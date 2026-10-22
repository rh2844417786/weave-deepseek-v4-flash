#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cooperative_groups.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <cstring>
#include <sstream>
#include <iomanip>
#include "peer_reduce.cuh"

namespace nt=weave::native_tail;
namespace cg=cooperative_groups;
#define NT_CUDA(call) do { auto e=(call); TORCH_CHECK(e==cudaSuccess,#call,": ",cudaGetErrorString(e)); } while(0)

namespace {
__global__ void transport_kernel(nt::TransportConfig cfg,const __nv_bfloat16* input) {
  auto grid=cg::this_grid();
  if(blockIdx.x==0 && threadIdx.x==0) nt::prepare_invocation(cfg);
  grid.sync();
  auto inv=nt::current_invocation(cfg);
  nt::U64 start=clock64();
  __shared__ int stop;
  for(int chunk=blockIdx.x;chunk<cfg.chunks;chunk+=gridDim.x) {
    if(threadIdx.x==0) {
      stop=0;
      while(clock64()-start<cfg.delay_cycles) if(nt::protocol_aborted(cfg,start)) {stop=1;break;}
    }
    __syncthreads();
    if(stop) break;
    int width=cfg.layout.hidden/(cfg.chunks*nt::kRanks);
    size_t n=size_t(cfg.tokens)*width*nt::kRanks;
    for(size_t i=threadIdx.x;i<n;i+=blockDim.x) {
      int owner=int(i/(size_t(cfg.tokens)*width));
      size_t within=i%(size_t(cfg.tokens)*width);
      int token=int(within/width),x=int(within%width);
      auto value=input[nt::output_index(cfg,chunk,owner,token,x)];
      if(!isfinite(__bfloat162float(value))) {
        atomicAdd(&nt::slot_control(cfg,cfg.rank,inv)->nonfinite,1u);nt::abort_local(cfg,nt::kNonfinite);
      }
      cfg.layout.partial(cfg.peers[cfg.rank],inv.slot)[nt::payload_index(cfg,chunk,owner,token,x)]=value;
    }
    nt::publish_local_chunk(cfg,inv,chunk);
  }
  while(true) {
    if(threadIdx.x==0) stop=nt::protocol_aborted(cfg,start)?2:
       (nt::transport_complete(cfg,inv)?1:0);
    __syncthreads();if(stop) break;
    if(threadIdx.x==0) stop=nt::protocol_headers_ready(cfg,inv)?0:3;
    __syncthreads();
    if(stop==0) nt::transport_try_work(cfg,inv);
    else __nanosleep(128);
  }
  grid.sync();
}

std::string uuid_string(int device) {
  cudaDeviceProp p; NT_CUDA(cudaGetDeviceProperties(&p,device));
  std::ostringstream out;
  for(unsigned char b:p.uuid.bytes) out<<std::hex<<std::setw(2)<<std::setfill('0')<<int(b);
  return out.str();
}
class Workspace {
public:
  int rank,capacity,hidden,device,clock_khz,workers,occupancy,sms;
  nt::U64 session;
  nt::Layout layout;
  void* bases[nt::kRanks]{};
  bool connected=false,closed=false;
  std::vector<int> peer_access=std::vector<int>(4,0),peer_atomics=std::vector<int>(4,0);
  cudaFuncAttributes attributes{};
  Workspace(int rank_,int capacity_,int hidden_,nt::U64 session_):rank(rank_),capacity(capacity_),hidden(hidden_),
      session(session_),layout(capacity_,hidden_) {
    TORCH_CHECK(rank>=0&&rank<4&&capacity>=0&&capacity<=8192&&hidden>0&&hidden%4==0&&session>0,"invalid native tail workspace contract");
    NT_CUDA(cudaGetDevice(&device));
    int cooperative=0;NT_CUDA(cudaDeviceGetAttribute(&cooperative,cudaDevAttrCooperativeLaunch,device));
    TORCH_CHECK(cooperative,"native tail requires cooperative launch; no oversubscribed fallback");
    cudaDeviceProp p;NT_CUDA(cudaGetDeviceProperties(&p,device));
    TORCH_CHECK(p.major>=9,"native tail currently supports Hopper or later");sms=p.multiProcessorCount;
    NT_CUDA(cudaDeviceGetAttribute(&clock_khz,cudaDevAttrClockRate,device));
    NT_CUDA(cudaFuncGetAttributes(&attributes,transport_kernel));
    NT_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy,transport_kernel,nt::kThreads,0));
    TORCH_CHECK(occupancy>0,"final transport kernel has zero resident occupancy");
    workers=std::min(sms*occupancy,32); // bounded by THIS final kernel, not a proxy kernel.
    NT_CUDA(cudaMalloc(&bases[rank],layout.bytes));
    try {
      NT_CUDA(cudaMemset(bases[rank],0,layout.bytes));
      nt::U64 canaries[16];std::fill_n(canaries,16,nt::kCanary);
      for(int slot=0;slot<nt::kSlots;++slot) for(bool after:{false,true})
        NT_CUDA(cudaMemcpy(layout.guard(bases[rank],slot,after),canaries,128,cudaMemcpyHostToDevice));
    } catch(...) {cudaFree(bases[rank]);bases[rank]=nullptr;throw;}
  }
  ~Workspace() { try {disconnect();close();} catch(...) {} }
  std::string uuid() const {return uuid_string(device);}
  pybind11::bytes handle() {
    TORCH_CHECK(!closed,"closed workspace");c10::cuda::CUDAGuard guard(device);
    cudaIpcMemHandle_t h;NT_CUDA(cudaIpcGetMemHandle(&h,bases[rank]));
    return pybind11::bytes(reinterpret_cast<char*>(&h),sizeof(h));
  }
  void connect(const std::vector<std::string>& handles,const std::vector<int>& ordinals,const std::vector<std::string>& uuids) {
    TORCH_CHECK(!closed&&!connected&&handles.size()==4&&ordinals.size()==4&&uuids.size()==4,"four ranks required");
    c10::cuda::CUDAGuard guard(device);
    for(int r=0;r<4;++r) {
      TORCH_CHECK(uuid_string(ordinals[r])==uuids[r],"rank CUDA visibility differs; use the same GPUQ-visible device list in every process");
      TORCH_CHECK(handles[r].size()==sizeof(cudaIpcMemHandle_t),"invalid IPC handle");
      if(r==rank) continue;
      int access=0,atomics=0;NT_CUDA(cudaDeviceCanAccessPeer(&access,device,ordinals[r]));
      NT_CUDA(cudaDeviceGetP2PAttribute(&atomics,cudaDevP2PAttrNativeAtomicSupported,device,ordinals[r]));
      peer_access[r]=access;peer_atomics[r]=atomics;
      TORCH_CHECK(access&&atomics,"four-rank IPC needs peer access and native system atomics for every pair");
      cudaIpcMemHandle_t h;std::memcpy(&h,handles[r].data(),sizeof(h));
      NT_CUDA(cudaIpcOpenMemHandle(&bases[r],h,cudaIpcMemLazyEnablePeerAccess));
    }
    connected=true;
  }
  std::vector<uint64_t> addresses() const {
    TORCH_CHECK(connected&&!closed,"connect IPC before requesting executor addresses");
    std::vector<uint64_t> out;for(void* p:bases) out.push_back(reinterpret_cast<uint64_t>(p));return out;
  }
  pybind11::dict resources() const {
    pybind11::dict d;d["cooperative_launch"]=true;d["sm_count"]=sms;
    d["active_blocks_per_sm"]=occupancy;d["workers"]=workers;d["threads"]=nt::kThreads;
    d["num_registers"]=attributes.numRegs;d["static_shared_bytes"]=attributes.sharedSizeBytes;
    d["local_bytes"]=attributes.localSizeBytes;d["workspace_bytes"]=layout.bytes;
    d["peer_access"]=peer_access;d["peer_native_atomics"]=peer_atomics;return d;
  }
  void run(torch::Tensor input,torch::Tensor output,int chunks,nt::U64 layer,double timeout_seconds,
           double delay_seconds,double cancel_seconds) {
    TORCH_CHECK(connected&&!closed,"IPC is not connected");c10::cuda::CUDAGuard guard(device);
    TORCH_CHECK(input.is_cuda()&&output.is_cuda()&&input.get_device()==device&&output.get_device()==device,"local CUDA tensors required");
    TORCH_CHECK(input.scalar_type()==torch::kBFloat16&&output.scalar_type()==torch::kBFloat16&&
      input.is_contiguous()&&output.is_contiguous()&&input.dim()==2&&input.size(1)==hidden&&
      input.size(0)<=capacity&&input.sizes()==output.sizes(),"expected matching contiguous BF16 [tokens,hidden] tensors");
    auto in_begin=reinterpret_cast<uintptr_t>(input.data_ptr());
    auto out_begin=reinterpret_cast<uintptr_t>(output.data_ptr());
    size_t input_bytes=input.numel()*input.element_size(),output_bytes=output.numel()*output.element_size();
    TORCH_CHECK(input_bytes==0 || in_begin+input_bytes<=out_begin || out_begin+output_bytes<=in_begin,
                "partial and output must not alias: later chunks may still read partial");
    TORCH_CHECK((chunks==1||chunks==2||chunks==4)&&hidden%(4*chunks)==0,"chunk/owner layout mismatch");
    TORCH_CHECK(timeout_seconds>0&&timeout_seconds<=300&&delay_seconds>=0&&cancel_seconds>=0,"invalid timeout/delay/cancellation");
    nt::TransportConfig cfg;cfg.layout=layout;cfg.rank=rank;cfg.tokens=input.size(0);cfg.chunks=chunks;
    cfg.session=session;cfg.layer=layer;cfg.timeout_cycles=nt::U64(timeout_seconds*clock_khz*1000.);
    cfg.delay_cycles=nt::U64(delay_seconds*clock_khz*1000.);cfg.cancel_after_cycles=nt::U64(cancel_seconds*clock_khz*1000.);
    cfg.output=reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>());
    for(int r=0;r<4;++r)cfg.peers[r]=bases[r];
    auto ptr=reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>());
    void* args[]={&cfg,&ptr};
    NT_CUDA(cudaLaunchCooperativeKernel(reinterpret_cast<void*>(transport_kernel),dim3(workers),dim3(nt::kThreads),args,0,
                                        at::cuda::getCurrentCUDAStream(device).stream()));
  }
  pybind11::dict snapshot(int chunks) {
    TORCH_CHECK(chunks==1||chunks==2||chunks==4,"invalid snapshot chunk count");
    TORCH_CHECK(!closed,"closed workspace");c10::cuda::CUDAGuard guard(device);
    NT_CUDA(cudaDeviceSynchronize());
    nt::Control host;NT_CUDA(cudaMemcpy(&host,bases[rank],sizeof(host),cudaMemcpyDeviceToHost));
    auto& s=host.slots[host.sequence?(host.sequence-1)%nt::kSlots:0];
    bool guards_ok=true;
    for(int slot=0;slot<nt::kSlots;++slot) for(bool after:{false,true}) {
      nt::U64 values[16];NT_CUDA(cudaMemcpy(values,layout.guard(bases[rank],slot,after),128,cudaMemcpyDeviceToHost));
      for(auto x:values)guards_ok &= x==nt::kCanary;
    }
    bool acknowledged=s.done==host.sequence&&host.sequence>0;
    std::vector<unsigned> tasks,smids;
    for(int i=0;i<chunks*5;++i){tasks.push_back(s.executed[i]);smids.push_back(s.smid[i]);}
    for(int c=0;c<chunks;++c)for(int r=0;r<4;++r) acknowledged &= s.ack[c][r]==host.sequence;
    pybind11::dict d;d["sequence"]=host.sequence;d["abort_code"]=host.abort_code;d["nonfinite"]=s.nonfinite;
    d["tasks"]=tasks;d["task_smids"]=smids;d["expected_tasks"]=chunks*5;d["acknowledged"]=acknowledged;
    d["canary_ok"]=guards_ok;d["session"]=s.session;d["layer"]=s.layer;return d;
  }
  void disconnect() {
    if(closed)return;c10::cuda::CUDAGuard guard(device);NT_CUDA(cudaDeviceSynchronize());
    for(int r=0;r<4;++r)if(r!=rank&&bases[r]) {NT_CUDA(cudaIpcCloseMemHandle(bases[r]));bases[r]=nullptr;}
    connected=false;
  }
  void close() {
    if(closed)return;TORCH_CHECK(!connected,"disconnect all peer mappings before freeing allocation");
    c10::cuda::CUDAGuard guard(device);
    if(bases[rank]) {NT_CUDA(cudaFree(bases[rank]));bases[rank]=nullptr;}closed=true;
  }
};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
  pybind11::class_<Workspace,std::shared_ptr<Workspace>>(m,"Workspace")
    .def(pybind11::init<int,int,int,nt::U64>()).def("handle",&Workspace::handle)
    .def("connect",&Workspace::connect).def("addresses",&Workspace::addresses)
    .def("resources",&Workspace::resources).def("run",&Workspace::run).def("snapshot",&Workspace::snapshot)
    .def("disconnect",&Workspace::disconnect).def("close",&Workspace::close)
    .def_property_readonly("uuid",&Workspace::uuid).def_readonly("device",&Workspace::device);
}
