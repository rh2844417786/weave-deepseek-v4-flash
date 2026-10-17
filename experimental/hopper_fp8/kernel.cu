// Hopper FP8 x packed MXFP4 candidate. No scalar GEMM fallback.
// The TMA staging, B128 layout, WGMMA fragment mapping and per-K32 partial
// accumulation are adapted from weave-deepgemm-repro commit 4fb1e6f,
// deep_gemm/impls/sm90_fp8_mxfp4_gemm_1d2d.cuh (MIT).
// See THIRD_PARTY_NOTICES.md and the vendored CUTLASS license.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cutlass/arch/barrier.h>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/arch/mma_sm90_gmma.hpp>
#include <cute/arch/mma_sm90_desc.hpp>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>
#include "math_contract.h"
#include "../../csrc/expert_numerics.h"

#define HOPPER_CUDA(call) do { auto hopper_cuda_status=(call); TORCH_CHECK(hopper_cuda_status==cudaSuccess, \
    "Hopper CUDA: ", cudaGetErrorString(hopper_cuda_status)); } while (0)
#define HOPPER_DRIVER(call) do { CUresult hopper_driver_status=(call); if(hopper_driver_status!=CUDA_SUCCESS) { \
    const char* hopper_driver_message=nullptr; cuGetErrorString(hopper_driver_status,&hopper_driver_message); \
    TORCH_CHECK(false,"Hopper tensor map: ", hopper_driver_message ? hopper_driver_message : "driver failure"); } } while (0)
namespace py = pybind11;
namespace weave::hopper {

struct Config {
  int rank, ep, max_batch, batch, h, i, experts, topk;
  int *counts, *active, *active_count, *slots;
  const __nv_bfloat16* x;
  const float* routing;
  uint8_t *a, *middle_fp8;
  float *a_scale, *middle_scale, *slot_outputs;
  __nv_bfloat16 *gate, *up, *middle, *down;
};
struct RoundBF16 {
  __device__ __forceinline__ float operator()(float x) const {
    return __bfloat162float(__float2bfloat16_rn(x));
  }
};
template<class Id>
__global__ void plan(Config cfg, const Id* ids) {
  const int count = plan_routes(ids, cfg.batch, cfg.topk, cfg.experts, cfg.rank,
      cfg.ep, cfg.max_batch, cfg.counts, cfg.active, cfg.slots);
  *cfg.active_count = count < 0 ? 0 : count;
  if(count<0) {
    assert(false && "Hopper routing: IDs out of range or duplicate per token");
    asm volatile("trap;");
  }
  for (int slot=0; slot<cfg.batch*cfg.topk; ++slot) {
    if(!isfinite(cfg.routing[slot])) {
      assert(false && "Hopper routing must be finite"); asm volatile("trap;");
    }
  }
}

// One CTA quantizes one active expert x K128 block, visiting its <=32
// real rows. Padding is written without reductions; this avoids launching
// thousands of CTAs just to fill the remaining rows of the one M64 tile.
template<bool Middle>
__global__ void quantize(Config cfg) {
  const int width=Middle?cfg.i:cfg.h, blocks=width/128;
  const int active_index=blockIdx.x/blocks;
  if (active_index >= *cfg.active_count) return;
  const int expert=cfg.active[active_index], kb=blockIdx.x%blocks;
  const int rows=cfg.counts[expert];
  auto* output=Middle?cfg.middle_fp8:cfg.a;
  auto* scales=Middle?cfg.middle_scale:cfg.a_scale;
  __shared__ float warp_max[4];
  for(int row=0;row<rows;++row) {
    float value;
    if constexpr(Middle) value=__bfloat162float(cfg.middle[(size_t(expert)*64+row)*width+kb*128+threadIdx.x]);
    else {
      const int token=cfg.slots[expert*cfg.max_batch+row]/cfg.topk;
      value=__bfloat162float(cfg.x[size_t(token)*width+kb*128+threadIdx.x]);
    }
    if(!isfinite(value)) {
      assert(false && "Hopper activation must be finite"); asm volatile("trap;");
    }
    float maximum=fabsf(value);
    #pragma unroll
    for(int offset=16;offset;offset/=2)
      maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffff,maximum,offset));
    if((threadIdx.x&31)==0) warp_max[threadIdx.x/32]=maximum;
    __syncthreads();
    maximum=fmaxf(fmaxf(warp_max[0],warp_max[1]),fmaxf(warp_max[2],warp_max[3]));
    const float scale=activation_scale(maximum);
    if(threadIdx.x==0) scales[(size_t(expert)*64+row)*blocks+kb]=scale;
    output[(size_t(expert)*64+row)*width+kb*128+threadIdx.x]=
        __nv_cvt_float_to_fp8(value/scale,__NV_SATFINITE,__NV_E4M3);
    // All readers finish before another row overwrites the four maxima.
    __syncthreads();
  }
  for(int row=rows;row<64;++row) {
    output[(size_t(expert)*64+row)*width+kb*128+threadIdx.x]=0;
    if(threadIdx.x==0) scales[(size_t(expert)*64+row)*blocks+kb]=1.f;
  }
}

__device__ __forceinline__ uint64_t descriptor(uint8_t* pointer) {
  // Initialize every descriptor bit, including reserved bits. Each swizzled
  // buffer is aligned 1024 so base_offset=0 matches the B128 addressing.
  cute::GmmaDescriptor desc{};
  desc.bitfield.start_address_=static_cast<uint32_t>(__cvta_generic_to_shared(pointer))>>4;
  desc.bitfield.leading_byte_offset_=0;
  desc.bitfield.stride_byte_offset_=1024>>4;
  desc.bitfield.base_offset_=0;
  desc.bitfield.layout_type_=1;  // GMMA B128
  return desc;
}
template<size_t... I>
__device__ __forceinline__ void mma(uint64_t a, uint64_t b, float* accum,
                                   std::index_sequence<I...>) {
  cute::SM90::GMMA::MMA_64x64x32_F32E4M3E4M3_SS_TN<>::fma(
      a,b,accum[I]...,cute::SM90::GMMA::ScaleOut::Zero);
}
__device__ __forceinline__ float e8m0(uint8_t code) {
  if(code==255) {
    assert(false && "Hopper MXFP4 E8M0 NaN scale is unsupported"); asm volatile("trap;");
  }
  return ldexpf(1.f,int(code)-127);
}

// Exactly one full warpgroup participates. Inactive capacity blocks return
// uniformly before barrier/TMA/WGMMA. Only one M64 tile per expert is issued.
__global__ __launch_bounds__(128,1) void gemm(
    const __grid_constant__ CUtensorMap a_map,
    const __grid_constant__ CUtensorMap b_map,
    const float* a_scales, const uint8_t* b_scales, __nv_bfloat16* output,
    const int* active, const int* active_count, const int* counts, int n, int k) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  const int ai=blockIdx.x/(n/64), nbase=(blockIdx.x%(n/64))*64;
  if(ai>=*active_count) return;
  const int expert=active[ai], valid_rows=counts[expert];
  using Barrier=cutlass::arch::ClusterTransactionBarrier;
  __shared__ __align__(128) uint8_t a_tma[64*128];
  __shared__ __align__(128) uint8_t b_tma[64*64];
  __shared__ __align__(1024) uint8_t a_smem[64*128];
  __shared__ __align__(1024) uint8_t b_smem[64*128];
  __shared__ __align__(16) uint64_t barrier_storage;
  auto* barrier=reinterpret_cast<Barrier*>(&barrier_storage);
  if(threadIdx.x==0) { barrier->init(1); cutlass::arch::fence_barrier_init(); }
  __syncthreads();
  const int lane=threadIdx.x%32, warp=threadIdx.x/32;
  const int r0=warp*16+lane/4, r1=r0+8, column=lane%4;
  float result[32]={0.f};
  unsigned phase=0;
  constexpr uint64_t hint=static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_FIRST);
  for(int stage=0;stage<k/128;++stage) {
    if(threadIdx.x==0) {
      // Expect bytes before launch. Two transactions complete the same phase.
      barrier->arrive_and_expect_tx(64*128+64*64);
      cute::SM90_TMA_LOAD_3D::copy(&a_map,&barrier_storage,hint,a_tma,stage*128,0,expert);
      cute::SM90_TMA_LOAD_3D::copy(&b_map,&barrier_storage,hint,b_tma,stage*64,nbase,expert);
    }
    barrier->wait(phase);
    __syncthreads();
    for(int index=threadIdx.x;index<64*128;index+=128) {
      const int row=index/128, col=index%128;
      a_smem[b128_offset(row,col)]=a_tma[index];
      b_smem[b128_offset(row,col)]=packed_to_e4m3(b_tma[row*64+col/2],col);
    }
    __syncthreads();
    cutlass::arch::fence_view_async_shared();
    __syncthreads();
    #pragma unroll
    for(int group=0;group<4;++group) {
      float partial[32]={0.f};
      #pragma unroll
      for(int z=0;z<32;++z) asm volatile("" : "+f"(partial[z]) :: "memory");
      asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
      mma(descriptor(a_smem+group*32),descriptor(b_smem+group*32),partial,
          std::make_index_sequence<32>{});
      asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
      asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
      #pragma unroll
      for(int z=0;z<32;++z) asm volatile("" : "+f"(partial[z]) :: "memory");
      const int kg=stage*4+group;
      const float as0=r0<valid_rows?a_scales[(size_t(expert)*64+r0)*(k/128)+stage]:0.f;
      const float as1=r1<valid_rows?a_scales[(size_t(expert)*64+r1)*(k/128)+stage]:0.f;
      #pragma unroll
      for(int fragment=0;fragment<8;++fragment) {
        const int c0=nbase+fragment*8+column*2;
        const float bs0=e8m0(b_scales[(size_t(expert)*n+c0)*(k/32)+kg]);
        const float bs1=e8m0(b_scales[(size_t(expert)*n+c0+1)*(k/32)+kg]);
        // Explicit RN operations keep oracle's K32 scaling/addition order,
        // independent of nvcc's contraction defaults.
        result[fragment*4]=__fadd_rn(result[fragment*4],__fmul_rn(__fmul_rn(partial[fragment*4],as0),bs0));
        result[fragment*4+1]=__fadd_rn(result[fragment*4+1],__fmul_rn(__fmul_rn(partial[fragment*4+1],as0),bs1));
        result[fragment*4+2]=__fadd_rn(result[fragment*4+2],__fmul_rn(__fmul_rn(partial[fragment*4+2],as1),bs0));
        result[fragment*4+3]=__fadd_rn(result[fragment*4+3],__fmul_rn(__fmul_rn(partial[fragment*4+3],as1),bs1));
      }
    }
    // The WGMMA wait above and this CTA rendezvous release both stage buffers.
    __syncthreads(); phase^=1;
  }
  #pragma unroll
  for(int f=0;f<8;++f) {
    const int c=nbase+f*8+column*2;
    if(r0<valid_rows) {
      output[(size_t(expert)*64+r0)*n+c]=__float2bfloat16_rn(result[f*4]);
      output[(size_t(expert)*64+r0)*n+c+1]=__float2bfloat16_rn(result[f*4+1]);
    }
    if(r1<valid_rows) {
      output[(size_t(expert)*64+r1)*n+c]=__float2bfloat16_rn(result[f*4+2]);
      output[(size_t(expert)*64+r1)*n+c+1]=__float2bfloat16_rn(result[f*4+3]);
    }
  }
#else
  asm volatile("trap;");
#endif
}
__global__ void swiglu(Config cfg,float clamp) {
  const int blocks_per_expert=(cfg.max_batch*cfg.i+255)/256;
  const int ai=blockIdx.x/blocks_per_expert;
  if(ai>=*cfg.active_count) return;
  const int e=cfg.active[ai], index=(blockIdx.x%blocks_per_expert)*256+threadIdx.x;
  if(index>=cfg.counts[e]*cfg.i) return;
  const size_t address=size_t(e)*64*cfg.i+index;
  cfg.middle[address]=__float2bfloat16_rn(weave::swiglu_product(
      __bfloat162float(cfg.gate[address]),__bfloat162float(cfg.up[address]),clamp,RoundBF16{}));
}
__global__ void route_slots(Config cfg) {
  const int blocks_per_expert=(cfg.max_batch*cfg.h+255)/256;
  const int ai=blockIdx.x/blocks_per_expert;
  if(ai>=*cfg.active_count) return;
  const int e=cfg.active[ai], index=(blockIdx.x%blocks_per_expert)*256+threadIdx.x;
  if(index>=cfg.counts[e]*cfg.h) return;
  const int row=index/cfg.h, col=index%cfg.h, slot=cfg.slots[e*cfg.max_batch+row];
  cfg.slot_outputs[size_t(slot)*cfg.h+col]=weave::routed_product(
      __bfloat162float(cfg.down[(size_t(e)*64+row)*cfg.h+col]),cfg.routing[slot],RoundBF16{});
}

// Maps are host encoded only at workspace construction / weight binding.
inline CUtensorMap tensor_map(void* address,int width,int rows,int experts,int box_width) {
  TORCH_CHECK(reinterpret_cast<uintptr_t>(address)%16==0,"TMA base pointer needs 16-byte alignment");
  CUtensorMap map{};
  const cuuint64_t dims[3]={cuuint64_t(width),cuuint64_t(rows),cuuint64_t(experts)};
  const cuuint64_t strides[2]={cuuint64_t(width),cuuint64_t(width)*rows};
  const cuuint32_t box[3]={cuuint32_t(box_width),64,1}, element_strides[3]={1,1,1};
  HOPPER_DRIVER(cuTensorMapEncodeTiled(&map,CU_TENSOR_MAP_DATA_TYPE_UINT8,3,address,dims,
      strides,box,element_strides,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_NONE,
      CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  return map;
}
class FP8Workspace;
struct FP8BoundWeights {
  std::shared_ptr<FP8Workspace> owner;
  std::vector<torch::Tensor> tensors;
  std::array<CUtensorMap,3> maps;
  FP8BoundWeights(std::shared_ptr<FP8Workspace> ws,const std::vector<torch::Tensor>& raw,
                  int local,int h,int i):owner(std::move(ws)),tensors(raw) {
    maps[0]=tensor_map(raw[0].data_ptr(),h/2,i,local,64);
    maps[1]=tensor_map(raw[1].data_ptr(),h/2,i,local,64);
    maps[2]=tensor_map(raw[2].data_ptr(),i/2,h,local,64);
  }
};
class FP8Workspace:public std::enable_shared_from_this<FP8Workspace> {
 public:
  FP8Workspace(int rank,int ep,int max_batch,int h,int i,int experts,int topk) {
    TORCH_CHECK(valid_dimensions(rank,ep,max_batch,h,i,experts,topk),
        "Hopper requires rank<ep<=4, B1..32, H/I multiples128 <=8192, E divisible by EP <=4096, topk1..16");
    HOPPER_CUDA(cudaGetDevice(&device_));
    cudaDeviceProp prop{}; HOPPER_CUDA(cudaGetDeviceProperties(&prop,device_));
    TORCH_CHECK(prop.major==9,"Hopper candidate requires SM90a hardware");
    require_outside_capture("workspace construction");
    cfg_={}; cfg_.rank=rank; cfg_.ep=ep; cfg_.max_batch=max_batch; cfg_.h=h;
    cfg_.i=i; cfg_.experts=experts; cfg_.topk=topk;
    const int local=experts/ep;
    auto opt=torch::TensorOptions().device(torch::kCUDA,device_);
    counts_=torch::empty({local},opt.dtype(torch::kInt32));
    active_=torch::empty({local},opt.dtype(torch::kInt32));
    count_=torch::empty({1},opt.dtype(torch::kInt32));
    expert_slots_=torch::empty({local,max_batch},opt.dtype(torch::kInt32));
    slots_=torch::empty({max_batch*topk,h},opt.dtype(torch::kFloat32));
    x_=torch::empty({max_batch,h},opt.dtype(torch::kBFloat16));
    routing_=torch::empty({max_batch,topk},opt.dtype(torch::kFloat32));
    a_=torch::empty({local,64,h},opt.dtype(torch::kUInt8));
    middle_fp8_=torch::empty({local,64,i},opt.dtype(torch::kUInt8));
    a_scale_=torch::empty({local,64,h/128},opt.dtype(torch::kFloat32));
    middle_scale_=torch::empty({local,64,i/128},opt.dtype(torch::kFloat32));
    gate_=torch::empty({local,64,i},opt.dtype(torch::kBFloat16));
    up_=torch::empty_like(gate_); middle_=torch::empty_like(gate_);
    down_=torch::empty({local,64,h},opt.dtype(torch::kBFloat16));
    cfg_.counts=counts_.data_ptr<int>(); cfg_.active=active_.data_ptr<int>();
    cfg_.active_count=count_.data_ptr<int>(); cfg_.slots=expert_slots_.data_ptr<int>();
    cfg_.slot_outputs=slots_.data_ptr<float>(); cfg_.routing=routing_.data_ptr<float>();
    cfg_.x=reinterpret_cast<__nv_bfloat16*>(x_.data_ptr<at::BFloat16>());
    cfg_.a=a_.data_ptr<uint8_t>(); cfg_.middle_fp8=middle_fp8_.data_ptr<uint8_t>();
    cfg_.a_scale=a_scale_.data_ptr<float>(); cfg_.middle_scale=middle_scale_.data_ptr<float>();
    cfg_.gate=reinterpret_cast<__nv_bfloat16*>(gate_.data_ptr<at::BFloat16>());
    cfg_.up=reinterpret_cast<__nv_bfloat16*>(up_.data_ptr<at::BFloat16>());
    cfg_.middle=reinterpret_cast<__nv_bfloat16*>(middle_.data_ptr<at::BFloat16>());
    cfg_.down=reinterpret_cast<__nv_bfloat16*>(down_.data_ptr<at::BFloat16>());
    a_map_=tensor_map(a_.data_ptr(),h,64,local,128);
    middle_map_=tensor_map(middle_fp8_.data_ptr(),i,64,local,128);
    views_.resize(max_batch+1);
    for(int b=1;b<=max_batch;++b) views_[b]={x_.narrow(0,0,b),routing_.narrow(0,0,b)};
  }
  std::shared_ptr<FP8BoundWeights> bind_weights(const std::vector<torch::Tensor>& raw) {
    c10::cuda::CUDAGuard guard(device_); require_outside_capture("weight binding");
    TORCH_CHECK(raw.size()==6,"bind_weights requires original w1,w3,w2,s1,s3,s2");
    const int local=cfg_.experts/cfg_.ep,h=cfg_.h,i=cfg_.i;
    validate(raw[0],torch::kUInt8,{local,i,h/2},"w1");
    validate(raw[1],torch::kUInt8,{local,i,h/2},"w3");
    validate(raw[2],torch::kUInt8,{local,h,i/2},"w2");
    validate(raw[3],torch::kUInt8,{local,i,h/32},"s1");
    validate(raw[4],torch::kUInt8,{local,i,h/32},"s3");
    validate(raw[5],torch::kUInt8,{local,h,i/32},"s2");
    return std::make_shared<FP8BoundWeights>(shared_from_this(),raw,local,h,i);
  }
  void prepare(torch::Tensor x,torch::Tensor ids,torch::Tensor routing) {
    c10::cuda::CUDAGuard guard(device_);
    TORCH_CHECK(x.dim()==2 && x.size(0)>=1 && x.size(0)<=cfg_.max_batch,"batch exceeds Hopper workspace capacity");
    const int b=int(x.size(0));
    validate(x,torch::kBFloat16,{b,cfg_.h},"input");
    TORCH_CHECK(ids.scalar_type()==torch::kInt32 || ids.scalar_type()==torch::kInt64,"IDs must be int32/int64");
    TORCH_CHECK(routing.scalar_type()==torch::kFloat32 || routing.scalar_type()==torch::kBFloat16 ||
        routing.scalar_type()==torch::kFloat16,"routing must be FP32/BF16/FP16");
    validate(ids,ids.scalar_type(),{b,cfg_.topk},"IDs");
    validate(routing,routing.scalar_type(),{b,cfg_.topk},"routing");
    views_[b][0].copy_(x); views_[b][1].copy_(routing);
    cfg_.batch=b;
    auto stream=current_stream();
    HOPPER_CUDA(cudaMemsetAsync(cfg_.slot_outputs,0,slots_.numel()*sizeof(float),stream));
    if(ids.scalar_type()==torch::kInt32) plan<<<1,1,0,stream>>>(cfg_,ids.data_ptr<int>());
    else plan<<<1,1,0,stream>>>(cfg_,ids.data_ptr<int64_t>());
    HOPPER_CUDA(cudaGetLastError());
    quantize<false><<<capacity()*(cfg_.h/128),128,0,stream>>>(cfg_);
    HOPPER_CUDA(cudaGetLastError());
    phase_=1;
  }
  void gate_up(const std::shared_ptr<FP8BoundWeights>& bound,float clamp=10.f) {
    c10::cuda::CUDAGuard guard(device_); check_bound(bound);
    TORCH_CHECK(phase_==1,"gate_up must follow prepare");
    TORCH_CHECK(std::isfinite(clamp) && clamp>0,"clamp must be finite and positive");
    launch_gemm(a_map_,bound->maps[0],cfg_.a_scale,bound->tensors[3],cfg_.gate,cfg_.i,cfg_.h);
    launch_gemm(a_map_,bound->maps[1],cfg_.a_scale,bound->tensors[4],cfg_.up,cfg_.i,cfg_.h);
    auto stream=current_stream();
    swiglu<<<capacity()*((cfg_.max_batch*cfg_.i+255)/256),256,0,stream>>>(cfg_,clamp);
    HOPPER_CUDA(cudaGetLastError());
    quantize<true><<<capacity()*(cfg_.i/128),128,0,stream>>>(cfg_);
    HOPPER_CUDA(cudaGetLastError()); phase_bound_=bound.get(); phase_=2;
  }
  void down(const std::shared_ptr<FP8BoundWeights>& bound) {
    c10::cuda::CUDAGuard guard(device_); check_bound(bound);
    TORCH_CHECK(phase_==2 && phase_bound_==bound.get(),"down must follow gate_up with the same bound weights");
    launch_gemm(middle_map_,bound->maps[2],cfg_.middle_scale,bound->tensors[5],cfg_.down,cfg_.h,cfg_.i);
    phase_=3;
  }
  void route() {
    c10::cuda::CUDAGuard guard(device_);
    TORCH_CHECK(phase_==3,"route must follow down");
    route_slots<<<capacity()*((cfg_.max_batch*cfg_.h+255)/256),256,0,current_stream()>>>(cfg_);
    HOPPER_CUDA(cudaGetLastError()); phase_=4;
  }
  void run(torch::Tensor x,torch::Tensor ids,torch::Tensor routing,
           const std::shared_ptr<FP8BoundWeights>& bound,float clamp=10.f) {
    check_bound(bound);
    TORCH_CHECK(std::isfinite(clamp) && clamp>0,"clamp must be finite and positive");
    prepare(x,ids,routing); gate_up(bound,clamp); down(bound); route();
  }
  torch::Tensor slot_outputs() const {return slots_;}
  torch::Tensor group_counts() const {return counts_;}
  torch::Tensor active_experts() const {return active_;}
  torch::Tensor active_count() const {return count_;}
  torch::Tensor expert_slots() const {return expert_slots_;}
  py::dict resources() const {
    c10::cuda::CUDAGuard guard(device_); require_outside_capture("resource inspection");
    cudaFuncAttributes attr{}; HOPPER_CUDA(cudaFuncGetAttributes(&attr,gemm));
    int occupancy=0; HOPPER_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy,gemm,128,0));
    py::dict result;
    result["numeric_contract"]=kNumericContract;
    result["kernel"]="SM90a_TMA_FP8_M64N64K32_stage128";
    result["threads"]=128; result["registers_per_thread"]=attr.numRegs;
    result["static_shared_bytes"]=attr.sharedSizeBytes;
    result["local_bytes"]=attr.localSizeBytes;
    result["active_blocks_per_sm"]=occupancy;
    result["binary_version"]=attr.binaryVersion;
    result["ptx_version"]=attr.ptxVersion;
    result["max_batch"]=cfg_.max_batch;
    result["local_experts"]=cfg_.experts/cfg_.ep;
    result["active_expert_capacity"]=active_capacity(cfg_.max_batch,cfg_.topk,cfg_.experts/cfg_.ep);
    result["gate_grid_capacity"]=active_capacity(cfg_.max_batch,cfg_.topk,cfg_.experts/cfg_.ep)*(cfg_.i/64);
    result["down_grid_capacity"]=active_capacity(cfg_.max_batch,cfg_.topk,cfg_.experts/cfg_.ep)*(cfg_.h/64);
    result["tma_transaction_bytes"]=64*128+64*64;
    result["a_map_shape"]=py::make_tuple(cfg_.experts/cfg_.ep,64,cfg_.h);
    result["middle_map_shape"]=py::make_tuple(cfg_.experts/cfg_.ep,64,cfg_.i);
    result["weight_storage"]="original readonly packed MXFP4 and E8M0 tensors; no copies";
    result["activation_quantization"]="per-row K128 maxabs/448; zero=1; E4M3 RNE SATFINITE";
    result["accumulation"]="FP32 K32 partial; (partial*a_scale)*b_scale then ordered FP32 add";
    result["output_boundary"]="BF16 GEMM/SwiGLU/routed product; FP32 local slot storage";
    result["concurrency"]="one workspace per stream; no concurrent reuse; bound storage immutable";
    return result;
  }
 private:
  int device_=-1,phase_=0;
  Config cfg_{};
  const FP8BoundWeights* phase_bound_=nullptr;
  torch::Tensor counts_,active_,count_,expert_slots_,slots_,x_,routing_;
  torch::Tensor a_,middle_fp8_,a_scale_,middle_scale_,gate_,up_,middle_,down_;
  std::vector<std::array<torch::Tensor,2>> views_;
  CUtensorMap a_map_{},middle_map_{};
  cudaStream_t current_stream() const {return at::cuda::getCurrentCUDAStream(device_).stream();}
  int capacity() const {return active_capacity(cfg_.batch,cfg_.topk,cfg_.experts/cfg_.ep);}
  void require_outside_capture(const char* operation) const {
    cudaStreamCaptureStatus status; HOPPER_CUDA(cudaStreamIsCapturing(current_stream(),&status));
    TORCH_CHECK(status==cudaStreamCaptureStatusNone,operation," must happen before CUDA graph capture");
  }
  void validate(const torch::Tensor& tensor,torch::ScalarType dtype,
                std::initializer_list<int64_t> shape,const char* name) const {
    TORCH_CHECK(tensor.is_cuda() && tensor.get_device()==device_ && tensor.is_contiguous() &&
        tensor.scalar_type()==dtype && tensor.dim()==int64_t(shape.size()),"invalid Hopper ",name," dtype/device/shape/contiguity");
    int dimension=0;
    for(auto size:shape) {TORCH_CHECK(tensor.size(dimension)==size,"invalid Hopper ",name," shape"); ++dimension;}
  }
  void check_bound(const std::shared_ptr<FP8BoundWeights>& bound) const {
    TORCH_CHECK(bound && bound->owner.get()==this,"bound weights belong to another workspace");
  }
  void launch_gemm(const CUtensorMap& amap,const CUtensorMap& bmap,const float* scales,
                   const torch::Tensor& bscales,__nv_bfloat16* out,int n,int k) {
    gemm<<<capacity()*(n/64),128,0,current_stream()>>>(amap,bmap,scales,bscales.data_ptr<uint8_t>(),out,
        cfg_.active,cfg_.active_count,cfg_.counts,n,k);
    HOPPER_CUDA(cudaGetLastError());
  }
};
} // namespace weave::hopper
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
  using namespace weave::hopper;
  py::class_<FP8BoundWeights,std::shared_ptr<FP8BoundWeights>>(m,"FP8BoundWeights");
  py::class_<FP8Workspace,std::shared_ptr<FP8Workspace>>(m,"FP8Workspace")
      .def(py::init<int,int,int,int,int,int,int>(),py::arg("rank"),py::arg("ep"),
           py::arg("max_batch"),py::arg("h"),py::arg("i"),py::arg("experts"),py::arg("topk"))
      .def("bind_weights",&FP8Workspace::bind_weights,py::arg("raw_six"))
      .def("run",&FP8Workspace::run,py::arg("x"),py::arg("ids"),py::arg("routing"),py::arg("bound"),py::arg("clamp")=10.f)
      .def("prepare",&FP8Workspace::prepare,py::arg("x"),py::arg("ids"),py::arg("routing"))
      .def("gate_up",&FP8Workspace::gate_up,py::arg("bound"),py::arg("clamp")=10.f)
      .def("down",&FP8Workspace::down,py::arg("bound"))
      .def("route",&FP8Workspace::route)
      .def("resources",&FP8Workspace::resources)
      .def_property_readonly("slot_outputs",&FP8Workspace::slot_outputs)
      .def_property_readonly("group_counts",&FP8Workspace::group_counts)
      .def_property_readonly("active_experts",&FP8Workspace::active_experts)
      .def_property_readonly("active_count",&FP8Workspace::active_count)
      .def_property_readonly("expert_slots",&FP8Workspace::expert_slots);
}
