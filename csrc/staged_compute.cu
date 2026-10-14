// P1 compute-only control: ordered gate/up, down, and local reduce launches.
// All dependencies use the current CUDA stream; no IPC or inter-CTA waiting.
#include <torch/extension.h>
#include <pybind11/stl.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cmath>
#include <array>
#include <memory>
#include <vector>
#include "staged_compute.cuh"
#include "staged_grid.h"
#include "online_staged.h"
#include "tile_math.cuh"

#define STAGED_CUDA_OK(call) do { cudaError_t result = (call); \
  TORCH_CHECK(result == cudaSuccess, "CUDA failure: ", cudaGetErrorString(result)); } while (0)

namespace {
constexpr int THREADS = 128;
constexpr int SHARED_BYTES = 8192;  // Four original 1024-BF16 warp workspaces.

struct StagedConfig {
  int rank, ep, batch, h, i, topk, experts, capacity;
  weave::tile_math::Weights weights;
  const __nv_bfloat16* x;
  const int *ids, *group_meta, *group_slots, *group_count;
  const float* routing;
  __nv_bfloat16* mid;
  float *slot_outputs, *output;
  float clamp;
};

__device__ inline void check(bool good) {
  if (!good) asm volatile("trap;");
}

struct StagedInput {
  const __nv_bfloat16* x;
  const int* slots;
  int h, topk;
  __device__ __nv_bfloat16 operator()(int row, int column) const {
    return x[size_t(slots[row] / topk) * h + column];
  }
};

struct MidInput {
  const __nv_bfloat16* mid;
  int i;
  __device__ __nv_bfloat16 operator()(int row, int column) const {
    return mid[size_t(row) * i + column];
  }
};

struct MidStore {
  __nv_bfloat16* mid;
  int i;
  __device__ void operator()(int row, int column, __nv_bfloat16 value) const {
    mid[size_t(row) * i + column] = value;
  }
};

struct SlotStore {
  float* output;
  const int* slots;
  int h;
  __device__ void operator()(int row, int column, float value) const {
    output[size_t(slots[row]) * h + column] = value;
  }
};

template<bool GateUp>
__global__ void staged_compute(StagedConfig cfg) {
  extern __shared__ __align__(32) unsigned char shared_bytes[];
  auto scratch = reinterpret_cast<__nv_bfloat16*>(shared_bytes) + (threadIdx.x / 32) * 1024;
  const int groups = cfg.group_count[0];
  check(groups >= 0 && groups <= cfg.capacity);
  const int columns = weave::tile_columns(GateUp ? cfg.i : cfg.h);
  const int tasks = weave::staged_tiles(groups, columns);
  for (int tile = blockIdx.x; tile < tasks; tile += gridDim.x) {
    const auto claim = weave::staged_tile(tile, columns);
    const int expert = cfg.group_meta[claim.group * 2];
    const int rows = cfg.group_meta[claim.group * 2 + 1];
    const int* slots = cfg.group_slots + size_t(claim.group) * weave::TILE_ROWS;
    check(expert >= 0 && expert < cfg.experts / cfg.ep && rows >= 1 && rows <= weave::TILE_ROWS);
    // Metadata content is validated by the offline CPU planner. These checks
    // fail safely if a caller uploads metadata for a different frozen trace.
    for (int row = threadIdx.x; row < rows; row += blockDim.x) {
      const int slot = slots[row];
      check(slot >= 0 && slot < cfg.batch * cfg.topk);
      check(cfg.ids[slot] == cfg.rank * (cfg.experts / cfg.ep) + expert);
    }
    __syncthreads();
    auto mid = cfg.mid + size_t(claim.group) * weave::TILE_ROWS * cfg.i;
    const int first = claim.column * weave::TILE_COLUMNS + (threadIdx.x / 32) * 16;
    if constexpr (GateUp) {
      weave::tile_math::gate_up(cfg.weights, expert, rows, cfg.h, cfg.i, first, cfg.clamp,
          StagedInput{cfg.x, slots, cfg.h, cfg.topk}, MidStore{mid, cfg.i}, scratch);
    } else {
      weave::tile_math::down(cfg.weights, expert, rows, cfg.h, cfg.i, first,
          MidInput{mid, cfg.i}, SlotStore{cfg.slot_outputs, slots, cfg.h}, scratch);
    }
    // Scratch reuse stays CTA-local; stream order separates the two stages.
    __syncthreads();
  }
}

__global__ void staged_reduce(StagedConfig cfg) {
  const int local_experts = cfg.experts / cfg.ep;
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < cfg.batch * cfg.h; index += blockDim.x * gridDim.x) {
    const int token = index / cfg.h, column = index % cfg.h;
    float sum = 0.f;
    // Keep original per-token slot order and routed-product BF16 boundaries.
    for (int slot = 0; slot < cfg.topk; ++slot) {
      const int pos = token * cfg.topk + slot;
      const int expert = cfg.ids[pos];
      const float route = cfg.routing[pos];
      if (column == 0) {
        check(expert >= 0 && expert < cfg.experts && isfinite(route) && route >= 0.f);
        for (int prev = 0; prev < slot; ++prev) check(cfg.ids[token * cfg.topk + prev] != expert);
      }
      if (expert / local_experts == cfg.rank)
        sum += weave::routed_product(cfg.slot_outputs[size_t(pos) * cfg.h + column], route,
                                     weave::tile_math::RoundBF16{});
      else
        cfg.slot_outputs[size_t(pos) * cfg.h + column] = 0.f;
    }
    cfg.output[index] = sum;
  }
}

__global__ void online_groups(StagedConfig cfg, int* meta, int* slots, int* count,
                               const int* ids32, const int64_t* ids64, int* copied_ids) {
  if (blockIdx.x || threadIdx.x) return;
  const int routes = cfg.batch * cfg.topk;
  check(ids64 ? weave::online_copy_ids(ids64, copied_ids, routes, cfg.experts)
              : weave::online_copy_ids(ids32, copied_ids, routes, cfg.experts));
  const int groups = weave::online_build_groups(cfg.ids, cfg.batch, cfg.topk,
      cfg.rank, cfg.ep, cfg.experts, cfg.capacity, meta, slots);
  check(groups >= 0 && groups <= cfg.capacity);
  for (int pos = 0; pos < cfg.batch * cfg.topk; ++pos)
    check(isfinite(cfg.routing[pos]) && cfg.routing[pos] >= 0.f);
  *count = groups;
}

__global__ void online_routed_slots(StagedConfig cfg) {
  const int local_experts = cfg.experts / cfg.ep;
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < cfg.batch * cfg.topk * cfg.h; index += blockDim.x * gridDim.x) {
    const int slot = index / cfg.h;
    // Nonowned slots were zeroed before compute; never round stale storage.
    if (cfg.ids[slot] / local_experts == cfg.rank)
      cfg.slot_outputs[index] = weave::routed_product(cfg.slot_outputs[index], cfg.routing[slot],
                                                     weave::tile_math::RoundBF16{});
  }
}

__global__ void online_finalize(StagedConfig cfg, __nv_bfloat16* output) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < cfg.batch * cfg.h; index += blockDim.x * gridDim.x) {
    const int token = index / cfg.h, column = index % cfg.h;
    const float sum = weave::online_origin_row(token, cfg.batch, cfg.rank, cfg.ep)
        ? weave::online_slot_sum(cfg.slot_outputs, token, column, cfg.topk, cfg.h) : 0.f;
    output[index] = __float2bfloat16_rn(sum);
  }
}

class StagedWorkspace;
struct StagedBoundWeights {
  // This handle keeps its raw checkpoint tensors and workspace alive.
  // Checkpoint storage must not be resized or mutated while bound.
  std::shared_ptr<StagedWorkspace> owner;
  std::vector<torch::Tensor> tensors;
  weave::tile_math::Weights weights;
  StagedBoundWeights(std::shared_ptr<StagedWorkspace> owner_,
                     const std::vector<torch::Tensor>& tensors_)
      : owner(std::move(owner_)), tensors(tensors_),
        weights{tensors_[0].data_ptr<unsigned char>(), tensors_[1].data_ptr<unsigned char>(),
                tensors_[2].data_ptr<unsigned char>(), tensors_[3].data_ptr<unsigned char>(),
                tensors_[4].data_ptr<unsigned char>(), tensors_[5].data_ptr<unsigned char>()} {}
};

class StagedWorkspace : public std::enable_shared_from_this<StagedWorkspace> {
 public:
  StagedWorkspace(int rank, int ep, int batch, int h, int i, int experts, int topk) {
    TORCH_CHECK(ep >= 1 && ep <= 4 && rank >= 0 && rank < ep && batch >= 1 && batch <= 32,
                "staged replay requires a valid EP rank and batch in 1..32");
    TORCH_CHECK(h > 0 && h <= 8192 && h % 32 == 0 && i > 0 && i <= 8192 && i % 32 == 0,
                "staged hidden/intermediate dimensions must be multiples of 32 in 32..8192");
    TORCH_CHECK(topk >= 1 && topk <= 16 && experts >= topk && experts % ep == 0,
                "invalid staged expert count or topk");
    STAGED_CUDA_OK(cudaGetDevice(&device_));
    cudaDeviceProp properties;
    STAGED_CUDA_OK(cudaGetDeviceProperties(&properties, device_));
    TORCH_CHECK(properties.major == 9, "staged kernel targets Hopper SM90");
    multiprocessors_ = properties.multiProcessorCount;
    cfg_ = {};
    cfg_.rank = rank; cfg_.ep = ep; cfg_.batch = batch; cfg_.h = h; cfg_.i = i;
    cfg_.experts = experts; cfg_.topk = topk; cfg_.clamp = 10.f;
    auto options = torch::TensorOptions().device(torch::kCUDA, device_);
    // One group per route is the maximum; backing addresses never resize.
    mid_ = torch::empty({int64_t(batch) * topk, weave::TILE_ROWS, i}, options.dtype(torch::kBFloat16));
    slots_ = torch::empty({int64_t(batch) * topk, h}, options.dtype(torch::kFloat32));
    output_ = torch::empty({batch, h}, options.dtype(torch::kFloat32));
    cfg_.mid = reinterpret_cast<__nv_bfloat16*>(mid_.data_ptr<at::BFloat16>());
    cfg_.slot_outputs = slots_.data_ptr<float>();
    cfg_.output = output_.data_ptr<float>();
    online_cfg_ = cfg_;
    online_cfg_.capacity = weave::online_group_capacity(batch, topk, experts / ep);
    online_x_ = torch::empty({batch, h}, options.dtype(torch::kBFloat16));
    online_ids_ = torch::empty({batch, topk}, options.dtype(torch::kInt32));
    online_routing_ = torch::empty({batch, topk}, options.dtype(torch::kFloat32));
    online_meta_ = torch::empty({online_cfg_.capacity, 2}, options.dtype(torch::kInt32));
    online_slots_ = torch::empty({online_cfg_.capacity, weave::TILE_ROWS}, options.dtype(torch::kInt32));
    online_count_ = torch::empty({1}, options.dtype(torch::kInt32));
    online_output_ = torch::empty({batch, h}, options.dtype(torch::kBFloat16));
    online_cfg_.x = reinterpret_cast<const __nv_bfloat16*>(online_x_.data_ptr<at::BFloat16>());
    online_cfg_.ids = online_ids_.data_ptr<int>();
    online_cfg_.routing = online_routing_.data_ptr<float>();
    online_cfg_.group_meta = online_meta_.data_ptr<int>();
    online_cfg_.group_slots = online_slots_.data_ptr<int>();
    online_cfg_.group_count = online_count_.data_ptr<int>();
    // Build all prefix views before serving; forward only copies and launches.
    online_inputs_.resize(batch + 1);
    for (int active = 1; active <= batch; ++active)
      online_inputs_[active] = {online_x_.narrow(0, 0, active), online_routing_.narrow(0, 0, active)};
  }

  void bind_weights(const std::vector<torch::Tensor>& raw) {
    validate_weights(raw);
    weights_ = raw;  // Keep original storage alive through all queued stages.
    cfg_.weights = {raw[0].data_ptr<unsigned char>(), raw[1].data_ptr<unsigned char>(),
                    raw[2].data_ptr<unsigned char>(), raw[3].data_ptr<unsigned char>(),
                    raw[4].data_ptr<unsigned char>(), raw[5].data_ptr<unsigned char>()};
  }

  std::shared_ptr<StagedBoundWeights> bind_online_weights(const std::vector<torch::Tensor>& raw) {
    c10::cuda::CUDAGuard guard(device_);
    validate_weights(raw);
    return std::make_shared<StagedBoundWeights>(shared_from_this(), raw);
  }

  void run_online(torch::Tensor x, torch::Tensor ids, torch::Tensor routing,
                  const std::shared_ptr<StagedBoundWeights>& bound, float clamp = 10.f) {
    c10::cuda::CUDAGuard guard(device_);
    TORCH_CHECK(bound && bound->owner.get() == this, "online bound weights belong to another workspace");
    TORCH_CHECK(x.dim() == 2 && x.size(0) >= 1 && x.size(0) <= cfg_.batch,
                "online decode batch must be in 1..workspace capacity");
    const int batch = int(x.size(0));
    validate(x, torch::kBFloat16, {batch, cfg_.h}, "online input");
    TORCH_CHECK(ids.scalar_type() == torch::kInt32 || ids.scalar_type() == torch::kInt64,
                "online IDs must be int32 or int64");
    TORCH_CHECK(routing.scalar_type() == torch::kFloat32 || routing.scalar_type() == torch::kBFloat16 ||
                routing.scalar_type() == torch::kFloat16, "online routing must be float32, bfloat16 or float16");
    validate(ids, ids.scalar_type(), {batch, cfg_.topk}, "online IDs");
    validate(routing, routing.scalar_type(), {batch, cfg_.topk}, "online routing");
    TORCH_CHECK(std::isfinite(clamp) && clamp > 0.f, "staged clamp must be finite and positive");
    online_inputs_[batch][0].copy_(x);
    online_inputs_[batch][1].copy_(routing);
    StagedConfig cfg = online_cfg_;
    cfg.batch = batch; cfg.clamp = clamp; cfg.weights = bound->weights;
    auto stream = at::cuda::getCurrentCUDAStream(device_).stream();
    STAGED_CUDA_OK(cudaMemsetAsync(cfg.slot_outputs, 0, size_t(batch) * cfg.topk * cfg.h * sizeof(float), stream));
    online_groups<<<1, 1, 0, stream>>>(cfg, online_meta_.data_ptr<int>(),
        online_slots_.data_ptr<int>(), online_count_.data_ptr<int>(),
        ids.scalar_type() == torch::kInt32 ? ids.data_ptr<int>() : nullptr,
        ids.scalar_type() == torch::kInt64 ? ids.data_ptr<int64_t>() : nullptr, online_ids_.data_ptr<int>());
    STAGED_CUDA_OK(cudaGetLastError());
    // Capacity is host-known. staged_compute guards against actual group_count;
    // the direct grid intentionally includes inactive capacity-bound blocks.
    staged_compute<true><<<weave::staged_blocks(0, cfg.capacity, weave::tile_columns(cfg.i)),
                           THREADS, SHARED_BYTES, stream>>>(cfg);
    STAGED_CUDA_OK(cudaGetLastError());
    staged_compute<false><<<weave::staged_blocks(0, cfg.capacity, weave::tile_columns(cfg.h)),
                            THREADS, SHARED_BYTES, stream>>>(cfg);
    STAGED_CUDA_OK(cudaGetLastError());
    const int blocks = (batch * cfg.topk * cfg.h + THREADS - 1) / THREADS;
    online_routed_slots<<<blocks, THREADS, 0, stream>>>(cfg);
    STAGED_CUDA_OK(cudaGetLastError());
    online_active_batch_ = batch;
  }

  // The caller performs NCCL SUM on the active FP32 slot prefix first.
  void finalize_online(int batch) {
    c10::cuda::CUDAGuard guard(device_);
    TORCH_CHECK(batch >= 1 && batch <= cfg_.batch && batch == online_active_batch_,
                "finalize batch must match the preceding online run");
    StagedConfig cfg = online_cfg_;
    cfg.batch = batch;
    auto stream = at::cuda::getCurrentCUDAStream(device_).stream();
    const int blocks = (batch * cfg.h + THREADS - 1) / THREADS;
    online_finalize<<<blocks, THREADS, 0, stream>>>(cfg,
        reinterpret_cast<__nv_bfloat16*>(online_output_.data_ptr<at::BFloat16>()));
    STAGED_CUDA_OK(cudaGetLastError());
  }

  void bind_inputs(torch::Tensor x, torch::Tensor ids, torch::Tensor routing,
                   torch::Tensor group_meta, torch::Tensor group_slots, torch::Tensor group_count) {
    const int b = cfg_.batch, h = cfg_.h, k = cfg_.topk;
    validate(x, torch::kBFloat16, {b, h}, "input");
    validate(ids, torch::kInt32, {b, k}, "global IDs");
    validate(routing, torch::kFloat32, {b, k}, "routing weights");
    TORCH_CHECK(group_meta.dim() == 2 && group_meta.size(0) >= 1 && group_meta.size(0) <= b * k,
                "group metadata capacity must be in 1..batch*topk");
    const int capacity = int(group_meta.size(0));
    validate(group_meta, torch::kInt32, {capacity, 2}, "group metadata");
    validate(group_slots, torch::kInt32, {capacity, weave::TILE_ROWS}, "group slots");
    validate(group_count, torch::kInt32, {1}, "device group count");
    inputs_ = {x, ids, routing, group_meta, group_slots, group_count};
    cfg_.x = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>());
    cfg_.ids = ids.data_ptr<int>(); cfg_.routing = routing.data_ptr<float>();
    cfg_.group_meta = group_meta.data_ptr<int>(); cfg_.group_slots = group_slots.data_ptr<int>();
    cfg_.group_count = group_count.data_ptr<int>(); cfg_.capacity = capacity;
  }

  // Bind once, then use these three calls in this order on the same current
  // stream for stage events. No GPU reads, synchronization, or allocation.
  void run_gate_up(int grid, float clamp = 10.f) {
    c10::cuda::CUDAGuard guard(device_);
    require_bound(); validate_grid(grid);
    TORCH_CHECK(std::isfinite(clamp) && clamp > 0.f, "staged clamp must be finite and positive");
    StagedConfig cfg = cfg_;
    cfg.clamp = clamp;
    const int blocks = weave::staged_blocks(grid, cfg.capacity, weave::tile_columns(cfg.i));
    auto stream = at::cuda::getCurrentCUDAStream(device_).stream();
    staged_compute<true><<<blocks, THREADS, SHARED_BYTES, stream>>>(cfg);
    STAGED_CUDA_OK(cudaGetLastError());
  }

  void run_down(int grid) {
    c10::cuda::CUDAGuard guard(device_);
    require_bound(); validate_grid(grid);
    const int blocks = weave::staged_blocks(grid, cfg_.capacity, weave::tile_columns(cfg_.h));
    auto stream = at::cuda::getCurrentCUDAStream(device_).stream();
    staged_compute<false><<<blocks, THREADS, SHARED_BYTES, stream>>>(cfg_);
    STAGED_CUDA_OK(cudaGetLastError());
  }

  void run_reduce() {
    c10::cuda::CUDAGuard guard(device_);
    require_bound();
    const int blocks = (cfg_.batch * cfg_.h + THREADS - 1) / THREADS;
    auto stream = at::cuda::getCurrentCUDAStream(device_).stream();
    staged_reduce<<<blocks, THREADS, 0, stream>>>(cfg_);
    STAGED_CUDA_OK(cudaGetLastError());
  }

  void run(torch::Tensor x, torch::Tensor ids, torch::Tensor routing,
           torch::Tensor group_meta, torch::Tensor group_slots, torch::Tensor group_count,
           int grid, float clamp = 10.f) {
    bind_inputs(x, ids, routing, group_meta, group_slots, group_count);
    run_gate_up(grid, clamp);
    run_down(grid);
    run_reduce();
  }

  torch::Tensor output() const { return output_; }
  torch::Tensor slot_outputs() const { return slots_; }
  torch::Tensor online_slot_outputs() const { return slots_; }
  torch::Tensor online_output() const { return online_output_; }
  torch::Tensor online_group_meta() const { return online_meta_; }
  torch::Tensor online_group_slots() const { return online_slots_; }
  torch::Tensor online_group_count() const { return online_count_; }

  pybind11::dict resources() const {
    c10::cuda::CUDAGuard guard(device_);
    pybind11::dict result, stages;
    stages["gate_up"] = kernel_resources(staged_compute<true>, SHARED_BYTES);
    stages["down"] = kernel_resources(staged_compute<false>, SHARED_BYTES);
    stages["reduce"] = kernel_resources(staged_reduce, 0);
    result["stages"] = stages;
    result["device"] = device_; result["multiprocessors"] = multiprocessors_;
    result["grid_controls"] = std::vector<int>{132, 264, 528, 0};
    result["group_capacity"] = cfg_.capacity;
    result["scratch_group_capacity"] = cfg_.batch * cfg_.topk;
    result["gate_tiles_per_group"] = weave::tile_columns(cfg_.i);
    result["down_tiles_per_group"] = weave::tile_columns(cfg_.h);
    result["reduce_grid_blocks"] = (cfg_.batch * cfg_.h + THREADS - 1) / THREADS;
    result["output_dtype"] = "float32";
    result["slot_output_boundary"] = "BF16-rounded down output stored as float32";
    result["online_group_capacity"] = online_cfg_.capacity;
    result["online_gate_grid_bound"] = weave::staged_blocks(0, online_cfg_.capacity, weave::tile_columns(cfg_.i));
    result["online_down_grid_bound"] = weave::staged_blocks(0, online_cfg_.capacity, weave::tile_columns(cfg_.h));
    result["online_grid_policy"] = "capacity_bound_direct_with_device_count_guard";
    result["online_max_batch"] = cfg_.batch;
    result["online_slot_output_dtype"] = "float32";
    result["online_output_dtype"] = "bfloat16";
    result["online_slot_output_boundary"] = "BF16-rounded routed products in zeroed float32 slots before NCCL SUM";
    result["online_output_boundary"] = "ordered global slot FP32 sum then BF16 contiguous origin mask";
    return result;
  }

 private:
  int device_ = -1, multiprocessors_ = 0;
  StagedConfig cfg_{};
  StagedConfig online_cfg_{};
  int online_active_batch_ = 0;
  torch::Tensor mid_, slots_, output_;
  torch::Tensor online_x_, online_ids_, online_routing_, online_meta_, online_slots_, online_count_, online_output_;
  std::vector<std::array<torch::Tensor, 2>> online_inputs_;
  std::vector<torch::Tensor> weights_, inputs_;

  void validate_weights(const std::vector<torch::Tensor>& raw) const {
    TORCH_CHECK(raw.size() == 6, "staged binding requires the six original MXFP4 tensors");
    const int local = cfg_.experts / cfg_.ep, h = cfg_.h, i = cfg_.i;
    const std::vector<std::vector<int64_t>> shapes = {
      {local, i, h / 2}, {local, i, h / 2}, {local, h, i / 2},
      {local, i, h / 32}, {local, i, h / 32}, {local, h, i / 32}};
    for (int j = 0; j < 6; ++j) validate(raw[j], torch::kUInt8, shapes[j], "raw MXFP4 weight");
  }

  void validate(const torch::Tensor& tensor, torch::ScalarType dtype,
                const std::vector<int64_t>& shape, const char* name) const {
    TORCH_CHECK(tensor.is_cuda() && tensor.get_device() == device_ && tensor.is_contiguous()
                && tensor.scalar_type() == dtype && tensor.sizes().vec() == shape,
                "invalid staged ", name, " tensor: check original shape, dtype, device, and contiguity");
  }

  static void validate_grid(int grid) {
    TORCH_CHECK(grid == 0 || grid == 132 || grid == 264 || grid == 528,
                "staged grid must be 132, 264, 528, or 0 for direct mapping");
  }

  void require_bound() const {
    TORCH_CHECK(weights_.size() == 6 && inputs_.size() == 6,
                "bind original weights and frozen inputs before running staged kernels");
  }

  template<class Kernel>
  static pybind11::dict kernel_resources(Kernel kernel, int dynamic_shared) {
    cudaFuncAttributes attrs;
    STAGED_CUDA_OK(cudaFuncGetAttributes(&attrs, kernel));
    int occupancy = 0;
    STAGED_CUDA_OK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy, kernel, THREADS, dynamic_shared));
    pybind11::dict result;
    result["numRegs"] = attrs.numRegs; result["localSizeBytes"] = attrs.localSizeBytes;
    result["sharedSizeBytes"] = attrs.sharedSizeBytes;
    result["maxDynamicSharedSizeBytes"] = attrs.maxDynamicSharedSizeBytes;
    result["maxThreadsPerBlock"] = attrs.maxThreadsPerBlock;
    result["dynamic_shared_bytes"] = dynamic_shared;
    result["threads_per_block"] = THREADS;
    result["active_blocks_per_sm"] = occupancy;
    result["binaryVersion"] = attrs.binaryVersion;
    result["ptxVersion"] = attrs.ptxVersion;
    return result;
  }
};
}  // namespace

void bind_staged(pybind11::module_& module) {
  namespace py = pybind11;
  py::class_<StagedBoundWeights, std::shared_ptr<StagedBoundWeights>>(module, "StagedBoundWeights");
  py::class_<StagedWorkspace, std::shared_ptr<StagedWorkspace>>(module, "StagedWorkspace")
      .def(py::init<int, int, int, int, int, int, int>(),
           py::arg("rank"), py::arg("ep"), py::arg("batch"), py::arg("h"), py::arg("i"),
           py::arg("experts"), py::arg("topk"))
      .def("bind_weights", &StagedWorkspace::bind_weights, py::arg("raw"))
      .def("bind_online_weights", &StagedWorkspace::bind_online_weights, py::arg("raw"))
      .def("run_online", &StagedWorkspace::run_online, py::arg("x"), py::arg("ids"),
           py::arg("routing"), py::arg("bound"), py::arg("clamp") = 10.f)
      .def("finalize_online", &StagedWorkspace::finalize_online, py::arg("batch"))
      .def("bind_inputs", &StagedWorkspace::bind_inputs, py::arg("x"), py::arg("ids"),
           py::arg("routing"), py::arg("group_meta"), py::arg("group_slots"), py::arg("group_count"))
      .def("run", &StagedWorkspace::run, py::arg("x"), py::arg("ids"), py::arg("routing"),
           py::arg("group_meta"), py::arg("group_slots"), py::arg("group_count"),
           py::arg("grid"), py::arg("clamp") = 10.f)
      .def("run_gate_up", &StagedWorkspace::run_gate_up, py::arg("grid"), py::arg("clamp") = 10.f)
      .def("run_down", &StagedWorkspace::run_down, py::arg("grid"))
      .def("run_reduce", &StagedWorkspace::run_reduce)
      .def_property_readonly("output", &StagedWorkspace::output)
      .def_property_readonly("slot_outputs", &StagedWorkspace::slot_outputs)
      .def_property_readonly("online_slot_outputs", &StagedWorkspace::online_slot_outputs)
      .def_property_readonly("online_output", &StagedWorkspace::online_output)
      .def_property_readonly("online_group_meta", &StagedWorkspace::online_group_meta)
      .def_property_readonly("online_group_slots", &StagedWorkspace::online_group_slots)
      .def_property_readonly("online_group_count", &StagedWorkspace::online_group_count)
      .def("resources", &StagedWorkspace::resources);
}

#undef STAGED_CUDA_OK
