// This tail is appended after the source-pinned native launch-selection code.
// Its kernel function pointers refer to StaticMarlin, so resource measurements
// describe the actual extracted helper binary, not the original kernel.
inline int64_t tensor_elements(tvm::ffi::TensorView value) {
  int64_t size=1;
  for (int axis=0;axis<value.dim();++axis) size*=value.size(axis);
  return size;
}

inline MarlinFuncPtr down_kernel(int block, int threads, int nb, int kb) {
  return get_marlin_kernel<nv_bfloat16,true,false>(host::kFE2M1f,
      div_ceil(block,16), nb, kb, block == 8, false, false, 2, threads, false);
}

inline void configure_down(int64_t rows, int64_t n, int64_t k,
                           int64_t block, int64_t dev, tvm::ffi::TensorView result) {
  RuntimeCheck(rows > 0 && n > 0 && k > 0 && n % 128 == 0 && k % 64 == 0,
               "invalid full-K MXFP4 task shape");
  RuntimeCheck(block == 8 || block == 16 || block == 32 || block == 48 || block == 64,
               "unsupported native M block");
  RuntimeCheck(result.device().device_type == kDLCPU && tensor_elements(result) == 16 &&
               result.dtype().bits == 64 && result.is_contiguous(), "invalid CPU geometry buffer");
  int current = -1, maximum = 0, sms = 0, major = 0;
  RuntimeDeviceCheck(cudaGetDevice(&current));
  RuntimeCheck(current == dev, "set current CUDA device before prepare");
  RuntimeDeviceCheck(cudaDeviceGetAttribute(&major,cudaDevAttrComputeCapabilityMajor,dev));
  RuntimeCheck(major >= 8, "native MXFP4 down needs SM80 or newer");
  RuntimeDeviceCheck(cudaDeviceGetAttribute(&maximum,cudaDevAttrMaxSharedMemoryPerBlockOptin,dev));
  RuntimeDeviceCheck(cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,dev));
  const auto config = determine_exec_config<nv_bfloat16,true,false>(
      host::kFE2M1f,rows,n,k,1,div_ceil(block,16),block == 8,4,32,
      false,true,false,false,maximum,sms);
  const auto tile = config.tb_cfg;
  RuntimeCheck(tile.num_threads > 0, "no valid native down specialization");
  const auto kernel = down_kernel(block,tile.num_threads,tile.thread_n/16,tile.thread_k/16);
  RuntimeCheck(kernel != MarlinDefault, "missing extracted native down specialization");
  cudaFuncAttributes attr;
  RuntimeDeviceCheck(cudaFuncGetAttributes(&attr,kernel));
  int scratch = config.blocks_per_sm > 1
      ? maximum / config.blocks_per_sm - kSharedMemoryLaunchReserve : maximum;
  scratch = (min(scratch,maximum - int(attr.sharedSizeBytes)) / 16) * 16;
  RuntimeCheck(is_valid_config(tile,block == 8,div_ceil(block,16),rows,n,k,
      4,32,false,true,false,false,scratch), "selected extracted down scratch does not fit");
  RuntimeDeviceCheck(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,scratch));
  int active = 0;
  RuntimeDeviceCheck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,kernel,tile.num_threads,scratch));
  RuntimeCheck(active > 0, "extracted down has zero measured residency");
  int64_t* out = static_cast<int64_t*>(result.data_ptr());
  const int64_t values[] = {tile.num_threads,div_ceil(block,16),tile.thread_n/16,
      tile.thread_k/16,block == 8,pipe_stages,scratch,attr.numRegs,
      int64_t(attr.sharedSizeBytes),int64_t(attr.localSizeBytes),active,sms,
      config.blocks_per_sm,attr.maxThreadsPerBlock,attr.binaryVersion,attr.ptxVersion};
  for (int i=0;i<16;++i) out[i]=values[i];
}

inline void launch_down(tvm::ffi::TensorView a, tvm::ffi::TensorView weight,
    tvm::ffi::TensorView scales, tvm::ffi::TensorView sorted,
    tvm::ffi::TensorView experts, tvm::ffi::TensorView padded,
    tvm::ffi::TensorView routing, tvm::ffi::TensorView output,
    tvm::ffi::TensorView visits, tvm::ffi::TensorView error,
    int64_t block,int64_t threads,int64_t nb,int64_t kb,int64_t scratch_bytes,int64_t grid_workers) {
  auto device = SymbolicDevice{};
  device.set_options<kDLCUDA>();
  TensorMatcher({-1,-1}).with_dtype<nv_bfloat16>().with_device(device).verify(a);
  TensorMatcher({a.size(0),-1}).with_dtype<nv_bfloat16>().with_device(device).verify(output);
  TensorMatcher({-1}).with_dtype<int32_t>().with_device(device).verify(sorted);
  TensorMatcher({-1}).with_dtype<int32_t>().with_device(device).verify(experts);
  TensorMatcher({1}).with_dtype<int32_t>().with_device(device).verify(padded);
  TensorMatcher({-1}).with_dtype<int64_t>().with_device(device).verify(visits);
  TensorMatcher({1}).with_dtype<int32_t>().with_device(device).verify(error);
  device.verify(weight.device()); device.verify(scales.device()); device.verify(routing.device());
  RuntimeCheck(a.is_contiguous() && output.is_contiguous() && weight.is_contiguous() &&
      scales.is_contiguous() && sorted.is_contiguous() && experts.is_contiguous() &&
      padded.is_contiguous() && routing.is_contiguous() && visits.is_contiguous() &&
      error.is_contiguous(), "all native tail tensors must be contiguous");
  const int rows=a.size(0), n=output.size(1), k=a.size(1);
  RuntimeCheck(block == 8 || block == 16 || block == 32 || block == 48 || block == 64,
               "unsupported native M block");
  const int capacity=min(int(tensor_elements(experts)),int(tensor_elements(sorted)/block));
  RuntimeCheck(nb > 0 && kb > 0 && n % (16*nb) == 0 && k % (16*kb) == 0,
               "invalid native N/K tile divisibility");
  RuntimeCheck(weight.dim() == 3 && weight.size(1) == k/16 && weight.size(2) == 2*n &&
      weight.dtype().bits == 32, "invalid native packed MXFP4 layout");
  RuntimeCheck(scales.dim() == 3 && scales.size(0) == weight.size(0) &&
      scales.size(1) == k/32 && scales.size(2) == n && scales.dtype().bits == 8,
      "invalid native E8M0 scale layout");
  RuntimeCheck(tensor_elements(routing) == rows && routing.dtype().bits == 32 &&
      routing.dtype().code == kDLFloat, "invalid routing weights");
  RuntimeCheck(capacity > 0,
      "native grouping capacity exceeds sorted IDs storage");
  const int tasks=capacity*(n/(16*nb));
  RuntimeCheck(tensor_elements(visits) == 0 || tensor_elements(visits) == tasks, "task bitmap capacity mismatch");
  RuntimeCheck(scratch_bytes > 0, "prepare down geometry before capture");
  RuntimeCheck(grid_workers >= 0 && grid_workers <= tasks, "invalid static worker count");
  const int grid=grid_workers ? int(grid_workers) : tasks;
  const auto kernel=down_kernel(block,threads,nb,kb);
  RuntimeCheck(kernel != MarlinDefault, "unsupported native down specialization");
  DownTileArgs args{static_cast<const int4*>(a.data_ptr()),
      static_cast<const int4*>(weight.data_ptr()),static_cast<const int4*>(scales.data_ptr()),
      static_cast<int4*>(output.data_ptr()),static_cast<const int32_t*>(sorted.data_ptr()),
      static_cast<const int32_t*>(experts.data_ptr()),static_cast<const int32_t*>(padded.data_ptr()),
      static_cast<const float*>(routing.data_ptr()),rows,n,k,int(weight.size(0)),capacity};
  auto stream=LaunchKernel::resolve_device(device.unwrap());
  kernel<<<grid,threads,scratch_bytes,stream>>>(args,
      tensor_elements(visits) ? static_cast<int64_t*>(visits.data_ptr()) : nullptr,
      static_cast<int*>(error.data_ptr()),scratch_bytes);
  RuntimeDeviceCheck(cudaGetLastError());
}
}  // namespace weave::native_tail
