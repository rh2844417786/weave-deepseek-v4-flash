"""Source-pinned private native Marlin overlay for measured CTA-pair budgets.

The arithmetic and attribute-aware native physical launch selection stay native.
Both GEMMs consume one routing plan in a fixed captured launch grid.
"""
import hashlib
import json
from pathlib import Path
import types

from .native_budget import ROOT, source_pins
from .native_budget_patch import replace_exact, verify_sources

PREFIX='kernels/jit/csrc/gemm/marlin_moe/'
OP='kernels/ops/moe/moe_wna16_marlin.py'
PATCHED=(PREFIX+'kernel.h',PREFIX+'marlin_template.h',PREFIX+'moe_wna16_marlin.cuh',OP)
KERNEL_EXTRA='int64_t* budget_state, const int64_t* cost_plan, int cost_role'

PLAN_FFI = r'''
// Shape/device verification is shared by the standalone overhead probe and the
// integrated path. The device planner never copies grouping or scalars to host.
void weave_cost_plan(
    tvm::ffi::TensorView sorted_ids, tvm::ffi::TensorView expert_ids,
    tvm::ffi::TensorView padded, int64_t block_size, int64_t total_rows,
    tvm::ffi::TensorView plan, tvm::ffi::TensorView table,
    tvm::ffi::TensorView control) {
  using namespace host;
  auto device = SymbolicDevice{};
  device.set_options<kDLCUDA>();
  TensorMatcher({-1}).with_dtype<int32_t>().with_device(device).verify(sorted_ids);
  TensorMatcher({-1}).with_dtype<int32_t>().with_device(device).verify(expert_ids);
  TensorMatcher({1}).with_dtype<int32_t>().with_device(device).verify(padded);
  TensorMatcher({32}).with_dtype<int64_t>().with_device(device).verify(plan);
  TensorMatcher({9,6,8}).with_dtype<float>().with_device(device).verify(table);
  TensorMatcher({3}).with_dtype<int32_t>().with_device(device).verify(control);
  RuntimeCheck(sorted_ids.is_contiguous() && expert_ids.is_contiguous() && padded.is_contiguous()
      && plan.is_contiguous() && table.is_contiguous() && control.is_contiguous(),
      "native cost grouping and plan/table/control must be contiguous");
  RuntimeCheck(total_rows >= 1 && total_rows <= 192, "native cost supports up to B32/topk6 routed rows");
  RuntimeCheck(block_size == 8 || (block_size >= 16 && block_size <= 64 && block_size % 16 == 0),
      "invalid native cost grouping block size");
  RuntimeCheck(sorted_ids.size(0) >= total_rows && sorted_ids.size(0) <= 192 + 256 * 63
      && expert_ids.size(0) > 0 && expert_ids.size(0) <= 192 + 256,
      "invalid native cost grouping capacity");
  auto stream = LaunchKernel::resolve_device(device.unwrap());
  weave_cost_plan_kernel<<<1,128,0,stream>>>(
      static_cast<const int32_t*>(sorted_ids.data_ptr()),
      static_cast<const int32_t*>(expert_ids.data_ptr()),
      static_cast<const int32_t*>(padded.data_ptr()),
      int(sorted_ids.size(0)), int(expert_ids.size(0)), int(block_size), int(total_rows),
      static_cast<int64_t*>(plan.data_ptr()), static_cast<const float*>(table.data_ptr()),
      static_cast<const int32_t*>(control.data_ptr()));
  RuntimeDeviceCheck(cudaGetLastError());
}
'''


def patch_sources(sources):
    pins=source_pins();result={}
    for name in PATCHED:
        text=sources[name]
        if hashlib.sha256(text.encode()).hexdigest()!=pins[name]:
            raise RuntimeError('native cost source mismatch: '+name)
        result[name]=text
    name=PREFIX+'kernel.h'
    result[name]='#include <cstdint>\n'+replace_exact(result[name],
        'int max_shared_mem\n','int max_shared_mem, '+KERNEL_EXTRA+'\n')

    name=PREFIX+'marlin_template.h'
    text='#include "native_cost.h"\n'+result[name]
    text=replace_exact(text,'int max_shared_mem)','int max_shared_mem, '+KERNEL_EXTRA+')',2)
    text=replace_exact(text,'gridDim.x','weave_workers',3)
    # The original EP valid-block scan ran in every Marlin CTA. Planning has
    # already counted this grouping once; retain native block remapping later.
    old='''  int num_tokens_past_padded = num_tokens_past_padded_ptr[0];
  int parallel = num_tokens_past_padded / moe_block_size;
  int num_valid_blocks = parallel;
  if constexpr (kIsEP) {
    for (int i = 0; i < parallel; i++) {
      if (expert_ids_ptr[i] == -1) num_valid_blocks--;
    }
  }
  int num_invalid_blocks = parallel - num_valid_blocks;
  parallel = num_valid_blocks;'''
    new='''  // Grouping statistics come from one preceding planner CTA.
  const int num_tokens_past_padded = int(cost_plan[2]) * moe_block_size;
  const int parallel = int(cost_plan[3]);
  const int num_invalid_blocks = int(cost_plan[2]) - parallel;'''
    text=replace_exact(text,old,new)
    insertion='''
  const int weave_workers = cost_plan[23] ? 0 : weave_cost_workers(
      int(gridDim.x), int(cost_plan[cost_role]), parallel, n_tiles, k_tiles);
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    budget_state[0] += 1;
    budget_state[1] += (weave_workers == 0);
    budget_state[2] = gridDim.x;
    budget_state[3] = weave_workers;
    budget_state[4] = cost_plan[2];
    budget_state[5] = parallel;
    budget_state[6] = n_tiles;
    budget_state[7] = k_tiles;
    const int effective_bucket = (weave_workers * 8 + int(gridDim.x) - 1) / int(gridDim.x);
    budget_state[8 + effective_bucket] += 1;
    budget_state[17] = cost_plan[4];
    budget_state[18] = cost_plan[5];
  }
  // All inactive workers leave uniformly before shared-memory/lock operations.
  if (int(blockIdx.x) >= weave_workers) return;
'''
    text=replace_exact(text,'  int iters = div_ceil(k_tiles * n_tiles * parallel, weave_workers);',
                       insertion+'  int iters = div_ceil(k_tiles * n_tiles * parallel, weave_workers);')
    result[name]=text

    name=PREFIX+'moe_wna16_marlin.cuh'
    text=result[name]
    prefix='    bool use_atomic_add,\n    bool use_fp32_reduce,\n'
    anchor=prefix+'    bool is_zp_float) {'
    if text.count(anchor)!=2:raise RuntimeError('native cost host signature count differs')
    text=text.replace(anchor,prefix+'    bool is_zp_float, '+KERNEL_EXTRA+') {',1)
    text=replace_exact(text,anchor,prefix+'''    bool is_zp_float,
    tvm::ffi::TensorView budget_state, tvm::ffi::TensorView cost_plan,
    tvm::ffi::TensorView cost_table, tvm::ffi::TensorView cost_control,
    int64_t cost_role, bool cost_refresh) {''')
    text=replace_exact(text,'use_fp32_reduce, max_shared_mem);',
        'use_fp32_reduce, max_shared_mem, budget_state, cost_plan, cost_role);')
    text=replace_exact(text,'      use_fp32_reduce,\n      is_zp_float);',
        '''      use_fp32_reduce,
      is_zp_float,
      static_cast<int64_t*>(budget_state.data_ptr()),
      static_cast<const int64_t*>(cost_plan.data_ptr()), static_cast<int>(cost_role));''')
    text=replace_exact(text,'}  // namespace device::marlin_moe\n',
                       '}  // namespace device::marlin_moe\n'+PLAN_FFI)
    anchor='  TensorMatcher({-1, -1}).with_dtype<scalar_t>().with_device(device).verify(a);'
    text=replace_exact(text,anchor,anchor+'''
  TensorMatcher({19}).with_dtype<int64_t>().with_device(device).verify(budget_state);
  TensorMatcher({32}).with_dtype<int64_t>().with_device(device).verify(cost_plan);
  TensorMatcher({9,6,8}).with_dtype<float>().with_device(device).verify(cost_table);
  TensorMatcher({3}).with_dtype<int32_t>().with_device(device).verify(cost_control);
  RuntimeCheck(budget_state.is_contiguous() && cost_plan.is_contiguous()
      && cost_table.is_contiguous() && cost_control.is_contiguous(), "native cost buffers must be contiguous");
  RuntimeCheck(kIsEP && b_q_weight.size(0) == 64, "native cost requires EP4 with 64 local experts");
  RuntimeCheck(cost_role == 0 || cost_role == 1, "native cost role must be gate/up(0) or down(1)");
  RuntimeCheck(size_m * top_k >= 1 && size_m * top_k <= 192, "native cost is bounded to B32/topk6");''')
    anchor='  // Early return for zero-size M (moved after all validation)'
    text=replace_exact(text,anchor,'''  if (cost_role == 0 || cost_refresh) {
    weave_cost_plan(sorted_token_ids, expert_ids, num_tokens_post_padded, moe_block_size,
        size_m * top_k, cost_plan, cost_table, cost_control);
  }

'''+anchor)
    result[name]=text

    text=result[OP]
    text=replace_exact(text,'        "moe_wna16_marlin",','        "weave_cost_marlin",')
    text=replace_exact(text,'        cuda_files=["gemm/marlin_moe/moe_wna16_marlin.cuh"],',
                       '        external_cuda_files=[WEAVE_COST_HEADER],')
    text=replace_exact(text,'        cuda_wrappers=[','        cuda_wrappers=[\n            ("weave_cost_plan", "weave_cost_plan"),')
    text=replace_exact(text,'    is_zp_float: bool = False,\n','''    is_zp_float: bool = False,
    *,
    budget_state: torch.Tensor,
    cost_plan: torch.Tensor,
    cost_table: torch.Tensor,
    cost_control: torch.Tensor,
    cost_role: int,
    cost_refresh: bool = False,
''')
    text=replace_exact(text,'        is_zp_float,\n',
                       '        is_zp_float,\n        budget_state, cost_plan, cost_table, cost_control, cost_role, cost_refresh,\n')
    text=replace_exact(text,'    module = _jit_moe_wna16_marlin_module(a.dtype, is_ep, has_bias)',
                       '''    module = _jit_moe_wna16_marlin_module(a.dtype, is_ep, has_bias)
    global WEAVE_PLANNER_MODULE
    WEAVE_PLANNER_MODULE = module''')
    text+='''

def weave_plan(sorted_ids, expert_ids, padded, block_size, total_rows, plan, table, control):
    """Execute the same compiled planner alone for calibration overhead timing."""
    if WEAVE_PLANNER_MODULE is None:
        raise RuntimeError("call weave_prepare before timing the native cost planner")
    WEAVE_PLANNER_MODULE.weave_cost_plan(
        sorted_ids, expert_ids, padded, block_size, total_rows, plan, table, control)
'''
    result[OP]=text
    return result


def build_overlay(package_root, cache_root):
    package_root,cache_root=Path(package_root),Path(cache_root)
    pins=verify_sources(package_root)
    changed=patch_sources({name:(package_root/name).read_text() for name in PATCHED})
    helper=(ROOT/'csrc/native_cost.h').read_bytes()
    digest=hashlib.sha256(json.dumps(pins,sort_keys=True).encode()+helper)
    for name,text in sorted(changed.items()):digest.update(name.encode()+b'\0'+text.encode())
    identity=digest.hexdigest();overlay=cache_root/('marlin-cost-'+identity)
    overlay.mkdir(parents=True,exist_ok=True)
    for name in pins:
        if name.startswith('kernels/jit/csrc/'):
            target=overlay/name;target.parent.mkdir(parents=True,exist_ok=True)
            raw=changed[name].encode() if name in changed else (package_root/name).read_bytes()
            if not target.exists() or target.read_bytes()!=raw:target.write_bytes(raw)
    helper_path=overlay/PREFIX/'native_cost.h'
    if not helper_path.exists() or helper_path.read_bytes()!=helper:helper_path.write_bytes(helper)
    (overlay/PREFIX/'cost_op.py').write_text(changed[OP])
    return changed[OP],overlay/PREFIX/'moe_wna16_marlin.cuh',identity


def load_cost_op(package_root, cache_root):
    source,header,identity=build_overlay(package_root,cache_root)
    module=types.ModuleType('weave_native_cost_marlin_'+identity)
    module.__file__=str(header.parent/'cost_op.py')
    module.WEAVE_COST_HEADER=str(header)
    module.WEAVE_PLANNER_MODULE=None
    exec(compile(source,module.__file__,'exec'),module.__dict__)
    op=module.moe_wna16_marlin_gemm
    def prepare(dtype,is_ep,has_bias):
        compiled=module._jit_moe_wna16_marlin_module(dtype,is_ep,has_bias)
        module.WEAVE_PLANNER_MODULE=compiled
        return compiled
    op.weave_prepare=prepare
    op.weave_overlay_sha256=identity
    op.weave_plan=module.weave_plan
    return op
