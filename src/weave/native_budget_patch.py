"""Build a private, source-verified Marlin JIT overlay. Installed files stay intact."""
import hashlib
import json
from pathlib import Path
import types

from .native_budget import ROOT, source_pins

PREFIX = 'kernels/jit/csrc/gemm/marlin_moe/'
OP = 'kernels/ops/moe/moe_wna16_marlin.py'
PATCHED = (PREFIX + 'kernel.h', PREFIX + 'marlin_template.h',
           PREFIX + 'moe_wna16_marlin.cuh', OP)
EXTRA = 'int64_t* budget_state, int budget_target_percent, int budget_ep_size, int budget_mode'


def verify_sources(package_root):
    pins = source_pins()
    for name, expected in pins.items():
        path = Path(package_root) / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError('native budget source mismatch: ' + name)
    return pins


def replace_exact(text, old, new, count=1):
    if text.count(old) != count:
        raise RuntimeError('native budget patch anchor differs: ' + old[:80])
    return text.replace(old, new)


def patch_sources(sources):
    pins = source_pins()
    result = {}
    for name in PATCHED:
        text = sources[name]
        if hashlib.sha256(text.encode()).hexdigest() != pins[name]:
            raise RuntimeError('native budget source mismatch: ' + name)
        result[name] = text
    name = PREFIX + 'kernel.h'
    result[name] = '#include <cstdint>\n' + replace_exact(
        result[name], 'int max_shared_mem\n', 'int max_shared_mem, ' + EXTRA + '\n')
    name = PREFIX + 'marlin_template.h'
    text = '#include "native_budget.h"\n' + result[name]
    text = replace_exact(text, 'int max_shared_mem)', 'int max_shared_mem, ' + EXTRA + ')', 2)
    # All three uses govern both stripe ownership and reduction lock layout.
    text = replace_exact(text, 'gridDim.x', 'weave_workers', 3)
    insertion = '''
  // Virtual grid is recomputed from current device routing on every replay.
  const int weave_total_routes = prob_m * top_k;
  const int weave_local_routes = weave_budget_local_routes(sorted_token_ids_ptr,
      expert_ids_ptr, num_tokens_past_padded, moe_block_size, weave_total_routes);
  const auto weave_choice = weave_budget_choose(int(gridDim.x), weave_local_routes,
      weave_total_routes, parallel, n_tiles, k_tiles, budget_target_percent,
      budget_ep_size, budget_mode);
  const int weave_workers = weave_choice.workers;
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    budget_state[0] += 1;
    budget_state[1] += (weave_workers == 0);
    budget_state[2] = gridDim.x;
    budget_state[3] = weave_workers;
    budget_state[4] = num_tokens_past_padded / moe_block_size;
    budget_state[5] = parallel;
    budget_state[6] = n_tiles;
    budget_state[7] = k_tiles;
    budget_state[8 + weave_choice.bucket] += 1;
    budget_state[17] = weave_local_routes;
    budget_state[18] = weave_total_routes;
  }
  // Entire CTA exits uniformly before locks or async shared-memory operations.
  if (int(blockIdx.x) >= weave_workers) return;
'''
    anchor = '  int iters = div_ceil(k_tiles * n_tiles * parallel, weave_workers);'
    result[name] = replace_exact(text, anchor, insertion + anchor)

    name = PREFIX + 'moe_wna16_marlin.cuh'
    text = result[name]
    prefix = '    bool use_atomic_add,\n    bool use_fp32_reduce,\n'
    anchor = prefix + '    bool is_zp_float) {'
    if text.count(anchor) != 2:
        raise RuntimeError('native budget host signature count differs')
    text = text.replace(anchor, prefix + '    bool is_zp_float, ' + EXTRA + ') {', 1)
    text = replace_exact(text, anchor, prefix + '''    bool is_zp_float,
    tvm::ffi::TensorView budget_state,
    int64_t budget_target_percent, int64_t budget_ep_size, int64_t budget_mode) {''')
    text = replace_exact(text, 'use_fp32_reduce, max_shared_mem);',
                         'use_fp32_reduce, max_shared_mem, budget_state, budget_target_percent, budget_ep_size, budget_mode);')
    text = replace_exact(text, '      use_fp32_reduce,\n      is_zp_float);', '''      use_fp32_reduce,
      is_zp_float,
      static_cast<int64_t*>(budget_state.data_ptr()),
      static_cast<int>(budget_target_percent), static_cast<int>(budget_ep_size), static_cast<int>(budget_mode));''')
    anchor = '  TensorMatcher({-1, -1}).with_dtype<scalar_t>().with_device(device).verify(a);'
    text = replace_exact(text, anchor, anchor + '''
  TensorMatcher({19}).with_dtype<int64_t>().with_device(device).verify(budget_state);
  RuntimeCheck(budget_state.is_contiguous(), "budget state must be contiguous");
  RuntimeCheck(budget_target_percent >= 1 && budget_target_percent <= 100, "invalid budget target");
  RuntimeCheck(budget_ep_size == 4 && kIsEP, "native budget v1 requires EP4");
  RuntimeCheck(budget_mode == 1 || budget_mode == 2, "invalid budget mode");''')
    result[name] = text

    text = result[OP]
    text = replace_exact(text, '        "moe_wna16_marlin",', '        "weave_budget_marlin",')
    text = replace_exact(text, '        cuda_files=["gemm/marlin_moe/moe_wna16_marlin.cuh"],',
                         '        external_cuda_files=[WEAVE_BUDGET_HEADER],')
    text = replace_exact(text, '    is_zp_float: bool = False,\n', '''    is_zp_float: bool = False,
    *,
    budget_state: torch.Tensor,
    budget_target_percent: int = 75,
    budget_ep_size: int = 4,
    budget_mode: int = 1,
''')
    text = replace_exact(text, '        is_zp_float,\n',
                         '        is_zp_float,\n        budget_state, budget_target_percent, budget_ep_size, budget_mode,\n')
    result[OP] = text
    return result


def build_overlay(package_root, cache_root):
    package_root, cache_root = Path(package_root), Path(cache_root)
    pins = verify_sources(package_root)
    changed = patch_sources({name: (package_root / name).read_text() for name in PATCHED})
    helper = (ROOT / 'csrc/native_budget.h').read_bytes()
    digest = hashlib.sha256(json.dumps(pins, sort_keys=True).encode() + helper)
    for name, text in sorted(changed.items()):
        digest.update(name.encode() + text.encode())
    identity = digest.hexdigest()
    overlay = cache_root / ('marlin-' + identity)
    # Per-rank cache roots avoid a writer racing the other ranks' compilers.
    overlay.mkdir(parents=True, exist_ok=True)
    for name in pins:
        if name.startswith('kernels/jit/csrc/'):
            target = overlay / name
            target.parent.mkdir(parents=True, exist_ok=True)
            raw = changed[name].encode() if name in changed else (package_root / name).read_bytes()
            if not target.exists() or target.read_bytes() != raw:
                target.write_bytes(raw)
    helper_path = overlay / PREFIX / 'native_budget.h'
    if not helper_path.exists() or helper_path.read_bytes() != helper:
        helper_path.write_bytes(helper)
    (overlay / PREFIX / 'budget_op.py').write_text(changed[OP])
    return changed[OP], overlay / PREFIX / 'moe_wna16_marlin.cuh', identity


def load_budget_op(package_root, cache_root):
    source, header, identity = build_overlay(package_root, cache_root)
    module = types.ModuleType('weave_native_budget_marlin_' + identity)
    module.__file__ = str(header.parent / 'budget_op.py')
    module.WEAVE_BUDGET_HEADER = str(header)
    exec(compile(source, module.__file__, 'exec'), module.__dict__)
    op = module.moe_wna16_marlin_gemm
    op.weave_prepare = module._jit_moe_wna16_marlin_module
    op.weave_overlay_sha256 = identity
    return op
