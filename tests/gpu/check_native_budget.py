"""Offline, real MXFP4 Marlin budget validation on four independent H100 ranks.

Run with the existing source-pinned SGLang image and a read-only original model:
  torchrun --standalone --nproc-per-node=4 tests/gpu/check_native_budget.py \
    --model-path /model --output-dir /work/fresh --cache-root /work/jit

No downloads, toy kernel, model serving, NCCL or performance measurements are
involved. The probe uses real checkpoint weights, installed native repacking
and grouping, original Marlin, and the actual compiled budget overlay. A whole
process watchdog bounds CUDA hangs; the outer controller must still own and
reap its container. Run under compute-sanitizer separately when available.
"""
import argparse
from collections import Counter
from datetime import timedelta
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import re
import sys
import threading
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.weave.hopper_evidence import atomic_report

H, I, E, TOPK, EP = 4096, 2048, 256, 6, 4
GUARD, GUARD_WORDS = 0x13572468, 64
UPSTREAM_COMMIT = '29481685462732237d80d86076d6563e1f658102'
# These support files are additional to the overlay's own transitive source
# manifest and the repository's existing adapter manifest.
HELPER_PINS = {
    'srt/layers/quantization/marlin_utils_fp4.py':
        'dbef823f827745342ae0ed5a34361da123654ede64eccb0b8d08a5dd5abfa864',
    'srt/layers/quantization/marlin_utils.py':
        '1f0529ca2b24e2af804eaaf51fe72d76484cf5c7ff91a768612c5c9ddfba0bf3',
    'srt/layers/moe/moe_runner/triton_utils/moe_align_block_size.py':
        '652dc416dc461b6f5b04cb9182dfba8fec439a30548a6cc1da8e38037bd2c0aa',
    'kernels/ops/quantization/gptq_marlin_repack.py':
        'c128316664f70a631d2a0a88a6d0ebc1e4ee435114425914bb69bb1c89f30ca7',
    'kernels/jit/csrc/gemm/marlin/gptq_marlin_repack.cuh':
        'a1fd81cbce9dacc6bb9d54129a2d266ed594a243d4fdd8fad7d1eb93cf02ad0b',
    'kernels/ops/moe/__init__.py':
        '76b0e99e33ed3ada54caca4edcbbb095b3c48ddd326a28bc3e64af862f0c0711',
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model-path', 'output-dir', 'cache-root'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--layer', type=int, default=0)
    parser.add_argument('--batches', type=int, nargs='+', default=[7, 32])
    parser.add_argument('--replay-cycles', type=int, default=3)
    parser.add_argument('--timeout-seconds', type=int, default=1800)
    args = parser.parse_args(argv)
    if args.layer not in range(43):
        parser.error('layer must be in 0..42')
    if sorted(set(args.batches)) != args.batches or any(b not in range(1, 33) for b in args.batches):
        parser.error('batches must be unique increasing integers in 1..32')
    if args.replay_cycles < 2 or not 30 <= args.timeout_seconds <= 7200:
        parser.error('at least two replay cycles and timeout in 30..7200 seconds required')
    return args


def route_rows(name, batch):
    """Mapped native EP IDs; remote slots share the -1 grouping bucket.

    Every nonempty pattern includes expert 0 in slot 0 for the independent
    oracle. Per-token local expert IDs are unique. No user routes are used.
    """
    if name not in ('balanced', 'heavy', 'light', 'empty'):
        raise ValueError('unknown route pattern')
    rows = []
    for token in range(batch):
        local = {'balanced': 1 + token % 2, 'heavy': TOPK,
                 'light': int(token == 0), 'empty': 0}[name]
        rows.append([(token * TOPK + slot) % (E // EP) if slot < local else -1
                     for slot in range(TOPK)])
    return rows


def group_counts(rows, block_size):
    counts = Counter(value for row in rows for value in row)
    return dict(global_blocks=sum((n + block_size - 1) // block_size for n in counts.values()),
                local_blocks=sum((n + block_size - 1) // block_size for e, n in counts.items() if e >= 0),
                local_rows=sum(n for e, n in counts.items() if e >= 0),
                total_rows=sum(counts.values()))


def replay_schedule(cycles):
    # Revisit full/split reduction lock layouts without clearing the workspace.
    return ['heavy'] + ['balanced', 'light', 'heavy', 'empty', 'heavy'] * cycles


def state_record(values):
    require(len(values) == 19, 'budget evidence state must have 19 int64 cells')
    return dict(calls=values[0], empty_calls=values[1], native_grid=values[2],
                active_workers=values[3], grouped_blocks_including_remote=values[4],
                local_blocks=values[5], n_tiles=values[6], k_tiles=values[7],
                histogram=values[8:17], local_routes=values[17], total_routes=values[18])


def validate_state(before, after, expected, mode):
    a, b = state_record(after), state_record(before)
    require(a['calls'] - b['calls'] == 1, 'one GEMM must increment device call count once')
    empty = expected['local_rows'] == 0
    require(a['empty_calls'] - b['empty_calls'] == int(empty), 'empty launch counter differs')
    require(a['grouped_blocks_including_remote'] == expected['global_blocks'] and
            a['local_blocks'] == expected['local_blocks'] and
            a['local_routes'] == expected['local_rows'] and
            a['total_routes'] == expected['total_rows'], 'device route/group counts differ')
    grid, active = a['native_grid'], a['active_workers']
    require(grid > 0 and a['n_tiles'] > 0 and a['k_tiles'] > 0, 'invalid native geometry')
    require((active == 0) if empty else (0 < active <= grid), 'invalid active worker count')
    if mode == 2 and not empty:
        require(active == grid, 'full virtual-grid control must use the native grid')
    if mode == 1 and not empty:
        require(active <= expected['local_blocks'] * a['n_tiles'] * a['k_tiles'],
                'adaptive workers exceed available tile iterations')
    bucket = (active * 8 + grid - 1) // grid if active else 0
    delta = [x - y for x, y in zip(a['histogram'], b['histogram'])]
    require(delta == [int(i == bucket) for i in range(9)], 'device bucket histogram differs')
    return a


def numerical(actual, reference):
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(reference).all())
    require(finite, 'nonfinite output or reference')
    delta = actual.double() - reference.double()
    l2 = float((delta.norm() / reference.double().norm().clamp_min(1e-12)).item())
    maximum = float(delta.abs().max().item()) if delta.numel() else 0.0
    require(math.isfinite(l2) and math.isfinite(maximum), 'nonfinite numerical metric')
    # All finite values are records. There is deliberately no accuracy gate.
    return dict(finite=True, relative_l2=l2, max_abs=maximum)


def guarded(size, dtype, device):
    storage = torch.full((size + 2 * GUARD_WORDS,), GUARD, dtype=dtype, device=device)
    middle = storage[GUARD_WORDS:-GUARD_WORDS]
    middle.zero_()
    return storage, middle


def inspect_guards(storage, middle, *, zero):
    require(bool((storage[:GUARD_WORDS] == GUARD).all()) and
            bool((storage[-GUARD_WORDS:] == GUARD).all()), 'workspace/state red-zone corruption')
    if zero:
        require(int(middle.count_nonzero().item()) == 0, 'Marlin reduction locks were not released')


def inspect_grouping(grouping, rows, block_size):
    sorted_ids, expert_ids, padded = grouping
    count = int(padded.item())
    expected = group_counts(rows, block_size)
    require(count == expected['global_blocks'] * block_size, 'native padded grouping size differs')
    require(0 <= count <= sorted_ids.numel(), 'native grouping exceeds allocated capacity')
    ids = sorted_ids[:count].cpu().tolist()
    experts = expert_ids[:count // block_size].cpu().tolist()
    flat = [expert for row in rows for expert in row]
    seen = []
    for offset, slot in enumerate(ids):
        require(slot >= 0, 'negative sorted slot index')
        if slot < len(flat):
            require(flat[slot] == experts[offset // block_size], 'native grouping expert/slot mismatch')
            seen.append(slot)
    require(sorted(seen) == list(range(len(flat))), 'native grouping lost or duplicated route slots')
    require(sum(e >= 0 for e in experts) == expected['local_blocks'], 'native local block count differs')
    return expected


class Arm:
    """One isolated lock workspace per captured arm, reused across replays."""
    def __init__(self, op, mode, stage, batch, layer, state_size, align):
        self.op, self.mode, self.stage, self.batch, self.align = op, mode, stage, batch, align
        device = layer.w13_weight.device
        self.rows = batch * TOPK
        self.block_size = next(size for size in (8, 16, 32, 48, 64)
                               if self.rows / (E // EP) / size < .9)
        self.ids = torch.tensor(route_rows('heavy', batch), dtype=torch.int32, device=device)
        self.routing = torch.tensor([.1, .15, .2, .1, .25, .2], device=device).repeat(batch, 1)
        generator = torch.Generator(device=device).manual_seed(1739 + batch)
        input_rows, k, n = (batch, H, 2 * I) if stage == 'w13' else (self.rows, I, H)
        self.x = (torch.randn((input_rows, k), generator=generator, device=device) * .125).bfloat16()
        self.output = torch.empty((self.rows, n), dtype=torch.bfloat16, device=device)
        self.qweight = layer.w13_weight if stage == 'w13' else layer.w2_weight
        self.scales = layer.w13_weight_scale if stage == 'w13' else layer.w2_weight_scale
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        self.lock_storage, self.workspace = guarded(sms * 4, torch.int32, device)
        self.state_storage, self.state = guarded(state_size, torch.int64, device)
        self.grouping = None
        self.graph = None
        self.captured_pointers = None

    def body(self):
        from sgl_kernel.scalar_type import scalar_types
        self.output.zero_()  # Native EP requires the skipped remote slots to start at zero.
        self.grouping = self.align(self.ids, self.block_size, E)
        sorted_ids, expert_ids, padded = self.grouping
        k, n = (H, 2 * I) if self.stage == 'w13' else (I, H)
        keywords = dict(moe_block_size=self.block_size,
                        top_k=TOPK if self.stage == 'w13' else 1,
                        mul_topk_weights=self.stage == 'w2', is_ep=True,
                        b_q_type=scalar_types.float4_e2m1f,
                        size_m=self.batch if self.stage == 'w13' else self.rows,
                        size_n=n, size_k=k, is_k_full=True, use_atomic_add=False,
                        use_fp32_reduce=True, is_zp_float=False)
        if self.mode:
            keywords.update(budget_state=self.state, budget_target_percent=75,
                            budget_ep_size=EP, budget_mode=self.mode)
        result = self.op(self.x, self.output, self.qweight, None, self.scales,
                         None, None, None, None, self.workspace, sorted_ids, expert_ids,
                         padded, self.routing, **keywords)
        require(result.data_ptr() == self.output.data_ptr(), 'GEMM failed to use supplied output')

    def pointers(self):
        tensors = [self.x, self.ids, self.routing, self.output, self.qweight,
                   self.scales, self.workspace, self.state] + list(self.grouping)
        return tuple(tensor.data_ptr() for tensor in tensors)

    def capture(self, owned_graphs):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self.body()
        stream.synchronize()
        inspect_guards(self.lock_storage, self.workspace, zero=True)
        self.graph = torch.cuda.CUDAGraph()
        owned_graphs.append(self.graph)
        with torch.cuda.graph(self.graph, stream=stream):
            self.body()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        self.state.zero_()
        self.captured_pointers = self.pointers()

    def run(self, rows, graph):
        self.ids.copy_(torch.tensor(rows, device=self.ids.device, dtype=torch.int32))
        # Leave locks untouched between calls: clearing them would mask bugs.
        inspect_guards(self.lock_storage, self.workspace, zero=True)
        before = self.state.cpu().tolist() if self.mode else None
        self.output.fill_(float('nan'))
        if graph:
            self.graph.replay()
        else:
            self.body()
        torch.cuda.synchronize()
        require(bool(torch.isfinite(self.output).all()), 'nonfinite Marlin output')
        expected = inspect_grouping(self.grouping, rows, self.block_size)
        inspect_guards(self.lock_storage, self.workspace, zero=True)
        inspect_guards(self.state_storage, self.state, zero=False)
        remote = self.ids.flatten() == -1
        require(int(self.output[remote].count_nonzero().item()) == 0, 'remote EP output slots are nonzero')
        if graph:
            require(self.pointers() == self.captured_pointers, 'captured tensor pointers changed')
        evidence = validate_state(before, self.state.cpu().tolist(), expected, self.mode) if self.mode else None
        return self.output.clone(), evidence


def reference_slot(arm, raw):
    """Three independent numeric records for expert 0/slot 0, outside capture."""
    from src.weave.hopper_reference import _mxfp4_values, _partial_gemm
    indices = (0, 1) if arm.stage == 'w13' else (2,)
    fp32, bf16, fp8 = [], [], []
    x = arm.x[:1]
    for index in indices:
        packed, scales = raw[index][0], raw[index + 3][0]
        values, factors = _mxfp4_values(packed, scales, torch)
        weight = (values.reshape(values.shape[0], -1, 32) * factors.unsqueeze(-1)).flatten(1)
        fp32.append((x.float() @ weight.T).bfloat16())
        bf16.append(x @ weight.bfloat16().T)
        fp8.append(_partial_gemm(x, packed, scales, torch))
    result = dict(bf16_oracle=torch.cat(fp32, dim=1), bf16_reference=torch.cat(bf16, dim=1),
                  fp8_oracle=torch.cat(fp8, dim=1))
    if arm.stage == 'w2':
        result = {name: (tensor.float() * arm.routing[0, 0]).bfloat16()
                  for name, tensor in result.items()}
    return result


def hash_tensor(tensor):
    digest = hashlib.sha256()
    for expert in tensor:
        digest.update(expert.contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def package_root():
    spec = importlib.util.find_spec('sglang')
    require(spec is not None and spec.submodule_search_locations, 'installed offline SGLang is required')
    return Path(next(iter(spec.submodule_search_locations)))


def main(argv=None):
    args = parse_args(argv)
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1')
    rank = int(os.environ.get('LOCAL_RANK', '0'))
    report_path = args.output_dir / f'native-budget-gpu-rank{rank}.json'
    require(not report_path.exists(), 'fresh output directory required; preserve prior evidence')
    report = dict(schema_version=1, status='running', rank=rank, world_size=EP,
                  numerical_policy='record_only', raw_content_exported=False,
                  formal_performance_claim=False, model_execution_evidence=False,
                  scope='real_checkpoint_native_marlin_operator_eager_and_graph',
                  baseline='unmodified_installed_native_marlin',
                  full_mode='patched_full_virtual_grid_control_not_native_baseline',
                  layer=args.layer, batches=args.batches, replay_cycles=args.replay_cycles,
                  cases=[], stage='startup', cleanup={'status': 'not_started'},
                  limitations=['One selected layer; not all 43 model layers or serving.',
                               'Independent rank-local GEMMs; no shared-expert/collective overlap.',
                               'Guard regions and lock release do not replace compute-sanitizer.',
                               'Oracle records sample expert 0/slot 0; native comparisons cover all outputs.',
                               'FP8 oracle is a precision comparator, not native MXFP4/BF16 semantics.'])
    save_lock, stopped = threading.Lock(), threading.Event()

    def save():
        with save_lock:
            atomic_report(report_path, report)

    def watchdog():
        if not stopped.wait(args.timeout_seconds):
            with save_lock:
                report.update(status='failed', stop_reason='process_timeout_or_cuda_hang')
                report['cleanup'] = {'status': 'incomplete', 'reason': 'watchdog_process_exit'}
                atomic_report(report_path, report)
            os._exit(124)

    save()
    threading.Thread(target=watchdog, daemon=True).start()
    owned_graphs, retained_arms, initialized, failure = [], [], False, None
    try:
        global torch
        import torch
        import torch.distributed as dist
        from src.weave.checkpoint_identity import checkpoint_stats, sha256_file
        from src.weave.runtime_contract import load_contract, verify_adapter_sources
        from src.weave.weights import load_raw_experts
        from src.weave.native_budget import STATE_SIZE, native_budget_source_sha256, source_pins
        from src.weave.native_budget_patch import load_budget_op

        require(torch.cuda.is_available(), 'CUDA unavailable: use the existing Linux SGLang image on four H100s')
        torch.cuda.set_device(rank)
        dist.init_process_group('gloo', timeout=timedelta(seconds=args.timeout_seconds))
        initialized = True
        require(dist.get_world_size() == EP and dist.get_rank() == rank, 'four single-node torchrun ranks required')
        contract = load_contract()
        props = torch.cuda.get_device_properties(rank)
        image_id = os.environ.get('WEAVE_IMAGE_ID', '')
        require(re.fullmatch(r'sha256:[0-9a-f]{64}', image_id) is not None, 'resolved existing WEAVE_IMAGE_ID required')
        require(platform.system() == 'Linux' and platform.machine() == 'x86_64', 'Linux x86_64 runtime required')
        runtime = dict(python=platform.python_version(), torch=str(torch.__version__),
                       cuda=torch.version.cuda, sglang=importlib.metadata.version('sglang'),
                       sgl_kernel=importlib.metadata.version('sgl-kernel'),
                       device=props.name, capability=[props.major, props.minor],
                       multiprocessors=props.multi_processor_count, image_id=image_id)
        # Native grouping is supplied by this image's installed sgl-kernel AOT
        # wheel. Record its binary inventory, without installing or fetching it.
        distribution = importlib.metadata.distribution('sgl-kernel')
        runtime['sgl_kernel_binary_sha256'] = {
            str(path): sha256_file(distribution.locate_file(path))
            for path in (distribution.files or ()) if str(path).endswith('.so')}
        require(runtime['sgl_kernel_binary_sha256'], 'installed sgl-kernel binary inventory unavailable')
        for observed, expected in (('python', 'python_version'), ('torch', 'torch_version'),
                                   ('cuda', 'torch_cuda'), ('sglang', 'sglang_version')):
            require(runtime[observed] == contract[expected], 'installed runtime differs from configs/runtime.json: ' + observed)
        require(torch.cuda.device_count() == EP and runtime['capability'] == [9, 0], 'four visible SM90 GPUs required')
        require(STATE_SIZE == 19, 'unexpected budget state schema')
        report['runtime'] = runtime
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision('highest')
        root = package_root()
        report['stage'] = 'source_verification'
        report['source_identity'] = dict(upstream_commit=UPSTREAM_COMMIT,
                                        adapter=verify_adapter_sources(root),
                                        helpers=verify_adapter_sources(root, HELPER_PINS),
                                        budget_upstream=verify_adapter_sources(root, source_pins()),
                                        budget_source_sha256=native_budget_source_sha256(),
                                        local={name: sha256_file(ROOT / name) for name in (
                                            'tests/gpu/check_native_budget.py', 'src/weave/weights.py',
                                            'src/weave/checkpoint_identity.py', 'src/weave/hopper_reference.py',
                                            'src/weave/native_budget.py', 'src/weave/native_budget_patch.py')})
        from sglang.kernels.ops.moe.moe_wna16_marlin import moe_wna16_marlin_gemm
        from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import moe_align_block_size
        from sglang.srt.layers.quantization.marlin_utils_fp4 import prepare_moe_mxfp4_layer_for_marlin
        patched = load_budget_op(root, args.cache_root / f'rank{rank}')
        # Bind the exact generated overlay returned by the real loader.
        report['source_identity']['overlay_sha256'] = patched.weave_overlay_sha256
        report['stage'] = 'checkpoint_read_and_repack'
        save()
        config = json.loads((args.model_path / 'config.json').read_text())
        require(tuple(config[key] for key in ('hidden_size', 'moe_intermediate_size', 'n_routed_experts',
                                             'num_experts_per_tok', 'num_hidden_layers')) == (H, I, E, TOPK, 43),
                'original H4096/I2048/E256/topk6/43-layer checkpoint required')
        before_stats = checkpoint_stats(args.model_path)
        raw = load_raw_experts(args.model_path, args.layer, rank, EP, device=f'cuda:{rank}')
        report['checkpoint'] = dict(config_sha256=sha256_file(args.model_path / 'config.json'),
                                   index_sha256=sha256_file(args.model_path / 'model.safetensors.index.json'),
                                   local_expert_tensor_sha256=[hash_tensor(t) for t in raw],
                                   identity_scope='loaded original local tensors in selected layer',
                                   experts_per_rank=E // EP)
        layer = torch.nn.Module()
        layer.w13_weight = torch.nn.Parameter(torch.cat(raw[:2], dim=1).view(torch.int8), requires_grad=False)
        layer.w2_weight = torch.nn.Parameter(raw[2].view(torch.int8), requires_grad=False)
        layer.w13_weight_scale = torch.nn.Parameter(torch.cat(raw[3:5], dim=1).view(torch.float8_e8m0fnu), requires_grad=False)
        layer.w2_weight_scale = torch.nn.Parameter(raw[5].view(torch.float8_e8m0fnu), requires_grad=False)
        layer.orig_dtype = torch.bfloat16
        prepare_moe_mxfp4_layer_for_marlin(layer)
        require(layer.w13_weight_scale.dtype == layer.w2_weight_scale.dtype == torch.float8_e8m0fnu,
                'native repack did not preserve E8M0 scale format')
        torch.cuda.synchronize()
        report['stage'] = 'compiled_operator_cases'
        save()
        for stage in ('w13', 'w2'):
            for batch in args.batches:
                arms = {}
                for mode, name in ((0, 'native'), (1, 'adaptive'), (2, 'full')):
                    for execution in ('eager', 'graph'):
                        arm = Arm(moe_wna16_marlin_gemm if mode == 0 else patched, mode, stage,
                                  batch, layer, STATE_SIZE, moe_align_block_size)
                        # Retain even a partially captured arm until teardown.
                        retained_arms.append(arm)
                        arms[f'{name}_{execution}'] = arm
                        if execution == 'graph':
                            arm.capture(owned_graphs)
                reference = reference_slot(arms['native_eager'], raw)
                positive_budgets, native_grids = set(), set()
                for sequence, pattern in enumerate(replay_schedule(args.replay_cycles)):
                    rows = route_rows(pattern, batch)
                    outputs, states = {}, {}
                    for name, arm in arms.items():
                        report['stage'] = f'{stage}_b{batch}_{sequence}_{name}'
                        save()
                        outputs[name], state = arm.run(rows, name.endswith('_graph'))
                        if state is not None:
                            states[name] = state
                    numerical_records = {name + '_vs_native_eager': numerical(value, outputs['native_eager'])
                                         for name, value in outputs.items() if name != 'native_eager'}
                    for name in ('adaptive', 'full'):
                        numerical_records[name + '_graph_vs_own_eager'] = numerical(outputs[name + '_graph'], outputs[name + '_eager'])
                    oracle_records = {} if pattern == 'empty' else {
                        name: {oracle: numerical(value[:1], target) for oracle, target in reference.items()}
                        for name, value in outputs.items()}
                    adaptive = states['adaptive_graph']
                    require(len({state['native_grid'] for state in states.values()}) == 1,
                            'native geometry differs between adaptive and full controls')
                    native_grids.add(adaptive['native_grid'])
                    if adaptive['active_workers']:
                        positive_budgets.add(adaptive['active_workers'])
                    record = dict(stage=stage, batch=batch, sequence=sequence, pattern=pattern,
                                  grouped_work=group_counts(rows, arms['native_eager'].block_size),
                                  states=states, numerical=numerical_records, oracle_numerical=oracle_records,
                                  oracle_slots=0 if pattern == 'empty' else 1,
                                  finite=True, guards_intact=True, locks_zero_after_call=True,
                                  remote_slots_zero=True, stable_graph_pointers=True,
                                  graph_reused=True, graph_count=len(owned_graphs))
                    report['cases'].append(record)
                    save()
                require(len(native_grids) == 1, 'physical grid changed within one capture')
                require(len(positive_budgets) >= 2, 'changing routes did not change positive adaptive graph budgets')
        require(checkpoint_stats(args.model_path) == before_stats, 'checkpoint changed during validation')
        report.update(case_count=len(report['cases']), stop_reason='complete_case_coverage',
                      expected_case_count=2 * len(args.batches) * len(replay_schedule(args.replay_cycles)),
                      actual_graph_replays=len(report['cases']) * 3)
        require(report['case_count'] == report['expected_case_count'], 'incomplete operator case coverage')
        report['source_identity']['helpers_after'] = verify_adapter_sources(root, HELPER_PINS)
        report['source_identity']['adapter_after'] = verify_adapter_sources(root)
        report['source_identity']['budget_upstream_after'] = verify_adapter_sources(root, source_pins())
        report['checks_complete'] = True
    except Exception as exc:
        failure = type(exc).__name__
        report.update(status='failed', error_type=failure, stop_reason='runtime_or_validity_failure')
        private = args.output_dir / 'private'
        private.mkdir(parents=True, exist_ok=True, mode=0o700)
        log = private / f'native-budget-gpu-rank{rank}.log'
        with log.open('a') as stream:
            traceback.print_exception(type(exc), exc, exc.__traceback__, file=stream)
        log.chmod(0o600)
        save()
    finally:
        try:
            if initialized:
                from src.weave.graph_cleanup import cleanup_graphs
                def publish_cleanup(state):
                    report['cleanup'] = state
                    save()
                statuses = [None] * EP
                dist.all_gather_object(statuses, dict(rank=rank, checks_complete=report.get('checks_complete', False),
                                                     error_type=failure))
                report['all_rank_checks'] = statuses
                if not all(item['checks_complete'] for item in statuses):
                    failure = failure or 'PeerValidationFailure'
                    report.update(status='failed', error_type=failure, stop_reason='rank_validation_failure')
                cleanup_graphs(owned_graphs, torch, dist, None, publish_cleanup)
            else:
                report['cleanup'] = {'status': 'not_initialized'}
        except Exception as exc:
            failure = failure or type(exc).__name__
            report.update(status='failed', error_type=failure, stop_reason='rank_failure_or_cleanup_failure')
        if not failure and report.get('checks_complete') and report['cleanup']['status'] == 'complete':
            report['status'] = 'complete'
        save()
        stopped.set()
    return 1 if report['status'] != 'complete' else 0


if __name__ == '__main__':
    raise SystemExit(main())
