"""Offline real-checkpoint safety probe for the measured native Marlin policy.

Run torchrun --standalone --nproc-per-node=4 tests/gpu/check_native_cost.py with
--model-path /model --output-dir /work/fresh --cache-root /work/jit
--table-dir /work/measured-tables. Full-MoE predictive validation is performed
inside calibration; this independent probe checks actual Marlin kernel safety.
"""
import argparse
from collections import Counter
from datetime import timedelta
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import sys
import threading
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tests.gpu import check_native_budget as base
from src.weave.hopper_evidence import atomic_report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model-path', 'output-dir', 'cache-root', 'table-dir'):
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


def validate_plan(before, after, rows, block_size):
    from src.weave.native_cost import CANDIDATES
    base.require(len(before) == len(after) == 32, 'measured plan must have 32 int64 cells')
    counts = Counter(expert for row in rows for expert in row if expert >= 0)
    grouped = base.group_counts(rows, block_size)
    expected = [grouped['global_blocks'], grouped['local_blocks'], grouped['local_rows'],
                grouped['total_rows'], len(counts), sum(n * n for n in counts.values()),
                max(counts.values(), default=0)]
    base.require(after[2:9] == expected, 'planner feature/group counts differ from actual routing')
    base.require(after[10] - before[10] == 1, 'one operator validation launch must plan once')
    base.require(after[20] == before[20] == 0 and after[23] == 0, 'invalid measured plan observed')
    selected = after[21]
    base.require(selected in range(9) and tuple(after[:2]) == tuple(CANDIDATES[selected]),
                 'planner selected an invalid candidate pair')
    base.require([a - b for a, b in zip(after[11:20], before[11:20])] ==
                 [int(i == selected) for i in range(9)], 'planner candidate histogram differs')
    base.require(after[9] > 0 and after[22] >= 0, 'planner predicted cost or anchor distance is invalid')
    return dict(pair=after[:2], features=[after[4], after[6], after[3], after[7], after[8]],
                selected_index=selected, predicted_moe_ns=after[9], calls=after[10],
                candidate_histogram=after[11:20], invalid_plan_count=after[20],
                nearest_distance_ppm=after[22])


class CostArm(base.Arm):
    def __init__(self, op, stage, batch, layer, align, table):
        super().__init__(op, 1, stage, batch, layer, 19, align)
        self.plan_storage, self.plan = base.guarded(32, torch.int64, self.x.device)
        self.table = torch.tensor(table, dtype=torch.float32, device=self.x.device)
        self.control = torch.tensor([0, 8, 8], dtype=torch.int32, device=self.x.device)

    def body(self):
        from sgl_kernel.scalar_type import scalar_types
        self.output.zero_()
        self.grouping = self.align(self.ids, self.block_size, base.E)
        k, n = (base.H, 2 * base.I) if self.stage == 'w13' else (base.I, base.H)
        result = self.op(self.x, self.output, self.qweight, None, self.scales,
            None, None, None, None, self.workspace, *self.grouping, self.routing,
            moe_block_size=self.block_size, top_k=base.TOPK if self.stage == 'w13' else 1,
            mul_topk_weights=self.stage == 'w2', is_ep=True, b_q_type=scalar_types.float4_e2m1f,
            size_m=self.batch if self.stage == 'w13' else self.rows, size_n=n, size_k=k,
            is_k_full=True, use_atomic_add=False, use_fp32_reduce=True, is_zp_float=False,
            budget_state=self.state, cost_plan=self.plan, cost_table=self.table,
            cost_control=self.control, cost_role=0 if self.stage == 'w13' else 1,
            cost_refresh=self.stage == 'w2')
        base.require(result.data_ptr() == self.output.data_ptr(), 'cost GEMM ignored supplied output')

    def pointers(self):
        return super().pointers() + tuple(t.data_ptr() for t in (self.plan, self.table, self.control))

    def capture(self, owned_graphs):
        super().capture(owned_graphs)
        self.plan.zero_()

    def run(self, rows, graph):
        before = self.plan.cpu().tolist()
        output, state = super().run(rows, graph)
        base.inspect_guards(self.plan_storage, self.plan, zero=False)
        plan = validate_plan(before, self.plan.cpu().tolist(), rows, self.block_size)
        role = 0 if self.stage == 'w13' else 1
        grid = state['native_grid']
        available = state['local_blocks'] * state['n_tiles'] * state['k_tiles']
        expected = min((grid * plan['pair'][role] + 7) // 8, available)
        base.require(state['active_workers'] == expected, 'active workers differ from selected pair/work cap')
        return output, dict(state, plan=plan)


def _select_group(profile, layer):
    matches = [key for key, value in profile['groups'].items() if layer in value.get('layers', [])]
    base.require(len(matches) == 1, 'measured profile must bind selected layer to one structural class')
    return matches[0]


def main(argv=None):
    args = parse_args(argv)
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1')
    rank = int(os.environ.get('LOCAL_RANK', '0'))
    report_path = args.output_dir / f'native-cost-gpu-rank{rank}.json'
    base.require(not report_path.exists(), 'fresh output directory required; preserve prior evidence')
    report = dict(schema_version=1, method='native_marlin_measured_cta_v2', status='running',
        rank=rank, world_size=4, numerical_policy='record_only', raw_content_exported=False,
        formal_performance_claim=False, model_execution_evidence=False, layer=args.layer,
        batches=args.batches, replay_cycles=args.replay_cycles, cases=[], stage='startup',
        scope='real_checkpoint_native_marlin_cost_operator_eager_and_graph',
        cleanup={'status': 'not_started'},
        limitations=['Selected layer native kernel safety only; full-MoE validation is in calibration profile.',
                     'Guard regions and lock release do not replace compute-sanitizer.',
                     'Finite numerical differences are record-only.'])
    lock, stopped = threading.Lock(), threading.Event()

    def save():
        with lock:
            atomic_report(report_path, report)

    def watchdog():
        if not stopped.wait(args.timeout_seconds):
            with lock:
                report.update(status='failed', stop_reason='process_timeout_or_cuda_hang')
                report['cleanup'] = dict(status='incomplete', reason='watchdog_process_exit')
                atomic_report(report_path, report)
            os._exit(124)

    save()
    threading.Thread(target=watchdog, daemon=True).start()
    owned_graphs, retained_arms, initialized, failure = [], [], False, None
    try:
        global torch
        import torch
        import torch.distributed as dist
        base.torch = torch
        from src.weave.checkpoint_identity import checkpoint_stats, sha256_file
        from src.weave.runtime_contract import load_contract, verify_adapter_sources
        from src.weave.weights import load_raw_experts
        from src.weave.native_budget import source_pins
        from src.weave.native_cost import load_profile, native_cost_source_sha256
        from src.weave.native_cost_patch import load_cost_op
        base.require(torch.cuda.is_available(), 'CUDA unavailable: use existing Linux SGLang image on four H100s')
        torch.cuda.set_device(rank)
        dist.init_process_group('gloo', timeout=timedelta(seconds=args.timeout_seconds))
        initialized = True
        base.require(dist.get_world_size() == 4 and dist.get_rank() == rank, 'four single-node torchrun ranks required')
        props = torch.cuda.get_device_properties(rank)
        image_id = os.environ.get('WEAVE_IMAGE_ID', '')
        base.require(re.fullmatch(r'sha256:[0-9a-f]{64}', image_id) is not None, 'resolved WEAVE_IMAGE_ID required')
        base.require(platform.system() == 'Linux' and platform.machine() == 'x86_64', 'Linux x86_64 runtime required')
        distribution = base._sgl_kernel_distribution()  # preserve both installed distribution names
        runtime = dict(python=platform.python_version(), torch=str(torch.__version__), cuda=torch.version.cuda,
            sglang=importlib.metadata.version('sglang'), sgl_kernel=distribution.version,
            device=props.name, capability=[props.major, props.minor],
            multiprocessors=props.multi_processor_count, image_id=image_id)
        runtime['sgl_kernel_binary_sha256'] = {str(p): sha256_file(distribution.locate_file(p))
                                             for p in (distribution.files or ()) if str(p).endswith('.so')}
        base.require(runtime['sgl_kernel_binary_sha256'], 'installed kernel binary inventory unavailable')
        contract = load_contract()
        for observed, expected in (('python', 'python_version'), ('torch', 'torch_version'),
                                   ('cuda', 'torch_cuda'), ('sglang', 'sglang_version')):
            base.require(runtime[observed] == contract[expected], 'runtime contract differs: ' + observed)
        base.require(torch.cuda.device_count() == 4 and runtime['capability'] == [9, 0], 'four visible SM90 GPUs required')
        report['runtime'] = runtime
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision('highest')
        root = base.package_root()
        report['source_identity'] = dict(adapter=verify_adapter_sources(root),
            helpers=verify_adapter_sources(root, base.HELPER_PINS),
            budget_upstream=verify_adapter_sources(root, source_pins()),
            cost_source_sha256=native_cost_source_sha256())
        checkpoint_digest = os.environ.get('WEAVE_CHECKPOINT_CONTENT_SHA256', '')
        base.require(re.fullmatch(r'[0-9a-f]{64}', checkpoint_digest) is not None and
                     checkpoint_digest != '0' * 64, 'verified checkpoint content SHA256 required')
        # This content digest is verified against the read-only mount by the
        # owning controller; additionally hash the actual local config here.
        expected_identity = dict(rank=rank, cost_source_sha256=native_cost_source_sha256(),
            upstream_sha256=report['source_identity']['budget_upstream'], image_id=image_id,
            checkpoint_content_sha256=checkpoint_digest,
            model_config_sha256=sha256_file(args.model_path / 'config.json'),
            sglang_commit=base.UPSTREAM_COMMIT,
            runtime={k: runtime[k] for k in ('python', 'torch', 'cuda', 'sglang', 'sgl_kernel')},
            gpu=dict(name=props.name, capability=[props.major, props.minor],
                     sm_count=props.multi_processor_count, total_memory=props.total_memory))
        profile = load_profile(args.table_dir / f'native-cost-rank{rank}.json', expected_identity)
        group = _select_group(profile, args.layer)
        report.update(profile_sha256=profile['profile_sha256'], profile_group=group)
        from sglang.kernels.ops.moe.moe_wna16_marlin import moe_wna16_marlin_gemm
        from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import moe_align_block_size
        from sglang.srt.layers.quantization.marlin_utils_fp4 import prepare_moe_mxfp4_layer_for_marlin
        patched = load_cost_op(root, args.cache_root / f'rank{rank}')
        patched.weave_prepare(torch.bfloat16, True, False)
        report['source_identity']['overlay_sha256'] = patched.weave_overlay_sha256
        config = json.loads((args.model_path / 'config.json').read_text())
        base.require(tuple(config[k] for k in ('hidden_size', 'moe_intermediate_size', 'n_routed_experts',
                    'num_experts_per_tok', 'num_hidden_layers')) == (4096, 2048, 256, 6, 43), 'original checkpoint geometry required')
        before_stats = checkpoint_stats(args.model_path)
        raw = load_raw_experts(args.model_path, args.layer, rank, 4, device=f'cuda:{rank}')
        report['checkpoint'] = dict(config_sha256=sha256_file(args.model_path / 'config.json'),
            index_sha256=sha256_file(args.model_path / 'model.safetensors.index.json'),
            local_expert_tensor_sha256=[base.hash_tensor(t) for t in raw])
        layer = torch.nn.Module()
        layer.w13_weight = torch.nn.Parameter(torch.cat(raw[:2], dim=1).view(torch.int8), requires_grad=False)
        layer.w2_weight = torch.nn.Parameter(raw[2].view(torch.int8), requires_grad=False)
        layer.w13_weight_scale = torch.nn.Parameter(torch.cat(raw[3:5], dim=1).view(torch.float8_e8m0fnu), requires_grad=False)
        layer.w2_weight_scale = torch.nn.Parameter(raw[5].view(torch.float8_e8m0fnu), requires_grad=False)
        layer.orig_dtype = torch.bfloat16
        prepare_moe_mxfp4_layer_for_marlin(layer)
        base.require(layer.w13_weight_scale.dtype == layer.w2_weight_scale.dtype == torch.float8_e8m0fnu,
                     'native repack did not preserve E8M0 scales')
        torch.cuda.synchronize()
        for stage in ('w13', 'w2'):
            for batch in args.batches:
                arms = {}
                for name in ('native', 'cost'):
                    for execution in ('eager', 'graph'):
                        arm = (base.Arm(moe_wna16_marlin_gemm, 0, stage, batch, layer, 19, moe_align_block_size)
                               if name == 'native' else CostArm(patched, stage, batch, layer, moe_align_block_size,
                                                               profile['tables'][group][str(batch)]['rows']))
                        retained_arms.append(arm)
                        arms[f'{name}_{execution}'] = arm
                        if execution == 'graph':
                            arm.capture(owned_graphs)
                reference = base.reference_slot(arms['native_eager'], raw)
                native_grids = set()
                for sequence, pattern in enumerate(base.replay_schedule(args.replay_cycles)):
                    rows, outputs, states = base.route_rows(pattern, batch), {}, {}
                    for name, arm in arms.items():
                        report['stage'] = f'{stage}_b{batch}_{sequence}_{name}'
                        outputs[name], state = arm.run(rows, name.endswith('_graph'))
                        if state is not None:
                            states[name] = state
                    native_grids.add(states['cost_graph']['native_grid'])
                    numerical = {name + '_vs_native_eager': base.numerical(value, outputs['native_eager'])
                                 for name, value in outputs.items() if name != 'native_eager'}
                    numerical['cost_graph_vs_own_eager'] = base.numerical(outputs['cost_graph'], outputs['cost_eager'])
                    oracle = {} if pattern == 'empty' else {name: {kind: base.numerical(value[:1], target)
                              for kind, target in reference.items()} for name, value in outputs.items()}
                    report['cases'].append(dict(stage=stage, batch=batch, sequence=sequence, pattern=pattern,
                        grouped_work=base.group_counts(rows, arms['native_eager'].block_size), states=states,
                        numerical=numerical, oracle_numerical=oracle, finite=True, guards_intact=True,
                        locks_zero_after_call=True, remote_slots_zero=True, stable_graph_pointers=True,
                        graph_reused=True, oracle_slots=0 if pattern == 'empty' else 1))
                    save()
                base.require(len(native_grids) == 1, 'native physical grid changed within one capture')
        base.require(checkpoint_stats(args.model_path) == before_stats, 'checkpoint changed during validation')
        report.update(case_count=len(report['cases']), expected_case_count=2 * len(args.batches) * len(base.replay_schedule(args.replay_cycles)),
                      actual_graph_replays=len(report['cases']) * 2, stop_reason='complete_case_coverage')
        base.require(report['case_count'] == report['expected_case_count'], 'incomplete operator case coverage')
        report['source_identity']['helpers_after'] = verify_adapter_sources(root, base.HELPER_PINS)
        report['source_identity']['adapter_after'] = verify_adapter_sources(root)
        report['source_identity']['budget_upstream_after'] = verify_adapter_sources(root, source_pins())
        report['checks_complete'] = True
    except Exception as exc:
        failure = type(exc).__name__
        report.update(status='failed', error_type=failure, stop_reason='runtime_or_validity_failure')
        private = args.output_dir / 'private'
        private.mkdir(parents=True, exist_ok=True, mode=0o700)
        path = private / f'native-cost-gpu-rank{rank}.log'
        with path.open('a') as stream:
            traceback.print_exception(type(exc), exc, exc.__traceback__, file=stream)
        path.chmod(0o600)
        save()
    finally:
        try:
            if initialized:
                from src.weave.graph_cleanup import cleanup_graphs
                def publish(state):
                    report['cleanup'] = state
                    save()
                statuses = [None] * 4
                dist.all_gather_object(statuses, dict(rank=rank, checks_complete=report.get('checks_complete', False), error_type=failure))
                report['all_rank_checks'] = statuses
                if not all(item['checks_complete'] for item in statuses):
                    failure = failure or 'PeerValidationFailure'
                    report.update(status='failed', error_type=failure, stop_reason='rank_validation_failure')
                cleanup_graphs(owned_graphs, torch, dist, None, publish)
            else:
                report['cleanup'] = {'status': 'not_initialized'}
        except Exception as exc:
            failure = failure or type(exc).__name__
            report.update(status='failed', error_type=failure, stop_reason='cleanup_failure')
        if not failure and report.get('checks_complete') and report['cleanup']['status'] == 'complete':
            report['status'] = 'complete'
        save()
        stopped.set()
    return int(report['status'] != 'complete')


if __name__ == '__main__':
    raise SystemExit(main())
