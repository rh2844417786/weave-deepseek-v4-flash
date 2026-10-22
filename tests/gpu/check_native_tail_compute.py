"""S1: real checkpoint MXFP4 static full-K down versus installed native down.

Offline existing SGLang image only. Run rank-locally or with torchrun (4 ranks).
No collective is used here; this isolates extraction/cancelled split-K costs.
Raw weights, activations, routing, bitmaps and error logs remain private.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import sys
import threading
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.weave.hopper_evidence import atomic_report
from src.weave.native_tail_patch import METHOD, audit_task_coverage, load_down_op, verify_sources
from tests.gpu import check_native_budget as support

H, I, E, TOPK, EP = support.H, support.I, support.E, support.TOPK, support.EP
require = support.require


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model-path', 'output-dir', 'cache-root'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--layer', type=int, default=0)
    parser.add_argument('--batches', type=int, nargs='+', default=[32, 65, 128])
    parser.add_argument('--m-blocks', type=int, nargs='+', default=[8, 16, 32, 48, 64])
    parser.add_argument('--replay-cycles', type=int, default=3)
    parser.add_argument('--timing-repeats', type=int, default=20)
    parser.add_argument('--timeout-seconds', type=int, default=3600)
    parser.add_argument('--sanitizer', choices=['none', 'memcheck', 'racecheck', 'synccheck'], default='none')
    args = parser.parse_args(argv)
    if args.layer not in range(43) or any(b < 1 for b in args.batches):
        parser.error('layer must be 0..42 and batches positive')
    if args.m_blocks != [8, 16, 32, 48, 64]:
        parser.error('S1 acceptance requires all native M blocks: 8 16 32 48 64')
    if not any(b > 32 for b in args.batches):
        parser.error('S1 acceptance requires an independent batch larger than 32')
    if args.replay_cycles < 2 or args.timing_repeats < 2 or args.timeout_seconds < 30:
        parser.error('require >=2 replays/timing repeats and >=30 second watchdog')
    return args


def routes(pattern, batch):
    if pattern == 'concentrated':
        return [list(range(TOPK)) for _ in range(batch)]
    return support.route_rows(pattern, batch)


def bf16_guarded(rows, cols, device):
    guard = 128
    storage = torch.full((rows * cols + 2 * guard,), 43.0, dtype=torch.bfloat16, device=device)
    return storage, storage[guard:-guard].view(rows, cols)


def guards(storage):
    require(bool((storage[:128] == 43).all()) and bool((storage[-128:] == 43).all()),
            'down output red-zone corruption')


def event_ms(operation, repeats):
    for _ in range(3):
        operation()
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(repeats):
        operation()
    end.record()
    end.synchronize()
    return begin.elapsed_time(end) / repeats


class DownCase:
    def __init__(self, layer, batch, block, native, extracted, align, clamp_limit):
        self.batch, self.block, self.native, self.extracted = batch, block, native, extracted
        self.align, self.layer, self.clamp_limit = align, layer, clamp_limit
        self.device = layer.w2_weight.device
        rows = batch * TOPK
        self.ids = torch.tensor(routes('concentrated', batch), dtype=torch.int32, device=self.device)
        self.routing = torch.tensor([.1, .15, .2, .1, .25, .2], device=self.device).repeat(batch, 1)
        self.grouping = align(self.ids, block, E)
        self.capacity = min(self.grouping[1].numel(), self.grouping[0].numel() // block)
        self.geometry = extracted.prepare(rows, H, I, block, self.device.index)
        self.tiles = H // (16 * self.geometry['n_blocks'])
        generator = torch.Generator(device=self.device).manual_seed(2921 + batch)
        self.hidden = (torch.randn((batch, H), generator=generator, device=self.device) * .125).bfloat16()
        self.up = torch.empty((rows, 2 * I), dtype=torch.bfloat16, device=self.device)
        self.x = torch.empty((rows, I), dtype=torch.bfloat16, device=self.device)
        self.native_storage, self.native_out = bf16_guarded(rows, H, self.device)
        self.tail_storage, self.tail_out = bf16_guarded(rows, H, self.device)
        sms = torch.cuda.get_device_properties(self.device).multi_processor_count
        self.lock_storage, self.locks = support.guarded(sms * 4, torch.int32, self.device)
        self.visit_storage, self.visits = support.guarded(self.capacity * self.tiles, torch.int64, self.device)
        self.error_storage, self.error = support.guarded(1, torch.int32, self.device)
        self.no_visits = torch.empty(0, dtype=torch.int64, device=self.device)
        self.graphs = {}

    def native_gemm(self, stage, output):
        from sgl_kernel.scalar_type import scalar_types
        front = stage == 'w13'
        self.native(self.hidden if front else self.x, output,
            self.layer.w13_weight if front else self.layer.w2_weight, None,
            self.layer.w13_weight_scale if front else self.layer.w2_weight_scale,
            None, None, None, None, self.locks, *self.grouping, self.routing,
            self.block, TOPK if front else 1, not front, True, scalar_types.float4_e2m1f,
            self.batch if front else self.batch * TOPK, 2 * I if front else H, H if front else I,
            is_k_full=True, use_atomic_add=False, use_fp32_reduce=True, is_zp_float=False)

    def set_pattern(self, pattern):
        rows = routes(pattern, self.batch)
        self.ids.copy_(torch.tensor(rows, device=self.device, dtype=torch.int32))
        current = self.align(self.ids, self.block, E)
        for target, source in zip(self.grouping, current):
            target.copy_(source)
        # Inputs are produced by the pinned native gate/up and activation once.
        self.up.zero_()
        self.native_gemm('w13', self.up)
        if self.clamp_limit is None:
            from sglang.kernels.ops.activation.activation import silu_and_mul
            silu_and_mul(self.up, self.x)
        else:
            from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import swiglu_limit_func
            swiglu_limit_func(self.x, self.up, self.clamp_limit)
        torch.cuda.synchronize()
        support.inspect_grouping(self.grouping, rows, self.block)
        sorted_ids, experts, padded = self.grouping
        return audit_task_coverage(sorted_ids.cpu().tolist(), experts.cpu().tolist(), int(padded.item()),
                                   self.batch * TOPK, H, self.block, 16 * self.geometry['n_blocks'],
                                   num_experts=E // EP)

    def tail(self, diagnostic, reuse=False):
        self.extracted(self.x, self.layer.w2_weight, self.layer.w2_weight_scale,
                       *self.grouping, self.routing, self.tail_out, self.block,
                       visits=self.visits if diagnostic else self.no_visits,
                       error=self.error, geometry=self.geometry, grid_workers=1 if reuse else 0)

    def body(self, name):
        if name == 'native':
            self.native_out.zero_()
            self.native_gemm('w2', self.native_out)
        else:
            self.tail_out.zero_()
            self.error.zero_()
            if name in ('tail', 'tail_reuse'):
                self.visits.zero_()
            self.tail(name in ('tail','tail_reuse'), name == 'tail_reuse')

    def capture(self, owned):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for name in ('native', 'tail', 'tail_timing'):
                self.body(name)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        for name in ('native', 'tail', 'tail_timing'):
            graph = torch.cuda.CUDAGraph()
            owned.append(graph)
            with torch.cuda.graph(graph, stream=stream):
                self.body(name)
            self.graphs[name] = graph
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        self.pointers = self.addresses()

    def addresses(self):
        tensors = (*self.grouping, self.routing, self.x, self.native_out, self.tail_out,
                   self.visits, self.error, self.locks, self.layer.w2_weight, self.layer.w2_weight_scale)
        return tuple(t.data_ptr() for t in tensors)

    def validate(self, coverage):
        require(int(self.error.item()) == 0, 'device routing/layout error')
        values = self.visits.cpu().tolist()
        expected = set(coverage['expected_task_ids'])
        require(values == [int(i in expected) for i in range(len(values))],
                'full task bitmap differs: missing or repeated down task')
        for storage, middle in ((self.visit_storage,self.visits),(self.error_storage,self.error)):
            support.inspect_guards(storage,middle,zero=False)
        support.inspect_guards(self.lock_storage,self.locks,zero=True)
        guards(self.native_storage); guards(self.tail_storage)
        require(self.addresses() == self.pointers, 'captured down pointer changed')
        for output in (self.native_out,self.tail_out):
            require(bool(torch.isfinite(output).all()), 'nonfinite down output')
            require(int(output[self.ids.flatten() == -1].count_nonzero().item()) == 0,
                    'remote expert output must remain zero')
        return dict(exact_task_bitmap_verified=True, local_tasks=len(expected),
                    valid_pairs=coverage['valid_pairs'], covered_elements=coverage['covered_elements'],
                    task_bitmap_sha256=hashlib.sha256(bytes(int(v) for v in values)).hexdigest(),
                    guards_intact=True, native_locks_zero=True, stable_graph_pointers=True)


def main(argv=None):
    args = parse_args(argv)
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1')
    rank = int(os.environ.get('LOCAL_RANK','0'))
    report_path = args.output_dir / f'native-tail-compute-rank{rank}.json'
    require(not report_path.exists(), 'fresh output directory required; preserve historical reports')
    report = dict(schema_version=1, method=METHOD, stage='S1', rank=rank, status='running',
                  numerical_policy='record_only', source_implemented=True, gpu_mechanisms_verified=False,
                  full_model_graph_execution_verified=False, throughput_result_observed=False,
                  scope='real_checkpoint_native_front_static_full_K_down', cases=[],
                  sanitizer=dict(requested=args.sanitizer, status='external_result_required' if args.sanitizer != 'none'
                                 else 'not_run'), cleanup=dict(status='not_started'),
                  raw_content_exported=False, limitations=[
                      'Static mapping only; no queue, peer communication, stealing or full-model service.',
                      'Output canaries and task bitmaps do not replace external compute-sanitizer.',
                      'Operator timings include output clearing; extracted timings retain device validity checks.',
                      'FP8 oracle is a precision comparator, not the native MXFP4/BF16 method.'])
    lock, stopped = threading.Lock(), threading.Event()
    def save():
        with lock: atomic_report(report_path,report)
    def watchdog():
        if not stopped.wait(args.timeout_seconds):
            with lock:
                report.update(status='failed',stop_reason='process_timeout_or_cuda_hang')
                report['cleanup'] = dict(status='incomplete',reason='watchdog_exit')
                atomic_report(report_path,report)
            os._exit(124)
    save(); threading.Thread(target=watchdog,daemon=True).start()
    owned, retained = [], []
    failure, cuda_started = None, False
    try:
        global torch
        import torch
        support.torch = torch
        from src.weave.checkpoint_identity import checkpoint_stats, sha256_file
        from src.weave.weights import load_raw_experts
        from src.weave.runtime_contract import load_contract
        require(torch.cuda.is_available(),'CUDA unavailable; run offline Linux SGLang image')
        torch.cuda.set_device(rank); cuda_started = True
        props = torch.cuda.get_device_properties(rank)
        contract = load_contract()
        image_id = os.environ.get('WEAVE_IMAGE_ID','')
        require(re.fullmatch(r'sha256:[0-9a-f]{64}',image_id) is not None,'resolved existing WEAVE_IMAGE_ID required')
        require(platform.system() == 'Linux' and platform.machine() == 'x86_64','Linux x86_64 required')
        runtime=dict(python=platform.python_version(),torch=str(torch.__version__),cuda=torch.version.cuda,
                     sglang=importlib.metadata.version('sglang'),sgl_kernel=support._sgl_kernel_distribution().version,
                     image_id=image_id,device=props.name,capability=[props.major,props.minor])
        for key,expected in (('python','python_version'),('torch','torch_version'),('cuda','torch_cuda'),('sglang','sglang_version')):
            require(runtime[key] == contract[expected],'runtime contract mismatch: '+key)
        require(runtime['capability'] == [9,0],'S1 acceptance target is H100 SM90')
        distribution=support._sgl_kernel_distribution()
        runtime['sgl_kernel_binary_sha256']={str(p):sha256_file(distribution.locate_file(p))
            for p in (distribution.files or ()) if str(p).endswith('.so')}
        require(runtime['sgl_kernel_binary_sha256'],'installed kernel binary inventory unavailable')
        report['runtime']=runtime
        package = support.package_root()
        report['upstream_sha256']=verify_sources(package)
        report['local_sha256']={name:sha256_file(ROOT/name) for name in (
            'src/weave/native_tail_patch.py','experimental/native_tail/marlin_down_tile.cuh',
            'experimental/native_tail/down_static_binding.cuh','tests/gpu/check_native_tail_compute.py',
            'tests/gpu/check_native_budget.py','src/weave/weights.py','src/weave/hopper_reference.py')}
        from sglang.kernels.ops.moe.moe_wna16_marlin import moe_wna16_marlin_gemm
        from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import moe_align_block_size
        from sglang.srt.layers.quantization.marlin_utils_fp4 import prepare_moe_mxfp4_layer_for_marlin
        extracted=load_down_op(package,args.cache_root/f'rank{rank}')
        report['overlay_sha256']=extracted.weave_overlay_sha256
        config=json.loads((args.model_path/'config.json').read_text())
        require(tuple(config[key] for key in ('hidden_size','moe_intermediate_size','n_routed_experts',
            'num_experts_per_tok','num_hidden_layers')) == (H,I,E,TOPK,43),'original checkpoint dimensions required')
        before=checkpoint_stats(args.model_path)
        raw=load_raw_experts(args.model_path,args.layer,rank,EP,device=f'cuda:{rank}')
        report['checkpoint']=dict(layer=args.layer,local_tensor_sha256=[support.hash_tensor(t) for t in raw],
            config_sha256=sha256_file(args.model_path/'config.json'),
            index_sha256=sha256_file(args.model_path/'model.safetensors.index.json'))
        layer=torch.nn.Module()
        layer.w13_weight=torch.nn.Parameter(torch.cat(raw[:2],dim=1).view(torch.int8),requires_grad=False)
        layer.w2_weight=torch.nn.Parameter(raw[2].view(torch.int8),requires_grad=False)
        layer.w13_weight_scale=torch.nn.Parameter(torch.cat(raw[3:5],dim=1).view(torch.float8_e8m0fnu),requires_grad=False)
        layer.w2_weight_scale=torch.nn.Parameter(raw[5].view(torch.float8_e8m0fnu),requires_grad=False)
        layer.orig_dtype=torch.bfloat16
        prepare_moe_mxfp4_layer_for_marlin(layer)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.set_float32_matmul_precision('highest')
        report['native_repacked_w2_sha256']=support.hash_tensor(layer.w2_weight)
        report['native_repacked_scale_sha256']=support.hash_tensor(layer.w2_weight_scale)
        # The front uses the checkpoint's native DeepSeek V4 clamp setting.
        clamp_limit=config.get('swiglu_limit',None)
        report['front_activation']=dict(name='native_swiglu_limit' if clamp_limit is not None else 'native_silu_and_mul',
                                       clamp_limit=clamp_limit)
        for batch in args.batches:
            for block in args.m_blocks:
                case=DownCase(layer,batch,block,moe_wna16_marlin_gemm,extracted,moe_align_block_size,clamp_limit)
                retained.append(case)
                case.set_pattern('concentrated');case.capture(owned)
                for pattern in ('concentrated','balanced','light','empty'):
                    coverage=case.set_pattern(pattern)
                    case.body('native');case.body('tail');torch.cuda.synchronize()
                    eager_native=case.native_out.clone();eager_tail=case.tail_out.clone()
                    numeric=dict(tail_eager_vs_native=support.numerical(eager_tail,eager_native))
                    for cycle in range(args.replay_cycles):
                        case.graphs['native'].replay();case.graphs['tail'].replay()
                        torch.cuda.synchronize()
                        evidence=case.validate(coverage)
                        numeric[f'tail_graph_cycle{cycle}_vs_native']=support.numerical(case.tail_out,case.native_out)
                        numeric[f'tail_graph_cycle{cycle}_vs_own_eager']=support.numerical(case.tail_out,eager_tail)
                    case.body('tail_reuse');torch.cuda.synchronize()
                    case.validate(coverage)
                    numeric['same_CTA_scratch_reuse_vs_native']=support.numerical(case.tail_out,case.native_out)
                    oracle={}
                    if pattern != 'empty':
                        ref=support.reference_slot(SimpleNamespace(stage='w2',x=case.x,routing=case.routing),raw)
                        oracle={arm:{name:support.numerical(value[:1],target) for name,target in ref.items()}
                                for arm,value in (('native',case.native_out),('tail',case.tail_out))}
                    native_ms=event_ms(case.graphs['native'].replay,args.timing_repeats)
                    down_ms=event_ms(case.graphs['tail_timing'].replay,args.timing_repeats)
                    require(int(case.error.item()) == 0,'timing graph device validity failure')
                    # Timing does not modify diagnostic visits, check once more.
                    case.validate(coverage)
                    report['cases'].append(dict(same_CTA_scratch_reuse_verified=True,batch=batch,m_block=block,pattern=pattern,
                        geometry=case.geometry,numerical=numeric,oracle_numerical=oracle,**evidence,
                        native_down_ms=native_ms,extracted_down_ms=down_ms,
                        extraction_time_ratio=down_ms/native_ms,performance_policy='record_only',
                        replay_cycles=args.replay_cycles))
                    save()
        require(len(report['cases']) == len(args.batches)*5*4,'incomplete S1 case coverage')
        require(checkpoint_stats(args.model_path) == before,'checkpoint changed during S1 check')
        verify_sources(package)
        report.update(checks_complete=True,gpu_mechanisms_verified=True,
                      case_count=len(report['cases']),stop_reason='complete_case_coverage')
    except Exception as exc:
        failure=type(exc).__name__
        report.update(status='failed',error_type=failure,stop_reason='runtime_or_validity_failure')
        private=args.output_dir/'private';private.mkdir(parents=True,exist_ok=True,mode=0o700)
        log=private/f'native-tail-compute-rank{rank}.log'
        with log.open('a') as stream: traceback.print_exception(type(exc),exc,exc.__traceback__,file=stream)
        log.chmod(0o600)
    finally:
        try:
            if cuda_started:
                torch.cuda.synchronize()
                for graph in owned: graph.reset()
                torch.cuda.synchronize()
            report['cleanup']=dict(status='complete',graphs_reset=len(owned),no_process_group_created=True)
        except Exception as exc:
            failure=failure or type(exc).__name__
            report.update(status='failed',stop_reason='cleanup_failure',error_type=failure)
            report['cleanup']=dict(status='failed')
        if not failure and report.get('checks_complete'): report['status']='complete'
        save();stopped.set()
    return int(report['status'] != 'complete')


if __name__ == '__main__':
    raise SystemExit(main())
