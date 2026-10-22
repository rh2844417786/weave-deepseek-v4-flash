"""Offline S3–S5 real-checkpoint resident-tail mechanism check on four H100s.

Uses original repacked expert weights and native gate/up+activation. Shared
partials are deterministic BF16 test inputs, so this is an operator mechanism
check, not full-model shared-expert or throughput evidence. Full task traces
are written only to the private output directory, outside all timed windows.
"""
import argparse
from dataclasses import asdict
from datetime import timedelta
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.weave.hopper_evidence import atomic_report
from src.weave.native_tail_patch import METHOD, verify_sources, load_down_op
from tests.gpu import check_native_tail_compute as compute
from tests.gpu import check_native_budget as support

H, I, E, TOPK, EP = support.H, support.I, support.E, support.TOPK, support.EP
require = support.require


def any_interval_overlap(left, right):
    """Actual nonempty task intervals, never envelopes across idle gaps."""
    left, right = sorted(left), sorted(right)
    i = j = 0
    while i < len(left) and j < len(right):
        a, b = left[i], right[j]
        if max(a[0], b[0]) < min(a[1], b[1]):
            return True
        if a[1] <= b[1]:
            i += 1
        else:
            j += 1
    return False


def check_visits(visits, expected_task_ids):
    expected = set(expected_task_ids)
    require(visits == [int(i in expected) for i in range(len(visits))],
            'exact down task bitmap differs (missing, repeated, or wrong task)')


def stolen_then_communication(worker_trace, down, communication):
    """(worker,start,end) tuples; the same initial COMM worker must do both."""
    for worker, _, end in down:
        if worker_trace[worker][0] != 1:
            continue
        if any(other == worker and start >= end for other, start, _ in communication):
            return True
    return False


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model-path', 'output-dir', 'cache-root'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--table-dir', type=Path)
    parser.add_argument('--sanitizer-smoke', action='store_true',
                        help='targeted B32 static matrix inside the controller compute-sanitizer stage')
    parser.add_argument('--batches', type=int, nargs='+', default=[32, 65])
    parser.add_argument('--layer', type=int, default=0)
    parser.add_argument('--replay-cycles', type=int, default=5)
    parser.add_argument('--timeout-seconds', type=int, default=3600)
    parser.add_argument('--controlled-delay-cycles', type=int, default=2000000)
    args = parser.parse_args(argv)
    if args.layer not in range(43) or any(batch < 1 for batch in args.batches):
        parser.error('layer must be 0..42 and batches positive')
    if args.sanitizer_smoke and (args.table_dir or args.batches != [32]):
        parser.error('--sanitizer-smoke requires static mode and exactly --batches 32')
    if not args.table_dir and not args.sanitizer_smoke and (32 not in args.batches or not any(batch > 32 for batch in args.batches)):
        parser.error('static mechanism acceptance requires batch32 and an independent larger batch')
    if args.replay_cycles < 5 or args.timeout_seconds < 30 or args.controlled_delay_cycles < 1:
        parser.error('need >=5 replays, >=30 second watchdog, positive controlled GPU delay')
    return args


def plan_dict(plan):
    return plan if isinstance(plan, dict) else asdict(plan)


def analyze_trace(snapshot, coverage, *, tiles, capacity, plan):
    visits, trace, events = snapshot['visits'], snapshot['trace'], snapshot['events']
    check_visits(visits, coverage['expected_task_ids'])
    require(len(events) == capacity * tiles + 20, 'diagnostic event capacity mismatch')
    chunks = plan['chunks']
    expected = set(coverage['expected_task_ids'])
    down, comm, later_down, early_comm = [], [], [], []
    stolen = 0
    down_count = [0] * len(trace)
    comm_count = [0] * len(trace)
    for task, row in enumerate(events[:capacity * tiles]):
        worker, begin, end, kind = row
        if task not in expected:
            require(row == [0, 0, 0, 0], 'unexpected down task event')
            continue
        require(kind == 1 and 0 <= worker < len(trace) and 0 < begin <= end,
                'malformed or missing down task interval')
        down.append((worker, begin, end)); down_count[worker] += 1
        if trace[worker][0] == 1:
            stolen += 1
        if task % tiles >= tiles // chunks:
            later_down.append((begin, end))
    for task, row in enumerate(events[capacity * tiles:]):
        worker, begin, end, kind = row
        if task >= 5 * chunks:
            require(row == [0, 0, 0, 0], 'unexpected communication task event')
            continue
        expected_kind = 2 if task < chunks else 3
        require(kind == expected_kind and 0 <= worker < len(trace) and 0 < begin <= end,
                'malformed or missing peer task interval')
        comm.append((worker, begin, end)); comm_count[worker] += 1
        chunk = task if task < chunks else (task - chunks) // 4
        if chunk == 0:
            early_comm.append((begin, end))
    for worker, row in enumerate(trace):
        require(row[0] == int(worker < plan['communication_workers']), 'worker role differs from selected plan')
        require(row[2] > 0 and row[3] >= row[2], 'worker interval missing')
        require(row[4] == down_count[worker] and row[5] == comm_count[worker],
                'worker counters disagree with exact task events')
        require(row[6] == (down_count[worker] if row[0] else 0), 'stolen task counter differs')
    delays = [row[8] - row[7] for row in trace if row[7] and row[8] and row[8] >= row[7]]
    return dict(exact_down_tasks_verified=True, exact_peer_tasks_verified=True,
                local_down_tasks=len(expected), local_communication_tasks=len(comm),
                initial_comm_down_tasks=stolen,
                stolen_worker_resumed_communication=stolen_then_communication(trace, down, comm),
                overlap_observed=any_interval_overlap(later_down, early_comm),
                overlap_scope='same_GPU_actual_chunk0_peer_task_and_later_chunk_down_task_intervals',
                ready_observed_to_start_ns=delays,
                ready_delay_scope='same_GPU_first_observed_ready_to_first_communication_start',
                distinct_observed_sms=len({row[1] for row in trace}),
                task_event_sha256=hashlib.sha256(json.dumps(events,separators=(',', ':')).encode()).hexdigest())


def make_down(case):
    from src.weave.native_tail_front import DownInputs, NativeGeometry
    sorted_ids, expert_ids, padded = case.grouping
    return DownInputs(activated_pairs=case.x, sorted_token_ids=sorted_ids,
        local_expert_ids=expert_ids, num_tokens_post_padded=padded,
        topk_weights=case.routing, topk_ids=case.ids, original_slot_mapping=sorted_ids,
        w2=case.layer.w2_weight, w2_scales=case.layer.w2_weight_scale,
        geometry=NativeGeometry(case.batch, TOPK, case.batch * TOPK, H, I, case.block, E // EP, case.capacity),
        owners=(case.hidden,case.up,case.locks,case.layer))


def load_checkpoint(args, rank, torch):
    from src.weave.checkpoint_identity import checkpoint_stats, sha256_file
    from src.weave.weights import load_raw_experts
    from sglang.srt.layers.quantization.marlin_utils_fp4 import prepare_moe_mxfp4_layer_for_marlin
    config = json.loads((args.model_path / 'config.json').read_text())
    require(tuple(config[k] for k in ('hidden_size', 'moe_intermediate_size', 'n_routed_experts',
        'num_experts_per_tok', 'num_hidden_layers')) == (H,I,E,TOPK,43), 'original checkpoint dimensions required')
    before = checkpoint_stats(args.model_path)
    raw = load_raw_experts(args.model_path,args.layer,rank,EP,device=f'cuda:{rank}')
    record = dict(layer=args.layer, local_expert_tensor_sha256=[support.hash_tensor(t) for t in raw],
        config_sha256=sha256_file(args.model_path/'config.json'),
        index_sha256=sha256_file(args.model_path/'model.safetensors.index.json'))
    layer = torch.nn.Module()
    layer.w13_weight = torch.nn.Parameter(torch.cat(raw[:2],dim=1).view(torch.int8),requires_grad=False)
    layer.w2_weight = torch.nn.Parameter(raw[2].view(torch.int8),requires_grad=False)
    layer.w13_weight_scale = torch.nn.Parameter(torch.cat(raw[3:5],dim=1).view(torch.float8_e8m0fnu),requires_grad=False)
    layer.w2_weight_scale = torch.nn.Parameter(raw[5].view(torch.float8_e8m0fnu),requires_grad=False)
    layer.orig_dtype = torch.bfloat16
    prepare_moe_mxfp4_layer_for_marlin(layer)
    record['repacked_w2_sha256'] = support.hash_tensor(layer.w2_weight)
    record['repacked_scales_sha256'] = support.hash_tensor(layer.w2_weight_scale)
    return layer, raw, config, before, record


def runtime_identity(rank, torch):
    from src.weave.checkpoint_identity import sha256_file
    from src.weave.runtime_contract import load_contract
    props = torch.cuda.get_device_properties(rank)
    distribution = support._sgl_kernel_distribution()
    image = os.environ.get('WEAVE_IMAGE_ID', '')
    require(re.fullmatch(r'sha256:[0-9a-f]{64}',image) is not None,'resolved existing WEAVE_IMAGE_ID required')
    require(platform.system() == 'Linux' and platform.machine() == 'x86_64','Linux x86_64 required')
    record = dict(python=platform.python_version(), torch=str(torch.__version__),cuda=torch.version.cuda,
        sglang=importlib.metadata.version('sglang'),sgl_kernel=distribution.version,
        image_id=image,device=props.name,capability=[props.major,props.minor])
    for key,expected in (('python','python_version'),('torch','torch_version'),('cuda','torch_cuda'),('sglang','sglang_version')):
        require(record[key] == load_contract()[expected], 'runtime contract mismatch: '+key)
    require(record['capability'] == [9,0], 'H100 SM90 target required')
    record['sgl_kernel_binary_sha256'] = {str(p):sha256_file(distribution.locate_file(p))
        for p in (distribution.files or ()) if str(p).endswith('.so')}
    require(record['sgl_kernel_binary_sha256'], 'installed kernel binary inventory unavailable')
    return record


def main(argv=None):
    args = parse_args(argv)
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1')
    rank = int(os.environ.get('LOCAL_RANK','0'))
    report_path = args.output_dir / f'native-tail-executor-rank{rank}.json'
    require(not report_path.exists(), 'fresh output directory required')
    mode = 'measured' if args.table_dir else 'static'
    report = dict(schema_version=1, method=METHOD, component='resident_executor', rank=rank,
        status='running', numerical_policy='record_only', mode=mode, source_implemented=True,
        gpu_mechanisms_verified=False, full_model_graph_execution_verified=False,
        throughput_result_observed=False, checks_complete=False, cases=[], cleanup=dict(status='not_started'),
        memcheck='external_compute_sanitizer_required', raw_content_exported=False,
        sanitizer_scope='targeted_B32' if args.sanitizer_smoke else 'not_requested',
        limitations=['Selected-layer operator graph with synthetic routing, not full model service.',
            'Shared inputs are deterministic BF16 partials, not checkpoint shared FP8 execution.',
            'Controlled stealing uses a diagnostic GPU delay on COMPUTE roles; no throughput claim.',
            'Uninjected observations are synthetic operator replays, not natural serving observations.',
            'Finite oracle, prediction and output differences are record_only.'])
    lock, stopped = threading.Lock(), threading.Event()
    def save():
        with lock: atomic_report(report_path,report)
    def watchdog():
        if not stopped.wait(args.timeout_seconds):
            with lock:
                report.update(status='failed',stop_reason='process_timeout_or_cuda_hang')
                report['cleanup']=dict(status='incomplete',reason='watchdog_exit')
                atomic_report(report_path,report)
            os._exit(124)
    save();threading.Thread(target=watchdog,daemon=True).start()
    initialized = False
    group = engine = None
    graphs, retained = [], []
    failure = None
    total_replays = planner_checks = 0
    initial_sequence = final_sequence = None
    observed = dict(overlap_observed=False, controlled_stealing_observed=False,
        controlled_stolen_worker_resumed_communication=False, uninjected_operator_stealing_observed=False)
    try:
        import torch
        import torch.distributed as dist
        compute.torch = support.torch = torch
        from src.weave.checkpoint_identity import checkpoint_stats, sha256_file
        from src.weave.native_tail_cost import select_joint
        from src.weave.native_tail_executor import NativeTailExecutor
        from src.weave.native_tail_runtime import native_tail_source_sha256
        from src.weave.native_tail_transport import assess_transport_evidence
        from sglang.kernels.ops.moe.moe_wna16_marlin import moe_wna16_marlin_gemm
        from sglang.kernels.ops.moe.moe_topk_sum import moe_topk_sum
        from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import moe_align_block_size
        require(torch.cuda.is_available(), 'CUDA unavailable; use offline existing SGLang image')
        require(int(os.environ.get('WORLD_SIZE','0')) == 4, 'exactly four torchrun ranks required')
        torch.cuda.set_device(rank)
        dist.init_process_group('nccl',timeout=timedelta(seconds=args.timeout_seconds))
        initialized = True
        group = dist.new_group(backend='gloo',timeout=timedelta(seconds=args.timeout_seconds))
        require(dist.get_rank(group) == rank and dist.get_world_size(group) == 4, 'four single-node ranks required')
        def sync():
            torch.cuda.synchronize();dist.barrier(group=group)
        report['runtime'] = runtime_identity(rank,torch)
        package = support.package_root()
        report['upstream_sha256'] = verify_sources(package)
        report['source_sha256'] = native_tail_source_sha256()
        report['harness_sha256'] = sha256_file(Path(__file__))
        layer, raw, config, checkpoint_before, checkpoint_record = load_checkpoint(args,rank,torch)
        report['checkpoint'] = checkpoint_record
        report['shared_input_source'] = 'deterministic_private_BF16_test_partial'
        scale = float(config['routed_scaling_factor'])
        report['routed_scaling_factor'] = scale
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision('highest')
        extracted = load_down_op(package,args.cache_root/f'rank{rank}'/'static_down')
        engine = NativeTailExecutor(torch,rank,package,group,max_batch=max(args.batches),
            cache_root=args.cache_root/f'rank{rank}'/'executor',
            directory=args.table_dir or args.output_dir/'private'/'static',mode=mode,validation=True)
        if args.table_dir:
            report['profile_sha256'] = engine.profiles[str(rank)]['profile_sha256']
            report['profile_file_sha256'] = sha256_file(args.table_dir/f'native-tail-rank{rank}.json')
            report['profiles_sha256'] = {r:p['profile_sha256'] for r,p in engine.profiles.items()}
        private = args.output_dir/'private'
        private.mkdir(parents=True,exist_ok=True,mode=0o700)
        for batch in args.batches:
            block = next(b for b in (8,16,32,48,64) if batch*TOPK/(E//EP)/b < .9)
            case = compute.DownCase(layer,batch,block,moe_wna16_marlin_gemm,extracted,
                                   moe_align_block_size,config.get('swiglu_limit',None))
            retained.append(case)
            coverage = case.set_pattern('concentrated')
            down = make_down(case)
            generator = torch.Generator(device=case.device).manual_seed(8723+rank+batch)
            shared = (torch.randn((batch,H),generator=generator,device=case.device)*.125).bfloat16()
            ctx = engine.prepare_inputs(down,shared,args.layer)
            retained.extend((down,shared,ctx))
            plans = [plan_dict(p) for p in ctx.plans]
            if mode == 'static':
                def index(comm,chunks,steal,serial=False):
                    target=dict(communication_workers=comm,chunks=chunks,steal_tiles=steal,serial=serial)
                    require(target in plans, 'required executable static mechanism plan is unavailable')
                    return plans.index(target)
                modes = [('serial',index(0,1,0,True),0),('C1',index(1,1,0),0),
                         ('C2',index(1,2,0),0),('steal_uninjected',index(2,2,1),0),
                         ('steal_controlled',index(2,2,1),args.controlled_delay_cycles)]
                engine.set_fixed(args.layer,batch,modes[0][1],diagnostic=True)
            else:
                modes = [('measured',None,0)]
                engine.set_search(args.layer,batch,diagnostic=True)
            reference_storage, reference = compute.bf16_guarded(batch,H,case.device)
            summed = torch.empty_like(reference)
            retained.extend((reference_storage,reference,summed))
            def front():
                case.up.zero_();case.native_gemm('w13',case.up)
                if case.clamp_limit is None:
                    from sglang.kernels.ops.activation.activation import silu_and_mul
                    silu_and_mul(case.up,case.x)
                else:
                    from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import swiglu_limit_func
                    swiglu_limit_func(case.x,case.up,case.clamp_limit)
            def reference_body():
                front();case.native_out.zero_();case.native_gemm('w2',case.native_out)
                moe_topk_sum(case.native_out.view(batch,TOPK,H),summed)
                reference.copy_(shared);reference.add_(summed,alpha=scale)
                dist.all_reduce(reference)
            def tail_body():
                front()
                return engine(down,shared,layer_id=args.layer,routed_scaling_factor=scale,output_width=H)
            # Warm up both compiled paths and NCCL before graph capture.
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    reference_body();tail_body()
            sync()
            reference_graph, tail_graph = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
            graphs.extend((reference_graph,tail_graph))
            with torch.cuda.graph(reference_graph,stream=stream): reference_body()
            with torch.cuda.graph(tail_graph,stream=stream): tail_body()
            sync()
            pointer_set = tuple(t.data_ptr() for t in (case.x,*case.grouping,case.routing,shared,
                ctx.pair,ctx.output,ctx.mode,ctx.visits,ctx.trace,ctx.events))
            sequence = engine.snapshot_context(args.layer,batch)['protocol']['sequence']
            if initial_sequence is None: initial_sequence = sequence
            for mode_name,selected,delay in modes:
                if selected is None: engine.set_search(args.layer,batch,diagnostic=True)
                else: engine.set_fixed(args.layer,batch,selected,diagnostic=True,delay_cycles=delay)
                for pattern in ('concentrated','balanced','light','empty_rank2'):
                    local_pattern = 'empty' if pattern == 'empty_rank2' and rank == 2 else (
                        'balanced' if pattern == 'empty_rank2' else pattern)
                    coverage = case.set_pattern(local_pattern)
                    for replay in range(args.replay_cycles):
                        # New payload every replay catches stale epoch/slot reuse.
                        shared.fill_((rank+1)*(replay+1)/64.)
                        reference_graph.replay();sync()
                        before_stats = ctx.stats.cpu().tolist()
                        tail_graph.replay();sync()
                        snapshot = engine.snapshot_context(args.layer,batch,private=True)
                        after_stats = ctx.stats.cpu().tolist()
                        require(after_stats[0]-before_stats[0] == 1 and after_stats[1]-before_stats[1] == 1
                            and after_stats[34]-before_stats[34] == 1, 'one front/tail/committed invocation coverage differs')
                        protocol = snapshot['protocol']
                        assessed = assess_transport_evidence(protocol)
                        require(assessed['valid'], 'protocol task/ack/nonfinite/canary check failed: '+str(assessed['failures']))
                        require(protocol['sequence'] == sequence+1, 'graph replay did not advance exactly one GPU sequence')
                        sequence=protocol['sequence']; final_sequence=sequence; total_replays+=1
                        require(snapshot['canary_ok'], 'executor pair/output canary corruption')
                        require(pointer_set == tuple(t.data_ptr() for t in (case.x,*case.grouping,case.routing,shared,
                            ctx.pair,ctx.output,ctx.mode,ctx.visits,ctx.trace,ctx.events)), 'graph binding addresses changed')
                        require(int(ctx.pair[case.ids.flatten() == -1].count_nonzero()) == 0,
                                'remote expert pair slots nonzero')
                        support.inspect_guards(case.lock_storage,case.locks,zero=True)
                        compute.guards(reference_storage);compute.guards(case.native_storage)
                        actual_plan=plans[snapshot['selected_index']]
                        if selected is not None:
                            require(snapshot['selected_index'] == selected and snapshot['chosen_indices'] == [selected]*4,
                                    'fixed cross-rank plan consensus differs')
                        mechanisms=analyze_trace(snapshot,coverage,tiles=case.tiles,capacity=case.capacity,plan=actual_plan)
                        numerical=dict(global_tail_vs_native_NCCL=support.numerical(ctx.output,reference),
                                       local_pairs_vs_native_down=support.numerical(ctx.pair,case.native_out))
                        oracle={}
                        if local_pattern != 'empty':
                            from types import SimpleNamespace
                            refs=support.reference_slot(SimpleNamespace(stage='w2',x=case.x,routing=case.routing),raw)
                            oracle={name:support.numerical(ctx.pair[:1],value) for name,value in refs.items()}
                        planner={}
                        if selected is None:
                            features=[None]*4
                            dist.all_gather_object(features,snapshot['features'],group=group)
                            cpu=select_joint([table['surface'] for table in ctx.rank_tables],features,ctx.plans,batch*TOPK)
                            require(cpu['indices'] == snapshot['chosen_indices'], 'GPU planner disagrees with CPU on current replay workload')
                            planner=dict(cpu_indices=cpu['indices'],gpu_indices=snapshot['chosen_indices'],
                                cpu_predicted_us=cpu['predicted_us'],gpu_leader_predicted_us=snapshot.get('leader_prediction_us'),
                                current_workload_verified=True,prediction_error_policy='record_only')
                            planner_checks+=1
                        observed['overlap_observed'] |= mechanisms['overlap_observed']
                        if delay:
                            observed['controlled_stealing_observed'] |= mechanisms['initial_comm_down_tasks'] > 0
                            observed['controlled_stolen_worker_resumed_communication'] |= mechanisms['stolen_worker_resumed_communication']
                        else:
                            observed['uninjected_operator_stealing_observed'] |= mechanisms['initial_comm_down_tasks'] > 0
                        tag=f'b{batch}-{mode_name}-{pattern}-{replay}-rank{rank}'
                        trace_path=private/(tag+'.json')
                        atomic_report(trace_path,dict(snapshot=snapshot,expected_task_ids=coverage['expected_task_ids']))
                        trace_path.chmod(0o600)
                        report['cases'].append(dict(batch=batch,m_block=block,mode=mode_name,pattern=pattern,
                            replay=replay,sequence=sequence,plan=actual_plan,resources=ctx.resources,
                            controlled_compute_delay_cycles=delay,mechanisms=mechanisms,numerical=numerical,
                            oracle_numerical=oracle,planner=planner,protocol_valid=True,canary_ok=True,
                            native_locks_zero=True,stable_graph_addresses=True,exact_task_coverage=True,
                            front_and_tail_calls=1,private_evidence_sha256=sha256_file(trace_path)))
                        save()
        expected=len(args.batches)*(1 if mode == 'measured' else 5)*4*args.replay_cycles
        require(len(report['cases']) == expected, 'incomplete executor case coverage')
        require(checkpoint_stats(args.model_path) == checkpoint_before,'checkpoint changed during executor check')
        require(native_tail_source_sha256() == report['source_sha256'],'native tail source changed during validation')
        verify_sources(package)
        report.update(observed,case_count=len(report['cases']),expected_case_count=expected,
            checks_complete=True,stop_reason='complete_case_coverage',
            graph_capture_replay=dict(verified=True,replays=total_replays,
                initial_sequence=initial_sequence,final_sequence=final_sequence,
                actual_native_front_captured=True,reference='native_down_topk_sum_shared_add_NCCL_graph'),
            planner_validation=dict(complete=mode == 'measured' and planner_checks == expected,
                                    checked_replays=planner_checks,features_from_current_replay=True))
    except Exception as exc:
        failure=type(exc).__name__
        report.update(status='failed',error_type=failure,stop_reason='runtime_or_validity_failure')
        private=args.output_dir/'private';private.mkdir(parents=True,exist_ok=True,mode=0o700)
        log=private/f'native-tail-executor-rank{rank}.log'
        with log.open('a') as stream: traceback.print_exception(type(exc),exc,exc.__traceback__,file=stream)
        log.chmod(0o600)
    finally:
        try:
            if initialized:
                torch.cuda.synchronize()
                for graph in graphs: graph.reset()
                torch.cuda.synchronize()
                if engine is not None: engine.close()
            report['cleanup']=dict(status='complete',graphs_reset=len(graphs),ipc_closed=engine is None or engine.closed)
        except Exception as exc:
            failure=failure or type(exc).__name__
            report.update(status='failed',error_type=failure,stop_reason='cleanup_failure')
            report['cleanup']=dict(status='failed')
        if not failure and report.get('checks_complete'): report['status']='complete'
        save()
    if initialized:
        try:
            reports=[None]*4
            dist.all_gather_object(reports,report,group=group)
            valid=all(r['status']=='complete' and r['cleanup']['status']=='complete' for r in reports)
            union={key:any(r.get(key,False) for r in reports) for key in observed}
            verified=valid and (mode == 'measured' or (union['overlap_observed'] and
                union['controlled_stealing_observed'] and union['controlled_stolen_worker_resumed_communication']))
            report['gpu_mechanisms_verified']=verified
            report['all_rank_checks_complete']=valid
            report['all_rank_mechanisms']=union
            if not valid: report['status']='failed'
            save()
            if rank == 0:
                atomic_report(args.output_dir/'summary.json',dict(method=METHOD,mode=mode,
                    status='complete' if valid else 'failed',numerical_policy='record_only',
                    source_sha256=report.get('source_sha256'),source_implemented=True,
                    gpu_mechanisms_verified=verified,full_model_graph_execution_verified=False,
                    throughput_result_observed=False,checks_complete=valid,cleanup=dict(status='complete' if valid else 'failed'),
                    mechanism_observations=union,ranks=[dict(rank=r['rank'],status=r['status'],
                        case_count=r.get('case_count'),profile_sha256=r.get('profile_sha256')) for r in reports]))
            dist.destroy_process_group(group)
            dist.destroy_process_group()
        except Exception as exc:
            report.update(status='failed',error_type=type(exc).__name__,stop_reason='rank_agreement_or_group_cleanup_failure')
            report['cleanup']=dict(status='failed');save()
    stopped.set()
    return int(report['status'] != 'complete')


if __name__ == '__main__':
    raise SystemExit(main())
