#!/usr/bin/env python3
"""EP4 CUDA-event calibration of original layer0 tiled MXFP4 experts, offline.

Run via torchrun in the existing pinned image after a current-source all-layer
core gate. Loading/building precedes the measuring window, which is <=1200s.
No package, model, image or dataset downloads are attempted.
"""
import argparse
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.weave.calibration import (BUCKETS,KERNEL,TimingStore,atomic_json,case_key,
                                   run_calibration,work_units,CalibrationJournal,
                                   execute_gpu_iteration,run_diagnostic_case,capture_local_action)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--core-run-dir',type=Path,required=True)
    parser.add_argument('--budget-seconds',type=float,default=1200.)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--invocation-id',default=os.environ.get('WEAVE_CALIBRATION_INVOCATION_ID','standalone'))
    parser.add_argument('--repetitions',type=int,default=3)
    parser.add_argument('--warmups',type=int,default=1)
    parser.add_argument('--kernel-timeout-seconds',type=float,default=60.,
                        help='device protocol wait guard, independent of remaining measurement budget')
    parser.add_argument('--diagnose-case',help='one tokens,c,K,mode case; debug-only, never publish profile')
    args = parser.parse_args(argv)
    if not 0 < args.budget_seconds <= 1200:
        parser.error('--budget-seconds must be finite in (0,1200]')
    if args.repetitions < 2 or args.warmups < 1:
        parser.error('at least two repetitions and one warmup required')
    if args.output_dir.resolve() == args.core_run_dir.resolve():
        parser.error('calibration output must differ from core evidence directory')
    if not 0<args.kernel_timeout_seconds<=1200:
        parser.error('kernel timeout must be finite in (0,1200]')
    if args.diagnose_case:
        if args.resume: parser.error('diagnostics use a fresh separate directory, not --resume')
        try:
            n,c,k,mode=args.diagnose_case.split(',')
            args.diagnose_case={'tokens':int(n),'c':int(c),'K':int(k),'mode':mode}
        except (ValueError,TypeError): parser.error('diagnose-case requires tokens,c,K,mode')
        if (args.diagnose_case['tokens'] not in BUCKETS or args.diagnose_case['c']<0
                or args.diagnose_case['K'] not in (1,2,4,8)
                or args.diagnose_case['mode'] not in ('compute','communication','mixed')):
            parser.error('diagnostic case must use a required bucket/c/K/mode')
        if args.output_dir.exists() and any(args.output_dir.iterdir()):
            # Host launcher writes launch.json/log in its directory; checked there.
            if any(p.name not in ('launch.json','calibration.log','gpu-before.json') for p in args.output_dir.iterdir()):
                parser.error('diagnostics require a fresh evidence directory')
    return args


def validate_core_evidence(core_dir,model_path,source_sha256,actual_runtime):
    """Refuse missing/stale/nonpassing current-source all43/three-kernel evidence."""
    from src.weave.checkpoint_identity import checkpoint_stats
    from src.weave.service_acceptance import validate_checkpoint_gate
    try:
        runtime = json.loads((core_dir/'runtime.json').read_text())
        identity = json.loads((core_dir/'checkpoint_identity.json').read_text())
        ranks = [json.loads((core_dir/f'checkpoint-rank{r}.json').read_text()) for r in range(4)]
        if identity.get('verified_stats') != checkpoint_stats(model_path):
            raise ValueError('read-only checkpoint files differ from core identity; rerun core gate')
        index_sha = hashlib.sha256((model_path/'model.safetensors.index.json').read_bytes()).hexdigest()
        if identity.get('files_sha256',{}).get('model.safetensors.index.json') != index_sha:
            raise ValueError('checkpoint index differs from content identity')
        gate = validate_checkpoint_gate(runtime,ranks,source_sha256=source_sha256,
                                       index_sha256=index_sha,checkpoint_sha256=identity['sha256'])
        if runtime.get('core_validation_status') != 'passed':
            raise ValueError('core functional gate must have passed')
        if gate['image_id'] != actual_runtime.get('resolved_image_id'):
            raise ValueError('actual image differs from passed core evidence')
        return gate,identity
    except (OSError,KeyError,TypeError,json.JSONDecodeError) as exc:
        raise ValueError(f'absent or malformed core evidence; rerun all43/three-kernel gate: {exc}') from exc


def main(argv=None):
    args = parse_args(argv)
    # These settings constrain all subsequent local model loading.
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_DATASETS_OFFLINE'] = '1'
    try:
        import torch
        import torch.distributed as dist
        from src.weave.ops import PeerRuntime,kernel_source_sha256
        from src.weave.weights import load_raw_experts
        from src.weave.checkpoint_identity import checkpoint_stats
        from src.weave.runtime_contract import validate_inventory
        from scripts.check_runtime import capture
    except ImportError as exc:
        raise RuntimeError('required packages absent; use the existing pinned SGLang image, no online pip') from exc
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA/Hopper required; execute torchrun4 inside the existing pinned image')
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    # Loading 43 checkpoint identities can be slow; this is setup, outside budget.
    dist.init_process_group('gloo',timeout=timedelta(seconds=1800))
    if dist.get_world_size() != 4 or dist.get_rank() != rank:
        raise ValueError('calibration requires local EP4 ranks/ordinals 0..3')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    stopped = {'signal':None}
    def signal_request(number,frame):
        # Deferred until the shared Gloo control point; a rank must not unwind
        # while other peers are still inside the persistent CUDA IPC kernel.
        stopped['signal'] = number
    for number in (signal.SIGTERM,signal.SIGINT):
        signal.signal(number,signal_request)

    journal=None
    def collective_action(function):
        """Make ordinary Python setup/I/O failures visible to every rank."""
        result,error = capture_local_action(function,journal=journal)
        errors = [None]*4
        dist.all_gather_object(errors,error)
        if any(errors):
            raise RuntimeError(f'collective calibration failure: {errors}')
        return result

    setup = [None]
    def rank0_setup():
        if rank:
            return None
        actual = capture()
        errors = validate_inventory(actual,'core')
        if errors or not actual.get('resolved_image_id'):
            raise ValueError(f'current runtime/image contract failed: {errors}; WEAVE_IMAGE_ID required')
        source = kernel_source_sha256()
        gate,checkpoint = validate_core_evidence(args.core_run_dir,args.model_path,source,actual)
        config = json.loads((args.model_path/'config.json').read_text())
        dimensions = [config.get(k) for k in ('hidden_size','moe_intermediate_size','n_routed_experts','num_experts_per_tok','num_hidden_layers')]
        if dimensions != [4096,2048,256,6,43]:
            raise ValueError(f'original model shape differs: {dimensions}')
        return {'actual_runtime':actual,'source_sha256':source,'core_gate':gate,
                'checkpoint_identity':checkpoint,'clamp':float(config['swiglu_limit'])}
    setup[0] = collective_action(rank0_setup)
    dist.broadcast_object_list(setup,src=0)
    setup = setup[0]
    initial_stats = collective_action(lambda:checkpoint_stats(args.model_path))
    runtime = collective_action(lambda:PeerRuntime(None,2048,4096,2048,6,256,kernel='tiled'))
    if runtime.source_sha256 != setup['source_sha256']:
        raise ValueError('kernel source changed since core evidence/setup')
    device = torch.device('cuda',rank)
    raw = collective_action(lambda:load_raw_experts(args.model_path,0,rank,4,device=device))
    prop = torch.cuda.get_device_properties(rank)
    identity = {'kernel':KERNEL,'source_sha256':runtime.source_sha256,
                'device_key':f'{prop.name};sm{prop.major}{prop.minor};{prop.multi_processor_count}',
                'hidden_size':4096,'intermediate_size':2048,'experts':256,'top_k':6,
                'rank':rank,'workers':prop.multi_processor_count,'ep_size':4,
                'checkpoint_content_sha256':setup['checkpoint_identity']['sha256'],
                'image_id':setup['actual_runtime']['resolved_image_id'],
                'runtime':{'python':platform.python_version(),'torch':str(torch.__version__),'cuda':torch.version.cuda},
                'measurement':{'buckets':list(BUCKETS),'repetitions':args.repetitions,'warmups':args.warmups}}
    identities = [None]*4
    dist.all_gather_object(identities,{k:v for k,v in identity.items() if k != 'rank'})
    if any(v != identities[0] for v in identities):
        raise ValueError('rank hardware/runtime/grid identities differ; identical benchmark order required')
    store = collective_action(lambda:TimingStore(args.output_dir/f'timings-rank{rank}.json',identity,resume=args.resume))
    completed = [None]*4
    dist.all_gather_object(completed,[case_key(p) for p in store.points])
    common = set(completed[0]).intersection(*(set(v) for v in completed[1:]))
    collective_action(lambda:store.reconcile_completed(common))
    # Preload deterministic BF16 inputs and routes. Weights are original layer0;
    # input fixtures are deliberately ordinary finite activations, not rates.
    def prepare_inputs():
        generator = torch.Generator(device=device).manual_seed(314159+rank)
        x_all = (torch.randn((2048,4096),generator=generator,device=device)*.125).bfloat16()
        routing_all = torch.full((2048,6),1/6,device=device,dtype=torch.float32)
        routes = {mode:torch.tensor([[owner*64+s for s in range(6)]]*2048,
                                    device=device,dtype=torch.int32)
                  for mode,owner in [('compute',rank),('communication',(rank+1)%4),('mixed',(rank+1)%4)]}
        events = [(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True))
                  for _ in range(args.repetitions)]
        torch.cuda.synchronize()
        return x_all,routing_all,routes,events
    x_all,routing_all,routes,events = collective_action(prepare_inputs)
    dist.barrier()
    start = [time.monotonic() if rank == 0 else None]
    dist.broadcast_object_list(start,src=0)
    start = start[0]
    # The parent starts its CUDA-block watchdog from this exact clock marker.
    journal=collective_action(lambda:CalibrationJournal(args.output_dir,identity,args.invocation_id,
                                                       start,args.budget_seconds,
                                                       initial_completed_cases=len(store.points)))
    collective_action(journal.save)
    control = {'remaining':args.budget_seconds,'reason':None}

    def should_stop():
        requested = [None]*4
        dist.all_gather_object(requested,stopped['signal'])
        decision = [None]
        if rank == 0:
            remaining = args.budget_seconds-(time.monotonic()-start)
            reason = 'signal' if any(requested) else ('deadline' if remaining <= 0 else None)
            decision[0] = {'stop':reason is not None,'remaining':remaining,'reason':reason}
        dist.broadcast_object_list(decision,src=0)
        control.update(decision[0])
        return decision[0]['stop']

    def measure_case(case):
        samples,workload,tiles = [],None,None
        n = case['tokens']
        mode = None if case['mode'] == 'mixed' else case['mode']
        def exclude_attempt():
            value = dict(case,samples_ms=samples,warmups=args.warmups,workload=workload,tile_counters=tiles)
            collective_action(lambda:store.record_excluded(value,reason=control['reason'] or 'interrupted'))
        for iteration in range(args.warmups+args.repetitions):
            if should_stop():
                exclude_attempt()
                return None
            collective_action(lambda:journal.begin(case,iteration,
                elapsed_seconds=time.monotonic()-start,completed_cases=len(store.points),
                kernel_timeout_seconds=args.kernel_timeout_seconds))
            dist.barrier()
            timed = iteration >= args.warmups
            if timed:
                begin,end = events[iteration-args.warmups]
            def execute_iteration():
                def launch(timeout):
                    runtime.run(x_all[:n],routes[case['mode']][:n],routing_all[:n],raw,
                            fixed_plan=(case['c'],case['K'],0),calibration_mode=mode,
                            timeout_seconds=timeout,clamp=setup['clamp'])
                execute_gpu_iteration(launch,torch.cuda.synchronize,journal=journal,
                    kernel_timeout_seconds=args.kernel_timeout_seconds,
                    begin_event=begin.record if timed else None,end_event=end.record if timed else None)
            collective_action(execute_iteration)
            # All peers have synchronized before this CPU publication. It is
            # outside CUDA begin/end events and cannot become a GPU duration.
            collective_action(journal.record_completed_iteration)
            # Absolute deadline checked after events/synchronize too. A late
            # duration never becomes a completed measured timing point.
            if timed:
                def read_elapsed():
                    journal.stage('event_elapsed_time')
                    return float(begin.elapsed_time(end))
                samples.append(collective_action(read_elapsed))
            if should_stop():
                exclude_attempt()
                return None
            def read_counters():
                journal.stage('workload_counter_read')
                return runtime.ws.workload.cpu().tolist(),runtime.ws.tile_counters.cpu().tolist()
            observed_work,observed_tiles = collective_action(read_counters)
            def verify_counters():
                if workload is not None and (workload != observed_work or tiles != observed_tiles):
                    raise ValueError('actual workload/tile counters changed across repetitions')
            collective_action(verify_counters)
            workload,tiles = observed_work,observed_tiles
        def validate_case():
            journal.stage('case_output_validation')
            value = dict(case,samples_ms=samples,warmups=args.warmups,workload=workload,
                         tile_counters=tiles,completed_before_deadline=True,
                         host_iteration_timings=list(journal.value['iterations']))
            value.update(work_units(value,4096,2048))
            output = runtime.ws.output[:n]
            if not torch.isfinite(output).all().item():
                raise ValueError('nonfinite calibration output')
            if case['mode'] == 'communication' and torch.count_nonzero(output).item():
                raise ValueError('communication-only calibration must return zero output')
            return value
        return collective_action(validate_case)

    try:
        # Disk failures are coordinated; peers must never enter the next CUDA
        # case while one process is unwinding a failed raw-point publication.
        original_append = store.append
        def append_point(*pos,**kwargs):
            def append():
                original_append(*pos,**kwargs)
                journal.committed(len(store.points))
            return collective_action(append)
        store.append = append_point
        if args.diagnose_case:
            if os.environ.get('CUDA_LAUNCH_BLOCKING') not in ('0','1'):
                raise ValueError('diagnostic case requires explicit CUDA_LAUNCH_BLOCKING=0 or1')
            result=collective_action(lambda:run_diagnostic_case(store,args.diagnose_case,measure_case,should_stop))
            collective_action(lambda:atomic_json(args.output_dir/f'diagnostic-rank{rank}.json',
                dict(result,rank=rank,identity=identity,invocation_id=args.invocation_id,
                     launch_blocking=os.environ['CUDA_LAUNCH_BLOCKING']=='1',
                     host_iteration_timings=list(journal.value['iterations']),
                     kernel_timeout_seconds=args.kernel_timeout_seconds)))
            return 0 if result['status']=='diagnostic_passed' else 124
        result = collective_action(lambda:run_calibration(store,measure_case,should_stop,
                                  lambda:args.budget_seconds-control['remaining'],args.budget_seconds))
        if result['status'] == 'measured':
            def publish():
                if initial_stats != checkpoint_stats(args.model_path):
                    raise ValueError('checkpoint changed during calibration; refusing measured publication')
                if kernel_source_sha256() != identity['source_sha256']:
                    raise ValueError('CUDA source changed during calibration; refusing measured publication')
                atomic_json(args.output_dir/f'profile-bank-rank{rank}.json',result['bundle'])
            collective_action(publish)
        summary = {'status':result['status'],'rank':rank,'completed_cases':result['completed_cases'],
                   'initial_completed_cases':journal.value['initial_completed_cases'],
                   'committed_this_invocation':len(store.points)-journal.value['initial_completed_cases'],
                   'kernel_timeout_seconds':args.kernel_timeout_seconds,
                   'budget_seconds':args.budget_seconds,'elapsed_seconds':time.monotonic()-start,
                   'stop_reason':control['reason'],'identity':identity,'core_gate':setup['core_gate'],
                   'actual_runtime':setup['actual_runtime']}
        collective_action(lambda:atomic_json(args.output_dir/f'calibration-rank{rank}.json',summary))
        print(json.dumps({'rank':rank,'status':result['status'],'completed_cases':result['completed_cases']}),flush=True)
        return 0 if result['status'] == 'measured' else 124
    except Exception as error:
        try: journal.fail(error)
        except OSError as evidence_error: error.add_note(f'failure evidence I/O failed: {evidence_error}')
        raise
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    sys.exit(main())
