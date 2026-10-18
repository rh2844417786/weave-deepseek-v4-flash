"""Four-H100 external-event smoke using the actual diagnostic graph runtime.

Run in the existing offline image. Four changed B32 inputs exercise captured
compute, NCCL SUM, device accounting and ordered finalize. Numerical differences
are record-only. Local CPU tests do not establish the CUDA event timing gate.
"""
import argparse
from datetime import timedelta
import os
from pathlib import Path
import platform
import re
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.weave.hopper_evidence import atomic_report
from tests.gpu import check_hopper_graph as graph_check
from tests.gpu import check_hopper_serving as smoke


def hashes():
    from src.weave.graph_stage_profile import graph_stage_source_sha256
    return dict(graph_check.hashes(), profile_source_sha256=graph_stage_source_sha256())


def expected_cases():
    return [dict(sequence=index, batch=32, variant=index) for index in range(1, 5)]


def evaluate(runtime, original, baseline, raw, bound, original_bound, bf16_bound,
             static, graph, original_graph, pointers, rank, case):
    label = f"stage_event_{case['sequence']}"
    values = smoke.agreed(label + ':inputs', lambda: smoke.synthetic_inputs(32, case['variant'], f'cuda:{rank}'))
    def prepare():
        smoke.require(not torch.equal(static[0], values[0]) and not torch.equal(static[1], values[1]),
                      'replay inputs must change')
        for target, value in zip(static, values):
            target.copy_(value)
        for name in ('group_counts', 'active_count', 'active_experts', 'expert_slots'):
            getattr(runtime.ws, name).fill_(-2)
        runtime._slots[32].fill_(float('nan'))
        runtime._outputs[32].fill_(float('nan'))
    smoke.agreed(label + ':prepare', prepare)
    before = smoke.agreed(label + ':counter_before', runtime.device_slot_collectives_snapshot)
    def replay():
        selected = runtime.profiler.begin_replay(case['sequence'], 32)
        smoke.require(selected, 'smoke must sample every replay')
        try:
            graph.replay()
            return runtime.profiler.finish_replay(case['sequence'], 32)
        except Exception:
            runtime.profiler.abort_replay(case['sequence'], 32)
            raise
    sample = smoke.agreed(label + ':sample', replay)
    after = smoke.agreed(label + ':counter_after', runtime.device_slot_collectives_snapshot)
    def inspect():
        smoke.require(after - before == 1, 'graph device collective counter differs')
        work = graph_check.metadata(runtime, static[1], rank)
        output, slots = runtime._outputs[32], runtime._slots[32]
        smoke.require(bool(torch.isfinite(output).all()) and bool(torch.isfinite(slots).all()),
                      'nonfinite stage graph output')
        smoke.require(torch.equal(output, graph_check.assembled(slots, 32, rank)), 'ordered assembly differs')
        smoke.require(smoke.addresses(runtime) == pointers, 'stage graph workspace addresses moved')
        smoke.require(sample['sequence'] == case['sequence'] and sample['b32_ordinal'] == case['sequence']
                      and len(sample['layers']) == 1, 'fresh stage sample identity differs')
        return output.clone(), slots.clone(), work
    output, slots, work = smoke.agreed(label + ':inspect', inspect)
    # All event timestamps and actual graph output are now frozen. References
    # may execute only after the sample read, never between replay and read.
    smoke.agreed(label + ':original_replay', lambda: (original_graph.replay(), torch.cuda.synchronize()))
    original_output = smoke.agreed(label + ':original_snapshot', lambda: original._outputs[32].clone())
    eager_output, _ = smoke.run_captured(original, static, original_bound, label + ':original_eager')
    from tests.gpu.check_hopper_candidate import numerical, local_reference
    from src.weave.hopper_reference import expert_reference_fp8
    from tests.gpu.check_checkpoint import expert_reference
    def numerical_records():
        records = {'original_graph': numerical(output, original_output),
                   'original_eager': numerical(output, eager_output)}
        x, ids, routing = static
        count = 2
        local = slots[:count * 6].clone()
        local[ids[:count].flatten() // 8 != rank] = 0
        for name, oracle in (('fp8_oracle', expert_reference_fp8), ('bf16_oracle', expert_reference)):
            reference = local_reference(x[:count], ids[:count], routing[:count], raw, rank, 32, 7., oracle)
            records[name] = numerical(local, reference)
        baseline.run_online(x[:count], ids[:count], routing[:count], bf16_bound, 7.)
        records['bf16_reference'] = numerical(local, baseline.online_slot_outputs[:count * 6])
        smoke.require(all(row['finite'] for row in records.values()), 'nonfinite numerical record')
        return records
    records = smoke.agreed(label + ':numerical_records', numerical_records)
    return dict(case, stage_sample=sample, finite=True, metadata_exact=True, assembly_exact=True,
                stable_addresses=True, input_update_verified=True, fresh_events_verified=True,
                device_collective_delta=after-before, numerical=records, **work)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    global torch, dist
    import torch
    import torch.distributed as dist
    import triton
    from src.weave.graph_stage_profile import GraphStageProfiler, HopperStageDecodeRuntime
    from src.weave.hopper_graph import HopperGraphDecodeRuntime
    from src.weave.ops import extension as bf16_extension
    from tests.gpu import check_hopper_candidate as probe
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1')
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    dist.init_process_group('gloo', timeout=timedelta(minutes=30))
    smoke.torch, smoke.dist, probe.torch = torch, dist, torch
    graph_check.torch, graph_check.dist = torch, dist
    group, owned_graphs, fresh = None, [], False
    report = dict(schema_version=1, status='running', rank=rank, world_size=dist.get_world_size(),
                  measurement_mode='instrumented_diagnostic', numerical_policy='record_only',
                  raw_content_exported=False, cases=[], case_count=0, stop_reason='runtime_failure')
    def save():
        atomic_report(args.output_dir / f'stage-event-rank{rank}.json', report)
    def cleanup_progress(state):
        report['cleanup'] = state
        if fresh:
            atomic_report(args.output_dir / f'graph-cleanup-rank{rank}.json', state)
    try:
        def inspect():
            smoke.require(dist.get_world_size() == 4 and dist.get_rank() == rank, 'four single-node ranks required')
            smoke.require(not (args.output_dir / f'stage-event-rank{rank}.json').exists(), 'fresh event report required')
            props = torch.cuda.get_device_properties(rank)
            smoke.require(torch.cuda.device_count() == 4 and 'H100' in props.name
                          and (props.major, props.minor) == (9, 0), 'four visible H100 GPUs required')
            image = os.environ.get('WEAVE_IMAGE_ID', '')
            smoke.require(re.fullmatch(r'sha256:[0-9a-f]{64}', image) is not None, 'resolved existing image required')
            return dict(torch=str(torch.__version__), cuda=torch.version.cuda, python=platform.python_version(),
                        triton=triton.__version__, device_name=props.name, capability=[props.major, props.minor],
                        device_count=torch.cuda.device_count(), image_id=image)
        report['runtime'] = smoke.agreed('inspect', inspect)
        report['image_id'] = report['runtime']['image_id']
        fresh = True
        report.update(smoke.agreed('sources', hashes))
        group = dist.new_group(ranks=list(range(4)), backend='nccl', timeout=timedelta(minutes=5))
        peer = SimpleNamespace(torch=torch, device=rank, rank=rank, ep=4, h=128, i=128, topk=6)
        profiler = smoke.agreed('create_profiler', lambda: GraphStageProfiler(torch, rank, 'weave', stride=1, capacity=4, layers=1))
        runtime = smoke.agreed('create_stage_runtime', lambda: HopperStageDecodeRuntime(peer, group, profiler, experts=32))
        original = smoke.agreed('create_original_runtime', lambda: HopperGraphDecodeRuntime(peer, group, experts=32))
        baseline = smoke.agreed('create_bf16', lambda: bf16_extension().StagedWorkspace(rank,4,32,128,128,32,6))
        def loaded():
            for item in (runtime, original):
                smoke.require(item.candidate_source_sha256 == report['candidate_source_sha256']
                              and item.assembly_source_sha256 == report['assembly_source_sha256']
                              and item.graph_runtime_source_sha256 == report['graph_runtime_source_sha256'],
                              'loaded graph sources differ from disk')
            smoke.require(runtime.profile_source_sha256 == report['profile_source_sha256']
                          and profiler.profile_source_sha256 == report['profile_source_sha256'],
                          'loaded stage sources differ from disk')
        smoke.agreed('loaded_sources', loaded)
        raw = smoke.agreed('weights', lambda: smoke.synthetic_weights(rank, f'cuda:{rank}'))
        bound = smoke.agreed('bind_stage', lambda: runtime.bind_weights(raw))
        smoke.agreed('register_layer', lambda: runtime.register_layer(0, bound))
        original_bound = smoke.agreed('bind_original', lambda: original.bind_weights(raw))
        bf16_bound = smoke.agreed('bind_bf16', lambda: baseline.bind_online_weights(raw))
        static = smoke.agreed('inputs', lambda: smoke.synthetic_inputs(32, 0, f'cuda:{rank}'))
        stream = smoke.agreed('capture_stream', lambda: torch.cuda.Stream())
        graphs = []
        for label, item, handle in (('stage', runtime, bound), ('original', original, original_bound)):
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                smoke.run_captured(item, static, handle, label + ':warm')
            torch.cuda.current_stream().wait_stream(stream)
            before = item.device_slot_collectives_snapshot()
            graph, _ = graph_check.capture(item, static, handle, stream, label + ':capture', owned_graphs=owned_graphs)
            after = item.device_slot_collectives_snapshot()
            smoke.agreed(label + ':capture_counter', lambda: smoke.require(before == after, 'capture executed counter'))
            graphs.append(graph)
        pointers = smoke.addresses(runtime)
        for case in expected_cases():
            report['cases'].append(evaluate(runtime, original, baseline, raw, bound, original_bound,
                bf16_bound, static, graphs[0], graphs[1], pointers, rank, case))
            smoke.agreed('save_progress', save)
        report['stage_profile'] = smoke.agreed('snapshot', profiler.snapshot)
        report['source_hashes_after'] = smoke.agreed('sources_after', hashes)
        smoke.agreed('complete', lambda: smoke.require(
            len(report['cases']) == 4 and all(report[key] == value for key, value in report['source_hashes_after'].items()),
            'incomplete event coverage or source changed'))
        report.update(status='gpu_checks_passed_cleanup_pending', gpu_checks_status='passed', stop_reason='none',
                      case_count=4, replay_count=4, graph_capture_count=2, external_events=True,
                      replay_device_collective_delta=sum(row['device_collective_delta'] for row in report['cases']),
                      shape=dict(batch=32, hidden=128, intermediate=128, experts=32, topk=6, layers=1))
        smoke.agreed('save', save)
    except Exception as exc:
        smoke.private_exception('stage_event_main', exc)
        report.update(status='failed', case_count=len(report['cases']),
                      stop_reason=str(exc) if isinstance(exc, smoke.AgreedFailure) else 'runtime_failure')
        if fresh:
            save()
    finally:
        from src.weave.graph_cleanup import cleanup_graphs
        try:
            cleanup_graphs(owned_graphs, torch, dist, group, cleanup_progress)
        except Exception as exc:
            smoke.private_exception('stage_event_cleanup', exc)
            report.update(status='failed', stop_reason='cleanup_failure')
        else:
            if report['status'] == 'gpu_checks_passed_cleanup_pending':
                report['status'] = 'passed'
        if fresh:
            save()
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
