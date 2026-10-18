"""Actual four-H100 graph smoke, exact decode B1..32 and eager prefill B65.

Public reports contain scalar evidence only. CUDA compilation/capture/replay
requires the existing offline server image; CPU tests do not establish this gate.
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
from src.weave.hopper_evidence import atomic_report, public_resources
from tests.gpu import check_hopper_serving as smoke

CAPTURE_BATCHES = tuple(range(32, 0, -1))


def expected_cases():
    return ([{'case': 'all_buckets', 'batch': batch, 'handle': 0, 'variant': 1}
             for batch in range(1, 33)]
            + [{'case': 'cross_bucket', 'batch': batch, 'handle': index % 2, 'variant': index + 2}
               for index, batch in enumerate((32, 1, 17, 7, 32))]
            + [{'case': 'after_prefill', 'batch': 32, 'handle': 0, 'variant': 9}])


def hashes():
    from src.weave.hopper_candidate import source_sha256
    from src.weave.hopper_decode import assembly_source_sha256
    from src.weave.hopper_graph import graph_runtime_source_sha256
    from src.weave.hopper_prefill_candidate import source_sha256 as prefill_source
    from src.weave.hopper_prefill import runtime_source_sha256
    from src.weave.ops import kernel_source_sha256
    return {'candidate_source_sha256': source_sha256(),
            'assembly_source_sha256': assembly_source_sha256(),
            'graph_runtime_source_sha256': graph_runtime_source_sha256(),
            'prefill_source_sha256': prefill_source(),
            'prefill_runtime_source_sha256': runtime_source_sha256(),
            'bf16_source_sha256': kernel_source_sha256()}


def metadata(runtime, ids, rank):
    rows = ids.cpu().tolist()
    expected = [[] for _ in range(8)]
    for token, row in enumerate(rows):
        for slot, expert in enumerate(row):
            if expert // 8 == rank:
                expected[expert % 8].append(token * 6 + slot)
    counts = [len(row) for row in expected]
    active = [expert for expert, count in enumerate(counts) if count]
    slots = [row + [-1] * (32 - len(row)) for row in expected]
    smoke.require(runtime.ws.group_counts.cpu().tolist() == counts, 'graph group counts differ')
    smoke.require(runtime.ws.active_count.cpu().tolist() == [len(active)], 'graph active count differs')
    smoke.require(runtime.ws.active_experts.cpu().tolist() == active + [-1] * (8 - len(active)),
                  'graph active experts differ')
    smoke.require(runtime.ws.expert_slots.cpu().tolist() == slots, 'graph stable expert slots differ')
    return {'local_slots': sum(counts), 'origin_rows': len(smoke.origin_rows(len(rows), rank))}


def assembled(slots, batch, rank):
    """Exact ordered assembly of the SAME graph-produced global slot tensor."""
    expected = torch.zeros((batch, 128), dtype=torch.float32, device=slots.device)
    for slot in range(6):
        expected.add_(slots[slot::6])
    result = expected.bfloat16()
    chunk = (batch + 3) // 4
    result[:min(rank * chunk, batch)].zero_()
    result[min((rank + 1) * chunk, batch):].zero_()
    return result


def capture(runtime, values, bound, stream, label):
    """Agree CPU launch failures before the captured NCCL collective.

    There are no D2H reads or CUDA synchronizations inside the capture body.
    The test-only Gloo agreement before NCCL makes rank-local launch errors
    coherent. NCCL itself has a bounded process-group timeout.
    """
    graph = smoke.agreed(label + ':allocate', lambda: torch.cuda.CUDAGraph())
    original = dist.all_reduce
    entered = False
    failure = None
    output = None

    def observed(tensor, *args, **kwargs):
        nonlocal entered
        smoke.require(not entered, 'capture issued more than one slot collective')
        entered = True
        def inspect():
            smoke.require(tensor is runtime._slots[values[0].shape[0]], 'capture used wrong slot prefix')
            smoke.require(kwargs.get('group') is runtime.device_group, 'capture used wrong TP group')
            smoke.require(kwargs.get('op') == dist.ReduceOp.SUM, 'capture did not use SUM')
        smoke.agreed(label + ':pre_collective', inspect)
        return original(tensor, *args, **kwargs)

    smoke.agreed(label + ':ready', lambda: torch.cuda.synchronize())
    dist.all_reduce = observed
    try:
        with torch.cuda.graph(graph, stream=stream):
            output = runtime.run(*values, bound, 7.)
    except Exception as exc:
        smoke.private_exception(label, exc)
        failure = type(exc).__name__
    finally:
        dist.all_reduce = original
    if not entered:
        try:
            smoke.agreed(label + ':pre_collective', lambda: smoke.require(False, 'capture missed collective'))
        except smoke.AgreedFailure:
            pass
    smoke.agreed(label + ':complete', lambda: smoke.require(failure is None and entered, 'graph capture failed'))
    return graph, output


def replay_and_snapshot(runtime, static, graph, batch, pointers, rank, label):
    """Poison graph-owned results, replay, then inspect before any eager decode.

    Negative metadata sentinels cannot describe valid routing, including empty
    ranks. NaN slots/output require the replay to repopulate all active results.
    Input copies and poisoning are test preparation outside graph capture.
    """
    def poison():
        for name in ('group_counts', 'active_count', 'active_experts', 'expert_slots'):
            getattr(runtime.ws, name).fill_(-2)
        runtime._slots[batch].fill_(float('nan'))
        runtime._outputs[batch].fill_(float('nan'))
    smoke.agreed(label + ':poison', poison)
    before = smoke.agreed(label + ':counter_before', runtime.device_slot_collectives_snapshot)
    smoke.agreed(label + ':replay_ready', lambda: None)
    smoke.agreed(label + ':replay', lambda: (graph.replay(), torch.cuda.synchronize()))
    after = smoke.agreed(label + ':counter_after', runtime.device_slot_collectives_snapshot)
    def snapshot():
        smoke.require(after - before == 1, 'actual graph replay device collective count differs')
        # Check exact newly planned routes before eager code can replace them.
        work = metadata(runtime, static[1], rank)
        output, slots = runtime._outputs[batch], runtime._slots[batch]
        smoke.require(bool(torch.isfinite(output).all()) and bool(torch.isfinite(slots).all()),
                      'nonfinite graph output or slots')
        smoke.require(smoke.addresses(runtime) == pointers, 'shared graph workspace addresses moved')
        smoke.require(torch.equal(output, assembled(slots, batch, rank)), 'graph ordered assembly differs')
        return output.clone(), slots.clone(), work, before, after
    return smoke.agreed(label + ':replay_snapshot', snapshot)


def evaluate(runtime, baseline, raw, bound, baseline_bound, static, graph, case, pointers, rank):
    """Update inputs, replay and snapshot, then compute record-only references."""
    batch, handle, variant = (case[key] for key in ('batch', 'handle', 'variant'))
    label = f"{case['case']}_{batch}_{variant}"
    values = smoke.agreed(label + ':inputs', lambda: smoke.synthetic_inputs(batch, variant, f'cuda:{rank}'))
    smoke.agreed(label + ':copy', lambda: [target.copy_(source) for target, source in zip(static, values)])
    output, slots, work, before, after = replay_and_snapshot(
        runtime, static, graph, batch, pointers, rank, label)
    # The graph result and its exact metadata/assembly evidence are already
    # frozen. Eager decode may now reuse the workspace for numerical records.
    eager, local_slots = smoke.run_captured(runtime, static, bound, label + ':eager')
    from tests.gpu.check_hopper_candidate import numerical, local_reference
    from src.weave.hopper_reference import expert_reference_fp8
    from tests.gpu.check_checkpoint import expert_reference

    def check():
        checks = {'eager_graph': numerical(output, eager)}
        count = min(batch, 2)
        x, ids, routing = static
        actual = slots[:count * 6].clone()
        remote = (ids[:count].flatten() // 8 != rank)
        actual[remote] = 0
        for name, oracle in (('fp8_oracle', expert_reference_fp8), ('bf16_oracle', expert_reference)):
            expected = local_reference(x[:count], ids[:count], routing[:count], raw, rank, 32, 7., oracle)
            checks[name] = numerical(actual, expected)
        baseline.run_online(x[:count], ids[:count], routing[:count], baseline_bound, 7.)
        checks['bf16_reference'] = numerical(actual, baseline.online_slot_outputs[:count * 6])
        smoke.require(all(metric['finite'] for metric in checks.values()), 'nonfinite graph/oracle differences')
        remote_cpu = static[1].cpu().flatten() // 8 != rank
        smoke.require(not bool(local_slots[rank][remote_cpu].count_nonzero()), 'nonlocal eager slots nonzero')
        return dict(case, **work, finite=True, metadata_exact=True, assembly_exact=True,
                    stable_addresses=True, binding_alive=True, device_collective_delta=after-before,
                    device_counter_before=before, device_counter_after=after,
                    input_update_verified=True, oracle_tokens=count, numerical=checks)
    return smoke.agreed(label + ':check', check)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    global torch, dist
    import torch
    import torch.distributed as dist
    import triton
    from src.weave.hopper_graph import HopperGraphDecodeRuntime
    from src.weave.hopper_prefill import HopperPrefillRuntime
    from src.weave.ops import extension as bf16_extension
    from tests.gpu import check_hopper_candidate as probe

    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1')
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    dist.init_process_group('gloo', timeout=timedelta(minutes=30))
    smoke.torch, smoke.dist, probe.torch = torch, dist, torch
    group = None
    fresh = False
    report = {'schema_version': 1, 'status': 'failed', 'numerical_policy': 'record_only',
              'raw_content_exported': False, 'rank': rank, 'world_size': dist.get_world_size(),
              'cases': [], 'case_count': 0, 'stop_reason': 'runtime_failure'}
    def save():
        atomic_report(args.output_dir / f'hopper-graph-rank{rank}.json', report)
    try:
        def inspect():
            smoke.require(dist.get_world_size() == 4 and dist.get_rank() == rank, 'four single-node ranks required')
            smoke.require(not (args.output_dir / f'hopper-graph-rank{rank}.json').exists(), 'fresh graph report required')
            props = torch.cuda.get_device_properties(rank)
            smoke.require(torch.cuda.device_count() == 4 and 'H100' in props.name
                          and (props.major, props.minor) == (9, 0), 'four visible H100 GPUs required')
            image = os.environ.get('WEAVE_IMAGE_ID', '')
            smoke.require(re.fullmatch(r'sha256:[0-9a-f]{64}', image) is not None, 'resolved existing image required')
            return {'torch': str(torch.__version__), 'cuda': torch.version.cuda,
                    'python': platform.python_version(), 'triton': triton.__version__,
                    'device_name': props.name, 'capability': [props.major, props.minor],
                    'device_count': torch.cuda.device_count(), 'image_id': image}
        report['runtime'] = smoke.agreed('inspect', inspect)
        fresh = True
        report.update(smoke.agreed('sources', hashes))
        group = dist.new_group(ranks=list(range(4)), backend='nccl', timeout=timedelta(minutes=5))
        peer = SimpleNamespace(torch=torch, device=rank, rank=rank, ep=4, h=128, i=128, topk=6)
        runtime = smoke.agreed('create_graph', lambda: HopperGraphDecodeRuntime(peer, group, experts=32))
        prefill = smoke.agreed('create_prefill', lambda: HopperPrefillRuntime(peer, group, experts=32, max_batch=65))
        baseline = smoke.agreed('create_bf16', lambda: bf16_extension().StagedWorkspace(rank,4,32,128,128,32,6))
        def loaded():
            smoke.require(runtime.candidate_source_sha256 == report['candidate_source_sha256']
                          and runtime.assembly_source_sha256 == report['assembly_source_sha256']
                          and runtime.graph_runtime_source_sha256 == report['graph_runtime_source_sha256']
                          and prefill.candidate_source_sha256 == report['prefill_source_sha256']
                          and prefill.runtime_source_sha256 == report['prefill_runtime_source_sha256'],
                          'loaded sources differ from disk')
        smoke.agreed('loaded_sources', loaded)
        device = f'cuda:{rank}'
        raw = [smoke.agreed(f'weights{handle}', lambda handle=handle: smoke.synthetic_weights(rank + handle*11, device))
               for handle in range(2)]
        bounds = [smoke.agreed(f'bind{handle}', lambda handle=handle: runtime.bind_weights(raw[handle])) for handle in range(2)]
        baseline_bounds = [smoke.agreed(f'bind_bf16{handle}', lambda handle=handle: baseline.bind_online_weights(raw[handle]))
                           for handle in range(2)]
        prefill_bound = smoke.agreed('bind_prefill', lambda: prefill.bind_weights(raw[0]))
        pointers = smoke.addresses(runtime)
        counter_pointer = runtime.device_slot_collectives.data_ptr()
        static = {batch: smoke.agreed(f'allocate_inputs{batch}', lambda batch=batch: smoke.synthetic_inputs(batch, 0, device))
                  for batch in CAPTURE_BATCHES}
        stream = smoke.agreed('capture_stream', lambda: torch.cuda.Stream())
        graphs = {}
        captures = []
        capture_counter_delta = 0
        for handle in range(2):
            for batch in CAPTURE_BATCHES:
                label = f'capture_h{handle}_b{batch}'
                # Each immutable bound handle and every exact batch is warmed
                # on the capture stream, before capture and JIT-free replay.
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    smoke.run_captured(runtime, static[batch], bounds[handle], label + ':warm')
                torch.cuda.current_stream().wait_stream(stream)
                start = smoke.agreed(label + ':counter_before', runtime.device_slot_collectives_snapshot)
                graph, output = capture(runtime, static[batch], bounds[handle], stream, label)
                end = smoke.agreed(label + ':counter_after', runtime.device_slot_collectives_snapshot)
                smoke.agreed(label + ':capture_counter', lambda: smoke.require(end == start, 'capture executed device counter'))
                capture_counter_delta += end-start
                graphs[handle, batch] = graph
                captures.append({'batch': batch, 'handle': handle})
                smoke.agreed(label + ':cached_output', lambda: smoke.require(output is runtime._outputs[batch], 'uncached graph output'))
        report.update(capture_batches=list(CAPTURE_BATCHES), capture_handles=[0,1],
                      graph_capture_count=len(captures), captures=captures,
                      capture_device_collective_delta=capture_counter_delta)
        for case in expected_cases():
            if case['case'] == 'after_prefill':
                def interleave():
                    from tests.gpu.check_hopper_prefill import inputs
                    # That helper's globals are deliberately explicit here.
                    from tests.gpu import check_hopper_prefill as prefill_check
                    prefill_check.torch = torch
                    return inputs(65, 0, 32, 128, device)
                values = smoke.agreed('prefill_inputs', interleave)
                counter_before = runtime.device_slot_collectives_snapshot()
                output, _ = smoke.run_captured(prefill, values, prefill_bound, 'prefill_interleave')
                def prefill_check():
                    smoke.require(bool(torch.isfinite(output).all()), 'nonfinite interleaved prefill')
                    smoke.require(prefill.ws is not runtime.ws, 'prefill reused decode workspace')
                    smoke.require(runtime.device_slot_collectives_snapshot() == counter_before, 'prefill incremented decode counter')
                    smoke.require(prefill.additional_slot_collectives == 1, 'prefill collective count differs')
                    return {'batch':65, 'finite':True, 'additional_slot_collectives':1, 'separate_workspace':True}
                report['prefill_interleave'] = smoke.agreed('prefill_check', prefill_check)
            item = evaluate(runtime, baseline, raw[case['handle']], bounds[case['handle']],
                            baseline_bounds[case['handle']], static[case['batch']],
                            graphs[case['handle'],case['batch']], case, pointers, rank)
            report['cases'].append(item)
            smoke.agreed('save_progress', save)
        report['ordered_finalizer'] = smoke.agreed('ordered_finalizer', lambda: smoke.ordered_finalizer_fixture(rank,device))
        report['source_hashes_after'] = smoke.agreed('sources_after', hashes)
        def complete():
            smoke.require(all(report[key] == value for key,value in report['source_hashes_after'].items()), 'source changed')
            smoke.require(counter_pointer == runtime.device_slot_collectives.data_ptr(), 'device counter moved')
            smoke.require(len(report['cases']) == 38, 'incomplete graph matrix')
            smoke.require(smoke.addresses(runtime) == pointers, 'workspace addresses moved')
        smoke.agreed('complete', complete)
        report.update(status='passed', stop_reason='none', case_count=len(report['cases']),
                      replay_count=len(report['cases']), replay_device_collective_delta=sum(item['device_collective_delta'] for item in report['cases']),
                      device_slot_collectives=runtime.device_slot_collectives_snapshot(),
                      eager_slot_collectives=runtime.additional_slot_collectives,
                      stable_addresses=True, resources=public_resources(runtime.resources))
        smoke.agreed('save', save)
        return 0
    except Exception as exc:
        smoke.private_exception('graph_main', exc)
        report.update(status='failed', case_count=len(report['cases']),
                      stop_reason=str(exc) if isinstance(exc,smoke.AgreedFailure) else 'runtime_failure')
        if fresh:
            save()
        return 1
    finally:
        if group is not None:
            dist.destroy_process_group(group)
        dist.destroy_process_group()


if __name__ == '__main__':
    raise SystemExit(main())
