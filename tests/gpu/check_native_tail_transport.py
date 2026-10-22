#!/usr/bin/env python3
"""Four-GPU offline S2 check. No model, new packages, or external downloads.

CUDA_VISIBLE_DEVICES must be the allocation returned by GPUQ. Example inside the
existing SGLang image:
  PYTHONPATH=src torchrun --standalone --nproc-per-node=4 \
    tests/gpu/check_native_tail_transport.py --output-dir /srv/weave-work/private/tail-s2

Repeat with compute-sanitizer --tool memcheck --target-processes all wrapping
 torchrun (or the site's supported child-process launcher). Canaries are a bounded
workspace check; they are not a substitute for the recorded memcheck result.
"""
import argparse
import json
import os
from pathlib import Path
import threading
import time
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--tokens', type=int, default=32)
    parser.add_argument('--hidden', type=int, default=4096)
    parser.add_argument('--chunks', type=int, choices=(1, 2, 4), default=1)
    parser.add_argument('--replays', type=int, default=9)
    parser.add_argument('--timing-repeats', type=int, default=20)
    parser.add_argument('--timeout-seconds', type=float, default=20.)
    parser.add_argument('--watchdog-seconds', type=float, default=900.)
    return parser.parse_args()


def main():
    args = parse_args()
    import torch
    import torch.distributed as dist
    from weave.native_tail_transport import NativeTailTransport, TransportShape, METHOD
    TransportShape(args.tokens, args.hidden, args.chunks)
    if args.replays < 5 or args.timing_repeats < 1:
        raise ValueError('at least five graph replays are required to exercise both reused slots')
    if int(os.environ.get('WORLD_SIZE', '0')) != 4:
        raise RuntimeError('launch with torchrun --nproc-per-node=4 on a GPUQ four-device allocation')
    # Local ordinal within CUDA_VISIBLE_DEVICES; no physical GPU ids are assumed.
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('nccl')
    group = dist.new_group(backend='gloo')
    rank = dist.get_rank(group)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    timer = threading.Timer(args.watchdog_seconds, lambda: os._exit(124))
    timer.daemon = True
    timer.start()
    cases = []
    runtime = None
    report = {'method': METHOD, 'component': 'standalone_transport',
              'numerical_policy': 'record_only', 'rank': rank,
              'source_implemented': True, 'gpu_mechanisms_verified': False,
              'full_model_graph_execution_verified': False, 'throughput_result_observed': False,
              'memcheck': 'external_compute_sanitizer_required', 'cleanup': False}

    def sync_ranks():
        torch.cuda.synchronize()
        dist.barrier(group=group)

    def differences(actual, reference):
        delta = actual.float()-reference.float()
        denominator = reference.float().norm().item()
        return {'relative_l2': delta.norm().item()/denominator if denominator else delta.norm().item(),
                'max_abs': delta.abs().max().item() if delta.numel() else 0.,
                'nonfinite_result': int((~torch.isfinite(actual)).sum().item())}

    def compare_case(name, rt, data, *, delay=0., graph=None):
        reference = data.clone()
        if reference.numel():
            dist.all_reduce(reference)
        sync_ranks()
        if graph is None:
            output = rt.run(data, chunks=args.chunks, layer=7,
                            timeout_seconds=args.timeout_seconds, delay_seconds=delay)
        else:
            graph.replay()
            output = rt.output[:len(data)]
        torch.cuda.synchronize()
        evidence = rt.snapshot(args.chunks)
        evidence.update(differences(output, reference))
        evidence['name'] = name
        evidence['valid'] &= evidence['nonfinite_result'] == 0
        cases.append(evidence)
        # This gate never reads finite error magnitudes.
        if not evidence['valid']:
            raise RuntimeError(f'{name}: invalid protocol, coverage, memory, or nonfinite output: {evidence}')
        return output

    try:
        runtime = NativeTailTransport(group, args.tokens, args.hidden)
        report.update(source_sha256=runtime.source_sha256, source_manifest=runtime.module.source_manifest,
                      resources=dict(runtime.ws.resources()), chunks=args.chunks)
        torch.manual_seed(413+rank)
        data = torch.randn((args.tokens, args.hidden), device='cuda', dtype=torch.bfloat16)
        compare_case('all_inputs_ready', runtime, data)
        compare_case('delayed_rank_3', runtime, data, delay=.02 if rank == 3 else 0.)
        zero_local = data.clone()
        if rank == 2:
            zero_local.zero_()
        compare_case('empty_local_contribution_rank_2', runtime, zero_local)
        compare_case('zero_token_payload', runtime, data[:0])

        # An actual cooperative launch is captured. Replay advances sequence on GPU.
        capture_stream = torch.cuda.Stream()
        capture_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(capture_stream):
            runtime.run(data, chunks=args.chunks, layer=7, timeout_seconds=args.timeout_seconds)
        sync_ranks()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=capture_stream):
            runtime.run(data, chunks=args.chunks, layer=7, timeout_seconds=args.timeout_seconds)
        sync_ranks()
        initial_sequence = runtime.snapshot(args.chunks)['sequence']
        for replay in range(args.replays):
            # Mutating input values catches a captured epoch or payload accidentally reused.
            data.fill_((rank+1)*(replay+1)/16.)
            compare_case(f'graph_replay_{replay}', runtime, data, graph=graph)
            if cases[-1]['sequence'] != initial_sequence+replay+1:
                raise RuntimeError('CUDA Graph replay did not advance exactly one GPU sequence')
        report['graph_capture_replay'] = {'verified': True, 'replays': args.replays,
                                          'initial_sequence': initial_sequence,
                                          'final_sequence': cases[-1]['sequence']}

        # Pre-generated partials are ready before timing. Each number is local-rank
        # elapsed time; no cross-GPU clock subtraction or invented shared timeline.
        native_ms, peer_ms = [], []
        native = data.clone()
        for _ in range(args.timing_repeats):
            native.copy_(data)
            sync_ranks()
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record(); dist.all_reduce(native); end.record(); end.synchronize()
            native_ms.append(begin.elapsed_time(end))
            sync_ranks()
            begin.record(); runtime.run(data, chunks=args.chunks, layer=7,
                                       timeout_seconds=args.timeout_seconds); end.record(); end.synchronize()
            peer_ms.append(begin.elapsed_time(end))
        timing_evidence = runtime.snapshot(args.chunks)
        if not timing_evidence['valid']:
            raise RuntimeError(f'invalid communication timing coverage: {timing_evidence}')
        report['communication_timing_ms'] = {'native_all_reduce': native_ms,
                                             'peer_transport': peer_ms,
                                             'inputs_ready_before_timing': True,
                                             'mode': 'eager', 'performance_policy': 'record_only'}
        # Destroy graph before freeing the addresses it captures.
        del graph
        sync_ranks()
        runtime.close(); runtime = None

        # Terminal protocol failures get NEW workspaces/sessions. A failed session
        # cannot be reset and silently reused. All peers must finish before teardown.
        failure_cases = [('timeout', 1), ('gpu_cancel', 5), ('identity_mismatch', 2), ('nonfinite', 3)]
        for name, expected_code in failure_cases:
            runtime = NativeTailTransport(group, args.tokens, args.hidden)
            sync_ranks()
            started = time.monotonic()
            failure_data = data.clone()
            if name == 'nonfinite' and rank == 2 and failure_data.numel():
                failure_data.view(-1)[0] = float('nan')
            if name == 'nonfinite' and not failure_data.numel():
                runtime.close(); runtime = None
                continue
            runtime.run(failure_data, chunks=args.chunks,
                        layer=(8 if name == 'identity_mismatch' and rank == 3 else 7),
                        timeout_seconds=.1 if name == 'timeout' else args.timeout_seconds,
                        delay_seconds=.5 if rank == 3 and name in ('timeout', 'gpu_cancel') else 0.,
                        cancel_seconds=.02 if rank == 0 and name == 'gpu_cancel' else 0.)
            torch.cuda.synchronize()
            evidence = runtime.snapshot(args.chunks)
            if evidence['abort_code'] != expected_code or not evidence['canary_ok']:
                raise RuntimeError(f'{name} did not terminate with expected bounded abort: {evidence}')
            runtime.close(); runtime = None
            cases.append({'name': name, 'expected_abort_code': expected_code,
                          'observed_abort_code': evidence['abort_code'], 'cleanup': True,
                          'elapsed_seconds': time.monotonic()-started, 'valid': True})
        report['cleanup'] = True
        report['gpu_mechanisms_verified'] = True
        report['valid'] = True
    except BaseException as exc:
        report['valid'] = False
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        if runtime is not None:
            try:
                runtime.close()
                report['cleanup'] = True
            except BaseException as exc:
                report['cleanup_error'] = f'{type(exc).__name__}: {exc}'
        report['cases'] = cases
        (args.output_dir/f'rank{rank}.json').write_text(json.dumps(report, indent=2)+'\n')
        timer.cancel()
    reports = [None]*4
    dist.all_gather_object(reports, report, group=group)
    if rank == 0:
        summary = {'method': METHOD, 'numerical_policy': 'record_only',
                   'source_implemented': True,
                   'gpu_mechanisms_verified': all(r['gpu_mechanisms_verified'] for r in reports),
                   'full_model_graph_execution_verified': False,
                   'throughput_result_observed': False,
                   'cleanup': all(r['cleanup'] for r in reports),
                   'ranks': reports}
        (args.output_dir/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
        print(json.dumps({k: v for k, v in summary.items() if k != 'ranks'}, indent=2))
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
