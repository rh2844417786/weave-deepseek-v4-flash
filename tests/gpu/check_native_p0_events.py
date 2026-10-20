"""Offline four-rank Torch-only CUDA Graph external-event/CUPTI smoke.

Run via torchrun inside the existing SGLang image. This tests event mechanics
with BF16 matmuls, two streams and NCCL; it does not establish native model,
Marlin or throughput evidence. Raw profiler traces remain in --private-dir.
"""
import argparse
from datetime import timedelta
import hashlib
import math
import os
from pathlib import Path
import platform
import re
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.weave.hopper_evidence import atomic_report
from src.weave.native_p0 import NativeP0Profiler, native_p0_source_sha256


def hashes():
    return dict(profile_source_sha256=native_p0_source_sha256(),
                smoke_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--private-dir', type=Path)
    args = parser.parse_args(argv)
    import torch
    import torch.distributed as dist
    from src.weave.graph_cleanup import cleanup_graphs
    from src.weave.native_p0_trace import LaunchTrace

    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1')
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    dist.init_process_group('gloo', timeout=timedelta(minutes=10))
    group, owned_graphs, fresh = None, [], False
    private = args.private_dir or args.output_dir / 'private'
    report_path = args.output_dir / f'native-p0-event-rank{rank}.json'
    report = dict(schema_version=1, status='running', rank=rank,
                  world_size=dist.get_world_size(), measurement_mode='instrumented_diagnostic',
                  numerical_policy='record_only', formal_performance_claim=False,
                  model_execution_evidence=False, raw_content_exported=False,
                  cases=[], case_count=0, stop_reason='runtime_failure')

    def save():
        atomic_report(report_path, report)

    def private_exception(label, exc):
        private.mkdir(parents=True, exist_ok=True, mode=0o700)
        path = private / f'native-p0-smoke-rank{rank}.log'
        with path.open('a') as output:
            output.write(f'\n{label}: {type(exc).__name__}\n')
            traceback.print_exception(type(exc), exc, exc.__traceback__, file=output)
        path.chmod(0o600)

    def agreed(label, operation):
        result, failure = None, None
        try:
            result = operation()
        except Exception as exc:
            private_exception(label, exc)
            failure = dict(rank=rank, error_type=type(exc).__name__)
        failures = [None] * dist.get_world_size()
        dist.all_gather_object(failures, failure)
        if any(item is not None for item in failures):
            raise RuntimeError(f'{label}: rank agreement failed')
        return result

    def require(condition, message):
        if not condition:
            raise RuntimeError(message)

    def cleanup_progress(state):
        report['cleanup'] = state
        if fresh:
            atomic_report(args.output_dir / f'graph-cleanup-rank{rank}.json', state)

    try:
        def inspect():
            require(dist.get_world_size() == 4 and dist.get_rank() == rank,
                    'four single-node ranks required')
            require(not report_path.exists(), 'fresh native P0 report required')
            props = torch.cuda.get_device_properties(rank)
            require(torch.cuda.device_count() == 4 and 'H100' in props.name and
                    (props.major, props.minor) == (9, 0), 'four visible H100 GPUs required')
            image_id = os.environ.get('WEAVE_IMAGE_ID', '')
            require(re.fullmatch(r'sha256:[0-9a-f]{64}', image_id) is not None,
                    'resolved existing image identity required')
            return dict(torch=str(torch.__version__), cuda=torch.version.cuda,
                        python=platform.python_version(), device_name=props.name,
                        capability=[props.major, props.minor], device_count=torch.cuda.device_count(),
                        image_id=image_id)
        report['runtime'] = agreed('runtime', inspect)
        report['image_id'] = report['runtime']['image_id']
        fresh = True
        report.update(agreed('sources', hashes))
        group = dist.new_group(list(range(4)), backend='nccl', timeout=timedelta(minutes=3))
        trace = LaunchTrace(torch, rank, private, expected_layers=None)
        profiler = agreed('profiler', lambda: NativeP0Profiler(
            torch, rank, stride=1, capacity=4, layers=1, trace=trace))
        profiler.register_layer(0, dict(path='synthetic_two_stream_bf16_mm',
                                       model_path_evidence=False))

        def buffers():
            torch.manual_seed(4096 + rank)
            device = f'cuda:{rank}'
            x = torch.randn((32, 4096), device=device, dtype=torch.bfloat16) / 64
            weight = torch.randn((4096, 4096), device=device, dtype=torch.bfloat16) / 64
            return x, weight, torch.empty_like(x), torch.empty_like(x), torch.empty_like(x)
        x, weight, routed, shared, merged = agreed('buffers', buffers)
        origin = x.clone()
        capture_stream = torch.cuda.Stream()
        shared_stream = torch.cuda.Stream()

        def body():
            main_stream = torch.cuda.current_stream()
            with profiler.span(0, 'moe'):
                # Fork precedes both branches. No wait is inserted between them.
                shared_stream.wait_stream(main_stream)
                with profiler.span(0, 'routed'):
                    with profiler.span(0, 'gate_up'):
                        torch.mm(x, weight, out=routed)
                    with profiler.span(0, 'activation'):
                        routed.relu_()
                    with profiler.span(0, 'down'):
                        torch.mm(routed, weight, out=merged)
                with torch.cuda.stream(shared_stream):
                    with profiler.span(0, 'shared'):
                        torch.mm(x, weight, out=shared)
                        shared.relu_()
                main_stream.wait_stream(shared_stream)
                with profiler.span(0, 'shared_merge'):
                    merged.add_(shared)
                with profiler.span(0, 'all_reduce'):
                    dist.all_reduce(merged, group=group)

        def warmup():
            capture_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(capture_stream):
                for _ in range(3):
                    body()
            torch.cuda.current_stream().wait_stream(capture_stream)
            torch.cuda.synchronize()
        agreed('warmup', warmup)
        graph = torch.cuda.CUDAGraph()
        owned_graphs.append(graph)  # Partial capture still must be reset.

        def capture():
            with torch.cuda.graph(graph, stream=capture_stream):
                body()
            torch.cuda.synchronize()
        agreed('capture', capture)

        for sequence in range(1, 5):
            def run():
                x.copy_(origin * sequence)
                require(profiler.begin_replay(sequence, 32), 'smoke requires every sample')
                try:
                    graph.replay()
                    sample = profiler.finish_replay(sequence, 32)
                except BaseException:
                    profiler.abort_replay(sequence, 32)
                    raise
                require(bool(torch.isfinite(merged).all()), 'nonfinite synthetic graph output')
                intervals = sample['layers']['0']['intervals']
                require(intervals['shared_merge']['start_ms'] + .001 >=
                        max(intervals['routed']['end_ms'], intervals['shared']['end_ms']),
                        'native join boundary precedes branch completion')
                require(sample['sequence'] == sequence and sample['b32_ordinal'] == sequence,
                        'sample freshness identity differs')
                norm = float(merged.float().norm().item())
                require(math.isfinite(norm), 'nonfinite output norm')
                return dict(sequence=sequence, batch=32, sample=sample, finite=True,
                            output_norm=norm, fresh_events_verified=True,
                            join_order_verified=True, input_scale=sequence)
            report['cases'].append(agreed(f'replay_{sequence}', run))
            agreed('progress', save)
        report['native_p0_profile'] = agreed('snapshot', profiler.snapshot)
        report['source_hashes_after'] = agreed('sources_after', hashes)

        def validate():
            snapshot = report['native_p0_profile']
            require(all(report[key] == value for key, value in report['source_hashes_after'].items()),
                    'source identity changed')
            require(snapshot['launch_trace']['status'] == 'complete',
                    'actual CUDA launch metadata missing')
            require(snapshot['profile_source_sha256'] == report['profile_source_sha256'],
                    'loaded profiler source identity differs')
            require(len(report['cases']) == 4 and snapshot['sampled_total'] == 4,
                    'incomplete smoke coverage')
            require(len({item['output_norm'] for item in report['cases']}) == 4,
                    'updated inputs did not change synthetic output')
        agreed('validate', validate)
        report.update(status='gpu_checks_passed_cleanup_pending', gpu_checks_status='passed',
                      stop_reason='none', case_count=4, replay_count=4, graph_capture_count=1,
                      external_events=True, two_streams=True, nccl_all_reduce=True,
                      generic_launch_metadata_verified=True,
                      shape=dict(batch=32, matrix_rows=32, matrix_columns=4096, dtype='bfloat16'))
        agreed('save', save)
    except Exception as exc:
        private_exception('main', exc)
        report.update(status='failed', stop_reason='runtime_failure',
                      error_type=type(exc).__name__, case_count=len(report['cases']))
        if fresh:
            save()
    finally:
        try:
            cleanup_graphs(owned_graphs, torch, dist, group, cleanup_progress)
        except Exception as exc:
            private_exception('cleanup', exc)
            report.update(status='failed', stop_reason='cleanup_failure')
        else:
            if report['status'] == 'gpu_checks_passed_cleanup_pending':
                report['status'] = 'passed'
        if fresh:
            save()
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
