"""Lazy source-pinned full graph hooks shared by native and Weave launchers."""
import hashlib
import importlib.abc
import importlib.machinery
import json
import os
from pathlib import Path
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from .graph_observer import BATCHES, GraphObserver, validate_graph_configuration, graph_stage_options

TARGET = 'sglang.srt.model_executor.runner_backend.full_cuda_graph_backend'
ROOT = Path(__file__).resolve().parents[2]


def graph_observer_source_sha256():
    digest = hashlib.sha256()
    for name in ('src/weave/graph_observer.py', 'src/weave/graph_hooks.py',
                 'src/weave/graph_stage_profile.py',
                 'timing_hooks/sitecustomize.py', 'configs/sglang_graph_sources.json'):
        digest.update((ROOT / name).read_bytes())
    return digest.hexdigest()


def verify_graph_sources(package_root):
    pins = json.loads((ROOT / 'configs/sglang_graph_sources.json').read_text())['sha256']
    for name, expected in pins.items():
        path = Path(package_root) / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError('graph C32 source mismatch: ' + name)
    return pins


def start_endpoint(observer, verify):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_GET(self):
            try:
                if self.path != '/graph-coverage':
                    raise ValueError('unsupported graph endpoint')
                verify()
                result, code = observer.snapshot(), 200
            except Exception as error:
                result, code = {'error': type(error).__name__, 'reason': str(error)}, 409
            data = json.dumps(result, allow_nan=False).encode()
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
    server = ThreadingHTTPServer(('127.0.0.1', int(os.environ.get('WEAVE_GRAPH_EVIDENCE_PORT_BASE', '30310')) + observer.rank), Handler)
    threading.Thread(target=server.serve_forever, daemon=True, name='weave-graph-coverage').start()
    return server


def wrap_model(model, observer, is_capturing):
    original = model.forward
    def forward(*args, **kwargs):
        batch = kwargs.get('forward_batch', args[2] if len(args) > 2 else None)
        if batch is None:
            raise ValueError('graph observer requires model forward batch')
        mode = batch.forward_mode
        phase = ('capture' if is_capturing() else 'warmup' if observer.capturing
                 else 'eager_decode' if mode.is_decode() else 'prefill' if mode.is_extend() else 'other')
        with observer.operation():
            result = original(*args, **kwargs)
            with observer.lock:
                observer.forwards[phase] += 1
            return result
    model.forward = forward
    model.weave_graph_observer = observer


def patch_backend(cls, setup):
    if getattr(cls, '_weave_graph_hook', False):
        return
    init, capture, replay = cls.__init__, cls.capture_one, cls.replay
    def initialize(self, cuda_graph_runner, **kwargs):
        observer = setup(cuda_graph_runner)
        self._weave_graph_observer, self._weave_graph_runner = observer, cuda_graph_runner
        init(self, cuda_graph_runner, **kwargs)
    def capture_one(self, shape_key, *args, **kwargs):
        observer = self._weave_graph_observer
        batch = shape_key.size
        if batch not in BATCHES or shape_key.stream_idx is not None or shape_key.variant_label is not None:
            raise ValueError('unsupported graph shape key')
        with observer.operation():
            observer.counts['capture_attempts'] += 1
            observer.capturing = True
            try:
                result = capture(self, shape_key, *args, **kwargs)
                observer.captured.add(batch)
                observer.counts['capture_successes'] += 1
                return result
            except BaseException:
                observer.counts['capture_failures'] += 1
                raise
            finally:
                observer.capturing = False
    def replay_one(self, shape_key, static_forward_batch, **kwargs):
        observer, runner = self._weave_graph_observer, self._weave_graph_runner
        raw, captured = runner.raw_bs, shape_key.size
        if raw != captured or sorted(observer.captured) != BATCHES:
            raise ValueError('graph replay requires complete exact batch capture without padding')
        counters = getattr(getattr(runner.model_runner.model, 'weave_runtime', None), 'model_batch_counters', None)
        with observer.operation():
            success = False
            sampled = False
            profiler = observer.stage_profiler
            sequence = observer.counts['successful_replays'] + 1
            if counters is not None:
                counters.begin('decode', raw, raw)
            try:
                if profiler is not None:
                    sampled = profiler.begin_replay(sequence, raw)
                # The source-pinned backend only calls CUDAGraph.replay and
                # returns its captured output. Read timing before host counters.
                result = replay(self, shape_key, static_forward_batch, **kwargs)
                success = True
                if sampled:
                    profiler.finish_replay(sequence, raw)
                return result
            except BaseException:
                if not success:
                    observer.counts['failed_replays'] += 1
                if profiler is not None:
                    profiler.abort_replay(sequence, raw)
                raise
            finally:
                # A timing read failure must not erase an actual graph replay.
                if success:
                    observer.counts['successful_replays'] += 1
                    observer.raw[str(raw)] += 1
                    observer.buckets[str(captured)] += 1
                if counters is not None:
                    counters.end(success=success)
    cls.__init__, cls.capture_one, cls.replay = initialize, capture_one, replay_one
    cls._weave_graph_hook = True


def setup_runner(runner):
    import sglang
    import torch
    from sglang.srt.runtime_context import get_parallel
    package_root = Path(sglang.__file__).parent
    pins = verify_graph_sources(package_root)
    validate_graph_configuration(runner.model_runner.server_args, get_parallel())
    if list(runner.capture_bs) != BATCHES or runner.captured_req_width != 1:
        raise ValueError('graph runner filtered the exact 1..32 buckets')
    model = runner.model_runner.model
    if hasattr(model, 'weave_graph_observer'):
        raise ValueError('multiple full graph backends for one model are unsupported')
    rank = runner.model_runner.tp_group.rank
    options = graph_stage_options()
    profiler = None
    if options is not None:
        from .graph_stage_profile import GraphStageProfiler
        kind = ('weave' if os.environ.get('SGLANG_EXTERNAL_MODEL_PACKAGE') == 'weave.sglang_ext.models'
                else 'native')
        profiler = GraphStageProfiler(torch, rank, kind, **options)
        model.weave_graph_stage_profiler = profiler
    observer = GraphObserver(rank, pins, graph_observer_source_sha256(), stage_profiler=profiler)
    wrap_model(model, observer, torch.cuda.is_current_stream_capturing)
    def verify():
        verify_graph_sources(package_root)
        if observer.observer_source != graph_observer_source_sha256():
            raise RuntimeError('graph observer source changed')
    model.weave_graph_server = start_endpoint(observer, verify)
    return observer


class Loader(importlib.abc.Loader):
    def __init__(self, loader):
        self.loader = loader
    def create_module(self, spec):
        return self.loader.create_module(spec)
    def exec_module(self, module):
        self.loader.exec_module(module)
        patch_backend(module.FullCudaGraphBackend, setup_runner)


class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != TARGET:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            raise ImportError('pinned full graph backend missing')
        spec.loader = Loader(spec.loader)
        return spec


def install_graph_hook():
    if TARGET in sys.modules:
        raise RuntimeError('graph hook must precede native backend import')
    if not any(isinstance(f, Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, Finder())
