import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
from src.weave.graph_observer import GraphObserver, BATCHES, canonical_graph_config, validate_graph_configuration
from src.weave.graph_hooks import patch_backend, wrap_model, verify_graph_sources
from src.weave.staged_decode import ModelBatchCounters, validate_decode_configuration


class GraphHooksTest(unittest.TestCase):
    def fixture(self):
        observer = GraphObserver(0, {'model': 'a'}, 'b')
        state = NS(capturing=False)
        mode = NS(is_decode=lambda: True, is_extend=lambda: False)
        batch = NS(forward_mode=mode, batch_size=32)
        model = NS(forward=lambda *args: None, weave_runtime=NS(model_batch_counters=ModelBatchCounters()))
        wrap_model(model, observer, lambda: state.capturing)
        runner = NS(raw_bs=32, model_runner=NS(model=model))
        class Backend:
            def __init__(self, runner): pass
            def capture_one(self, key, fn):
                fn(); fn()
                state.capturing = True
                try: fn()
                finally: state.capturing = False
            def replay(self, key, batch):
                if getattr(batch, 'fail', False): raise RuntimeError('replay failure')
                return 'replayed'
        patch_backend(Backend, lambda _: observer)
        return Backend(runner), observer, model, batch, runner

    def test_only_actual_replay_counts(self):
        backend, observer, model, batch, runner = self.fixture()
        for n in BATCHES:
            backend.capture_one(NS(size=n, stream_idx=None, variant_label=None), lambda: model.forward(None, None, batch))
        snap = observer.snapshot()
        self.assertEqual(snap['model_forwards']['warmup'], 64)
        self.assertEqual(snap['model_forwards']['capture'], 32)
        self.assertEqual(snap['successful_replays'], 0)
        backend.replay(NS(size=32), batch)
        snap = observer.snapshot()
        self.assertEqual(snap['successful_replays'], 1)
        self.assertEqual(snap['raw_batch_histogram'], {'32': 1})
        self.assertEqual(model.weave_runtime.model_batch_counters.snapshot()['by_phase']['decode'], {'32': 1})
        model.forward(None, None, batch)
        self.assertEqual(observer.snapshot()['model_forwards']['eager_decode'], 1)
        runner.raw_bs = 31
        with self.assertRaises(ValueError): backend.replay(NS(size=32), batch)

    def test_active_snapshot_and_failure(self):
        backend, observer, model, batch, runner = self.fixture()
        with observer.operation():
            with self.assertRaises(ValueError): observer.snapshot()
        def fail(): raise RuntimeError('capture failed')
        with self.assertRaises(RuntimeError): backend.capture_one(NS(size=32, stream_idx=None, variant_label=None), fail)
        snap = observer.snapshot()
        self.assertEqual(snap['capture_failures'], 1)
        self.assertEqual(snap['captured_batches'], [])
        self.assertFalse(observer.capturing)

    def test_failed_replay_releases_activity_and_batch_counters(self):
        backend, observer, model, batch, runner = self.fixture()
        observer.captured.update(BATCHES)
        batch.fail = True
        with self.assertRaises(RuntimeError): backend.replay(NS(size=32), batch)
        snap = observer.snapshot()
        self.assertEqual(snap['failed_replays'], 1)
        self.assertEqual(snap['successful_replays'], 0)
        self.assertEqual(snap['raw_batch_histogram'], {})
        self.assertEqual(model.weave_runtime.model_batch_counters.snapshot()['failed_forwards'], 1)

    def test_config_and_source_fail_closed(self):
        config = canonical_graph_config()
        args = NS(cuda_graph_config=NS(**{k:NS(**v) for k,v in config.items()}), max_running_requests=32, disable_cuda_graph_padding=True)
        validate_graph_configuration(args)
        args.cuda_graph_config.decode.bs = [1,32]
        with self.assertRaises(ValueError): validate_graph_configuration(args)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError): verify_graph_sources(Path(tmp))

    def test_graph_optin_keeps_legacy_eager_constraint(self):
        kw = dict(decode_implementation='hopper_fp8', kernel='tiled', hot_cache=True, allow_decode_graph=True, fixed_plan=(1,1,0))
        with self.assertRaises(ValueError): validate_decode_configuration(**kw)
        validate_decode_configuration(**kw, graph_c32=True)
        kw['decode_implementation']='staged_direct'
        with self.assertRaises(ValueError): validate_decode_configuration(**kw, graph_c32=True)

class GraphLaunchTest(unittest.TestCase):
    def test_common_exact_configuration_and_rejections(self):
        import json, os, subprocess, sys
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES':'0,1,2,3', 'WEAVE_MAX_RUNNING_REQUESTS':'32', 'WEAVE_CUDA_GRAPH_MAX_BS_DECODE':'32'}
        base = [sys.executable, 'scripts/run_native_ep.py', '--graph-c32', '--request-timing', '--decode-graph', 'full']
        for backend in ('native', 'weave_functional'):
            args = base + ['--backend', backend]
            if backend == 'weave_functional':
                args += ['--kernel','tiled','--decode-implementation','hopper_fp8','--prefill-implementation','hopper_fp8']
            result = subprocess.run(args, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            argv = json.loads(result.stdout)['docker_argv']
            self.assertEqual(json.loads(argv[argv.index('--cuda-graph-config')+1]), canonical_graph_config())
            self.assertNotIn('--cuda-graph-max-bs-decode', argv)
            self.assertIn('--disable-cuda-graph-padding', argv)
            self.assertIn('WEAVE_GRAPH_C32=1', argv)
        for extra in (['--graph-evidence-port-base','30110'], ['--stall-diagnostic'], ['--decode-graph','disabled']):
            result = subprocess.run(base + extra, env=env, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
