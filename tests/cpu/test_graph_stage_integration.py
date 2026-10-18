"""Offline tests of opt-in service hooks and actual replay accounting."""
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from src.weave.graph_hooks import patch_backend, setup_runner
from src.weave.graph_observer import BATCHES, GraphObserver, canonical_graph_config, graph_stage_options
from src.weave.staged_decode import ModelBatchCounters


class FakeProfiler:
    def __init__(self, observer, events):
        self.observer, self.events = observer, events
        self.ordinal = 0
        self.finish_error = False

    def begin_replay(self, sequence, batch):
        assert self.observer.active
        self.events.append(('begin', sequence, batch))
        if batch == 32:
            self.ordinal += 1
            return self.ordinal % 2 == 0
        return False

    def finish_replay(self, sequence, batch):
        assert self.observer.active
        self.events.append(('finish', sequence, batch))
        if self.finish_error:
            raise RuntimeError('event timing failed')

    def abort_replay(self, sequence, batch):
        self.events.append(('abort', sequence, batch))

    def snapshot(self):
        assert self.observer.lock._is_owned()
        return {'samples': [], 'measurement_mode': 'instrumented_diagnostic'}


class GraphStageHooksTest(unittest.TestCase):
    def fixture(self, enabled=True):
        events = []
        observer = GraphObserver(0, {}, 'source')
        if enabled:
            observer.stage_profiler = FakeProfiler(observer, events)
        observer.captured.update(BATCHES)
        counters = ModelBatchCounters()
        model = NS(weave_runtime=NS(model_batch_counters=counters))
        runner = NS(raw_bs=32, model_runner=NS(model=model))

        class Backend:
            def __init__(self, runner): pass
            def capture_one(self, key): events.append(('capture', key.size))
            def replay(self, key, batch):
                events.append(('replay', key.size))
                if getattr(batch, 'fail', False):
                    raise RuntimeError('graph failed')
                return 'output'

        patch_backend(Backend, lambda _: observer)
        return Backend(runner), observer, runner, counters, events

    def test_sampler_boundaries_sequence_and_default(self):
        backend, observer, runner, counters, events = self.fixture()
        backend.capture_one(NS(size=32, stream_idx=None, variant_label=None))
        backend.replay(NS(size=32), NS())
        runner.raw_bs = 1
        backend.replay(NS(size=1), NS())
        runner.raw_bs = 32
        self.assertEqual(backend.replay(NS(size=32), NS()), 'output')
        self.assertEqual(events, [('capture', 32), ('begin', 1, 32), ('replay', 32),
                                 ('begin', 2, 1), ('replay', 1), ('begin', 3, 32),
                                 ('replay', 32), ('finish', 3, 32)])
        snapshot = observer.snapshot()
        self.assertEqual(snapshot['successful_replays'], 3)
        self.assertEqual(snapshot['raw_batch_histogram'], {'32': 2, '1': 1})
        self.assertIn('stage_profile', snapshot)
        with observer.operation():
            with self.assertRaises(ValueError): observer.snapshot()
        backend, observer, runner, counters, events = self.fixture(enabled=False)
        backend.replay(NS(size=32), NS())
        self.assertEqual(events, [('replay', 32)])
        self.assertNotIn('stage_profile', observer.snapshot())

    def test_graph_failure_aborts_sample_without_completed_replay(self):
        backend, observer, runner, counters, events = self.fixture()
        backend.replay(NS(size=32), NS())
        with self.assertRaisesRegex(RuntimeError, 'graph failed'):
            backend.replay(NS(size=32), NS(fail=True))
        self.assertEqual(events[-3:], [('begin', 2, 32), ('replay', 32), ('abort', 2, 32)])
        snap = observer.snapshot()
        self.assertEqual((snap['successful_replays'], snap['failed_replays']), (1, 1))
        self.assertEqual(counters.snapshot()['failed_forwards'], 1)
        self.assertFalse(observer.active)

    def test_timing_failure_preserves_actual_replay_coverage(self):
        backend, observer, runner, counters, events = self.fixture()
        backend.replay(NS(size=32), NS())
        observer.stage_profiler.finish_error = True
        with self.assertRaisesRegex(RuntimeError, 'event timing failed'):
            backend.replay(NS(size=32), NS())
        snap = observer.snapshot()
        self.assertEqual((snap['successful_replays'], snap['failed_replays']), (2, 0))
        self.assertEqual(snap['raw_batch_histogram'], {'32': 2})
        self.assertEqual(counters.snapshot()['failed_forwards'], 0)
        self.assertEqual(events[-1], ('abort', 2, 32))

    def test_setup_attaches_shared_profiler_before_model_warmup(self):
        import src.weave.graph_hooks as hooks
        env = {'WEAVE_GRAPH_C32': '1', 'WEAVE_REQUEST_TIMING': '1', 'WEAVE_GRAPH_STAGE_PROFILE': '1'}
        config = canonical_graph_config()
        args = NS(cuda_graph_config=NS(**{k: NS(**v) for k, v in config.items()}),
                  max_running_requests=32, disable_cuda_graph_padding=True)
        parallel = NS(tp_size=4, moe_ep_size=4, moe_tp_size=1, attn_dp_size=1)
        fake_modules = {'sglang': NS(__file__='/unused/sglang/__init__.py'),
                        'torch': NS(cuda=NS(is_current_stream_capturing=lambda: False)),
                        'sglang.srt.runtime_context': NS(get_parallel=lambda: parallel)}
        for enabled, kind in ((False, 'native'), (True, 'native'), (True, 'weave')):
            model = NS(forward=lambda: None)
            runner = NS(capture_bs=BATCHES, captured_req_width=1,
                        model_runner=NS(model=model, server_args=args, tp_group=NS(rank=2)))
            case_env = dict(env)
            if not enabled: case_env.pop('WEAVE_GRAPH_STAGE_PROFILE')
            if kind == 'weave': case_env['SGLANG_EXTERNAL_MODEL_PACKAGE'] = 'weave.sglang_ext.models'
            constructed = []
            def construct(torch, rank, selected_kind, **options):
                constructed.append((rank, selected_kind, options))
                return NS(kind=selected_kind)
            with patch.dict(os.environ, case_env, clear=True), patch.dict(sys.modules, {
                    **fake_modules, 'src.weave.graph_stage_profile': NS(GraphStageProfiler=construct)}), \
                    patch.object(hooks, 'verify_graph_sources', return_value={}), \
                    patch.object(hooks, 'graph_observer_source_sha256', return_value='source'), \
                    patch.object(hooks, 'start_endpoint', return_value='server'):
                observer = setup_runner(runner)
            if enabled:
                self.assertEqual(constructed, [(2, kind, {'stride': 128, 'capacity': 64})])
                self.assertIs(observer.stage_profiler, model.weave_graph_stage_profiler)
            else:
                self.assertEqual(constructed, [])
                self.assertFalse(hasattr(model, 'weave_graph_stage_profiler'))

    def test_bound_layer_registration_uses_each_installed_handle(self):
        from tests.cpu.test_hopper_adapter import HopperAdapterTest
        module, expert, _, _ = HopperAdapterTest().expert()
        registered = []
        expert.runtime.graph_stage_profile = True
        expert.runtime.hopper_decode.register_layer = lambda layer, bound: registered.append((layer, bound))
        native = NS(quant_method=object(), moe_runner_config=NS(inplace=True),
                    should_fuse_routed_scaling_factor_in_topk=False)
        replacements = [module.RoutedExperts(native, expert.runtime, expert.raw_weights, None,
                                             (1, 1, 0), 10, layer) for layer in range(43)]
        self.assertEqual([layer for layer, _ in registered], list(range(43)))
        for (_, bound), replacement in zip(registered, replacements):
            self.assertIs(bound, replacement.hopper_bound_weights)


class GraphStageLaunchTest(unittest.TestCase):
    def test_environment_guards(self):
        env = {'WEAVE_GRAPH_C32': '1', 'WEAVE_REQUEST_TIMING': '1', 'WEAVE_GRAPH_STAGE_PROFILE': '1'}
        self.assertIsNone(graph_stage_options({}))
        self.assertEqual(graph_stage_options(env), {'stride': 128, 'capacity': 64})
        for change in ({'WEAVE_GRAPH_STAGE_PROFILE': 'yes'}, {'WEAVE_GRAPH_C32': '0'},
                       {'WEAVE_REQUEST_TIMING': '0'}, {'WEAVE_EXECUTION_PROFILE': '1'},
                       {'WEAVE_PROFILE': '/bank'}, {'WEAVE_REPLAY_CAPTURE_DIR': '/capture'},
                       {'WEAVE_GRAPH_STAGE_STRIDE': '0'}, {'WEAVE_GRAPH_STAGE_STRIDE': '65537'},
                       {'WEAVE_GRAPH_STAGE_CAPACITY': '257'}, {'WEAVE_GRAPH_STAGE_CAPACITY': 'x'}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                graph_stage_options({**env, **change})

    def launch(self, *extra):
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '0,1,2,3',
               'WEAVE_MAX_RUNNING_REQUESTS': '32', 'WEAVE_CUDA_GRAPH_MAX_BS_DECODE': '32'}
        return subprocess.run([sys.executable, 'scripts/run_native_ep.py', *extra],
                              cwd=Path(__file__).resolve().parents[2], env=env,
                              capture_output=True, text=True)

    def test_both_groups_opt_in_and_default_has_no_stage_environment(self):
        base = ['--graph-c32', '--request-timing', '--decode-graph', 'full']
        for backend in ('native', 'weave_functional'):
            args = base + ['--backend', backend]
            if backend == 'weave_functional':
                args += ['--kernel', 'tiled', '--decode-implementation', 'hopper_fp8',
                         '--prefill-implementation', 'hopper_fp8']
            for enabled in (False, True):
                result = self.launch(*args, *(['--graph-stage-profile', '--graph-stage-stride', '8',
                                             '--graph-stage-capacity', '4'] if enabled else []))
                self.assertEqual(result.returncode, 0, result.stderr)
                command = json.loads(result.stdout)['docker_argv']
                self.assertEqual('WEAVE_GRAPH_STAGE_PROFILE=1' in command, enabled)
                if enabled:
                    self.assertIn('WEAVE_GRAPH_STAGE_STRIDE=8', command)
                    self.assertIn('WEAVE_GRAPH_STAGE_CAPACITY=4', command)

    def test_invalid_launch_contracts_rejected_before_docker(self):
        base = ['--graph-c32', '--request-timing', '--decode-graph', 'full', '--graph-stage-profile']
        for extras in (['--stall-diagnostic'], ['--execution-profile'], ['--profile', '/bank'],
                       ['--decode-graph', 'disabled'], ['--graph-stage-stride', '0'],
                       ['--graph-stage-capacity', '257']):
            result = self.launch(*base, *extras)
            self.assertNotEqual(result.returncode, 0)
            self.assertNotIn('docker_argv', result.stdout)
        self.assertNotEqual(self.launch('--graph-stage-profile').returncode, 0)
        self.assertNotEqual(self.launch('--graph-stage-stride', '8').returncode, 0)


if __name__ == '__main__':
    unittest.main()
