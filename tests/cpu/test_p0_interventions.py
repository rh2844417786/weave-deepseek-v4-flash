"""P0 intervention boundaries using CPU CUDA doubles, not GPU evidence."""
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

from src.weave.native_p0 import NativeP0Profiler
from src.weave import native_p0_hooks as hooks
from src.weave.native_p0_hooks import native_p0_options
from tests.cpu.test_graph_stage_profile import FakeCuda
from tests.cpu.test_native_p0_hooks import fixture

ROOT = Path(__file__).resolve().parents[2]


class UnreadableBuffer:
    def detach(self):
        raise AssertionError('readback is disabled')


class UnusedTrace:
    def begin(self):
        raise AssertionError('readback must not start a trace')


class InterventionTests(unittest.TestCase):
    def test_modes_capture_events_and_retain_buffers_independently(self):
        for mode, allocated, nodes, retained in (
                ('full', 1032, 86, 43), ('events', 1032, 86, 0), ('buffers', 0, 0, 43)):
            with self.subTest(mode=mode):
                cuda = FakeCuda()
                profiler = NativeP0Profiler(SimpleNamespace(cuda=cuda), 0,
                                            mode=mode, readback=False)
                refs = []
                for layer in range(43):
                    profiler.register_layer(layer, {'native_layer': layer})
                    cuda.capturing = True
                    with profiler.span(layer, 'moe'):
                        buffers = tuple(UnreadableBuffer() for _ in range(3))
                        refs.extend(weakref.ref(value) for value in buffers)
                        profiler.bind_grouping(layer, *buffers, 8, 256)
                    cuda.capturing = False
                del buffers
                gc.collect()
                self.assertEqual(len(cuda.events), allocated)
                self.assertEqual(len(cuda.operations), nodes)
                self.assertEqual(sum(ref() is not None for ref in refs), retained * 3)
                snapshot = profiler.snapshot()
                self.assertEqual(snapshot['intervention'], dict(
                    mode=mode, readback=False, event_node_count=nodes,
                    allocated_event_count=allocated,
                    retained_grouping_layer_count=retained,
                    retained_grouping_tensor_ref_count=retained * 3))
                self.assertEqual(len(snapshot['layer_metadata']), 43)
                for row in snapshot['layer_metadata'].values():
                    self.assertEqual(row['observed_span_calls'], {'moe': 1})
                    self.assertEqual(row['captured_spans'], [] if mode == 'buffers' else ['moe'])
                self.assertEqual(snapshot['external_events'], mode != 'buffers')

    def test_no_readback_counts_actual_b32_without_events_copies_or_traces(self):
        for mode in ('full', 'events', 'buffers'):
            with self.subTest(mode=mode):
                cuda = FakeCuda()
                profiler = NativeP0Profiler(SimpleNamespace(cuda=cuda), 0, stride=1,
                                            mode=mode, readback=False, trace=UnusedTrace())
                cuda.log.clear()
                for sequence, batch in enumerate((1, 32, 31, 32), 1):
                    self.assertFalse(profiler.begin_replay(sequence, batch))
                    self.assertIsNone(profiler.finish_replay(sequence, batch))
                self.assertEqual(cuda.log, [])
                snapshot = profiler.snapshot()
                self.assertEqual(snapshot['b32_seen'], 2)
                self.assertEqual(snapshot['sampled_total'], 0)
                self.assertEqual(snapshot['samples'], [])
                self.assertIsNone(snapshot['launch_trace'])
                with self.assertRaisesRegex(ValueError, 'sequence'):
                    profiler.begin_replay(4, 32)

    def test_invalid_mode_and_nonfull_readback_rejected_before_allocation(self):
        for options in ({'mode': 'invalid'}, {'mode': 'buffers'}, {'mode': 'events'},
                        {'readback': '0'}):
            cuda = FakeCuda()
            with self.assertRaises(ValueError):
                NativeP0Profiler(SimpleNamespace(cuda=cuda), 0, **options)
            self.assertEqual(cuda.events, [])

    def test_mode_and_readback_environment_require_explicit_p0(self):
        env = dict(WEAVE_NATIVE_P0='1', WEAVE_GRAPH_C32='1', WEAVE_REQUEST_TIMING='1',
                   WEAVE_NATIVE_P0_PRIVATE_DIR='/work/p0')
        for mode in ('full', 'events', 'buffers'):
            options = native_p0_options(dict(env, WEAVE_NATIVE_P0_MODE=mode,
                                            WEAVE_NATIVE_P0_READBACK='0'))
            self.assertEqual(options['mode'], mode)
            self.assertIs(options['readback'], False)
        for change in ({'WEAVE_NATIVE_P0_MODE': 'bad'},
                       {'WEAVE_NATIVE_P0_MODE': 'buffers'},
                       {'WEAVE_NATIVE_P0_MODE': 'events'},
                       {'WEAVE_NATIVE_P0_READBACK': 'false'}):
            with self.assertRaises(ValueError):
                native_p0_options(dict(env, **change))
        for key, value in (('WEAVE_NATIVE_P0_MODE', 'full'), ('WEAVE_NATIVE_P0_READBACK', '0')):
            with self.assertRaisesRegex(ValueError, 'require'):
                native_p0_options({key: value})
        with self.assertRaisesRegex(ValueError, 'readback'):
            native_p0_options(dict(env, WEAVE_FIXED_PREFIX='1'))

    def test_install_skips_launch_trace_resource_for_no_readback(self):
        # Source checking and CUDA allocation stay real except for unavailable
        # installed SGLang modules and device Event primitives on this CPU host.
        for mode in ('full', 'events', 'buffers'):
            model, _, modules, _ = fixture()
            model.model.layers *= 43
            cuda = FakeCuda()
            for module in modules.values():
                module.__file__ = str(ROOT / 'tests/cpu/test_native_p0_hooks.py')
            by_name = {hooks.MODULES[key]: value for key, value in modules.items()}
            package = SimpleNamespace(__file__=str(ROOT / '__init__.py'))
            with patch.dict(sys.modules, {'sglang': package}), \
                    patch.object(hooks, 'verify_native_p0_sources', return_value={
                        'tests/cpu/test_native_p0_hooks.py': 'fixture'}), \
                    patch('src.weave.native_p0_trace.LaunchTrace',
                          side_effect=AssertionError('no readback must not construct LaunchTrace')), \
                    patch.object(hooks.importlib, 'import_module', side_effect=by_name.__getitem__):
                profiler = hooks.install_native_p0(model, SimpleNamespace(cuda=cuda), 0,
                    private_dir='/unused', package_root=ROOT, mode=mode, readback=False)
            self.assertEqual(len(profiler.snapshot()['layer_metadata']), 43)
            self.assertIsNone(profiler.trace)
            self.assertEqual(len(cuda.events), 0 if mode == 'buffers' else 1032)


class InterventionLaunchTests(unittest.TestCase):
    def launch(self, *args):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='0,1,2,3',
                   WEAVE_MAX_RUNNING_REQUESTS='32', WEAVE_CUDA_GRAPH_MAX_BS_DECODE='32')
        return subprocess.run([sys.executable, str(ROOT / 'scripts/run_native_ep.py'),
                               '--work-root', '/tmp/weave-work', *args],
                              env=env, capture_output=True, text=True)

    def test_launcher_selects_modes_only_when_p0_enabled(self):
        base = ('--native-p0', '--graph-c32', '--decode-graph', 'full', '--request-timing')
        for mode in ('full', 'events', 'buffers'):
            run = self.launch(*base, '--native-p0-mode', mode, '--native-p0-no-readback')
            self.assertEqual(run.returncode, 0, run.stderr)
            argv = json.loads(run.stdout)['docker_argv']
            self.assertIn('WEAVE_NATIVE_P0_MODE=' + mode, argv)
            self.assertIn('WEAVE_NATIVE_P0_READBACK=0', argv)
        for args in (('--native-p0-mode', 'buffers'), ('--native-p0-no-readback',),
                     (*base, '--native-p0-mode', 'buffers'), (*base, '--native-p0-mode', 'events')):
            self.assertNotEqual(self.launch(*args).returncode, 0)

    def test_fixed_prefix_launch_maps_private_paths_without_reading_teacher(self):
        base = ('--graph-c32', '--decode-graph', 'full', '--request-timing')
        run = self.launch(*base, '--fixed-prefix-dir', '/tmp/weave-work/probe/native_a',
                          '--fixed-prefix-teacher', '/tmp/weave-work/probe/future-teacher.json')
        self.assertEqual(run.returncode, 0, run.stderr)
        argv = json.loads(run.stdout)['docker_argv']
        self.assertIn('WEAVE_FIXED_PREFIX=1', argv)
        self.assertIn('WEAVE_FIXED_PREFIX_DIR=/work/probe/native_a', argv)
        self.assertIn('WEAVE_FIXED_PREFIX_TEACHER=/work/probe/future-teacher.json', argv)
        self.assertIn('--enable-custom-logit-processor', argv)
        self.assertFalse(any('WEAVE_NATIVE_P0' in part for part in argv))
        seed = self.launch(*base, '--fixed-prefix-dir', '/tmp/weave-work/probe/native_a')
        self.assertEqual(seed.returncode, 0, seed.stderr)
        self.assertFalse(any('WEAVE_FIXED_PREFIX_TEACHER' in part
                             for part in json.loads(seed.stdout)['docker_argv']))

    def test_fixed_prefix_rejects_conflicts_and_paths_outside_work_root(self):
        base = ('--graph-c32', '--decode-graph', 'full', '--request-timing',
                '--fixed-prefix-dir', '/tmp/weave-work/probe/native_a')
        for args in (
                ('--fixed-prefix-teacher', '/tmp/weave-work/teacher.json'),
                ('--fixed-prefix-dir', '/tmp/weave-work/probe'),
                (*base, '--backend', 'weave_functional'),
                (*base, '--graph-stage-profile'), (*base, '--native-p0'),
                (*base, '--fixed-prefix-dir', '/tmp/outside'),
                (*base, '--fixed-prefix-teacher', '/tmp/outside.json'),
                (*base, '--fixed-prefix-teacher', 'relative.json')):
            run = self.launch(*args)
            self.assertNotEqual(run.returncode, 0, run.stdout)

    def test_launch_records_intervention_and_private_probe_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp).resolve()
            model = work / 'model'
            model.mkdir()
            docker = work / 'docker'
            docker.write_text('#!/usr/bin/env python3\nimport sys\n'
                'if sys.argv[1:3] == ["image", "inspect"]: print("sha256:fixture-image")\n'
                'elif sys.argv[1] == "run": sys.exit(0)\n')
            docker.chmod(0o755)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='0,1,2,3',
                       WEAVE_MAX_RUNNING_REQUESTS='32', WEAVE_CUDA_GRAPH_MAX_BS_DECODE='32',
                       PATH=str(work) + os.pathsep + os.environ['PATH'])
            probe = work / 'probe/p0_buffers'
            teacher = work / 'probe/teacher.json'
            run = subprocess.run([sys.executable, str(ROOT / 'scripts/run_native_ep.py'),
                '--work-root', str(work), '--model-path', str(model), '--execute',
                '--native-p0', '--native-p0-mode', 'buffers', '--native-p0-no-readback',
                '--graph-c32', '--decode-graph', 'full', '--request-timing',
                '--fixed-prefix-dir', str(probe), '--fixed-prefix-teacher', str(teacher)],
                env=env, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            paths = list(work.glob('native-ep-*/launch.json'))
            self.assertEqual(len(paths), 1)
            report = json.loads(paths[0].read_text())
            self.assertEqual(report['native_p0_configuration'], {'stride': 128, 'capacity': 64})
            self.assertEqual(report['native_p0_intervention'], {'mode': 'buffers', 'readback': False})
            self.assertTrue(report['fixed_prefix'])
            self.assertEqual(report['fixed_prefix_configuration'],
                             {'private_dir': str(probe), 'teacher_path': str(teacher)})
            self.assertEqual(report['numerical_policy'], 'record_only')
            self.assertFalse(report['used_for_formal_performance'])
            self.assertEqual(report['generation_status'], 'not_verified')


if __name__ == '__main__':
    unittest.main()
