import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

try:
    from scripts.run_compute_replay import docker_argv,export_replay
except ImportError:
    docker_argv=export_replay=None


class ComputeReplayLaunchTest(unittest.TestCase):
    def test_probe_semantic_change_invalidates_replay_resume_identity(self):
        import scripts.run_compute_replay as module
        names=('scripts/run_compute_replay.py','scripts/compute_replay_child.py','tests/gpu/replay_compute.py',
               'tests/gpu/check_staged.py','tests/gpu/check_checkpoint.py','src/weave/replay_benchmark.py',
               'src/weave/replay_trace.py','src/weave/staged_plan.py','src/weave/weights.py','src/weave/ops.py',
               'src/weave/execution_profile.py')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for name in names:
                p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('original')
            with patch.object(module,'ROOT',root):
                before=module.replay_source_sha256()
                (root/'src/weave/execution_profile.py').write_text('changed probe semantics')
                self.assertNotEqual(module.replay_source_sha256(),before)

    def test_export_accepts_exact_capture_descriptors_but_rejects_tensor_values(self):
        from src.weave.replay_trace import aggregate_manifests
        from tests.cpu.test_replay_trace import manifest
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run=root/'run';run.mkdir()
            (run/'experiment.json').write_text(json.dumps({'status':'partial'}))
            value=aggregate_manifests([manifest(rank) for rank in range(4)])
            path=run/'capture_manifest.json';path.write_text(json.dumps(value))
            export_replay(run,root/'good')
            self.assertTrue((root/'good/capture_manifest.json').is_file())
            value['ranks'][0]['tensors']['ids']['values']=[12,19]
            path.write_text(json.dumps(value))
            with self.assertRaises(ValueError):export_replay(run,root/'bad')
            self.assertFalse((root/'bad').exists())

    def test_resumed_controller_ignores_previous_invocation_deadline(self):
        import scripts.run_compute_replay as module
        self.assertTrue(hasattr(module,'measurement_start'))
        self.assertIsNone(module.measurement_start({'invocation_id':'old','measurement_started_monotonic':10.},'new'))
        self.assertEqual(module.measurement_start({'invocation_id':'new','measurement_started_monotonic':20.},'new'),20.)

    def test_replay_container_keeps_original_assets_readonly_and_allocated_devices(self):
        self.assertIsNotNone(docker_argv)
        argv=docker_argv('sha256:existing',Path('/srv/work'),Path('/models/v4'),Path('/srv/work/trace'),
                         'compute-replay-test','0,2,4,6',3,20,3,7200,120)
        self.assertEqual(argv[argv.index('--gpus')+1],'"device=0,2,4,6"')
        self.assertIn('--pull=never',argv)
        self.assertIn('/srv/work/trace:/trace:ro',argv)
        self.assertIn('/models/v4:/model:ro',argv)
        self.assertIn('HF_HUB_OFFLINE=1',argv)
        self.assertIn('GLOO_SOCKET_IFNAME=lo',argv)

    def test_export_omits_private_tensors_and_full_compile_log(self):
        self.assertIsNotNone(export_replay)
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp)/'run';run.mkdir()
            (run/'experiment.json').write_text(json.dumps({'status':'partial'}))
            (run/'replay-rank0.json').write_text(json.dumps({'status':'partial','points':[]}))
            (run/'rank0.pt').write_bytes(b'private tensors')
            (run/'build-and-replay.log').write_text('private exception detail\nptxas info    : Used 64 registers\n')
            output=Path(tmp)/'public';m=export_replay(run,output)
            self.assertFalse((output/'rank0.pt').exists())
            self.assertFalse((output/'build-and-replay.log').exists())
            self.assertEqual((output/'compilation-resources.txt').read_text(),'ptxas info    : Used 64 registers\n')
            self.assertIn('replay-rank0.json',m['files'])

    def test_cached_replay_exports_ptxas_from_its_core_build(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp)/'run';run.mkdir()
            core=Path(tmp)/'core';core.mkdir()
            (run/'experiment.json').write_text(json.dumps({'status':'partial','core_run_dir':str(core)}))
            (run/'build-and-replay.log').write_text('ninja: no work to do.\n')
            (core/'build-and-test.log').write_text('private log\nptxas info    : 0 bytes spill stores\n')
            (core/'rank0.json').write_text(json.dumps({'status':'functional_checks_passed'}))
            output=Path(tmp)/'public';m=export_replay(run,output)
            self.assertEqual((output/'core/compilation-resources.txt').read_text(),
                             'ptxas info    : 0 bytes spill stores\n')
            self.assertIn('core/rank0.json',m['files'])
            self.assertFalse((output/'core/build-and-test.log').exists())
