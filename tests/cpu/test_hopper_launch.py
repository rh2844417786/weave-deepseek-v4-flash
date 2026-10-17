import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.run_hopper_candidate import docker_command,export_public,measurement_start


class HopperLaunchTest(unittest.TestCase):
    def test_vendor_generator_rejects_wrong_archive_before_extracting(self):
        from scripts.vendor_hopper_headers import materialize
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);archive=root/'archive.tar.gz';archive.write_bytes(b'wrong archive')
            with self.assertRaisesRegex(ValueError,'SHA256'):materialize(archive,root/'vendor')
            self.assertFalse((root/'vendor').exists())

    def test_worker_freshness_accepts_controller_metadata_but_rejects_old_points(self):
        from tests.gpu.check_hopper_candidate import require_fresh_output
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'identity.json').write_text('{}');(root/'runtime.json').write_text('{}')
            require_fresh_output(root)
            (root/'hopper-rank0.json').write_text('{}')
            with self.assertRaises(ValueError):require_fresh_output(root)

    def test_offline_readonly_assets_and_dynamic_devices(self):
        argv=docker_command('sha256:old',Path('/srv/work'),Path('/models/v4'),Path('/srv/trace'),
                            'hopper-candidate-test','0,2,5,7',3,20,7200,'00000000-0000-4000-8000-000000000000')
        self.assertIn('--pull=never',argv)
        self.assertEqual(argv[argv.index('--gpus')+1],'"device=0,2,5,7"')
        self.assertIn('/models/v4:/model:ro',argv);self.assertIn('/srv/trace:/capture:ro',argv)
        self.assertIn('HF_HUB_OFFLINE=1',argv)
        self.assertIn('/workspace/scripts/hopper_candidate_child.py',argv)
        self.assertNotIn('pip',argv)

    def test_stale_progress_cannot_end_current_setup_budget(self):
        self.assertIsNone(measurement_start({'invocation_id':'old','measurement_started_monotonic':1.},'new'))
        self.assertEqual(measurement_start({'invocation_id':'new','measurement_started_monotonic':2.},'new'),2.)
        with self.assertRaises(ValueError):measurement_start({'invocation_id':'new','measurement_started_monotonic':float('nan')},'new')

    def test_export_retains_scalar_evidence_not_private_trace_or_full_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp)/'run';run.mkdir()
            (run/'experiment.json').write_text(json.dumps({'status':'failed'}))
            (run/'runtime.json').write_text(json.dumps({'runtime_status':'passed'}))
            (run/'rank0.pt').write_bytes(b'private tensor')
            (run/'build-and-replay-private.log').write_text('private values\nptxas info    : Used 128 registers\n')
            dest=Path(tmp)/'public';m=export_public(run,dest)
            self.assertFalse((dest/'rank0.pt').exists());self.assertFalse((dest/'build-and-replay-private.log').exists())
            self.assertEqual((dest/'compilation-resources.txt').read_text(),'ptxas info    : Used 128 registers\n')
            for name,h in m['files'].items():self.assertEqual(hashlib.sha256((dest/name).read_bytes()).hexdigest(),h)
