import json
from pathlib import Path
import tempfile
import unittest

from scripts.run_c32_comparison import service_command, client_command, gate_command, export_public


class C32ComparisonLaunchTest(unittest.TestCase):
    def test_three_groups_share_eager_c32_and_only_staged_switches_decode(self):
        for variant in ('native','persistent_cached','staged_direct'):
            argv=service_command(variant,Path('/srv/work'),Path('/model'),variant,30001,30110,'a'*64)
            self.assertEqual(argv[argv.index('--decode-graph')+1],'disabled')
            self.assertIn('--request-timing',argv)
            self.assertNotIn('--execution-profile',argv)
            self.assertEqual(argv[argv.index('--decode-implementation')+1],
                             'staged_direct' if variant=='staged_direct' else 'persistent')

    def test_clients_never_use_gpus_and_gate_honors_scheduler_devices(self):
        argv=client_command('sha256:old',Path('/srv/work'),Path('/srv/data.jsonl'),
                            'c32-job/native','client-unique',30001,600,False)
        self.assertIn('--pull=never',argv)
        self.assertIn('NVIDIA_VISIBLE_DEVICES=void',argv)
        self.assertIn('/srv/data.jsonl:/dataset.jsonl:ro',argv)
        self.assertEqual(argv[argv.index('--concurrency')+1],'32')
        gate=gate_command('sha256:old',Path('/srv/work'),Path('/models/v4'),'c32-job','0,2,4,6')
        self.assertEqual(gate[gate.index('--gpus')+1],'"device=0,2,4,6"')
        self.assertIn('/models/v4:/model:ro',gate)
        self.assertIn('HF_HUB_OFFLINE=1',gate)

    def test_export_leaves_acceptance_sequences_and_logs_private(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp)/'run';run.mkdir();(run/'native').mkdir()
            (run/'experiment.json').write_text(json.dumps({'status':'failed'}))
            (run/'native/acceptance-private.json').write_text(json.dumps({'output_token_ids':[19]}))
            (run/'native/server.log').write_text('private generation')
            (run/'native/client-summary.json').write_text(json.dumps({'status':'passed','decode_tokens_per_second':1.}))
            export_public(run,Path(tmp)/'public')
            self.assertFalse((Path(tmp)/'public/native/acceptance-private.json').exists())
            self.assertFalse((Path(tmp)/'public/native/server.log').exists())
            self.assertTrue((Path(tmp)/'public/native/client-summary.json').exists())

    def test_hopper_suite_uses_actual_hopper_smoke_and_existing_offline_image(self):
        service=service_command('hopper_fp8',Path('/srv/work'),Path('/model'),'hopper',30001,30110,'a'*64)
        self.assertEqual(service[service.index('--decode-implementation')+1],'hopper_fp8')
        gate=gate_command('sha256:resolved',Path('/srv/work'),Path('/model'),'hopper','0,2,4,6','hopper')
        self.assertIn('/workspace/tests/gpu/check_hopper_serving.py',gate)
        self.assertNotIn('/workspace/tests/gpu/check_online_staged.py',gate)
        self.assertIn('--nproc-per-node=4',gate)
        self.assertIn('--pull=never',gate)
        self.assertIn('WEAVE_IMAGE_ID=sha256:resolved',gate)
        self.assertNotIn('--model-path',gate)

    def test_hopper_preview_selects_three_groups_and_replay_bundle(self):
        import contextlib
        import io
        from unittest.mock import patch
        from scripts.run_c32_comparison import main
        from src.weave.gpu_allocation import GPUAllocation
        output=io.StringIO()
        with patch('scripts.run_c32_comparison.allocated_gpus',return_value=GPUAllocation.parse('0,1,2,3')),contextlib.redirect_stdout(output):
            code=main(['--work-root','/srv/work','--dataset','/srv/data.jsonl','--comparison-suite','hopper','--hopper-evidence-dir','/srv/proof'])
        value=json.loads(output.getvalue())
        self.assertEqual(code,0)
        self.assertEqual(set(value['groups']),{'native','staged_direct','hopper_fp8'})
        self.assertEqual(value['generation_comparison_policy'],'record_only')
        self.assertIn('/workspace/tests/gpu/check_hopper_serving.py',value['online_gate'])

    def test_hopper_export_copies_proof_and_assembly_but_never_private_payloads(self):
        import shutil
        from tests.cpu.test_hopper_serving_evidence import BUNDLE
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp)/'run';run.mkdir();(run/'hopper_fp8').mkdir();(run/'online-gate').mkdir()
            (run/'experiment.json').write_text(json.dumps({'status':'comparison_complete','comparison_suite':'hopper'}))
            shutil.copytree(BUNDLE,run/'replay-proof')
            (run/'hopper_fp8/acceptance-private.json').write_text(json.dumps({'records':[{'output_ids':[19]}]}))
            (run/'hopper_fp8/client-summary.json').write_text(json.dumps({'status':'passed'}))
            (run/'online-gate/hopper-serving-rank0.json').write_text(json.dumps({'status':'passed'}))
            public=Path(tmp)/'public';manifest=export_public(run,public)
            self.assertEqual((public/'replay-proof/manifest.json').read_bytes(),(BUNDLE/'manifest.json').read_bytes())
            self.assertIn('hopper_fp8/client-summary.json',manifest['files'])
            self.assertIn('online-gate/hopper-serving-rank0.json',manifest['files'])
            self.assertFalse((public/'hopper_fp8/acceptance-private.json').exists())
