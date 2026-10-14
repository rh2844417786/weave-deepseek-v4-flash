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
