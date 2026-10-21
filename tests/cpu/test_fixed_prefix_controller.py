import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from scripts.run_fixed_prefix import main,export_public

class FixedControllerTests(unittest.TestCase):
    def test_preview_five_modes_and_boundaries(self):
        output=io.StringIO()
        with patch('scripts.run_fixed_prefix.subprocess.run',side_effect=AssertionError('execution inpreview')),contextlib.redirect_stdout(output):
            self.assertEqual(main(['--work-root','/work','--records','/data/main.jsonl']),0)
        d=json.loads(output.getvalue());self.assertEqual(d['steps'],8);self.assertEqual(d['decode_steps'],7);self.assertEqual(len(d['services']),5)
        for arm,cmd in d['services'].items():
            self.assertTrue(cmd[cmd.index('--run-name')+1].startswith('native-ep-'))
            self.assertEqual('--native-p0' in cmd,arm.startswith('p0_'))
            if arm.startswith('p0_'):self.assertIn('--native-p0-no-readback',cmd)
            self.assertIn('--fixed-prefix-dir',cmd)

    def test_public_scalar_records_export_and_raw_tokens_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run=root/'run';(run/'native_a/private').mkdir(parents=True)
            (run/'experiment.json').write_text('{"status":"diagnostic_complete"}')
            (run/'comparison.json').write_text('{"comparisons":[{"records":[{"step":0,"max_abs":123.0}]}]}')
            (run/'native_a/measure-rank0.json').write_text('{"records":[{"step":0,"input_ids_sha256":"commitment"}]}')
            (run/'native_a/private/teacher.json').write_text('{"output_ids":[1,2,3]}')
            m=export_public(run,root/'public');self.assertIn('comparison.json',m['files']);self.assertIn('native_a/measure-rank0.json',m['files']);self.assertNotIn('native_a/private/teacher.json',m['files'])
            (run/'comparison.json').write_text('{"records":[{"output_ids":[1,2,3]}]}')
            with self.assertRaises(ValueError):export_public(run,root/'bad')

    def test_waits_for_late_rank_proof_after_http_completion(self):
        from scripts.run_fixed_prefix import wait_phase_proofs
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            for rank in range(4):(folder/f'measure-rank{rank}.json').write_text(json.dumps({'status':'running' if rank==3 else 'passed'}))
            def finish(_): (folder/'measure-rank3.json').write_text('{"status":"passed"}')
            with patch('scripts.run_fixed_prefix.time.sleep',side_effect=finish),patch('scripts.run_fixed_prefix.validate_phase_proofs',return_value={'checked':True}) as validate:
                r=wait_phase_proofs(folder,phase='measure',steps=8,config={},teacher={},identity={},timeout=5,server=SimpleNamespace(poll=lambda:None),cancel_check=lambda:None)
            self.assertEqual(r,{'checked':True});validate.assert_called_once()

if __name__=='__main__':unittest.main()
