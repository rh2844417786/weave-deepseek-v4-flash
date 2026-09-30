import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]


class ExportCoreEvidenceTest(unittest.TestCase):
    def test_existing_rank_reports_export_without_raw_service_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'cuda-functional.fixture';source.mkdir()
            output=Path(tmp)/'published'
            (source/'runtime.json').write_text(json.dumps({'validation_stage':'core','runtime_status':'passed',
                                                        'core_validation_status':'passed','child_exit_code':0}))
            for rank in range(4):
                (source/f'rank{rank}.json').write_text(json.dumps({'status':'functional_checks_passed',
                    'core_graph_replay':'passed','full_model':'not_run','performance':'not_run'}))
            (source/'exit_code.txt').write_text('0\n')
            (source/'server.log').write_text('raw generated answer')
            run=subprocess.run([sys.executable,'scripts/export_core_evidence.py','--run-dir',str(source),
                                '--output-dir',str(output)],cwd=ROOT,capture_output=True,text=True)
            self.assertEqual(run.returncode,0,run.stderr)
            manifest=json.loads((output/'evidence_manifest.json').read_text())
            self.assertEqual(manifest['reported_core_status'],'passed')
            self.assertEqual(manifest['scope'],'small_synthetic_core_checks')
            self.assertEqual((output/'rank3.json').read_bytes(),(source/'rank3.json').read_bytes())
            self.assertFalse((output/'server.log').exists())

    def test_missing_rank_cannot_be_exported_as_passed(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'cuda-functional.fixture';source.mkdir()
            output=Path(tmp)/'published'
            (source/'runtime.json').write_text(json.dumps({'validation_stage':'core','runtime_status':'passed',
                                                        'core_validation_status':'passed','child_exit_code':0}))
            run=subprocess.run([sys.executable,'scripts/export_core_evidence.py','--run-dir',str(source),
                                '--output-dir',str(output)],cwd=ROOT,capture_output=True,text=True)
            self.assertEqual(run.returncode,0,run.stderr)
            manifest=json.loads((output/'evidence_manifest.json').read_text())
            self.assertEqual(manifest['reported_core_status'],'incomplete')
            self.assertEqual(manifest['missing_rank_reports'],[0,1,2,3])


if __name__=='__main__': unittest.main()
