import unittest
import json
from pathlib import Path
import tempfile
from src.weave.experiment import ExperimentLedger
from scripts.run_full_experiment import measuring_deadline


class FullExperimentTest(unittest.TestCase):
    def test_old_calibration_marker_cannot_expire_new_invocation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'progress.json'
            path.write_text(json.dumps({'invocation_id':'old','measuring_started_monotonic':1.,'budget_seconds':1200}))
            self.assertIsNone(measuring_deadline(path,'new'))
            path.write_text(json.dumps({'invocation_id':'new','measuring_started_monotonic':100.,'budget_seconds':1200}))
            self.assertEqual(measuring_deadline(path,'new'),1300.)
            path.write_text(json.dumps({'invocation_id':'new','measuring_started_monotonic':float('nan'),'budget_seconds':1200}))
            with self.assertRaises(ValueError): measuring_deadline(path,'new')

    def test_formal_groups_cannot_run_before_required_phases(self):
        ledger=ExperimentLedger()
        with self.assertRaises(ValueError): ledger.record('native_formal',{'status':'complete'})
        ledger.record('core',{'status':'passed'})
        with self.assertRaises(ValueError): ledger.record('calibration',{'status':'partial'})
        self.assertEqual(ledger.completed,['core'])
        ledger.record('calibration',{'status':'measured'})
        ledger.record('model_acceptance',{'status':'passed'})
        ledger.record('native_formal',{'status':'deadline'})
        ledger.record('weave_formal',{'status':'deadline'})
        ledger.record('comparison',{'status':'no_common_completed','relative_decode_speed':None})
        self.assertEqual(ledger.outcome,'no_common_completed')

    def test_identity_mismatch_and_empty_decode_work_cannot_claim_a_speed_result(self):
        for result in ({'status':'identity_mismatch'}, {'status':'compared','relative_decode_speed':None}):
            ledger=ExperimentLedger()
            for name,status in [('core','passed'),('calibration','measured'),('model_acceptance','passed'),
                                ('native_formal','complete'),('weave_formal','complete')]:
                ledger.record(name,{'status':status})
            if result['status']=='identity_mismatch':
                with self.assertRaises(ValueError): ledger.record('comparison',result)
            else:
                ledger.record('comparison',result)
                self.assertEqual(ledger.outcome,'no_decode_metric')
