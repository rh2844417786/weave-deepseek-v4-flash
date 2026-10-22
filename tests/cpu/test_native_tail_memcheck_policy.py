"""API-report filtering must not turn sanitizer or application failures into passes."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from scripts import run_native_tail as controller


class MemcheckApiPolicyTests(unittest.TestCase):
    def args(self,root):
        return NS(work_root=root,model_path=Path('/model'),table_dir=root/'calibration',gpu_timeout=30)

    def test_only_memcheck_disables_api_reports_and_keeps_error_exit_code(self):
        root=Path('/srv/weave-work');args=self.args(root)
        command=controller.gpu_command('executor_memcheck',args,'image',root/'run'/'executor_memcheck')
        self.assertIn('--report-api-errors',command)
        self.assertEqual(command[command.index('--report-api-errors')+1],'no')
        self.assertEqual(command[command.index('--error-exitcode')+1],'99')
        for stage in ('compute','transport','executor','measured_validation'):
            self.assertNotIn('--report-api-errors',controller.gpu_command(stage,args,'image',root/'run'/stage))

    def run_stage(self,root,exit_code):
        child=NS(returncode=exit_code,poll=lambda:exit_code)
        owner=NS(spawn=Mock(return_value=child),stop=Mock())
        identity={'checkpoint_content_sha256':'a'*64}
        with patch.object(controller,'gpu_command',return_value=['docker','--name','test-memcheck']),patch.object(
                controller,'summarize_gpu_reports',return_value=dict(status='complete',ranks={})):
            return controller.run_gpu_stage('executor_memcheck',self.args(root),root,identity,
                'image',None,owner,lambda:None)

    def test_success_report_distinguishes_prior_counts_from_unobservable_current_count(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);result=self.run_stage(root,0)
            memory=result['memory_check']
            self.assertTrue(memory['verified'])
            self.assertEqual(memory['report_api_errors'],'no')
            self.assertTrue(memory['application_nonzero_exit_is_hard_failure'])
            current=memory['api_error_observations']['current_run']
            self.assertIsNone(current['filtered_count'])
            self.assertIsNone(current['categories'])
            prior=memory['api_error_observations']['historical_reference']
            self.assertEqual(prior['observed_count'],116)
            self.assertEqual(prior['run_name'],'native-tail-06ee57bda6e0')
            self.assertEqual({c['api'] for c in prior['categories']},
                             {'cuGetProcAddress_v2','cudaFuncGetAttributes'})
            self.assertFalse(prior['is_current_run_measurement'])
            controller.safe_tail_public(memory)
            self.assertEqual(json.loads((root/'executor_memcheck/check.json').read_text())['memory_check'],memory)

    def test_memory_error_and_own_api_exception_still_fail_and_preserve_policy(self):
        for code in (99,1):
            with self.subTest(exit_code=code),tempfile.TemporaryDirectory() as directory:
                root=Path(directory)
                with self.assertRaisesRegex(RuntimeError,'gpu_stage_failed'):self.run_stage(root,code)
                result=json.loads((root/'executor_memcheck/check.json').read_text())
                self.assertEqual(result['status'],'failed')
                memory=result['memory_check']
                self.assertFalse(memory['verified'])
                self.assertEqual(memory['actual_process_exit_code'],code)
                self.assertEqual(memory['report_api_errors'],'no')
