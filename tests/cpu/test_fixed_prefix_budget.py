import json
import os
import subprocess
import sys
import tempfile
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace
import unittest
from src.weave.fixed_prefix_budget import admission_budget, validate_effective_budget

ROOT=Path(__file__).resolve().parents[2]
LENGTHS=[r['input_tokens'] for r in json.loads((ROOT/'reports/server/fixed-prefix-d21fe5a23861/config.json').read_text())['requests']]

class FixedPrefixBudgetTests(unittest.TestCase):
    def test_real_27_plus_5_failure_and_new_whole_cohort_budget(self):
        old=admission_budget(LENGTHS,page_size=256,chunked_prefill_size=8192,max_prefill_tokens=16384)
        self.assertEqual(old['raw_input_tokens'],6059)
        self.assertEqual(old['page_rounded_input_tokens'],9472)
        self.assertEqual(old['rows_within_prefill_token_budget'],27)
        self.assertEqual(old['page_rounded_prefix_tokens'][26],8192)
        self.assertFalse(old['whole_cohort_fits_token_budget'])
        new=admission_budget(LENGTHS,page_size=256,chunked_prefill_size=16384,max_prefill_tokens=16384)
        self.assertEqual(new['rows_within_prefill_token_budget'],32)
        self.assertTrue(new['whole_cohort_fits_token_budget'])

    def test_effective_server_limits_checked_not_only_command_flags(self):
        good=dict(page_size=256,chunked_prefill_size=16384,max_prefill_tokens=16384,max_running_requests=32,prefill_max_requests=None,internal_states=[{'effective_max_running_requests_per_dp':32}])
        proof=validate_effective_budget(LENGTHS,good)
        self.assertEqual(proof['status'],'passed');self.assertEqual(proof['page_size'],256)
        for update in [dict(chunked_prefill_size=8192),dict(max_prefill_tokens=8192),dict(page_size=128),dict(prefill_max_requests=27),dict(max_running_requests=27),dict(internal_states=[]),dict(internal_states=[{}]),dict(internal_states=[{'effective_max_running_requests_per_dp':27}])]:
            with self.subTest(update=update),self.assertRaises(ValueError):validate_effective_budget(LENGTHS,{**good,**update})
        self.assertEqual(proof['effective_max_running_requests_per_dp'],[32])
        for key in ('page_size','chunked_prefill_size','max_prefill_tokens','max_running_requests','prefill_max_requests','internal_states'):
            bad=dict(good);bad.pop(key)
            with self.assertRaises(ValueError):validate_effective_budget(LENGTHS,bad)

    def test_invalid_lengths_do_not_produce_budget_proof(self):
        for lengths in ([],LENGTHS[:-1],[0]*32,[True]*32,[1.5]*32):
            with self.assertRaises(ValueError):admission_budget(lengths,page_size=256,chunked_prefill_size=16384,max_prefill_tokens=16384)

    def test_live_failure_is_saved_without_unrelated_server_state(self):
        from scripts.run_fixed_prefix import check_prefill_budget
        config={'requests':[{'input_tokens':n} for n in LENGTHS]}
        response=dict(page_size=256,chunked_prefill_size=8192,max_prefill_tokens=16384,max_running_requests=32,prefill_max_requests=None,internal_states=[{'effective_max_running_requests_per_dp':32,'text':'private'}])
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'budget.json'
            with patch('scripts.run_fixed_prefix.http_json',return_value=response),self.assertRaises(ValueError):
                check_prefill_budget('http://127.0.0.1:30001',config,output)
            raw=output.read_text();report=json.loads(raw)
            self.assertEqual(report['status'],'failed');self.assertEqual(report['effective_limits']['chunked_prefill_size'],8192)
            self.assertNotIn('private',raw);self.assertNotIn('internal_states',raw)
            response['chunked_prefill_size']=16384
            with patch('scripts.run_fixed_prefix.http_json',return_value=response):check_prefill_budget('http://127.0.0.1:30001',config,output)
            self.assertEqual(json.loads(output.read_text())['page_rounded_input_tokens'],9472)

    def test_real_launch_previews_scope_16k_to_fixed_prefix_only(self):
        from scripts.run_fixed_prefix import service_command
        from src.weave.fixed_prefix_evidence import ARMS,normalize_fixed_launch
        env={**os.environ,'CUDA_VISIBLE_DEVICES':'0,2,4,6','WEAVE_MAX_RUNNING_REQUESTS':'32','WEAVE_CUDA_GRAPH_MAX_BS_DECODE':'32'}
        launches=[]
        for arm in ARMS:
            command=service_command(arm,Path('/tmp/fixed-budget'),Path('/model'),'budget-test',30001,30310,'a'*64,Path('/tmp/fixed-budget')/arm,Path('/tmp/fixed-budget/teacher.json'))
            command.remove('--execute');run=subprocess.run(command,env=env,capture_output=True,text=True)
            self.assertEqual(run.returncode,0,run.stderr);argv=json.loads(run.stdout)['docker_argv']
            for flag in ('--chunked-prefill-size','--max-prefill-tokens'):
                self.assertEqual(argv.count(flag),1);self.assertEqual(argv[argv.index(flag)+1],'16384')
            launches.append(normalize_fixed_launch(argv))
        self.assertTrue(all(argv==launches[0] for argv in launches))
        run=subprocess.run([sys.executable,str(ROOT/'scripts/run_native_ep.py'),'--work-root','/tmp/fixed-budget','--backend','native','--graph-c32','--decode-graph','full','--request-timing'],env=env,capture_output=True,text=True)
        self.assertEqual(run.returncode,0,run.stderr);argv=json.loads(run.stdout)['docker_argv']
        self.assertEqual(argv[argv.index('--chunked-prefill-size')+1],'8192');self.assertNotIn('--max-prefill-tokens',argv)

if __name__=='__main__':unittest.main()
