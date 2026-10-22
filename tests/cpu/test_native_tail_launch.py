import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT=Path(__file__).resolve().parents[2]

class TailLaunchTests(unittest.TestCase):
    def launch(self,*extra):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='1,3,5,7',WEAVE_MAX_RUNNING_REQUESTS='32',WEAVE_CUDA_GRAPH_MAX_BS_DECODE='32')
        return subprocess.run([sys.executable,'scripts/run_native_ep.py','--backend','native',
            '--work-root','/tmp/weave-work','--decode-graph','full','--graph-c32','--request-timing',*extra],
            cwd=ROOT,env=env,capture_output=True,text=True)

    def test_tail_tables_and_separate_evidence_path_keep_original_graph_contract(self):
        result=self.launch('--native-tail','measured','--native-tail-directory','/tmp/weave-work/tables')
        self.assertEqual(result.returncode,0,result.stderr)
        command=json.loads(result.stdout)['docker_argv']
        self.assertIn('WEAVE_NATIVE_TAIL=measured',command)
        self.assertIn('WEAVE_NATIVE_TAIL_DIRECTORY=/work/tables',command)
        self.assertTrue(any(x.startswith('WEAVE_NATIVE_TAIL_EVIDENCE_DIR=/work/') for x in command))
        self.assertFalse(any(x.startswith('WEAVE_NATIVE_BUDGET=') for x in command))
        self.assertIn('--pull=never',command)
        graph=json.loads(command[command.index('--cuda-graph-config')+1])
        self.assertEqual(graph['decode'],dict(backend='full',max_bs=32,bs=list(range(1,33))))
        self.assertEqual(graph['prefill']['backend'],'disabled')
        self.assertIn('--disable-cuda-graph-padding',command)
        self.assertEqual(command[command.index('--watchdog-timeout')+1],'1800')

    def test_calibration_timeout_and_missing_or_stacked_methods(self):
        result=self.launch('--native-tail','calibration','--native-tail-directory','/tmp/weave-work/cal')
        self.assertEqual(result.returncode,0,result.stderr)
        command=json.loads(result.stdout)['docker_argv']
        self.assertEqual(command[command.index('--watchdog-timeout')+1],'43200')
        for flags in [('--native-tail','measured'),
                      ('--native-tail','static','--native-tail-directory','/outside'),
                      ('--native-tail','static','--native-tail-directory','/tmp/weave-work/cal','--native-budget','adaptive'),
                      ('--native-tail','static','--native-tail-directory','/tmp/weave-work/cal','--native-p0')]:
            with self.subTest(flags=flags):self.assertNotEqual(self.launch(*flags).returncode,0)
