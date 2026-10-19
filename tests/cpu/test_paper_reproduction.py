import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


class PaperControllerTests(unittest.TestCase):
    def test_paper_preview_cannot_launch_old_operator_as_paper(self):
        from scripts.run_gpqa_c32 import main
        out=io.StringIO()
        with contextlib.redirect_stdout(out),patch('scripts.run_gpqa_c32.subprocess.run',side_effect=AssertionError('preview offline')):
            self.assertEqual(main(['--work-root','/work','--records','/data','--prefill-evidence-dir','/prior',
                '--paper-proof-dir','/work/paper','--decode-graph','full']),0)
        data=json.loads(out.getvalue());self.assertEqual(set(data['groups']),{'native','paper_persistent'})
        self.assertFalse(data['complete_weave_claim']);self.assertFalse(data['graph_gate_required'])
        cmd=data['groups']['paper_persistent']
        for flag in ('--decode-implementation','--prefill-implementation'):
            self.assertEqual(cmd[cmd.index(flag)+1],'hopper_persistent')
        self.assertEqual(cmd[cmd.index('--profile')+1],'/work/paper/calibrate')
        self.assertEqual(data['seconds'],1200);self.assertEqual(data['concurrency'],32)

    def test_paper_rejects_eager_or_operator_profiler(self):
        from scripts.run_gpqa_c32 import main
        base=['--work-root','/work','--records','/data','--prefill-evidence-dir','/prior','--paper-proof-dir','/work/paper']
        for extra in ([],['--decode-graph','full','--graph-stage-profile']):
            with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):main(base+extra)

    def test_owned_offline_stage_commands_and_dynamic_gpu_assignment(self):
        from scripts.run_paper_reproduction import stage_command
        names=[]
        for stage in ('smoke','calibrate','checkpoint'):
            cmd=stage_command('sha256:image',Path('/work'),Path('/model'),'run','0,2,4,6',stage,'c'*64)
            names.append(cmd[cmd.index('--name')+1]);self.assertIn('--pull=never',cmd)
            self.assertIn('"device=0,2,4,6"',cmd)
            self.assertEqual([cmd[i+1] for i,v in enumerate(cmd) if v=='--output-dir'],[f'/work/run/{stage}']*2)
            self.assertIn('/workspace/tests/gpu/check_hopper_persistent.py',cmd)
            if stage=='checkpoint':self.assertEqual(cmd[cmd.index('--profile-dir')+1],'/work/run/calibrate')
        self.assertEqual(len(set(names)),3)

    def test_public_export_never_copies_private_gpu_log(self):
        from scripts.run_paper_reproduction import export
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'run';root.mkdir();(root/'smoke').mkdir()
            (root/'paper-experiment.json').write_text('{"status":"failed"}')
            (root/'smoke/private.log').write_text('private activations')
            (root/'smoke/paper-smoke-rank0.json').write_text('{"status":"failed","numerical_policy":"record_only"}')
            dst=Path(tmp)/'public';export(root,dst)
            self.assertFalse((dst/'smoke/private.log').exists())
            self.assertEqual((dst/'smoke/paper-smoke-rank0.json').stat().st_mode&0o777,0o664)
            self.assertEqual(len(json.loads((dst/'manifest.json').read_text())['files']),2)


if __name__=='__main__':unittest.main()
