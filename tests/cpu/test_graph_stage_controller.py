import contextlib
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch


class StageControllerTests(unittest.TestCase):
    def test_diagnostic_preview_two_full_graph_groups_no_formal_claim(self):
        from scripts.run_gpqa_c32 import main
        out=io.StringIO()
        with contextlib.redirect_stdout(out),patch('scripts.run_gpqa_c32.subprocess.run',side_effect=AssertionError('offline preview')):
            self.assertEqual(main(['--work-root','/work','--records','/records','--prefill-evidence-dir','/proof',
                '--decode-graph','full','--graph-stage-profile']),0)
        value=json.loads(out.getvalue())
        self.assertEqual(value['measurement_mode'],'instrumented_diagnostic')
        self.assertFalse(value['formal_gpqa'])
        self.assertTrue(value['stage_event_gate_required'])
        self.assertEqual(value['seconds'],1200)
        for command in value['groups'].values():
            self.assertIn('--graph-stage-profile',command)
            self.assertEqual(command[command.index('--graph-stage-stride')+1],'128')
            self.assertEqual(command[command.index('--decode-graph')+1],'full')

    def test_profile_cannot_silently_run_eager(self):
        from scripts.run_gpqa_c32 import main
        with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
            main(['--work-root','/work','--records','/records','--prefill-evidence-dir','/proof','--graph-stage-profile'])

    def test_second_gpu_gate_has_distinct_owned_container_and_output(self):
        from scripts.run_graph_gate import graph_gate_command,stage_event_gate_command
        args=('sha256:image',Path('/work'),Path('/model'),'run-id','0,2,4,6')
        normal=graph_gate_command(*args);stage=stage_event_gate_command(*args)
        self.assertNotEqual(normal[normal.index('--name')+1],stage[stage.index('--name')+1])
        self.assertIn('/workspace/tests/gpu/check_graph_stage_events.py',stage)
        self.assertEqual([stage[i+1] for i,x in enumerate(stage) if x=='--output-dir'],['/work/run-id/stage-event-gate']*2)
        self.assertIn('--pull=never',stage)
