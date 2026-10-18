"""Graph ownership and parent timeout regressions; no CUDA required."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


class GraphCleanupTests(unittest.TestCase):
    def fixture(self, fail_reset=False):
        events, reports = [], []
        class Graph:
            live = True
            def reset(self):
                events.append('reset')
                if fail_reset: raise RuntimeError('reset failure')
                self.live = False
        graphs = [Graph(), Graph()]
        # Aliases deliberately stay live, reproducing main's dict and loop local.
        aliases = {0: graphs[0], 1: graphs[1]}
        group = object()
        def destroy(value=None):
            self.assertTrue(all(not g.live for g in aliases.values()))
            events.append('nccl' if value is group else 'gloo')
        torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: events.append('sync')))
        dist = SimpleNamespace(monitored_barrier=lambda **kw: events.append('barrier'),destroy_process_group=destroy)
        return graphs, group, torch, dist, events, reports

    def test_resets_all_graphs_before_communicator_even_with_live_aliases(self):
        from src.weave.graph_cleanup import cleanup_graphs
        graphs,group,torch,dist,events,reports=self.fixture()
        result=cleanup_graphs(graphs,torch,dist,group,lambda row:reports.append(copy.deepcopy(row)))
        self.assertEqual(events,['sync','reset','reset','sync','barrier','nccl','gloo'])
        self.assertEqual(result['status'],'complete')
        self.assertEqual(result['graphs_reset'],2)
        self.assertTrue(all(r['status']!='complete' for r in reports[:-1]))
        self.assertEqual(result['completed_phases'],['synchronize','reset_graphs','synchronize_after_reset','rank_barrier','destroy_nccl','destroy_gloo'])

    def test_failed_reset_records_failure_and_does_not_destroy_communicator(self):
        from src.weave.graph_cleanup import cleanup_graphs
        graphs,group,torch,dist,events,reports=self.fixture(fail_reset=True)
        with self.assertRaises(RuntimeError):
            cleanup_graphs(graphs,torch,dist,group,lambda row:reports.append(copy.deepcopy(row)))
        self.assertNotIn('nccl',events)
        self.assertEqual(reports[-1]['status'],'failed')
        self.assertEqual(reports[-1]['phase'],'reset_graphs')
        self.assertEqual(reports[-1]['error_type'],'RuntimeError')

    def test_parent_does_not_accept_passed_gpu_reports_while_child_is_alive(self):
        from scripts.run_gpqa_c32 import wait_graph_gate
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            child=subprocess.Popen([sys.executable,'-c',
                'import json,pathlib,sys,time; p=pathlib.Path(sys.argv[1]); '
                '(p/"hopper-graph-rank0.json").write_text(json.dumps({"status":"passed"})); '
                '(p/"graph-cleanup-rank0.json").write_text(json.dumps({"started_monotonic":time.monotonic(),"status":"running","phase":"destroy_nccl"})); '
                'time.sleep(60)',str(root)])
            try:
                with self.assertRaisesRegex(TimeoutError,'graph_cleanup_deadline'):
                    wait_graph_gate(child,root,gate_timeout=10.,cleanup_timeout=.2)
                self.assertIsNone(child.poll())
            finally:
                child.kill();child.wait(timeout=5)

    def test_controller_exports_cleanup_timeout_without_starting_services(self):
        import contextlib
        import io
        from unittest.mock import MagicMock
        from scripts import run_gpqa_c32 as controller
        from src.weave.gpu_allocation import GPUAllocation
        with tempfile.TemporaryDirectory() as tmp, contextlib.ExitStack() as stack:
            root=Path(tmp);owners=[MagicMock(),MagicMock()];output=io.StringIO()
            stack.enter_context(patch.object(controller,'OwnedChild',side_effect=owners))
            stack.enter_context(patch.object(controller,'allocated_gpus',return_value=GPUAllocation.parse('0,2,4,6')))
            stack.enter_context(patch('src.weave.gpqa_c32.load_gpqa_workload',return_value=SimpleNamespace(identity={})))
            stack.enter_context(patch('src.weave.gpqa_c32.validate_prefill_bundle',return_value={'identity':{},'core_gate':{}}))
            stack.enter_context(patch('src.weave.gpqa_c32.copy_verified_prefill_bundle'))
            stack.enter_context(patch.object(controller.subprocess,'run',return_value=SimpleNamespace(stdout='sha256:image')))
            def timeout(child,folder,**kwargs):
                self.assertEqual(kwargs['cleanup_timeout'],120.)
                kwargs['cancel_check']()
                (folder/'graph-cleanup-rank0.json').write_text(json.dumps({'status':'running','phase':'destroy_nccl','started_monotonic':1.}))
                raise TimeoutError('graph_cleanup_deadline')
            wait=stack.enter_context(patch.object(controller,'wait_graph_gate',side_effect=timeout))
            stack.enter_context(contextlib.redirect_stdout(output))
            code=controller.main(['--work-root',str(root),'--records',str(root/'records'),
                '--prefill-evidence-dir',str(root/'proof'),'--decode-graph','full','--execute'])
            self.assertEqual(code,1);wait.assert_called_once()
            result=json.loads(output.getvalue());public=Path(result['export_directory'])
            report=json.loads((public/'experiment.json').read_text())
            self.assertEqual(report['stage'],'graph_gate')
            self.assertEqual(report['error_type'],'TimeoutError')
            self.assertEqual(report['groups'],{})
            self.assertTrue((public/'graph-gate/graph-cleanup-rank0.json').is_file())
            self.assertEqual(owners[0].spawn.call_count,1)
            for owner in owners:owner.stop.assert_called_once()

    def test_parent_requires_successful_process_exit(self):
        from scripts.run_gpqa_c32 import wait_graph_gate
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError,'graph_gpu_gate_failed'):
                wait_graph_gate(SimpleNamespace(poll=lambda:1,returncode=1),Path(tmp),gate_timeout=1,cleanup_timeout=1)
            wait_graph_gate(SimpleNamespace(poll=lambda:0,returncode=0),Path(tmp),gate_timeout=1,cleanup_timeout=1)


if __name__=='__main__':unittest.main()
