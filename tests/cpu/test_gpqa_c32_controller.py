import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.run_gpqa_c32 import client_command, export_public, main, measurement_deadline, validate_runtime, baseline_coverage


class GPQAControllerTests(unittest.TestCase):
    def test_client_offline_cpu_prepared_directory_and_fixed_contract(self):
        cmd=client_command('sha256:pinned',Path('/work'),Path('/prepared/records.jsonl'),'job/native','unique',30001)
        self.assertIn('/prepared:/prepared:ro',cmd)
        self.assertIn('NVIDIA_VISIBLE_DEVICES=void',cmd)
        self.assertIn('--network=host',cmd)
        self.assertIn('--pull=never',cmd)
        self.assertNotIn('--gpus',cmd)
        self.assertIn('--controller-handshake',cmd)
        self.assertNotIn('--seconds',cmd)
        self.assertNotIn('--concurrency',cmd)
        self.assertEqual(cmd[cmd.index('--records')+1],'/prepared/records.jsonl')

    def test_marker_deadline_is_fixed_and_immutable(self):
        marker={'started_monotonic':110.,'deadline_monotonic':1310.}
        self.assertEqual(measurement_deadline(marker,100.,111.,20.),1330.)
        for changed in ({**marker,'deadline_monotonic':1311.},{**marker,'started_monotonic':99.},{**marker,'deadline_monotonic':float('nan')}):
            with self.assertRaises(ValueError):measurement_deadline(changed,100.,111.,20.)

    def test_current_runtime_rejected_before_client(self):
        identity={'image_id':'sha256:pinned','source_sha256':'a'*64}
        good={'runtime_status':'passed','resolved_image_id':'sha256:pinned','kernel_source_sha256':'a'*64,'host_gpu_allocation':'0,2,4,6'}
        validate_runtime(good,identity,'0,2,4,6')
        with self.assertRaises(ValueError):validate_runtime({**good,'resolved_image_id':'sha256:other'},identity,'0,2,4,6')

    def test_preview_never_inspects_image_or_runs_gate(self):
        stream=io.StringIO()
        with patch('scripts.run_gpqa_c32.subprocess.run',side_effect=AssertionError('execution in preview')),contextlib.redirect_stdout(stream):
            self.assertEqual(main(['--work-root','/work','--records','/prepared/records.jsonl','--prefill-evidence-dir','/proof']),0)
        value=json.loads(stream.getvalue())
        self.assertEqual(set(value['groups']),{'native','hopper_prefill'})
        self.assertFalse(value['rerun_gpu_matrix'])
        self.assertEqual(value['seconds'],1200)
        self.assertNotIn('online_gate',value)

    def test_export_allowlist_raw_rejection_and_modes(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp)/'private';run.mkdir();(run/'native').mkdir()
            (run/'experiment.json').write_text('{"status":"failed"}')
            (run/'records.jsonl').write_text('private tokens')
            (run/'native/server-private.log').write_text('private text')
            (run/'native/gpqa-rows.json').write_text('{"rows":[{"row_index":3}]}')
            output=Path(tmp)/'public';manifest=export_public(run,output)
            self.assertEqual(set(manifest['files']),{'experiment.json','native/gpqa-rows.json'})
            self.assertEqual((output/'native/gpqa-rows.json').stat().st_mode & 0o777,0o664)
            (run/'native/gpqa-rows.json').write_text('{"rows":[{"rid":"private"}]}')
            with self.assertRaises(ValueError):export_public(run,Path(tmp)/'bad')

    def test_validation_failure_still_records_both_owned_cleanup_paths(self):
        from src.weave.gpu_allocation import GPUAllocation
        from unittest.mock import MagicMock
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            children=[MagicMock(),MagicMock()]
            stream=io.StringIO()
            with patch('scripts.run_gpqa_c32.OwnedChild',side_effect=children), patch('scripts.run_gpqa_c32.allocated_gpus',return_value=GPUAllocation.parse('0,1,2,3')), patch('src.weave.gpqa_c32.load_gpqa_workload',side_effect=ValueError('private failure payload')), contextlib.redirect_stdout(stream):
                result=main(['--work-root',str(root),'--records',str(root/'records.jsonl'),'--prefill-evidence-dir',str(root/'proof'),'--execute'])
            self.assertEqual(result,1)
            for child in children:child.stop.assert_called_once()
            result=json.loads(stream.getvalue())
            public=Path(result['export_directory'])
            report=json.loads((public/'experiment.json').read_text())
            self.assertEqual(report['status'],'failed')
            self.assertEqual(report['error_type'],'ValueError')
            self.assertNotIn('private failure payload',(public/'experiment.json').read_text())
            self.assertEqual((Path(result['work_directory'])/'controller-error-private.log').read_text(),'private failure payload\n')

    def test_baseline_coverage_shares_twenty_second_budget_across_ranks(self):
        clock=[100.]
        timeouts=[]
        def get(url,timeout):
            timeouts.append(timeout)
            clock[0]+=4.
            return {'rank':len(timeouts)-1}
        with patch('scripts.run_gpqa_c32.time.monotonic',side_effect=lambda:clock[0]), patch('scripts.run_gpqa_c32.http_json',side_effect=get):
            result=baseline_coverage(30110,deadline=120.)
        self.assertEqual(len(result),4)
        self.assertEqual(timeouts,[20.,16.,12.,8.])

    def test_baseline_coverage_rejects_expired_or_overrun_budget(self):
        with patch('scripts.run_gpqa_c32.time.monotonic',return_value=120.), patch('scripts.run_gpqa_c32.http_json') as get:
            with self.assertRaises(TimeoutError):baseline_coverage(30110,deadline=120.)
            get.assert_not_called()
        with patch('scripts.run_gpqa_c32.time.monotonic',side_effect=[100.,121.]), patch('scripts.run_gpqa_c32.http_json',return_value={} ) as get:
            with self.assertRaises(TimeoutError):baseline_coverage(30110,deadline=120.)
            self.assertEqual(get.call_count,1)

    def test_controller_accepts_marker_created_during_same_poll_iteration(self):
        from types import SimpleNamespace
        from unittest.mock import MagicMock
        from src.weave.gpu_allocation import GPUAllocation
        from src.weave.calibration import atomic_json as real_atomic
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);clock=[100.];state={}
            identity={'record_sha256':'a'*64}
            proof={'identity':{'source_sha256':'b'*64,'image_id':'sha256:pinned','checkpoint_content_sha256':'c'*64},'core_gate':{}}
            server=MagicMock();server.poll.return_value=None
            client=MagicMock();client.poll.side_effect=[None,0];client.returncode=0
            server_owner=MagicMock();client_owner=MagicMock()
            def spawn_server(command,lifecycle,**kwargs):
                if 'server_dir' in state:raise RuntimeError('stop_after_native')
                state['server_dir']=lifecycle.parent;lifecycle.parent.mkdir()
                real_atomic(lifecycle.parent/'runtime.json',{'runtime_status':'passed','resolved_image_id':'sha256:pinned','kernel_source_sha256':'b'*64,'host_gpu_allocation':'0,1,2,3'})
                (lifecycle.parent/'server.log').write_text('')
                return server
            def spawn_client(command,lifecycle,**kwargs):
                state['folder']=lifecycle.parent
                real_atomic(lifecycle.parent/'warmup-ready.json',{'ready_monotonic':101.})
                real_atomic(lifecycle.parent/'measurement-summary.json',{'workload':identity})
                return client
            def publish(path,value):
                real_atomic(path,value)
                if Path(path).name=='measurement-go.json':
                    clock[0]=103.
                    real_atomic(Path(path).parent/'load-window.json',{'started_monotonic':103.,'deadline_monotonic':1303.})
                elif Path(path).name=='group.json' and value.get('stage')=='measurement':clock[0]=104.
            server_owner.spawn.side_effect=spawn_server;client_owner.spawn.side_effect=spawn_client
            def now():
                # The client ready marker is published after the loop starts;
                # reading the clock during baseline capture observes that event.
                if 'folder' in state and clock[0]==100.:clock[0]=102.
                return clock[0]
            output=io.StringIO()
            with patch('scripts.run_gpqa_c32.OwnedChild',side_effect=[server_owner,client_owner]), patch('scripts.run_gpqa_c32.allocated_gpus',return_value=GPUAllocation.parse('0,1,2,3')), patch('scripts.run_gpqa_c32.subprocess.run',return_value=SimpleNamespace(stdout='sha256:pinned')), patch('src.weave.gpqa_c32.load_gpqa_workload',return_value=SimpleNamespace(identity=identity)), patch('src.weave.gpqa_c32.validate_prefill_bundle',return_value=proof), patch('src.weave.gpqa_c32.validate_client_summary'), patch('scripts.run_gpqa_c32.wait_ready'), patch('scripts.run_gpqa_c32.flush_cache_when_idle'), patch('scripts.run_gpqa_c32.assert_ports_free'), patch('scripts.run_gpqa_c32.scheduler_counts',return_value={'max_running_requests':32}), patch('scripts.run_gpqa_c32.atomic_json',side_effect=publish), patch('scripts.run_gpqa_c32.time.monotonic',side_effect=now), patch('scripts.run_gpqa_c32.time.sleep'), patch('scripts.run_gpqa_c32.export_public'), contextlib.redirect_stdout(output):
                main(['--work-root',str(root),'--records',str(root/'records.jsonl'),'--prefill-evidence-dir',str(root/'proof'),'--execute'])
            native=json.loads((state['folder']/'group.json').read_text())
            self.assertEqual(native['status'],'passed',native)
