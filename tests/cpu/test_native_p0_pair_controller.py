import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from scripts.run_native_p0_pair import main, export_public, client_command, load_deadline

class PairControllerTests(unittest.TestCase):
    def test_preview_two_native_graph_arms_and_same_client(self):
        output=io.StringIO()
        with patch('scripts.run_native_p0_pair.subprocess.run',side_effect=AssertionError('preview executed')),contextlib.redirect_stdout(output):
            self.assertEqual(main(['--work-root','/w','--records','/data/main.jsonl']),0)
        d=json.loads(output.getvalue());self.assertEqual(d['order'],['native','p0']);self.assertEqual(d['request_count'],32)
        self.assertFalse(d['formal_performance']);self.assertEqual(d['completion_policy'],'natural_EOS_or_32768_per_request')
        for key,cmd in d['services'].items():
            self.assertEqual(cmd[cmd.index('--backend')+1],'native');self.assertEqual(cmd[cmd.index('--decode-graph')+1],'full')
            self.assertEqual('--native-p0' in cmd,key=='p0')
            self.assertTrue(cmd[cmd.index('--run-name')+1].startswith('native-ep-'))
        self.assertNotIn('--seconds',d['clients']['native'])
        self.assertIn('/workspace/scripts/bench_native_p0_pair.py',d['clients']['native'])

    def test_client_offline_and_deadline(self):
        cmd=client_command('image',Path('/w'),Path('/data/main.jsonl'),'run/native','job',30001,3600)
        self.assertIn('--pull=never',cmd);self.assertIn('NVIDIA_VISIBLE_DEVICES=void',cmd);self.assertIn('--controller-handshake',cmd)
        self.assertEqual(load_deadline({'started_monotonic':20.,'deadline_monotonic':3620.},10.,21.,3600.,60.),3710.)
        with self.assertRaises(ValueError):load_deadline({'started_monotonic':20.,'deadline_monotonic':1220.},10.,21.,3600.,60.)

    def test_export_does_not_copy_private_or_raw_trace(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run=root/'run';run.mkdir();(run/'native').mkdir()
            (run/'experiment.json').write_text('{"status":"paired_check_complete"}')
            (run/'native/paired-client-summary.json').write_text('{"rows":[]}')
            (run/'native/client-private.log').write_text('private')
            d=export_public(run,root/'public');self.assertEqual(set(d['files']),{'experiment.json','native/paired-client-summary.json'})
            (run/'native/paired-client-summary.json').write_text('{"text":"private"}')
            with self.assertRaises(ValueError):export_public(run,root/'bad')

    def test_complete_two_arm_flow_keeps_content_mismatch_successful(self):
        import time
        from types import SimpleNamespace
        from contextlib import ExitStack
        from tests.cpu.test_native_p0_pair import clients
        from src.weave.ops import kernel_source_sha256
        from src.weave.gpu_allocation import GPUAllocation
        a,b=clients();workload=SimpleNamespace(identity=a['source_workload_identity'])
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);results=iter([a,b]);owners=[]
            class Process:
                returncode=0
                def __init__(self,folder=None):self.folder=folder;self.first=True
                def poll(self):
                    if self.folder is None:return None
                    if self.first:
                        self.first=False;now=time.monotonic()
                        (self.folder/'warmup-ready.json').write_text(json.dumps({'ready_monotonic':now}))
                        return None
                    now=time.monotonic()
                    (self.folder/'paired-load-start.json').write_text(json.dumps({'started_monotonic':now,'deadline_monotonic':now+3600}))
                    return 0
            class Owner:
                def __init__(self):self.requested=None;self.protected=False;self.stops=0;owners.append(self)
                def check_cancelled(self):pass
                def stop(self):self.stops+=1
                def spawn(self,command,lifecycle,**kwargs):
                    lifecycle=Path(lifecycle);lifecycle.parent.mkdir(parents=True,exist_ok=True);lifecycle.write_text('{"cleanup_status":"absent","exit_code":0}')
                    if '--backend' in command:
                        p0='--native-p0' in command;folder=lifecycle.parent
                        argv=['docker','run','--name',folder.name,'image','python','--output-dir',str(folder)]
                        if p0:argv[2:2]=['-e','WEAVE_NATIVE_P0=1']
                        (folder/'launch.json').write_text(json.dumps(dict(backend='native',graph_mode='full',native_p0=p0,docker_argv=argv)))
                        (folder/'runtime.json').write_text(json.dumps(dict(runtime_status='passed',resolved_image_id='sha256:image',kernel_source_sha256=kernel_source_sha256(),host_gpu_allocation='0,2,4,6')))
                        return Process()
                    folder=lifecycle.parent;(folder/'paired-client-summary.json').write_text(json.dumps(next(results)));return Process(folder)
            output=io.StringIO()
            with ExitStack() as stack:
                patches={
                    'scripts.run_native_p0_pair.OwnedChild':Owner,
                    'scripts.run_native_p0_pair.allocated_gpus':lambda:GPUAllocation.parse('0,2,4,6'),
                    'scripts.run_native_p0_pair.subprocess.run':lambda *a,**k:SimpleNamespace(stdout='sha256:image'),
                    'src.weave.gpqa_c32.load_gpqa_workload':lambda *a,**k:workload,
                    'src.weave.native_p0_pair_client.selection_identity':lambda w:a['selection'],
                    'scripts.run_native_p0_pair.native_checkpoint_proof':lambda *a:dict(checkpoint_content_sha256='f'*64,verified_stats={}),
                    'src.weave.checkpoint_identity.checkpoint_stats':lambda p:{},
                    'scripts.run_native_p0_pair.validate_prior_smoke':lambda *a:dict(status='passed'),
                    'scripts.run_native_p0_pair.assert_ports_free':lambda p:None,
                    'scripts.run_native_p0_pair.wait_ready':lambda *a,**k:None,
                    'scripts.run_native_p0_pair.flush_cache_when_idle':lambda *a,**k:None,
                    'scripts.run_native_p0_pair.baseline_coverage':lambda *a,**k:[],
                    'scripts.run_native_p0_pair.paired_graph_check':lambda *a,**k:dict(status='passed',p0_runtime_sampled=False),
                }
                for name,value in patches.items():stack.enter_context(patch(name,value))
                stack.enter_context(contextlib.redirect_stdout(output))
                self.assertEqual(main(['--work-root',str(root),'--records',str(root/'main.jsonl'),'--execute']),0)
            result=json.loads(output.getvalue());public=Path(result['export_directory'])
            comparison=json.loads((public/'comparison.json').read_text())
            self.assertEqual(result['status'],'paired_check_complete');self.assertEqual(comparison['output_text_digest_mismatched'],32)
            self.assertEqual(json.loads((public/'launch-comparison.json').read_text())['status'],'matched')
            self.assertTrue(all(owner.stops>=3 for owner in owners))

if __name__=='__main__':unittest.main()
