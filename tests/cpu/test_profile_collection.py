import copy
from http.server import BaseHTTPRequestHandler,HTTPServer
import json
from pathlib import Path
import socket
import tempfile
import threading
import unittest

try:
    from scripts.collect_execution_profile import collect_profiles,reset_profiles
except ImportError:
    collect_profiles=reset_profiles=None
from src.weave.execution_profile import TraceConfig,summarize_profile
from tests.cpu.test_execution_profile import fixture


class ProfileCollectionTest(unittest.TestCase):
    def test_fixed_eager_capture_requires_forward_samples_and_the_requested_plan(self):
        import scripts.collect_execution_profile as module
        self.assertTrue(hasattr(module,'validate_capture'))
        identity,heads,totals,records=fixture()
        identity.update(graph_enabled=False,profile_bank_sha256=None,schedule_mode='explicit_fixed',
                        fixed_plan=[1,1,1],image_id='sha256:'+'d'*64,
                        host_gpu_allocation='0,2,4,6',device_key='fixture;sm90;2')
        config=TraceConfig(stride=2,capacity=2,layers=1)
        value=summarize_profile(config,identity,heads,totals,records,workers=2)
        value['layers']['0']['decode']=copy.deepcopy(value['layers']['0']['prefill'])
        gate={'source_sha256':'a'*64,'checkpoint_content_sha256':'c'*64,'image_id':'sha256:'+'d'*64}
        bound={'rank':0,'device_key':'fixture;sm90;2'}
        with self.assertRaises(ValueError):module.validate_capture(value,config,gate,bound,'0,2,4,6',fixed_plan=[1,1,1])
        from src.weave.forward_profile import ForwardSampler
        from tests.cpu.test_forward_profile import Event
        from types import SimpleNamespace
        clock=[0.]
        sampler=ForwardSampler(SimpleNamespace(Event=lambda **kw:Event(clock),
            is_current_stream_capturing=lambda:False),stride=2,capacity=2,clock=lambda:clock[0])
        for _ in range(5):
            ticket=sampler.begin('decode',32,32);clock[0]+=.01;sampler.end(ticket)
        value['forward_profile']=sampler.snapshot()
        module.validate_capture(value,config,gate,bound,'0,2,4,6',fixed_plan=[1,1,1])
        for key,new in [('graph_enabled',True),('fixed_plan',[1,1,0]),('profile_bank_sha256','b'*64)]:
            bad=copy.deepcopy(value);bad[key]=new
            with self.subTest(key=key),self.assertRaises(ValueError):
                module.validate_capture(bad,config,gate,bound,'0,2,4,6',fixed_plan=[1,1,1])

    def test_four_rank_localhost_snapshots_are_bound_and_saved_publicly(self):
        self.assertIsNotNone(collect_profiles)
        identity,heads,totals,records=fixture()
        identity.update(image_id='sha256:'+'d'*64,host_gpu_allocation='0,2,4,6',device_key='fixture;sm90;2')
        payload=summarize_profile(TraceConfig(stride=2,capacity=2,layers=1),identity,heads,totals,records,workers=2)
        payload['layers']['0']['decode']=copy.deepcopy(payload['layers']['0']['prefill'])
        servers=[]
        try:
            for _ in range(30):
                servers=[]
                first=HTTPServer(('127.0.0.1',0),BaseHTTPRequestHandler);base=first.server_port;first.server_close()
                try:
                    for rank in range(4):
                        data=copy.deepcopy(payload);data['rank']=rank
                        class Handler(BaseHTTPRequestHandler):
                            value=data
                            def do_GET(self):
                                if self.path!='/execution-profile': self.send_error(404);return
                                body=json.dumps(self.value).encode();self.send_response(200)
                                self.send_header('Content-Type','application/json');self.end_headers();self.wfile.write(body)
                            def do_POST(self):
                                if self.path!='/execution-profile/reset': self.send_error(404);return
                                self.rfile.read(int(self.headers.get('Content-Length','0')))
                                body=json.dumps({'status':'reset','rank':self.value['rank']}).encode()
                                self.send_response(200);self.send_header('Content-Type','application/json');self.end_headers();self.wfile.write(body)
                            def log_message(self,*args): pass
                        server=HTTPServer(('127.0.0.1',base+rank),Handler);servers.append(server)
                    break
                except OSError:
                    for server in servers: server.server_close()
            self.assertEqual(len(servers),4)
            for server in servers: threading.Thread(target=server.serve_forever,daemon=True).start()
            config=TraceConfig(stride=2,capacity=2,layers=1)
            gate={'source_sha256':'a'*64,'checkpoint_content_sha256':'c'*64,'image_id':'sha256:'+'d'*64}
            profiles={'profiles':[{'rank':rank,'sha256':'b'*64,'device_key':'fixture;sm90;2'} for rank in range(4)]}
            self.assertEqual(reset_profiles(base),{'status':'reset','ranks':[0,1,2,3]})
            with tempfile.TemporaryDirectory() as tmp:
                result=collect_profiles(base,Path(tmp),config,gate,profiles,'0,2,4,6')
                self.assertEqual(result['status'],'profiled')
                self.assertEqual(result['ranks'],[0,1,2,3])
                self.assertEqual(len(list(Path(tmp).glob('execution-profile-rank*.json'))),4)
                self.assertEqual((Path(tmp)/'execution-profile-rank0.json').stat().st_mode&0o777,0o664)
                servers[2].RequestHandlerClass.value['source_sha256']='f'*64
                with self.assertRaises(ValueError):
                    collect_profiles(base,Path(tmp)/'bad',config,gate,profiles,'0,2,4,6')
                self.assertFalse((Path(tmp)/'bad').exists())
        finally:
            for server in servers: server.shutdown();server.server_close()

    def test_instrumented_comparison_cannot_be_labelled_formal_performance(self):
        from src.weave.performance import compare_groups,summarize_response
        from tests.cpu.test_gpqa_performance import response
        row=summarize_response(37,'owned',[1,2],response())
        group={'records':[row],'measurement_mode':'instrumented_diagnostic'}
        result=compare_groups(group,copy.deepcopy(group))
        self.assertIn('used_for_formal_performance',result)
        self.assertFalse(result['used_for_formal_performance'])
        self.assertEqual(result['measurement_mode'],'instrumented_diagnostic')
        with self.assertRaises(ValueError):
            compare_groups(group,{'records':[row],'measurement_mode':'formal_performance'})
