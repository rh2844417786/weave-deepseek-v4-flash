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
