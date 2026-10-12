"""Localhost-only idle snapshots, outside the measured serving interval."""
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import threading


def start_evidence_server(model, base_port):
    from .sglang_backend import coverage,execution_profile
    runtime = model.weave_runtime

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path not in ('/coverage','/execution-profile'):
                self.send_error(404)
                return
            try:
                runtime.torch.cuda.set_device(runtime.device)
                runtime.torch.cuda.synchronize(runtime.device)
                payload = json.dumps(coverage(model) if self.path=='/coverage' else execution_profile(model),allow_nan=False).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            except Exception as exc:
                self.send_error(500, str(exc))

        def do_POST(self):
            if self.path!='/execution-profile/reset':
                self.send_error(404);return
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<=length<=64: raise ValueError('bounded reset body required')
                body=self.rfile.read(length) if length else b'{}'
                if json.loads(body)!={}: raise ValueError('empty JSON reset body required')
            except (ValueError,json.JSONDecodeError):
                self.send_error(400,'empty JSON reset body required');return
            try:
                runtime.torch.cuda.set_device(runtime.device)
                runtime.reset_execution_profile()
                payload=json.dumps({'status':'reset','rank':runtime.rank}).encode()
                self.send_response(200);self.send_header('Content-Type','application/json')
                self.send_header('Content-Length',str(len(payload)));self.end_headers()
                self.wfile.write(payload)
            except Exception as exc: self.send_error(500,str(exc))

        def log_message(self, *args):
            pass

    server = HTTPServer(('127.0.0.1', base_port+runtime.rank), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True,
                              name=f'weave-evidence-rank{runtime.rank}')
    thread.start()
    return server
