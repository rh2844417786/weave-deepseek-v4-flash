"""Source-pinned lazy native forward hook and rank-local control endpoint."""
import hashlib
import importlib.abc
import importlib.machinery
import json
import os
from pathlib import Path
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

TARGET='sglang.srt.models.deepseek_v4'
NATIVE_SHA256='d13c8434b7ddb6f8d502c6c59d613b0425f9d3781eef5ab4f6936dd8ad831881'

def stall_source_sha256():
    root=Path(__file__).resolve().parents[2]
    digest=hashlib.sha256()
    for path in (root/'src/weave/stall_timeline.py',Path(__file__),root/'src/weave/request_timing.py',root/'timing_hooks/sitecustomize.py'):
        digest.update(path.read_bytes())
    return digest.hexdigest()

def verify_source(path):
    if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=NATIVE_SHA256:
        raise RuntimeError('stall diagnostic requires pinned native model source')

def start_endpoint(collector, torch, source_path):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args): pass
        def answer(self):
            try:
                with torch.cuda.device(collector.device):
                    verify_source(source_path)
                    if stall_source_sha256()!=collector.source_sha256:
                        raise RuntimeError('stall instrumentation source changed since bootstrap')
                    if self.command=='POST' and self.path=='/arm':
                        size=int(self.headers.get('Content-Length','0'))
                        if size>2 or self.rfile.read(size) not in (b'',b'{}'): raise ValueError('arm accepts empty object only')
                        result=collector.arm()
                    elif self.command=='GET' and self.path=='/snapshot': result=collector.snapshot()
                    elif self.command=='GET' and self.path=='/status': result=collector.status()
                    else: raise ValueError('unsupported diagnostic endpoint')
                code=200
            except Exception as error:
                result={'error':type(error).__name__,'reason':str(error)}; code=409
            data=json.dumps(result,allow_nan=False).encode()
            self.send_response(code); self.send_header('Content-Type','application/json'); self.send_header('Content-Length',str(len(data))); self.end_headers(); self.wfile.write(data)
        do_POST=answer
        do_GET=answer
    server=ThreadingHTTPServer(('127.0.0.1',int(os.environ.get('WEAVE_STALL_PORT_BASE','30210'))+collector.rank),Handler)
    threading.Thread(target=server.serve_forever,daemon=True,name='weave-stall-control').start()
    return server

def patch_model_class(cls, source_path):
    if getattr(cls,'_weave_stall_installed',False): return
    original=cls.forward
    def forward(self,*args,**kwargs):
        import torch
        from sglang.srt.distributed import get_tp_group
        from .stall_timeline import StallTimeline,TorchBackend
        collector=getattr(self,'_weave_stall_collector',None)
        if collector is None:
            verify_source(source_path)
            device=torch.cuda.current_device()
            layers=list(self.model.layers)
            if len(layers)!=43: raise RuntimeError('stall diagnostic requires all 43 routed layers')
            collector=StallTimeline(TorchBackend(torch,device),rank=get_tp_group().rank,device=device,
                source_sha256=stall_source_sha256(),native_source_sha256=NATIVE_SHA256,
                identity={'image':os.environ.get('WEAVE_STALL_IMAGE',os.environ.get('WEAVE_IMAGE_ID','unspecified')),'candidate':os.environ.get('WEAVE_STALL_CANDIDATE','unspecified')})
            for index,layer in enumerate(layers):
                expert=layer.mlp.experts
                if getattr(expert,'_weave_stall_wrapped',False): raise RuntimeError('routed expert already wrapped')
                actual=expert.forward
                def wrapped(*a,_actual=actual,_layer=index,**kw):
                    with collector.routed(_layer): return _actual(*a,**kw)
                expert.forward=wrapped
                expert._weave_stall_wrapped=True
            self._weave_stall_collector=collector
            self._weave_stall_server=start_endpoint(collector,torch,source_path)
        batch=kwargs.get('forward_batch',args[2] if len(args)>2 else None)
        ids=kwargs.get('input_ids',args[0] if args else None)
        if batch is None or ids is None: raise RuntimeError('missing native forward batch')
        phase='decode' if batch.forward_mode.is_decode() else 'prefill'
        with collector.forward(phase,ids.shape[0]): return original(self,*args,**kwargs)
    cls.forward=forward
    cls._weave_stall_installed=True

class Loader(importlib.abc.Loader):
    def __init__(self,loader,origin): self.loader,self.origin=loader,origin
    def create_module(self,spec): return self.loader.create_module(spec)
    def exec_module(self,module):
        verify_source(self.origin)
        self.loader.exec_module(module)
        patch_model_class(module.DeepseekV4ForCausalLM,self.origin)
class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname!=TARGET: return None
        spec=importlib.machinery.PathFinder.find_spec(fullname,path)
        if spec is None or spec.loader is None or spec.origin is None: raise ImportError('native model source missing')
        spec.loader=Loader(spec.loader,spec.origin)
        return spec

def install_stall_hook():
    if TARGET in sys.modules: raise RuntimeError('stall hook must precede native import')
    if not any(isinstance(f,Finder) for f in sys.meta_path): sys.meta_path.insert(0,Finder())
