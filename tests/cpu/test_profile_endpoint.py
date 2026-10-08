import copy
from types import SimpleNamespace
import threading
import unittest
from unittest.mock import patch

from src.weave.evidence_server import start_evidence_server
from src.weave.ops import PeerRuntime
from src.weave.execution_profile import TraceConfig
from scripts.check_sglang_service import http_json
from tests.cpu.test_execution_profile import fixture


class DeviceBuffer:
    """Only the external CUDA tensor storage is substituted."""
    def __init__(self,value): self.value=copy.deepcopy(value)
    def cpu(self): return self
    def tolist(self): return copy.deepcopy(self.value)
    def zero_(self):
        def zero(value): return [zero(v) for v in value] if isinstance(value,list) else 0
        self.value=zero(self.value)


class ProfileEndpointTest(unittest.TestCase):
    def test_profile_endpoint_reads_real_aggregation_and_resets_real_buffers(self):
        identity,heads,totals,records=fixture()
        runtime=PeerRuntime.__new__(PeerRuntime)
        runtime.rank=runtime.device=0;runtime.trace_workers=2
        runtime.trace_config=TraceConfig(stride=2,capacity=2,layers=1)
        runtime.trace_buffers=tuple(DeviceBuffer(v) for v in (heads,totals,records))
        runtime.torch=SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda *args:None,set_device=lambda *args:None))
        model=SimpleNamespace(weave_runtime=runtime)
        backend=SimpleNamespace(coverage=lambda model:{'rank':0},
            execution_profile=lambda model:runtime.execution_profile_snapshot(identity))
        with patch.dict('sys.modules',{'src.weave.sglang_backend':backend}):
            server=start_evidence_server(model,0)
        try:
            url=f'http://127.0.0.1:{server.server_port}'
            value=http_json(url+'/execution-profile')
            self.assertEqual(value['layers']['0']['prefill']['calls'],5)
            self.assertFalse(value['used_for_formal_performance'])
            self.assertEqual(http_json(url+'/execution-profile/reset',{}),{'status':'reset','rank':0})
            value=http_json(url+'/execution-profile')
            self.assertEqual(value['layers']['0']['prefill']['calls'],0)
            self.assertEqual(value['layers']['0']['prefill']['recent_samples'],[])
        finally: server.shutdown();server.server_close()
