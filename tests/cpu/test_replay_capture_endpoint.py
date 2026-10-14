from types import SimpleNamespace
import unittest
from unittest.mock import patch,Mock
from urllib.error import HTTPError

from scripts.check_sglang_service import http_json
from src.weave.evidence_server import start_evidence_server
from src.weave.replay_trace import CaptureConfig,CaptureLifecycle
from tests.cpu.test_replay_trace import identity


class ReplayCaptureEndpointTest(unittest.TestCase):
    def test_active_coverage_rejected_before_cuda_synchronization(self):
        counters=Mock();counters.snapshot.side_effect=ValueError('active')
        cuda=Mock();runtime=SimpleNamespace(rank=0,device=0,model_batch_counters=counters,
                                          torch=SimpleNamespace(cuda=cuda))
        model=SimpleNamespace(weave_runtime=runtime)
        backend=SimpleNamespace(coverage=Mock(return_value={}),execution_profile=Mock(return_value={}))
        with patch.dict('sys.modules',{'src.weave.sglang_backend':backend}):server=start_evidence_server(model,0)
        try:
            with self.assertRaises(HTTPError):http_json(f'http://127.0.0.1:{server.server_port}/coverage')
            cuda.synchronize.assert_not_called();cuda.set_device.assert_not_called()
        finally:server.shutdown();server.server_close()

    def test_status_and_arm_use_only_cpu_state_and_preserve_active_forward_guard(self):
        capture=SimpleNamespace(identity=identity(),lifecycle=CaptureLifecycle(CaptureConfig(),rank=0))
        # Any accidental CUDA synchronization/device access fails this test.
        runtime=SimpleNamespace(rank=0,device=0,replay_capture=capture)
        model=SimpleNamespace(weave_runtime=runtime)
        backend=SimpleNamespace(coverage=None,execution_profile=None)
        with patch.dict('sys.modules',{'src.weave.sglang_backend':backend}):
            server=start_evidence_server(model,0)
        try:
            base=f'http://127.0.0.1:{server.server_port}'
            self.assertEqual(http_json(base+'/replay-capture/arm',{}),{'rank':0,'status':'armed'})
            capture.lifecycle.begin('decode',32,32)
            value=http_json(base+'/replay-capture/status')
            self.assertTrue(value['coverage']['active'])
            self.assertEqual(value['coverage']['eligible_forwards'],1)
            self.assertEqual(value['identity'],identity())
            with self.assertRaises(HTTPError):http_json(base+'/replay-capture/arm',{})
            with self.assertRaises(HTTPError):http_json(base+'/replay-capture/arm',{'ids':[1,2]})
            capture.lifecycle.end()
        finally:server.shutdown();server.server_close()


if __name__=='__main__':unittest.main()
