import types
import unittest
from unittest.mock import patch
from src.weave.ops import PeerRuntime


class PeerInitializationTest(unittest.TestCase):
    def test_rank_allocation_failure_is_agreed_before_handle_exchange(self):
        events=[]
        dist=types.ModuleType('torch.distributed')
        dist.get_rank=lambda group:0
        dist.get_world_size=lambda group:4
        dist.get_backend=lambda group:'gloo'
        def gather(items,value,group=None):
            events.append(value)
            items[:]=[value]*4
        dist.all_gather_object=gather
        torch=types.ModuleType('torch')
        torch.cuda=types.SimpleNamespace(current_device=lambda:0)
        torch.distributed=dist
        def allocate(*args): raise RuntimeError('out of memory')
        module=types.SimpleNamespace(Workspace=allocate,weave_source_sha256='source')
        with patch.dict('sys.modules',{'torch':torch,'torch.distributed':dist}),patch('src.weave.ops.extension',return_value=module):
            with self.assertRaisesRegex(RuntimeError,'IPC allocation'):
                PeerRuntime(None,32,4096,2048,6,256,kernel='tiled')
        self.assertTrue(any(isinstance(value,str) and 'out of memory' in value for value in events))
        self.assertFalse(any(isinstance(value,bytes) for value in events))
