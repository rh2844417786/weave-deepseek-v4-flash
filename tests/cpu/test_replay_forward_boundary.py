"""Exercise the actual wrapper without requiring the installed SGLang runtime."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.weave.replay_trace import CaptureConfig,CaptureLifecycle


class NativeBoundary:
    """Substitute only the external model execution; keep wrapper/lifecycle real."""
    def forward(self,*args,**kwargs):
        life=self.weave_runtime.replay_capture.lifecycle
        if life.slot is not None:
            for layer in range(43):life.record_layer(layer)
        if self.raise_forward:raise RuntimeError('native model failed')
        return 'model-result'


class ReplayForwardBoundaryTest(unittest.TestCase):
    def model(self):
        modules={'sglang.srt.models.deepseek_v4':SimpleNamespace(DeepseekV4ForCausalLM=NativeBoundary),
                 'weave.sglang_backend':SimpleNamespace(install_weave_backend=None,coverage=None)}
        path=Path(__file__).resolve().parents[2]/'src/weave/sglang_ext/models/deepseek_v4.py'
        spec=importlib.util.spec_from_file_location('test_replay_model_boundary',path)
        module=importlib.util.module_from_spec(spec)
        with patch.dict('sys.modules',modules):spec.loader.exec_module(module)
        model=module.DeepseekV4ForCausalLM();model.raise_forward=False
        life=CaptureLifecycle(CaptureConfig(),rank=0);life.arm()
        model.weave_runtime=SimpleNamespace(replay_capture=SimpleNamespace(lifecycle=life))
        return model,life

    def test_all43_layers_share_one_forward_sequence_and_selected_slot(self):
        model,life=self.model()
        mode=SimpleNamespace(is_decode=lambda:True,is_extend=lambda:False)
        batch=SimpleNamespace(forward_mode=mode,batch_size=32)
        for _ in range(128):
            self.assertEqual(model.forward(SimpleNamespace(shape=(32,)),None,batch),'model-result')
        state=life.snapshot()
        self.assertEqual(state['eligible_forwards'],128)
        self.assertEqual(state['captured_layers'],[43,43])
        self.assertFalse(state['active'])
        self.assertTrue(state['complete'])

    def test_failed_native_forward_invalidates_capture(self):
        model,life=self.model();model.raise_forward=True
        mode=SimpleNamespace(is_decode=lambda:True,is_extend=lambda:False)
        with self.assertRaisesRegex(RuntimeError,'native model failed'):
            model.forward(input_ids=SimpleNamespace(shape=(32,)),
                          forward_batch=SimpleNamespace(forward_mode=mode,batch_size=32))
        self.assertTrue(life.snapshot()['failed'])
        self.assertFalse(life.snapshot()['active'])


if __name__=='__main__':unittest.main()
