import unittest
from types import SimpleNamespace as NS
from src.weave.native_tail_runtime import install_tail_dispatch, tail_options, NativeTailRuntime


class RuntimeTests(unittest.TestCase):
    def test_options_reject_stacked_methods_and_need_explicit_tables_or_calibration(self):
        self.assertIsNone(tail_options({}))
        env={'WEAVE_NATIVE_TAIL':'measured','WEAVE_NATIVE_TAIL_DIRECTORY':'/srv/work/tables',
             'WEAVE_GRAPH_C32':'1','WEAVE_REQUEST_TIMING':'1'}
        self.assertEqual(tail_options(env),dict(mode='measured',directory='/srv/work/tables',repeats=7))
        for extra in ({'WEAVE_NATIVE_BUDGET':'full'},{'WEAVE_NATIVE_P0':'1'},
                      {'WEAVE_NATIVE_TAIL_DIRECTORY':'relative'},{'WEAVE_NATIVE_TAIL':'adaptive'},
                      {'WEAVE_NATIVE_TAIL_REPEATS':'1'}):
            with self.assertRaises(ValueError):tail_options(dict(env,**extra))
        with self.assertRaisesRegex(ValueError,'run_native_tail.py'):
            tail_options(dict(env,WEAVE_NATIVE_TAIL_DIRECTORY=''))

    def test_prefill_and_pre_capture_decode_keep_native_then_capture_uses_explicit_helper(self):
        log=[]
        class MLP:
            layer_id=0
            def forward(self, hidden_states, forward_batch=None, gemm_output_zero_allocator=None,
                        input_ids=None, input_ids_global=None, skip_shared_experts=False):
                log.append(('native',input_ids_global));return 'native-output'
        mlp=MLP();model=NS(model=NS(layers=[NS(mlp=mlp)]))
        model.forward=lambda a,b,forward_batch:mlp.forward('hidden',forward_batch,None,'local','global')
        observer=NS(capturing=False,captured=set());model.weave_graph_observer=observer
        class Runtime:
            def active(self):return observer.capturing or bool(observer.captured)
            def maybe_calibrate(self,*args):log.append(('calibrate',args[0],args[1]))
            def invoke_tail(self, layer, original, args, kwargs):
                log.append(('tail',layer,args[4]));return 'global-output'
        install_tail_dispatch(model,Runtime(),layers=1)
        batch=lambda decode:NS(forward_mode=NS(is_decode=lambda:decode),batch_size=2)
        self.assertEqual(model.forward(None,None,batch(False)),'native-output')
        self.assertEqual(model.forward(None,None,batch(True)),'native-output')
        observer.capturing=True
        self.assertEqual(model.forward(None,None,batch(True)),'global-output')
        self.assertEqual(log,[('native','global'),('native','global'),('calibrate',0,2),('tail',0,'global')])
        log.clear()
        self.assertEqual(model.forward(None,None,batch(False)),'native-output')
        self.assertEqual(log,[('native','global')])

    def test_duplicate_install_fails_without_mutating_the_first_wrappers(self):
        model=NS(model=NS(layers=[NS(mlp=NS(forward=lambda hidden:hidden))]),forward=lambda *a,**k:None)
        install_tail_dispatch(model,NS(),layers=1)
        saved=model.forward
        with self.assertRaisesRegex(RuntimeError,'already installed'):
            install_tail_dispatch(model,NS(),layers=1)
        self.assertIs(model.forward,saved)

    def test_calibration_runs_only_in_capture_one_warmup(self):
        from unittest.mock import patch
        runtime=object.__new__(NativeTailRuntime)
        runtime.mode='calibration'
        runtime.model=NS()
        runtime.torch=NS(cuda=NS(is_current_stream_capturing=lambda:False))
        runtime.maybe_calibrate(0,2,None,(),{})
        runtime.model.weave_graph_observer=NS(capturing=False,captured=set())
        runtime.maybe_calibrate(0,2,None,(),{})
        runtime.model.weave_graph_observer.capturing=True
        runtime.torch.cuda.is_current_stream_capturing=lambda:True
        runtime.maybe_calibrate(0,2,None,(),{})


if __name__ == '__main__':unittest.main()
