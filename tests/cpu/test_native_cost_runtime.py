import unittest
from types import SimpleNamespace as NS
from src.weave.native_cost_runtime import cost_options, classify_layers, install_cost_dispatch

class Parameters:
    def __init__(self, dims=(64,256,1024)):self.shape=dims;self.dtype='torch.int32'
class Expert:
    quant_method=NS()
    def named_parameters(self,recurse=False):return [('w13_weight',Parameters())]
class TopK:pass
class HashTopK:pass

class RuntimeTests(unittest.TestCase):
    def mlp(self,topk):
        return NS(experts=Expert(),shared_experts=None,topk=topk,gate=Expert(),
                  _shared_expert_tp1=False,is_hash=isinstance(topk,HashTopK))
    def test_classes_cover_every_layer_and_keep_hash_routing_separate(self):
        layers=[NS(mlp=self.mlp(HashTopK() if i<3 else TopK())) for i in range(43)]
        groups,mapping=classify_layers(NS(model=NS(layers=layers)))
        self.assertEqual(len(groups),2)
        self.assertEqual(sorted(i for g in groups.values() for i in g['layers']),list(range(43)))
        self.assertEqual([g['representative_layer'] for g in groups.values()],[0,3])
        self.assertNotEqual(mapping[0],mapping[3])

    def test_cost_mode_requires_measured_table_or_explicit_calibration(self):
        self.assertIsNone(cost_options({}))
        env={'WEAVE_NATIVE_BUDGET':'cost','WEAVE_GRAPH_C32':'1','WEAVE_REQUEST_TIMING':'1'}
        with self.assertRaisesRegex(ValueError,'run_native_cost.py'):cost_options(env)
        options=cost_options(dict(env,WEAVE_NATIVE_COST_TABLE_DIR='/work/tables'))
        self.assertEqual(options['mode'],'cost')
        options=cost_options(dict(env,WEAVE_NATIVE_COST_CALIBRATE_DIR='/work/new'))
        self.assertEqual(options['mode'],'calibration')
        for extra in ({'WEAVE_NATIVE_COST_TABLE_DIR':'/work/a','WEAVE_NATIVE_COST_CALIBRATE_DIR':'/work/b'},
                      {'WEAVE_NATIVE_COST_TABLE_DIR':'relative'},
                      {'WEAVE_NATIVE_COST_TABLE_DIR':'/work/a','WEAVE_NATIVE_P0':'1'}):
            with self.assertRaises(ValueError):cost_options(dict(env,**extra))

    def test_model_prefill_and_decode_choose_correct_callers_and_release_context(self):
        log=[]
        mod=NS(moe_wna16_marlin_gemm=lambda *a,**k:log.append(('native',k)))
        class MLP:
            def forward(self,hidden,**kwargs):
                for i in range(2):mod.moe_wna16_marlin_gemm(hidden,None,None,None,None,None,None,
                    is_ep=True,mul_topk_weights=bool(i),top_k=1 if i else 6,size_m=42 if i else 7,size_n=4096,size_k=2048 if i else 4096)
                return hidden
        mlp=MLP();model=NS(model=NS(layers=[NS(mlp=mlp)]))
        model.forward=lambda a,b,forward_batch:mlp.forward('hidden')
        class Runtime:
            mode='cost'
            def maybe_calibrate(self,*args):raise AssertionError('formal mode cannot calibrate')
            def dispatch(self,layer,index,args,kwargs,observer=None,original_native=False,original=None):
                log.append(('cost',(layer,index)));return 'cost'
        runtime=Runtime()
        install_cost_dispatch(model,mod,NS(tensor_model_parallel_all_reduce=lambda x:x),runtime,layers=1)
        model.forward(None,None,NS(forward_mode=NS(is_decode=lambda:False),batch_size=7))
        self.assertEqual([x[0] for x in log],['native','native'])
        log.clear();model.forward(None,None,NS(forward_mode=NS(is_decode=lambda:True),batch_size=7))
        self.assertEqual(log,[('cost',(0,0)),('cost',(0,1))])
        log.clear();mod.moe_wna16_marlin_gemm()
        self.assertEqual(log[0][0],'native')

class GraphSetupTests(unittest.TestCase):
    def test_cost_setup_cannot_fall_into_heuristic_parser(self):
        import os
        from unittest.mock import patch
        from src.weave.graph_hooks import setup_budget
        env={'WEAVE_NATIVE_BUDGET':'cost','WEAVE_GRAPH_C32':'1','WEAVE_REQUEST_TIMING':'1',
             'WEAVE_NATIVE_COST_TABLE_DIR':'/work/table'}
        with patch.dict(os.environ,env,clear=True), \
             patch('src.weave.native_cost_runtime.install_native_cost',return_value='cost') as install, \
             patch('src.weave.native_budget_runtime.budget_options',side_effect=AssertionError('heuristic parser')):
            self.assertEqual(setup_budget('model','torch',0,'root'),'cost')
            self.assertEqual(install.call_args.kwargs,dict(mode='cost',directory='/work/table',repeats=7))

    def test_original_native_installs_no_budget_runtime(self):
        import os
        from unittest.mock import patch
        from src.weave.graph_hooks import setup_budget
        with patch.dict(os.environ,{},clear=True):
            self.assertIsNone(setup_budget('model','torch',0,'root'))

class StartupBoundaryTests(unittest.TestCase):
    def test_pre_graph_autotune_cannot_trigger_calibration(self):
        from src.weave.native_cost_runtime import NativeCostRuntime
        runtime=object.__new__(NativeCostRuntime)
        runtime.torch=NS(cuda=NS(is_current_stream_capturing=lambda:False))
        runtime.model=NS(weave_graph_observer=NS(capturing=False))
        # No tensors/class tables are needed before the true capture_one boundary.
        runtime.maybe_calibrate(0,None,(),{})

    def test_repeat_count_matches_launcher_without_arbitrary_maximum(self):
        env={'WEAVE_NATIVE_BUDGET':'cost','WEAVE_GRAPH_C32':'1','WEAVE_REQUEST_TIMING':'1',
             'WEAVE_NATIVE_COST_CALIBRATE_DIR':'/work/new'}
        for value in ('2','101'):
            self.assertEqual(cost_options(dict(env,WEAVE_NATIVE_COST_REPEATS=value))['repeats'],int(value))
        with self.assertRaises(ValueError):cost_options(dict(env,WEAVE_NATIVE_COST_REPEATS='1'))
