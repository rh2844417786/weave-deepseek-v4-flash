import unittest
from types import SimpleNamespace as NS
from src.weave.native_budget_runtime import install_dispatch, budget_options

class RuntimeTests(unittest.TestCase):
    def fixture(self, calls=2):
        log=[]
        mod=NS(moe_wna16_marlin_gemm=lambda *a,**kw: log.append(('native',kw)))
        def candidate(*a,**kw):log.append(('budget',kw));return 'result'
        class MLP:
            def forward(self,hidden):
                for i in range(calls):
                    mod.moe_wna16_marlin_gemm(hidden, None, None, None, None, None, None,
                        is_ep=True,top_k=6 if i==0 else 1,mul_topk_weights=bool(i))
                return hidden
        mlp=MLP();model=NS(model=NS(layers=[NS(mlp=mlp)]))
        model.forward=lambda a,b,forward_batch: mlp.forward('hidden')
        runtime=NS(states={(0,'gate_up'):'up-state',(0,'down'):'down-state'},
                   mode='adaptive',target_percent=75)
        install_dispatch(model,mod,candidate,runtime,layers=1)
        return model,mod,log

    def test_prefill_keeps_original_and_decode_dispatch_has_stable_buffers(self):
        model,mod,log=self.fixture()
        batch=lambda decode:NS(forward_mode=NS(is_decode=lambda:decode))
        model.forward(None,None,batch(False))
        self.assertEqual([x[0] for x in log],['native','native'])
        log.clear();model.forward(None,None,batch(True))
        self.assertEqual([x[1]['budget_state'] for x in log],['up-state','down-state'])
        self.assertEqual(log[0][1]['budget_target_percent'],75)
        log.clear();mod.moe_wna16_marlin_gemm()
        self.assertEqual(log[0][0],'native') # layer context reset

    def test_missing_or_extra_expert_calls_fail_before_silent_native_fallback(self):
        for calls in (1,3):
            model,mod,log=self.fixture(calls)
            with self.assertRaisesRegex(RuntimeError,'two routed Marlin'):
                model.forward(None,None,NS(forward_mode=NS(is_decode=lambda:True)))
            log.clear();mod.moe_wna16_marlin_gemm()
            self.assertEqual(log[0][0],'native')

    def test_explicit_environment_and_no_profile_mixing(self):
        self.assertIsNone(budget_options({}))
        env={'WEAVE_NATIVE_BUDGET':'adaptive','WEAVE_NATIVE_BUDGET_DIR':'/work/budget',
             'WEAVE_GRAPH_C32':'1','WEAVE_REQUEST_TIMING':'1'}
        self.assertEqual(budget_options(env)['target_percent'],75)
        for key in ('WEAVE_NATIVE_P0','WEAVE_GRAPH_STAGE_PROFILE','WEAVE_FIXED_PREFIX'):
            with self.assertRaises(ValueError):budget_options(dict(env,**{key:'1'}))
        with self.assertRaises(ValueError):budget_options(dict(env,WEAVE_NATIVE_BUDGET_DIR='relative'))
