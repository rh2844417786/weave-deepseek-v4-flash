"""Execute the pinned front against CPU stand-ins for the unavailable CUDA ops."""
import math
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from src.weave import native_tail_front as front

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / 'third_party/sglang_marlin_2948168'


class Tensor:
    def __init__(self, shape, dtype='bf16', values=None):
        self.shape, self.dtype, self.device = tuple(shape), dtype, 'cuda:0'
        self.values = list(values) if values is not None else [0.] * math.prod(shape)

    def is_contiguous(self): return True
    def size(self, dim): return self.shape[dim]
    def numel(self): return math.prod(self.shape)
    def view(self, *shape):
        if -1 in shape:
            shape = tuple(len(self.values) // math.prod([x for x in shape if x != -1]) if x == -1 else x for x in shape)
        return Tensor(shape, self.dtype, self.values)
    def __getitem__(self, item): return Tensor((len(self.values[item]),), self.dtype, self.values[item])
    def zero_(self): self.values[:] = [0.] * len(self.values); return self


class FrontTests(unittest.TestCase):
    def fixture(self):
        calls = []
        torch = NS(float16='f16', bfloat16='bf16', half='f16', int32='i32', int='i32',
                   float32='f32', float8_e8m0fnu='e8m0',
                   empty=lambda shape, **kw: Tensor(shape, kw['dtype']),
                   zeros=lambda shape, **kw: Tensor((shape,) if isinstance(shape, int) else shape, kw['dtype']),
                   cuda=NS(get_device_capability=lambda d:(9,0)))
        def gemm(*args, **kwargs):
            if kwargs['mul_topk_weights']:
                raise AssertionError('old native down must not execute')
            calls.append(('gate_up', args, kwargs))
            out = args[1]
            out.values[:] = [1.] * len(out.values)
            return out
        def silu(source, out):
            calls.append(('activation', source, out))
            out.values[:] = [1. / (1. + math.exp(-1.))] * len(out.values)
        native = NS(torch=torch, get_scalar_type=lambda *a:'mxfp4',
                    moe_wna16_marlin_gemm=gemm, silu_and_mul=silu)
        grouped = (Tensor((24,), 'i32', range(24)), Tensor((3,), 'i32', [0,-1,1]), Tensor((1,), 'i32',[24]))
        def align(*args): calls.append(('group', args)); return grouped
        module = NS(moe_align_block_size=align)
        inputs = dict(hidden_states=Tensor((2,16)), w1=Tensor((2,1,64),'i32'),
                      w2=Tensor((2,1,32),'i32'), w1_scale=Tensor((2,1,32),'e8m0'),
                      w2_scale=Tensor((2,1,16),'e8m0'), gating_output=Tensor((2,2),'f32'),
                      topk_weights=Tensor((2,2),'f32',[.25,.75,.6,.4]),
                      topk_ids=Tensor((2,2),'i32',[0,-1,1,0]), expert_map=Tensor((4,),'i32'),
                      global_num_experts=4, workspace=Tensor((4,),'i32'), num_bits=4)
        build = front.build_front((FIXTURE / front.FUSED_SOURCE).read_text(), native)
        return build, inputs, calls, module, grouped

    def test_real_pinned_front_stops_after_activation_and_preserves_native_routes(self):
        build, inputs, calls, module, grouped = self.fixture()
        with patch.dict(sys.modules, {'sglang.srt.layers.moe.fused_moe_triton':module}):
            down = build(**inputs)
        self.assertEqual([x[0] for x in calls], ['group','gate_up','activation'])
        self.assertEqual(down.activated_pairs.shape, (4,16))
        self.assertEqual(down.activated_pairs.values, [1/(1+math.exp(-1))]*64)
        self.assertIs(down.sorted_token_ids, grouped[0])
        self.assertIs(down.local_expert_ids, grouped[1])
        self.assertIs(down.original_slot_mapping, grouped[0])
        self.assertIs(down.topk_weights, inputs['topk_weights'])
        self.assertIs(down.topk_ids, inputs['topk_ids'])
        self.assertIs(down.w2, inputs['w2'])
        self.assertEqual((down.geometry.tokens, down.geometry.topk, down.geometry.pair_rows,
                          down.geometry.block_size_m, down.geometry.n, down.geometry.k), (2,2,4,8,16,16))
        self.assertFalse(calls[1][2]['mul_topk_weights'])
        self.assertTrue(calls[1][2]['is_ep'])

    def test_unsupported_precision_fails_before_native_work(self):
        build, inputs, calls, module, _ = self.fixture()
        inputs['hidden_states'].dtype = 'f16'
        with self.assertRaisesRegex(RuntimeError, 'BF16.*MXFP4'):
            build(**inputs)
        self.assertEqual(calls, [])

    def test_partial_group_allocation_does_not_create_an_out_of_bounds_task(self):
        build,inputs,calls,module,grouped=self.fixture()
        grouped[0].shape=(25,);grouped[0].values=list(range(25))
        grouped[1].shape=(4,);grouped[1].values=[0,-1,1,-1]
        with patch.dict(sys.modules, {'sglang.srt.layers.moe.fused_moe_triton':module}):
            down=build(**inputs)
        self.assertEqual(down.geometry.group_capacity,3)

    def test_modified_source_is_rejected_before_compilation(self):
        raw = (FIXTURE / front.FUSED_SOURCE).read_text()
        with self.assertRaisesRegex(RuntimeError, 'source mismatch'):
            front.build_front(raw + '\n', NS())

    def test_native_runner_keeps_activation_options_and_weights_without_running_down(self):
        observed = []
        def receive(**kwargs): observed.append(kwargs); return 'down-inputs'
        torch = NS(float16='f16', bfloat16='bf16', float8_e8m0fnu='e8m0')
        runner = front.build_runner_front((FIXTURE / front.RUNNER_SOURCE).read_text(), NS(torch=torch), receive)
        quant = NS(w13_qweight='w13', w2_qweight='w2', w13_scales=Tensor((1,), 'e8m0'),
            w2_scales=Tensor((1,), 'e8m0'), w13_g_idx_sort_indices=None, w2_g_idx_sort_indices=None,
            weight_bits=4, w13_g_idx=None, w2_g_idx=None, is_k_full=True, w13_qzeros=None,
            w2_qzeros=None, expert_map='map', global_num_experts=256, w13_global_scale=None,
            w2_global_scale=None, w13_bias=None, w2_bias=None)
        topk = NS(router_logits='logits', topk_weights='weights', topk_ids='ids')
        config = NS(is_gated=True, activation='silu', routed_scaling_factor=2.5,
                    swiglu_limit=7.0, gemm1_clamp_limit=None, gemm1_alpha=None)
        workspace = NS(marlin_make_workspace=lambda *args, **kwargs:'native-workspace')
        with patch.dict(sys.modules, {'sglang.srt.layers.quantization.marlin_utils':workspace}):
            result = runner(NS(hidden_states=Tensor((2,16)), topk_output=topk), quant, config)
        self.assertEqual(result, 'down-inputs')
        self.assertEqual(observed[0]['clamp_limit'],7.0)
        self.assertEqual(observed[0]['routed_scaling_factor'],2.5)
        self.assertEqual(observed[0]['topk_weights'],'weights')
        self.assertEqual(observed[0]['workspace'],'native-workspace')


class DecodeTests(unittest.TestCase):
    def fixture(self, hash_layer=True):
        log=[]
        class Stream:
            def __init__(self, name): self.name=name
            def wait_stream(self, other): log.append(('wait',self.name,other.name))
            def __enter__(self): log.append(('stream',self.name))
            def __exit__(self,*args): pass
        main,alt=Stream('main'),Stream('shared')
        torch=NS(bfloat16='bf16',cuda=NS(current_stream=lambda:main,stream=lambda x:x),
                 nn=NS(functional=NS(pad=lambda *a,**k: self.fail('unexpected padding'))))
        class Dispatcher:
            skip_local_expert_mapping=False
            def dispatch(self,**kw):
                log.append(('dispatch',kw['topk_output']))
                return NS(hidden_states=kw['hidden_states'],topk_output=kw['topk_output'])
        class Quant: pass
        class TopK:
            apply_routed_scaling_factor_on_output=False
            def __call__(self, hidden, logits, **kw):
                log.append(('topk',kw));return NS(topk_ids='mapped',topk_weights='weights')
        experts=NS(quant_method=Quant(),dispatcher=Dispatcher(),reduce_results=False,
            _dwdp_bound=False,should_fuse_routed_scaling_factor_in_topk=False,
            w13_weight=Tensor((2,1,64),'i32'),
            moe_runner_config=NS(inplace=False,apply_router_weight_on_input=False,
                                 activation='silu',is_gated=True),
            runner=NS(lora_enabled=False,runner_backend=NS(is_marlin=lambda:True)))
        def gate(hidden, allocator): log.append(('gate',allocator));return 'logits'
        def shared(hidden, allocator, **kw): log.append(('shared',allocator,kw));return 'shared-partial'
        mlp=NS(tp_size=4,moe_ep_size=4,_shared_expert_tp1=False,shared_experts_is_fp8=True,
            shared_experts=object(),_enable_a2a_moe=False,num_fused_shared_experts=0,
            _fuse_shared_experts_inside_sbo=False,experts=experts,alt_stream=alt,topk=TopK(),
            gate=gate,_forward_shared_experts=shared,_maybe_quant_moe_input_once=lambda h:None,
            is_hash=hash_layer,is_nextn=False,layer_id=3,routed_scaling_factor=2.5)
        native=NS(get_exec=lambda:NS(moe=NS(enable_eplb=False)),
                  get_forward=lambda:NS(flashinfer_trtllm_bypass=False),
                  should_skip_post_experts_all_reduce=lambda **kw:False)
        def runner(dispatch,quant,config): log.append(('front',));return 'down-inputs'
        helper=front.NativeDecodeFront(torch,native,runner,lambda experts:'quant',
                                        Dispatcher,Quant)
        def tail(down, shared, **kw): log.append(('tail',down,shared,kw));return 'global-output'
        return helper,mlp,tail,log

    def test_hash_parameters_shared_dependency_and_scale_reach_exactly_one_tail(self):
        helper,mlp,tail,log=self.fixture()
        result=helper(mlp,Tensor((2,16)),tail,gemm_output_zero_allocator='allocator',
                      input_ids='local-id',input_ids_global='global-id')
        self.assertEqual(result,'global-output')
        self.assertEqual([x[0] for x in log],['wait','gate','topk','dispatch','front','stream','shared','wait','tail'])
        self.assertEqual(log[2][1],{'input_ids':'global-id','expert_location_dispatch_info':None})
        self.assertEqual(log[-2],('wait','main','shared'))
        self.assertEqual(log[-1],('tail','down-inputs','shared-partial',dict(layer_id=3,routed_scaling_factor=2.5,output_width=16)))

    def test_tp1_shared_and_missing_hash_ids_fail_before_launch(self):
        for tp1 in (True,False):
            helper,mlp,tail,log=self.fixture()
            mlp._shared_expert_tp1=tp1
            with self.assertRaisesRegex(RuntimeError,'shared TP1|input_ids_global'):
                helper(mlp,Tensor((2,16)),tail)
            self.assertEqual(log,[])

    def test_scaling_already_in_topk_is_rejected_instead_of_doubled(self):
        helper,mlp,tail,log=self.fixture()
        mlp.topk.apply_routed_scaling_factor_on_output=True
        with self.assertRaisesRegex(RuntimeError,'scaling'):
            helper(mlp,Tensor((2,16)),tail,input_ids_global='ids')
        self.assertEqual(log,[])

    def test_fp16_model_input_cannot_silently_change_output_dtype(self):
        helper,mlp,tail,log=self.fixture()
        with self.assertRaisesRegex(RuntimeError,'BF16'):
            helper(mlp,Tensor((2,16),'f16'),tail,input_ids_global='ids')
        self.assertEqual(log,[])


if __name__ == '__main__': unittest.main()
