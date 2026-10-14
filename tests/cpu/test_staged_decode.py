"""The adapter orchestration remains testable without a local CUDA installation."""
import importlib
import types
import unittest
from unittest.mock import patch

from src.weave import ops


class Tensor:
    def __init__(self, rows, columns, label='tensor', owner=None):
        self.shape = (rows, columns)
        self.label, self.owner = label, owner
        self.slices = 0

    def __getitem__(self, key):
        self.slices += 1
        return Tensor(key.stop - (key.start or 0), self.shape[1], self.label, self)


class Workspace:
    def __init__(self, *dimensions):
        self.dimensions = dimensions
        self.online_slot_outputs = Tensor(192, 4096, 'slots')
        self.online_output = Tensor(32, 4096, 'output')
        self.events = []
        self.binds = []

    def resources(self):
        return {'online_group_capacity':72,'online_gate_grid_bound':2304,
                'online_down_grid_bound':4608,
                'online_grid_policy':'capacity_bound_direct_with_device_count_guard'}

    def bind_online_weights(self, weights):
        bound = types.SimpleNamespace(owner=self, weights=tuple(weights))
        self.binds.append(bound)
        return bound

    def run_online(self, x, ids, routing, bound, clamp):
        if bound.owner is not self:
            raise ValueError('foreign workspace')
        self.events.append(('run', x, ids, routing, bound, clamp))

    def finalize_online(self, batch):
        self.events.append(('finalize', batch))


class StagedDecodeTest(unittest.TestCase):
    def module(self):
        try:
            return importlib.import_module('src.weave.staged_decode')
        except ModuleNotFoundError:
            self.fail('staged decode runtime is not implemented')

    def runtime(self):
        module = self.module()
        group = object()
        dist = types.SimpleNamespace(ReduceOp=types.SimpleNamespace(SUM='SUM'))
        torch = types.SimpleNamespace(distributed=dist,
            cuda=types.SimpleNamespace(is_current_stream_capturing=lambda:False))
        peer = types.SimpleNamespace(torch=torch, rank=2, ep=4, h=4096, i=2048, topk=6)
        extension = types.SimpleNamespace(StagedWorkspace=Workspace)
        with patch.object(ops, 'extension', return_value=extension):
            runtime = module.StagedDecodeRuntime(peer, group)
        dist.all_reduce = lambda tensor, op, group: runtime.ws.events.append(
            ('reduce', tensor, op, group))
        return runtime, group

    def test_full_inputs_one_active_slot_collective_then_origin_finalize(self):
        runtime, group = self.runtime()
        bound = runtime.bind_weights(tuple(object() for _ in range(6)))
        x, ids, routing = Tensor(7,4096), Tensor(7,6), Tensor(7,6)
        result = runtime.run(x, ids, routing, bound, 10)
        events = runtime.ws.events
        self.assertEqual([event[0] for event in events], ['run','reduce','finalize'])
        self.assertIs(events[0][1], x)
        self.assertEqual(events[1][1].shape, (42,4096))
        self.assertEqual(events[1][2:], ('SUM', group))
        self.assertEqual(events[2], ('finalize',7))
        self.assertEqual(result.shape, (7,4096))
        self.assertIs(result.owner, runtime.ws.online_output)
        self.assertEqual(runtime.additional_slot_collectives, 1)

    def test_workspace_and_prefix_views_are_reused_across_layers_and_calls(self):
        runtime, _ = self.runtime()
        raw_a, raw_b = tuple(object() for _ in range(6)), tuple(object() for _ in range(6))
        bound_a, bound_b = runtime.bind_weights(raw_a), runtime.bind_weights(raw_b)
        reads = runtime.ws.online_slot_outputs.slices, runtime.ws.online_output.slices
        results = [runtime.run(Tensor(32,4096),Tensor(32,6),Tensor(32,6),bound,10)
                   for bound in (bound_a,bound_b,bound_a)]
        self.assertEqual(runtime.ws.dimensions, (2,4,32,4096,2048,256,6))
        self.assertTrue(all(result is results[0] for result in results))
        self.assertEqual(reads, (32,32))
        self.assertEqual(reads, (runtime.ws.online_slot_outputs.slices,runtime.ws.online_output.slices))
        self.assertIs(bound_a.weights[0], raw_a[0])
        self.assertIs(bound_b.owner, runtime.ws)
        self.assertEqual(getattr(runtime,'resources',{}).get('online_grid_policy'),
                         'capacity_bound_direct_with_device_count_guard')

    def test_unsupported_batches_or_capture_fail_before_any_collective(self):
        runtime, _ = self.runtime()
        bound = runtime.bind_weights(tuple(object() for _ in range(6)))
        for batch in (0,33):
            with self.subTest(batch=batch), self.assertRaisesRegex(ValueError, '1..32'):
                runtime.run(Tensor(batch,4096),Tensor(batch,6),Tensor(batch,6),bound,10)
        runtime.torch.cuda.is_current_stream_capturing=lambda:True
        with self.assertRaisesRegex(RuntimeError,'eager'):
            runtime.run(Tensor(1,4096),Tensor(1,6),Tensor(1,6),bound,10)
        self.assertEqual(runtime.ws.events, [])

    def test_collective_failure_does_not_finalize_or_count_success(self):
        runtime,_=self.runtime()
        def fail(*args,**kwargs):raise RuntimeError('collective failed')
        runtime.torch.distributed.all_reduce=fail
        bound=runtime.bind_weights(tuple(object() for _ in range(6)))
        with self.assertRaisesRegex(RuntimeError,'collective failed'):
            runtime.run(Tensor(1,4096),Tensor(1,6),Tensor(1,6),bound,10)
        self.assertEqual([event[0] for event in runtime.ws.events],['run'])
        self.assertEqual(runtime.additional_slot_collectives,0)

    def test_staged_configuration_rejects_every_unsupported_mode(self):
        validate = self.module().validate_decode_configuration
        valid = dict(decode_implementation='staged_direct',kernel='tiled',hot_cache=True,
                     allow_decode_graph=False,fixed_plan=(1,1,0),profile_path=None,
                     execution_profile=None,replay_capture=False,forward_profile=False)
        validate(**valid)
        for change in ({'kernel':'wmma'},{'hot_cache':False},{'allow_decode_graph':True},
                       {'fixed_plan':(0,1,0)},{'profile_path':'bank'},
                       {'execution_profile':object()},{'replay_capture':True},
                       {'forward_profile':True},{'decode_implementation':'unknown'}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate(**{**valid,**change})

    def test_cpu_model_counts_keep_failed_forwards_out_of_success_histogram(self):
        counts = self.module().ModelBatchCounters()
        counts.begin('decode',7,7)
        with self.assertRaisesRegex(ValueError,'active'): counts.snapshot()
        counts.end(success=True)
        before = counts.snapshot()
        counts.begin('decode',32,32); counts.end(success=True)
        counts.begin('prefill',50,50); counts.end(success=False)
        after = counts.snapshot()
        self.assertEqual(before['by_phase']['decode'], {'7':1})
        self.assertEqual(after['by_phase']['decode'], {'7':1,'32':1})
        self.assertEqual(after['by_phase']['prefill'], {})
        self.assertEqual(after['decode_valid_batch_histogram'], {'7':1,'32':1})
        self.assertEqual(after['successful_forwards'], 2)
        self.assertEqual(after['failed_forwards'], 1)
        self.assertEqual(after['sequence'], 3)
        self.assertFalse(after['active'])


if __name__=='__main__':unittest.main()
