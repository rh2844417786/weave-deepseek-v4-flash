"""Fake-event orchestration tests; no GPU timing claims are made here."""
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.weave import graph_stage_profile as profile


class FakeEvent:
    def __init__(self, cuda, **options):
        self.cuda, self.options, self.stamp = cuda, options, None
        cuda.events.append(self)
    def record(self):
        self.cuda.log.append('event')
        if self.cuda.capturing:
            self.cuda.operations.append(self.record_timestamp)
        else:
            self.record_timestamp()
    def record_timestamp(self):
        self.stamp = self.cuda.time
    def synchronize(self):
        self.cuda.log.append('sync_end')
    def elapsed_time(self, end):
        self.cuda.log.append('read')
        return end.stamp - self.stamp


class FakeCuda:
    def __init__(self):
        self.events, self.log, self.operations = [], [], []
        self.capturing, self.time = False, 0.
    def Event(self, **options):
        return FakeEvent(self, **options)
    def is_current_stream_capturing(self):
        return self.capturing
    def advance(self, value):
        if self.capturing:
            self.operations.append(lambda: self.advance(value))
        else:
            self.time += value
    def replay(self):
        for operation in self.operations:
            operation()


class ProfileTest(unittest.TestCase):
    def setUp(self):
        self.cuda = FakeCuda()
        self.torch = SimpleNamespace(cuda=self.cuda)
    def capture_layer(self, sampler, bound, durations=(2., 3., 1.)):
        self.cuda.operations = []
        self.cuda.capturing = True
        events = sampler.events_for(bound)
        events[0].record()
        for index, duration in enumerate(durations):
            self.cuda.advance(duration)
            events[index + 1].record()
        self.cuda.capturing = False
    def test_events_materialized_before_capture_external_four_only(self):
        sampler = profile.GraphStageProfiler(self.torch, 0, 'weave', layers=1)
        bound = object()
        sampler.register_layer(2, bound)
        self.assertEqual(len(self.cuda.events), 6)
        self.assertTrue(all(event.stamp is not None for event in self.cuda.events))
        self.assertEqual([event.options.get('external', False) for event in self.cuda.events], [False]*2+[True]*4)
        sampler.register_layer(2, bound)
        self.assertEqual(len(self.cuda.events), 6)
        self.cuda.capturing = True
        with self.assertRaisesRegex(RuntimeError, 'before graph capture'):
            sampler.register_layer(2, bound)
    def test_only_every_128_b32_samples_actual_sequence(self):
        sampler = profile.GraphStageProfiler(self.torch, 0, 'native')
        for sequence in range(1, 257):
            selected = sampler.begin_replay(sequence, 32 if sequence % 2 == 0 else 7)
            self.assertEqual(selected, sequence == 256)
            if selected:
                self.cuda.advance(4.)
                result = sampler.finish_replay(sequence, 32)
        self.assertEqual(result['sequence'], 256)
        self.assertEqual(result['b32_ordinal'], 128)
        self.assertEqual(result['layers'], [])
        self.assertEqual(sampler.snapshot()['layers_expected'], 0)
        self.assertEqual(sampler.snapshot()['b32_seen'], 128)
        self.assertEqual(self.cuda.log.count('sync_end'), 1)
    def test_capture_replay_overwrites_events_reads_after_end_sync_and_retains_values(self):
        sampler = profile.GraphStageProfiler(self.torch, 1, 'weave', stride=1, layers=1)
        bound = object(); sampler.register_layer(0, bound)
        self.capture_layer(sampler, bound)
        self.assertTrue(sampler.begin_replay(37, 32))
        self.cuda.advance(.5); self.cuda.replay(); self.cuda.advance(.25)
        self.cuda.log.clear()
        first = sampler.finish_replay(37, 32)
        self.assertEqual(self.cuda.log[:3], ['event', 'sync_end', 'read'])
        self.assertEqual(first['whole_graph_ms'], 6.75)
        self.assertEqual(first['residual_ms'], .75)
        self.assertEqual(first['layers'][0], dict(layer=0, expert_pipeline_ms=2., slot_sum_ms=3., finalize_ms=1., routed_total_ms=6.))
        first_stamp = sampler.events_for(bound)[0].stamp
        self.cuda.advance(50); self.cuda.replay()  # unsampled replay overwrites timestamps
        self.assertNotEqual(first_stamp, sampler.events_for(bound)[0].stamp)
        self.assertEqual(sampler.snapshot()['samples'][0], first)
        first['layers'][0]['slot_sum_ms'] = 999
        self.assertEqual(sampler.snapshot()['samples'][0]['layers'][0]['slot_sum_ms'], 3.)
    def test_ring_baseline_sequence_and_native_outer_only(self):
        sampler = profile.GraphStageProfiler(self.torch, 0, 'native', stride=1, capacity=64)
        for sequence in range(1, 90):
            sampler.begin_replay(sequence, 32); self.cuda.advance(1); sampler.finish_replay(sequence, 32)
        snapshot = sampler.snapshot()
        self.assertEqual(len(snapshot['samples']), 64)
        self.assertEqual(snapshot['samples'][0]['sequence'], 26)
        self.assertEqual(snapshot['sampled_total'], 89)
        self.assertEqual([row['sequence'] for row in snapshot['samples'] if row['sequence'] > 85], [86,87,88,89])
        self.assertEqual(len(self.cuda.events), 2)
    def test_stale_layer_events_fail_even_when_durations_close(self):
        sampler = profile.GraphStageProfiler(self.torch, 0, 'weave', stride=1, layers=1)
        bound = object(); sampler.register_layer(0, bound); self.capture_layer(sampler, bound)
        self.cuda.replay(); self.cuda.advance(10)
        sampler.begin_replay(1, 32); self.cuda.advance(7)
        with self.assertRaisesRegex(RuntimeError, 'negative CUDA'):
            sampler.finish_replay(1, 32)
        sampler.abort_replay(1, 32)
        self.assertEqual(sampler.snapshot()['aborted_samples'], 1)
    def test_pending_mismatch_missing_layers_and_nonfinite_fail(self):
        sampler = profile.GraphStageProfiler(self.torch, 0, 'weave', stride=1, layers=1)
        with self.assertRaisesRegex(RuntimeError, 'all expected'):
            sampler.begin_replay(1, 32)
        sampler.register_layer(0, object())
        sampler.begin_replay(2, 32)
        with self.assertRaisesRegex(RuntimeError, 'idle'):
            sampler.snapshot()
        with self.assertRaisesRegex(RuntimeError, 'not finished'):
            sampler.begin_replay(3, 32)
        with self.assertRaisesRegex(RuntimeError, 'does not match'):
            sampler.finish_replay(3, 32)
        sampler.abort_replay(2, 32)
        with self.assertRaisesRegex(RuntimeError, 'nonfinite'):
            profile.GraphStageProfiler._duration(SimpleNamespace(elapsed_time=lambda _:math.nan), object())
    def test_source_helper_includes_separate_module_and_original_identity(self):
        self.assertEqual(len(profile.graph_stage_source_sha256()), 64)
        self.assertEqual(profile._LOADED_PROFILE_SOURCE_SHA256, profile.graph_stage_source_sha256())
        source = Path(profile.__file__).read_text()
        self.assertIn("'graph_runtime': graph_runtime_source_sha256()", source)
        self.assertNotEqual(profile.graph_stage_source_sha256(), profile.graph_runtime_source_sha256())


class StageRuntimeTest(unittest.TestCase):
    def setUp(self):
        from tests.cpu.test_hopper_graph import HopperGraphTest
        self.fixture = HopperGraphTest(); self.fixture.setUp(); self.addCleanup(self.fixture.doCleanups)
        self.cuda = FakeCuda(); self.fixture.torch.cuda = self.cuda
        self.sampler = profile.GraphStageProfiler(self.fixture.torch, 2, 'weave', stride=1, layers=1)
        self.runtime = profile.HopperStageDecodeRuntime.__new__(profile.HopperStageDecodeRuntime)
        self.runtime.__dict__.update(self.fixture.runtime.__dict__)
        self.runtime.profiler = self.sampler
        self.bound = self.fixture.bound
        self.runtime.register_layer(0, self.bound)
        self.patcher = patch.object(profile, 'finalize_slots', side_effect=lambda *args:self.fixture.events.append(('finalize',)))
        self.patcher.start(); self.addCleanup(self.patcher.stop)
    def run_batch(self, batch):
        from tests.cpu.test_staged_decode import Tensor
        return self.runtime.run(Tensor(batch,128),Tensor(batch,6),Tensor(batch,6),self.bound,7.)
    def test_b32_boundaries_wrap_inclusive_compute_sum_counter_finalize(self):
        order = []
        self.runtime.ws.run = lambda *args:order.append('compute')
        self.runtime.torch.distributed.all_reduce = lambda *args,**kwargs:order.append('sum')
        self.runtime.device_slot_collectives.add_ = lambda value:order.append('counter')
        events = self.sampler.events_for(self.bound)
        for index,event in enumerate(events):event.record = lambda index=index:order.append(index)
        with patch.object(profile,'finalize_slots',side_effect=lambda *args:order.append('finalize')):
            self.run_batch(32)
        self.assertEqual(order, [0,'compute',1,'sum',2,'counter','finalize',3])
        self.assertEqual(self.runtime.additional_slot_collectives, 1)
    def test_other_batches_delegate_unchanged_without_event_nodes(self):
        self.cuda.log.clear(); self.run_batch(7)
        self.assertEqual(self.cuda.log, [])
        self.assertEqual([event[0] for event in self.fixture.events], ['compute','reduce','counter','finalize'])
    def test_actual_replay_counter_runs_without_python_runtime(self):
        self.run_batch(32)
        counter = self.runtime.device_slot_collectives
        original = counter.add_
        def add(value):
            if self.cuda.capturing:self.cuda.operations.append(lambda: original(value))
            else:original(value)
        counter.add_ = add
        self.cuda.capturing = True; self.run_batch(32); self.cuda.capturing = False
        self.assertEqual(counter.value,1)
        for _ in range(4):self.cuda.replay()
        self.assertEqual(counter.value,5)
        self.assertEqual(self.runtime.additional_slot_collectives,1)


class SmokeSourceTest(unittest.TestCase):
    def test_gpu_case_matrix_identity_and_cleanup_contract(self):
        from tests.gpu import check_graph_stage_events as smoke
        self.assertEqual(smoke.expected_cases(), [dict(sequence=n,batch=32,variant=n) for n in range(1,5)])
        self.assertEqual(smoke.hashes()['profile_source_sha256'], profile.graph_stage_source_sha256())
        text = Path(smoke.__file__).read_text()
        self.assertIn('HopperStageDecodeRuntime(peer, group, profiler, experts=32)',text)
        self.assertIn('cleanup_graphs(owned_graphs, torch, dist, group, cleanup_progress)',text)
        self.assertIn("f'graph-cleanup-rank{rank}.json'", text)
        self.assertIn("numerical_policy='record_only'",text)


if __name__ == '__main__':
    unittest.main()
