"""CPU orchestration and interval tests, not evidence of CUDA execution."""
from types import SimpleNamespace
import unittest

from src.weave import native_p0 as p0
from tests.cpu.test_graph_stage_profile import FakeCuda


class HostBuffer:
    def __init__(self, cuda, values):
        self.cuda, self.values = cuda, values

    def detach(self):
        return self

    def cpu(self):
        if 'sync_end' not in self.cuda.log:
            raise AssertionError('metadata copied before graph completion')
        self.cuda.log.append('copy')
        return self

    def reshape(self, *_):
        return self

    def tolist(self):
        return self.values


class NativeP0Test(unittest.TestCase):
    def setUp(self):
        self.cuda = FakeCuda()
        self.sampler = p0.NativeP0Profiler(SimpleNamespace(cuda=self.cuda), 0,
                                         stride=1, capacity=2, layers=1)
        self.sampler.register_layer(0, {'path': 'observed'})

    def capture(self):
        self.cuda.capturing = True
        with self.sampler.span(0, 'moe'):
            with self.sampler.span(0, 'routed'):
                self.cuda.advance(3)
            with self.sampler.span(0, 'shared'):
                self.cuda.advance(2)
            with self.sampler.span(0, 'shared_merge'):
                self.cuda.advance(1)
        self.cuda.capturing = False

    def sample(self, sequence):
        self.assertTrue(self.sampler.begin_replay(sequence, 32))
        self.cuda.advance(.5)
        self.cuda.replay()
        self.cuda.advance(.25)
        return self.sampler.finish_replay(sequence, 32)

    def test_capture_only_external_events_and_optional_absence(self):
        count = len(self.cuda.events)
        with self.sampler.span(0, 'router'):
            pass
        self.capture()
        self.assertEqual(len(self.cuda.events), count)
        self.assertTrue(all(event.options['external'] for event in self.cuda.events[2:]))
        row = self.sample(1)['layers']['0']
        self.assertNotIn('router', row['intervals'])
        self.assertEqual(row['moe_ms'], 6)
        self.assertEqual(row['branch_overlap_ms'], 0)
        self.assertEqual(row['shared_tail_ms'], 2)
        self.assertEqual(row['moe_post_branches_ms'], 1)
        self.assertEqual(row['intervals']['shared_merge']['start_ms'], 5.5)
        self.assertEqual(self.cuda.log.count('sync_end'), 1)

    def test_overlapping_intervals_are_intersection_not_duration_sum(self):
        intervals = {'moe': {'start_ms': 0, 'end_ms': 10, 'duration_ms': 10},
                     'routed': {'start_ms': 1, 'end_ms': 7, 'duration_ms': 6},
                     'shared': {'start_ms': 2, 'end_ms': 9, 'duration_ms': 7}}
        self.assertEqual(p0.derive_layer_metrics(intervals),
                         dict(moe_ms=10, branch_overlap_ms=5, shared_tail_ms=2,
                              routed_tail_ms=0, branches_done_ms=9, moe_post_branches_ms=1))
        self.assertEqual(p0.derive_layer_metrics({'moe': intervals['moe']}), {'moe_ms': 10})

    def test_stride_counts_only_b32_and_retains_global_sequence(self):
        self.sampler.stride = 2
        self.capture()
        self.assertFalse(self.sampler.begin_replay(1, 7))
        self.assertFalse(self.sampler.begin_replay(2, 32))
        result = self.sample(4)
        self.assertEqual(result['sequence'], 4)
        self.assertEqual(result['b32_ordinal'], 2)
        self.assertEqual(result['whole_graph_ms'], 6.75)
        with self.assertRaisesRegex(ValueError, 'sequence'):
            self.sampler.begin_replay(4, 32)

    def test_stale_events_and_pending_reads_rejected(self):
        self.capture()
        self.sample(1)
        self.cuda.advance(20)
        self.sampler.begin_replay(2, 32)
        with self.assertRaisesRegex(RuntimeError, 'idle'):
            self.sampler.snapshot()
        with self.assertRaisesRegex(RuntimeError, 'stale|window'):
            self.sampler.finish_replay(2, 32)
        self.sampler.abort_replay(2, 32)
        self.assertEqual(self.sampler.snapshot()['aborted_samples'], 1)

    def test_ring_snapshots_do_not_alias_and_event_reads_follow_sync(self):
        self.capture()
        first = self.sample(1)
        self.sample(2)
        self.sample(3)
        snapshot = self.sampler.snapshot()
        self.assertEqual([x['sequence'] for x in snapshot['samples']], [2, 3])
        snapshot['samples'][0]['layers'].clear()
        self.assertTrue(self.sampler.snapshot()['samples'][0]['layers'])
        self.assertEqual(first['sequence'], 1)
        self.assertLess(self.cuda.log.index('sync_end'), self.cuda.log.index('read'))

    def test_grouping_excludes_remote_groups_and_padding_and_never_exports_ids(self):
        self.cuda.capturing = True
        self.sampler.bind_grouping(0, HostBuffer(self.cuda, [0, 1, 99, 99, 2, 3, 4, 99, 8, 99, 99, 99]),
                                  HostBuffer(self.cuda, [2, -1, 2]), HostBuffer(self.cuda, [12]), 4, 10)
        self.cuda.capturing = False
        self.capture()
        row = self.sample(1)['layers']['0']['grouping']
        self.assertEqual(row, dict(valid_local_pairs=3, valid_groups=2, active_experts=1,
                                   max_rows=3, row_histogram={'3': 1}, num_tokens_post_pad=12,
                                   block_size=4, valid_pair_capacity=10))
        self.assertNotIn('sorted_token_ids', repr(self.sampler.snapshot()))

    def test_registration_during_capture_and_incomplete_layer_capture_rejected(self):
        self.cuda.capturing = True
        with self.assertRaisesRegex(RuntimeError, 'before'):
            self.sampler.register_layer(0, {})
        self.cuda.capturing = False
        with self.assertRaisesRegex(RuntimeError, 'moe'):
            self.sampler.begin_replay(1, 32)

    def test_trace_wraps_first_sample_and_marks_its_extra_overhead(self):
        cuda = self.cuda
        class Trace:
            def begin(self):
                cuda.log.append('trace_begin')
            def finish(self):
                cuda.log.append('trace_finish')
                return {'status': 'complete', 'kernel_count': 8}
            def abort(self):
                cuda.log.append('trace_abort')
        self.sampler.trace = Trace()
        self.capture()
        self.cuda.log.clear()
        first = self.sample(1)
        second = self.sample(2)
        self.assertTrue(first['launch_trace_active'])
        self.assertFalse(second['launch_trace_active'])
        self.assertEqual(self.cuda.log[0], 'trace_begin')
        self.assertEqual(self.cuda.log.count('trace_finish'), 1)
        self.assertLess(self.cuda.log.index('sync_end'), self.cuda.log.index('trace_finish'))
        self.assertEqual(self.sampler.snapshot()['launch_trace']['kernel_count'], 8)

    def test_aborting_trace_clears_pending_state(self):
        calls = []
        self.sampler.trace = SimpleNamespace(begin=lambda: calls.append('begin'),
                                            abort=lambda: calls.append('abort'))
        self.capture()
        self.sampler.begin_replay(1, 32)
        self.sampler.abort_replay(1, 32)
        self.assertEqual(calls, ['begin', 'abort'])
        self.assertEqual(self.sampler.snapshot()['launch_trace']['status'], 'aborted')
        self.assertEqual(self.sampler.snapshot()['aborted_samples'], 1)


if __name__ == '__main__':
    unittest.main()
