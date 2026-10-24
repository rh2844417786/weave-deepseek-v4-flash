import unittest

class TailTimingTests(unittest.TestCase):
    def test_actual_overlap_does_not_fill_gaps_or_double_count_workers(self):
        from src.weave.native_tail_timing import interval_overlap_us
        self.assertEqual(interval_overlap_us([(1000,2000),(10000,11000)],[(4000,5000)]),0)
        self.assertEqual(interval_overlap_us([(1000,5000),(2000,6000)],[(3000,7000),(4000,8000)]),3.)

    def test_late_publication_explains_no_overlap_on_one_gpu(self):
        from src.weave.native_tail_timing import snapshot_timing
        state=dict(start_ns=1000,planner_begin_ns=1100,planner_end_ns=3100,planned_ns=3200,
            completion_begin_ns=11000,finish_ns=12000,local_groups=1,
            chunk_published_ns=[7100,8100,0,0],
            chunk_times_ns=[[3200,4000,4000,7000,7200,8000,8000,9000],
                            [4000,5000,5000,8000,8200,9000,9000,10000], [0]*8,[0]*8],
            events=[[0,3200,4000,1],[1,4000,5000,1]]+
              [[0,7200,8000,2],[0,8200,9000,2]]+
              [[0,8000,9000,3]]*4+[[0,9000,10000,3]]*4+[[0]*4]*10)
        row=snapshot_timing(state,chunks=2,tiles=2,capacity=1)
        self.assertEqual(row['planner_us'],2.)
        self.assertEqual(row['chunk0_finalize_end_after_last_down_us'],2.)
        self.assertEqual(row['chunk0_publish_after_last_down_us'],2.1)
        self.assertEqual(row['first_peer_task_after_last_down_us'],2.2)
        self.assertEqual(row['chunk0_communication_overlap_us'],0.)
        self.assertEqual(row['measurement_scope'],'same_gpu_diagnostic_replay')
        state['local_groups']=0
        state['events'][:2]=[[0]*4]*2
        row=snapshot_timing(state,chunks=2,tiles=2,capacity=1)
        self.assertIsNone(row['first_peer_task_after_last_down_us'])
