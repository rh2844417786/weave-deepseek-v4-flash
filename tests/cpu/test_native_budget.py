import math
import unittest
from src.weave.native_budget import select_workers, state_row, STATE_SIZE


class PolicyTests(unittest.TestCase):
    def test_routing_changes_same_shape_without_exceeding_native_grid(self):
        choices = [select_workers(528, local, 192, 30, 32, 112) for local in (12, 48, 84)]
        self.assertEqual([x[0] for x in choices], [132, 396, 528])
        self.assertEqual([x[1] for x in choices], [2, 6, 8])

    def test_empty_local_and_full_control(self):
        self.assertEqual(select_workers(528, 0, 192, 0, 32, 112), (0, 0))
        self.assertEqual(select_workers(528, 1, 192, 1, 32, 112, mode='full'), (528, 8))
        self.assertEqual(select_workers(528, 1, 6, 1, 2, 2), (4, 1))

    def test_stripes_cover_work_exactly_for_all_selected_workers(self):
        for grid in (4, 33, 132, 528):
            for routes in (1, 3, 12, 48, 96, 192):
                for nk in ((1, 1), (16, 64), (32, 112)):
                    blocks = min(routes, 64)
                    count, bucket = select_workers(grid, routes, 192, blocks, *nk)
                    work = blocks * math.prod(nk)
                    for alignment in (1, 2, 4):
                        iters = math.ceil(math.ceil(work / count) / alignment) * alignment
                        spans = [(i * iters, min((i + 1) * iters, work))
                                 for i in range(count) if i * iters < work]
                        self.assertEqual(spans[0][0], 0)
                        self.assertEqual(spans[-1][1], work)
                        self.assertTrue(all(a[1] == b[0] for a, b in zip(spans, spans[1:])))
                        self.assertLessEqual(count, grid)

    def test_counter_report_names_are_aggregate_only(self):
        state = [0] * STATE_SIZE
        state[0:8] = [10, 2, 528, 396, 48, 30, 32, 112]
        state[8] = 2; state[14] = 8
        state[17:19] = [48, 192]
        row = state_row(2, 'gate_up', state)
        self.assertEqual(row['calls'], sum(row['bucket_counts']))
        self.assertEqual(row['last_local_routes'], 48)
        self.assertEqual(row['last_total_routes'], 192)

    def test_invalid_configuration_and_routes_rejected(self):
        for kwargs in ({'target_percent':0}, {'mode':'guess'}, {'ep_size':0}):
            with self.assertRaises(ValueError):select_workers(528,1,192,1,32,112,**kwargs)
        with self.assertRaises(ValueError):select_workers(528,193,192,1,32,112)
