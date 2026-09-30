"""Hand-computed oracles for token ownership, deduplicated dispatch and combine."""

import unittest

from src.weave.routing import TokenKey, TokenRoute, ExpertResult, build_dispatch_plan, combine_results


class RoutingTest(unittest.TestCase):
    def example(self):
        return build_dispatch_plan([
            TokenRoute(TokenKey(0, 0), (2, 3), (0.25, 0.75)),
            TokenRoute(TokenKey(1, 0), (0, 3), (0.5, 0.5)),
        ], num_experts=4, ep_size=2, top_k=2)

    def test_one_remote_token_retains_two_expert_tasks(self):
        plan = self.example()
        self.assertEqual(plan.dispatch_tokens[1], (TokenKey(0, 0), TokenKey(1, 0)))
        self.assertEqual(len(plan.tasks[1]), 3)
        self.assertEqual([task.local_expert for task in plan.tasks[1]], [0, 1, 1])
        self.assertEqual(plan.workloads[1].remote_pairs, 2)
        self.assertEqual(plan.workloads[1].remote_unique_tokens, 1)
        self.assertEqual(plan.workloads[1].local_pairs, 1)

    def test_combine_restores_distinct_origins_and_slot_weights(self):
        values = [
            ExpertResult(TokenKey(1, 0), 1, (6.0, 8.0)),
            ExpertResult(TokenKey(0, 0), 1, (12.0, 16.0)),
            ExpertResult(TokenKey(1, 0), 0, (2.0, 4.0)),
            ExpertResult(TokenKey(0, 0), 0, (4.0, 8.0)),
        ]
        self.assertEqual(combine_results(self.example(), values, hidden_size=2), {
            TokenKey(0, 0): (10.0, 14.0), TokenKey(1, 0): (4.0, 6.0),
        })

    def test_ep4_top6_has_one_payload_per_destination(self):
        plan = build_dispatch_plan([
            TokenRoute(TokenKey(0, 7), (0, 1, 64, 65, 128, 192), (0.25,) * 6)
        ])
        self.assertEqual([len(tokens) for tokens in plan.dispatch_tokens], [1, 1, 1, 1])
        self.assertEqual([len(tasks) for tasks in plan.tasks], [2, 2, 1, 1])

    def test_empty_ranks_and_empty_batch_are_preserved(self):
        plan = build_dispatch_plan([TokenRoute(TokenKey(0, 0), (0, 1), (0.5, 0.5))], top_k=2)
        self.assertEqual(plan.tasks[1:], ((), (), ()))
        empty = build_dispatch_plan([])
        self.assertEqual(empty.dispatch_tokens, ((), (), (), ()))
        self.assertEqual(combine_results(empty, [], hidden_size=2), {})

    def test_weights_are_not_renormalized(self):
        plan = build_dispatch_plan([TokenRoute(TokenKey(0, 0), (0, 1), (0.75, 0.75))], top_k=2)
        out = combine_results(plan, [ExpertResult(TokenKey(0, 0), i, (2.0,)) for i in (0, 1)], 1)
        self.assertEqual(out[TokenKey(0, 0)], (3.0,))

    def test_missing_or_duplicate_return_slot_is_rejected(self):
        one = ExpertResult(TokenKey(0, 0), 0, (1.0,))
        with self.assertRaisesRegex(ValueError, "missing"):
            combine_results(self.example(), [one], 1)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            combine_results(self.example(), [one, one], 1)

    def test_bad_routing_is_rejected_before_dispatch(self):
        good = TokenRoute(TokenKey(0, 0), (0, 1), (0.5, 0.5))
        with self.assertRaisesRegex(ValueError, "duplicate token"):
            build_dispatch_plan([good, good], top_k=2)
        for ids in ((-1, 1), (0, 256), (0, 0)):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                build_dispatch_plan([TokenRoute(TokenKey(0, 0), ids, (0.5, 0.5))], top_k=2)
        with self.assertRaises(ValueError):
            build_dispatch_plan([TokenRoute(TokenKey(4, 0), (0, 1), (0.5, 0.5))], top_k=2)
        with self.assertRaises(ValueError):
            build_dispatch_plan([TokenRoute(TokenKey(0, 0), (0, 1), (float('nan'), 0.5))], top_k=2)

    def test_wrong_return_width_or_identity_is_rejected(self):
        for result in (ExpertResult(TokenKey(0, 0), 0, ()), ExpertResult(TokenKey(0, 9), 0, (1.0,)), ExpertResult(TokenKey(0, 0), 2, (1.0,))):
            with self.subTest(result=result), self.assertRaises(ValueError):
                combine_results(self.example(), [result], 1)


if __name__ == '__main__':
    unittest.main()
