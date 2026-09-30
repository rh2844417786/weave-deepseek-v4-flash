"""Protect GPQA prompt content and deterministic ordering."""

import unittest

from src.weave.gpqa import build_message, ordered_indices


class GPQAPrepareTest(unittest.TestCase):
    def test_choices_use_original_csv_index_without_leaking_gold_fields(self):
        row = {
            "Question": "  Which answer?  ",
            "Correct Answer": "correct",
            "Incorrect Answer 1": "wrong1",
            "Incorrect Answer 2": "wrong2",
            "Incorrect Answer 3": "wrong3",
            "Explanation": "private gold explanation",
        }
        message = build_message(row, original_index=3)
        self.assertIn("Which answer?\n\nA. correct\nB. wrong3\nC. wrong1\nD. wrong2", message)
        self.assertNotIn("private gold explanation", message)
        self.assertNotIn("Correct Answer", message)

    def test_seed_42_fixes_question_order(self):
        self.assertEqual(ordered_indices(5), [3, 1, 2, 4, 0])
        self.assertEqual(ordered_indices(5), ordered_indices(5))


if __name__ == "__main__":
    unittest.main()
