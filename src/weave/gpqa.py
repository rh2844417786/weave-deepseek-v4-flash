"""Deterministic GPQA Main question and choice preparation."""

import random


PROMPT_PREFIX = (
    "Answer the following multiple-choice question. Explain your reasoning briefly, "
    "then state the final answer as A, B, C, or D.\n\n"
)


def build_message(row: dict, original_index: int) -> str:
    """Use only the question and the four answer texts, with no gold label."""
    answers = [row["Correct Answer"]] + [row[f"Incorrect Answer {i}"] for i in (1, 2, 3)]
    random.Random(42 + original_index).shuffle(answers)
    return PROMPT_PREFIX + row["Question"].strip() + "\n\n" + "\n".join(
        f"{letter}. {answer.strip()}" for letter, answer in zip("ABCD", answers)
    )


def ordered_indices(count: int) -> list:
    """Return source row indices in the fixed benchmark order."""
    if count < 0:
        raise ValueError("count must be nonnegative")
    indices = list(range(count))
    random.Random(42).shuffle(indices)
    return indices
