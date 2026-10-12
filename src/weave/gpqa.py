"""Deterministic GPQA Main question and choice preparation."""

import random
import hashlib
import json
from pathlib import Path


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


def load_prepared_records(path):
    """Validate saved offline provenance before acceptance or formal data reuse.

    The manifest carries the integrity contract: the record hash must match the
    prepared file, every row must hold a valid natural token-id list, and row
    identities must be unique. Dataset-specific provenance pins are recorded by
    the preparer but are not required here, so one runner can serve any pinned
    prepared set (GPQA Main, SWE-Bench, ...).
    """
    path=Path(path)
    manifest=json.loads((path.parent/'manifest.json').read_text())
    if manifest.get('record_sha256')!=hashlib.sha256(path.read_bytes()).hexdigest():
        raise ValueError('record_sha256 does not match the prepared records file')
    records=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not records:
        raise ValueError('prepared records must not be empty')
    seen=set()
    for record in records:
        ids=record.get('input_token_ids')
        if (not isinstance(ids,list) or not ids or any(type(v) is not int or v<0 for v in ids)
                or record.get('input_tokens',len(ids))!=len(ids)):
            raise ValueError('natural prepared input token count inconsistent')
        row=record.get('row_index')
        if type(row) is not int or row in seen:
            raise ValueError('unique integer prepared row identities required')
        seen.add(row)
    return records,manifest
