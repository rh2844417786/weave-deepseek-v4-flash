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

    Config files may differ from the publisher's pin, as permitted by the
    original preparer, but their observed digest and match flag are required.
    Tokenizer and encoder files must match the exact pins.
    """
    path=Path(path)
    manifest=json.loads((path.parent/'manifest.json').read_text())
    root=Path(__file__).resolve().parents[2]
    pinned=json.loads((root/'data/gpqa/deepseek_model_manifest.json').read_text())
    sources=json.loads((root/'data/gpqa/source_manifest.json').read_text())
    main=[record for record in sources['records'] if Path(record['path']).name=='gpqa_main.csv']
    if len(main)!=1 or main[0]['rows']!=448:
        raise ValueError('one pinned 448-row GPQA Main source required')
    observed=manifest.get('observed_model_metadata_sha256')
    if (manifest.get('model_id')!=pinned['model_id']
            or manifest.get('model_revision')!=pinned['revision']
            or manifest.get('pinned_model_metadata_sha256')!=pinned['sha256']
            or not isinstance(observed,dict)):
        raise ValueError('pinned DeepSeek tokenizer/Think High encoder provenance required')
    for name,expected in pinned['sha256'].items():
        actual=observed.get(name)
        valid=isinstance(actual,str) and len(actual)==64 and all(c in '0123456789abcdef' for c in actual)
        if not valid or (name!='config.json' and actual!=expected):
            raise ValueError(f'observed DeepSeek metadata SHA256 mismatch or missing: {name}')
    if manifest.get('config_matches_pinned') is not (observed['config.json']==pinned['sha256']['config.json']):
        raise ValueError('observed config match flag inconsistent')
    if (manifest.get('gpqa_main_sha256')!=main[0]['sha256']
            or manifest.get('count')!=448 or manifest.get('order_seed')!=42
            or manifest.get('reasoning_mode')!='Think High' or manifest.get('thinking_mode')!='thinking'
            or manifest.get('reasoning_effort')!='high' or manifest.get('max_new_tokens')!=32768
            or manifest.get('normal_eos_allowed') is not True
            or manifest.get('record_sha256')!=hashlib.sha256(path.read_bytes()).hexdigest()):
        raise ValueError('hash-verified pinned 448-row seed42 Think High GPQA manifest required')
    records=[json.loads(line) for line in path.read_text().splitlines()]
    if len(records)!=448 or [record['row_index'] for record in records]!=ordered_indices(448):
        raise ValueError('448 unique original GPQA row identities required')
    for record in records:
        ids=record['input_token_ids']
        if not isinstance(ids,list) or not ids or any(type(v) is not int or v<0 for v in ids) or record.get('input_tokens',len(ids))!=len(ids):
            raise ValueError('natural prepared input token count inconsistent')
    return records,manifest
