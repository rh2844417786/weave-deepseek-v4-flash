#!/usr/bin/env python3
"""Count natural GPQA prompt lengths without printing or exporting examples."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import random
from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer

PROMPT_PREFIX = (
    "Answer the following multiple-choice question. Explain your reasoning briefly, "
    "then state the final answer as A, B, C, or D.\n\n"
)


def build_message(row, index):
    answers = [row["Correct Answer"]] + [row[f"Incorrect Answer {i}"] for i in (1, 2, 3)]
    random.Random(42 + index).shuffle(answers)
    return PROMPT_PREFIX + row["Question"].strip() + "\n\n" + "\n".join(
        f"{letter}. {answer.strip()}" for letter, answer in zip("ABCD", answers)
    )


def percentile(values, p):
    ordered = sorted(values)
    index = (len(ordered) - 1) * p
    lo, hi = math.floor(index), math.ceil(index)
    return round(ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo), 2)


def stats(values):
    return {
        "count": len(values), "min": min(values), "median": percentile(values, 0.5),
        "p95": percentile(values, 0.95), "max": max(values),
        "mean": round(sum(values) / len(values), 2),
        "at_least_2048": sum(n >= 2048 for n in values),
        "at_least_4096": sum(n >= 4096 for n in values),
        "at_least_8192": sum(n >= 8192 for n in values),
    }


def raise_exception(message):
    raise ValueError(message)


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=root / "data/gpqa")
    parser.add_argument("--tokenizer-dir", type=Path, default=root / "data/tokenizers/Qwen3.5-35B-A3B")
    parser.add_argument("--output", type=Path, default=root / "data/gpqa/length_audit.json")
    args = parser.parse_args()
    tokenizer = Tokenizer.from_file(str(args.tokenizer_dir / "tokenizer.json"))
    template_bytes = (args.tokenizer_dir / "chat_template.jinja").read_bytes()
    environment = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])
    environment.globals["raise_exception"] = raise_exception
    template = environment.from_string(template_bytes.decode())
    result = {
        "tokenizer_id": "Qwen/Qwen3.5-35B-A3B",
        "tokenizer_revision": "59d61f3ce65a6d9863b86d2e96597125219dc754",
        "tokenizer_sha256": hashlib.sha256((args.tokenizer_dir / "tokenizer.json").read_bytes()).hexdigest(),
        "chat_template_sha256": hashlib.sha256(template_bytes).hexdigest(),
        "libraries": {name: importlib.metadata.version(name) for name in ("tokenizers", "jinja2")},
        "method": "Official Jinja chat template + tokenizers.encode(add_special_tokens=False); single user message; no system message; no few-shot examples; no padding or truncation",
        "input_columns": ["Question", "Correct Answer", "Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3"],
        "explanation_and_gold_label_in_prompt": False,
        "answer_choice_seed": "42 + zero-based CSV row index",
        "instruction": PROMPT_PREFIX.strip(),
        "prompt_modes": {},
    }
    for thinking in (False, True):
        mode = "thinking_on" if thinking else "thinking_off"
        result["prompt_modes"][mode] = {}
        for split in ("main", "diamond", "extended"):
            path = args.data_root / "raw" / f"gpqa_{split}.csv"
            with path.open(newline="", encoding="utf-8-sig") as stream:
                rows = list(csv.DictReader(stream))
            counts = []
            for index, row in enumerate(rows):
                rendered = template.render(messages=[{"role": "user", "content": build_message(row, index)}], add_generation_prompt=True, enable_thinking=thinking, tools=None)
                suffix = "<|im_start|>assistant\n<think>\n" if thinking else "<|im_start|>assistant\n<think>\n\n</think>\n\n"
                if not rendered.endswith(suffix):
                    raise ValueError(f"Unexpected generation suffix for {mode}; inspect the official template")
                counts.append(len(tokenizer.encode(rendered, add_special_tokens=False).ids))
            result["prompt_modes"][mode][split] = stats(counts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
