"""Write the multiple-choice benchmarks of the study as JSON Lines files.

HellaSwag is already local (data/benchmarks/hellaswag.jsonl). This writes PIQA and ARC-Easy.

Example: python -m scripts.prepare_benchmarks --out data/benchmarks
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset


def piqa_example(row: dict) -> dict:
    """A PIQA question with its two solutions."""
    return {"ctx": row["goal"], "endings": [row["sol1"], row["sol2"]], "label": row["label"]}


def arc_example(row: dict) -> dict:
    """An ARC question with its choices, the answer being the position of its letter."""
    choices = row["choices"]
    return {
        "ctx": f"Question: {row['question']}\nAnswer:",
        "endings": list(choices["text"]),
        "label": choices["label"].index(row["answerKey"]),
    }


def write_examples(path: Path, examples: list[dict]) -> None:
    """Write one example per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(example) + "\n" for example in examples))


def main() -> None:
    """Download PIQA (validation) and ARC-Easy (test) and write them."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/benchmarks"))
    args = parser.parse_args()
    piqa = load_dataset("baber/piqa", split="validation")
    write_examples(args.out / "piqa.jsonl", [piqa_example(row) for row in piqa])
    arc = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
    write_examples(args.out / "arc_easy.jsonl", [arc_example(row) for row in arc])


if __name__ == "__main__":
    main()
