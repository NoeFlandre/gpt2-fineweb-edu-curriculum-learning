"""Write the multiple-choice benchmarks of the study as JSON Lines files.

HellaSwag is already in data/benchmarks/hellaswag.jsonl. This writes PIQA and ARC-Easy.

Example: python -m scripts.prepare_benchmarks --out data/benchmarks
"""

import argparse
import io
import json
import urllib.request
import zipfile
from pathlib import Path

from datasets import load_dataset

# The validation split of ybisk/piqa. That dataset is a loading script, which the datasets
# library no longer runs, so this reads the files the script downloads: dev.jsonl and its labels.
PIQA_ZIP = "https://storage.googleapis.com/ai2-mosaic/public/physicaliqa/physicaliqa-train-dev.zip"
PIQA_FOLDER = "physicaliqa-train-dev"


def piqa_example(row: dict) -> dict:
    """A PIQA question with its two solutions."""
    return {"ctx": row["goal"], "endings": [row["sol1"], row["sol2"]], "label": row["label"]}


def piqa_validation(archive: zipfile.ZipFile) -> list[dict]:
    """The PIQA validation examples: each dev.jsonl row with the label on the same line."""
    rows = archive.read(f"{PIQA_FOLDER}/dev.jsonl").decode().splitlines()
    labels = archive.read(f"{PIQA_FOLDER}/dev-labels.lst").decode().splitlines()
    return [
        piqa_example({**json.loads(row), "label": int(label)})
        for row, label in zip(rows, labels, strict=True)
    ]


def download_piqa() -> zipfile.ZipFile:
    """Download the PIQA zip from the location that ybisk/piqa declares."""
    with urllib.request.urlopen(PIQA_ZIP, timeout=120) as response:
        return zipfile.ZipFile(io.BytesIO(response.read()))


def arc_example(row: dict) -> dict:
    """An ARC question with its choices; the label is the position of the answer."""
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


def write_benchmarks(out: Path) -> None:
    """Download PIQA (validation) and ARC-Easy (test) and write them into out."""
    write_examples(out / "piqa.jsonl", piqa_validation(download_piqa()))
    arc = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
    write_examples(out / "arc_easy.jsonl", [arc_example(row) for row in arc])


def main() -> None:
    """Read the output folder and write the benchmarks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/benchmarks"))
    write_benchmarks(parser.parse_args().out)


if __name__ == "__main__":
    main()
