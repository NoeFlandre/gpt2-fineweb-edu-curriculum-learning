"""Build the token shards the arms train on.

- raw/train: FineWeb documents, each scored by the FineWeb-Edu classifier.
- raw/val: validation documents taken after the training documents.
- edu/train: FineWeb-Edu documents with their published scores.

Example: python -m scripts.prepare_data --out data/arms
"""

import argparse
from collections.abc import Iterator
from pathlib import Path
from typing import Protocol

import numpy as np
import tiktoken
from datasets import load_dataset
from gpt2_scratch.data.build import build_arrays, take_documents
from gpt2_scratch.data.shards import save_shard
from gpt2_scratch.data.tokenize import encode_document
from gpt2_scratch.experiment.scoring import EducationalScorer

SCORE_BATCH = 64


class TextScorer(Protocol):
    """Anything that maps texts to integer educational scores."""

    def score(self, texts: list[str]) -> np.ndarray:
        """Return one integer score per text."""


FINEWEB = "HuggingFaceFW/fineweb"
FINEWEB_EDU = "HuggingFaceFW/fineweb-edu"


def parse_args() -> argparse.Namespace:
    """Read the command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/arms"))
    parser.add_argument("--train-tokens", type=int, default=100_000_000)
    parser.add_argument("--val-tokens", type=int, default=2_000_000)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def stream_rows(repo: str) -> Iterator[dict]:
    """Yield the rows of the sample-10BT config of a dataset, without downloading it all."""
    yield from load_dataset(repo, name="sample-10BT", split="train", streaming=True)


def score_texts(scorer: TextScorer, texts: list[str]) -> np.ndarray:
    """Classifier score of every text, computed in batches."""
    batches = []
    for start in range(0, len(texts), SCORE_BATCH):
        batches.append(scorer.score(texts[start : start + SCORE_BATCH]))
        if len(batches) % 200 == 0:
            print(f"scored {start + SCORE_BATCH} of {len(texts)} documents", flush=True)
    return np.concatenate(batches) if batches else np.zeros(0, dtype=np.uint8)


def main() -> None:
    """Build the raw and edu shards."""
    args = parse_args()
    encoder = tiktoken.get_encoding("gpt2")

    def encode(text: str) -> np.ndarray:
        """Encode one document with its end-of-text prefix."""
        return encode_document(text, encoder)

    scorer = EducationalScorer.from_pretrained(args.device)
    raw = (row["text"] for row in stream_rows(FINEWEB))
    print("building raw train shard", flush=True)
    arrays, texts = take_documents(raw, args.train_tokens, encode)
    tokens, scores = build_arrays(arrays, score_texts(scorer, texts).tolist(), args.train_tokens)
    save_shard(args.out / "raw", "train", 0, tokens, scores)

    print("building raw validation shard", flush=True)
    arrays, texts = take_documents(raw, args.val_tokens, encode)
    tokens, scores = build_arrays(arrays, score_texts(scorer, texts).tolist(), args.val_tokens)
    save_shard(args.out / "raw", "val", 0, tokens, scores)

    edu_scores: list[int] = []

    def edu_texts() -> Iterator[str]:
        """Yield FineWeb-Edu texts and remember their published scores."""
        for row in stream_rows(FINEWEB_EDU):
            edu_scores.append(row["int_score"])
            yield row["text"]

    print("building edu train shard", flush=True)
    arrays, _ = take_documents(edu_texts(), args.train_tokens, encode)
    tokens, scores = build_arrays(arrays, edu_scores[: len(arrays)], args.train_tokens)
    save_shard(args.out / "edu", "train", 0, tokens, scores)


if __name__ == "__main__":
    main()
