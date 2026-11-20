"""Build the ordering study's data from FineWeb-Edu.

Writes the training pool in each of the four orders, and the held-out validation split with its
score buckets and their token counts.

Example: python -m scripts.prepare_data --out data/ordering
"""

import argparse
import json
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
import tiktoken
from datasets import load_dataset
from gpt2_scratch.data.corpus import (
    SCORE_EDGES,
    arm_tokens,
    bucket_ids,
    split_pool_and_validation,
    token_scores,
)
from gpt2_scratch.data.shards import save_shard
from gpt2_scratch.data.tokenize import encode_document
from gpt2_scratch.experiment.arms import ORDERINGS, arm_order
from gpt2_scratch.training.config import TrainConfig

EDU = "HuggingFaceFW/fineweb-edu"
VAL_TOKENS = 2_000_000
STREAM_BUFFER = 10_000


def build_study(
    rows: Iterable[dict],
    encode: Callable[[str], np.ndarray],
    out: Path,
    pool_tokens: int,
    val_tokens: int,
    seed: int,
) -> None:
    """Write each arm's token shard, and the validation shard with its score buckets."""
    pool, validation = split_pool_and_validation(rows, encode, pool_tokens, val_tokens)
    scores = np.array([doc.score for doc in pool])
    lengths = np.array([len(doc.tokens) for doc in pool])
    for arm in ORDERINGS:
        order = arm_order(arm, scores, lengths, pool_tokens, seed)
        save_shard(out / "arms" / arm, "train", 0, arm_tokens(pool, order, pool_tokens))
    validation_tokens = arm_tokens(validation, np.arange(len(validation)), val_tokens)
    per_token = token_scores(validation)[: len(validation_tokens)]
    buckets = bucket_ids(per_token)
    save_shard(out / "validation", "val", 0, validation_tokens)
    np.save(out / "validation" / "val_buckets.npy", buckets)
    report = {"edges": SCORE_EDGES.tolist(), "tokens": np.bincount(buckets, minlength=4).tolist()}
    (out / "validation" / "buckets.json").write_text(json.dumps(report))


def main() -> None:
    """Stream FineWeb-Edu and build the study's data."""
    cfg = TrainConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/ordering"))
    parser.add_argument("--pool-tokens", type=int, default=cfg.max_steps * cfg.total_batch_size)
    parser.add_argument("--val-tokens", type=int, default=VAL_TOKENS)
    parser.add_argument("--seed", type=int, default=cfg.seed)
    args = parser.parse_args()
    tokenizer = tiktoken.get_encoding("gpt2")
    rows = load_dataset(EDU, name="sample-10BT", split="train", streaming=True).shuffle(
        seed=args.seed, buffer_size=STREAM_BUFFER
    )
    build_study(
        rows,
        lambda text: encode_document(text, tokenizer),
        args.out,
        args.pool_tokens,
        args.val_tokens,
        args.seed,
    )


if __name__ == "__main__":
    main()
