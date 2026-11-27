"""Build the ordering study's data from FineWeb-Edu.

The validation split is the same for every data seed. It is drawn once, from the stream of
VALIDATION_SEED after that stream's pool, and written to --validation (default data/validation).
Each data seed's training pool is drawn from its own stream, leaving out the validation documents.
It is written to --out (default data/data<data seed>) with each arm's token shard and a copy of the
validation split.

Example: python -m scripts.prepare_data --data-seed 43
"""

import argparse
import json
import shutil
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import tiktoken
from datasets import load_dataset
from gpt2_scratch.data.corpus import (
    BUCKET_COUNT,
    SCORE_EDGES,
    Document,
    arm_tokens,
    collect_documents,
    token_buckets,
    validation_after_pool,
)
from gpt2_scratch.data.shards import save_shard
from gpt2_scratch.data.tokenize import encode_document
from gpt2_scratch.experiment.arms import ORDERINGS, arm_order
from gpt2_scratch.experiment.naming import DATA_SEED
from gpt2_scratch.training.config import TrainConfig

EDU = "HuggingFaceFW/fineweb-edu"
VALIDATION_SEED = 42  # the validation split comes from this stream, whatever the data seed
VAL_TOKENS = 2_000_000
STREAM_BUFFER = 10_000
POOL_TOKENS = TrainConfig().max_steps * TrainConfig().total_batch_size


def stream(seed: int) -> Iterator[dict]:
    """The FineWeb-Edu sample, shuffled with this seed."""
    rows = load_dataset(EDU, name="sample-10BT", split="train", streaming=True)
    return iter(rows.shuffle(seed=seed, buffer_size=STREAM_BUFFER))


def save_validation(out: Path, validation: list[Document]) -> None:
    """Write the validation shard, its score buckets, and the keys of its documents.

    buckets.json is written last: its presence means the split is complete.
    """
    save_shard(out, "val", 0, np.concatenate([doc.tokens for doc in validation]))
    buckets = token_buckets(validation)
    np.save(out / "val_buckets.npy", buckets)
    (out / "doc_keys.txt").write_text("".join(f"{doc.key}\n" for doc in validation))
    counts = np.bincount(buckets, minlength=BUCKET_COUNT)
    report = {"edges": SCORE_EDGES.tolist(), "tokens": counts.tolist()}
    (out / "buckets.json").write_text(json.dumps(report))


def save_pool(out: Path, pool: list[Document], pool_tokens: int, data_seed: int) -> None:
    """Write each arm's training shard: the pool in that arm's order."""
    shard_tokens = pool_tokens + 1
    scores = np.array([doc.score for doc in pool])
    lengths = np.array([len(doc.tokens) for doc in pool])
    for arm in ORDERINGS:
        order = arm_order(arm, scores, lengths, pool_tokens, data_seed)
        save_shard(out / "arms" / arm, "train", 0, arm_tokens(pool, order, shard_tokens))


def main() -> None:
    """Build the validation split if it is missing, then this data seed's pool."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-seed",
        type=int,
        default=DATA_SEED,
        help="seed of the training pool, its orders and its stream",
    )
    parser.add_argument("--out", type=Path, help="default: data/data<data seed>")
    parser.add_argument(
        "--validation",
        type=Path,
        default=Path("data/validation"),
        help="the fixed validation split",
    )
    parser.add_argument("--pool-tokens", type=int, default=POOL_TOKENS)
    args = parser.parse_args()
    out = args.out or Path(f"data/data{args.data_seed}")
    tokenizer = tiktoken.get_encoding("gpt2")

    def encode(text: str) -> np.ndarray:
        return encode_document(text, tokenizer)

    if not (args.validation / "buckets.json").exists():
        # The default pool budget (plus one token, as the shards are built) fixes the validation
        # split, whatever --pool-tokens is.
        validation = validation_after_pool(
            stream(VALIDATION_SEED), encode, POOL_TOKENS + 1, VAL_TOKENS
        )
        save_validation(args.validation, validation)
    skip = frozenset((args.validation / "doc_keys.txt").read_text().split())
    pool = collect_documents(stream(args.data_seed), encode, args.pool_tokens + 1, skip)
    save_pool(out, pool, args.pool_tokens, args.data_seed)
    shutil.copytree(args.validation, out / "validation")


if __name__ == "__main__":
    main()
