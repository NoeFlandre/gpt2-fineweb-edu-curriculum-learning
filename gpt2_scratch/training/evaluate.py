"""Validation loss, overall and per FineWeb-Edu score bucket."""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from gpt2_scratch.data.shards import list_shards, load_shard

BUCKETS = ("25_30", "30_35", "35_40", "40_50")


def load_validation(directory: Path | str) -> tuple[np.ndarray, np.ndarray]:
    """The validation tokens and the score bucket of each token."""
    directory = Path(directory)
    tokens = load_shard(list_shards(directory, "val")[0])
    return tokens, np.load(directory / "val_buckets.npy")


@torch.no_grad()
def validation_by_bucket(
    model,
    tokens: np.ndarray,
    buckets: np.ndarray,
    device: str,
    batch_size: int,
    seq_len: int,
) -> dict[str, float]:
    """Mean loss over all validation tokens, and over the tokens of each score bucket.

    Tokens are scored in consecutive non-overlapping windows; a token belongs to the bucket
    of its own score. The model's training mode is restored afterwards.
    """
    was_training = model.training
    model.eval()
    losses = torch.zeros(len(BUCKETS), dtype=torch.float64)
    counts = torch.zeros(len(BUCKETS), dtype=torch.float64)
    windows = (len(tokens) - 1) // seq_len
    offsets = np.arange(seq_len)
    for begin in range(0, windows, batch_size):
        starts = np.arange(begin, min(begin + batch_size, windows)) * seq_len
        positions = starts[:, None] + offsets
        x = torch.from_numpy(tokens[positions].astype(np.int64)).to(device)
        y = torch.from_numpy(tokens[positions + 1].astype(np.int64)).to(device)
        target_buckets = torch.from_numpy(buckets[positions + 1].astype(np.int64)).reshape(-1)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, _ = model(x)
        per_token = (
            F.cross_entropy(
                logits.float().reshape(-1, logits.size(-1)), y.reshape(-1), reduction="none"
            )
            .cpu()
            .double()
        )
        losses.index_add_(0, target_buckets, per_token)
        counts.index_add_(0, target_buckets, torch.ones_like(per_token))
    model.train(was_training)
    result = {"val_loss": float(losses.sum() / counts.sum())}
    for index, name in enumerate(BUCKETS):
        mean = float(losses[index] / counts[index]) if counts[index] > 0 else float("nan")
        result[f"val_loss_{name}"] = mean
    return result
