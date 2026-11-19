"""Token shards: uint16 token ids with one uint8 educational score per token."""

from pathlib import Path

import numpy as np

SCORES_SUFFIX = ".scores.npy"


def save_shard(directory: Path, split: str, index: int, tokens, scores) -> Path:
    """Write a shard of tokens and their per-token scores.

    Args:
        directory: Folder to write into; created if missing.
        split: Split name, for example "train" or "val".
        index: Shard index within the split.
        tokens: Token ids.
        scores: Educational score of each token's document, 0 to 5.

    Returns:
        The path of the token file.

    Raises:
        ValueError: If tokens and scores have different lengths.
    """
    if len(tokens) != len(scores):
        raise ValueError("tokens and scores must have the same length")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{split}_{index:06d}.npy"
    np.save(path, np.asarray(tokens, dtype=np.uint16))
    np.save(_scores_path(path), np.asarray(scores, dtype=np.uint8))
    return path


def list_shards(directory: Path, split: str) -> list[Path]:
    """Return the token files of a split in order.

    Raises:
        FileNotFoundError: If the split has no shards.
    """
    paths = sorted(
        path
        for path in Path(directory).glob(f"{split}_*.npy")
        if not path.name.endswith(SCORES_SUFFIX)
    )
    if not paths:
        raise FileNotFoundError(f"no {split} shards in {directory}")
    return paths


def load_shard(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load the tokens and per-token scores of one shard."""
    return np.load(path), np.load(_scores_path(path))


def _scores_path(token_path: Path) -> Path:
    return token_path.with_name(token_path.stem + SCORES_SUFFIX)
