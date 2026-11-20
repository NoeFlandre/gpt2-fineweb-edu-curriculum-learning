"""Token shards: one NumPy file of uint16 token ids per shard."""

from pathlib import Path

import numpy as np


def save_shard(directory: Path | str, split: str, index: int, tokens: np.ndarray) -> Path:
    """Write one shard of tokens and return its path."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{split}_{index:06d}.npy"
    np.save(path, np.asarray(tokens, dtype=np.uint16))
    return path


def load_shard(path: Path | str) -> np.ndarray:
    """Read the tokens of one shard."""
    return np.load(path)


def list_shards(directory: Path | str, split: str) -> list[Path]:
    """Every shard of a split, sorted by name."""
    paths = sorted(Path(directory).glob(f"{split}_*.npy"))
    if not paths:
        raise FileNotFoundError(f"no {split} shards in {directory}")
    return paths
