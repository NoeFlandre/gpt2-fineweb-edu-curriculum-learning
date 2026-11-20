"""Sequential batches of inputs and targets read from token shards."""

from pathlib import Path

import numpy as np
import torch

from gpt2_scratch.data.shards import list_shards, load_shard


class TokenLoader:
    """Yield consecutive (batch, sequence) batches, moving to the next shard when one runs out."""

    def __init__(self, directory: Path, split: str, batch_size: int, seq_len: int) -> None:
        """Open the first shard of a split.

        Args:
            directory: Folder holding the shards.
            split: Split name.
            batch_size: Sequences per batch.
            seq_len: Tokens per sequence.
        """
        self.shards = list_shards(directory, split)
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.reset()

    def reset(self) -> None:
        """Restart from the first token of the first shard."""
        self._open(0)

    def state(self) -> dict:
        """Where the next batch starts, so a resumed run reads the same data."""
        return {"shard_index": self.shard_index, "position": self.position}

    def restore(self, state: dict) -> None:
        """Continue from a position saved by state()."""
        self._open(state["shard_index"])
        self.position = state["position"]

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return inputs and their next-token targets."""
        span = self.batch_size * self.seq_len
        if self.position + span + 1 > len(self.tokens):
            self._open((self.shard_index + 1) % len(self.shards))
        start = self.position
        end = start + span + 1
        tokens = torch.from_numpy(self.tokens[start:end].astype(np.int64))
        self.position = start + span
        shape = (self.batch_size, self.seq_len)
        return tokens[:-1].view(shape), tokens[1:].view(shape)

    def _open(self, index: int) -> None:
        self.shard_index = index
        self.tokens = load_shard(self.shards[index])
        self.position = 0
