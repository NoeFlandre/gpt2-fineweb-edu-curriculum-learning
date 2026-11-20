"""Position-wise feed-forward network."""

import torch
import torch.nn as nn

from gpt2_scratch.model.config import GPT2Config


class MLP(nn.Module):
    """Expand to four times the width, apply GELU, then project back."""

    def __init__(self, config: GPT2Config) -> None:
        """Create the two linear layers."""
        super().__init__()
        self.l1 = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.l2 = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the feed-forward network to every position."""
        return self.l2(self.gelu(self.l1(x)))
