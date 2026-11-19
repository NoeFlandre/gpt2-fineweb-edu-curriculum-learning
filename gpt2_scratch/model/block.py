"""Transformer block: attention and a feed-forward network, each with a residual connection."""

import torch
import torch.nn as nn

from gpt2_scratch.model.attention import CausalSelfAttention
from gpt2_scratch.model.config import GPT2Config
from gpt2_scratch.model.mlp import MLP


class Block(nn.Module):
    """Pre-norm transformer block."""

    def __init__(self, config: GPT2Config) -> None:
        """Create the two layer norms, the attention and the feed-forward network."""
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply attention, then the feed-forward network, each added back to the input."""
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))
