"""Causal multi-head self-attention."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from gpt2_scratch.model.config import GPT2Config


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention that only looks at earlier positions."""

    def __init__(self, config: GPT2Config) -> None:
        """Create the query, key and value projection and the output projection."""
        super().__init__()
        assert config.n_embd % config.n_head == 0, "embedding size must split evenly into heads"
        self.attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.proj.WEIGHT_SCALE_INIT = 1  # ty: ignore[invalid-assignment]
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Attend over a batch of sequences shaped (batch, time, channels)."""
        batch, time, channels = x.size()
        q, k, v = self.attn(x).split(self.n_embd, dim=2)
        head_dim = channels // self.n_head

        def split_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(batch, time, self.n_head, head_dim).transpose(1, 2)

        y = F.scaled_dot_product_attention(
            split_heads(q), split_heads(k), split_heads(v), is_causal=True
        )
        y = y.transpose(1, 2).contiguous().view(batch, time, channels)
        return self.proj(y)
