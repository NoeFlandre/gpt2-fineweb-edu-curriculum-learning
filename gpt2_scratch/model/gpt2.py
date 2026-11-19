"""GPT-2 language model."""

import inspect
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from gpt2_scratch.model.block import Block
from gpt2_scratch.model.config import GPT2Config
from gpt2_scratch.model.loss import weighted_mean


class GPT2(nn.Module):
    """Decoder-only transformer with tied input and output embeddings."""

    def __init__(self, config: GPT2Config) -> None:
        """Build the network and initialize its weights."""
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "WEIGHT_SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return logits and, when targets are given, the next-token loss.

        Args:
            idx: Token ids, shape (batch, time).
            targets: Next-token ids, same shape as idx.
            weights: Optional per-token loss weights, same shape as targets.
        """
        _, length = idx.size()
        assert length <= self.config.block_size, "sequence longer than the block size"
        pos = torch.arange(0, length, dtype=torch.long, device=idx.device)
        x = self.wpe(pos) + self.wte(idx)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))
        if targets is None:
            return logits, None
        per_token = F.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.reshape(-1), reduction="none"
        )
        if weights is None:
            return logits, per_token.mean()
        return logits, weighted_mean(per_token, weights.reshape(-1))

    def configure_optimizers(
        self, weight_decay: float, learning_rate: float, device: str
    ) -> torch.optim.Optimizer:
        """Create AdamW, decaying matrices but not biases or LayerNorm parameters."""
        params = [p for p in self.parameters() if p.requires_grad]
        return torch.optim.AdamW(
            _weight_decay_groups(params, weight_decay),
            lr=learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
            fused=_supports_fused(device),
        )


def _weight_decay_groups(params: Sequence[torch.Tensor], weight_decay: float) -> list[dict]:
    """Decay matrices only; biases and LayerNorm parameters are left alone."""
    decay = [p for p in params if p.dim() >= 2]
    no_decay = [p for p in params if p.dim() < 2]
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def _supports_fused(device: str) -> bool:
    return "fused" in inspect.signature(torch.optim.AdamW).parameters and "cuda" in device
