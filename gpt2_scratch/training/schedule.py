"""Learning-rate schedule: linear warmup, then cosine decay to the minimum, or a constant peak."""

import math

from gpt2_scratch.training.config import TrainConfig


def get_lr(step: int, cfg: TrainConfig) -> float:
    """Learning rate for an optimizer step."""
    if step < cfg.warmup_steps:
        return cfg.max_lr * (step + 1) / cfg.warmup_steps
    if cfg.schedule == "constant":
        return cfg.max_lr
    if step > cfg.max_steps:
        return cfg.min_lr
    progress = (step - cfg.warmup_steps) / (cfg.max_steps - cfg.warmup_steps)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return cfg.min_lr + (cfg.max_lr - cfg.min_lr) * cosine
