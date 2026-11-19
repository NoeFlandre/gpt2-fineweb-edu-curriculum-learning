"""Loss helpers shared by the model and the training loop."""

import torch

EPSILON = 1e-8


def weighted_mean(per_token_loss: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Average per-token losses, counting each token by its weight.

    With all weights equal to one this is the plain mean. If every weight is zero the
    result is zero, not NaN.
    """
    return (per_token_loss * weights).sum() / weights.sum().clamp_min(EPSILON)
