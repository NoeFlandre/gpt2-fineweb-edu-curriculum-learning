"""Unweighted validation loss."""

import torch

from gpt2_scratch.data.loader import TokenLoader


@torch.no_grad()
def validation_loss(model, loader: TokenLoader, device: str, batches: int) -> float:
    """Mean next-token loss over a number of batches, with every token counted equally.

    The model's training mode is restored afterwards.
    """
    was_training = model.training
    model.eval()
    loader.reset()
    total = 0.0
    for _ in range(batches):
        x, y, _ = loader.next_batch()
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            _, loss = model(x.to(device), y.to(device))
        total += loss.item()
    model.train(was_training)
    return total / batches
