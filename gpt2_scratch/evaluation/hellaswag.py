"""HellaSwag completion accuracy.

Each example has a context and four endings. The model scores every ending by its average
per-token loss over the ending tokens; the lowest-loss ending is its prediction.
"""

import json
from collections.abc import Callable, Sequence
from pathlib import Path

import torch
import torch.nn.functional as F

Encode = Callable[[str], list[int]]
CHOICES = 4


def load_examples(path: Path) -> list[dict]:
    """Read one HellaSwag example per line of a JSON Lines file."""
    with Path(path).open() as file:
        return [json.loads(line) for line in file]


def render_example(example: dict, encode: Encode) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Build the four padded context-plus-ending rows of one example.

    Returns:
        Tokens of shape (4, width), a mask that is 1 on ending tokens, and the label.
    """
    context = encode(example["ctx"])
    endings = [encode(" " + ending) for ending in example["endings"]]
    rows = [context + ending for ending in endings]
    masks = [[0] * len(context) + [1] * len(ending) for ending in endings]
    return _pad(rows), _pad(masks), example["label"]


def _pad(rows: list[list[int]]) -> torch.Tensor:
    padded = torch.zeros(len(rows), max(len(row) for row in rows), dtype=torch.long)
    for index, row in enumerate(rows):
        padded[index, : len(row)] = torch.tensor(row)
    return padded


def completion_losses(
    logits: torch.Tensor, tokens: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Average loss of each ending, computed over its ending tokens only."""
    shift_logits = logits[:, :-1, :].reshape(-1, logits.size(-1))
    shift_targets = tokens[:, 1:].reshape(-1)
    shift_mask = mask[:, 1:].float()
    losses = F.cross_entropy(shift_logits, shift_targets, reduction="none")
    losses = losses.view(CHOICES, -1)
    return (losses * shift_mask).sum(dim=1) / shift_mask.sum(dim=1)


def predict(logits: torch.Tensor, tokens: torch.Tensor, mask: torch.Tensor) -> int:
    """Index of the ending with the lowest average loss."""
    return int(completion_losses(logits, tokens, mask).argmin().item())


@torch.no_grad()
def accuracy(model, examples: Sequence[dict], device: str, encode: Encode) -> float:
    """Fraction of examples whose prediction matches the label."""
    correct = 0
    for example in examples:
        tokens, mask, label = render_example(example, encode)
        logits, _ = model(tokens.to(device))
        correct += predict(logits.cpu(), tokens, mask) == label
    return correct / len(examples)
