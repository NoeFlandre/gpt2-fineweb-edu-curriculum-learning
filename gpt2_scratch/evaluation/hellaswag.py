"""HellaSwag completion accuracy, scored as in the original repository.

Each example has a context and four endings. Each ending is scored by its average per-token loss
over the ending tokens (normalized accuracy), and the lowest-loss ending is the prediction. The
model runs under bf16 autocast, as in the original evaluation.
"""

import json
from collections.abc import Callable, Sequence
from pathlib import Path

import torch
import torch.nn.functional as F

Encode = Callable[[str], list[int]]
CHOICES = 4
EXAMPLES_PER_BATCH = 8


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


def _stack(
    rendered: Sequence[tuple[torch.Tensor, torch.Tensor, int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stack rendered examples into (N, 4, width) tokens and masks, and (N,) labels."""
    width = max(tokens.size(1) for tokens, _, _ in rendered)
    tokens = torch.zeros(len(rendered), CHOICES, width, dtype=torch.long)
    masks = torch.zeros_like(tokens)
    for index, (row_tokens, row_mask, _) in enumerate(rendered):
        tokens[index, :, : row_tokens.size(1)] = row_tokens
        masks[index, :, : row_mask.size(1)] = row_mask
    labels = torch.tensor([label for _, _, label in rendered])
    return tokens, masks, labels


def completion_losses(
    logits: torch.Tensor, tokens: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Average loss of each ending over its ending tokens.

    Args:
        logits: Shape (N, 4, width, vocab).
        tokens: Shape (N, 4, width).
        mask: Shape (N, 4, width), 1 on ending tokens.

    Returns:
        Shape (N, 4).
    """
    shift_logits = logits[:, :, :-1, :].float()
    shift_targets = tokens[:, :, 1:]
    shift_mask = mask[:, :, 1:].float()
    losses = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)), shift_targets.reshape(-1), reduction="none"
    )
    losses = losses.view(shift_targets.shape)
    return (losses * shift_mask).sum(dim=-1) / shift_mask.sum(dim=-1)


def predict(logits: torch.Tensor, tokens: torch.Tensor, mask: torch.Tensor) -> int:
    """Index of the ending with the lowest average loss, for one example of shape (4, width)."""
    return int(completion_losses(logits[None], tokens[None], mask[None])[0].argmin().item())


@torch.no_grad()
def accuracy(model, examples: Sequence[dict], device: str, encode: Encode) -> float:
    """Fraction of examples whose lowest-loss ending is the labelled one.

    Examples are batched in order of length, which only reduces padding; each example is still
    scored on its own endings, so the count of correct predictions is unchanged.
    """
    rendered = [render_example(example, encode) for example in examples]
    order = sorted(range(len(rendered)), key=lambda index: rendered[index][0].size(1))
    correct = 0
    for start in range(0, len(order), EXAMPLES_PER_BATCH):
        tokens, mask, labels = _stack(
            [rendered[index] for index in order[start : start + EXAMPLES_PER_BATCH]]
        )
        tokens, mask = tokens.to(device), mask.to(device)
        count, width = tokens.size(0), tokens.size(2)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, _ = model(tokens.reshape(-1, width))
        logits = logits.reshape(count, CHOICES, width, -1)
        predictions = completion_losses(logits, tokens, mask).argmin(dim=1).cpu()
        correct += int((predictions == labels).sum())
    return correct / len(examples)
