"""Multiple-choice benchmarks, scored like the original HellaSwag evaluation.

Each example has a context and a list of endings. Every ending is scored by its average per-token
loss over the ending tokens (normalized accuracy); the lowest-loss ending is the prediction.
"""

import json
from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn.functional as F

Encode = Callable[[str], list[int]]
EXAMPLES_PER_BATCH = 8


def load_examples(path: Path) -> list[dict]:
    """Read one example per line of a JSON Lines file."""
    with Path(path).open() as file:
        return [json.loads(line) for line in file]


def render_example(example: dict, encode: Encode) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Build one padded row per ending: context followed by that ending.

    Returns:
        Tokens of shape (endings, width), a mask that is 1 on ending tokens, and the label.
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
    rendered: list[tuple[torch.Tensor, torch.Tensor, int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad a batch of examples to one shape: tokens, masks, valid endings and labels."""
    endings = max(tokens.size(0) for tokens, _, _ in rendered)
    width = max(tokens.size(1) for tokens, _, _ in rendered)
    tokens = torch.zeros(len(rendered), endings, width, dtype=torch.long)
    masks = torch.zeros_like(tokens)
    valid = torch.zeros(len(rendered), endings, dtype=torch.bool)
    for index, (row_tokens, row_mask, _) in enumerate(rendered):
        count, length = row_tokens.shape
        tokens[index, :count, :length] = row_tokens
        masks[index, :count, :length] = row_mask
        valid[index, :count] = True
    labels = torch.tensor([label for _, _, label in rendered])
    return tokens, masks, valid, labels


def completion_losses(
    logits: torch.Tensor, tokens: torch.Tensor, mask: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    """Average loss of each ending over its ending tokens; padded endings get infinity.

    Args:
        logits: Shape (examples, endings, width, vocab).
        tokens: Shape (examples, endings, width).
        mask: Shape (examples, endings, width), 1 on ending tokens.
        valid: Shape (examples, endings), False for padding.

    Returns:
        Shape (examples, endings).
    """
    shift_logits = logits[:, :, :-1, :].float()
    shift_targets = tokens[:, :, 1:]
    shift_mask = mask[:, :, 1:].float()
    per_token = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_targets.reshape(-1),
        reduction="none",
    ).view(shift_targets.shape)
    average = (per_token * shift_mask).sum(dim=-1) / shift_mask.sum(dim=-1).clamp_min(1.0)
    return torch.where(valid, average, torch.full_like(average, float("inf")))


@torch.no_grad()
def accuracy(
    model, examples: list[dict], device: str, encode: Encode, batch_size: int = EXAMPLES_PER_BATCH
) -> float:
    """Fraction of examples whose lowest-loss ending is the labelled one.

    Examples are batched in order of length, which only reduces padding; each example is still
    scored on its own endings, so the count of correct predictions is unchanged.
    """
    rendered = [render_example(example, encode) for example in examples]
    order = sorted(range(len(rendered)), key=lambda index: rendered[index][0].size(1))
    correct = 0
    for start in range(0, len(order), batch_size):
        batch = [rendered[index] for index in order[start : start + batch_size]]
        tokens, masks, valid, labels = _stack(batch)
        count, endings, width = tokens.shape
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, _ = model(tokens.reshape(-1, width).to(device))
        losses = completion_losses(
            logits.reshape(count, endings, width, -1),
            tokens.to(device),
            masks.to(device),
            valid.to(device),
        )
        correct += int((losses.argmin(dim=1).cpu() == labels).sum())
    return correct / len(examples)
