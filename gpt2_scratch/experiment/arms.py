"""Experimental arms: which documents are trained on and how much each token counts."""

from collections.abc import Callable
from dataclasses import dataclass

import torch

MAX_SCORE = 5


def uniform_weight(scores: torch.Tensor) -> torch.Tensor:
    """Every token counts the same."""
    return torch.ones_like(scores, dtype=torch.float32)


def soft_weight(scores: torch.Tensor) -> torch.Tensor:
    """A token counts in proportion to its document's educational score."""
    return scores.float() / MAX_SCORE


@dataclass(frozen=True)
class Arm:
    """One experimental arm.

    Attributes:
        name: Identifier used in folder names and reports.
        source: Shard folder to train on: "raw" (all FineWeb documents) or "edu" (FineWeb-Edu).
        weight: Maps each target token's score to its loss weight.
    """

    name: str
    source: str
    weight: Callable[[torch.Tensor], torch.Tensor]


ARMS: dict[str, Arm] = {
    "uniform": Arm("uniform", source="raw", weight=uniform_weight),
    "hard": Arm("hard", source="edu", weight=uniform_weight),
    "soft": Arm("soft", source="raw", weight=soft_weight),
}


def get_arm(name: str) -> Arm:
    """Return the arm registered under name.

    Raises:
        ValueError: If no arm has that name.
    """
    if name not in ARMS:
        raise ValueError(f"unknown arm {name!r}, expected one of {sorted(ARMS)}")
    return ARMS[name]
