"""Document orderings of the study: every arm trains on the same pool, in a different order."""

import numpy as np

ORDERINGS = ("shuffle", "ascending", "descending", "cooldown")
COOLDOWN_SHARE = 0.8


class UnknownOrderingError(ValueError):
    """An ordering name that is not one of ORDERINGS."""

    def __init__(self, name: str) -> None:
        super().__init__(f"unknown ordering {name!r}")


def arm_order(
    name: str, scores: np.ndarray, lengths: np.ndarray, budget: int, seed: int
) -> np.ndarray:
    """Indices of the pool's documents in the order this arm trains on them.

    Args:
        name: One of ORDERINGS.
        scores: Continuous educational score of each document.
        lengths: Token count of each document.
        budget: Tokens the arm trains on; the cooldown split is taken at a share of it.
        seed: Seed for the random shuffles.
    """
    if name not in ORDERINGS:
        raise UnknownOrderingError(name)
    rng = np.random.default_rng(seed)
    if name == "shuffle":
        return rng.permutation(len(scores))
    ascending = np.argsort(scores, kind="stable")
    if name == "ascending":
        return ascending
    if name == "descending":
        return ascending[::-1]
    return _cooldown(ascending, lengths, budget, rng)


def _cooldown(
    ascending: np.ndarray, lengths: np.ndarray, budget: int, rng: np.random.Generator
) -> np.ndarray:
    """Shuffle the lowest-scoring documents that fill 80% of the budget, then the rest."""
    cumulative = np.cumsum(lengths[ascending])
    split = int(np.searchsorted(cumulative, COOLDOWN_SHARE * budget, side="right"))
    return np.concatenate([rng.permutation(ascending[:split]), rng.permutation(ascending[split:])])
