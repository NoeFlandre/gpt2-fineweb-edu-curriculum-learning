"""Assemble token shards from documents."""

from collections.abc import Callable, Iterable, Sequence

import numpy as np
from numpy.typing import ArrayLike


def take_documents(
    texts: Iterable[str], n_tokens: int, encode: Callable[[str], ArrayLike]
) -> tuple[list[np.ndarray], list[str]]:
    """Encode documents until at least n_tokens tokens have been collected.

    Returns:
        The token array and the text of each kept document.
    """
    arrays: list[np.ndarray] = []
    kept: list[str] = []
    total = 0
    for text in texts:
        if total >= n_tokens:
            break
        ids = np.asarray(encode(text), dtype=np.uint16)
        arrays.append(ids)
        kept.append(text)
        total += len(ids)
    return arrays, kept


def build_arrays(
    token_arrays: Sequence[np.ndarray], doc_scores: Sequence[int], max_tokens: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Join documents into one token array with each token carrying its document's score.

    Args:
        token_arrays: Token ids of each document.
        doc_scores: Educational score of each document.
        max_tokens: If given, keep only this many leading tokens.

    Returns:
        The tokens and their per-token scores.
    """
    tokens = np.concatenate(token_arrays)
    scores = np.concatenate(
        [
            np.full(len(ids), score, dtype=np.uint8)
            for ids, score in zip(token_arrays, doc_scores, strict=True)
        ]
    )
    if max_tokens is not None:
        tokens, scores = tokens[:max_tokens], scores[:max_tokens]
    return tokens, scores
