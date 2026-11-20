"""The study's corpus: the training pool, each arm's token stream, and the score buckets."""

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass

import numpy as np

MIN_INT_SCORE = 3
Encode = Callable[[str], np.ndarray]


@dataclass(frozen=True)
class Document:
    """One encoded document (starting with its end-of-text token) and its continuous score."""

    tokens: np.ndarray
    score: float


def collect_documents(rows: Iterator[dict], encode: Encode, budget: int) -> list[Document]:
    """Encode eligible rows, in order, until they hold at least budget tokens.

    A row is eligible when its integer score is at least MIN_INT_SCORE and it has text.
    """
    documents: list[Document] = []
    total = 0
    for row in rows:
        if row["int_score"] < MIN_INT_SCORE or not row["text"].strip():
            continue
        tokens = encode(row["text"])
        documents.append(Document(tokens, float(row["score"])))
        total += len(tokens)
        if total >= budget:
            break
    return documents


def split_pool_and_validation(
    rows: Iterable[dict], encode: Encode, pool_tokens: int, val_tokens: int
) -> tuple[list[Document], list[Document]]:
    """Take the training pool first, then validation documents from the rows that follow."""
    rows = iter(rows)
    pool = collect_documents(rows, encode, pool_tokens)
    validation = collect_documents(rows, encode, val_tokens)
    return pool, validation


def arm_tokens(documents: list[Document], order: np.ndarray, budget: int) -> np.ndarray:
    """The first budget tokens of the documents concatenated in the given order."""
    tokens = np.concatenate([documents[index].tokens for index in order])
    return tokens[:budget]


def token_scores(documents: list[Document]) -> np.ndarray:
    """The score of each document, repeated for each of its tokens."""
    return np.concatenate(
        [np.full(len(doc.tokens), doc.score, dtype=np.float32) for doc in documents]
    )


SCORE_EDGES = np.array([3.0, 3.5, 4.0])


def bucket_ids(scores: np.ndarray) -> np.ndarray:
    """The score bucket (0 to 3) of each token: [2.5, 3.0), [3.0, 3.5), [3.5, 4.0), [4.0, 5.0]."""
    return np.searchsorted(SCORE_EDGES, scores, side="right").astype(np.uint8)
