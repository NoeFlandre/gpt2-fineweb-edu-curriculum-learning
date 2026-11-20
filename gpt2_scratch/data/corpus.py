"""The study's corpus: the training pool, each arm's token stream, and the score buckets."""

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass

import numpy as np

MIN_INT_SCORE = 3
SCORE_EDGES = np.array([3.0, 3.5, 4.0])
BUCKET_COUNT = len(SCORE_EDGES) + 1
Encode = Callable[[str], np.ndarray]


@dataclass(frozen=True)
class Document:
    """One encoded document (starting with its end-of-text token) and its continuous score."""

    tokens: np.ndarray
    score: float


class StreamTooShortError(ValueError):
    """The rows ended before every score bucket held its quota of tokens."""

    def __init__(self, filled: list[int], quota: int) -> None:
        super().__init__(f"rows ended with {filled} tokens per bucket; each needs {quota}")


def _eligible(row: dict) -> bool:
    return row["int_score"] >= MIN_INT_SCORE and bool(row["text"].strip())


def bucket_of(score: float) -> int:
    """The score bucket (0 to 3) of one score: [2.5, 3.0), [3.0, 3.5), [3.5, 4.0), [4.0, 5.0]."""
    return int(np.searchsorted(SCORE_EDGES, score, side="right"))


def collect_documents(rows: Iterator[dict], encode: Encode, budget: int) -> list[Document]:
    """Encode eligible rows, in order, until they hold at least budget tokens."""
    documents: list[Document] = []
    total = 0
    for row in rows:
        if not _eligible(row):
            continue
        tokens = encode(row["text"])
        documents.append(Document(tokens, float(row["score"])))
        total += len(tokens)
        if total >= budget:
            break
    return documents


def collect_stratified(rows: Iterator[dict], encode: Encode, quota: int) -> list[Document]:
    """Encode eligible rows until every score bucket holds quota tokens.

    A row whose bucket is already full is skipped before it is encoded. The row that fills a
    bucket is cut to the room left in it.
    """
    documents: list[Document] = []
    filled = [0] * BUCKET_COUNT
    for row in rows:
        if not _eligible(row):
            continue
        score = float(row["score"])
        bucket = bucket_of(score)
        room = quota - filled[bucket]
        if room <= 0:
            continue
        tokens = encode(row["text"])[:room]
        documents.append(Document(tokens, score))
        filled[bucket] += len(tokens)
        if min(filled) >= quota:
            return documents
    raise StreamTooShortError(filled, quota)


def split_pool_and_validation(
    rows: Iterable[dict], encode: Encode, pool_tokens: int, val_tokens: int
) -> tuple[list[Document], list[Document]]:
    """Take the training pool first, then a stratified validation set from the rows that follow.

    The validation set holds val_tokens / BUCKET_COUNT tokens in each score bucket.
    """
    rows = iter(rows)
    pool = collect_documents(rows, encode, pool_tokens)
    validation = collect_stratified(rows, encode, val_tokens // BUCKET_COUNT)
    return pool, validation


def arm_tokens(documents: list[Document], order: np.ndarray, budget: int) -> np.ndarray:
    """The first budget tokens of the documents concatenated in the given order."""
    tokens = np.concatenate([documents[index].tokens for index in order])
    return tokens[:budget]


def token_buckets(documents: list[Document]) -> np.ndarray:
    """The score bucket of each token, repeated for each token of its document."""
    buckets = [bucket_of(doc.score) for doc in documents]
    lengths = [len(doc.tokens) for doc in documents]
    return np.repeat(np.array(buckets, dtype=np.uint8), lengths)
