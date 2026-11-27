"""The study's corpus: validation split, training pools, arm streams and score buckets."""

import hashlib
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass

import numpy as np

MIN_INT_SCORE = 3
SCORE_EDGES = np.array([3.0, 3.5, 4.0])
BUCKET_COUNT = len(SCORE_EDGES) + 1
Encode = Callable[[str], np.ndarray]


@dataclass(frozen=True)
class Document:
    """One encoded document (starting with its end-of-text token), its score and its row key."""

    tokens: np.ndarray
    score: float
    key: str


class StreamTooShortError(ValueError):
    """The rows ended before every score bucket held its quota of tokens."""

    def __init__(self, filled: list[int], quota: int) -> None:
        super().__init__(f"rows ended with {filled} tokens per bucket; each needs {quota}")


def _eligible(row: dict) -> bool:
    return row["int_score"] >= MIN_INT_SCORE and bool(row["text"].strip())


def row_key(row: dict) -> str:
    """The identity of a row across data seeds: the hash of its text."""
    return hashlib.sha256(row["text"].encode()).hexdigest()


def bucket_of(score: float) -> int:
    """The score bucket (0 to 3) of one score: [2.5, 3.0), [3.0, 3.5), [3.5, 4.0), [4.0, 5.0]."""
    return int(np.searchsorted(SCORE_EDGES, score, side="right"))


def collect_documents(
    rows: Iterator[dict], encode: Encode, budget: int, skip: frozenset[str] = frozenset()
) -> list[Document]:
    """Encode eligible rows, in order, until they hold at least budget tokens.

    Rows whose key is in skip are left out before they are encoded.
    """
    documents: list[Document] = []
    total = 0
    for row in rows:
        if not _eligible(row):
            continue
        key = row_key(row)
        if key in skip:
            continue
        tokens = encode(row["text"])
        documents.append(Document(tokens, float(row["score"]), key))
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
        documents.append(Document(tokens, score, row_key(row)))
        filled[bucket] += len(tokens)
        if min(filled) >= quota:
            return documents
    raise StreamTooShortError(filled, quota)


def validation_after_pool(
    rows: Iterable[dict], encode: Encode, pool_tokens: int, val_tokens: int
) -> list[Document]:
    """The study's validation set: a stratified sample of the rows that follow a pool.

    The first pool_tokens of eligible rows are consumed, as the pool of that stream would take them.
    The validation set then holds val_tokens / BUCKET_COUNT tokens in each score bucket.
    """
    rows = iter(rows)
    collect_documents(rows, encode, pool_tokens)
    return collect_stratified(rows, encode, val_tokens // BUCKET_COUNT)


def arm_tokens(documents: list[Document], order: np.ndarray, budget: int) -> np.ndarray:
    """The first budget tokens of the documents concatenated in the given order."""
    tokens = np.concatenate([documents[index].tokens for index in order])
    return tokens[:budget]


def token_buckets(documents: list[Document]) -> np.ndarray:
    """The score bucket of each token, repeated for each token of its document."""
    buckets = [bucket_of(doc.score) for doc in documents]
    lengths = [len(doc.tokens) for doc in documents]
    return np.repeat(np.array(buckets, dtype=np.uint8), lengths)
