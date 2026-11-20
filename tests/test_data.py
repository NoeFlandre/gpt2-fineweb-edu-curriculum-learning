import numpy as np
import pytest
import torch
from gpt2_scratch.data.corpus import (
    Document,
    StreamTooShortError,
    arm_tokens,
    bucket_of,
    collect_documents,
    collect_stratified,
    split_pool_and_validation,
    token_buckets,
)
from gpt2_scratch.data.loader import TokenLoader
from gpt2_scratch.data.shards import list_shards, load_shard, save_shard

EOT = 50256
BUCKET_SCORES = [2.6, 3.2, 3.7, 4.2]


def encode(text):
    return np.array([EOT, *[ord(char) % 200 for char in text]], dtype=np.uint16)


def row(text, int_score=3, score=3.1):
    return {"text": text, "int_score": int_score, "score": score}


def spread_rows(count):
    """Rows whose scores cycle through the four buckets, each score unique."""
    return [row(f"doc{i}", 3, BUCKET_SCORES[i % 4] + i / 1000) for i in range(count)]


def test_collect_documents_keeps_rows_whose_integer_score_is_three_or_more():
    rows = iter([row("keep", 3, 3.1), row("drop", 2, 2.4), row("also", 5, 4.8)])

    docs = collect_documents(rows, encode, budget=1000)

    assert [doc.score for doc in docs] == [3.1, 4.8]


def test_collect_documents_skips_rows_without_text():
    rows = iter([row("   ", 3, 3.1), row("keep", 3, 3.2)])

    docs = collect_documents(rows, encode, budget=1000)

    assert [doc.score for doc in docs] == [3.2]


def test_collect_documents_stops_once_the_budget_is_reached():
    rows = iter([row("x" * 9, 3, 3.1), row("y" * 9, 3, 3.2), row("z" * 9, 3, 3.3)])

    docs = collect_documents(rows, encode, budget=20)

    assert len(docs) == 2
    assert next(rows)["score"] == 3.3


def test_collect_documents_stops_at_the_document_that_first_reaches_the_budget():
    rows = iter([row("x" * 9, 3, 3.1), row("y" * 9, 3, 3.2), row("z" * 9, 3, 3.3)])

    docs = collect_documents(rows, encode, budget=11)

    assert len(docs) == 2


def test_documents_start_with_the_end_of_text_token():
    docs = collect_documents(iter([row("hi", 3, 3.1)]), encode, budget=1)

    assert docs[0].tokens[0] == EOT


def test_collect_stratified_fills_each_bucket_to_its_quota():
    rows = iter(
        [
            row("drop", 2, 4.5),
            row("abcd", 3, 2.6),
            row("efgh", 3, 2.7),
            row("ijkl", 3, 2.8),
            row("mnop", 3, 3.2),
            row("qr", 3, 3.3),
            row("st", 3, 3.7),
            row("uvwxyz", 3, 3.8),
            row("abc", 3, 4.2),
            row("defgh", 3, 4.4),
        ]
    )

    docs = collect_stratified(rows, encode, quota=6)

    assert [doc.score for doc in docs] == [2.6, 2.7, 3.2, 3.3, 3.7, 3.8, 4.2, 4.4]
    assert [len(doc.tokens) for doc in docs] == [5, 1, 5, 1, 3, 3, 4, 2]


def test_collect_stratified_fails_when_the_rows_end_before_every_bucket_is_full():
    with pytest.raises(
        StreamTooShortError, match=r"rows ended with \[3, 0, 0, 0\] tokens per bucket"
    ):
        collect_stratified(iter([row("ab", 3, 2.6)]), encode, quota=6)


def test_split_keeps_validation_documents_out_of_the_pool():
    pool, validation = split_pool_and_validation(
        spread_rows(40), encode, pool_tokens=20, val_tokens=40
    )

    assert len(pool) == 4
    assert {doc.score for doc in pool}.isdisjoint({doc.score for doc in validation})


def test_validation_holds_the_same_number_of_tokens_in_every_bucket():
    _, validation = split_pool_and_validation(
        spread_rows(40), encode, pool_tokens=20, val_tokens=40
    )

    counts = np.bincount(token_buckets(validation), minlength=4)
    assert counts.tolist() == [10, 10, 10, 10]


def test_arm_tokens_are_exactly_the_budget_in_document_order():
    docs = [
        Document(np.array([EOT, 1, 2], dtype=np.uint16), 3.1),
        Document(np.array([EOT, 3], dtype=np.uint16), 3.2),
    ]

    tokens = arm_tokens(docs, np.array([1, 0]), budget=4)

    assert tokens.tolist() == [EOT, 3, EOT, 1]
    assert tokens.dtype == np.uint16


def test_score_buckets_follow_the_fixed_edges_from_2_5_to_5():
    scores = [2.5, 2.99, 3.0, 3.49, 3.5, 3.99, 4.0, 5.0]

    assert [bucket_of(score) for score in scores] == [0, 0, 1, 1, 2, 2, 3, 3]


def test_token_buckets_repeat_each_document_bucket_for_its_tokens():
    docs = [
        Document(np.array([EOT, 1], dtype=np.uint16), 3.5),
        Document(np.array([EOT], dtype=np.uint16), 4.5),
    ]

    buckets = token_buckets(docs)

    assert buckets.tolist() == [2, 2, 3]
    assert buckets.dtype == np.uint8


def test_shards_round_trip_tokens(tmp_path):
    tokens = np.arange(10, dtype=np.uint16)
    save_shard(tmp_path, "train", 0, tokens)

    loaded = load_shard(list_shards(tmp_path, "train")[0])

    np.testing.assert_array_equal(loaded, tokens)


def test_list_shards_is_sorted_and_only_returns_the_requested_split(tmp_path):
    save_shard(tmp_path, "train", 1, np.zeros(4, np.uint16))
    save_shard(tmp_path, "train", 0, np.zeros(4, np.uint16))
    save_shard(tmp_path, "val", 0, np.zeros(4, np.uint16))

    names = [path.name for path in list_shards(tmp_path, "train")]

    assert names == ["train_000000.npy", "train_000001.npy"]


def test_list_shards_fails_when_the_split_is_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="no train shards"):
        list_shards(tmp_path, "train")


def test_loader_yields_inputs_and_targets_shifted_by_one(tmp_path):
    save_shard(tmp_path, "train", 0, np.arange(200, dtype=np.uint16))
    loader = TokenLoader(tmp_path, "train", batch_size=2, seq_len=8)

    x, y = loader.next_batch()

    assert x.shape == y.shape == (2, 8)
    assert x.dtype == torch.long
    assert torch.equal(x[0, 1:], y[0, :-1])


def test_loader_moves_to_the_next_shard_then_wraps_around(tmp_path):
    save_shard(tmp_path, "train", 0, np.arange(20, dtype=np.uint16))
    save_shard(tmp_path, "train", 1, np.arange(100, 120, dtype=np.uint16))
    loader = TokenLoader(tmp_path, "train", batch_size=2, seq_len=8)
    loader.next_batch()

    x, _ = loader.next_batch()
    assert x[0, 0].item() == 100

    loader.reset()
    x, _ = loader.next_batch()
    assert x[0, 0].item() == 0


def test_loader_state_lets_a_restored_loader_continue_the_same_stream(tmp_path):
    save_shard(tmp_path, "train", 0, np.arange(200, dtype=np.uint16))
    loader = TokenLoader(tmp_path, "train", batch_size=2, seq_len=8)
    loader.next_batch()
    saved = loader.state()
    expected, _ = loader.next_batch()

    restored = TokenLoader(tmp_path, "train", batch_size=2, seq_len=8)
    restored.restore(saved)
    actual, _ = restored.next_batch()

    assert torch.equal(actual, expected)
