import numpy as np
import pytest
import tiktoken
import torch
from gpt2_scratch.data.build import build_arrays, take_documents
from gpt2_scratch.data.loader import TokenLoader
from gpt2_scratch.data.shards import list_shards, load_shard, save_shard
from gpt2_scratch.data.tokenize import encode_document

ENCODER = tiktoken.get_encoding("gpt2")


def make_shard(directory, split, index, length, offset=0):
    tokens = (offset + np.arange(length)).astype(np.uint16)
    scores = (np.arange(length) % 6).astype(np.uint8)
    save_shard(directory, split, index, tokens, scores)


def test_document_starts_with_the_end_of_text_token():
    tokens = encode_document("Hello world", ENCODER)

    assert tokens[0] == ENCODER.eot_token
    assert tokens[1:].tolist() == ENCODER.encode_ordinary("Hello world")


def test_encoded_documents_are_uint16():
    assert encode_document("Hello", ENCODER).dtype == np.uint16


def test_end_of_text_only_appears_at_the_start():
    tokens = encode_document("<|endoftext|>", ENCODER)

    assert tokens.tolist().count(ENCODER.eot_token) == 1


def test_shard_round_trip_keeps_tokens_and_scores(tmp_path):
    tokens = np.arange(10, dtype=np.uint16)
    scores = (np.arange(10) % 6).astype(np.uint8)
    save_shard(tmp_path, "train", 0, tokens, scores)

    loaded_tokens, loaded_scores = load_shard(list_shards(tmp_path, "train")[0])

    np.testing.assert_array_equal(loaded_tokens, tokens)
    np.testing.assert_array_equal(loaded_scores, scores)


def test_list_shards_is_sorted_and_ignores_score_files(tmp_path):
    make_shard(tmp_path, "train", 1, 4)
    make_shard(tmp_path, "train", 0, 4)

    names = [path.name for path in list_shards(tmp_path, "train")]

    assert names == ["train_000000.npy", "train_000001.npy"]


def test_list_shards_only_returns_the_requested_split(tmp_path):
    make_shard(tmp_path, "train", 0, 4)
    make_shard(tmp_path, "val", 0, 4)

    assert [path.name for path in list_shards(tmp_path, "val")] == ["val_000000.npy"]


def test_list_shards_fails_when_the_split_is_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="no train shards"):
        list_shards(tmp_path, "train")


def test_save_shard_rejects_mismatched_lengths(tmp_path):
    with pytest.raises(ValueError, match="same length"):
        save_shard(tmp_path, "train", 0, np.zeros(3, np.uint16), np.zeros(2, np.uint8))


def test_loader_batches_have_the_requested_shape(tmp_path):
    make_shard(tmp_path, "train", 0, 200)
    loader = TokenLoader(tmp_path, "train", batch_size=2, seq_len=8)

    x, y, s = loader.next_batch()

    assert x.shape == y.shape == s.shape == (2, 8)
    assert x.dtype == torch.long


def test_targets_are_inputs_shifted_by_one(tmp_path):
    make_shard(tmp_path, "train", 0, 200)
    loader = TokenLoader(tmp_path, "train", batch_size=2, seq_len=8)

    x, y, _ = loader.next_batch()

    assert x[0, 0].item() == 0
    assert y[0, 0].item() == 1
    assert torch.equal(x[0, 1:], y[0, :-1])


def test_scores_are_aligned_with_the_targets(tmp_path):
    make_shard(tmp_path, "train", 0, 200)
    loader = TokenLoader(tmp_path, "train", batch_size=2, seq_len=8)

    _, y, s = loader.next_batch()

    expected = torch.from_numpy((np.arange(1, 17) % 6).astype(np.int64)).view(2, 8)
    assert torch.equal(s, expected)
    assert y[0, 0].item() == 1


def test_consecutive_batches_continue_the_stream(tmp_path):
    make_shard(tmp_path, "train", 0, 200)
    loader = TokenLoader(tmp_path, "train", batch_size=2, seq_len=8)

    loader.next_batch()
    x, _, _ = loader.next_batch()

    assert x[0, 0].item() == 16


def test_loader_moves_to_the_next_shard_then_wraps_around(tmp_path):
    make_shard(tmp_path, "train", 0, 20)
    make_shard(tmp_path, "train", 1, 20, offset=100)
    loader = TokenLoader(tmp_path, "train", batch_size=2, seq_len=8)

    loader.next_batch()
    x_second_shard, _, _ = loader.next_batch()
    assert x_second_shard[0, 0].item() == 100

    loader.reset()
    x_again, _, _ = loader.next_batch()
    assert x_again[0, 0].item() == 0


def test_take_documents_stops_once_the_budget_is_reached():
    texts = ["a" * 10, "b" * 10, "c" * 10, "d" * 10]

    token_arrays, kept_texts = take_documents(iter(texts), 15, lambda text: list(text.encode()))

    assert kept_texts == texts[:2]
    assert len(token_arrays) == 2


def test_build_arrays_repeats_each_document_score_per_token():
    docs = [np.array([5, 6, 7], dtype=np.uint16), np.array([8], dtype=np.uint16)]

    tokens, scores = build_arrays(docs, [4, 1])

    assert tokens.tolist() == [5, 6, 7, 8]
    assert scores.tolist() == [4, 4, 4, 1]


def test_build_arrays_trims_to_the_token_budget():
    docs = [np.arange(5, dtype=np.uint16), np.arange(5, dtype=np.uint16)]

    tokens, scores = build_arrays(docs, [2, 3], max_tokens=7)

    assert len(tokens) == len(scores) == 7
    assert scores.tolist() == [2, 2, 2, 2, 2, 3, 3]


def test_take_documents_keeps_everything_when_the_stream_runs_out():
    texts = ["abc", "de"]

    token_arrays, kept_texts = take_documents(iter(texts), 1_000, lambda text: list(text.encode()))

    assert kept_texts == texts
    assert len(token_arrays) == 2


def test_take_documents_keeps_one_document_for_a_tiny_budget():
    token_arrays, kept_texts = take_documents(
        iter(["abc", "def"]), 1, lambda text: list(text.encode())
    )

    assert kept_texts == ["abc"]


def test_take_documents_stops_exactly_when_the_budget_is_met():
    texts = ["aaaaaaaaaa", "bbbbbbbbbb", "cccccccccc"]

    _, kept_texts = take_documents(iter(texts), 10, lambda text: list(text.encode()))

    assert kept_texts == texts[:1]


def test_take_documents_returns_uint16_token_arrays():
    token_arrays, _ = take_documents(iter(["ab"]), 10, lambda text: [1, 2, 300])

    assert token_arrays[0].dtype == np.uint16
    assert token_arrays[0].tolist() == [1, 2, 300]


def test_build_arrays_returns_uint8_scores():
    _, scores = build_arrays([np.array([1, 2], dtype=np.uint16)], [5])

    assert scores.dtype == np.uint8


def test_build_arrays_rejects_mismatched_documents_and_scores():
    with pytest.raises(ValueError):
        build_arrays([np.array([1], dtype=np.uint16)], [1, 2])
