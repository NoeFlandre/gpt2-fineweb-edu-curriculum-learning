import math

import numpy as np
import torch
from gpt2_scratch.data.shards import save_shard
from gpt2_scratch.model.config import GPT2Config
from gpt2_scratch.model.gpt2 import GPT2
from gpt2_scratch.training.evaluate import load_validation, validation_by_bucket

TINY = GPT2Config(block_size=16, vocab_size=64, n_layer=1, n_head=2, n_embd=16)
BUCKET_KEYS = {
    "val_loss",
    "val_loss_25_30",
    "val_loss_30_35",
    "val_loss_35_40",
    "val_loss_40_50",
}


def test_one_bucket_reports_the_same_loss_as_the_overall_loss():
    torch.manual_seed(0)
    model = GPT2(TINY)
    tokens = (np.arange(400) % 60).astype(np.uint16)
    buckets = np.zeros(400, dtype=np.uint8)

    result = validation_by_bucket(model, tokens, buckets, "cpu", batch_size=2, seq_len=16)

    assert set(result) == BUCKET_KEYS
    assert result["val_loss"] == result["val_loss_25_30"]
    assert math.isnan(result["val_loss_40_50"])


def test_validation_is_deterministic_and_restores_the_training_mode():
    torch.manual_seed(0)
    model = GPT2(TINY)
    tokens = (np.arange(400) % 60).astype(np.uint16)
    buckets = (np.arange(400) % 4).astype(np.uint8)

    first = validation_by_bucket(model, tokens, buckets, "cpu", batch_size=2, seq_len=16)
    second = validation_by_bucket(model, tokens, buckets, "cpu", batch_size=2, seq_len=16)

    assert first == second
    assert model.training


def test_bucket_losses_are_finite_when_every_bucket_has_tokens():
    torch.manual_seed(0)
    model = GPT2(TINY)
    tokens = (np.arange(400) % 60).astype(np.uint16)
    buckets = (np.arange(400) % 4).astype(np.uint8)

    result = validation_by_bucket(model, tokens, buckets, "cpu", batch_size=2, seq_len=16)

    assert all(math.isfinite(result[key]) for key in BUCKET_KEYS)


def test_load_validation_reads_the_tokens_and_their_buckets(tmp_path):
    folder = tmp_path / "validation"
    save_shard(folder, "val", 0, np.arange(8, dtype=np.uint16))
    np.save(folder / "val_buckets.npy", np.zeros(8, dtype=np.uint8))

    tokens, buckets = load_validation(folder)

    assert tokens.tolist() == list(range(8))
    assert buckets.tolist() == [0] * 8
