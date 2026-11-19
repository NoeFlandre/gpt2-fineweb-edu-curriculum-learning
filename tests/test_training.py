import json
import math

import numpy as np
import pytest
import torch
from gpt2_scratch.data.loader import TokenLoader
from gpt2_scratch.data.shards import save_shard
from gpt2_scratch.experiment.arms import ARMS
from gpt2_scratch.model.config import GPT2Config
from gpt2_scratch.training.config import TrainConfig
from gpt2_scratch.training.evaluate import validation_loss
from gpt2_scratch.training.metrics import MetricsLog
from gpt2_scratch.training.runner import train
from gpt2_scratch.training.schedule import get_lr

TINY = GPT2Config(block_size=16, vocab_size=2048, n_layer=1, n_head=2, n_embd=16)


def make_shard(directory, split, index, length):
    tokens = (np.arange(length) % 2000).astype(np.uint16)
    scores = (np.arange(length) % 6).astype(np.uint8)
    save_shard(directory, split, index, tokens, scores)


def make_data_root(root):
    make_shard(root / "raw", "train", 0, 2000)
    make_shard(root / "raw", "val", 0, 2000)
    make_shard(root / "edu", "train", 0, 2000)
    return root


def test_config_derives_steps_and_gradient_accumulation():
    cfg = TrainConfig(
        total_tokens=2 * 524288,
        total_batch_size=524288,
        micro_batch_size=16,
        sequence_length=1024,
    )

    assert cfg.max_steps == 2
    assert cfg.grad_accum_steps == 32
    assert cfg.min_lr == pytest.approx(cfg.max_lr * 0.1)


def test_config_rejects_batches_that_do_not_split_into_micro_batches():
    with pytest.raises(ValueError, match="divisible"):
        TrainConfig(total_batch_size=1000, micro_batch_size=16, sequence_length=1024)


def test_warmup_rises_linearly_to_the_max_learning_rate():
    cfg = TrainConfig(total_tokens=100 * 524288, warmup_steps=10, max_lr=1.0)

    assert get_lr(0, cfg) == pytest.approx(0.1)
    assert get_lr(9, cfg) == pytest.approx(1.0)


def test_cosine_decay_is_halfway_at_the_midpoint():
    cfg = TrainConfig(total_tokens=100 * 524288, warmup_steps=10, max_lr=1.0)

    assert get_lr(55, cfg) == pytest.approx(0.55)


def test_learning_rate_ends_at_the_minimum_and_stays_there():
    cfg = TrainConfig(total_tokens=100 * 524288, warmup_steps=10, max_lr=1.0)

    assert get_lr(100, cfg) == pytest.approx(cfg.min_lr)
    assert get_lr(150, cfg) == pytest.approx(cfg.min_lr)


def test_metrics_log_writes_one_json_object_per_line(tmp_path):
    log = MetricsLog(tmp_path / "metrics.jsonl")

    log.write(step=0, train_loss=1.5)
    log.write(step=1, train_loss=1.2)

    lines = (tmp_path / "metrics.jsonl").read_text().splitlines()
    assert [json.loads(line) for line in lines] == [
        {"step": 0, "train_loss": 1.5},
        {"step": 1, "train_loss": 1.2},
    ]


def test_metrics_log_starts_from_an_empty_file(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text("stale\n")

    MetricsLog(path)

    assert path.read_text() == ""


def test_validation_loss_is_finite_and_restores_the_training_mode(tmp_path):
    make_shard(tmp_path, "val", 0, 2000)
    from gpt2_scratch.model.gpt2 import GPT2

    torch.manual_seed(0)
    model = GPT2(TINY)
    loader = TokenLoader(tmp_path, "val", batch_size=2, seq_len=16)

    loss = validation_loss(model, loader, device="cpu", batches=2)

    assert math.isfinite(loss)
    assert model.training


def test_validation_loss_is_deterministic(tmp_path):
    make_shard(tmp_path, "val", 0, 2000)
    from gpt2_scratch.model.gpt2 import GPT2

    torch.manual_seed(0)
    model = GPT2(TINY)
    loader = TokenLoader(tmp_path, "val", batch_size=2, seq_len=16)

    first = validation_loss(model, loader, device="cpu", batches=2)
    second = validation_loss(model, loader, device="cpu", batches=2)

    assert first == pytest.approx(second)


def test_train_writes_metrics_and_a_final_model(tmp_path):
    root = make_data_root(tmp_path / "data")
    cfg = TrainConfig(
        total_tokens=4 * 64,
        total_batch_size=64,
        micro_batch_size=2,
        sequence_length=16,
        warmup_steps=1,
        val_every=2,
        val_batches=2,
        use_compile=False,
    )
    out = tmp_path / "run"

    summary = train(cfg, ARMS["soft"], TINY, root, out, device="cpu")

    records = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]
    train_steps = [r["step"] for r in records if "train_loss" in r]
    val_steps = [r["step"] for r in records if "val_loss" in r]
    assert train_steps == [0, 1, 2, 3]
    # The final evaluation comes after the last update, at step 4.
    assert val_steps == [0, 2, 4]
    assert (out / "model.pt").exists()
    assert summary["steps"] == 4


def test_train_uses_the_arm_source_directory(tmp_path):
    root = make_data_root(tmp_path / "data")
    cfg = TrainConfig(
        total_tokens=64,
        total_batch_size=64,
        micro_batch_size=2,
        sequence_length=16,
        warmup_steps=1,
        val_every=1,
        val_batches=1,
        use_compile=False,
    )

    summary = train(cfg, ARMS["hard"], TINY, root, tmp_path / "run", device="cpu")

    assert summary["steps"] == 1


def test_train_records_hellaswag_accuracy_when_examples_are_given(tmp_path):
    root = make_data_root(tmp_path / "data")
    cfg = TrainConfig(
        total_tokens=64,
        total_batch_size=64,
        micro_batch_size=2,
        sequence_length=16,
        warmup_steps=1,
        val_every=1,
        val_batches=1,
        use_compile=False,
    )
    examples = [{"ctx": "ab", "endings": ["cd", "ef", "gh", "ij"], "label": 0}]
    out = tmp_path / "run"

    summary = train(
        cfg,
        ARMS["uniform"],
        TINY,
        root,
        out,
        device="cpu",
        hellaswag=examples,
        encode=lambda text: list(text.encode()),
    )

    records = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]
    assert records[-1]["hellaswag_acc"] == summary["hellaswag_acc"]
    assert 0.0 <= summary["hellaswag_acc"] <= 1.0


def test_train_rejects_hellaswag_examples_without_an_encoder(tmp_path):
    root = make_data_root(tmp_path / "data")
    cfg = TrainConfig(
        total_tokens=64,
        total_batch_size=64,
        micro_batch_size=2,
        sequence_length=16,
        use_compile=False,
    )
    examples = [{"ctx": "ab", "endings": ["cd", "ef", "gh", "ij"], "label": 0}]

    with pytest.raises(ValueError, match="encode is required"):
        train(cfg, ARMS["uniform"], TINY, root, tmp_path / "run", "cpu", hellaswag=examples)


def test_train_forwards_every_metric_record_to_the_tracker(tmp_path):
    from tests.test_tracking import RecordingTracker

    root = make_data_root(tmp_path / "data")
    cfg = TrainConfig(
        total_tokens=2 * 64,
        total_batch_size=64,
        micro_batch_size=2,
        sequence_length=16,
        warmup_steps=1,
        val_every=1,
        val_batches=1,
        use_compile=False,
    )
    tracker = RecordingTracker()
    out = tmp_path / "run"

    train(cfg, ARMS["uniform"], TINY, root, out, device="cpu", tracker=tracker)

    records = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]
    assert len(tracker.logged) == len(records)


class CrashingTracker:
    """Fails once, on the training record of one step, after that record is written."""

    def __init__(self, crash_step):
        self.crash_step = crash_step

    def log(self, metrics, step):
        if "train_loss" in metrics and step == self.crash_step:
            raise RuntimeError("crash")

    def finish(self):
        pass


def test_an_interrupted_run_resumes_from_its_last_checkpoint(tmp_path):
    root = make_data_root(tmp_path / "data")
    cfg = TrainConfig(
        total_tokens=4 * 64,
        total_batch_size=64,
        micro_batch_size=2,
        sequence_length=16,
        warmup_steps=1,
        val_every=10,
        val_batches=1,
        checkpoint_every=2,
        use_compile=False,
    )
    out = tmp_path / "run"

    with pytest.raises(RuntimeError, match="crash"):
        train(cfg, ARMS["soft"], TINY, root, out, device="cpu", tracker=CrashingTracker(2))
    assert (out / "checkpoint.pt").exists()

    summary = train(cfg, ARMS["soft"], TINY, root, out, device="cpu")

    records = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]
    train_steps = [r["step"] for r in records if "train_loss" in r]
    assert train_steps == [0, 1, 2, 3]
    assert summary["steps"] == 4
    assert not (out / "checkpoint.pt").exists()


def test_loader_state_lets_a_restored_loader_continue_the_same_stream(tmp_path):
    make_shard(tmp_path, "train", 0, 200)
    loader = TokenLoader(tmp_path, "train", batch_size=2, seq_len=8)
    loader.next_batch()
    saved = loader.state()
    expected, _, _ = loader.next_batch()

    restored = TokenLoader(tmp_path, "train", batch_size=2, seq_len=8)
    restored.restore(saved)
    actual, _, _ = restored.next_batch()

    assert torch.equal(actual, expected)


def test_on_checkpoint_is_called_at_each_publish_step(tmp_path):
    root = make_data_root(tmp_path / "data")
    cfg = TrainConfig(
        total_tokens=4 * 64,
        total_batch_size=64,
        micro_batch_size=2,
        sequence_length=16,
        warmup_steps=1,
        val_every=10,
        val_batches=1,
        checkpoint_every=1,
        publish_every=2,
        use_compile=False,
    )
    published = []

    train(
        cfg,
        ARMS["uniform"],
        TINY,
        root,
        tmp_path / "run",
        device="cpu",
        on_checkpoint=lambda out, step: published.append(step),
    )

    assert published == [2, 4]


def test_resume_refuses_a_checkpoint_from_another_configuration(tmp_path):
    from dataclasses import replace

    root = make_data_root(tmp_path / "data")
    cfg = TrainConfig(
        total_tokens=4 * 64,
        total_batch_size=64,
        micro_batch_size=2,
        sequence_length=16,
        warmup_steps=1,
        val_every=10,
        val_batches=1,
        checkpoint_every=2,
        use_compile=False,
    )
    out = tmp_path / "run"
    with pytest.raises(RuntimeError, match="crash"):
        train(cfg, ARMS["soft"], TINY, root, out, device="cpu", tracker=CrashingTracker(2))

    with pytest.raises(ValueError, match="another run"):
        train(replace(cfg, max_lr=1e-3), ARMS["soft"], TINY, root, out, device="cpu")
