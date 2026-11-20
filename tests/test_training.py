import json
import math
from dataclasses import replace

import numpy as np
import pytest
from gpt2_scratch.data.shards import save_shard
from gpt2_scratch.model.config import GPT2Config
from gpt2_scratch.training.config import TrainConfig
from gpt2_scratch.training.runner import train
from gpt2_scratch.training.schedule import get_lr
from tests.test_tracking import RecordingTracker

TINY = GPT2Config(block_size=16, vocab_size=2048, n_layer=1, n_head=2, n_embd=16)
BUCKET_KEYS = {
    "val_loss",
    "val_loss_25_30",
    "val_loss_30_35",
    "val_loss_35_40",
    "val_loss_40_50",
}


def make_data_root(root, arm="shuffle"):
    save_shard(root / "arms" / arm, "train", 0, (np.arange(2000) % 2000).astype(np.uint16))
    save_shard(root / "validation", "val", 0, (np.arange(400) % 2000).astype(np.uint16))
    np.save(root / "validation" / "val_buckets.npy", (np.arange(400) % 4).astype(np.uint8))
    return root


def small_cfg(**overrides):
    settings = {
        "total_tokens": 4 * 64,
        "total_batch_size": 64,
        "micro_batch_size": 2,
        "sequence_length": 16,
        "warmup_steps": 1,
        "eval_every": 2,
        "checkpoint_every": 2,
        "publish_every": 2,
        "use_compile": False,
    }
    settings.update(overrides)
    return TrainConfig(**settings)


def read_records(out):
    return [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]


class CrashingTracker:
    """Fails once, on the training record of one step, after that record is written."""

    def __init__(self, crash_step):
        self.crash_step = crash_step

    def log(self, metrics, step):
        if "train_loss" in metrics and step == self.crash_step:
            raise RuntimeError("crash")

    def finish(self):
        pass


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


def test_the_study_has_1716_steps_of_524288_tokens_per_arm():
    cfg = TrainConfig()

    assert cfg.total_tokens == 900_000_000
    assert cfg.max_steps == 1716
    assert cfg.max_steps * cfg.total_batch_size == 899_678_208
    assert cfg.grad_accum_steps == 64


def test_the_study_warms_up_over_about_one_percent_of_its_steps():
    cfg = TrainConfig()

    assert cfg.warmup_steps == 18
    assert get_lr(cfg.warmup_steps - 1, cfg) == pytest.approx(cfg.max_lr)
    assert get_lr(cfg.max_steps, cfg) == pytest.approx(cfg.min_lr)


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


def test_train_records_curves_at_every_evaluation_and_a_final_model(tmp_path):
    root = make_data_root(tmp_path / "data")
    out = tmp_path / "run"

    summary = train(small_cfg(), "shuffle", TINY, root, out, device="cpu")

    records = read_records(out)
    assert [r["step"] for r in records if "train_loss" in r] == [0, 1, 2, 3]
    assert [r["step"] for r in records if "val_loss" in r] == [0, 2, 4]
    assert (out / "model.pt").exists()
    assert summary["steps"] == 4


def test_every_evaluation_records_the_overall_and_the_four_score_bucket_losses(tmp_path):
    root = make_data_root(tmp_path / "data")
    out = tmp_path / "run"

    train(small_cfg(), "shuffle", TINY, root, out, device="cpu")

    evaluation = next(r for r in read_records(out) if "val_loss" in r)
    assert set(evaluation) >= BUCKET_KEYS


def test_benchmark_accuracies_are_recorded_under_their_names(tmp_path):
    root = make_data_root(tmp_path / "data")
    out = tmp_path / "run"
    examples = {"piqa": [{"ctx": "ab", "endings": ["cd", "ef"], "label": 0}]}

    summary = train(
        small_cfg(),
        "shuffle",
        TINY,
        root,
        out,
        device="cpu",
        benchmarks=examples,
        encode=lambda text: list(text.encode()),
    )

    final = next(r for r in reversed(read_records(out)) if "piqa_acc" in r)
    assert final["piqa_acc"] == summary["piqa_acc"]
    assert 0.0 <= summary["piqa_acc"] <= 1.0


def test_benchmarks_need_an_encoder(tmp_path):
    root = make_data_root(tmp_path / "data")
    examples = {"piqa": [{"ctx": "ab", "endings": ["cd", "ef"], "label": 0}]}

    with pytest.raises(ValueError, match="encode is required"):
        train(small_cfg(), "shuffle", TINY, root, tmp_path / "run", "cpu", benchmarks=examples)


def test_train_forwards_every_metric_record_to_the_tracker(tmp_path):
    root = make_data_root(tmp_path / "data")
    tracker = RecordingTracker()
    out = tmp_path / "run"

    train(small_cfg(), "shuffle", TINY, root, out, device="cpu", tracker=tracker)

    assert len(tracker.logged) == len(read_records(out))


def test_train_uses_the_requested_arm_directory(tmp_path):
    root = make_data_root(tmp_path / "data", arm="cooldown")

    summary = train(small_cfg(), "cooldown", TINY, root, tmp_path / "run", device="cpu")

    assert summary["steps"] == 4


def test_an_interrupted_run_resumes_from_its_last_checkpoint(tmp_path):
    root = make_data_root(tmp_path / "data")
    out = tmp_path / "run"
    cfg = small_cfg(eval_every=10)

    with pytest.raises(RuntimeError, match="crash"):
        train(cfg, "shuffle", TINY, root, out, device="cpu", tracker=CrashingTracker(2))
    assert (out / "checkpoint.pt").exists()

    summary = train(cfg, "shuffle", TINY, root, out, device="cpu")

    assert [r["step"] for r in read_records(out) if "train_loss" in r] == [0, 1, 2, 3]
    assert summary["steps"] == 4
    assert not (out / "checkpoint.pt").exists()


def test_resume_refuses_a_checkpoint_from_another_configuration(tmp_path):
    root = make_data_root(tmp_path / "data")
    cfg = small_cfg(eval_every=10)
    out = tmp_path / "run"
    with pytest.raises(RuntimeError, match="crash"):
        train(cfg, "shuffle", TINY, root, out, device="cpu", tracker=CrashingTracker(2))

    with pytest.raises(ValueError, match="another run"):
        train(replace(cfg, max_lr=1e-3), "shuffle", TINY, root, out, device="cpu")


def test_on_checkpoint_is_called_at_each_publish_step(tmp_path):
    root = make_data_root(tmp_path / "data")
    published = []

    train(
        small_cfg(checkpoint_every=1, publish_every=2),
        "shuffle",
        TINY,
        root,
        tmp_path / "run",
        device="cpu",
        on_checkpoint=lambda out, step: published.append(step),
    )

    assert published == [2, 4]


def test_losses_are_finite_at_every_evaluation(tmp_path):
    root = make_data_root(tmp_path / "data")
    out = tmp_path / "run"

    train(small_cfg(), "shuffle", TINY, root, out, device="cpu")

    assert all(math.isfinite(r["val_loss"]) for r in read_records(out) if "val_loss" in r)
