import json
import sys
from pathlib import Path

import numpy as np
import tiktoken
from gpt2_scratch.data.shards import load_shard
from gpt2_scratch.data.tokenize import encode_document
from gpt2_scratch.publishing import MODEL_REPO
from gpt2_scratch.training.tracking import TRACKIO_BUCKET, TRACKIO_PROJECT, TRACKIO_SPACE
from scripts import prepare_data, report, train, upload


class FakeScorer:
    def __init__(self):
        self.calls = []

    def score(self, texts):
        self.calls.append(len(texts))
        return np.full(len(texts), 4, dtype=np.uint8)

    @classmethod
    def from_pretrained(cls, device):
        return cls()


def fake_load_dataset(repo, name, split, streaming):
    assert name == "sample-10BT" and split == "train" and streaming
    if repo == prepare_data.FINEWEB:
        return [{"text": "raw document one"}, {"text": "raw document two"}] * 80
    return [{"text": "edu document", "int_score": 3}] * 200


def test_prepare_parse_args_defaults(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prepare_data"])

    args = prepare_data.parse_args()

    assert args.train_tokens == 100_000_000
    assert args.val_tokens == 2_000_000
    assert args.out == Path("data/arms")


def test_score_texts_batches_and_joins_the_scores():
    scorer = FakeScorer()
    texts = ["x"] * 150

    scores = prepare_data.score_texts(scorer, texts)

    assert scorer.calls == [64, 64, 22]
    assert len(scores) == 150


def test_score_texts_handles_no_documents():
    assert prepare_data.score_texts(FakeScorer(), []).tolist() == []


def test_prepare_main_writes_raw_and_edu_shards(monkeypatch, tmp_path):
    monkeypatch.setattr(prepare_data, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(prepare_data, "EducationalScorer", FakeScorer)
    monkeypatch.setattr(
        sys,
        "argv",
        ["prepare_data", "--out", str(tmp_path), "--train-tokens", "300", "--val-tokens", "100"],
    )

    prepare_data.main()

    raw_train_tokens, raw_train_scores = load_shard(tmp_path / "raw" / "train_000000.npy")
    assert len(raw_train_tokens) == 300
    assert set(raw_train_scores.tolist()) == {4}
    assert (tmp_path / "raw" / "val_000000.npy").exists()
    edu_tokens, edu_scores = load_shard(tmp_path / "edu" / "train_000000.npy")
    assert len(edu_tokens) == 300
    assert set(edu_scores.tolist()) == {3}


def test_report_main_plots_every_run_folder(monkeypatch, tmp_path):
    for name in ["soft", "uniform"]:
        folder = tmp_path / "runs" / name
        folder.mkdir(parents=True)
        (folder / "metrics.jsonl").write_text(
            json.dumps({"step": 0, "train_loss": 9.0})
            + "\n"
            + json.dumps({"step": 0, "val_loss": 9.5})
            + "\n"
        )
    (tmp_path / "runs" / "figures").mkdir()
    monkeypatch.setattr(
        sys, "argv", ["report", "--runs", str(tmp_path / "runs"), "--out", str(tmp_path / "figs")]
    )

    report.main()

    assert (tmp_path / "figs" / "training_loss.png").exists()


def test_upload_main_uploads_the_runs_then_publishes_the_dashboard(monkeypatch, tmp_path):
    uploads = []
    published = []

    class FakeApi:
        def upload_folder(self, **kwargs):
            uploads.append(kwargs)

        def upload_file(self, **kwargs):
            uploads.append(kwargs)

    monkeypatch.setattr(upload, "HfApi", FakeApi)
    monkeypatch.setattr(
        upload,
        "publish",
        lambda project, space_id, bucket_id: (
            published.append((project, space_id, bucket_id)) or "url"
        ),
    )
    monkeypatch.setattr(sys, "argv", ["upload", "--runs", str(tmp_path)])

    upload.main()

    assert uploads[0]["path_in_repo"] == "runs"
    assert uploads[0]["repo_id"] == MODEL_REPO
    assert published == [(TRACKIO_PROJECT, TRACKIO_SPACE, TRACKIO_BUCKET)]


def test_train_parse_args_reads_the_arm_and_output(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train", "--arm", "soft", "--out", "runs/soft"])

    args = train.parse_args()

    assert args.arm == "soft"
    assert args.out == Path("runs/soft")


def test_train_main_wires_the_arm_and_tracker_into_the_runner(monkeypatch):
    captured = {}

    class FakeTracker:
        @classmethod
        def start(cls, project, name, config, space_id, bucket_id):
            captured.update(project=project, name=name)
            return cls()

        def finish(self):
            captured["finished"] = True

    def fake_train(
        cfg, arm, model_config, data, out, device, hellaswag, encode, tracker, on_checkpoint
    ):
        assert on_checkpoint is None
        captured.update(arm=arm.name, out=out, examples=len(hellaswag), device=device)
        assert isinstance(tracker, FakeTracker)
        return {"steps": 0}

    monkeypatch.setattr(train, "TrackioTracker", FakeTracker)
    monkeypatch.setattr(train, "train", fake_train)
    monkeypatch.setattr(train, "load_examples", lambda path: [{"label": 0}] * 3)
    monkeypatch.setattr(
        sys,
        "argv",
        ["train", "--arm", "hard", "--out", "runs/hard", "--device", "cpu", "--no-hub"],
    )

    train.main()

    assert captured["arm"] == "hard"
    assert captured["name"] == "hard"
    assert captured["project"] == TRACKIO_PROJECT
    assert captured["finished"] is True
    assert captured["examples"] == 3


def test_score_texts_reports_progress_every_two_hundred_batches(monkeypatch, capsys):
    monkeypatch.setattr(prepare_data, "SCORE_BATCH", 1)

    prepare_data.score_texts(FakeScorer(), ["x"] * 200)

    assert "scored 200 of 200 documents" in capsys.readouterr().out


def test_prepare_main_keeps_a_short_edu_stream_as_is(monkeypatch, tmp_path):
    def short_edu(repo, name, split, streaming):
        if repo == prepare_data.FINEWEB_EDU:
            return [{"text": "edu document", "int_score": 3}] * 5
        return fake_load_dataset(repo, name, split, streaming)

    monkeypatch.setattr(prepare_data, "load_dataset", short_edu)
    monkeypatch.setattr(prepare_data, "EducationalScorer", FakeScorer)
    monkeypatch.setattr(
        sys,
        "argv",
        ["prepare_data", "--out", str(tmp_path), "--train-tokens", "300", "--val-tokens", "100"],
    )

    prepare_data.main()

    edu_tokens, edu_scores = load_shard(tmp_path / "edu" / "train_000000.npy")
    assert len(edu_tokens) == len(edu_scores) == 5 * len(_edu_document_tokens())


def _edu_document_tokens():
    return encode_document("edu document", tiktoken.get_encoding("gpt2"))
