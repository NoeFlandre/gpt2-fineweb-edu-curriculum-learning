import json
import sys

import numpy as np
from gpt2_scratch.data.shards import list_shards, load_shard
from gpt2_scratch.publishing import MODEL_REPO
from gpt2_scratch.training.tracking import TRACKIO_BUCKET, TRACKIO_PROJECT, TRACKIO_SPACE
from scripts import prepare_benchmarks, prepare_data, train, upload

EOT = 50256


def encode(text):
    return np.array([EOT, *[ord(char) % 200 for char in text]], dtype=np.uint16)


def edu_row(index):
    return {"text": f"document number {index}", "int_score": 3, "score": 3.0 + index / 100}


def test_piqa_rows_become_two_ending_examples():
    row = {"goal": "Cook rice", "sol1": "boil water", "sol2": "freeze it", "label": 0}

    example = prepare_benchmarks.piqa_example(row)

    assert example == {"ctx": "Cook rice", "endings": ["boil water", "freeze it"], "label": 0}


def test_arc_rows_put_the_question_before_each_choice_and_find_the_answer_position():
    row = {
        "question": "What warms a room?",
        "choices": {"text": ["ice", "heater", "snow"], "label": ["A", "B", "C"]},
        "answerKey": "B",
    }

    example = prepare_benchmarks.arc_example(row)

    assert example == {
        "ctx": "Question: What warms a room?\nAnswer:",
        "endings": ["ice", "heater", "snow"],
        "label": 1,
    }


def test_write_examples_writes_one_json_object_per_line(tmp_path):
    path = tmp_path / "piqa.jsonl"

    prepare_benchmarks.write_examples(path, [{"ctx": "a", "endings": ["b"], "label": 0}])

    assert json.loads(path.read_text()) == {"ctx": "a", "endings": ["b"], "label": 0}


def test_build_study_writes_each_arm_and_the_validation_split(tmp_path):
    rows = [edu_row(index) for index in range(40)]

    prepare_data.build_study(rows, encode, tmp_path, pool_tokens=60, val_tokens=20, seed=1)

    for arm in ("shuffle", "ascending", "descending", "cooldown"):
        tokens = load_shard(list_shards(tmp_path / "arms" / arm, "train")[0])
        assert len(tokens) == 60
    validation = load_shard(list_shards(tmp_path / "validation", "val")[0])
    buckets = np.load(tmp_path / "validation" / "val_buckets.npy")
    assert len(validation) == len(buckets) == 20
    report = json.loads((tmp_path / "validation" / "buckets.json").read_text())
    assert report["edges"] == [3.0, 3.5, 4.0]
    assert sum(report["tokens"]) == 20


def test_train_main_wires_the_arm_benchmarks_and_tracker_into_the_runner(monkeypatch):
    captured = {}

    class FakeTracker:
        @classmethod
        def start(cls, project, name, config, space_id, bucket_id):
            captured.update(project=project, name=name, space=space_id, bucket=bucket_id)
            return cls()

        def finish(self):
            captured["finished"] = True

    def fake_train(
        cfg, arm, model_config, data, out, device, benchmarks, encode, tracker, on_checkpoint
    ):
        captured.update(arm=arm, out=out, benchmarks=sorted(benchmarks), device=device)
        assert on_checkpoint is None
        return {"steps": 0}

    monkeypatch.setattr(train, "TrackioTracker", FakeTracker)
    monkeypatch.setattr(train, "train", fake_train)
    monkeypatch.setattr(train, "load_examples", lambda path: [{"label": 0}])
    monkeypatch.setattr(
        sys,
        "argv",
        ["train", "--arm", "cooldown", "--out", "runs/cooldown", "--device", "cpu", "--no-hub"],
    )

    train.main()

    assert captured["arm"] == "cooldown"
    assert captured["benchmarks"] == ["arc_easy", "hellaswag", "piqa"]
    assert captured["project"] == TRACKIO_PROJECT
    assert captured["finished"] is True


def test_upload_main_uploads_the_runs_and_card_then_publishes_the_dashboard(monkeypatch, tmp_path):
    uploads = []
    published = []

    class FakeApi:
        def upload_folder(self, **kwargs):
            uploads.append(("folder", kwargs))

        def upload_file(self, **kwargs):
            uploads.append(("file", kwargs))

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

    assert [kind for kind, _ in uploads] == ["folder", "file"]
    assert uploads[0][1]["repo_id"] == MODEL_REPO
    assert uploads[1][1]["path_in_repo"] == "README.md"
    assert published == [(TRACKIO_PROJECT, TRACKIO_SPACE, TRACKIO_BUCKET)]
