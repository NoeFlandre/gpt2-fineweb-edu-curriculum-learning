import io
import json
import sys
import zipfile

import numpy as np
from gpt2_scratch.data.loader import TokenLoader
from gpt2_scratch.data.shards import list_shards, load_shard
from gpt2_scratch.publishing import MODEL_REPO
from gpt2_scratch.training.config import TrainConfig
from gpt2_scratch.training.tracking import TRACKIO_BUCKET, TRACKIO_PROJECT, TRACKIO_SPACE
from scripts import prepare_benchmarks, prepare_data, report, train, upload

EOT = 50256


def encode(text):
    return np.array([EOT, *[ord(char) % 200 for char in text]], dtype=np.uint16)


BUCKET_SCORES = [2.6, 3.2, 3.7, 4.2]


def edu_row(index):
    score = BUCKET_SCORES[index % 4] + index / 1000
    return {"text": f"document number {index}", "int_score": round(score), "score": score}


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


def test_build_study_writes_each_arm_and_the_stratified_validation_split(tmp_path):
    rows = [edu_row(index) for index in range(40)]

    prepare_data.build_study(rows, encode, tmp_path, pool_tokens=64, val_tokens=40, seed=1)

    for arm in ("shuffle", "ascending", "descending", "cooldown"):
        tokens = load_shard(list_shards(tmp_path / "arms" / arm, "train")[0])
        assert len(tokens) == 65
    validation = load_shard(list_shards(tmp_path / "validation", "val")[0])
    buckets = np.load(tmp_path / "validation" / "val_buckets.npy")
    assert len(validation) == len(buckets) == 40
    assert np.bincount(buckets, minlength=4).tolist() == [10, 10, 10, 10]
    report = json.loads((tmp_path / "validation" / "buckets.json").read_text())
    assert report["edges"] == [3.0, 3.5, 4.0]
    assert report["tokens"] == [10, 10, 10, 10]


def test_each_arm_keeps_the_final_target_inside_its_shard(tmp_path):
    rows = [edu_row(index) for index in range(40)]
    prepare_data.build_study(rows, encode, tmp_path, pool_tokens=64, val_tokens=40, seed=1)
    tokens = load_shard(list_shards(tmp_path / "arms" / "shuffle", "train")[0])
    loader = TokenLoader(tmp_path / "arms" / "shuffle", "train", batch_size=2, seq_len=4)

    for _ in range(8):  # eight micro-batches of 2 x 4 tokens read the 64 input tokens
        _, y = loader.next_batch()

    assert y.flatten().tolist() == tokens[57:65].tolist()


def test_train_main_wires_the_arm_benchmarks_and_tracker_into_the_runner(monkeypatch):
    captured = {}

    class FakeTracker:
        @classmethod
        def start(cls, project, name, space_id, bucket_id, **_kwargs):
            captured.update(project=project, name=name, space=space_id, bucket=bucket_id)
            return cls()

        def finish(self):
            captured["finished"] = True

    def fake_train(
        _cfg, arm, _model_config, _data, out, device, *, benchmarks, on_checkpoint, **_kwargs
    ):
        captured.update(arm=arm, out=out, benchmarks=sorted(benchmarks), device=device)
        assert on_checkpoint is None
        return {"steps": 0}

    monkeypatch.setattr(train, "TrackioTracker", FakeTracker)
    monkeypatch.setattr(train, "train", fake_train)
    monkeypatch.setattr(train, "load_examples", lambda _path: [{"label": 0}])
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


def piqa_zip_bytes(rows, labels):
    """The bytes of a PIQA zip shaped like the one ybisk/piqa downloads, with a dev split."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("physicaliqa-train-dev/dev.jsonl", "\n".join(json.dumps(r) for r in rows))
        archive.writestr("physicaliqa-train-dev/dev-labels.lst", "\n".join(labels))
    return buffer.getvalue()


def piqa_archive(rows, labels):
    return zipfile.ZipFile(io.BytesIO(piqa_zip_bytes(rows, labels)))


def test_download_piqa_opens_the_zip_from_the_location_ybisk_piqa_declares(monkeypatch):
    opened = []
    zip_bytes = piqa_zip_bytes([{"goal": "g", "sol1": "a", "sol2": "b"}], ["1"])

    def fake_urlopen(url, timeout):
        opened.append((url, timeout))
        return io.BytesIO(zip_bytes)

    monkeypatch.setattr(prepare_benchmarks.urllib.request, "urlopen", fake_urlopen)

    archive = prepare_benchmarks.download_piqa()

    assert opened == [(prepare_benchmarks.PIQA_ZIP, 120)]
    assert archive.read("physicaliqa-train-dev/dev-labels.lst") == b"1"


def test_main_starts_fresh_when_the_hub_has_no_checkpoint(monkeypatch, tmp_path, capsys):
    hub = HubRecorder()
    monkeypatch.setattr(train, "fetch_checkpoint", lambda *_args: False)
    monkeypatch.setattr(train, "HfApi", lambda: hub)
    monkeypatch.setattr(train, "TrackioTracker", SilentTracker)
    monkeypatch.setattr(train, "train", lambda *_args, **_kwargs: {"steps": 0})
    monkeypatch.setattr(train, "load_examples", lambda _path: [])
    monkeypatch.setattr(
        sys, "argv", ["train", "--arm", "shuffle", "--out", str(tmp_path), "--device", "cpu"]
    )

    train.main()

    assert hub.events == [("create", MODEL_REPO, "model", True)]
    assert "resuming" not in capsys.readouterr().out


def test_piqa_validation_reads_each_dev_row_with_its_integer_label():
    rows = [
        {"goal": "Cook rice", "sol1": "boil water", "sol2": "freeze it"},
        {"goal": "Dry socks", "sol1": "hang them", "sol2": "freeze them"},
    ]

    examples = prepare_benchmarks.piqa_validation(piqa_archive(rows, ["0", "1"]))

    assert examples == [
        {"ctx": "Cook rice", "endings": ["boil water", "freeze it"], "label": 0},
        {"ctx": "Dry socks", "endings": ["hang them", "freeze them"], "label": 1},
    ]


def test_prepare_benchmarks_writes_piqa_and_arc_easy(monkeypatch, tmp_path):
    piqa_row = {"goal": "Cook rice", "sol1": "boil water", "sol2": "freeze it"}
    arc_row = {
        "question": "What warms a room?",
        "choices": {"text": ["ice", "heater"], "label": ["A", "B"]},
        "answerKey": "B",
    }
    monkeypatch.setattr(
        prepare_benchmarks, "download_piqa", lambda: piqa_archive([piqa_row], ["0"])
    )
    monkeypatch.setattr(prepare_benchmarks, "load_dataset", lambda *_args, **_kwargs: [arc_row])

    prepare_benchmarks.write_benchmarks(tmp_path)

    piqa = json.loads((tmp_path / "piqa.jsonl").read_text())
    assert piqa == {"ctx": "Cook rice", "endings": ["boil water", "freeze it"], "label": 0}
    arc = json.loads((tmp_path / "arc_easy.jsonl").read_text())
    assert arc["label"] == 1


def test_prepare_benchmarks_main_writes_into_the_output_folder(monkeypatch, tmp_path):
    written = []
    monkeypatch.setattr(prepare_benchmarks, "write_benchmarks", written.append)
    monkeypatch.setattr(sys, "argv", ["prepare_benchmarks", "--out", str(tmp_path)])

    prepare_benchmarks.main()

    assert written == [tmp_path]


def test_prepare_data_main_streams_fineweb_edu_into_the_study(monkeypatch, tmp_path):
    captured = {}

    class FakeEncoder:
        eot_token = 50256

        def encode_ordinary(self, text):
            return [ord(char) for char in text]

    class FakeStream:
        def shuffle(self, seed, buffer_size):
            captured["shuffle"] = (seed, buffer_size)
            return ["row"]

    def fake_load_dataset(dataset, **kwargs):
        captured["dataset"] = (dataset, kwargs)
        return FakeStream()

    def fake_build_study(rows, encode, out, pool_tokens, val_tokens, seed):
        captured.update(
            rows=rows,
            encoded=encode("a").tolist(),
            out=out,
            pool_tokens=pool_tokens,
            val_tokens=val_tokens,
            seed=seed,
        )

    monkeypatch.setattr(prepare_data.tiktoken, "get_encoding", lambda _name: FakeEncoder())
    monkeypatch.setattr(prepare_data, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(prepare_data, "build_study", fake_build_study)
    monkeypatch.setattr(sys, "argv", ["prepare_data", "--out", str(tmp_path)])

    prepare_data.main()

    cfg = TrainConfig()
    assert captured["dataset"] == (
        prepare_data.EDU,
        {"name": "sample-10BT", "split": "train", "streaming": True},
    )
    assert captured["shuffle"] == (cfg.seed, prepare_data.STREAM_BUFFER)
    assert captured["rows"] == ["row"]
    assert captured["encoded"] == [50256, 97]
    assert captured["out"] == tmp_path
    assert captured["pool_tokens"] == cfg.max_steps * cfg.total_batch_size
    assert captured["val_tokens"] == prepare_data.VAL_TOKENS
    assert captured["seed"] == cfg.seed


def test_run_folders_keeps_the_folders_that_hold_metrics(tmp_path):
    for name in ("shuffle", "cooldown", "figures"):
        (tmp_path / name).mkdir()
    for name in ("shuffle", "cooldown"):
        (tmp_path / name / "metrics.jsonl").write_text("{}\n")

    assert report.run_folders(tmp_path) == {
        "cooldown": tmp_path / "cooldown",
        "shuffle": tmp_path / "shuffle",
    }


def test_report_main_plots_every_run_and_lists_the_figures(monkeypatch, tmp_path, capsys):
    (tmp_path / "shuffle").mkdir()
    (tmp_path / "shuffle" / "metrics.jsonl").write_text("{}\n")
    plotted = []

    def fake_plot_runs(runs, out):
        plotted.append((runs, out))
        return [out / "training_loss.png"]

    monkeypatch.setattr(report, "plot_runs", fake_plot_runs)
    monkeypatch.setattr(
        sys, "argv", ["report", "--runs", str(tmp_path), "--out", str(tmp_path / "figs")]
    )

    report.main()

    assert plotted == [({"shuffle": tmp_path / "shuffle"}, tmp_path / "figs")]
    assert str(tmp_path / "figs" / "training_loss.png") in capsys.readouterr().out


def run_main_and_capture_config(monkeypatch, tmp_path, *options):
    received = []

    def fake_train(cfg, *_args, **_kwargs):
        received.append(cfg)
        return {"steps": 0}

    monkeypatch.setattr(train, "TrackioTracker", SilentTracker)
    monkeypatch.setattr(train, "train", fake_train)
    monkeypatch.setattr(train, "load_examples", lambda _path: [])
    argv = ["train", "--arm", "shuffle", "--out", str(tmp_path), "--device", "cpu", "--no-hub"]
    monkeypatch.setattr(sys, "argv", [*argv, *options])

    train.main()

    return received[0]


def test_train_main_passes_the_smoke_schedule_options_to_the_runner(monkeypatch, tmp_path):
    cfg = run_main_and_capture_config(
        monkeypatch,
        tmp_path,
        "--total-tokens",
        "2097152",
        "--eval-every",
        "2",
        "--checkpoint-every",
        "2",
    )

    assert (cfg.total_tokens, cfg.eval_every, cfg.checkpoint_every) == (2097152, 2, 2)


def test_train_main_uses_the_study_schedule_without_options(monkeypatch, tmp_path):
    cfg = run_main_and_capture_config(monkeypatch, tmp_path)

    assert cfg == TrainConfig()


def test_hub_checkpointer_defers_a_failed_upload(monkeypatch, tmp_path, capsys):
    published = []

    def fake_publish(_api, repo, arm, _out, step):
        if step == 4:
            rate_limit = "rate limited"
            raise RuntimeError(rate_limit)
        published.append((repo, arm, step))

    monkeypatch.setattr(train, "HfApi", lambda: "api")
    monkeypatch.setattr(train, "publish_checkpoint", fake_publish)
    checkpoint = train._hub_checkpointer("user/repo", "shuffle")

    checkpoint(tmp_path, 2)
    checkpoint(tmp_path, 4)

    assert published == [("user/repo", "shuffle", 2)]
    assert "checkpoint upload deferred (RuntimeError) at step 4" in capsys.readouterr().out


class HubRecorder:
    """Stands in for the Hub API and records the calls made while a run starts."""

    def __init__(self):
        self.events = []

    def create_repo(self, repo_id, repo_type, exist_ok):
        self.events.append(("create", repo_id, repo_type, exist_ok))


class SilentTracker:
    @classmethod
    def start(cls, **_kwargs):
        return cls()

    def finish(self):
        pass


def test_main_resumes_from_the_hub_checkpoint_and_publishes_its_checkpoints(
    monkeypatch, tmp_path, capsys
):
    hub = HubRecorder()
    published = []

    def fake_train(*_args, on_checkpoint, **_kwargs):
        on_checkpoint(tmp_path, 100)
        return {"steps": 0}

    def fake_fetch(_repo, _arm, _out):
        hub.events.append("fetch")
        return True

    monkeypatch.setattr(train, "fetch_checkpoint", fake_fetch)
    monkeypatch.setattr(train, "HfApi", lambda: hub)
    monkeypatch.setattr(
        train, "publish_checkpoint", lambda _api, _repo, _arm, _out, step: published.append(step)
    )
    monkeypatch.setattr(train, "TrackioTracker", SilentTracker)
    monkeypatch.setattr(train, "train", fake_train)
    monkeypatch.setattr(train, "load_examples", lambda _path: [])
    monkeypatch.setattr(
        sys, "argv", ["train", "--arm", "shuffle", "--out", str(tmp_path), "--device", "cpu"]
    )

    train.main()

    assert published == [100]
    assert hub.events == [("create", MODEL_REPO, "model", True), "fetch"]
    assert "resuming shuffle" in capsys.readouterr().out
