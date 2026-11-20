import json

from gpt2_scratch.training.metrics import MetricsLog
from tests.test_tracking import RecordingTracker


def test_a_fresh_log_starts_empty(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text('{"step": 0}\n')

    MetricsLog(path)

    assert path.read_text() == ""


def test_resuming_keeps_only_the_records_before_the_resume_step(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text('{"step": 0}\n{"step": 1}\n{"step": 2}\n')

    log = MetricsLog(path, keep_before=2)
    log.write(step=2, train_loss=1.0)

    steps = [json.loads(line)["step"] for line in path.read_text().splitlines()]
    assert steps == [0, 1, 2]


def test_resuming_without_an_earlier_file_starts_empty(tmp_path):
    path = tmp_path / "metrics.jsonl"

    MetricsLog(path, keep_before=5)

    assert path.read_text() == ""


def test_write_appends_a_record_and_forwards_it_to_the_tracker(tmp_path):
    path = tmp_path / "metrics.jsonl"
    tracker = RecordingTracker()
    log = MetricsLog(path, tracker)

    log.write(step=3, val_loss=2.5)

    assert json.loads(path.read_text()) == {"step": 3, "val_loss": 2.5}
    assert tracker.logged == [(3, {"val_loss": 2.5})]
