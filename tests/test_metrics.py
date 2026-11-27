from gpt2_scratch.training.metrics import MetricsLog, last_step, read_records


def test_last_step_is_the_last_record_that_has_one():
    assert last_step([{"step": 0}, {"step": 50}, {"val_loss": 4.0}]) == 50
    assert last_step([{"val_loss": 4.0}]) == 0
    assert last_step([]) == 0


def test_read_records_skips_a_line_cut_short_by_a_killed_job(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text('{"step": 0}\n{"step": 50, "val_loss": ')

    assert read_records(path) == [{"step": 0}]


def test_read_records_is_empty_without_a_file(tmp_path):
    assert read_records(tmp_path / "missing.jsonl") == []


def test_resuming_keeps_the_records_before_its_start_after_a_cut_line(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text('{"step": 0}\n{"step": 50, "train_loss": 4.')

    MetricsLog(path, keep_before=50)

    assert read_records(path) == [{"step": 0}]
