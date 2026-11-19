import json

from gpt2_scratch.reporting.plots import load_metrics, plot_runs

PNG_MAGIC = b"\x89PNG"


def write_run(directory, val_losses, accuracy):
    directory.mkdir(parents=True)
    records = [{"step": s, "train_loss": 10.0 - s} for s in range(len(val_losses))]
    records += [{"step": s, "val_loss": v} for s, v in enumerate(val_losses)]
    records.append({"step": len(val_losses) - 1, "hellaswag_acc": accuracy})
    (directory / "metrics.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return directory


def test_load_metrics_returns_every_record(tmp_path):
    run = write_run(tmp_path / "soft", [5.0, 4.0], 0.27)

    records = load_metrics(run / "metrics.jsonl")

    assert len(records) == 5
    assert records[0] == {"step": 0, "train_loss": 10.0}


def test_plot_runs_writes_three_png_figures(tmp_path):
    runs = {
        "uniform": write_run(tmp_path / "uniform", [5.0, 4.0], 0.26),
        "soft": write_run(tmp_path / "soft", [5.0, 3.5], 0.28),
    }
    out = tmp_path / "figures"

    written = plot_runs(runs, out)

    assert sorted(path.name for path in written) == [
        "hellaswag.png",
        "training_loss.png",
        "validation_loss.png",
    ]
    for path in written:
        assert path.read_bytes()[:4] == PNG_MAGIC


def test_hellaswag_bars_skip_runs_without_a_score(tmp_path):
    runs = {
        "scored": write_run(tmp_path / "scored", [5.0], 0.27),
        "unscored": tmp_path / "unscored",
    }
    runs["unscored"].mkdir()
    (runs["unscored"] / "metrics.jsonl").write_text('{"step": 0, "train_loss": 9.0}\n')

    written = plot_runs(runs, tmp_path / "figures")

    assert "hellaswag.png" in [path.name for path in written]
