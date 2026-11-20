import json

from gpt2_scratch.reporting.plots import plot_runs


def write_run(folder, records):
    folder.mkdir(parents=True)
    (folder / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    return folder


def test_plot_runs_writes_the_training_validation_bucket_and_benchmark_figures(tmp_path):
    run = write_run(
        tmp_path / "shuffle",
        [
            {"step": 0, "train_loss": 11.0},
            {
                "step": 0,
                "val_loss": 11.0,
                "val_loss_25_30": 11.1,
                "val_loss_30_35": 11.0,
                "val_loss_35_40": 10.9,
                "val_loss_40_50": 10.8,
                "hellaswag_acc": 0.25,
                "piqa_acc": 0.5,
                "arc_easy_acc": 0.25,
            },
        ],
    )

    figures = plot_runs({"shuffle": run}, tmp_path / "figures")

    assert sorted(path.name for path in figures) == [
        "benchmarks.png",
        "score_buckets.png",
        "training_loss.png",
        "validation_loss.png",
    ]
    assert all(path.stat().st_size > 0 for path in figures)


def test_plot_runs_tolerates_runs_with_only_some_metrics(tmp_path):
    run = write_run(tmp_path / "soft", [{"step": 0, "train_loss": 11.0}])

    figures = plot_runs({"soft": run}, tmp_path / "figures")

    assert len(figures) == 4
