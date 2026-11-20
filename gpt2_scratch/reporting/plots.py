"""Figures comparing the ordering arms: losses, score quartiles and benchmarks."""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BUCKET_KEYS = ["val_loss_25_30", "val_loss_30_35", "val_loss_35_40", "val_loss_40_50"]
BENCHMARK_KEYS = ["hellaswag_acc", "piqa_acc", "arc_easy_acc"]


def load_metrics(path: Path) -> list[dict]:
    """Read every record of a metrics.jsonl file."""
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def plot_runs(runs: Mapping[str, Path], out_dir: Path) -> list[Path]:
    """Plot the curves of every run.

    Args:
        runs: Arm name to the folder holding its metrics.jsonl.
        out_dir: Folder for the figures; created if missing.

    Returns:
        The written figure paths.
    """
    metrics = {name: load_metrics(Path(folder) / "metrics.jsonl") for name, folder in runs.items()}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    return [
        _lines(metrics, ["train_loss"], "Training loss", "Loss", out / "training_loss.png"),
        _lines(metrics, ["val_loss"], "Validation loss", "Loss", out / "validation_loss.png"),
        _lines(
            metrics,
            BUCKET_KEYS,
            "Validation loss by FineWeb-Edu score bucket",
            "Loss",
            out / "score_buckets.png",
        ),
        _lines(
            metrics,
            BENCHMARK_KEYS,
            "Benchmark accuracy (normalized)",
            "Accuracy",
            out / "benchmarks.png",
        ),
    ]


def _series(records: list[dict], key: str) -> tuple[list[int], list[float]]:
    points = [(r["step"], r[key]) for r in records if key in r]
    return [step for step, _ in points], [value for _, value in points]


def _lines(
    metrics: Mapping[str, list[dict]],
    keys: Sequence[str],
    title: str,
    ylabel: str,
    path: Path,
) -> Path:
    figure, axis = plt.subplots(figsize=(10, 5))
    for name, records in metrics.items():
        for key in keys:
            label = name if len(keys) == 1 else f"{name} {key}"
            axis.plot(*_series(records, key), marker="o", label=label)
    axis.set_title(title)
    axis.set_xlabel("Step")
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.3)
    axis.legend()
    figure.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(figure)
    return path
