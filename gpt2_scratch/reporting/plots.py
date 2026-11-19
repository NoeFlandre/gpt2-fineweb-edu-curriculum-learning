"""Figures comparing experimental arms."""

import json
from collections.abc import Mapping
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

GPT2_BASELINE_ACCURACY = 0.294463


def load_metrics(path: Path) -> list[dict]:
    """Read every record of a metrics.jsonl file."""
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def plot_runs(runs: Mapping[str, Path], out_dir: Path) -> list[Path]:
    """Plot training loss, validation loss and HellaSwag accuracy for each run.

    Args:
        runs: Run name to the folder holding its metrics.jsonl.
        out_dir: Folder for the figures; created if missing.

    Returns:
        The written figure paths.
    """
    metrics = {name: load_metrics(Path(folder) / "metrics.jsonl") for name, folder in runs.items()}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = [
        _line_plot(metrics, "train_loss", "Training loss", "Loss", out_dir / "training_loss.png"),
        _line_plot(metrics, "val_loss", "Validation loss", "Loss", out_dir / "validation_loss.png"),
        _line_plot(
            metrics,
            "hellaswag_acc",
            "HellaSwag accuracy (normalized)",
            "Accuracy",
            out_dir / "hellaswag.png",
            baseline=GPT2_BASELINE_ACCURACY,
        ),
    ]
    return written


def _series(records: list[dict], key: str) -> tuple[list[int], list[float]]:
    points = [(r["step"], r[key]) for r in records if key in r]
    return [step for step, _ in points], [value for _, value in points]


def _line_plot(
    metrics: dict,
    key: str,
    title: str,
    ylabel: str,
    path: Path,
    baseline: float | None = None,
) -> Path:
    figure, axis = plt.subplots(figsize=(10, 5))
    for name, records in metrics.items():
        axis.plot(*_series(records, key), marker="o", label=name)
    if baseline is not None:
        axis.axhline(baseline, color="gray", linestyle="--", label="GPT-2 124M")
    axis.set_title(title)
    axis.set_xlabel("Step")
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.3)
    axis.legend()
    figure.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(figure)
    return path
