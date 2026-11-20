"""Plot every run found under a folder.

Example: python -m scripts.report --runs runs
"""

import argparse
from pathlib import Path

from gpt2_scratch.reporting.plots import plot_runs


def run_folders(runs: Path) -> dict[str, Path]:
    """The folders under runs that hold a metrics file, by name."""
    return {path.name: path for path in sorted(runs.iterdir()) if (path / "metrics.jsonl").exists()}


def main() -> None:
    """Read the runs folder and write the figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--out", type=Path, default=Path("runs/figures"))
    args = parser.parse_args()
    for figure in plot_runs(run_folders(args.runs), args.out):
        print(figure)


if __name__ == "__main__":
    main()
