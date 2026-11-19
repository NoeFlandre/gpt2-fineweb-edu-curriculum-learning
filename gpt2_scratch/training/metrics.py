"""Metrics log: one JSON object per line, forwarded to an optional experiment tracker."""

import json
from pathlib import Path
from typing import Protocol


class Tracker(Protocol):
    """Receives each metric record as it is written."""

    def log(self, metrics: dict, step: int) -> None:
        """Record metrics at a step."""

    def finish(self) -> None:
        """Close the run."""


def _kept_lines(path: Path, keep_before: int) -> str:
    """Records of an earlier run that come before the step a resumed run starts from."""
    if keep_before <= 0 or not path.exists():
        return ""
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    return "".join(json.dumps(line) + "\n" for line in lines if line["step"] < keep_before)


class MetricsLog:
    """Writes metric records to a JSON Lines file."""

    def __init__(self, path: Path, tracker: Tracker | None = None, keep_before: int = 0) -> None:
        """Start the file afresh, keeping only records of steps before keep_before.

        Records are also sent to tracker if given.
        """
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(_kept_lines(self.path, keep_before))
        self.tracker = tracker

    def write(self, **fields) -> None:
        """Append one record and forward it to the tracker."""
        with self.path.open("a") as file:
            file.write(json.dumps(fields) + "\n")
        if self.tracker is not None:
            metrics = {key: value for key, value in fields.items() if key != "step"}
            self.tracker.log(metrics, step=fields["step"])
