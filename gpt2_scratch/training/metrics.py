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


class MetricsLog:
    """Writes metric records to a JSON Lines file, starting from an empty file."""

    def __init__(self, path: Path, tracker: Tracker | None = None) -> None:
        """Create or empty the file at path; records are also sent to tracker if given."""
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("")
        self.tracker = tracker

    def write(self, **fields) -> None:
        """Append one record and forward it to the tracker."""
        with self.path.open("a") as file:
            file.write(json.dumps(fields) + "\n")
        if self.tracker is not None:
            metrics = {key: value for key, value in fields.items() if key != "step"}
            self.tracker.log(metrics, step=fields["step"])
