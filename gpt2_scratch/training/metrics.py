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


def read_records(path: Path) -> list[dict]:
    """The complete records of a metrics file; a line cut short by a killed job is skipped."""
    if not path.exists():
        return []
    records = []
    for line in path.read_text().splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def last_step(records: list[dict]) -> int:
    """The step of the last record that has one, or 0 when there is none."""
    steps = [record["step"] for record in records if "step" in record]
    return int(steps[-1]) if steps else 0


def _kept_lines(path: Path, keep_before: int) -> str:
    """The records of an earlier run that precede the step a resumed run starts from."""
    records = read_records(path) if keep_before > 0 else []
    return "".join(json.dumps(record) + "\n" for record in records if record["step"] < keep_before)


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
