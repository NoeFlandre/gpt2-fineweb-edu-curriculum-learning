"""Reading the metric records a training run writes to its metrics file."""

import json
from pathlib import Path


def read_records(metrics: Path) -> list[dict]:
    """The complete records of a metrics file; a line cut short by a killed job is skipped."""
    if not metrics.exists():
        return []
    records = []
    for line in metrics.read_text().splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def last_step(records: list[dict]) -> int:
    """The step of the last record that has one, or 0 when there is none."""
    steps = [record["step"] for record in records if "step" in record]
    return int(steps[-1]) if steps else 0
