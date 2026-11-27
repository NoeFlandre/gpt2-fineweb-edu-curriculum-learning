"""Decide whether a training chain goes on, from the progress its last job made.

An arm trains as a chain of Grid'5000 jobs. Each link queues its successor when it starts, so the
successor begins as soon as the link ends, however it ended. Before doing any work, a link asks
`begin`: it counts how many links in a row ended without a new metric record, and stops the chain
once that reaches MAX_FAILURES, so a crashing run cannot resubmit itself forever.

Usage: python3 scripts/g5k/chain.py begin RUN_DIR STATE_FILE
Prints "run" or "stop".
"""

import argparse
import json
from pathlib import Path

from gpt2_scratch.training.metrics import last_step, read_records

MAX_FAILURES = 3
METRICS = "metrics.jsonl"


def next_state(state: dict, step: int) -> dict:
    """The chain state once a link begins at `step`.

    The first link of a chain has no earlier state and starts with no failures. Otherwise the
    failure count restarts at zero when the previous link moved past the step it began at, and
    grows by one when it did not.
    """
    if "step" not in state:
        return {"step": step, "failures": 0}
    failures = 0 if step > state["step"] else state["failures"] + 1
    return {"step": step, "failures": failures}


def begin(run_dir: Path, state_file: Path, max_failures: int = MAX_FAILURES) -> str:
    """Record that a link is starting, and return "stop" once the chain has failed too often."""
    state = json.loads(state_file.read_text()) if state_file.exists() else {}
    state = next_state(state, last_step(read_records(run_dir / METRICS)))
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps(state))
    return "stop" if state["failures"] >= max_failures else "run"


def main() -> None:
    """Run the begin command and print its verdict."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["begin"])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("state_file", type=Path)
    args = parser.parse_args()
    print(begin(args.run_dir, args.state_file))


if __name__ == "__main__":
    main()
