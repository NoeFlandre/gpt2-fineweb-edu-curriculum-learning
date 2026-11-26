"""Show the training chains of this site: each run's last metrics and its live OAR jobs.

Run on a frontend. With --busy, exit 1 when a live job name starts with one of the prefixes, which
launch.sh uses to refuse starting a chain that is already running.

Usage: python3 scripts/g5k/status.py [--runs runs] [--busy PREFIX ...]
"""

import argparse
import json
import subprocess
from pathlib import Path

from scripts.g5k.run_log import last_step, read_records

FINISHED = ("Terminated", "Error")
METRICS = "metrics.jsonl"
FINISHED_DIR = Path.home() / "gpt2-g5k" / "state"


def live_jobs(oarstat: dict | list) -> list[dict]:
    """The jobs of an `oarstat -J` listing that have not ended."""
    jobs = oarstat.values() if isinstance(oarstat, dict) else oarstat
    return [job for job in jobs if job.get("state") not in FINISHED]


def last_validation(parsed: list[dict]) -> float | None:
    """The last validation loss recorded, or None when there is none yet."""
    losses = [record["val_loss"] for record in parsed if "val_loss" in record]
    return losses[-1] if losses else None


def oarstat_jobs() -> list[dict]:
    """Live jobs of the current user on this site."""
    output = subprocess.run(
        ["oarstat", "-u", "-J"], capture_output=True, text=True, check=True
    ).stdout
    return live_jobs(json.loads(output or "{}"))


def busy(jobs: list[dict], prefixes: list[str]) -> list[dict]:
    """The live jobs whose name starts with one of the prefixes."""
    return [job for job in jobs if str(job.get("name") or "").startswith(tuple(prefixes))]


def job_id(job: dict) -> str:
    """The OAR job id of an `oarstat -J` entry."""
    return str(job.get("Job_Id", job.get("id", "?")))


def report(runs: Path, jobs: list[dict], finished_dir: Path = FINISHED_DIR) -> str:
    """One line per run with its last step and validation loss, then the live jobs."""
    lines = []
    for run in sorted(p for p in runs.glob("seed*/*") if p.is_dir()):
        label = f"{run.name}-{run.parent.name}"
        parsed = read_records(run / METRICS)
        step = last_step(parsed)
        val = last_validation(parsed)
        val_text = f"{val:.3f}" if val is not None else "-"
        done = "published" if (finished_dir / f"{label}.done").exists() else "running"
        lines.append(f"{label:<22} step {step:>5}  val_loss {val_text:>6}  {done}")
    lines.extend(f"job {job_id(job)}  {job.get('name')}  {job.get('state')}" for job in jobs)
    return "\n".join(lines) or "no runs yet"


def main() -> None:
    """Print the report, or check that no matching job is live."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--busy", nargs="+", metavar="PREFIX")
    args = parser.parse_args()
    jobs = oarstat_jobs()
    if args.busy is not None:
        running = busy(jobs, args.busy)
        for job in running:
            print(f"live job {job_id(job)}: {job.get('name')}")
        raise SystemExit(1 if running else 0)
    print(report(args.runs, jobs))


if __name__ == "__main__":
    main()
