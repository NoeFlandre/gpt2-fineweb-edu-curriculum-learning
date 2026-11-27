"""Rebuild the Trackio dashboard from every run, and publish it to the dashboard bucket.

A training link keeps the dashboard current as it trains (SiteDashboard). A sync replaces the
bucket's dashboard files with the content of one local database, so every rebuild logs the complete
dashboard:

- the runs already in the bucket are read from its files (runs.json, configs and metrics parquet);
- the runs of the given seeds are read from their metrics files on this site, and replace any run
  of the same name;
- the bucket files are backed up under backups/<UTC time>/ first, unless --no-backup;
- every run is logged into a fresh Trackio database (TRACKIO_DIR must point at an empty folder), and
  that database is synced once, so the bucket ends up with all the runs.

The rebuilds of a site take turns on a lock in runs/, so one rebuild's read and write never
interleave with another's.

Usage: TRACKIO_DIR=<empty folder> python -m scripts.g5k.dashboard --runs runs --seeds 43
"""

import argparse
import fcntl
import io
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import pyarrow.parquet as pq
from gpt2_scratch.model.config import STUDY_CONFIG
from gpt2_scratch.training.config import TrainConfig
from gpt2_scratch.training.tracking import (
    PUBLISH_EVERY,
    TRACKIO_BUCKET,
    TRACKIO_PROJECT,
    TRACKIO_SPACE,
)
from huggingface_hub import HfApi, get_token

from scripts.g5k.run_log import read_records

DASHBOARD_FILES = ("runs.json", "metrics.parquet", "aux/configs.parquet", "settings.json")
NOT_METRICS = {"id", "run_id", "timestamp", "run_name", "step", "log_id", "space_id"}


@dataclass(frozen=True)
class Run:
    """One training run for the dashboard: its name, its configuration and its metric records."""

    name: str
    config: dict
    records: list[tuple[int, dict]]


def fetch(bucket: str, name: str, token: str | None) -> bytes:
    """The content of one file of the bucket."""
    url = f"https://huggingface.co/buckets/{bucket}/resolve/{name}"
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    return urllib.request.urlopen(request, timeout=120).read()


def backup_bucket(bucket: str, token: str | None, stamp: str) -> None:
    """Copy the dashboard files of the bucket to backups/<stamp>/ before anything replaces them."""
    additions: list[tuple[str | Path | bytes, str]] = [
        (fetch(bucket, name, token), f"backups/{stamp}/{name}") for name in DASHBOARD_FILES
    ]
    HfApi().batch_bucket_files(bucket, add=additions, token=token)


def _metrics_of(row: dict) -> dict:
    """The metric values of one row of the dashboard's metrics table."""
    values = {}
    for key, value in row.items():
        if key in NOT_METRICS or value is None:
            continue
        if isinstance(value, float) and math.isnan(value):
            continue
        values[key] = value
    return values


def bucket_runs(runs_json: list[dict], configs: list[dict], metrics: list[dict]) -> list[Run]:
    """The runs of the dashboard as the bucket files describe them."""
    configs_by_name = {row["run_name"]: row for row in configs}
    rows_by_name: dict[str, list[dict]] = {}
    for row in metrics:
        rows_by_name.setdefault(row["run_name"], []).append(row)
    runs = []
    for run in runs_json:
        name = run["name"]
        config_row = configs_by_name[name]
        config = {
            "arm": config_row["arm"],
            "train": config_row["train"],
            "model": config_row["model"],
        }
        records = []
        for row in sorted(rows_by_name.get(name, []), key=lambda r: r["step"]):
            values = _metrics_of(row)
            if values:
                records.append((int(row["step"]), values))
        runs.append(Run(name, config, records))
    return runs


def _seed_records(metrics_file: Path) -> list[tuple[int, dict]]:
    """The (step, metrics) records of one metrics file."""
    records = []
    for record in read_records(metrics_file):
        values = {key: value for key, value in record.items() if key != "step"}
        if values:
            records.append((int(record["step"]), values))
    return records


def site_runs(root: Path, seed: int) -> list[Run]:
    """The runs of one training seed, read from the metrics files under root/seed<N>/<arm>/."""
    runs = []
    for folder in sorted((root / f"seed{seed}").glob("*")):
        metrics_file = folder / "metrics.jsonl"
        if not metrics_file.exists():
            continue
        config = {
            "arm": folder.name,
            "train": asdict(TrainConfig(seed=seed)),
            "model": asdict(STUDY_CONFIG),
        }
        runs.append(Run(f"{folder.name}-seed{seed}", config, _seed_records(metrics_file)))
    return runs


def merged(from_bucket: list[Run], from_sites: list[Run]) -> list[Run]:
    """The bucket's runs, with any run that this site also has replaced by the site's version."""
    names = {run.name for run in from_sites}
    kept = [run for run in from_bucket if run.name not in names]
    return kept + from_sites


def write_dashboard(runs: list[Run], project: str, space: str, bucket: str) -> None:
    """Log every run into the local Trackio database, then sync it to the dashboard bucket once."""
    import trackio  # reads TRACKIO_DIR when it is imported, so the job sets it first

    for run in runs:
        trackio.init(project=project, name=run.name, config=run.config)
        for step, values in run.records:
            trackio.log(values, step=step)
        trackio.finish()
    trackio.sync(project, space_id=space, bucket_id=bucket, sdk="static", private=False)


def _read_parquet(bucket: str, name: str, token: str | None) -> list[dict]:
    return pq.read_table(io.BytesIO(fetch(bucket, name, token))).to_pylist()


@contextmanager
def _site_lock(runs: Path) -> Generator[None]:
    """Hold the site's dashboard lock, so that one rebuild at a time reads and writes the bucket."""
    with (runs / ".dashboard.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


class SiteDashboard:
    """The dashboard of one seed's runs on this site, kept current as they train.

    It is a metrics Tracker. Every PUBLISH_EVERY training steps, and at the end of the run, a child
    process rebuilds the dashboard from the metrics files of the seed's runs under runs (see main).
    The child runs in the background, so training does not wait for it. A rebuild still running
    when the next one is due is skipped: the next one carries its data. Output goes to
    runs/.dashboard.log.
    """

    def __init__(self, seed: int, runs: Path = Path("runs")) -> None:
        """Remember the seed to publish and the folder that holds the runs."""
        self.seed = seed
        self.runs = runs
        self.child: subprocess.Popen | None = None
        self.workdir = ""
        self.where = ""

    def log(self, metrics: dict, step: int) -> None:
        """Rebuild the dashboard every PUBLISH_EVERY training steps, unless a rebuild is running."""
        if "train_loss" in metrics and step % PUBLISH_EVERY == 0 and not self._running():
            self._publish(f"step {step}")

    def finish(self) -> None:
        """Publish the final state of the dashboard, once the rebuild in flight is done."""
        if self.child is not None:
            self.child.wait()
        child = self._publish("the end of the run")
        child.wait()
        self._settle()

    def _running(self) -> bool:
        return self.child is not None and self.child.poll() is None

    def _publish(self, where: str) -> subprocess.Popen:
        """Start a rebuild in a child process, with a fresh Trackio database."""
        self._settle()
        self.where = where
        self.workdir = tempfile.mkdtemp(prefix="trackio-")
        command = [
            sys.executable,
            "-m",
            "scripts.g5k.dashboard",
            "--runs",
            str(self.runs),
            "--seeds",
            str(self.seed),
            "--no-backup",
        ]
        with (self.runs / ".dashboard.log").open("a") as log:
            child = subprocess.Popen(
                command,
                env={**os.environ, "TRACKIO_DIR": self.workdir},
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        self.child = child
        return child

    def _settle(self) -> None:
        """Report the last rebuild if it failed, and remove its database."""
        if self.child is None:
            return
        if self.child.returncode != 0:
            print(
                f"dashboard publish deferred (exit {self.child.returncode}) at {self.where}; "
                f"see {self.runs / '.dashboard.log'}",
                flush=True,
            )
        shutil.rmtree(self.workdir, ignore_errors=True)
        self.child = None


def main() -> None:
    """Back up the dashboard, rebuild it with every run, and report what it now holds."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--bucket", default=TRACKIO_BUCKET)
    parser.add_argument("--space", default=TRACKIO_SPACE)
    parser.add_argument("--project", default=TRACKIO_PROJECT)
    parser.add_argument("--no-backup", action="store_true", help="skip the backup of the bucket")
    args = parser.parse_args()
    token = get_token()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    with _site_lock(args.runs):
        if not args.no_backup:
            backup_bucket(args.bucket, token, stamp)
        current = bucket_runs(
            json.loads(fetch(args.bucket, "runs.json", token)),
            _read_parquet(args.bucket, "aux/configs.parquet", token),
            _read_parquet(args.bucket, "metrics.parquet", token),
        )
        from_sites = [run for seed in args.seeds for run in site_runs(args.runs, seed)]
        runs = merged(current, from_sites)
        write_dashboard(runs, args.project, args.space, args.bucket)
    backup = "" if args.no_backup else f" (backup: backups/{stamp}/)"
    print(f"dashboard rebuilt with {len(runs)} runs{backup}")
    for run in runs:
        print(f"  {run.name}: {len(run.records)} steps")


if __name__ == "__main__":
    main()
