"""Experiment tracking with Trackio, published as a live static dashboard on Hugging Face.

Runs log to a local Trackio project. Every optimizer step, the project is synced to a
static Space backed by a Hugging Face bucket, so the dashboard stays current during training.
"""

import trackio
from trackio import context_vars

TRACKIO_PROJECT = "gpt2-fineweb-edu-experiments"
TRACKIO_SPACE = "NoeFlandre/GPT2-Fineweb-Edu-Experiments-trackio"
TRACKIO_BUCKET = "NoeFlandre/GPT2-Fineweb-Edu-Experiments-trackio-bucket"


def publish(project: str, space_id: str, bucket_id: str) -> str:
    """Sync a project to a public static dashboard Space backed by a bucket.

    Returns:
        The URL of the dashboard.
    """
    return trackio.sync(
        project, space_id=space_id, bucket_id=bucket_id, sdk="static", private=False
    )


class TrackioTracker:
    """A Trackio run whose project is published after every optimizer step."""

    def __init__(self, project: str, space_id: str, bucket_id: str) -> None:
        """Remember where the dashboard lives."""
        self.project = project
        self.space_id = space_id
        self.bucket_id = bucket_id

    @classmethod
    def start(
        cls,
        project: str,
        name: str,
        config: dict,
        space_id: str = TRACKIO_SPACE,
        bucket_id: str = TRACKIO_BUCKET,
    ) -> "TrackioTracker":
        """Start a run named name in project and remember where to publish it."""
        trackio.init(project=project, name=name, config=config)
        return cls(project, space_id, bucket_id)

    def log(self, metrics: dict, step: int) -> None:
        """Record metrics at a step; publish after each training step."""
        trackio.log(metrics, step=step)
        if "train_loss" in metrics:
            self._publish()

    def finish(self) -> None:
        """Close the run and publish its final state."""
        trackio.finish()
        self._publish()

    def _publish(self) -> None:
        _flush_pending_metrics()
        publish(self.project, self.space_id, self.bucket_id)


def _flush_pending_metrics() -> None:
    """Write queued metrics to the local database so that a publish sees the latest step.

    Trackio sends metrics from a background thread every few seconds. This forces the
    pending batch out now. It uses trackio 0.45.x internals, so trackio is pinned.
    """
    run = context_vars.current_run.get()
    if run is None:
        return
    with run._client_lock:
        run._flush_queues_inline()
