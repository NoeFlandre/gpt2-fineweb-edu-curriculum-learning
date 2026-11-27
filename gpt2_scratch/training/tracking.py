"""Experiment tracking with Trackio, published as a live static dashboard on Hugging Face.

Every metric is logged as it is produced. Every PUBLISH_EVERY training steps, and at the end, the
project is synced to a static Space backed by a Hugging Face bucket, so the dashboard stays current
during training.
"""

import trackio
from trackio import context_vars

TRACKIO_PROJECT = "gpt2-fineweb-edu-curriculum-learning"
TRACKIO_SPACE = "NoeFlandre/gpt2-fineweb-edu-curriculum-learning"
TRACKIO_BUCKET = "NoeFlandre/gpt2-fineweb-edu-curriculum-learning"
PUBLISH_EVERY = 50


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
        group: str | None = None,
    ) -> "TrackioTracker":
        """Start a run named name in project, in group, and remember where to publish it."""
        trackio.init(project=project, name=name, group=group, config=config)
        return cls(project, space_id, bucket_id)

    def log(self, metrics: dict, step: int) -> None:
        """Record metrics at a step; publish every PUBLISH_EVERY training steps."""
        trackio.log(metrics, step=step)
        if "train_loss" in metrics and step % PUBLISH_EVERY == 0:
            self._publish(f"step {step}")

    def finish(self) -> None:
        """Close the run and publish its final state."""
        trackio.finish()
        self._publish("the end of the run")

    def _publish(self, where: str) -> None:
        """Sync the dashboard; a failed sync is reported and the next one tries again."""
        try:
            _flush_pending_metrics()
            publish(self.project, self.space_id, self.bucket_id)
        except Exception as error:  # the run goes on; the next sync tries again
            print(f"dashboard sync deferred ({type(error).__name__}) at {where}", flush=True)


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
