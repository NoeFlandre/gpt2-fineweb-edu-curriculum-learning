import threading

from gpt2_scratch.training import tracking
from gpt2_scratch.training.metrics import MetricsLog


class RecordingTracker:
    def __init__(self):
        self.logged = []
        self.finished = False

    def log(self, metrics, step):
        self.logged.append((step, metrics))

    def finish(self):
        self.finished = True


def test_metrics_log_forwards_each_record_to_the_tracker_with_its_step(tmp_path):
    tracker = RecordingTracker()
    log = MetricsLog(tmp_path / "metrics.jsonl", tracker=tracker)

    log.write(step=3, train_loss=1.5, lr=0.1)

    assert tracker.logged == [(3, {"train_loss": 1.5, "lr": 0.1})]


def test_metrics_log_without_a_tracker_only_writes_the_file(tmp_path):
    log = MetricsLog(tmp_path / "metrics.jsonl")

    log.write(step=0, val_loss=9.0)

    assert (tmp_path / "metrics.jsonl").read_text() == '{"step": 0, "val_loss": 9.0}\n'


class FakeTrackio:
    def __init__(self):
        self.init_kwargs: dict = {}
        self.logs: list[tuple[dict, int | None]] = []
        self.finished = False
        self.sync_calls: list[tuple] = []

    def init(self, **kwargs):
        self.init_kwargs = kwargs

    def log(self, metrics, step=None):
        self.logs.append((metrics, step))

    def finish(self):
        self.finished = True

    def sync(self, project, space_id=None, bucket_id=None, sdk=None, private=None):
        self.sync_calls.append((project, space_id, bucket_id, sdk, private))
        return "https://huggingface.co/spaces/user/space"


def test_tracker_starts_a_local_run_and_forwards_logs(monkeypatch):
    fake = FakeTrackio()
    monkeypatch.setattr(tracking, "trackio", fake)

    tracker = tracking.TrackioTracker.start(
        project="gpt2",
        name="soft",
        config={"lr": 6e-4},
        space_id="user/space",
        bucket_id="user/bucket",
    )
    tracker.log({"val_loss": 9.0}, step=0)

    assert fake.init_kwargs == {"project": "gpt2", "name": "soft", "config": {"lr": 6e-4}}
    assert fake.logs == [({"val_loss": 9.0}, 0)]


def test_tracker_publishes_the_dashboard_after_every_training_step(monkeypatch):
    fake = FakeTrackio()
    monkeypatch.setattr(tracking, "trackio", fake)
    tracker = tracking.TrackioTracker.start(
        project="gpt2", name="soft", config={}, space_id="user/space", bucket_id="user/bucket"
    )

    tracker.log({"val_loss": 9.0}, step=0)
    assert fake.sync_calls == []

    tracker.log({"train_loss": 2.0, "lr": 0.1}, step=0)
    tracker.log({"train_loss": 1.9, "lr": 0.1}, step=1)

    assert fake.sync_calls == [
        ("gpt2", "user/space", "user/bucket", "static", False),
        ("gpt2", "user/space", "user/bucket", "static", False),
    ]


def test_tracker_publishes_once_more_when_the_run_finishes(monkeypatch):
    fake = FakeTrackio()
    monkeypatch.setattr(tracking, "trackio", fake)
    tracker = tracking.TrackioTracker.start(
        project="gpt2", name="soft", config={}, space_id="user/space", bucket_id="user/bucket"
    )

    tracker.finish()

    assert fake.finished
    assert len(fake.sync_calls) == 1


def test_publish_syncs_the_project_as_a_public_static_dashboard(monkeypatch):
    fake = FakeTrackio()
    monkeypatch.setattr(tracking, "trackio", fake)

    url = tracking.publish("gpt2", space_id="user/space", bucket_id="user/bucket")

    assert fake.sync_calls == [("gpt2", "user/space", "user/bucket", "static", False)]
    assert url == "https://huggingface.co/spaces/user/space"


class FakeRun:
    def __init__(self, events):
        self.events = events
        self._client_lock = threading.Lock()

    def _flush_queues_inline(self):
        self.events.append("flush")


def test_publish_flushes_the_pending_metrics_before_syncing(monkeypatch):
    from trackio import context_vars

    events = []
    fake = FakeTrackio()
    fake.sync = lambda *args, **kwargs: events.append("sync") or "url"
    monkeypatch.setattr(tracking, "trackio", fake)
    token = context_vars.current_run.set(FakeRun(events))  # ty: ignore[invalid-argument-type]
    try:
        tracking.TrackioTracker("gpt2", "user/space", "user/bucket")._publish()
    finally:
        context_vars.current_run.reset(token)

    assert events == ["flush", "sync"]
