import threading

from gpt2_scratch.training import tracking


class RecordingTracker:
    """Keeps every record it receives, for assertions."""

    def __init__(self):
        self.logged = []

    def log(self, metrics, step):
        self.logged.append((step, metrics))

    def finish(self):
        pass


def test_the_names_belong_to_the_ordering_study():
    assert tracking.TRACKIO_PROJECT == "gpt2-fineweb-edu-ordering"
    assert tracking.TRACKIO_SPACE == "NoeFlandre/gpt2-fineweb-edu-ordering-trackio"
    assert tracking.TRACKIO_BUCKET == "NoeFlandre/gpt2-fineweb-edu-ordering-trackio-bucket"


def test_start_creates_a_named_trackio_run(monkeypatch):
    calls = []
    monkeypatch.setattr(tracking.trackio, "init", lambda **kwargs: calls.append(kwargs))

    tracker = tracking.TrackioTracker.start(
        project="p", name="shuffle", config={"a": 1}, space_id="s", bucket_id="b"
    )

    assert calls == [{"project": "p", "name": "shuffle", "config": {"a": 1}}]
    assert (tracker.project, tracker.space_id, tracker.bucket_id) == ("p", "s", "b")


def test_a_training_record_publishes_the_dashboard(monkeypatch):
    published = []
    monkeypatch.setattr(tracking.trackio, "log", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        tracking,
        "publish",
        lambda project, _space_id, _bucket_id: published.append(project) or "url",
    )
    tracker = tracking.TrackioTracker("p", "space", "bucket")

    tracker.log({"val_loss": 3.0}, step=0)
    assert published == []

    tracker.log({"train_loss": 3.1}, step=1)
    assert published == ["p"]


def test_finish_closes_the_run_and_publishes_its_final_state(monkeypatch):
    published = []
    monkeypatch.setattr(tracking.trackio, "finish", lambda: None)
    monkeypatch.setattr(
        tracking,
        "publish",
        lambda _project, _space_id, bucket_id: published.append(bucket_id) or "url",
    )

    tracking.TrackioTracker("p", "space", "bucket").finish()

    assert published == ["bucket"]


def test_flushing_without_an_active_run_does_nothing():
    tracking._flush_pending_metrics()


def test_flushing_pushes_the_queued_metrics_of_the_active_run(monkeypatch):
    flushed = []

    class FakeRun:
        _client_lock = threading.Lock()

        def _flush_queues_inline(self):
            flushed.append(True)

    class FakeRunVar:
        def get(self):
            return FakeRun()

    monkeypatch.setattr(tracking.context_vars, "current_run", FakeRunVar())

    tracking._flush_pending_metrics()

    assert flushed == [True]


def test_publish_syncs_the_project_as_a_public_static_space(monkeypatch):
    calls = []

    def fake_sync(project, **kwargs):
        calls.append((project, kwargs))
        return "https://dashboard"

    monkeypatch.setattr(tracking.trackio, "sync", fake_sync)

    url = tracking.publish("p", "space", "bucket")

    assert url == "https://dashboard"
    assert calls == [
        ("p", {"space_id": "space", "bucket_id": "bucket", "sdk": "static", "private": False})
    ]
