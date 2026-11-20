from pathlib import Path

import pytest
from gpt2_scratch import publishing
from huggingface_hub.utils import EntryNotFoundError

NO_CHECKPOINT = "no checkpoint"


class FakeApi:
    def __init__(self):
        self.commits = []

    def create_commit(self, repo_id, repo_type, operations, commit_message):
        self.commits.append((repo_id, repo_type, operations, commit_message))


def test_publish_checkpoint_uploads_the_checkpoint_and_metrics_in_one_commit(tmp_path):
    out = tmp_path / "uniform"
    out.mkdir()
    (out / "checkpoint.pt").write_bytes(b"weights")
    (out / "metrics.jsonl").write_text("{}\n")
    api = FakeApi()

    publishing.publish_checkpoint(api, "user/repo", "uniform", out, step=50)

    assert len(api.commits) == 1
    repo_id, repo_type, operations, message = api.commits[0]
    assert (repo_id, repo_type) == ("user/repo", "model")
    assert sorted(op.path_in_repo for op in operations) == [
        "runs/uniform/checkpoint.pt",
        "runs/uniform/metrics.jsonl",
    ]
    assert message == "uniform: checkpoint at step 50"
    assert not (out / "checkpoint.pt").exists()
    assert (out / "metrics.jsonl").exists()


def test_a_failed_checkpoint_upload_keeps_the_local_copy_for_the_next_attempt(tmp_path):
    out = tmp_path / "uniform"
    out.mkdir()
    (out / "checkpoint.pt").write_bytes(b"weights")
    (out / "metrics.jsonl").write_text("{}\n")

    class UnreachableApi:
        def create_commit(self, **_kwargs):
            message = "hub unreachable"
            raise ConnectionError(message)

    with pytest.raises(ConnectionError):
        publishing.publish_checkpoint(UnreachableApi(), "user/repo", "uniform", out, step=50)

    assert (out / "checkpoint.pt").read_bytes() == b"weights"


def test_fetch_checkpoint_reports_none_when_the_hub_has_no_checkpoint(monkeypatch, tmp_path):
    def missing(*_args, **_kwargs):
        raise EntryNotFoundError(NO_CHECKPOINT)

    monkeypatch.setattr(publishing, "hf_hub_download", missing)

    assert publishing.fetch_checkpoint("user/repo", "uniform", tmp_path) is False


def test_fetch_checkpoint_also_brings_back_the_metrics_so_the_curves_stay_whole(
    monkeypatch, tmp_path
):
    out = tmp_path / "uniform"
    fetched = []

    def download(repo, filename, *, repo_type, local_dir):
        assert (repo, repo_type) == ("user/repo", "model")
        path = Path(local_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(filename.encode())
        fetched.append(filename)
        return str(path)

    monkeypatch.setattr(publishing, "hf_hub_download", download)

    assert publishing.fetch_checkpoint("user/repo", "uniform", out) is True
    assert sorted(fetched) == ["runs/uniform/checkpoint.pt", "runs/uniform/metrics.jsonl"]
    assert (out / "metrics.jsonl").read_bytes() == b"runs/uniform/metrics.jsonl"


def test_fetch_checkpoint_moves_the_download_into_the_run_folder(monkeypatch, tmp_path):
    out = tmp_path / "uniform"

    def download(repo, filename, *, repo_type, local_dir):
        assert (repo, repo_type) == ("user/repo", "model")
        path = Path(local_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"weights" if filename.endswith("checkpoint.pt") else b"curves")
        (Path(local_dir) / ".cache").mkdir(exist_ok=True)
        return str(path)

    monkeypatch.setattr(publishing, "hf_hub_download", download)

    assert publishing.fetch_checkpoint("user/repo", "uniform", out) is True
    assert (out / "checkpoint.pt").read_bytes() == b"weights"
    assert (out / "metrics.jsonl").read_bytes() == b"curves"
    assert not (out / "runs").exists()
    assert not (out / ".cache").exists()


def test_fetch_checkpoint_keeps_a_local_checkpoint_without_downloading(monkeypatch, tmp_path):
    (tmp_path / "checkpoint.pt").write_bytes(b"local")

    def forbidden(*_args, **_kwargs):
        pytest.fail("a local checkpoint must not be downloaded again")

    monkeypatch.setattr(publishing, "hf_hub_download", forbidden)

    assert publishing.fetch_checkpoint("user/repo", "uniform", tmp_path) is True


def test_fetch_checkpoint_leaves_no_download_folders_when_the_hub_has_none(monkeypatch, tmp_path):
    out = tmp_path / "uniform"

    def missing(_repo, _filename, *, local_dir, **_kwargs):
        (Path(local_dir) / "runs" / "uniform").mkdir(parents=True)
        (Path(local_dir) / ".cache").mkdir()
        raise EntryNotFoundError(NO_CHECKPOINT)

    monkeypatch.setattr(publishing, "hf_hub_download", missing)

    assert publishing.fetch_checkpoint("user/repo", "uniform", out) is False
    assert not (out / "runs").exists()
    assert not (out / ".cache").exists()
