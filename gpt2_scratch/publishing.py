"""Keeping runs on the Hugging Face model repository, so that they can be resumed from there."""

import shutil
from pathlib import Path

from huggingface_hub import CommitOperationAdd, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError

MODEL_REPO = "NoeFlandre/gpt2-fineweb-edu-curriculum-learning"
CHECKPOINT = "checkpoint.pt"
METRICS = "metrics.jsonl"


def remote_path(arm: str, name: str) -> str:
    """Where a run file lives inside the model repository."""
    return f"runs/{arm}/{name}"


def publish_checkpoint(api, repo: str, arm: str, out_dir: Path, step: int) -> None:
    """Upload a run's latest checkpoint and metrics in one commit, then remove the local checkpoint.

    If the upload fails, the local checkpoint stays, so the next attempt can upload it.
    """
    operations = [
        CommitOperationAdd(path_in_repo=remote_path(arm, name), path_or_fileobj=str(out_dir / name))
        for name in (CHECKPOINT, METRICS)
    ]
    api.create_commit(
        repo_id=repo,
        repo_type="model",
        operations=operations,
        commit_message=f"{arm}: checkpoint at step {step}",
    )
    (out_dir / CHECKPOINT).unlink()


def fetch_checkpoint(repo: str, arm: str, out_dir: Path) -> bool:
    """Download a run's checkpoint and metrics when its local folder has no checkpoint.

    The metrics come with the checkpoint, so the curves before the resume step are kept.

    Returns:
        True when the folder has a checkpoint afterwards, False when the Hub has none either.
    """
    if (out_dir / CHECKPOINT).exists():
        return True
    try:
        downloaded = {
            name: hf_hub_download(
                repo, remote_path(arm, name), repo_type="model", local_dir=out_dir
            )
            for name in (CHECKPOINT, METRICS)
        }
    except EntryNotFoundError:
        _discard_download(out_dir)
        return False
    for name, path in downloaded.items():
        Path(path).replace(out_dir / name)
    _discard_download(out_dir)
    return True


def _discard_download(out_dir: Path) -> None:
    """Remove the folders that a download into out_dir leaves behind."""
    shutil.rmtree(out_dir / "runs", ignore_errors=True)
    shutil.rmtree(out_dir / ".cache", ignore_errors=True)
