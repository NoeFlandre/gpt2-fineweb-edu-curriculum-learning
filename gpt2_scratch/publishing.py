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


def _commit(api, repo: str, run: str, out_dir: Path, names: tuple[str, ...], message: str) -> None:
    """Upload the named files of a run in one commit."""
    operations = [
        CommitOperationAdd(path_in_repo=remote_path(run, name), path_or_fileobj=str(out_dir / name))
        for name in names
    ]
    api.create_commit(
        repo_id=repo, repo_type="model", operations=operations, commit_message=message
    )


def publish_checkpoint(api, repo: str, arm: str, out_dir: Path, step: int) -> None:
    """Upload a run's latest checkpoint and metrics in one commit, then remove the local checkpoint.

    If the upload fails, the local checkpoint stays, so the next attempt can upload it.
    """
    _commit(api, repo, arm, out_dir, (CHECKPOINT, METRICS), f"{arm}: checkpoint at step {step}")
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


FINAL_FILES = (CHECKPOINT, "model.pt", "config.json", METRICS)


def publish_final(api, repo: str, run: str, out_dir: Path) -> None:
    """Upload a finished run's checkpoint, weights, configuration and metrics in one commit.

    run is the folder in the repository, such as init42-data42/shuffle.
    """
    _commit(api, repo, run, out_dir, FINAL_FILES, f"{run}: final checkpoint, weights and metrics")
