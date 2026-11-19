"""Keeping runs on the Hugging Face model repository, so that they can be resumed from there."""

import shutil
from pathlib import Path

from huggingface_hub import CommitOperationAdd, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError

MODEL_REPO = "NoeFlandre/GPT2-Fineweb-Edu-Experiments"
CHECKPOINT = "checkpoint.pt"
METRICS = "metrics.jsonl"


def remote_path(arm: str, name: str) -> str:
    """Where a run file lives inside the model repository."""
    return f"runs/{arm}/{name}"


def publish_checkpoint(api, repo: str, arm: str, out_dir: Path, step: int) -> None:
    """Upload a run's latest checkpoint and metrics in one commit."""
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


def fetch_checkpoint(repo: str, arm: str, out_dir: Path) -> bool:
    """Download a run's checkpoint when its local folder has none.

    Returns:
        True when the folder has a checkpoint afterwards, False when the Hub has none either.
    """
    target = out_dir / CHECKPOINT
    if target.exists():
        return True
    try:
        downloaded = hf_hub_download(
            repo, remote_path(arm, CHECKPOINT), repo_type="model", local_dir=out_dir
        )
    except EntryNotFoundError:
        return False
    Path(downloaded).replace(target)
    shutil.rmtree(out_dir / "runs", ignore_errors=True)
    shutil.rmtree(out_dir / ".cache", ignore_errors=True)
    return True
