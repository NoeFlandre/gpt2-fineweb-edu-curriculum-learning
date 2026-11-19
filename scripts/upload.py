"""Upload runs and the model card to the model repo, then publish the Trackio dashboard.

Example: python -m scripts.upload
"""

import argparse
from pathlib import Path

from gpt2_scratch.publishing import MODEL_REPO
from gpt2_scratch.training.tracking import TRACKIO_BUCKET, TRACKIO_PROJECT, TRACKIO_SPACE, publish
from huggingface_hub import HfApi


def parse_args() -> argparse.Namespace:
    """Read the command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=MODEL_REPO)
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--card", type=Path, default=Path("hub/README.md"))
    parser.add_argument("--message", default="Upload experiment runs")
    parser.add_argument("--project", default=TRACKIO_PROJECT)
    parser.add_argument("--space", default=TRACKIO_SPACE)
    parser.add_argument("--bucket", default=TRACKIO_BUCKET)
    return parser.parse_args()


def main() -> None:
    """Upload the runs folder and the model card, then publish the Trackio dashboard."""
    args = parse_args()
    api = HfApi()
    api.upload_folder(
        folder_path=str(args.runs),
        path_in_repo="runs",
        repo_id=args.repo,
        repo_type="model",
        commit_message=args.message,
    )
    api.upload_file(
        path_or_fileobj=str(args.card),
        path_in_repo="README.md",
        repo_id=args.repo,
        repo_type="model",
        commit_message="Update the model card",
    )
    print(f"uploaded {args.runs} to https://huggingface.co/{args.repo}")
    print(publish(args.project, space_id=args.space, bucket_id=args.bucket))


if __name__ == "__main__":
    main()
