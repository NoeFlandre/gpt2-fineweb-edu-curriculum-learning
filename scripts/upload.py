"""Upload runs to the model repo and publish the Trackio dashboard.

Example: python -m scripts.upload
"""

import argparse
from pathlib import Path

from gpt2_scratch.training.tracking import TRACKIO_BUCKET, TRACKIO_PROJECT, TRACKIO_SPACE, publish
from huggingface_hub import HfApi

REPO = "NoeFlandre/GPT2-Fineweb-Edu-Experiments"


def parse_args() -> argparse.Namespace:
    """Read the command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=REPO)
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--message", default="Upload experiment runs")
    parser.add_argument("--project", default=TRACKIO_PROJECT)
    parser.add_argument("--space", default=TRACKIO_SPACE)
    parser.add_argument("--bucket", default=TRACKIO_BUCKET)
    return parser.parse_args()


def main() -> None:
    """Upload the runs folder, then publish the Trackio dashboard."""
    args = parse_args()
    HfApi().upload_folder(
        folder_path=str(args.runs),
        path_in_repo="runs",
        repo_id=args.repo,
        repo_type="model",
        commit_message=args.message,
    )
    print(f"uploaded {args.runs} to https://huggingface.co/{args.repo}")
    print(publish(args.project, space_id=args.space, bucket_id=args.bucket))


if __name__ == "__main__":
    main()
