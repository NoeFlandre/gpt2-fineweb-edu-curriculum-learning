"""Publish a finished arm to the model repository, then delete its weights from this site.

Usage: python -m scripts.g5k.publish --arm ARM --out runs/ARM
"""

import argparse
from pathlib import Path

from gpt2_scratch.publishing import MODEL_REPO, publish_final
from huggingface_hub import HfApi

WEIGHTS = ("checkpoint.pt", "model.pt", "config.json")


def main() -> None:
    """Upload the arm's checkpoint, weights, configuration and metrics, then remove local copies."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repo", default=MODEL_REPO)
    args = parser.parse_args()
    publish_final(HfApi(), args.repo, args.arm, args.out)
    for name in WEIGHTS:
        (args.out / name).unlink()
    print(f"published {args.arm} to {args.repo}")


if __name__ == "__main__":
    main()
