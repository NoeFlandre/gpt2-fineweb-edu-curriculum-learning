"""Publish a finished arm to the model repository, then delete its local copies from this site.

Usage: python -m scripts.g5k.publish --arm ARM --seed SEED --out runs/<run id>/ARM
"""

import argparse
from pathlib import Path

from gpt2_scratch.experiment.naming import run_id
from gpt2_scratch.publishing import MODEL_REPO, publish_final
from huggingface_hub import HfApi

LOCAL_FILES = ("checkpoint.pt", "model.pt", "config.json")


def main() -> None:
    """Upload the arm's checkpoint, weights, configuration and metrics, then remove local copies."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repo", default=MODEL_REPO)
    args = parser.parse_args()
    run = f"{run_id(args.seed)}/{args.arm}"
    publish_final(HfApi(), args.repo, run, args.out)
    for name in LOCAL_FILES:
        (args.out / name).unlink()
    print(f"published {run} to {args.repo}")


if __name__ == "__main__":
    main()
