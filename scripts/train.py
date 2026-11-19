"""Train one arm, then score HellaSwag.

Example: python -m scripts.train --arm soft --out runs/soft
"""

import argparse
from dataclasses import asdict
from pathlib import Path

import tiktoken
import torch
from gpt2_scratch.evaluation.hellaswag import load_examples
from gpt2_scratch.experiment.arms import ARMS, get_arm
from gpt2_scratch.model.config import GPT2Config
from gpt2_scratch.training.config import TrainConfig
from gpt2_scratch.training.runner import train
from gpt2_scratch.training.tracking import (
    TRACKIO_BUCKET,
    TRACKIO_PROJECT,
    TRACKIO_SPACE,
    TrackioTracker,
)

DEFAULT_HELLASWAG = Path("data/hellaswag/hellaswag_val.jsonl")


def parse_args() -> argparse.Namespace:
    """Read the command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=sorted(ARMS))
    parser.add_argument("--data", type=Path, default=Path("data/arms"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--hellaswag", type=Path, default=DEFAULT_HELLASWAG)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--project", default=TRACKIO_PROJECT, help="Trackio project name")
    parser.add_argument("--space", default=TRACKIO_SPACE, help="Static dashboard Space")
    parser.add_argument("--bucket", default=TRACKIO_BUCKET, help="Bucket behind the dashboard")
    return parser.parse_args()


def main() -> None:
    """Train the arm, tracking it in Trackio, and print its summary."""
    args = parse_args()
    encoder = tiktoken.get_encoding("gpt2")
    arm = get_arm(args.arm)
    cfg = TrainConfig()
    model_config = GPT2Config(vocab_size=50304)
    tracker = TrackioTracker.start(
        project=args.project,
        name=arm.name,
        config={"arm": arm.name, "train": asdict(cfg), "model": asdict(model_config)},
        space_id=args.space,
        bucket_id=args.bucket,
    )
    try:
        summary = train(
            cfg,
            arm,
            model_config,
            args.data,
            args.out,
            args.device,
            hellaswag=load_examples(args.hellaswag),
            encode=encoder.encode_ordinary,
            tracker=tracker,
        )
    finally:
        tracker.finish()
    print(summary)


if __name__ == "__main__":
    main()
