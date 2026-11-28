"""Train one ordering arm, tracking curves in Trackio and keeping its checkpoints on the Hub.

Example: python -m scripts.train --arm cooldown --seed 42 \
    --out runs/lr6e-4-cosine_params45.2M_tokens899.7M_init42_data42/cooldown

If the output folder has no checkpoint, the one published for this arm is downloaded first, so an
interrupted run resumes from the Hub. Pass --no-hub to keep everything local.
"""

import argparse
from dataclasses import asdict
from pathlib import Path

import tiktoken
import torch
from gpt2_scratch.evaluation.multiple_choice import load_examples
from gpt2_scratch.experiment.arms import ORDERINGS
from gpt2_scratch.experiment.naming import DATA_SEED, run_id
from gpt2_scratch.model.config import STUDY_CONFIG
from gpt2_scratch.publishing import MODEL_REPO, fetch_checkpoint, publish_checkpoint
from gpt2_scratch.training.config import SCHEDULES, TrainConfig
from gpt2_scratch.training.runner import train
from gpt2_scratch.training.tracking import (
    TRACKIO_BUCKET,
    TRACKIO_PROJECT,
    TRACKIO_SPACE,
    TrackioTracker,
)
from huggingface_hub import HfApi

from scripts.g5k.dashboard import SiteDashboard

BENCHMARKS = ("hellaswag", "piqa", "arc_easy")
SMOKE = "For a smoke test only; the study uses the default"


def parse_args() -> argparse.Namespace:
    """Read the command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=ORDERINGS)
    parser.add_argument("--data", type=Path, help="data folder (default: data/data<data seed>)")
    parser.add_argument("--benchmarks", type=Path, default=Path("data/benchmarks"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--repo", default=MODEL_REPO, help="Model repo for checkpoints")
    parser.add_argument("--no-hub", action="store_true", help="keep checkpoints local only")
    parser.add_argument(
        "--dashboard",
        choices=("trackio", "site", "none"),
        default="trackio",
        help="trackio: a live Trackio run; site: every run of this seed on this site, rebuilt as "
        "they train (Grid'5000); none: no dashboard",
    )
    parser.add_argument("--project", default=TRACKIO_PROJECT, help="Trackio project name")
    parser.add_argument("--space", default=TRACKIO_SPACE, help="Static dashboard Space")
    parser.add_argument("--bucket", default=TRACKIO_BUCKET, help="Bucket behind the dashboard")
    defaults = TrainConfig()
    parser.add_argument(
        "--seed",
        type=int,
        default=defaults.seed,
        help="model initialization seed",
    )
    parser.add_argument(
        "--data-seed",
        type=int,
        default=DATA_SEED,
        help="seed of the data: the pool, the orders and the validation split",
    )
    parser.add_argument(
        "--schedule", choices=SCHEDULES, default=defaults.schedule, help="learning-rate schedule"
    )
    parser.add_argument("--total-tokens", type=int, default=defaults.total_tokens, help=SMOKE)
    parser.add_argument("--eval-every", type=int, default=defaults.eval_every, help=SMOKE)
    parser.add_argument(
        "--checkpoint-every", type=int, default=defaults.checkpoint_every, help=SMOKE
    )
    return parser.parse_args()


def load_benchmarks(directory: Path) -> dict[str, list[dict]]:
    """Every benchmark of the study, by name."""
    return {name: load_examples(Path(directory) / f"{name}.jsonl") for name in BENCHMARKS}


def _hub_checkpointer(repo: str, arm: str):
    """Publish each checkpoint to the Hub; a failed upload is reported and retried later."""
    api = HfApi()

    def checkpoint(out_dir: Path, step: int) -> None:
        try:
            publish_checkpoint(api, repo, arm, out_dir, step)
        except Exception as error:  # the run goes on; the next checkpoint tries again
            print(f"checkpoint upload deferred ({type(error).__name__}) at step {step}", flush=True)

    return checkpoint


def _prepare_hub(repo: str, arm: str, out: Path) -> None:
    """Create the model repo if needed, then resume from its checkpoint when the folder has none."""
    HfApi().create_repo(repo, repo_type="model", exist_ok=True)
    if fetch_checkpoint(repo, arm, out):
        print(f"resuming {arm} from the checkpoint in {out}", flush=True)


def main() -> None:
    """Train the arm, tracking it in Trackio, and print its final evaluation."""
    args = parse_args()
    encoder = tiktoken.get_encoding("gpt2")
    cfg = TrainConfig(
        total_tokens=args.total_tokens,
        eval_every=args.eval_every,
        checkpoint_every=args.checkpoint_every,
        seed=args.seed,
        schedule=args.schedule,
    )
    model_config = STUDY_CONFIG
    data = args.data or Path(f"data/data{args.data_seed}")
    run = f"{run_id(args.seed, args.data_seed, args.schedule)}/{args.arm}"
    if not args.no_hub:
        _prepare_hub(args.repo, run, args.out)
    tracker = None
    if args.dashboard == "trackio":
        tracker = TrackioTracker.start(
            project=args.project,
            name=f"{args.arm}-{run_id(args.seed, args.data_seed, args.schedule)}",
            config={"arm": args.arm, "train": asdict(cfg), "model": asdict(model_config)},
            space_id=args.space,
            bucket_id=args.bucket,
            group=args.arm,
        )
    elif args.dashboard == "site":
        tracker = SiteDashboard(args.seed, args.data_seed)
    try:
        summary = train(
            cfg,
            args.arm,
            model_config,
            data,
            args.out,
            args.device,
            benchmarks=load_benchmarks(args.benchmarks),
            encode=encoder.encode_ordinary,
            tracker=tracker,
            on_checkpoint=None if args.no_hub else _hub_checkpointer(args.repo, run),
        )
    finally:
        if tracker is not None:
            tracker.finish()
    print(summary)


if __name__ == "__main__":
    main()
