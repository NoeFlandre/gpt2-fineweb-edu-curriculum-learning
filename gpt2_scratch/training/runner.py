"""Train one experimental arm, with validation loss and HellaSwag accuracy recorded as curves."""

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path

import torch

from gpt2_scratch.data.loader import TokenLoader
from gpt2_scratch.evaluation.hellaswag import Encode, accuracy
from gpt2_scratch.experiment.arms import Arm
from gpt2_scratch.model.config import GPT2Config
from gpt2_scratch.model.gpt2 import GPT2
from gpt2_scratch.training.config import TrainConfig
from gpt2_scratch.training.evaluate import validation_loss
from gpt2_scratch.training.metrics import MetricsLog, Tracker
from gpt2_scratch.training.schedule import get_lr

CHECKPOINT = "checkpoint.pt"


def train(
    cfg: TrainConfig,
    arm: Arm,
    model_config: GPT2Config,
    data_root: Path,
    out_dir: Path,
    device: str,
    hellaswag: Sequence[dict] = (),
    encode: Encode | None = None,
    tracker: Tracker | None = None,
    on_checkpoint: Callable[[Path, int], None] | None = None,
) -> dict:
    """Train a model on the arm's data, recording curves along the way.

    Validation loss and, when examples are given, HellaSwag accuracy are recorded every few steps
    and once more after the last update. Progress is checkpointed, so an interrupted run resumes
    from its last checkpoint and reads the same data. Every publish_every steps, on_checkpoint is
    called with the output folder and the step, so the checkpoint can be kept elsewhere.

    Writes metrics.jsonl, model.pt and config.json into out_dir.

    Returns:
        A summary with the number of steps, the final validation loss and, when HellaSwag
        examples are given, the final HellaSwag accuracy.
    """
    if hellaswag and encode is None:
        raise ValueError("encode is required to score HellaSwag")
    torch.manual_seed(cfg.seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    identity = {"arm": arm.name, "train": asdict(cfg), "model": asdict(model_config)}
    train_loader = TokenLoader(
        Path(data_root) / arm.source, "train", cfg.micro_batch_size, cfg.sequence_length
    )
    val_loader = TokenLoader(
        Path(data_root) / "raw", "val", cfg.micro_batch_size, cfg.sequence_length
    )
    model = GPT2(model_config).to(device)
    optimizer = model.configure_optimizers(cfg.weight_decay, cfg.max_lr, device)
    start = _resume(out_dir, model, optimizer, train_loader, device, identity)
    log = MetricsLog(out_dir / "metrics.jsonl", tracker, keep_before=start)
    step_model = torch.compile(model) if cfg.use_compile else model

    for step in range(start, cfg.max_steps):
        _evaluate(step, model, val_loader, device, cfg, hellaswag, encode, log)
        log.write(
            step=step, **_train_step(step_model, optimizer, train_loader, arm, cfg, device, step)
        )
        if (step + 1) % cfg.checkpoint_every == 0:
            _checkpoint(out_dir, model, optimizer, train_loader, step + 1, identity)
            if on_checkpoint is not None and (step + 1) % cfg.publish_every == 0:
                on_checkpoint(out_dir, step + 1)
    final = _evaluate(cfg.max_steps, model, val_loader, device, cfg, hellaswag, encode, log)

    summary: dict = {"steps": cfg.max_steps, "final_val_loss": final["val_loss"]}
    if hellaswag:
        summary["hellaswag_acc"] = final["hellaswag_acc"]
    _save(model, out_dir, identity)
    (out_dir / CHECKPOINT).unlink(missing_ok=True)
    return summary


def _is_due(step: int, every: int, cfg: TrainConfig) -> bool:
    return step % every == 0 or step == cfg.max_steps


def _evaluate(step, model, val_loader, device, cfg, hellaswag, encode, log) -> dict:
    """Record validation loss and HellaSwag accuracy at the steps where each is due."""
    fields: dict = {}
    if _is_due(step, cfg.val_every, cfg):
        fields["val_loss"] = validation_loss(model, val_loader, device, cfg.val_batches)
    if hellaswag and encode is not None and _is_due(step, cfg.hellaswag_every, cfg):
        fields["hellaswag_acc"] = accuracy(model, hellaswag, device, encode)
    if fields:
        log.write(step=step, **fields)
    return fields


def _train_step(
    step_model, optimizer, loader, arm: Arm, cfg: TrainConfig, device, step: int
) -> dict:
    """Accumulate gradients over one optimizer step and return its metrics."""
    started = time.time()
    optimizer.zero_grad(set_to_none=True)
    loss_accum = 0.0
    for _ in range(cfg.grad_accum_steps):
        x, y, scores = loader.next_batch()
        weights = arm.weight(scores.to(device))
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            _, loss = step_model(x.to(device), y.to(device), weights=weights)
        loss = loss / cfg.grad_accum_steps
        loss.backward()
        loss_accum += loss.item()
    grad_norm = torch.nn.utils.clip_grad_norm_(step_model.parameters(), cfg.grad_clip)
    lr = get_lr(step, cfg)
    for group in optimizer.param_groups:
        group["lr"] = lr
    optimizer.step()
    elapsed = time.time() - started
    return {
        "train_loss": loss_accum,
        "lr": lr,
        "grad_norm": float(grad_norm),
        "tokens_per_s": cfg.total_batch_size / elapsed,
    }


def _resume(
    out_dir: Path, model, optimizer, loader: TokenLoader, device: str, identity: dict
) -> int:
    """Restore the model, optimizer and data position from the last checkpoint, or return 0."""
    path = out_dir / CHECKPOINT
    if not path.exists():
        return 0
    saved = torch.load(path, map_location=device)
    if saved["identity"] != identity:
        raise ValueError(f"{path} belongs to another run; remove it to start this one over")
    model.load_state_dict(saved["model"])
    optimizer.load_state_dict(saved["optimizer"])
    loader.restore(saved["loader"])
    return saved["step"]


def _checkpoint(
    out_dir: Path, model, optimizer, loader: TokenLoader, next_step: int, identity: dict
) -> None:
    """Save what a run needs to continue from next_step, replacing the previous checkpoint."""
    path = out_dir / CHECKPOINT
    temporary = path.with_suffix(".tmp")
    torch.save(
        {
            "identity": identity,
            "step": next_step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "loader": loader.state(),
        },
        temporary,
    )
    temporary.replace(path)


def _save(model: GPT2, out_dir: Path, identity: dict) -> None:
    torch.save(model.state_dict(), out_dir / "model.pt")
    (out_dir / "config.json").write_text(json.dumps(identity, indent=2))
