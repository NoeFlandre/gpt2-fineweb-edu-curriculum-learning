"""Train one experimental arm and write its metrics and final weights."""

import json
import time
from collections.abc import Sequence
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
) -> dict:
    """Train a model on the arm's data, then score HellaSwag if examples are given.

    Writes metrics.jsonl, model.pt and config.json into out_dir.

    Returns:
        A summary with the number of steps, the last validation loss and, when
        HellaSwag examples are given, their accuracy.
    """
    torch.manual_seed(cfg.seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = MetricsLog(out_dir / "metrics.jsonl", tracker)
    train_loader = TokenLoader(
        Path(data_root) / arm.source, "train", cfg.micro_batch_size, cfg.sequence_length
    )
    val_loader = TokenLoader(
        Path(data_root) / "raw", "val", cfg.micro_batch_size, cfg.sequence_length
    )
    model = GPT2(model_config).to(device)
    optimizer = model.configure_optimizers(cfg.weight_decay, cfg.max_lr, device)
    step_model = torch.compile(model) if cfg.use_compile else model

    last_val = float("nan")
    for step in range(cfg.max_steps):
        if _is_eval_step(step, cfg):
            last_val = validation_loss(model, val_loader, device, cfg.val_batches)
            log.write(step=step, val_loss=last_val)
        log.write(
            step=step, **_train_step(step_model, optimizer, train_loader, arm, cfg, device, step)
        )

    summary: dict = {"steps": cfg.max_steps, "final_val_loss": last_val}
    if hellaswag:
        summary["hellaswag_acc"] = _score_hellaswag(model, hellaswag, device, encode, log, cfg)
    _save(model, out_dir, arm, cfg, model_config)
    return summary


def _is_eval_step(step: int, cfg: TrainConfig) -> bool:
    return step % cfg.val_every == 0 or step == cfg.max_steps - 1


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


def _score_hellaswag(model, examples, device, encode: Encode | None, log: MetricsLog, cfg) -> float:
    if encode is None:
        raise ValueError("encode is required to score HellaSwag")
    score = accuracy(model, examples, device, encode)
    log.write(step=cfg.max_steps - 1, hellaswag_acc=score)
    return score


def _save(model: GPT2, out_dir: Path, arm: Arm, cfg: TrainConfig, model_config: GPT2Config) -> None:
    torch.save(model.state_dict(), out_dir / "model.pt")
    settings = {"arm": arm.name, "train": asdict(cfg), "model": asdict(model_config)}
    (out_dir / "config.json").write_text(json.dumps(settings, indent=2))
