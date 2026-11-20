"""Train one ordering arm, recording validation and benchmark curves along the way."""

import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path

import torch

from gpt2_scratch.data.loader import TokenLoader
from gpt2_scratch.evaluation.multiple_choice import Encode, accuracy
from gpt2_scratch.model.config import GPT2Config
from gpt2_scratch.model.gpt2 import GPT2
from gpt2_scratch.training.config import TrainConfig
from gpt2_scratch.training.evaluate import load_validation, validation_by_bucket
from gpt2_scratch.training.metrics import MetricsLog, Tracker
from gpt2_scratch.training.schedule import get_lr

CHECKPOINT = "checkpoint.pt"


def train(
    cfg: TrainConfig,
    arm: str,
    model_config: GPT2Config,
    data_root: Path,
    out_dir: Path,
    device: str,
    benchmarks: Mapping[str, Sequence[dict]] | None = None,
    encode: Encode | None = None,
    tracker: Tracker | None = None,
    on_checkpoint: Callable[[Path, int], None] | None = None,
) -> dict:
    """Train one ordering arm, evaluating it along the way.

    Every eval_every steps, and after the last update, the validation loss (overall and per score
    quartile) and each benchmark's accuracy are recorded. Progress is checkpointed, so an
    interrupted run resumes from its last checkpoint. Every publish_every steps, on_checkpoint is
    called with the output folder and the step.

    Writes metrics.jsonl, model.pt and config.json into out_dir.

    Returns:
        A summary with the number of steps and the final evaluation.
    """
    benchmarks = benchmarks or {}
    if benchmarks and encode is None:
        raise ValueError("encode is required to score benchmarks")
    torch.manual_seed(cfg.seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    identity = {"arm": arm, "train": asdict(cfg), "model": asdict(model_config)}
    loader = TokenLoader(
        Path(data_root) / "arms" / arm, "train", cfg.micro_batch_size, cfg.sequence_length
    )
    validation_tokens, validation_buckets = load_validation(Path(data_root) / "validation")
    model = GPT2(model_config).to(device)
    optimizer = model.configure_optimizers(cfg.weight_decay, cfg.max_lr, device)
    start = _resume(out_dir, model, optimizer, loader, device, identity)
    log = MetricsLog(out_dir / "metrics.jsonl", tracker, keep_before=start)
    step_model = torch.compile(model) if cfg.use_compile else model

    def evaluate(step: int) -> dict:
        if not _is_due(step, cfg):
            return {}
        fields = validation_by_bucket(
            model,
            validation_tokens,
            validation_buckets,
            device,
            cfg.micro_batch_size,
            cfg.sequence_length,
        )
        for name, examples in benchmarks.items():
            fields[f"{name}_acc"] = accuracy(model, examples, device, encode)
        log.write(step=step, **fields)
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        return fields

    for step in range(start, cfg.max_steps):
        evaluate(step)
        log.write(step=step, **_train_step(step_model, optimizer, loader, cfg, device, step))
        if (step + 1) % cfg.checkpoint_every == 0:
            _checkpoint(out_dir, model, optimizer, loader, step + 1, identity)
            if on_checkpoint is not None and (step + 1) % cfg.publish_every == 0:
                on_checkpoint(out_dir, step + 1)
    final = evaluate(cfg.max_steps)

    _save(model, out_dir, identity)
    (out_dir / CHECKPOINT).unlink(missing_ok=True)
    return {"steps": cfg.max_steps, **final}


def _is_due(step: int, cfg: TrainConfig) -> bool:
    return step % cfg.eval_every == 0 or step == cfg.max_steps


def _train_step(
    step_model, optimizer, loader: TokenLoader, cfg: TrainConfig, device: str, step: int
) -> dict:
    """Accumulate gradients over one optimizer step and return its metrics."""
    started = time.time()
    optimizer.zero_grad(set_to_none=True)
    loss_accum = 0.0
    for _ in range(cfg.grad_accum_steps):
        x, y = loader.next_batch()
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            _, loss = step_model(x.to(device), y.to(device))
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
