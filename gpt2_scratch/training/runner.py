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
Publish = Callable[[Path, int], None]


class CheckpointMismatchError(ValueError):
    """A checkpoint in the output folder was written by a different run."""

    def __init__(self, path: Path) -> None:
        super().__init__(f"{path} belongs to another run; remove it to start this one over")


def train(
    cfg: TrainConfig,
    arm: str,
    model_config: GPT2Config,
    data_root: Path,
    out_dir: Path,
    device: str,
    encode: Encode,
    benchmarks: Mapping[str, Sequence[dict]] | None = None,
    tracker: Tracker | None = None,
    on_checkpoint: Publish | None = None,
) -> dict:
    """Train one ordering arm, evaluating it along the way.

    Every eval_every steps, and after the last update, the validation loss (overall and per score
    bucket) and each benchmark's accuracy are recorded. Progress is checkpointed, so an interrupted
    run resumes from its last checkpoint. Every publish_every steps, on_checkpoint is called with
    the output folder and the step.

    Args:
        cfg: Training settings.
        arm: Name of the ordering, which selects the training shards.
        model_config: Model size settings.
        data_root: Folder holding the arms and the validation split.
        out_dir: Folder for metrics, checkpoints and the final weights.
        device: Torch device, such as "cuda" or "cpu".
        encode: Turns text into token ids, for scoring the benchmarks.
        benchmarks: Examples of each multiple-choice benchmark, by name.
        tracker: Receives every metric record, if given.
        on_checkpoint: Called with the output folder and the step at each publish step.

    Returns:
        A summary with the number of steps and the final evaluation.
    """
    benchmarks = benchmarks or {}
    torch.manual_seed(cfg.seed)
    run = _Run(
        cfg,
        arm,
        model_config,
        Path(data_root),
        Path(out_dir),
        device,
        benchmarks,
        encode,
        tracker,
        on_checkpoint,
    )
    for step in range(run.start, cfg.max_steps):
        run.evaluate(step)
        run.train_step(step)
        if (step + 1) % cfg.checkpoint_every == 0:
            run.checkpoint(step + 1)
    final = run.evaluate(cfg.max_steps)
    run.finish()
    return {"steps": cfg.max_steps, **final}


def _is_due(step: int, cfg: TrainConfig) -> bool:
    return step % cfg.eval_every == 0 or step == cfg.max_steps


class _Run:
    """The state of one training run: model, data, logs and checkpoints."""

    def __init__(
        self,
        cfg: TrainConfig,
        arm: str,
        model_config: GPT2Config,
        data_root: Path,
        out_dir: Path,
        device: str,
        benchmarks: Mapping[str, Sequence[dict]],
        encode: Encode,
        tracker: Tracker | None,
        on_checkpoint: Publish | None,
    ) -> None:
        self.cfg = cfg
        self.out_dir = out_dir
        self.device = device
        self.benchmarks = benchmarks
        self.encode = encode
        self.on_checkpoint = on_checkpoint
        self.identity = {"arm": arm, "train": asdict(cfg), "model": asdict(model_config)}
        out_dir.mkdir(parents=True, exist_ok=True)
        self.loader = TokenLoader(
            data_root / "arms" / arm, "train", cfg.micro_batch_size, cfg.sequence_length
        )
        self.validation = load_validation(data_root / "validation")
        self.model = GPT2(model_config).to(device)
        self.optimizer = self.model.configure_optimizers(cfg.weight_decay, cfg.max_lr, device)
        self.start = self._resume()
        self.log = MetricsLog(out_dir / "metrics.jsonl", tracker, keep_before=self.start)
        self.step_model = torch.compile(self.model) if cfg.use_compile else self.model

    def evaluate(self, step: int) -> dict[str, float]:
        """Record the validation and benchmark metrics if this step is due for them."""
        if not _is_due(step, self.cfg):
            return {}
        fields = validation_by_bucket(
            self.model,
            *self.validation,
            self.device,
            self.cfg.micro_batch_size,
            self.cfg.sequence_length,
        )
        for name, examples in self.benchmarks.items():
            fields[f"{name}_acc"] = accuracy(self.model, examples, self.device, self.encode)
        self.log.write(step=step, **fields)
        torch.cuda.empty_cache()
        return fields

    def train_step(self, step: int) -> None:
        """Run one optimizer step and record its metrics."""
        self.log.write(step=step, **self._update(step))

    def checkpoint(self, next_step: int) -> None:
        """Save what a run needs to continue from next_step, then publish it when due."""
        path = self.out_dir / CHECKPOINT
        temporary = path.with_suffix(".tmp")
        torch.save(
            {
                "identity": self.identity,
                "step": next_step,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "loader": self.loader.state(),
            },
            temporary,
        )
        temporary.replace(path)
        if self.on_checkpoint is not None and next_step % self.cfg.publish_every == 0:
            self.on_checkpoint(self.out_dir, next_step)

    def finish(self) -> None:
        """Save the final weights and configuration, and remove the checkpoint."""
        torch.save(self.model.state_dict(), self.out_dir / "model.pt")
        (self.out_dir / "config.json").write_text(json.dumps(self.identity, indent=2))
        (self.out_dir / CHECKPOINT).unlink(missing_ok=True)

    def _update(self, step: int) -> dict[str, float]:
        """Accumulate gradients over one optimizer step and return its metrics."""
        started = time.time()
        self.optimizer.zero_grad(set_to_none=True)
        loss_total = 0.0
        for _ in range(self.cfg.grad_accum_steps):
            x, y = self.loader.next_batch()
            with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                _, loss = self.step_model(x.to(self.device), y.to(self.device))
            loss = loss / self.cfg.grad_accum_steps
            loss.backward()
            loss_total += loss.item()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
        lr = get_lr(step, self.cfg)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.optimizer.step()
        elapsed = time.time() - started
        return {
            "train_loss": loss_total,
            "lr": lr,
            "grad_norm": float(grad_norm),
            "tokens_per_s": self.cfg.total_batch_size / elapsed,
        }

    def _resume(self) -> int:
        """Restore the model, optimizer and data position from the last checkpoint, or return 0."""
        path = self.out_dir / CHECKPOINT
        if not path.exists():
            return 0
        saved = torch.load(path, map_location=self.device)
        if saved["identity"] != self.identity:
            raise CheckpointMismatchError(path)
        self.model.load_state_dict(saved["model"])
        self.optimizer.load_state_dict(saved["optimizer"])
        self.loader.restore(saved["loader"])
        return saved["step"]
