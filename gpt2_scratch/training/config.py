"""Training settings shared by every ordering arm."""

from dataclasses import dataclass

BATCH_SPLIT_MESSAGE = "total_batch_size must be divisible by micro_batch_size * sequence_length"


@dataclass(frozen=True)
class TrainConfig:
    """Hyperparameters of one training run.

    The defaults are the ordering study: 900M tokens per arm, 524,288 tokens per optimizer step
    (1716 steps), with warmup over about 1% of the steps.
    """

    total_tokens: int = 900_000_000
    total_batch_size: int = 524_288
    micro_batch_size: int = 8
    sequence_length: int = 1024
    max_lr: float = 6e-4
    min_lr_ratio: float = 0.1
    warmup_steps: int = 18
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_every: int = 50
    checkpoint_every: int = 50
    publish_every: int = 100
    seed: int = 42
    use_compile: bool = True

    def __post_init__(self) -> None:
        """Check that the batch splits into whole micro-batches."""
        if self.total_batch_size % (self.micro_batch_size * self.sequence_length) != 0:
            raise ValueError(BATCH_SPLIT_MESSAGE)

    @property
    def max_steps(self) -> int:
        """Optimizer steps needed to see total_tokens."""
        return self.total_tokens // self.total_batch_size

    @property
    def grad_accum_steps(self) -> int:
        """Micro-batches accumulated into one optimizer step."""
        return self.total_batch_size // (self.micro_batch_size * self.sequence_length)

    @property
    def min_lr(self) -> float:
        """Learning rate at the end of the cosine decay."""
        return self.max_lr * self.min_lr_ratio
