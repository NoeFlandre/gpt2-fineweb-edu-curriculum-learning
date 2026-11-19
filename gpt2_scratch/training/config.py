"""Training settings shared by every experimental arm."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainConfig:
    """Hyperparameters of one training run.

    The defaults follow the original run, except for the token budget and warmup.
    """

    total_tokens: int = 100_000_000
    total_batch_size: int = 524_288
    micro_batch_size: int = 16
    sequence_length: int = 1024
    max_lr: float = 6e-4
    min_lr_ratio: float = 0.1
    warmup_steps: int = 10
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    val_every: int = 10
    val_batches: int = 40
    seed: int = 42
    use_compile: bool = True

    def __post_init__(self) -> None:
        """Check that the batch splits into whole micro-batches."""
        if self.total_batch_size % (self.micro_batch_size * self.sequence_length) != 0:
            raise ValueError(
                "total_batch_size must be divisible by micro_batch_size * sequence_length"
            )

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
