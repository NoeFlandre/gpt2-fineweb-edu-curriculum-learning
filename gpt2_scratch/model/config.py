"""GPT-2 model hyperparameters."""

from dataclasses import dataclass


@dataclass(frozen=True)
class GPT2Config:
    """Model size settings. The defaults are GPT-2 small."""

    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

    @property
    def n_params(self) -> int:
        """Trainable parameters, with the output head tied to the token embedding."""
        d = self.n_embd
        embeddings = self.vocab_size * d + self.block_size * d
        per_block = 12 * d * d + 13 * d  # attention and MLP weights and biases, two LayerNorms
        return embeddings + self.n_layer * per_block + 2 * d  # plus the final LayerNorm


# The ordering study's model: about 45M parameters, 19M of them outside the embeddings.
STUDY_CONFIG = GPT2Config(block_size=1024, vocab_size=50304, n_layer=6, n_head=8, n_embd=512)
