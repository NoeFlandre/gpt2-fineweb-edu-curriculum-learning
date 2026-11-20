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


# The ordering study's model: about 45M parameters, 19M of them outside the embeddings.
STUDY_CONFIG = GPT2Config(block_size=1024, vocab_size=50304, n_layer=6, n_head=8, n_embd=512)
