"""Educational scores for raw documents, from the FineWeb-Edu classifier."""

from collections.abc import Sequence

import numpy as np
import torch

from gpt2_scratch.experiment.arms import MAX_SCORE

CLASSIFIER_ID = "HuggingFaceFW/fineweb-edu-classifier"
MAX_LENGTH = 512


def to_int_score(raw: np.ndarray) -> np.ndarray:
    """Round classifier outputs to integer scores in the range 0 to MAX_SCORE."""
    return np.rint(np.clip(raw, 0, MAX_SCORE)).astype(np.uint8)


class EducationalScorer:
    """Scores documents with the FineWeb-Edu classifier."""

    def __init__(self, tokenizer, model, device: str = "cpu") -> None:
        """Wrap an already loaded tokenizer and classifier."""
        self.tokenizer = tokenizer
        self.model = model
        self.device = device

    @classmethod
    def from_pretrained(cls, device: str = "cuda") -> "EducationalScorer":
        """Load the published FineWeb-Edu classifier onto device."""
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(CLASSIFIER_ID)
        model = AutoModelForSequenceClassification.from_pretrained(CLASSIFIER_ID)
        return cls(tokenizer, model.to(device).eval(), device)

    @torch.no_grad()
    def score(self, texts: Sequence[str]) -> np.ndarray:
        """Return the integer educational score of each text."""
        inputs = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}
        with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
            logits = self.model(**inputs).logits
        logits = logits.squeeze(-1).float().cpu().numpy()
        return to_int_score(logits)
