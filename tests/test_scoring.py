from types import SimpleNamespace

import numpy as np
import torch
from gpt2_scratch.experiment.scoring import CLASSIFIER_ID, EducationalScorer, to_int_score


def test_to_int_score_rounds_to_the_nearest_integer():
    raw = np.array([2.4, 2.6, 4.49])

    assert to_int_score(raw).tolist() == [2, 3, 4]


def test_to_int_score_clips_to_the_zero_to_five_range():
    raw = np.array([-1.3, 5.7, 9.0])

    assert to_int_score(raw).tolist() == [0, 5, 5]


def test_to_int_score_returns_uint8():
    assert to_int_score(np.array([1.0])).dtype == np.uint8


class FakeTokenizer:
    def __call__(self, texts, padding, truncation, max_length, return_tensors):
        assert truncation and max_length == 512 and return_tensors == "pt"
        ids = torch.tensor([[len(text)] for text in texts])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


class FakeModel:
    def __call__(self, input_ids, attention_mask):
        return SimpleNamespace(logits=input_ids.float() / 10)


def test_scorer_maps_classifier_logits_to_integer_scores():
    scorer = EducationalScorer(FakeTokenizer(), FakeModel())

    assert scorer.score(["a" * 10, "a" * 33, "a" * 90]).tolist() == [1, 3, 5]


class FakeClassifier:
    def __init__(self):
        self.device = None
        self.evaluated = False

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        self.evaluated = True
        return self


def test_from_pretrained_loads_the_classifier_in_eval_mode(monkeypatch):
    import transformers

    classifier = FakeClassifier()
    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: FakeTokenizer()
    )
    monkeypatch.setattr(
        transformers.AutoModelForSequenceClassification,
        "from_pretrained",
        lambda *a, **k: classifier,
    )

    scorer = EducationalScorer.from_pretrained(device="cpu")

    assert scorer.model is classifier
    assert classifier.device == "cpu"
    assert classifier.evaluated


def test_from_pretrained_defaults_to_cuda_and_the_published_classifier(monkeypatch):
    import transformers

    requested = []
    classifier = FakeClassifier()

    tokenizer = FakeTokenizer()

    def record_tokenizer(name, *args, **kwargs):
        requested.append(name)
        return tokenizer

    def record_model(name, *args, **kwargs):
        requested.append(name)
        return classifier

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", record_tokenizer)
    monkeypatch.setattr(
        transformers.AutoModelForSequenceClassification, "from_pretrained", record_model
    )

    scorer = EducationalScorer.from_pretrained()

    assert requested == [CLASSIFIER_ID, CLASSIFIER_ID]
    assert classifier.device == "cuda"
    assert scorer.tokenizer is tokenizer
    assert scorer.model is classifier
    assert scorer.device == "cuda"
