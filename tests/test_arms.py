import pytest
import torch
from gpt2_scratch.experiment.arms import ARMS, get_arm, soft_weight, uniform_weight


def test_uniform_weight_is_one_for_every_token():
    scores = torch.tensor([[0, 3, 5]])

    assert torch.equal(uniform_weight(scores), torch.ones(1, 3))


def test_soft_weight_is_score_divided_by_five():
    scores = torch.tensor([0, 3, 5], dtype=torch.uint8)

    assert torch.allclose(soft_weight(scores), torch.tensor([0.0, 0.6, 1.0]))


def test_registry_has_the_three_arms_with_their_sources():
    assert set(ARMS) == {"uniform", "hard", "soft"}
    assert ARMS["uniform"].source == "raw"
    assert ARMS["hard"].source == "edu"
    assert ARMS["soft"].source == "raw"


def test_hard_arm_counts_every_token_equally():
    scores = torch.tensor([3, 4])

    assert torch.equal(ARMS["hard"].weight(scores), uniform_weight(scores))


def test_soft_arm_uses_soft_weights():
    scores = torch.tensor([1, 5])

    assert torch.allclose(ARMS["soft"].weight(scores), soft_weight(scores))


def test_get_arm_returns_the_registered_arm():
    assert get_arm("soft") is ARMS["soft"]


def test_get_arm_rejects_unknown_names():
    with pytest.raises(ValueError, match="unknown arm"):
        get_arm("cosine")
