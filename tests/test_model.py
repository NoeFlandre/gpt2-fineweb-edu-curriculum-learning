import pytest
import torch
import torch.nn.functional as F
from gpt2_scratch.model.config import GPT2Config
from gpt2_scratch.model.gpt2 import GPT2
from gpt2_scratch.model.loss import weighted_mean

TINY = GPT2Config(block_size=16, vocab_size=64, n_layer=1, n_head=2, n_embd=16)


def test_weighted_mean_with_equal_weights_is_the_plain_mean():
    losses = torch.tensor([1.0, 2.0, 3.0])

    assert weighted_mean(losses, torch.ones(3)).item() == pytest.approx(2.0)


def test_zero_weight_tokens_are_ignored():
    losses = torch.tensor([1.0, 100.0, 3.0])

    assert weighted_mean(losses, torch.tensor([1.0, 0.0, 1.0])).item() == pytest.approx(2.0)


def test_weights_scale_each_contribution():
    losses = torch.tensor([1.0, 3.0])

    assert weighted_mean(losses, torch.tensor([1.0, 3.0])).item() == pytest.approx(2.5)


def test_all_zero_weights_give_zero_not_nan():
    loss = weighted_mean(torch.tensor([1.0, 2.0]), torch.zeros(2)).item()

    assert loss == 0.0


def test_forward_returns_logits_of_shape_batch_time_vocab():
    torch.manual_seed(0)
    model = GPT2(TINY)
    idx = torch.randint(0, TINY.vocab_size, (2, 8))

    logits, loss = model(idx)

    assert logits.shape == (2, 8, TINY.vocab_size)
    assert loss is None


def test_forward_without_weights_matches_the_mean_cross_entropy():
    torch.manual_seed(0)
    model = GPT2(TINY)
    idx = torch.randint(0, TINY.vocab_size, (2, 8))
    targets = torch.randint(0, TINY.vocab_size, (2, 8))

    logits, loss = model(idx, targets)

    expected = F.cross_entropy(logits.view(-1, TINY.vocab_size), targets.view(-1))
    assert loss.item() == pytest.approx(expected.item(), rel=1e-6)


def test_forward_with_unit_weights_matches_the_unweighted_loss():
    torch.manual_seed(0)
    model = GPT2(TINY)
    idx = torch.randint(0, TINY.vocab_size, (2, 8))
    targets = torch.randint(0, TINY.vocab_size, (2, 8))

    _, plain = model(idx, targets)
    _, weighted = model(idx, targets, weights=torch.ones(2, 8))

    assert weighted.item() == pytest.approx(plain.item(), rel=1e-6)


def test_forward_weights_change_the_loss():
    torch.manual_seed(0)
    model = GPT2(TINY)
    idx = torch.randint(0, TINY.vocab_size, (2, 8))
    targets = torch.randint(0, TINY.vocab_size, (2, 8))
    weights = torch.ones(2, 8)
    weights[0] = 0.0

    _, plain = model(idx, targets)
    _, weighted = model(idx, targets, weights=weights)

    assert weighted.item() != pytest.approx(plain.item())


def test_optimizer_decays_matrices_but_not_vectors():
    model = GPT2(TINY)

    optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device="cpu")

    decay, no_decay = optimizer.param_groups
    assert decay["weight_decay"] == 0.1
    assert no_decay["weight_decay"] == 0.0
    assert all(p.dim() >= 2 for p in decay["params"])
    assert all(p.dim() < 2 for p in no_decay["params"])
