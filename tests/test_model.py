import pytest
import torch
import torch.nn.functional as F
from gpt2_scratch.model.config import STUDY_CONFIG, GPT2Config
from gpt2_scratch.model.gpt2 import GPT2

TINY = GPT2Config(block_size=16, vocab_size=64, n_layer=1, n_head=2, n_embd=16)


def test_forward_returns_logits_of_shape_batch_time_vocab():
    torch.manual_seed(0)
    model = GPT2(TINY)
    idx = torch.randint(0, TINY.vocab_size, (2, 8))

    logits, loss = model(idx)

    assert logits.shape == (2, 8, TINY.vocab_size)
    assert loss is None


def test_forward_loss_is_the_mean_cross_entropy():
    torch.manual_seed(0)
    model = GPT2(TINY)
    idx = torch.randint(0, TINY.vocab_size, (2, 8))
    targets = torch.randint(0, TINY.vocab_size, (2, 8))

    logits, loss = model(idx, targets)

    expected = F.cross_entropy(logits.view(-1, TINY.vocab_size), targets.view(-1))
    assert loss.item() == pytest.approx(expected.item(), rel=1e-6)


def test_optimizer_decays_matrices_but_not_vectors():
    model = GPT2(TINY)

    optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device="cpu")

    decay, no_decay = optimizer.param_groups
    assert decay["weight_decay"] == 0.1
    assert no_decay["weight_decay"] == 0.0
    assert all(p.dim() >= 2 for p in decay["params"])
    assert all(p.dim() < 2 for p in no_decay["params"])


def test_input_and_output_embeddings_are_tied():
    model = GPT2(TINY)

    assert model.lm_head.weight is model.wte.weight


def test_study_model_has_about_45_million_parameters():
    model = GPT2(STUDY_CONFIG)

    total = sum(p.numel() for p in model.parameters())

    assert 45.0e6 < total < 45.3e6


def test_study_model_has_about_19_million_non_embedding_parameters():
    model = GPT2(STUDY_CONFIG)
    embedding = model.wte.weight.numel() + model.wpe.weight.numel()

    non_embedding = sum(p.numel() for p in model.parameters()) - embedding

    assert 18.5e6 < non_embedding < 19.3e6


def test_study_model_is_six_layers_of_width_512_with_block_1024():
    assert (STUDY_CONFIG.n_layer, STUDY_CONFIG.n_embd, STUDY_CONFIG.n_head) == (6, 512, 8)
    assert STUDY_CONFIG.block_size == 1024
