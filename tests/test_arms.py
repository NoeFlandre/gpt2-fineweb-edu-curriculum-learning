import numpy as np
import pytest
from gpt2_scratch.experiment.arms import ORDERINGS, arm_order

SCORES = np.array([3.0, 4.5, 2.5, 3.5, 4.0])
LENGTHS = np.full(5, 100)


def test_the_orderings_are_the_four_arms():
    assert ORDERINGS == ("shuffle", "ascending", "descending", "cooldown")


def test_shuffle_visits_every_document_once():
    order = arm_order("shuffle", SCORES, LENGTHS, budget=500, seed=1)

    assert sorted(order.tolist()) == [0, 1, 2, 3, 4]


def test_shuffle_depends_on_the_seed():
    first = arm_order("shuffle", SCORES, LENGTHS, budget=500, seed=1)
    second = arm_order("shuffle", SCORES, LENGTHS, budget=500, seed=2)

    assert first.tolist() != second.tolist()


def test_ascending_puts_the_lowest_score_first():
    order = arm_order("ascending", SCORES, LENGTHS, budget=500, seed=1)

    assert SCORES[order].tolist() == [2.5, 3.0, 3.5, 4.0, 4.5]


def test_descending_puts_the_highest_score_first():
    order = arm_order("descending", SCORES, LENGTHS, budget=500, seed=1)

    assert SCORES[order].tolist() == [4.5, 4.0, 3.5, 3.0, 2.5]


def test_cooldown_puts_the_lowest_80_percent_of_tokens_first():
    scores = np.arange(10, dtype=float)
    lengths = np.full(10, 100)

    order = arm_order("cooldown", scores, lengths, budget=1000, seed=3)

    assert sorted(order[:8].tolist()) == list(range(8))
    assert sorted(order[8:].tolist()) == [8, 9]


def test_cooldown_splits_by_tokens_not_by_documents():
    scores = np.arange(4, dtype=float)
    lengths = np.array([700, 100, 100, 100])

    order = arm_order("cooldown", scores, lengths, budget=1000, seed=0)

    assert sorted(order[:2].tolist()) == [0, 1]
    assert sorted(order[2:].tolist()) == [2, 3]


def test_cooldown_shuffles_the_final_stage_instead_of_sorting_it():
    scores = np.arange(10, dtype=float)
    lengths = np.full(10, 100)

    final_stages = {
        tuple(arm_order("cooldown", scores, lengths, budget=1000, seed=seed)[8:].tolist())
        for seed in range(20)
    }

    assert final_stages == {(8, 9), (9, 8)}


def test_cooldown_shuffles_the_first_stage_too():
    scores = np.arange(10, dtype=float)
    lengths = np.full(10, 100)

    first_stages = {
        tuple(arm_order("cooldown", scores, lengths, budget=1000, seed=seed)[:8].tolist())
        for seed in range(20)
    }

    assert len(first_stages) > 1


def test_unknown_orderings_are_rejected():
    with pytest.raises(ValueError, match="unknown ordering"):
        arm_order("random-walk", SCORES, LENGTHS, budget=500, seed=1)
