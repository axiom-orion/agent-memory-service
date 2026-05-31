"""LoCoMo metric tests. `locomo_f1` is pinned to values verified byte-identical to
the upstream snap-research/locomo `task_eval/evaluation.py::f1_score`."""
from __future__ import annotations

import pytest

from eval.locomo.metrics import locomo_f1, recall_at_k, reciprocal_rank

# (prediction, ground_truth, expected_f1) — verified against LoCoMo's f1_score
F1_CASES = [
    ("7 May 2023", "7 May, 2023", 1.0),
    ("Paris, France", "paris france", 1.0),
    ("the cat and the dog", "cat dog", 1.0),
    ("Bob Tran", "Bob Tran", 1.0),
    ("She went running", "running was the activity", 0.333333),
    ("no information available", "Caroline visited the museum", 0.0),
]


@pytest.mark.parametrize("pred,gold,expected", F1_CASES)
def test_locomo_f1_matches_official(pred, gold, expected):
    assert locomo_f1(pred, gold) == pytest.approx(expected, abs=1e-5)


def test_recall_at_k():
    ranked = ["D1:1", "D2:3", "D5:2", "D1:9"]
    assert recall_at_k(ranked, {"D2:3", "D1:9"}, 5) == 1.0
    assert recall_at_k(ranked, {"D2:3", "D9:9"}, 2) == 0.5
    assert recall_at_k(ranked, set(), 5) is None  # no gold -> undefined


def test_reciprocal_rank():
    ranked = ["D1:1", "D2:3", "D5:2"]
    assert reciprocal_rank(ranked, {"D2:3"}) == pytest.approx(0.5)
    assert reciprocal_rank(ranked, {"D9:9"}) == 0.0
