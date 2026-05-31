"""Metrics. `locomo_f1` is a faithful reimplementation of LoCoMo's official
token-F1 (snap-research/locomo, task_eval/evaluation.py): SQuAD-style
normalisation + Porter stemming + token-overlap F1. Verified byte-identical to
the upstream implementation across the cases pinned in tests/test_locomo_metrics.py.
"""
from __future__ import annotations

import re
import string
from collections import Counter

_ARTICLES = re.compile(r"\b(a|an|the|and)\b")
_stemmer = None


def _stem(word: str) -> str:
    global _stemmer
    if _stemmer is None:
        try:
            from nltk.stem import PorterStemmer
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "locomo_f1 needs nltk's PorterStemmer for parity with the official "
                "LoCoMo metric. Install with: pip install nltk") from e
        _stemmer = PorterStemmer()
    return _stemmer.stem(word)


def normalize_answer(s: str) -> str:
    s = s.replace(",", "").lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def locomo_f1(prediction: str, ground_truth: str) -> float:
    pred = [_stem(w) for w in normalize_answer(prediction).split()]
    gold = [_stem(w) for w in normalize_answer(ground_truth).split()]
    common = Counter(pred) & Counter(gold)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred)
    recall = num_same / len(gold)
    return (2 * precision * recall) / (precision + recall)


# --- retrieval metrics (deterministic, no LLM) ----------------------------- #
def recall_at_k(ranked_ids: list[str], gold: set[str], k: int) -> float | None:
    if not gold:
        return None
    return len(set(ranked_ids[:k]) & gold) / len(gold)


def reciprocal_rank(ranked_ids: list[str], gold: set[str], cutoff: int = 10) -> float:
    for rank, did in enumerate(ranked_ids[:cutoff], 1):
        if did in gold:
            return 1.0 / rank
    return 0.0
