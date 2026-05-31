"""Retrieval scoring: similarity blended with recency decay and importance."""
from __future__ import annotations

import math

import numpy as np

from .config import MemoryPolicy
from .types import MemoryItem


def recency_weight(item: MemoryItem, now_day: int, half_life: float) -> float:
    """Exponential decay on age; 1.0 when fresh, 0.5 at one half-life."""
    age = max(0, now_day - item.created_day)
    return math.pow(0.5, age / half_life)


def score(query_vec: np.ndarray, item: MemoryItem, now_day: int,
          policy: MemoryPolicy) -> float:
    sim = float(np.dot(query_vec, item.embedding)) if item.embedding is not None else 0.0
    s = policy.w_similarity * sim
    if policy.use_recency:
        s += policy.w_recency * recency_weight(item, now_day, policy.recency_half_life_days)
        s += policy.w_importance * item.importance
    return s
