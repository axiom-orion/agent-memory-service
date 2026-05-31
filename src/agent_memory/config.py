"""Configuration: retrieval/lifecycle policy and global settings."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class MemoryPolicy:
    """Toggles the behaviours the eval ablates, plus scoring weights."""
    use_recency: bool = True          # blend recency + importance into the score
    use_consolidation: bool = True    # distil episodic -> semantic (dedupe)
    use_supersession: bool = True     # keep only the latest value per fact key

    # scoring weights (only the recency/importance terms are gated by use_recency)
    w_similarity: float = 1.0
    w_recency: float = 0.25
    w_importance: float = 0.1
    recency_half_life_days: float = 21.0

    @property
    def label(self) -> str:
        if not (self.use_recency or self.use_consolidation or self.use_supersession):
            return "flat-vector"
        parts = ["recency"] if self.use_recency else []
        if self.use_consolidation:
            parts.append("consolidation")
        if self.use_supersession:
            parts.append("supersession")
        return "+".join(parts)


@dataclass(frozen=True)
class Settings:
    interactions_path: Path = ROOT / "data" / "sessions" / "interactions.jsonl"
    queries_path: Path = ROOT / "eval" / "queries.jsonl"
    cache_dir: Path = ROOT / ".cache"
    embed_model: str = os.environ.get("EMBED_MODEL",
                                      "sentence-transformers/all-MiniLM-L6-v2")
    working_capacity: int = 8         # items held in working memory before eviction


settings = Settings()
