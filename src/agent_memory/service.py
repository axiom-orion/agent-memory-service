"""MemoryService -- the facade tying the stores, consolidation, retention,
scoring, and audit log into a single remember / consolidate / recall surface."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .audit import AuditLog
from .config import MemoryPolicy, settings
from .consolidation import consolidate
from .embeddings import Embedder
from .retention import apply_retention
from .scoring import score
from .stores import EpisodicMemory, ProceduralMemory, SemanticMemory, WorkingMemory
from .types import MemoryItem, MemoryType
from .vector_index import VectorIndex


class MemoryService:
    def __init__(self, policy: MemoryPolicy | None = None,
                 embedder: Embedder | None = None):
        self.policy = policy or MemoryPolicy()
        self.embedder = embedder or Embedder()
        self.working = WorkingMemory(settings.working_capacity)
        self.episodic = EpisodicMemory()
        self.semantic = SemanticMemory()
        self.procedural = ProceduralMemory()
        self.audit = AuditLog()
        self._consolidated = False
        self._dim: int | None = None

    # --- writes ------------------------------------------------------------- #
    def remember(self, content: str, day: int, *, subject: str | None = None,
                 attribute: str | None = None, value: str | None = None,
                 importance: float = 0.5, ttl_days: int | None = None,
                 provenance: list[str] | None = None) -> MemoryItem:
        item = MemoryItem(
            id=f"M{len(self.episodic.items()) + 1:04d}", content=content,
            mtype=MemoryType.EPISODIC, created_day=day, importance=importance,
            subject=subject, attribute=attribute, value=value, ttl_days=ttl_days,
            provenance=provenance or [])
        item.embedding = self.embedder.encode([content])[0]
        self._dim = item.embedding.shape[0]
        self.episodic.add(item)
        # mirror into working memory; evictions are flushed back to episodic
        self.working.add(item)
        self.audit.record(day, "write", item.id,
                          f"{attribute}={value}" if attribute else "episodic")
        return item

    def ingest_records(self, records: list[dict]) -> None:
        # batch-embed for speed, then register
        texts = [r["text"] for r in records]
        vecs = self.embedder.encode(texts)
        for r, v in zip(records, vecs, strict=True):
            item = MemoryItem(
                id=r["id"], content=r["text"], mtype=MemoryType.EPISODIC,
                created_day=r["ts_day"], importance=r.get("importance", 0.5),
                subject=r.get("subject"), attribute=r.get("attribute"),
                value=r.get("value"), provenance=[r.get("provenance", "")])
            item.embedding = v
            self._dim = v.shape[0]
            self.episodic.add(item)
            self.audit.record(r["ts_day"], "write", item.id,
                              f"{r.get('attribute')}={r.get('value')}"
                              if r.get("attribute") else "episodic")

    # --- lifecycle ---------------------------------------------------------- #
    def consolidate(self, now_day: int) -> None:
        if self.policy.use_consolidation:
            consolidate(self.episodic, self.semantic, self.embedder,
                        self.policy, self.audit, now_day)
        self._consolidated = True

    def forget(self, now_day: int) -> None:
        pool = self.semantic.all() if self.policy.use_consolidation else self.episodic.items()
        apply_retention(pool, now_day, self.policy, self.audit)

    # --- retrieval ---------------------------------------------------------- #
    def _candidates(self, now_day: int) -> list[MemoryItem]:
        if self.policy.use_consolidation:
            pool = self.semantic.active() + self.episodic.non_facts()
        else:
            pool = self.episodic.items()
        return [m for m in pool
                if not m.expired(now_day) and m.superseded_by is None
                and m.embedding is not None]

    def recall(self, query: str, k: int = 5, now_day: int = 0) -> list[MemoryItem]:
        cands = self._candidates(now_day)
        if not cands:
            return []
        qv = self.embedder.encode([query], use_cache=False)[0]
        # similarity shortlist via the index (scales), then blended re-score
        index = VectorIndex(self._dim or qv.shape[0]).build(
            [c.id for c in cands], np.stack([c.embedding for c in cands]))
        by_id = {c.id: c for c in cands}
        pool = [by_id[i] for i, _ in index.search(qv, max(k * 4, 20))]
        ranked = sorted(pool, key=lambda it: -score(qv, it, now_day, self.policy))[:k]
        for it in ranked:
            it.access_count += 1
            it.last_access_day = now_day
            self.audit.record(now_day, "recall", it.id, query[:48])
        return ranked


# --------------------------------------------------------------------------- #
def load_interactions(path: Path | None = None) -> list[dict]:
    path = path or settings.interactions_path
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python data/generate_sessions.py` (or `make gen-data`).")
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_queries(path: Path | None = None) -> list[dict]:
    path = path or settings.queries_path
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]
