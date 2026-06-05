"""MemoryService -- the facade tying the stores, consolidation, retention,
scoring, and audit log into a single remember / consolidate / recall surface.

Retrieval is backed by a persistent :class:`VectorIndex` (FAISS, exact-by-default).
The index keys on int64 ids; the stores key on string ids, so the service owns the
mapping (``_register``) and exposes the active set as ``get_active_vectors()`` --
``(float32[n, d], int64[n])`` -- which is exactly what ``VectorIndex.rebuild`` and
the ``/admin/rebuild`` endpoint consume.

The index is rebuilt from the *active* set rather than maintained incrementally:
memory mutates (consolidation, supersession, forgetting), and at this scale a full
rebuild is sub-millisecond. Writes mark the index dirty; the next recall (or an
explicit ``rebuild_index`` / ``/admin/rebuild``) reconstructs it. There is no
background thread -- on Cloud Run the cadence comes from Cloud Scheduler.
"""
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
from .vector_index import VectorIndex, VectorIndexConfig


class MemoryService:
    def __init__(self, policy: MemoryPolicy | None = None,
                 embedder: Embedder | None = None,
                 index_type: str = "flat"):
        self.policy = policy or MemoryPolicy()
        self.embedder = embedder or Embedder()
        self.working = WorkingMemory(settings.working_capacity)
        self.episodic = EpisodicMemory()
        self.semantic = SemanticMemory()
        self.procedural = ProceduralMemory()
        self.audit = AuditLog()
        self._consolidated = False
        self._dim: int | None = None
        # --- vector index + str<->int id mapping ------------------------------ #
        self._index_type = index_type            # "flat" (exact, default) | "hnsw"
        self.index: VectorIndex | None = None
        self._vec_of: dict[str, int] = {}        # string id -> stable int64 id
        self._item_of_vec: dict[int, MemoryItem] = {}  # int64 id -> live item
        self._next_vec_id: int = 0
        self._index_dirty: bool = True           # rebuild before the next search
        self._index_now_day: int | None = None   # now_day the index was built for

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
        self._index_dirty = True
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
        self._index_dirty = True

    # --- lifecycle ---------------------------------------------------------- #
    def consolidate(self, now_day: int) -> None:
        if self.policy.use_consolidation:
            consolidate(self.episodic, self.semantic, self.embedder,
                        self.policy, self.audit, now_day)
        self._consolidated = True
        self._index_dirty = True

    def forget(self, now_day: int) -> None:
        pool = self.semantic.all() if self.policy.use_consolidation else self.episodic.items()
        apply_retention(pool, now_day, self.policy, self.audit)
        self._index_dirty = True

    # --- vector index ------------------------------------------------------- #
    def _register(self, item: MemoryItem) -> int:
        """Assign (once) and return the stable int64 vector id for an item."""
        vid = self._vec_of.get(item.id)
        if vid is None:
            self._next_vec_id += 1
            vid = self._next_vec_id
            self._vec_of[item.id] = vid
            item.vec_id = vid
        self._item_of_vec[vid] = item
        return vid

    def get_active_vectors(self, now_day: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """The active retrieval set as ``(float32[n, d], int64[n])``.

        This is the store's contract with the index: it returns exactly the items
        that should be searchable *now* -- expired and superseded items excluded --
        so ``VectorIndex.rebuild(*store.get_active_vectors())`` drops them. Empty
        active set yields zero-row arrays so the index resets cleanly.
        """
        cands = self._candidates(now_day)
        dim = self._dim or VectorIndexConfig().dimension
        if not cands:
            return (np.zeros((0, dim), dtype=np.float32),
                    np.zeros((0,), dtype=np.int64))
        vecs = np.ascontiguousarray(
            np.stack([c.embedding for c in cands]), dtype=np.float32)
        ids = np.fromiter((self._register(c) for c in cands),
                          dtype=np.int64, count=len(cands))
        return vecs, np.ascontiguousarray(ids)

    def rebuild_index(self, now_day: int = 0) -> int:
        """Rebuild the FAISS index from the active set. Returns the active count.

        Called on the first recall after a write, by ``/admin/rebuild``, and by the
        Cloud Scheduler cadence. Recreates the index if the embedding dimension or
        index type changed; otherwise swaps its contents in place.
        """
        vecs, ids = self.get_active_vectors(now_day)
        dim = int(vecs.shape[1]) if vecs.shape[0] else (
            self._dim or VectorIndexConfig().dimension)
        if (self.index is None
                or self.index.config.dimension != dim
                or self.index.config.index_type != self._index_type):
            self.index = VectorIndex(
                VectorIndexConfig(dimension=dim, index_type=self._index_type))
        self.index.rebuild(vecs, ids)
        self._index_dirty = False
        self._index_now_day = now_day
        return int(ids.shape[0])

    def counts(self, now_day: int = 0) -> dict:
        """Active vs superseded item counts for the /stats endpoint (C3)."""
        superseded = sum(1 for it in self.semantic.all()
                         if it.superseded_by is not None)
        return {"active": len(self._candidates(now_day)), "superseded": superseded}

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
        # Keep the index consistent with the active set: rebuild on the first recall
        # after a write, or whenever the as-of day changes (expiry is day-relative).
        if self._index_dirty or self.index is None or self._index_now_day != now_day:
            self.rebuild_index(now_day)
        if self.index is None or self.index.stats()["ntotal"] == 0:
            return []
        qv = self.embedder.encode([query], use_cache=False)[0]
        # similarity shortlist via the index (scales), then blended re-score
        _, ids = self.index.search(np.asarray(qv, dtype=np.float32), max(k * 4, 20))
        pool: list[MemoryItem] = []
        for vid in ids[0].tolist():
            if vid == -1:
                continue
            it = self._item_of_vec.get(int(vid))
            # defensive: never surface a stale/expired item even if the index is
            # momentarily behind the store (between a supersession and a rebuild).
            if it is None or it.superseded_by is not None or it.expired(now_day):
                continue
            pool.append(it)
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
