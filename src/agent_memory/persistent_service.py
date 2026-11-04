"""MemoryService with a durable record behind it.

Same surface, same retrieval math — see pg_store.py for why faiss stays the
retrieval engine and Postgres is only the record.

The lifecycle this fixes:

    before:  remember() -> RAM -> Cloud Run scales to zero -> gone
    after:   remember() -> RAM + Postgres
             cold start -> hydrate() -> RAM + faiss index rebuilt from stored vectors

A caller with a gap between runs (the cason-heritage Keeper runs WEEKLY, so its
instance is always cold) now actually recalls what it learned last time. That
was the entire point of wiring KEEPER_MEMORY_URL, and without this it would have
been a no-op that looked wired.
"""
from __future__ import annotations

from .extractor import Extractor
from .pg_store import PgStore
from .service import MemoryService
from .types import MemoryItem, MemoryType


class PersistentMemoryService(MemoryService):
    """Write-through to Postgres, rehydrate on construction."""

    def __init__(self, *args, store: PgStore, hydrate: bool = True, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.store = store
        self.hydrated = 0
        if hydrate:
            self.hydrated = self.hydrate()

    # ---- rehydration ------------------------------------------------------
    def hydrate(self) -> int:
        """Rebuild in-RAM state from the durable record. Returns items restored.

        Superseded items are loaded too. They are not retrievable (the active
        filter excludes them), but the record keeps its own history of what it
        used to say — the same reason the supersession ledger is kept on the
        cason-heritage side.
        """
        items = self.store.load(include_superseded=True)
        if not items:
            return 0

        max_vec = -1
        for it in items:
            if it.mtype == MemoryType.SEMANTIC:
                self.semantic.add(it)
            elif it.mtype == MemoryType.PROCEDURAL:
                self.procedural.add(it.subject or it.id, it)
            else:
                # WORKING is a live mirror of EPISODIC, never a separate record.
                # Restoring it as episodic is correct and keeps remember()'s id
                # sequence (M0001, M0002, ...) continuous across a cold start.
                self.episodic.add(it)
            if it.embedding is not None and self._dim is None:
                self._dim = int(it.embedding.shape[0])
            if it.vec_id is not None:
                self._vec_of[it.id] = it.vec_id
                self._item_of_vec[it.vec_id] = it
                max_vec = max(max_vec, int(it.vec_id))

        self._next_vec_id = max_vec + 1
        self._index_dirty = True     # index rebuilds from these vectors on next search
        return len(items)

    # ---- write-through ----------------------------------------------------
    def _all_items(self) -> list[MemoryItem]:
        seen: dict[str, MemoryItem] = {}
        for it in (self.episodic.items() + self.semantic.all()
                   + self.procedural.items() + self.working.items()):
            seen[it.id] = it        # working mirrors episodic objects; dedupe by id
        return list(seen.values())

    def flush(self) -> int:
        """Persist everything currently in RAM. Used after operations that mutate
        many items at once (consolidation supersedes; forget expires)."""
        return self.store.upsert(self._all_items())

    def remember(self, *args, **kwargs) -> MemoryItem:
        item = super().remember(*args, **kwargs)
        self.store.upsert([item])   # one row, not a full flush — writes stay O(1)
        return item

    def ingest_records(self, records: list[dict],
                       extractor: Extractor | None = None) -> list[str]:
        before = {it.id for it in self._all_items()}
        unavailable = super().ingest_records(records, extractor)
        new = [it for it in self._all_items() if it.id not in before]
        if new:
            self.store.upsert(new)
        return unavailable

    def consolidate(self, now_day: int) -> None:
        super().consolidate(now_day)
        self.flush()                # supersession rewrites superseded_by across items

    def forget(self, now_day: int) -> None:
        super().forget(now_day)
        self.flush()                # expiry changes lifecycle fields in place

    # ---- introspection ----------------------------------------------------
    def counts(self, now_day: int = 0) -> dict:
        c = super().counts(now_day)
        c["hydrated_on_boot"] = self.hydrated
        c["durable"] = True
        try:
            c.update(self.store.counts())
        except Exception as e:
            # Degrade, never crash: a database blip must not take down /stats.
            c["persisted_error"] = str(e)[:160]
        return c
