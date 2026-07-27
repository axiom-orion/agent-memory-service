"""Durable record for the memory service: Postgres + pgvector.

WHY THIS SHAPE — read before "improving" it into a pgvector ANN search.

The obvious move is to replace `VectorIndex` (faiss) with a pgvector `<=>` query
and be done. That would be wrong here, for a reason that has nothing to do with
performance:

This project's published results — current-fact accuracy, stale-retrieval rate,
token savings, and the LoCoMo-10 evaluation — were all measured against faiss
EXACT inner-product search. Swapping the retrieval engine for approximate ANN
would mean the deployed service no longer behaves like the system those numbers
describe, and every one of those numbers would silently become a claim about
software that is no longer running.

So the split is deliberate:

    Postgres  = the durable RECORD    (survives scale-to-zero, redeploys, restarts)
    faiss     = the retrieval ENGINE  (unchanged, so the measured behaviour holds)

On cold start we rehydrate the items from Postgres and rebuild the faiss index
from their stored embeddings. Retrieval math is bit-for-bit what it was. The
embedding column is a real pgvector `vector`, so an ANN path stays available
later — but taking it would require re-running the evaluation first.

The service was previously in-process RAM on a single Cloud Run instance:
scale-to-zero wiped it, which made durable cross-run memory impossible for any
caller with a gap between runs (the cason-heritage Keeper runs WEEKLY).
"""
from __future__ import annotations

import json
import os
from typing import Iterable

import numpy as np

from .types import MemoryItem, MemoryType

SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS memory_items (
    id              text PRIMARY KEY,
    namespace       text        NOT NULL DEFAULT 'default',
    content         text        NOT NULL,
    mtype           text        NOT NULL,
    created_day     integer     NOT NULL,
    importance      real        NOT NULL DEFAULT 0.5,
    subject         text,
    attribute       text,
    value           text,
    ttl_days        integer,
    superseded_by   text,
    support         integer     NOT NULL DEFAULT 1,
    last_access_day integer,
    access_count    integer     NOT NULL DEFAULT 0,
    provenance      jsonb       NOT NULL DEFAULT '[]'::jsonb,
    vec_id          bigint,
    embedding       vector(%(dim)s),
    updated_at      timestamptz NOT NULL DEFAULT now()
);

-- The supersession key. Not unique: superseded rows are retained on purpose, so
-- the record keeps its own history of what it used to say.
CREATE INDEX IF NOT EXISTS memory_items_key_idx
    ON memory_items (namespace, subject, attribute)
    WHERE attribute IS NOT NULL;

-- Rehydration reads the ACTIVE set first; make that the cheap path.
CREATE INDEX IF NOT EXISTS memory_items_active_idx
    ON memory_items (namespace)
    WHERE superseded_by IS NULL;
"""

_COLUMNS = (
    "id, namespace, content, mtype, created_day, importance, subject, attribute, value, "
    "ttl_days, superseded_by, support, last_access_day, access_count, provenance, vec_id, embedding"
)


def dsn_from_env() -> str | None:
    """The connection string, or None when persistence is not configured.

    Absence is not an error: with no DSN the service runs exactly as it always
    did (in-RAM, ephemeral). It must degrade, not refuse to boot.
    """
    return os.environ.get("DATABASE_URL") or os.environ.get("POSTGRES_URL") or None


class PgStore:
    """Write-through durable record. Not a retrieval engine — see module docstring."""

    def __init__(self, dsn: str, *, namespace: str = "default", dimension: int = 384) -> None:
        self.dsn = dsn
        self.namespace = namespace
        self.dimension = dimension
        self._pool = None

    # ---- connection -------------------------------------------------------
    def _connect(self):
        import psycopg  # imported lazily: the dependency is optional
        return psycopg.connect(self.dsn, autocommit=True)

    def ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(SCHEMA % {"dim": self.dimension})

    def healthy(self) -> bool:
        try:
            with self._connect() as conn:
                conn.execute("SELECT 1")
            return True
        except Exception:
            return False

    # ---- serialization ----------------------------------------------------
    @staticmethod
    def _vec_literal(v) -> str | None:
        """pgvector's text input format: '[1,2,3]'."""
        if v is None:
            return None
        arr = np.asarray(v, dtype=np.float32).ravel()
        return "[" + ",".join(repr(float(x)) for x in arr.tolist()) + "]"

    def _row(self, item: MemoryItem) -> tuple:
        return (
            item.id, self.namespace, item.content,
            item.mtype.value if isinstance(item.mtype, MemoryType) else str(item.mtype),
            int(item.created_day), float(item.importance),
            item.subject, item.attribute, item.value,
            item.ttl_days, item.superseded_by, int(item.support),
            item.last_access_day, int(item.access_count),
            json.dumps(list(item.provenance or [])),
            item.vec_id, self._vec_literal(item.embedding),
        )

    @staticmethod
    def _item(row) -> MemoryItem:
        (id_, _ns, content, mtype, created_day, importance, subject, attribute, value,
         ttl_days, superseded_by, support, last_access_day, access_count,
         provenance, vec_id, embedding) = row
        emb = None
        if embedding is not None:
            raw = embedding if isinstance(embedding, str) else str(embedding)
            emb = np.asarray(json.loads(raw), dtype=np.float32)
        # provenance arrives as a parsed list when psycopg's jsonb adapter is
        # registered, and as raw JSON text when it is not. `list()` on the text
        # form would silently shred a citation into single characters, so decode
        # explicitly rather than trusting the driver's configuration.
        prov = provenance or []
        if isinstance(prov, (str, bytes)):
            try:
                prov = json.loads(prov)
            except (ValueError, TypeError):
                prov = []
        return MemoryItem(
            id=id_, content=content, mtype=MemoryType(mtype), created_day=int(created_day),
            importance=float(importance), subject=subject, attribute=attribute, value=value,
            ttl_days=ttl_days, superseded_by=superseded_by, support=int(support),
            last_access_day=last_access_day, access_count=int(access_count),
            provenance=list(prov), embedding=emb, vec_id=vec_id,
        )

    # ---- write ------------------------------------------------------------
    def upsert(self, items: Iterable[MemoryItem]) -> int:
        rows = [self._row(it) for it in items]
        if not rows:
            return 0
        placeholders = "(" + ",".join(["%s"] * 17) + ")"
        sql = (
            f"INSERT INTO memory_items ({_COLUMNS}) VALUES {placeholders} "
            "ON CONFLICT (id) DO UPDATE SET "
            "content = EXCLUDED.content, mtype = EXCLUDED.mtype, "
            "created_day = EXCLUDED.created_day, importance = EXCLUDED.importance, "
            "subject = EXCLUDED.subject, attribute = EXCLUDED.attribute, value = EXCLUDED.value, "
            "ttl_days = EXCLUDED.ttl_days, superseded_by = EXCLUDED.superseded_by, "
            "support = EXCLUDED.support, last_access_day = EXCLUDED.last_access_day, "
            "access_count = EXCLUDED.access_count, provenance = EXCLUDED.provenance, "
            "vec_id = EXCLUDED.vec_id, embedding = EXCLUDED.embedding, updated_at = now()"
        )
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.executemany(sql, rows)
        return len(rows)

    # ---- read -------------------------------------------------------------
    def load(self, *, include_superseded: bool = True) -> list[MemoryItem]:
        where = "WHERE namespace = %s" + ("" if include_superseded else " AND superseded_by IS NULL")
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(f"SELECT {_COLUMNS} FROM memory_items {where} ORDER BY created_day, id",
                            (self.namespace,))
                return [self._item(r) for r in cur.fetchall()]

    def counts(self) -> dict:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT count(*), count(*) FILTER (WHERE superseded_by IS NULL) "
                    "FROM memory_items WHERE namespace = %s", (self.namespace,))
                total, active = cur.fetchone()
        return {"persisted_total": int(total), "persisted_active": int(active)}
