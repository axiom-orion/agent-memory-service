"""Live-database tests. Skipped unless DATABASE_URL points at a real Postgres.

    DATABASE_URL='postgresql://...' python -m pytest tests/test_pg_live.py -q

These assert the one behaviour that the offline tests cannot: that a memory
written by one process is recalled by a DIFFERENT, COLD process. That is the
whole reason this layer exists — a Cloud Run instance is always cold when a
weekly caller comes back, and the previous in-RAM service returned nothing.

Every test namespaces itself and cleans up, so it is safe against a shared
database.
"""
from __future__ import annotations

import uuid

import numpy as np
import pytest

from agent_memory.pg_store import PgStore, dsn_from_env
from agent_memory.types import MemoryItem, MemoryType

DSN = dsn_from_env()
pytestmark = pytest.mark.skipif(not DSN, reason="DATABASE_URL not set — live DB tests skipped")


@pytest.fixture()
def store():
    s = PgStore(DSN, namespace=f"test-{uuid.uuid4().hex[:12]}")
    s.ensure_schema()
    yield s
    with s._connect() as conn:
        conn.execute("DELETE FROM memory_items WHERE namespace = %s", (s.namespace,))


def _item(id_: str, **over) -> MemoryItem:
    base = dict(
        id=id_, content=f"content for {id_}", mtype=MemoryType.SEMANTIC,
        created_day=1, importance=0.7, subject="ransom-sr", attribute="spouse",
        value="Phoebe Munden", provenance=["Glynn Co. GA deed book"],
        embedding=np.linspace(0, 1, 384, dtype=np.float32), vec_id=7,
    )
    base.update(over)
    return MemoryItem(**base)


def test_schema_is_idempotent(store):
    store.ensure_schema()
    store.ensure_schema()
    assert store.healthy()


def test_write_then_read_in_a_fresh_store(store):
    """The core promise: a second PgStore object — standing in for a cold
    instance — sees what the first one wrote."""
    store.upsert([_item("M0001")])

    cold = PgStore(DSN, namespace=store.namespace)
    loaded = cold.load()

    assert len(loaded) == 1
    assert loaded[0].id == "M0001"
    assert loaded[0].value == "Phoebe Munden"
    assert loaded[0].provenance == ["Glynn Co. GA deed book"]
    expected = np.linspace(0, 1, 384, dtype=np.float32)
    np.testing.assert_allclose(loaded[0].embedding, expected, atol=1e-6)


def test_upsert_is_idempotent_and_updates_in_place(store):
    store.upsert([_item("M0001", value="Ann Munden")])
    store.upsert([_item("M0001", value="Phoebe Munden")])
    loaded = store.load()
    assert len(loaded) == 1, "re-writing the same id must update, not duplicate"
    assert loaded[0].value == "Phoebe Munden"


def test_superseded_rows_are_retained_but_filtered(store):
    """Corrections keep their history — the record remembers what it used to
    say — while the active set excludes them."""
    store.upsert([_item("M0001", superseded_by="M0002"), _item("M0002", superseded_by=None)])
    assert len(store.load(include_superseded=True)) == 2
    active = store.load(include_superseded=False)
    assert [i.id for i in active] == ["M0002"]
    assert store.counts() == {"persisted_total": 2, "persisted_active": 1}


def test_namespaces_are_isolated(store):
    other = PgStore(DSN, namespace=f"test-{uuid.uuid4().hex[:12]}")
    other.ensure_schema()
    try:
        store.upsert([_item("M0001")])
        assert other.load() == []
    finally:
        with other._connect() as conn:
            conn.execute("DELETE FROM memory_items WHERE namespace = %s", (other.namespace,))


def test_a_cold_service_recalls_what_a_previous_one_learned(store):
    """End to end, and the actual bug this fixes.

    Service A remembers something and is discarded — exactly what Cloud Run does
    on scale-to-zero. Service B boots cold against the same record and recalls
    it. Before this layer, B returned nothing and the caller re-asked a question
    it had already answered.
    """
    from agent_memory.config import MemoryPolicy
    from agent_memory.embeddings import HashingEmbedder
    from agent_memory.persistent_service import PersistentMemoryService

    kw = dict(policy=MemoryPolicy(), embedder=HashingEmbedder(), index_type="flat")

    a = PersistentMemoryService(store=store, **kw)
    a.remember("Bo Williams names Ann Munden as William's wife.", day=1,
               subject="william", attribute="spouse", value="Ann Munden")
    del a  # the instance goes away, as it does on scale-to-zero

    b = PersistentMemoryService(store=PgStore(DSN, namespace=store.namespace), **kw)
    assert b.hydrated >= 1, "a cold service must rehydrate the durable record"

    hits = b.recall("who was William's wife", k=3, now_day=2)
    assert hits, "a cold service recalled nothing — the durable record is not wired"
    assert any("Ann Munden" in h.content or h.value == "Ann Munden" for h in hits)

    counts = b.counts()
    assert counts["durable"] is True
    assert counts["hydrated_on_boot"] >= 1
