"""Durable-record tests that need no database.

The round-trip is where silent data loss hides: a field dropped in _row() or
mis-parsed in _item() would not raise, it would just quietly return a memory
that lost its supersession pointer or its embedding. Every field is asserted.

Live-database behaviour (schema creation, upsert conflict handling, rehydration
into a cold service) is covered by tests/test_pg_live.py, which skips unless
DATABASE_URL is set.
"""
from __future__ import annotations

import numpy as np
import pytest

from agent_memory.pg_store import PgStore, dsn_from_env
from agent_memory.types import MemoryItem, MemoryType


def _item(**over) -> MemoryItem:
    base = dict(
        id="M0007", content="Ransom Sr. married Phoebe Munden.",
        mtype=MemoryType.SEMANTIC, created_day=12, importance=0.8,
        subject="ransom-sr", attribute="spouse", value="Phoebe Munden",
        ttl_days=None, superseded_by="M0011", support=3,
        last_access_day=14, access_count=5,
        provenance=["Glynn Co. GA deed book", "1799 land warrant"],
        embedding=np.arange(384, dtype=np.float32) / 384.0, vec_id=41,
    )
    base.update(over)
    return MemoryItem(**base)


def test_round_trip_preserves_every_field():
    store = PgStore("postgresql://unused", namespace="ns1")
    original = _item()
    restored = PgStore._item(store._row(original))

    assert restored.id == original.id
    assert restored.content == original.content
    assert restored.mtype == original.mtype
    assert restored.created_day == original.created_day
    assert restored.importance == pytest.approx(original.importance)
    assert restored.subject == original.subject
    assert restored.attribute == original.attribute
    assert restored.value == original.value
    assert restored.ttl_days == original.ttl_days
    assert restored.support == original.support
    assert restored.last_access_day == original.last_access_day
    assert restored.access_count == original.access_count
    assert restored.provenance == original.provenance
    assert restored.vec_id == original.vec_id


def test_supersession_pointer_survives():
    """The single most important field. Losing superseded_by would resurrect a
    corrected fact as if it were still true — the exact failure this project's
    stale-retrieval metric exists to measure."""
    store = PgStore("postgresql://unused")
    restored = PgStore._item(store._row(_item(superseded_by="M0011")))
    assert restored.superseded_by == "M0011"

    active = PgStore._item(store._row(_item(superseded_by=None)))
    assert active.superseded_by is None


def test_embedding_survives_bit_for_bit():
    """faiss stays the retrieval engine, so a rehydrated vector must be exactly
    what was indexed before. Any drift here silently changes recall."""
    store = PgStore("postgresql://unused")
    original = _item()
    restored = PgStore._item(store._row(original))
    assert restored.embedding is not None
    assert restored.embedding.shape == original.embedding.shape
    assert restored.embedding.dtype == np.float32
    np.testing.assert_array_equal(restored.embedding, original.embedding)


def test_missing_embedding_is_not_an_error():
    store = PgStore("postgresql://unused")
    restored = PgStore._item(store._row(_item(embedding=None, vec_id=None)))
    assert restored.embedding is None
    assert restored.vec_id is None


def test_vector_literal_is_pgvector_format():
    assert PgStore._vec_literal(np.array([1.0, -2.5, 0.0], dtype=np.float32)) == "[1.0,-2.5,0.0]"
    assert PgStore._vec_literal(None) is None


def test_empty_upsert_is_a_noop_and_never_connects():
    """Must not open a connection for an empty batch — otherwise a run with
    nothing to say would fail on an unreachable database."""
    store = PgStore("postgresql://definitely-not-a-real-host:5432/none")
    assert store.upsert([]) == 0


def test_no_dsn_means_no_persistence(monkeypatch):
    """Absence of a DSN is a valid configuration, not an error — the service
    must fall back to in-RAM rather than refuse to boot."""
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("POSTGRES_URL", raising=False)
    assert dsn_from_env() is None
    monkeypatch.setenv("POSTGRES_URL", "postgresql://x")
    assert dsn_from_env() == "postgresql://x"


def test_schema_declares_the_configured_dimension():
    from agent_memory.pg_store import SCHEMA
    rendered = SCHEMA % {"dim": 768}
    assert "vector(768)" in rendered
    assert "CREATE EXTENSION IF NOT EXISTS vector" in rendered
    # Superseded rows are retained on purpose, so the supersession-key index
    # must not be unique. (Assert on the DDL, not the prose — the comment above
    # the index legitimately contains the word.)
    assert "CREATE UNIQUE INDEX" not in rendered.upper()
