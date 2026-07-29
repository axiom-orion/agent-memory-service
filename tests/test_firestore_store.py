"""Firestore durable-record tests. No network, no credentials.

Serialization is asserted field by field for the same reason as the Postgres
backend: a field dropped in _doc() or mis-parsed in _item() would not raise, it
would quietly return a memory that lost its supersession pointer or its vector.

Write/read behaviour is exercised against a fake client that mimics the small
slice of the Firestore API this store uses (collection/document/batch/stream),
so the contract is tested without a project, credentials, or an emulator.

Live behaviour against the real database is covered by verifying the DEPLOYED
service, where Cloud Run supplies credentials via the runtime service account.
"""
from __future__ import annotations

import numpy as np
import pytest

from agent_memory.firestore_store import FirestoreStore, firestore_project_from_env
from agent_memory.types import MemoryItem, MemoryType


# ---- a minimal fake of the Firestore surface this store touches -------------
class _Snap:
    def __init__(self, data): self._data = data
    def to_dict(self): return dict(self._data)


class _Doc:
    """A document. Mirrors the real API: a document can hold subcollections,
    which is what makes the agent_memory/{ns}/items/{id} path legal."""
    def __init__(self, backing, doc_id): self.backing, self.id = backing, doc_id
    def collection(self, name): return _Col(self.backing.setdefault(("sub", self.id, name), {}))


class _Batch:
    def __init__(self): self.ops = []
    def set(self, doc, payload): self.ops.append((doc, payload))
    def commit(self):
        for doc, payload in self.ops:
            doc.backing[doc.id] = payload
        self.ops = []


class _Col:
    def __init__(self, backing): self.backing = backing
    def document(self, doc_id): return _Doc(self.backing, doc_id)
    def limit(self, n): return _Limited(self.backing, n)
    def stream(self): return iter([_Snap(v) for k, v in self.backing.items()
                                   if not isinstance(k, tuple)])


class _Limited(_Col):
    def __init__(self, backing, n):
        super().__init__(backing)
        self.n = n

    def stream(self):
        return iter(list(super().stream())[:self.n])


class _Client:
    def __init__(self): self.root = {}
    def collection(self, _name): return _Col(self.root)
    def batch(self): return _Batch()


@pytest.fixture()
def store():
    return FirestoreStore(project="test", namespace="ns1", client=_Client())


def _item(id_="M0007", **over) -> MemoryItem:
    base = dict(
        id=id_, content="Ransom Sr. married Phoebe Munden.",
        mtype=MemoryType.SEMANTIC, created_day=12, importance=0.8,
        subject="ransom-sr", attribute="spouse", value="Phoebe Munden",
        ttl_days=None, superseded_by="M0011", support=3,
        last_access_day=14, access_count=5,
        provenance=["Glynn Co. GA deed book", "1799 land warrant"],
        embedding=np.arange(384, dtype=np.float32) / 384.0, vec_id=41,
    )
    base.update(over)
    return MemoryItem(**base)


def test_round_trip_preserves_every_field(store):
    original = _item()
    restored = FirestoreStore._item(store._doc(original))
    for f in ("id", "content", "mtype", "created_day", "subject", "attribute", "value",
              "ttl_days", "superseded_by", "support", "last_access_day", "access_count",
              "provenance", "vec_id"):
        assert getattr(restored, f) == getattr(original, f), f
    assert restored.importance == pytest.approx(original.importance)


def test_supersession_pointer_survives(store):
    """Losing superseded_by would resurrect a corrected fact as if still true —
    the exact failure this project's stale-retrieval metric measures."""
    assert FirestoreStore._item(store._doc(_item(superseded_by="M0011"))).superseded_by == "M0011"
    assert FirestoreStore._item(store._doc(_item(superseded_by=None))).superseded_by is None


def test_embedding_survives_bit_for_bit(store):
    """faiss stays the retrieval engine, so a rehydrated vector must be exactly
    what was indexed. Any drift here silently changes recall."""
    original = _item()
    restored = FirestoreStore._item(store._doc(original))
    assert restored.embedding.dtype == np.float32
    np.testing.assert_array_equal(restored.embedding, original.embedding)


def test_embedding_is_stored_as_a_typed_vector(store):
    """Typed now so find_nearest is available later without a migration."""
    from google.cloud.firestore_v1.vector import Vector
    assert isinstance(store._doc(_item())["embedding"], Vector)


def test_a_plain_list_embedding_still_loads(store):
    """Documents written before the typed form must not become unreadable."""
    doc = store._doc(_item())
    doc["embedding"] = [0.5, 0.25, 0.125]
    restored = FirestoreStore._item(doc)
    np.testing.assert_allclose(restored.embedding, [0.5, 0.25, 0.125])


def test_missing_embedding_is_not_an_error(store):
    restored = FirestoreStore._item(store._doc(_item(embedding=None, vec_id=None)))
    assert restored.embedding is None and restored.vec_id is None


def test_write_then_read_back(store):
    store.upsert([_item("M0001", created_day=2, superseded_by=None),
                  _item("M0002", created_day=1, superseded_by=None)])
    loaded = store.load()
    assert [i.id for i in loaded] == ["M0002", "M0001"], "must be ordered by created_day"


def test_upsert_updates_in_place(store):
    store.upsert([_item("M0001", value="Ann Munden", superseded_by=None)])
    store.upsert([_item("M0001", value="Phoebe Munden", superseded_by=None)])
    loaded = store.load()
    assert len(loaded) == 1, "re-writing the same id must update, not duplicate"
    assert loaded[0].value == "Phoebe Munden"


def test_superseded_rows_retained_but_filtered(store):
    store.upsert([_item("M0001", superseded_by="M0002"), _item("M0002", superseded_by=None)])
    assert len(store.load(include_superseded=True)) == 2
    assert [i.id for i in store.load(include_superseded=False)] == ["M0002"]
    assert store.counts() == {"persisted_total": 2, "persisted_active": 1}


def test_empty_upsert_is_a_noop_and_never_connects():
    """No client is supplied, so touching Firestore would raise. It must not."""
    assert FirestoreStore(project="nope", namespace="ns").upsert([]) == 0


def test_batching_respects_the_500_write_cap(store):
    """Firestore rejects a batch over 500 writes; 600 must still all land."""
    items = [_item(f"M{i:04d}", superseded_by=None) for i in range(600)]
    assert store.upsert(items) == 600
    assert len(store.load()) == 600


def test_backend_selection_is_opt_in(monkeypatch):
    for var in ("MEMORY_BACKEND", "FIRESTORE_PROJECT", "GOOGLE_CLOUD_PROJECT", "GCP_PROJECT"):
        monkeypatch.delenv(var, raising=False)
    assert firestore_project_from_env() is None, "must not activate by accident"

    monkeypatch.setenv("MEMORY_BACKEND", "firestore")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "vorionsys")
    assert firestore_project_from_env() == "vorionsys"

    monkeypatch.delenv("MEMORY_BACKEND")
    monkeypatch.setenv("FIRESTORE_PROJECT", "explicit")
    assert firestore_project_from_env() == "explicit"


def test_ensure_schema_is_a_noop(store):
    assert store.ensure_schema() is None
