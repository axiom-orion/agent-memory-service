"""Vector index + index/store reconciliation tests.

Model-free: pure FAISS/numpy plus the deterministic HashingEmbedder, so the index
properties the service depends on are checked without downloading a transformer.

Covers the index contract the hardening introduced:
  * exact (flat) is the default                       -> test_default_index_type_is_flat
  * self-retrieval on a fixed synthetic set           -> test_self_retrieval_flat / _recall_target
  * id removal on the flat index                       -> test_remove_on_flat
  * HNSW carries external ids (IndexIDMap2) w/o raising -> test_hnsw_add_with_ids_*
  * rebuild from the active set drops excluded ids      -> test_rebuild_excludes_dropped_id
  * supersession -> rebuild excludes superseded ids     -> test_supersession_rebuild_*
  * empty active set resets the index cleanly           -> test_empty_*
"""
from __future__ import annotations

import numpy as np
import pytest

from agent_memory import MemoryPolicy, MemoryService
from agent_memory.embeddings import HashingEmbedder
from agent_memory.service import load_interactions
from agent_memory.vector_index import VectorIndex, VectorIndexConfig

DIM = 384


def _unit_vectors(n: int, dim: int = DIM, seed: int = 23) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, dim)).astype("float32")
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v


# --------------------------- index-level ----------------------------------- #
def test_default_index_type_is_flat():
    ix = VectorIndex()
    assert ix.config.index_type == "flat"          # exact-by-default
    assert ix.config.dimension == 384
    assert ix.stats()["ntotal"] == 0


def test_self_retrieval_flat():
    v = _unit_vectors(200)
    ids = np.arange(1, 201, dtype="int64")
    ix = VectorIndex(VectorIndexConfig(dimension=DIM))
    ix.add(v, ids)
    hits = sum(int(ix.search(v[i], 1)[1][0][0] == ids[i]) for i in range(v.shape[0]))
    assert hits == v.shape[0]                       # every vector retrieves itself


def test_recall_on_fixed_synthetic_set_meets_target():
    v = _unit_vectors(500, seed=7)
    ids = np.arange(1000, 1500, dtype="int64")
    ix = VectorIndex(VectorIndexConfig(dimension=DIM))
    ix.add(v, ids)
    correct = sum(int(ix.search(v[i], 1)[1][0][0] == ids[i]) for i in range(v.shape[0]))
    assert correct / v.shape[0] >= 0.99             # exact search -> ~perfect recall@1


def test_remove_on_flat():
    v = _unit_vectors(10)
    ids = np.arange(1, 11, dtype="int64")
    ix = VectorIndex(VectorIndexConfig(dimension=DIM))
    ix.add(v, ids)
    removed = ix.remove(np.array([3, 7], dtype="int64"))
    assert removed == 2
    assert ix.stats()["ntotal"] == 8
    returned = {int(x) for x in ix.search(v[2], 10)[1][0] if x != -1}
    assert 3 not in returned and 7 not in returned


def test_hnsw_add_with_ids_does_not_raise():
    v = _unit_vectors(64)
    ids = np.arange(1, 65, dtype="int64")
    ix = VectorIndex(VectorIndexConfig(dimension=DIM, index_type="hnsw"))
    ix.add(v, ids)                                   # IndexIDMap2 wraps HNSW (no native ids)
    assert ix.stats()["ntotal"] == 64
    assert ix.stats()["index_type"] == "hnsw"
    _, r = ix.search(v[0], 5)
    assert r.shape[1] >= 1


def test_hnsw_remove_raises():
    ix = VectorIndex(VectorIndexConfig(dimension=DIM, index_type="hnsw"))
    ix.add(_unit_vectors(8), np.arange(1, 9, dtype="int64"))
    with pytest.raises(NotImplementedError):
        ix.remove(np.array([1], dtype="int64"))      # HNSW: soft-delete + rebuild instead


def test_rebuild_excludes_dropped_id():
    v = _unit_vectors(3, seed=1)
    ix = VectorIndex(VectorIndexConfig(dimension=DIM))
    ix.add(v, np.array([10, 20, 30], dtype="int64"))
    ix.rebuild(v[[0, 2]], np.array([10, 30], dtype="int64"))   # 20 dropped from active set
    assert ix.stats()["ntotal"] == 2
    returned = {int(x) for x in ix.search(v[1], 3)[1][0] if x != -1}
    assert 20 not in returned


def test_empty_search_and_empty_rebuild():
    ix = VectorIndex(VectorIndexConfig(dimension=DIM))
    s, r = ix.search(_unit_vectors(1)[0], 5)
    assert s.shape[1] == 0 and r.shape[1] == 0       # empty index -> empty result
    ix.add(_unit_vectors(4), np.arange(1, 5, dtype="int64"))
    ix.rebuild(np.zeros((0, DIM), "float32"), np.zeros((0,), "int64"))  # empty active set
    assert ix.stats()["ntotal"] == 0                 # resets cleanly


# --------------------- store/service reconciliation ------------------------ #
@pytest.fixture(scope="module")
def consolidated_service():
    recs = load_interactions()
    svc = MemoryService(policy=MemoryPolicy(), embedder=HashingEmbedder())
    svc.ingest_records(recs)
    now = max(r["ts_day"] for r in recs) + 1
    svc.consolidate(now)
    svc.forget(now)
    svc.rebuild_index(now)
    return svc, now


def test_get_active_vectors_shapes(consolidated_service):
    svc, now = consolidated_service
    vecs, ids = svc.get_active_vectors(now)
    assert vecs.dtype == np.float32 and ids.dtype == np.int64
    assert vecs.ndim == 2 and vecs.shape[1] == DIM
    assert vecs.shape[0] == ids.shape[0] > 0
    assert len(set(ids.tolist())) == ids.shape[0]    # ids are unique


def test_supersession_rebuild_excludes_superseded_ids(consolidated_service):
    svc, now = consolidated_service
    _, ids = svc.get_active_vectors(now)
    live = set(ids.tolist())
    # every live index id maps to an active (non-superseded, non-expired) item
    assert all(svc._item_of_vec[i].superseded_by is None for i in live)
    # the manager fact was superseded (Alice Reyes -> Bob Tran)
    superseded = [it for it in svc.semantic.all() if it.superseded_by is not None]
    assert any(it.value == "Alice Reyes" for it in superseded)
    # ... and no superseded item is in the live index id-space
    sup_ids = {svc._vec_of[it.id] for it in superseded if it.id in svc._vec_of}
    assert sup_ids.isdisjoint(live)


def test_recall_returns_current_not_stale(consolidated_service):
    svc, now = consolidated_service
    hits = svc.recall("Who is my current manager?", k=5, now_day=now)
    values = [h.value for h in hits if h.value]
    assert "Bob Tran" in values                      # current value retrievable
    assert "Alice Reyes" not in values               # superseded value never surfaces
    assert all(isinstance(h.vec_id, int) for h in hits)  # stable int id assigned


def test_empty_service_rebuild_and_recall():
    svc = MemoryService(policy=MemoryPolicy(), embedder=HashingEmbedder())
    assert svc.rebuild_index(0) == 0                 # empty active set
    assert svc.recall("anything", k=3, now_day=0) == []
