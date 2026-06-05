# src/agent_memory/vector_index.py
"""
Vector index for agent-memory-service.

Default is EXACT inner-product search (IndexFlatIP). At this service's scale
(per-user / per-session memory: ~10^2-10^4 active vectors) exact search is
sub-millisecond and returns true nearest neighbors. Measured on faiss-cpu:
N=1k -> 0.05 ms, N=5k -> 0.27 ms, N=20k -> 1.3 ms per query.

HNSW is available behind config for the regime where exact search stops being
cheap (~>10^5 vectors), at the cost of *approximate* recall. It is intentionally
NOT the default: switching to HNSW below the crossover trades recall for a
latency win you do not have.

faiss HNSW does not implement add_with_ids; to carry external ids it must be
wrapped in IndexIDMap2. That wrapping is handled here so callers always get ids.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import faiss
import numpy as np


@dataclass(frozen=True)
class VectorIndexConfig:
    dimension: int = 384
    index_type: Literal["flat", "hnsw"] = "flat"  # exact by default; see module docstring
    normalize: bool = True                        # cosine via IP on L2-normalized vectors
    hnsw_m: int = 32                              # only used when index_type == "hnsw"
    hnsw_ef_construction: int = 200
    hnsw_ef_search: int = 64


class VectorIndex:
    """Inner-product vector index with id support and a clean rebuild path."""

    def __init__(self, config: VectorIndexConfig | None = None) -> None:
        self.config = config or VectorIndexConfig()
        self._hnsw: faiss.IndexHNSWFlat | None = None
        self._index = self._new_index()

    def _new_index(self) -> faiss.Index:
        dim = self.config.dimension
        if self.config.index_type == "hnsw":
            base = faiss.IndexHNSWFlat(dim, self.config.hnsw_m)
            base.hnsw.efConstruction = self.config.hnsw_ef_construction
            base.hnsw.efSearch = self.config.hnsw_ef_search
            self._hnsw = base
            return faiss.IndexIDMap2(base)  # HNSW has no native add_with_ids
        self._hnsw = None
        return faiss.IndexIDMap2(faiss.IndexFlatIP(dim))

    def _prepare(self, vectors: np.ndarray) -> np.ndarray:
        v = np.array(vectors, dtype=np.float32, copy=True)  # copy: never mutate caller
        if v.ndim == 1:
            v = v.reshape(1, -1)
        if v.shape[1] != self.config.dimension:
            raise ValueError(f"expected dim {self.config.dimension}, got {v.shape[1]}")
        v = np.ascontiguousarray(v)
        if self.config.normalize:
            faiss.normalize_L2(v)  # in-place on our copy
        return v

    def set_ef_search(self, ef_search: int) -> None:
        if self._hnsw is not None:
            self._hnsw.hnsw.efSearch = ef_search

    def add(self, vectors: np.ndarray, ids: np.ndarray) -> None:
        v = self._prepare(vectors)
        i = np.ascontiguousarray(np.asarray(ids, dtype=np.int64))
        if i.shape[0] != v.shape[0]:
            raise ValueError("vectors and ids length mismatch")
        self._index.add_with_ids(v, i)

    def search(self, query: np.ndarray, k: int = 5) -> tuple[np.ndarray, np.ndarray]:
        if self._index.ntotal == 0:
            return np.empty((1, 0), dtype=np.float32), np.empty((1, 0), dtype=np.int64)
        v = self._prepare(query)
        return self._index.search(v, min(k, self._index.ntotal))

    def rebuild(self, vectors: np.ndarray, ids: np.ndarray) -> None:
        """Rebuild from the ACTIVE set (post supersession/forget). For HNSW this
        restores topology; for flat it is a clean swap. Cheap at this scale."""
        self._index = self._new_index()
        if np.asarray(ids).shape[0] > 0:
            self.add(vectors, ids)

    def remove(self, ids: np.ndarray) -> int:
        """Exact index supports id removal; HNSW does not — soft-delete + rebuild()."""
        if self.config.index_type == "hnsw":
            raise NotImplementedError("HNSW cannot remove ids; soft-delete then rebuild()")
        sel = faiss.IDSelectorArray(np.ascontiguousarray(np.asarray(ids, dtype=np.int64)))
        return int(self._index.remove_ids(sel))

    def save(self, path: str) -> None:
        faiss.write_index(self._index, path)

    def load(self, path: str) -> None:
        self._index = faiss.read_index(path)
        self._hnsw = (faiss.downcast_index(self._index.index)
                      if self.config.index_type == "hnsw" else None)
        if self._hnsw is not None:
            self._hnsw.hnsw.efSearch = self.config.hnsw_ef_search

    def stats(self) -> dict:
        return {
            "index_type": self.config.index_type,
            "ntotal": int(self._index.ntotal),
            "dimension": self.config.dimension,
            "ef_search": (self._hnsw.hnsw.efSearch if self._hnsw is not None else None),
        }
