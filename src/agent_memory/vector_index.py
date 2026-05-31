"""Dense index over memory-item embeddings. FAISS inner-product; numpy fallback.

Kept tiny and rebuildable: memory mutates (consolidation, forgetting), so the
index is rebuilt from the live item set rather than incrementally maintained."""
from __future__ import annotations

import numpy as np

try:
    import faiss
    _HAS_FAISS = True
except Exception:  # pragma: no cover
    _HAS_FAISS = False


class VectorIndex:
    def __init__(self, dim: int):
        self.dim = dim
        self.backend = "faiss-flat" if _HAS_FAISS else "numpy"
        self._faiss = None
        self._mat: np.ndarray | None = None
        self._ids: list[str] = []

    def build(self, ids: list[str], embeddings: np.ndarray) -> VectorIndex:
        self._ids = ids
        emb = np.ascontiguousarray(embeddings, dtype=np.float32)
        if _HAS_FAISS and len(ids):
            idx = faiss.IndexFlatIP(self.dim)
            idx.add(emb)
            self._faiss = idx
        else:
            self._mat = emb if len(ids) else np.zeros((0, self.dim), dtype=np.float32)
        return self

    def search(self, query_vec: np.ndarray, k: int) -> list[tuple[str, float]]:
        if not self._ids:
            return []
        q = np.ascontiguousarray(query_vec.reshape(1, -1), dtype=np.float32)
        if self._faiss is not None:
            scores, idx = self._faiss.search(q, min(k, len(self._ids)))
            return [(self._ids[i], float(s))
                    for i, s in zip(idx[0], scores[0], strict=False) if i != -1]
        sims = self._mat @ q[0]
        order = np.argsort(-sims)[:k]
        return [(self._ids[i], float(sims[i])) for i in order]
