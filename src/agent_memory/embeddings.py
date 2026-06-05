"""Dense embedding pipeline (MiniLM) with on-disk caching.

Two embedders share one interface -- ``encode(texts, use_cache=True) -> float32[n, d]``
returning L2-normalized rows (so inner product == cosine, which is what the FAISS
index assumes):

  * :class:`Embedder`        -- MiniLM (``all-MiniLM-L6-v2``), real semantic recall.
  * :class:`HashingEmbedder` -- deterministic, dependency-free bag-of-words hashing
                                for tests/dev. NOT production recall quality.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np

from .config import settings


class Embedder:
    def __init__(self, model_name: str | None = None, cache_dir: Path | None = None):
        self.model_name = model_name or settings.embed_model
        self.cache_dir = cache_dir or settings.cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _key(self, texts: list[str]) -> Path:
        h = hashlib.sha256(
            (self.model_name + "\x00" + "\x00".join(texts)).encode()).hexdigest()[:24]
        return self.cache_dir / f"emb-{h}.npy"

    def encode(self, texts: list[str], use_cache: bool = True) -> np.ndarray:
        if use_cache:
            ck = self._key(texts)
            if ck.exists():
                return np.load(ck)
        vecs = np.asarray(self._load().encode(
            texts, normalize_embeddings=True, show_progress_bar=False, batch_size=64),
            dtype=np.float32)
        if use_cache:
            np.save(self._key(texts), vecs)
        return vecs


class HashingEmbedder:
    """Deterministic, dependency-free embeddings for tests/dev.

    Hashes word tokens into a fixed-dimension bag-of-words vector, L2-normalized so
    inner product equals cosine -- the same contract the FAISS index expects of
    MiniLM. Identical text yields an identical vector (so self-retrieval is exact);
    token overlap drives similarity. This is **not** a substitute for MiniLM's
    semantic recall -- it exists so the index/serving paths can be exercised without
    downloading torch + a transformer model.
    """

    def __init__(self, dim: int | None = None):
        self.dim = dim or 384

    def encode(self, texts: list[str], use_cache: bool = True) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for tok in re.findall(r"[a-z0-9]+", text.lower()):
                h = int(hashlib.blake2b(tok.encode(), digest_size=8).hexdigest(), 16)
                out[row, h % self.dim] += 1.0
            norm = float(np.linalg.norm(out[row]))
            if norm > 0.0:
                out[row] /= norm
        return out
