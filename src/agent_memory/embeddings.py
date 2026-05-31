"""Dense embedding pipeline (MiniLM) with on-disk caching."""
from __future__ import annotations

import hashlib
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
