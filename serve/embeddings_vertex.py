"""Vertex AI text-embeddings backend, duck-compatible with
`agent_memory.embeddings.Embedder` (exposes `.encode(texts) -> np.ndarray`).

Selected via EMBEDDINGS_BACKEND=vertex. Requires `pip install '.[vertex]'` and
Application Default Credentials (local: `gcloud auth application-default login`;
Cloud Run: an attached service account with the Vertex AI User role).

NOTE: this path calls a live Google Cloud API and was not exercised in the
authoring environment. Validate against your project before trusting it.
"""
from __future__ import annotations

import os

import numpy as np

_VERTEX_MAX_BATCH = 250  # text-embedding-* accepts up to 250 instances per call


class VertexEmbedder:
    def __init__(self, model_name: str | None = None,
                 project: str | None = None, location: str | None = None):
        self.model_name = model_name or os.environ.get("VERTEX_EMBED_MODEL", "text-embedding-004")
        self.project = project or os.environ.get("GOOGLE_CLOUD_PROJECT")
        self.location = location or os.environ.get("VERTEX_LOCATION", "us-central1")
        self._model = None

    def _load(self):
        if self._model is None:
            import vertexai
            from vertexai.language_models import TextEmbeddingModel
            vertexai.init(project=self.project, location=self.location)
            self._model = TextEmbeddingModel.from_pretrained(self.model_name)
        return self._model

    def encode(self, texts: list[str], use_cache: bool = True) -> np.ndarray:
        model = self._load()
        vecs: list[list[float]] = []
        for i in range(0, len(texts), _VERTEX_MAX_BATCH):
            batch = model.get_embeddings(texts[i:i + _VERTEX_MAX_BATCH])
            vecs.extend(e.values for e in batch)
        arr = np.asarray(vecs, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms  # L2-normalised to match the local backend
