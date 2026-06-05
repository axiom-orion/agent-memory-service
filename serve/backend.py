"""Backend selection for the serving layer."""
from __future__ import annotations

import os

from agent_memory.config import MemoryPolicy
from agent_memory.embeddings import Embedder
from agent_memory.service import MemoryService


def get_embedder():
    """Select the embedding backend from EMBEDDINGS_BACKEND:

      local  (default) -- all-MiniLM-L6-v2, real semantic recall (production demo)
      vertex           -- Vertex AI text-embeddings (off-box, set GOOGLE_CLOUD_PROJECT)
      hash             -- deterministic dependency-free hashing (tests/dev only)
    """
    backend = os.environ.get("EMBEDDINGS_BACKEND", "local").lower()
    if backend == "vertex":
        from .embeddings_vertex import VertexEmbedder
        return VertexEmbedder()
    if backend == "hash":
        from agent_memory.embeddings import HashingEmbedder
        return HashingEmbedder()
    return Embedder()


def build_service() -> MemoryService:
    # Full lifecycle (recency + consolidation + supersession) enabled for the API.
    # index_type stays "flat" (exact) by default; INDEX_TYPE=hnsw opts into ANN.
    return MemoryService(policy=MemoryPolicy(), embedder=get_embedder(),
                         index_type=os.environ.get("INDEX_TYPE", "flat"))
