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
    kwargs = dict(policy=MemoryPolicy(), embedder=get_embedder(),
                  index_type=os.environ.get("INDEX_TYPE", "flat"))

    # Durable record when DATABASE_URL is set; otherwise the original in-RAM
    # service, unchanged. Absence of a DSN is a valid configuration, not an
    # error — but on a scale-to-zero host it means memory does not survive
    # between requests, which is only acceptable for a demo.
    from agent_memory.pg_store import dsn_from_env
    dsn = dsn_from_env()
    if not dsn:
        return MemoryService(**kwargs)

    from agent_memory.pg_store import PgStore
    from agent_memory.persistent_service import PersistentMemoryService
    store = PgStore(
        dsn,
        namespace=os.environ.get("MEMORY_NAMESPACE", "default"),
        dimension=int(os.environ.get("EMBEDDING_DIM", "384")),
    )
    store.ensure_schema()
    return PersistentMemoryService(store=store, **kwargs)
