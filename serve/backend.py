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

    # Durable record when one is configured; otherwise the original in-RAM
    # service, unchanged. Configuring nothing is valid, not an error — but on a
    # scale-to-zero host it means memory does not survive between requests,
    # which is only acceptable for a demo.
    #
    #   DATABASE_URL / POSTGRES_URL  -> Postgres + pgvector (portable, any host)
    #   MEMORY_BACKEND=firestore     -> Firestore (no secret; runtime service account)
    #
    # Postgres wins if both are set, because an explicit DSN is an explicit
    # choice. Neither backend is asked to search — faiss remains the retrieval
    # engine (see pg_store.py).
    store = None
    ns = os.environ.get("MEMORY_NAMESPACE", "default")
    dim = int(os.environ.get("EMBEDDING_DIM", "384"))

    from agent_memory.pg_store import dsn_from_env
    dsn = dsn_from_env()
    if dsn:
        from agent_memory.pg_store import PgStore
        store = PgStore(dsn, namespace=ns, dimension=dim)
    else:
        from agent_memory.firestore_store import firestore_project_from_env
        project = firestore_project_from_env()
        if project:
            from agent_memory.firestore_store import FirestoreStore
            store = FirestoreStore(project, namespace=ns, dimension=dim)

    if store is None:
        return MemoryService(**kwargs)

    from agent_memory.persistent_service import PersistentMemoryService
    store.ensure_schema()
    return PersistentMemoryService(store=store, **kwargs)
