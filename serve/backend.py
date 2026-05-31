"""Backend selection for the serving layer."""
from __future__ import annotations

import os

from agent_memory.config import MemoryPolicy
from agent_memory.embeddings import Embedder
from agent_memory.service import MemoryService


def get_embedder():
    """local (default, all-MiniLM-L6-v2) or vertex (Vertex AI text-embeddings)."""
    if os.environ.get("EMBEDDINGS_BACKEND", "local").lower() == "vertex":
        from .embeddings_vertex import VertexEmbedder
        return VertexEmbedder()
    return Embedder()


def build_service() -> MemoryService:
    # Full lifecycle (recency + consolidation + supersession) enabled for the API.
    return MemoryService(policy=MemoryPolicy(), embedder=get_embedder())
