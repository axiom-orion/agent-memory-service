"""agent_memory: a typed, provenance-tracked memory service for LLM agents.

Working / episodic / semantic / procedural stores, with consolidation,
recency-aware retrieval, supersession-based forgetting, and an append-only
audit log for explainability."""
from .config import MemoryPolicy, Settings, settings
from .service import MemoryService
from .types import AuditEntry, MemoryItem, MemoryType

__all__ = [
    "MemoryService", "MemoryItem", "MemoryType", "AuditEntry",
    "MemoryPolicy", "Settings", "settings",
]
__version__ = "0.1.0"
