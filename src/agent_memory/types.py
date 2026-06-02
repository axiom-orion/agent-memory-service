"""Core datatypes."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class MemoryType(str, Enum):
    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


@dataclass(slots=True)
class MemoryItem:
    id: str
    content: str
    mtype: MemoryType
    created_day: int                       # integer day timestamp (synthetic clock)
    importance: float = 0.5                # salience in [0, 1]
    # structured fact (optional). (subject, attribute) is the supersession key.
    subject: str | None = None
    attribute: str | None = None
    value: str | None = None
    # lifecycle
    ttl_days: int | None = None            # None = no expiry
    superseded_by: str | None = None       # id of the item that replaced this fact
    support: int = 1                       # how many source statements back this item
    last_access_day: int | None = None
    access_count: int = 0
    # lineage
    provenance: list[str] = field(default_factory=list)
    embedding: np.ndarray | None = field(default=None, repr=False)
    # stable int64 id for the FAISS index (the index keys on ints; the store keys
    # on string ids). Assigned once, on first registration with the index, and kept
    # stable across rebuilds so a memory's vector id never changes underneath it.
    vec_id: int | None = None

    @property
    def is_fact(self) -> bool:
        return self.attribute is not None

    @property
    def key(self) -> tuple[str, str] | None:
        return (self.subject or "", self.attribute) if self.attribute else None

    def expired(self, now_day: int) -> bool:
        return self.ttl_days is not None and (now_day - self.created_day) > self.ttl_days

    def token_estimate(self) -> int:
        return len(self.content.split())


@dataclass(slots=True)
class AuditEntry:
    day: int
    op: str                                # write | supersede | consolidate | forget | recall
    item_id: str
    detail: str = ""
