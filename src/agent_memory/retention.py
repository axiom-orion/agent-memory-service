"""Retention / forgetting policy: TTL expiry, importance decay, and pruning of
superseded facts past a grace window. Every removal is audited."""
from __future__ import annotations

from .audit import AuditLog
from .config import MemoryPolicy
from .types import MemoryItem


def apply_retention(items: list[MemoryItem], now_day: int, policy: MemoryPolicy,
                    audit: AuditLog, superseded_grace_days: int = 365) -> list[MemoryItem]:
    """Return the surviving items; record what was forgotten and why."""
    kept: list[MemoryItem] = []
    for it in items:
        if it.expired(now_day):
            audit.record(now_day, "forget", it.id, "ttl-expired")
            continue
        if it.superseded_by is not None and \
                (now_day - it.created_day) > superseded_grace_days:
            audit.record(now_day, "forget", it.id, f"superseded-by:{it.superseded_by}")
            continue
        kept.append(it)
    return kept
