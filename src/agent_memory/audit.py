"""Append-only audit log -- the explainability substrate.

Answers "why does the agent believe X, where did it come from, and when was a
prior value forgotten" -- the audit/retention/explainability requirement that
regulated deployments (SOX/HIPAA/GDPR lineage) impose on a memory layer."""
from __future__ import annotations

from .types import AuditEntry


class AuditLog:
    def __init__(self) -> None:
        self._entries: list[AuditEntry] = []

    def record(self, day: int, op: str, item_id: str, detail: str = "") -> None:
        self._entries.append(AuditEntry(day=day, op=op, item_id=item_id, detail=detail))

    def for_item(self, item_id: str) -> list[AuditEntry]:
        return [e for e in self._entries if e.item_id == item_id]

    def by_op(self, op: str) -> list[AuditEntry]:
        return [e for e in self._entries if e.op == op]

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self):
        return iter(self._entries)
