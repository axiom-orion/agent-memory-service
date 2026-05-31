"""The four memory stores of the standard cognitive-memory taxonomy.

  working    -- bounded, current-session scratch; evicts under capacity pressure
  episodic   -- append-only, time-stamped log of what happened
  semantic   -- distilled facts ("what is true"); fact-keyed, supports supersession
  procedural -- learned how-to procedures keyed by a trigger (minimal here)

Stores are thin containers. Consolidation (episodic -> semantic) and recency-aware
retrieval live in consolidation.py and service.py so the policy can ablate them.
"""
from __future__ import annotations

from .types import MemoryItem, MemoryType


class WorkingMemory:
    """Bounded short-term buffer. When over capacity, evicts the lowest-importance
    item (ties broken by age) and returns it so the caller can flush it to episodic."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._items: list[MemoryItem] = []

    def add(self, item: MemoryItem) -> MemoryItem | None:
        item.mtype = MemoryType.WORKING
        self._items.append(item)
        if len(self._items) <= self.capacity:
            return None
        victim = min(self._items, key=lambda it: (it.importance, -it.created_day))
        self._items.remove(victim)
        return victim

    def items(self) -> list[MemoryItem]:
        return list(self._items)


class EpisodicMemory:
    """Append-only log of events/interactions."""

    def __init__(self):
        self._items: list[MemoryItem] = []

    def add(self, item: MemoryItem) -> None:
        item.mtype = MemoryType.EPISODIC
        self._items.append(item)

    def items(self) -> list[MemoryItem]:
        return list(self._items)

    def facts(self) -> list[MemoryItem]:
        return [m for m in self._items if m.is_fact]

    def non_facts(self) -> list[MemoryItem]:
        return [m for m in self._items if not m.is_fact]


class SemanticMemory:
    """Distilled facts. Holds at most one *active* item per (subject, attribute) key
    when supersession is on; otherwise distinct values for a key may coexist."""

    def __init__(self):
        self._items: dict[str, MemoryItem] = {}

    def add(self, item: MemoryItem) -> None:
        item.mtype = MemoryType.SEMANTIC
        self._items[item.id] = item

    def supersede(self, key: tuple[str, str], new_item: MemoryItem) -> list[str]:
        """Mark all currently-active items for `key` as superseded by new_item.
        Returns the ids that were superseded (for the audit log)."""
        superseded: list[str] = []
        for it in self._items.values():
            if it.key == key and it.superseded_by is None and it.id != new_item.id:
                it.superseded_by = new_item.id
                superseded.append(it.id)
        return superseded

    def active(self) -> list[MemoryItem]:
        return [it for it in self._items.values() if it.superseded_by is None]

    def all(self) -> list[MemoryItem]:
        return list(self._items.values())


class ProceduralMemory:
    """Learned procedures keyed by a trigger phrase (kept minimal)."""

    def __init__(self):
        self._items: dict[str, MemoryItem] = {}

    def add(self, trigger: str, item: MemoryItem) -> None:
        item.mtype = MemoryType.PROCEDURAL
        self._items[trigger] = item

    def get(self, trigger: str) -> MemoryItem | None:
        return self._items.get(trigger)

    def items(self) -> list[MemoryItem]:
        return list(self._items.values())
