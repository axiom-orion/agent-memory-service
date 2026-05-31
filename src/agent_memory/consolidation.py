"""Consolidation: episodic fact statements -> semantic facts.

This is the step that distinguishes a memory *service* from a vector log:
  * dedupe        -- collapse repeated statements of the same value into one
                     terse fact (cuts the tokens needed to ground an answer)
  * supersession  -- when a fact's value changes over time, keep only the latest
                     as active and mark earlier values superseded, so retrieval
                     returns the *current* value rather than a stale one

Without supersession (the ablation rung), distinct values for a key coexist as
separate semantic facts -- deduped, but a stale value can still surface.
"""
from __future__ import annotations

from collections import defaultdict

from .audit import AuditLog
from .config import MemoryPolicy
from .embeddings import Embedder
from .stores import EpisodicMemory, SemanticMemory
from .types import MemoryItem, MemoryType


def _humanize(attribute: str) -> str:
    return attribute.replace("_", " ")


def consolidate(episodic: EpisodicMemory, semantic: SemanticMemory,
                embedder: Embedder, policy: MemoryPolicy, audit: AuditLog,
                now_day: int) -> None:
    """Populate `semantic` from `episodic` fact statements per the policy."""
    groups: dict[tuple[str, str], list[MemoryItem]] = defaultdict(list)
    for m in episodic.facts():
        groups[m.key].append(m)

    new_items: list[MemoryItem] = []
    sid = 0
    for key, stmts in groups.items():
        stmts.sort(key=lambda m: m.created_day)
        subject, attribute = key

        # group statements by value
        by_value: dict[str, list[MemoryItem]] = defaultdict(list)
        for m in stmts:
            by_value[m.value].append(m)

        if policy.use_supersession:
            # only the most-recently-asserted value survives as active
            latest_value = stmts[-1].value
            values_in_order = sorted(
                by_value, key=lambda v: max(x.created_day for x in by_value[v]))
            active_item = None
            for v in values_in_order:
                support = by_value[v]
                sid += 1
                item = MemoryItem(
                    id=f"S{sid:04d}", mtype=MemoryType.SEMANTIC,
                    content=f"{_humanize(attribute)}: {v}",
                    created_day=max(x.created_day for x in support),
                    importance=max(x.importance for x in support),
                    subject=subject, attribute=attribute, value=v,
                    support=len(support),
                    provenance=[x.id for x in support])
                semantic.add(item)
                new_items.append(item)
                if v == latest_value:
                    active_item = item
            # mark every non-latest value as superseded by the active one
            if active_item is not None:
                for sup_id in semantic.supersede(key, active_item):
                    audit.record(now_day, "supersede", sup_id,
                                 f"superseded-by:{active_item.id}")
            audit.record(now_day, "consolidate", active_item.id if active_item else "",
                         f"{key} -> '{latest_value}' from {len(stmts)} statements")
        else:
            # dedupe only: one fact per distinct value; all coexist (stale possible)
            for v, support in by_value.items():
                sid += 1
                item = MemoryItem(
                    id=f"S{sid:04d}", mtype=MemoryType.SEMANTIC,
                    content=f"{_humanize(attribute)}: {v}",
                    created_day=max(x.created_day for x in support),
                    importance=max(x.importance for x in support),
                    subject=subject, attribute=attribute, value=v,
                    support=len(support),
                    provenance=[x.id for x in support])
                semantic.add(item)
                new_items.append(item)
                audit.record(now_day, "consolidate", item.id,
                             f"{key}='{v}' from {len(support)} statements")

    # embed the new semantic items in one batch
    if new_items:
        vecs = embedder.encode([it.content for it in new_items])
        for it, v in zip(new_items, vecs, strict=True):
            it.embedding = v
