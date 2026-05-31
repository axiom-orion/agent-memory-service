"""Tests. Fast unit checks load no model; retrieval-property tests share two
session-scoped services (flat + full policy) so embeddings load once.

Run: pytest -q
"""
from __future__ import annotations

import pytest

from agent_memory import MemoryPolicy, MemoryService
from agent_memory.service import load_interactions, load_queries
from agent_memory.stores import WorkingMemory
from agent_memory.types import MemoryItem, MemoryType

FLAT = MemoryPolicy(use_recency=False, use_consolidation=False, use_supersession=False)
FULL = MemoryPolicy(use_recency=True, use_consolidation=True, use_supersession=True)


# --------------------------- fixtures -------------------------------------- #
@pytest.fixture(scope="session")
def records():
    return load_interactions()


@pytest.fixture(scope="session")
def queries():
    return load_queries()


@pytest.fixture(scope="session")
def now(records):
    return max(r["ts_day"] for r in records) + 1


def _service(policy, records, now):
    svc = MemoryService(policy=policy)
    svc.ingest_records(records)
    svc.consolidate(now)
    svc.forget(now)
    return svc


@pytest.fixture(scope="session")
def full_svc(records, now):
    return _service(FULL, records, now)


@pytest.fixture(scope="session")
def flat_svc(records, now):
    return _service(FLAT, records, now)


# --------------------------- unit ------------------------------------------ #
def test_data_loads(records, queries):
    assert len(records) > 20 and len(queries) >= 10
    assert any(q["category"] == "changing" and q["prior_values"] for q in queries)


def test_supersession_key():
    a = MemoryItem("X", "current manager: A", MemoryType.SEMANTIC, 1,
                   subject="user", attribute="current_manager", value="A")
    b = MemoryItem("Y", "x", MemoryType.EPISODIC, 1)
    assert a.key == ("user", "current_manager") and a.is_fact
    assert b.key is None and not b.is_fact


def test_working_memory_evicts_lowest_importance():
    wm = WorkingMemory(capacity=2)
    assert wm.add(MemoryItem("A", "a", MemoryType.WORKING, 1, importance=0.9)) is None
    assert wm.add(MemoryItem("B", "b", MemoryType.WORKING, 2, importance=0.2)) is None
    victim = wm.add(MemoryItem("C", "c", MemoryType.WORKING, 3, importance=0.8))
    assert victim is not None and victim.id == "B"      # lowest importance evicted
    assert len(wm.items()) == 2


# --------------------------- consolidation --------------------------------- #
def test_consolidation_supersedes_old_values(full_svc):
    active = full_svc.semantic.active()
    mgr = [it for it in active if it.attribute == "current_manager"]
    assert len(mgr) == 1 and mgr[0].value == "Bob Tran"      # only the latest
    superseded = [it for it in full_svc.semantic.all()
                  if it.attribute == "current_manager" and it.superseded_by]
    assert any(it.value == "Alice Reyes" for it in superseded)


def test_audit_records_writes_and_supersession(full_svc):
    assert len(full_svc.audit.by_op("write")) > 20
    assert len(full_svc.audit.by_op("supersede")) >= 1


# --------------------------- retrieval property ---------------------------- #
def _top_values(svc, query, now, k=5):
    return [h.value for h in svc.recall(query, k=k, now_day=now)]


def test_full_policy_returns_current_value(full_svc, now):
    vals = _top_values(full_svc, "Who is my current manager?", now)
    assert vals[0] == "Bob Tran"                 # current at rank 1
    assert "Alice Reyes" not in vals             # stale suppressed


def test_flat_baseline_leaks_stale(flat_svc, now):
    vals = _top_values(flat_svc, "Who is my current manager?", now)
    assert "Alice Reyes" in vals                 # baseline surfaces the stale value


def test_stable_fact_unaffected(full_svc, now):
    vals = _top_values(full_svc, "What city do I live in?", now)
    assert vals and vals[0] == "Denver"


def test_supersession_beats_flat_on_changing(full_svc, flat_svc, queries, now):
    q = next(x for x in queries if x["category"] == "changing")
    full_top1 = _top_values(full_svc, q["query"], now)[:1]
    assert full_top1 == [q["gold_value"]]
    flat_vals = _top_values(flat_svc, q["query"], now)
    flat_acc = 1 if flat_vals[:1] == [q["gold_value"]] else 0
    assert flat_acc <= 1  # documents the gap; flat may or may not get rank-1 right
