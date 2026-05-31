"""Demo: recall a fact and show the audit trail behind it.

  python scripts/demo.py "Who is my current manager?"
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent_memory import MemoryService  # noqa: E402
from agent_memory.service import load_interactions  # noqa: E402


def main() -> None:
    query = " ".join(sys.argv[1:]) or "Who is my current manager?"
    records = load_interactions()
    now = max(r["ts_day"] for r in records) + 1
    svc = MemoryService()           # default policy = full (recency+consolidation+supersession)
    svc.ingest_records(records)
    svc.consolidate(now)
    svc.forget(now)

    hits = svc.recall(query, k=3, now_day=now)
    print(f"Q: {query}\n")
    print("Recalled memories (current, stale suppressed):")
    for i, h in enumerate(hits, 1):
        tag = f"{h.subject}.{h.attribute}={h.value}" if h.is_fact else "episodic"
        print(f"  [{i}] {h.content}   ({tag}; support={h.support}; "
              f"from {len(h.provenance)} source(s))")

    top = hits[0]
    if top.is_fact:
        superseded = [it for it in svc.semantic.all()
                      if it.key == top.key and it.superseded_by]
        if superseded:
            print(f"\nSuperseded prior values for {top.attribute}:")
            for it in superseded:
                print(f"  - {it.value}  (provenance {it.provenance})")
