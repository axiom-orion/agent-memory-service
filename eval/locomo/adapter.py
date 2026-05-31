"""Ingest a LoCoMo conversation into a MemoryService as episodic memories.

Each dialogue turn becomes one episodic item: id = the turn's dia_id (so retrieval
can be scored against LoCoMo's gold evidence ids), timestamp = session index
(a monotonic clock for recency), content = "Speaker: text"."""
from __future__ import annotations

from agent_memory.service import MemoryService

from .loader import LocomoSample


def ingest_sample(service: MemoryService, sample: LocomoSample) -> None:
    records = [
        {"id": t.dia_id, "text": f"{t.speaker}: {t.text}", "ts_day": t.session,
         "importance": 0.5, "provenance": [f"locomo:{sample.sample_id}:{t.dia_id}"]}
        for t in sample.turns
    ]
    service.ingest_records(records)
