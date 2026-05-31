"""Smoke test for the HTTP serving layer (local embeddings backend)."""
from __future__ import annotations

from fastapi.testclient import TestClient

from serve.app import app


def test_health_remember_recall_roundtrip():
    with TestClient(app) as client:
        assert client.get("/healthz").json()["status"] == "ok"

        client.post("/ingest", json={"records": [
            {"id": "A1", "text": "Mel moved to Lisbon for a new job", "ts_day": 1},
            {"id": "A2", "text": "Caroline adopted a dog named Pixel", "ts_day": 2},
            {"id": "A3", "text": "they discussed hiking the Azores in summer", "ts_day": 3},
        ]})

        hits = client.post("/recall", json={"query": "where did Mel move", "k": 3}).json()
        assert hits, "recall returned no hits"
        assert hits[0]["id"] == "A1"  # the relevant memory ranks first
        assert "X-Process-Time-Ms" in client.post(
            "/recall", json={"query": "dog", "k": 1}).headers
