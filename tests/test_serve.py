"""HTTP serving tests against the C3 contract.

Uses the deterministic `hash` embedding backend so no transformer is downloaded; the
lifespan loads the bundled synthetic corpus and builds the FAISS index. Env is set
before importing the app because the rate-limit / auth knobs are read at import time.
"""
from __future__ import annotations

import os

os.environ.setdefault("EMBEDDINGS_BACKEND", "hash")
os.environ.setdefault("ADMIN_TOKEN", "test-token")
os.environ.setdefault("RATE_LIMIT_PER_MINUTE", "100000")

from fastapi.testclient import TestClient  # noqa: E402

from serve.app import app  # noqa: E402

AUTH = {"Authorization": "Bearer test-token"}


def test_health():
    with TestClient(app) as client:
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert body["corpus_size"] > 0


def test_recall_c3_contract():
    with TestClient(app) as client:
        r = client.post("/recall", json={"query": "Who is my current manager?", "k": 5})
        assert r.status_code == 200
        records = r.json()["records"]
        assert records, "recall returned no records"
        rec = records[0]
        assert set(rec) == {"id", "content", "type", "importance", "superseded"}
        assert isinstance(rec["id"], int)
        assert rec["type"] in {"working", "episodic", "semantic", "procedural"}
        # the current value surfaces; the superseded value never does
        contents = " ".join(x["content"] for x in records)
        assert "Bob Tran" in contents
        assert "Alice Reyes" not in contents
        assert "X-Process-Time-Ms" in client.post("/recall", json={"query": "x"}).headers


def test_stats_c3_contract():
    with TestClient(app) as client:
        stats = client.get("/stats").json()
        assert set(stats) == {"active", "superseded"}
        assert stats["active"] > 0
        assert stats["superseded"] >= 1


def test_admin_rebuild_requires_bearer():
    with TestClient(app) as client:
        assert client.post("/admin/rebuild").status_code == 401
        assert client.post(
            "/admin/rebuild", headers={"Authorization": "Bearer wrong"}).status_code == 401
        ok = client.post("/admin/rebuild", headers=AUTH)
        assert ok.status_code == 200
        assert ok.json()["rebuilt"] > 0


def test_mutations_are_guarded_and_work_with_auth():
    with TestClient(app) as client:
        assert client.post("/ingest", json={"records": []}).status_code == 401
        assert client.post("/remember", json={"text": "x"}).status_code == 401
        n = client.post("/ingest", headers=AUTH, json={"records": [
            {"id": "Z1", "text": "Zelda the cat likes tuna", "ts_day": 1}]})
        assert n.json()["ingested"] == 1
        client.post("/admin/rebuild", headers=AUTH)
        hits = client.post("/recall", json={"query": "Zelda the cat", "k": 5}).json()["records"]
        assert any("Zelda" in h["content"] for h in hits)


def test_rate_limit_returns_429(monkeypatch):
    import serve.app as appmod
    monkeypatch.setattr(appmod, "_RL_PER_MIN", 3)
    appmod._rl_hits.clear()
    with TestClient(app) as client:
        codes = [client.post("/recall", json={"query": "x"}).status_code for _ in range(8)]
    assert 429 in codes
    assert codes.count(200) <= 3


def test_federation_seam_contract():
    """The exact endpoints + record shape cason's Keeper (memory-client.js) depends on:
    POST /ingest (admin) -> POST /recall (public) -> GET /stats. Pinned here so the
    cross-repo seam can't drift out from under the consumer."""
    with TestClient(app) as client:
        ing = client.post("/ingest", headers=AUTH, json={"records": [
            {"id": "SEAM1", "text": "The Keeper confirmed the 1635 Harwood patent.",
             "ts_day": 3, "subject": "thomas", "attribute": "patent", "value": "1635"}]})
        assert ing.json()["ingested"] == 1
        client.post("/admin/rebuild", headers=AUTH)
        recs = client.post("/recall", json={"query": "Harwood patent", "k": 5}).json()["records"]
        assert recs and set(recs[0]) == {"id", "content", "type", "importance", "superseded"}
        stats = client.get("/stats").json()
        assert set(stats) == {"active", "superseded"} and stats["active"] > 0


def test_pag_verify_endpoint_reports_integrity_and_actor():
    with TestClient(app) as client:
        v0 = client.get("/pag/verify").json()
        assert v0["ok"] is True and v0["length"] >= 1
        assert v0["signed"] is False                     # unsigned by default, said plainly
        assert v0["actor"]["attestation_level"] in {"config-hash", "declared"}
        # a write extends the chain; integrity still holds
        client.post("/remember", headers=AUTH, json={"text": "durable note", "day": 9})
        v1 = client.get("/pag/verify").json()
        assert v1["ok"] is True and v1["length"] > v0["length"]


def test_pag_snapshot_is_admin_guarded_and_matches_the_chain():
    with TestClient(app) as client:
        assert client.get("/pag/snapshot").status_code == 401
        snap = client.get("/pag/snapshot", headers=AUTH).json()
        head = client.get("/pag/verify").json()["head"]
        assert snap["head"] == head and len(snap["entries"]) >= 1
        assert snap["entries"][0]["op"] == "model-attest"   # identity first
