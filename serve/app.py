"""FastAPI surface for the agent memory service.

Public (rate-limited, read-only — serves a bundled synthetic corpus, never user data):
  GET  /health        -> liveness + active embeddings backend
  POST /recall        -> ranked retrieval        { query, k? } -> { records: [...] }   (C3)
  GET  /stats         -> active vs superseded counts          -> { active, superseded } (C3)

Bearer-guarded (set ADMIN_TOKEN; disabled — 503 — when unset, never left open):
  POST /admin/rebuild -> rebuild index from the active set     -> { rebuilt }           (C3)
  POST /remember /ingest /consolidate /forget -> mutate the in-memory store

The store is in-memory and single-instance: this deployment demonstrates the serving,
latency, and cost path. There is no background maintenance thread (Cloud Run freezes idle
instances); the index is rebuilt on write and on demand via /admin/rebuild, which a Cloud
Scheduler job calls on a cadence. Durable multi-instance memory (pgvector/Supabase or
Vertex Vector Search) is the documented production extension -- see DEPLOY.md.
"""
from __future__ import annotations

import os
import secrets
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .backend import build_service, get_embedder

_state: dict = {}

# A tiny synthetic fallback so the public endpoint always serves *something* even if the
# bundled corpus file is missing from the image. The real demo loads data/sessions/.
_FALLBACK_CORPUS = [
    {"id": "D0001", "text": "My manager is Alice Reyes.", "ts_day": 2,
     "subject": "user", "attribute": "current_manager", "value": "Alice Reyes",
     "importance": 0.8, "provenance": "synthetic"},
    {"id": "D0002", "text": "My manager is now Bob Tran after the reorg.", "ts_day": 40,
     "subject": "user", "attribute": "current_manager", "value": "Bob Tran",
     "importance": 0.8, "provenance": "synthetic"},
    {"id": "D0003", "text": "I live in Denver.", "ts_day": 1,
     "subject": "user", "attribute": "home_city", "value": "Denver",
     "importance": 0.7, "provenance": "synthetic"},
    {"id": "D0004", "text": "Reminder to prep the quarterly review deck.", "ts_day": 5,
     "subject": None, "attribute": None, "value": None,
     "importance": 0.3, "provenance": "synthetic"},
]


def _load_corpus() -> list[dict]:
    """The bundled synthetic corpus (deterministic, no user data)."""
    try:
        from agent_memory.service import load_interactions
        return load_interactions()
    except Exception as exc:  # pragma: no cover - packaging fallback
        print(f"[warmup] bundled corpus unavailable ({exc}); using fallback corpus")
        return list(_FALLBACK_CORPUS)


@asynccontextmanager
async def lifespan(app: FastAPI):
    svc = build_service()
    records = _load_corpus()
    svc.ingest_records(records)
    now = max((r.get("ts_day", 0) for r in records), default=0) + 1
    svc.consolidate(now)          # episodic -> semantic, supersede stale fact values
    svc.forget(now)               # apply TTL / prune superseded past grace
    svc.rebuild_index(now)        # build the FAISS index over the active set
    svc.embedder.encode(["warmup"])  # ensure the model/client is hot before traffic
    _state["svc"] = svc
    _state["now"] = now
    _state["backend"] = os.environ.get("EMBEDDINGS_BACKEND", "local")
    _state["corpus_size"] = len(records)
    yield
    _state.clear()


app = FastAPI(title="agent-memory-service", version="0.2.0", lifespan=lifespan)


# --------------------------------------------------------------------------- #
# Rate limiting: in-process fixed-window per client IP. Single-instance by design
# (deploy with --max-instances 1), so per-instance counting is global. Disabled when
# RATE_LIMIT_PER_MINUTE <= 0. /health is exempt so health checks are never throttled.
# --------------------------------------------------------------------------- #
_RL_WINDOW = 60.0
_RL_PER_MIN = int(os.environ.get("RATE_LIMIT_PER_MINUTE", "60"))
_rl_hits: dict[str, deque] = defaultdict(deque)


def _client_ip(request: Request) -> str:
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _rate_ok(ip: str, now: float) -> tuple[bool, int]:
    dq = _rl_hits[ip]
    cutoff = now - _RL_WINDOW
    while dq and dq[0] < cutoff:
        dq.popleft()
    if len(dq) >= _RL_PER_MIN:
        return False, max(1, int(_RL_WINDOW - (now - dq[0])) + 1)
    dq.append(now)
    if len(_rl_hits) > 10000:  # opportunistic bound on the IP table
        for k in [k for k, v in list(_rl_hits.items()) if not v]:
            _rl_hits.pop(k, None)
    return True, 0


@app.middleware("http")
async def rate_limit_and_timing(request: Request, call_next):
    start = time.perf_counter()
    if _RL_PER_MIN > 0 and request.url.path != "/health":
        ok, retry = _rate_ok(_client_ip(request), time.monotonic())
        if not ok:
            return JSONResponse(
                {"detail": "rate limit exceeded"}, status_code=429,
                headers={"Retry-After": str(retry),
                         "X-RateLimit-Limit": str(_RL_PER_MIN)})
    response = await call_next(request)
    response.headers["X-Process-Time-Ms"] = f"{(time.perf_counter() - start) * 1000:.2f}"
    return response


# --------------------------------------------------------------------------- #
# Bearer auth for mutating + admin routes. ADMIN_TOKEN is provided at deploy time
# (Secret Manager), never baked into the image. Unset => those routes return 503
# (disabled) rather than being left open.
# --------------------------------------------------------------------------- #
def require_admin(authorization: str | None = Header(default=None)) -> None:
    # .strip() the expected value too: secret managers (and Windows tooling) can leave a
    # trailing newline/CR on the stored token, which must not silently break auth.
    expected = (os.environ.get("ADMIN_TOKEN") or "").strip()
    if not expected:
        raise HTTPException(status_code=503,
                            detail="admin endpoints disabled: ADMIN_TOKEN not set")
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="missing bearer token")
    token = authorization.split(" ", 1)[1].strip()
    if not secrets.compare_digest(token, expected):
        raise HTTPException(status_code=401, detail="invalid bearer token")


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #
class RememberReq(BaseModel):
    text: str
    day: int = 0
    subject: str | None = None
    attribute: str | None = None
    value: str | None = None
    importance: float = Field(0.5, ge=0.0, le=1.0)


class Record(BaseModel):
    id: str
    text: str
    ts_day: int = 0
    importance: float = 0.5
    subject: str | None = None
    attribute: str | None = None
    value: str | None = None


class IngestReq(BaseModel):
    records: list[Record]


class RecallReq(BaseModel):
    query: str
    k: int = Field(5, ge=1, le=50)
    now_day: int | None = None


class MemoryRecord(BaseModel):
    """C3 contract record."""
    id: int
    content: str
    type: str
    importance: float
    superseded: bool


class RecallResp(BaseModel):
    records: list[MemoryRecord]


class StatsResp(BaseModel):
    active: int
    superseded: int


# --------------------------------------------------------------------------- #
# Public, read-only
# --------------------------------------------------------------------------- #
# NOTE: served at /health, not /healthz — Google's front end reserves/intercepts
# /healthz on *.run.app, so that path never reaches the container.
@app.get("/health")
def health():
    return {"status": "ok", "embeddings_backend": _state.get("backend", "local"),
            "corpus_size": _state.get("corpus_size", 0)}


@app.post("/recall", response_model=RecallResp)
def recall(req: RecallReq):
    svc = _state["svc"]
    now = req.now_day if req.now_day is not None else _state.get("now", 0)
    hits = svc.recall(req.query, k=req.k, now_day=now)
    return RecallResp(records=[
        MemoryRecord(id=int(h.vec_id or 0), content=h.content, type=h.mtype.value,
                     importance=float(h.importance),
                     superseded=h.superseded_by is not None)
        for h in hits])


@app.get("/stats", response_model=StatsResp)
def stats():
    svc = _state["svc"]
    return StatsResp(**svc.counts(_state.get("now", 0)))


# --------------------------------------------------------------------------- #
# Admin / mutating (Bearer-guarded)
# --------------------------------------------------------------------------- #
@app.post("/admin/rebuild", dependencies=[Depends(require_admin)])
def admin_rebuild():
    """Rebuild the index from the active set: VectorIndex.rebuild(*get_active_vectors()).
    Idempotent; safe on an empty active set (the index resets cleanly)."""
    svc = _state["svc"]
    return {"rebuilt": svc.rebuild_index(_state.get("now", 0))}


@app.post("/remember", dependencies=[Depends(require_admin)])
def remember(req: RememberReq):
    item = _state["svc"].remember(
        req.text, day=req.day, subject=req.subject, attribute=req.attribute,
        value=req.value, importance=req.importance)
    return {"id": item.id, "vec_id": item.vec_id}


@app.post("/ingest", dependencies=[Depends(require_admin)])
def ingest(req: IngestReq):
    _state["svc"].ingest_records([r.model_dump() for r in req.records])
    return {"ingested": len(req.records)}


@app.post("/consolidate", dependencies=[Depends(require_admin)])
def consolidate(now_day: int = 0):
    _state["svc"].consolidate(now_day)
    return {"ok": True}


@app.post("/forget", dependencies=[Depends(require_admin)])
def forget(now_day: int = 0):
    _state["svc"].forget(now_day)
    return {"ok": True}


__all__ = ["app", "get_embedder"]
