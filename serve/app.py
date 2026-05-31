"""FastAPI surface for the agent memory service.

Endpoints:
  GET  /healthz       -> liveness + active embeddings backend
  POST /remember      -> store one memory (optional subject/attribute/value triple)
  POST /ingest        -> store a batch of records
  POST /recall        -> ranked retrieval for a query
  POST /consolidate   -> run episodic->semantic consolidation (+ supersession)
  POST /forget        -> apply TTL / retention

The store is in-memory and single-instance: this deployment demonstrates the
serving, latency, and cost path. Durable multi-instance memory (pgvector/Supabase
or Vertex Vector Search) is the documented production extension -- see DEPLOY.md.
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from pydantic import BaseModel, Field

from .backend import build_service, get_embedder

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    import os
    svc = build_service()
    svc.embedder.encode(["warmup"])  # load model / init client before traffic
    _state["svc"] = svc
    _state["backend"] = os.environ.get("EMBEDDINGS_BACKEND", "local")
    yield
    _state.clear()


app = FastAPI(title="agent-memory-service", version="0.1.0", lifespan=lifespan)


@app.middleware("http")
async def add_process_time(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Process-Time-Ms"] = f"{(time.perf_counter() - start) * 1000:.2f}"
    return response


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


class IngestReq(BaseModel):
    records: list[Record]


class RecallReq(BaseModel):
    query: str
    k: int = Field(5, ge=1, le=50)
    now_day: int = 0


class Hit(BaseModel):
    id: str
    content: str
    created_day: int
    importance: float
    subject: str | None = None
    attribute: str | None = None
    value: str | None = None


@app.get("/healthz")
def healthz():
    return {"status": "ok", "embeddings_backend": _state.get("backend", "local")}


@app.post("/remember")
def remember(req: RememberReq):
    item = _state["svc"].remember(
        req.text, day=req.day, subject=req.subject, attribute=req.attribute,
        value=req.value, importance=req.importance)
    return {"id": item.id}


@app.post("/ingest")
def ingest(req: IngestReq):
    _state["svc"].ingest_records([r.model_dump() for r in req.records])
    return {"ingested": len(req.records)}


@app.post("/recall", response_model=list[Hit])
def recall(req: RecallReq):
    hits = _state["svc"].recall(req.query, k=req.k, now_day=req.now_day)
    return [Hit(id=h.id, content=h.content, created_day=h.created_day,
                importance=h.importance, subject=h.subject,
                attribute=h.attribute, value=h.value) for h in hits]


@app.post("/consolidate")
def consolidate(now_day: int = 0):
    _state["svc"].consolidate(now_day)
    return {"ok": True}


@app.post("/forget")
def forget(now_day: int = 0):
    _state["svc"].forget(now_day)
    return {"ok": True}


__all__ = ["app", "get_embedder"]
