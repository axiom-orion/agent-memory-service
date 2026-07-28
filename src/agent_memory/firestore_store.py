"""Durable record on Firestore. Same contract as PgStore — see pg_store.py first.

WHY FIRESTORE AND NOT POSTGRES

The service deploys to Cloud Run in a GCP project that already has a Firestore
database, so this backend needs no connection string, no secret to rotate, and
no connection pooler: it authenticates as the Cloud Run runtime service account.
Every failure this service's operators actually hit is a credential or config
failure, not a query failure, and this removes that whole class.

The trade that makes it viable: faiss remains the retrieval engine (see
pg_store.py for why that is non-negotiable while the published evaluation
numbers stand), so the store is never asked to do a nearest-neighbour search.
It only has to durably hold MemoryItems and their embeddings, which a document
database does perfectly well. Postgres/pgvector stays supported for anyone
deploying outside GCP.

Embeddings are stored as a Firestore `Vector`, not a plain array. We do not use
`find_nearest` today, but the typed form costs nothing and means the ANN path is
available without a migration — on the same condition as pgvector's: re-run the
evaluation before switching retrieval engines.

LAYOUT
    agent_memory/{namespace}/items/{item_id}

A namespace is a document holding an `items` subcollection, so two callers on
one database cannot see each other's memories, and a namespace can be dropped
in one operation.
"""
from __future__ import annotations

import os
from typing import Iterable

import numpy as np

from .types import MemoryItem, MemoryType

ROOT_COLLECTION = "agent_memory"
_BATCH_LIMIT = 450          # Firestore caps a batch at 500 writes; leave headroom


def firestore_project_from_env() -> str | None:
    """The project to talk to, or None when Firestore is not configured.

    Absence is not an error: with nothing configured the service runs exactly as
    it always did (in-RAM, ephemeral). It must degrade, not refuse to boot.
    """
    if os.environ.get("MEMORY_BACKEND", "").lower() == "firestore":
        return (os.environ.get("FIRESTORE_PROJECT")
                or os.environ.get("GOOGLE_CLOUD_PROJECT")
                or os.environ.get("GCP_PROJECT"))
    return os.environ.get("FIRESTORE_PROJECT") or None


class FirestoreStore:
    """Write-through durable record. Not a retrieval engine."""

    def __init__(self, project: str | None = None, *, namespace: str = "default",
                 dimension: int = 384, client=None) -> None:
        self.project = project
        self.namespace = namespace
        self.dimension = dimension
        self._client = client

    # ---- connection -------------------------------------------------------
    @property
    def client(self):
        if self._client is None:
            from google.cloud import firestore  # lazy: the dependency is optional
            self._client = firestore.Client(project=self.project)
        return self._client

    def _items(self):
        return (self.client.collection(ROOT_COLLECTION)
                .document(self.namespace)
                .collection("items"))

    def ensure_schema(self) -> None:
        """Firestore is schemaless — nothing to create. Kept so the two backends
        are drop-in interchangeable behind PersistentMemoryService."""
        return None

    def healthy(self) -> bool:
        try:
            next(self._items().limit(1).stream(), None)
            return True
        except Exception:
            return False

    # ---- serialization ----------------------------------------------------
    @staticmethod
    def _vector(v):
        if v is None:
            return None
        from google.cloud.firestore_v1.vector import Vector
        return Vector([float(x) for x in np.asarray(v, dtype=np.float32).ravel().tolist()])

    def _doc(self, item: MemoryItem) -> dict:
        return {
            "id": item.id,
            "content": item.content,
            "mtype": item.mtype.value if isinstance(item.mtype, MemoryType) else str(item.mtype),
            "created_day": int(item.created_day),
            "importance": float(item.importance),
            "subject": item.subject,
            "attribute": item.attribute,
            "value": item.value,
            "ttl_days": item.ttl_days,
            "superseded_by": item.superseded_by,
            "support": int(item.support),
            "last_access_day": item.last_access_day,
            "access_count": int(item.access_count),
            "provenance": list(item.provenance or []),
            "vec_id": item.vec_id,
            "embedding": self._vector(item.embedding),
        }

    @staticmethod
    def _item(d: dict) -> MemoryItem:
        emb = d.get("embedding")
        if emb is not None:
            # A Firestore Vector round-trips as a Vector; a plain list is also
            # accepted so documents written before the typed form still load.
            raw = getattr(emb, "value", emb)
            emb = np.asarray(list(raw), dtype=np.float32)
        prov = d.get("provenance") or []
        return MemoryItem(
            id=d["id"], content=d["content"], mtype=MemoryType(d["mtype"]),
            created_day=int(d["created_day"]), importance=float(d.get("importance", 0.5)),
            subject=d.get("subject"), attribute=d.get("attribute"), value=d.get("value"),
            ttl_days=d.get("ttl_days"), superseded_by=d.get("superseded_by"),
            support=int(d.get("support", 1)), last_access_day=d.get("last_access_day"),
            access_count=int(d.get("access_count", 0)), provenance=list(prov),
            embedding=emb, vec_id=d.get("vec_id"),
        )

    # ---- write ------------------------------------------------------------
    def upsert(self, items: Iterable[MemoryItem]) -> int:
        docs = [(it.id, self._doc(it)) for it in items]
        if not docs:
            return 0        # never open a connection for an empty batch
        col = self._items()
        written = 0
        for start in range(0, len(docs), _BATCH_LIMIT):
            batch = self.client.batch()
            for doc_id, payload in docs[start:start + _BATCH_LIMIT]:
                batch.set(col.document(doc_id), payload)   # set(): full overwrite = upsert
            batch.commit()
            written += len(docs[start:start + _BATCH_LIMIT])
        return written

    # ---- read -------------------------------------------------------------
    def load(self, *, include_superseded: bool = True) -> list[MemoryItem]:
        out = [self._item(s.to_dict()) for s in self._items().stream()]
        if not include_superseded:
            out = [i for i in out if i.superseded_by is None]
        # Sorted in Python, not Firestore: ordering by two fields would demand a
        # composite index, and at this scale (10^2-10^4 items) it is free here.
        out.sort(key=lambda i: (i.created_day, i.id))
        return out

    def counts(self) -> dict:
        total = active = 0
        for s in self._items().stream():
            total += 1
            if s.to_dict().get("superseded_by") is None:
                active += 1
        return {"persisted_total": total, "persisted_active": active}
