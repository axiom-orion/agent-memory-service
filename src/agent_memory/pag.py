"""PAG — the Provenance Attestation Graph (the append-only slice, fire-walled).

This service deliberately *forgets* (TTL pruning, in-place semantic supersession) — the
opposite of a provenance ledger, which must never forget and must be tamper-evident. So
PAG is NOT the whole memory layer: it is the **append-only slice only** — a hash-chained,
optionally signed, actor-attributed log that every audit operation tees into. The
consolidation/TTL layer stays "memory" and is explicitly not the provenance of record.

What each entry adds over the plain :class:`~agent_memory.audit.AuditLog` line it mirrors:

  * **content addressing** — ``content_hash`` is the SHA-256 of the canonical payload, so
    an entry's substance is identified by its bytes, not its position;
  * **a hash chain** — ``entry_hash = sha256(prev_hash + canonical(entry))``: mutate or
    drop any entry and every later hash breaks, so ``verify()`` localises tamper to an
    index instead of shrugging;
  * **actor identity** — ``agent_id`` / ``model_id`` / ``attestation_level`` record *who*
    (which agent, under which model identity, at which attestation grade) performed the
    operation — the field the plain audit log never had;
  * **an optional HMAC signature** — set ``PAG_SIGNING_KEY`` and every entry is signed;
    without a key entries are honestly **unsigned** and ``verify()`` says so — a missing
    key is reported, never faked;
  * **a replay snapshot** — ``snapshot()`` exports the whole chain; ``restore()`` refuses
    a tampered one. "Why was this refused months later" is a verification away.

Attestation levels, weakest to strongest (truth-in-claims — say only what is checked):
``none`` < ``declared`` (a model name was recorded, nothing verified) < ``config-hash``
(deterministic embedder configuration hashed) < ``behavioral`` < ``weight-fingerprint``
(loaded weights digested — see :func:`attest_embedder`).
"""
from __future__ import annotations

import hashlib
import hmac as hmac_mod
import json
import os
from dataclasses import asdict, dataclass, field

from .audit import AuditLog

PAG_SCHEMA = "pag/1"
GENESIS = "pag:genesis"


def _canonical(obj: object) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()


def _sha(*parts: bytes) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p)
    return h.hexdigest()


@dataclass(frozen=True, slots=True)
class ActorIdentity:
    """Who is writing: the agent, the model identity it runs under, and how that
    identity is attested. Defaults are deliberately humble ("none"), never inflated."""

    agent_id: str = "memory-service"
    model_id: str | None = None
    attestation_level: str = "none"


@dataclass(frozen=True, slots=True)
class ProvenanceEntry:
    seq: int
    day: int
    op: str                      # write | supersede | consolidate | forget | recall | model-attest
    item_id: str
    detail: str
    agent_id: str
    model_id: str | None
    attestation_level: str
    content_hash: str            # sha256 of the canonical (op, item_id, detail, payload)
    prev_hash: str
    entry_hash: str
    sig: str | None = None       # HMAC-SHA256(entry_hash) when a signing key is configured
    payload: dict | None = None  # optional structured substance (e.g. a model attestation)


def _body_hash(prev_hash: str, e: dict) -> str:
    body = {k: e[k] for k in
            ("seq", "day", "op", "item_id", "detail", "agent_id", "model_id",
             "attestation_level", "content_hash", "payload")}
    return _sha(prev_hash.encode(), _canonical(body))


def _signing_key() -> bytes | None:
    raw = os.environ.get("PAG_SIGNING_KEY", "").strip()
    return raw.encode() if raw else None


@dataclass(frozen=True, slots=True)
class VerifyReport:
    ok: bool
    length: int
    broken_at: int | None        # seq of the first entry whose chain hash fails
    signed: bool                 # were entries signed at all (a key was configured)
    sig_failures: list[int] = field(default_factory=list)


class ProvenanceLog:
    """Append-only by construction: there is no update or delete surface, and the
    chain makes silent mutation detectable rather than merely impolite."""

    def __init__(self, signing_key: bytes | None = None):
        self._entries: list[ProvenanceEntry] = []
        # an empty key means "unsigned", consistently — append and verify must agree
        self._key = (signing_key if signing_key is not None else _signing_key()) or None

    # --- write (the only mutation there is) -------------------------------- #
    def append(self, day: int, op: str, item_id: str, detail: str = "", *,
               actor: ActorIdentity | None = None,
               payload: dict | None = None) -> ProvenanceEntry:
        a = actor or ActorIdentity()
        prev = self._entries[-1].entry_hash if self._entries else GENESIS
        content_hash = _sha(_canonical({"op": op, "item_id": item_id,
                                        "detail": detail, "payload": payload}))
        draft = {
            "seq": len(self._entries), "day": day, "op": op, "item_id": item_id,
            "detail": detail, "agent_id": a.agent_id, "model_id": a.model_id,
            "attestation_level": a.attestation_level, "content_hash": content_hash,
            "payload": payload,
        }
        entry_hash = _body_hash(prev, draft)
        sig = (hmac_mod.new(self._key, entry_hash.encode(), hashlib.sha256).hexdigest()
               if self._key else None)
        entry = ProvenanceEntry(prev_hash=prev, entry_hash=entry_hash, sig=sig, **draft)
        self._entries.append(entry)
        return entry

    # --- read ---------------------------------------------------------------- #
    def entries(self) -> list[ProvenanceEntry]:
        return list(self._entries)

    def for_item(self, item_id: str) -> list[ProvenanceEntry]:
        return [e for e in self._entries if e.item_id == item_id]

    def head(self) -> str:
        return self._entries[-1].entry_hash if self._entries else GENESIS

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self):
        return iter(self._entries)

    # --- verification ---------------------------------------------------------- #
    def verify(self) -> VerifyReport:
        return verify_entries([asdict(e) for e in self._entries], self._key)

    # --- replay snapshot --------------------------------------------------------- #
    def snapshot(self) -> dict:
        return {"schema": PAG_SCHEMA, "head": self.head(),
                "entries": [asdict(e) for e in self._entries]}

    @classmethod
    def restore(cls, snap: dict, signing_key: bytes | None = None) -> ProvenanceLog:
        """Rebuild a log from a snapshot, refusing one whose chain does not verify."""
        key = signing_key if signing_key is not None else _signing_key()
        report = verify_entries(snap.get("entries", []), key)
        if not report.ok:
            raise ValueError(f"snapshot failed verification (broken at seq {report.broken_at}, "
                             f"sig failures {report.sig_failures})")
        log = cls(signing_key=key)
        for d in snap.get("entries", []):
            log._entries.append(ProvenanceEntry(**d))
        if snap.get("head") not in (None, log.head()):
            raise ValueError("snapshot head does not match its entries")
        return log


def verify_entries(entries: list[dict], key: bytes | None) -> VerifyReport:
    """Recompute the whole chain (and signatures, when a key is present)."""
    prev = GENESIS
    broken_at: int | None = None
    sig_failures: list[int] = []
    for d in entries:
        expected = _body_hash(prev, d)
        if d.get("prev_hash") != prev or d.get("entry_hash") != expected:
            broken_at = int(d.get("seq", -1))
            break
        if key is not None:
            want = hmac_mod.new(key, expected.encode(), hashlib.sha256).hexdigest()
            if d.get("sig") != want:
                sig_failures.append(int(d.get("seq", -1)))
        prev = d["entry_hash"]
    return VerifyReport(ok=broken_at is None and not sig_failures, length=len(entries),
                        broken_at=broken_at, signed=key is not None,
                        sig_failures=sig_failures)


class AttestedAuditLog(AuditLog):
    """The existing audit log, teeing every record into the PAG with actor identity.

    Drop-in: ``record(day, op, item_id, detail)`` keeps its exact signature, so the
    consolidation/retention call sites are untouched — they now leave a chained,
    attributable trail for free.
    """

    def __init__(self, pag: ProvenanceLog, actor: ActorIdentity | None = None):
        super().__init__()
        self.pag = pag
        self.actor = actor or ActorIdentity()

    def record(self, day: int, op: str, item_id: str, detail: str = "") -> None:
        super().record(day, op, item_id, detail)
        self.pag.append(day, op, item_id, detail, actor=self.actor)


def attest_embedder(embedder: object) -> dict:
    """Identity of the embedding model, at the strongest grade obtainable *cheaply*.

    * ``HashingEmbedder`` -> ``config-hash``: it has no weights; its identity IS its
      configuration (algorithm + dimension), hashed.
    * ``Embedder`` (MiniLM) -> ``declared`` here: the model *name* is recorded without
      loading weights (laziness is load-bearing — construction must not download a
      model). Call :meth:`MemoryService.attest_model` to upgrade to
      ``weight-fingerprint`` by actually loading and digesting the weights.
    """
    dim = getattr(embedder, "dim", None)
    if dim is not None and not hasattr(embedder, "model_name"):
        ident = {"algorithm": "hashing-bow/blake2b8", "dim": int(dim)}
        return {"model": f"hashing-bow-{int(dim)}", "grade": "config-hash",
                "fingerprint": "cfg:" + _sha(_canonical(ident))[:32]}
    name = getattr(embedder, "model_name", None)
    return {"model": str(name) if name else type(embedder).__name__,
            "grade": "declared", "fingerprint": None}


def weight_fingerprint(state_dict: dict) -> str:
    """Loaded-state digest over tensor values (name, shape, canonical float32 bytes) —
    the same deployable-exact form as genealogy-graphrag's attest module: catches
    post-load tampering and survives benign re-serialization; NOT quantization-robust."""
    import numpy as np
    h = hashlib.sha256()
    for name in sorted(state_dict):
        t = state_dict[name]
        if hasattr(t, "detach"):
            t = t.detach().to("cpu").float().numpy()
        a = np.ascontiguousarray(np.asarray(t), dtype="<f4")
        h.update(name.encode())
        h.update(repr(tuple(int(d) for d in a.shape)).encode())
        h.update(a.tobytes())
    return "wfp:" + h.hexdigest()[:32]
