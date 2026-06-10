"""PAG (Provenance Attestation Graph) — the claims, as executable assertions.

The fire-wall principle under test: PAG is the append-only slice only — hash-chained,
optionally signed, actor-attributed — and tamper is *localised*, not just detected.
Uses the HashingEmbedder throughout (deterministic, no model download), the same
discipline as the rest of the suite.
"""
from __future__ import annotations

import dataclasses

import pytest

from agent_memory.config import MemoryPolicy
from agent_memory.embeddings import HashingEmbedder
from agent_memory.pag import (
    GENESIS,
    ActorIdentity,
    AttestedAuditLog,
    ProvenanceLog,
    attest_embedder,
    verify_entries,
    weight_fingerprint,
)
from agent_memory.service import MemoryService

KEY = b"test-signing-key"


def _log(key: bytes | None = None) -> ProvenanceLog:
    log = ProvenanceLog(signing_key=key if key is not None else b"")
    # signing_key=b"" -> falsy -> unsigned; explicit, not env-dependent
    return log


# --- the chain -------------------------------------------------------------- #

def test_chain_links_and_verifies():
    log = _log()
    e1 = log.append(1, "write", "M0001", "episodic")
    e2 = log.append(1, "write", "M0002", "episodic")
    assert e1.prev_hash == GENESIS
    assert e2.prev_hash == e1.entry_hash
    r = log.verify()
    assert r.ok and r.length == 2 and r.broken_at is None and not r.signed


def test_tamper_is_localised_to_the_first_broken_entry():
    log = _log()
    for i in range(5):
        log.append(1, "write", f"M{i:04d}", "episodic")
    entries = [dataclasses.asdict(e) for e in log.entries()]
    entries[2]["detail"] = "history, rewritten"          # mutate the substance
    r = verify_entries(entries, None)
    assert not r.ok and r.broken_at == 2                 # named, not shrugged at


def test_dropping_an_entry_breaks_the_chain():
    log = _log()
    for i in range(4):
        log.append(1, "write", f"M{i:04d}")
    entries = [dataclasses.asdict(e) for e in log.entries()]
    del entries[1]
    assert not verify_entries(entries, None).ok


def test_content_addressing_is_deterministic_and_substance_sensitive():
    a = _log().append(1, "write", "M0001", "x", payload={"k": "v"})
    b = _log().append(1, "write", "M0001", "x", payload={"k": "v"})
    c = _log().append(1, "write", "M0001", "x", payload={"k": "OTHER"})
    assert a.content_hash == b.content_hash
    assert a.content_hash != c.content_hash


# --- signatures: honest when absent, checked when present ------------------- #

def test_signed_chain_verifies_and_detects_resigning_attempts():
    log = ProvenanceLog(signing_key=KEY)
    log.append(1, "write", "M0001")
    log.append(2, "supersede", "S0001", "S0001 -> S0002")
    r = log.verify()
    assert r.ok and r.signed and r.sig_failures == []
    forged = [dataclasses.asdict(e) for e in log.entries()]
    forged[1]["sig"] = "0" * 64                          # chain intact, signature wrong
    r2 = verify_entries(forged, KEY)
    assert not r2.ok and r2.sig_failures == [1] and r2.broken_at is None


def test_unsigned_is_reported_not_faked():
    r = _log().verify()
    assert r.signed is False                              # a missing key is named


# --- append-only by construction --------------------------------------------- #

def test_no_mutation_surface():
    log = _log()
    assert not any(m in dir(ProvenanceLog) for m in ("update", "delete", "remove", "pop"))
    log.append(1, "write", "M0001")
    with pytest.raises(dataclasses.FrozenInstanceError):
        log.entries()[0].detail = "rewritten"             # entries are frozen


# --- snapshot: replayable, and a tampered one is refused ---------------------- #

def test_snapshot_roundtrip_and_refusal():
    log = ProvenanceLog(signing_key=KEY)
    for i in range(3):
        log.append(i, "write", f"M{i:04d}", "episodic")
    snap = log.snapshot()
    restored = ProvenanceLog.restore(snap, signing_key=KEY)
    assert restored.head() == log.head() and len(restored) == 3
    snap["entries"][1]["detail"] = "rewritten"
    with pytest.raises(ValueError):
        ProvenanceLog.restore(snap, signing_key=KEY)


# --- the tee: existing audit semantics, plus identity --------------------------- #

def test_attested_audit_log_is_a_dropin_and_tees_with_identity():
    pag = _log()
    actor = ActorIdentity(agent_id="keeper", model_id="hashing-bow-64",
                          attestation_level="config-hash")
    audit = AttestedAuditLog(pag, actor)
    audit.record(3, "write", "M0001", "episodic")
    # plain AuditLog behavior preserved
    assert len(audit) == 1 and audit.for_item("M0001")[0].op == "write"
    # ...and the chained, attributed mirror exists
    e = pag.for_item("M0001")[0]
    assert e.agent_id == "keeper" and e.attestation_level == "config-hash"
    assert pag.verify().ok


def test_service_operations_land_in_the_pag_with_identity():
    svc = MemoryService(policy=MemoryPolicy(), embedder=HashingEmbedder(dim=64))
    svc.remember("Ada moved to Lisbon", day=1, subject="ada",
                 attribute="location", value="Lisbon")
    svc.remember("Ada moved to Madrid", day=5, subject="ada",
                 attribute="location", value="Madrid")
    svc.consolidate(now_day=6)
    ops = {e.op for e in svc.pag}
    assert "model-attest" in ops and "write" in ops and "consolidate" in ops
    assert svc.pag.entries()[0].op == "model-attest"     # identity first, memories after
    assert all(e.agent_id == "memory-service" for e in svc.pag)
    assert svc.pag.verify().ok


# --- model attestation grades (truth-in-claims) -------------------------------- #

def test_hashing_embedder_attests_at_config_hash_grade():
    att = attest_embedder(HashingEmbedder(dim=64))
    assert att["grade"] == "config-hash" and att["fingerprint"].startswith("cfg:")
    assert att == attest_embedder(HashingEmbedder(dim=64))     # deterministic
    assert att != attest_embedder(HashingEmbedder(dim=128))    # config-sensitive


def test_attest_model_restates_config_hash_for_weightless_embedder():
    svc = MemoryService(policy=MemoryPolicy(), embedder=HashingEmbedder(dim=64))
    att = svc.attest_model(now_day=9)
    assert att["grade"] == "config-hash"                  # no weights -> no inflation
    attests = [e for e in svc.pag if e.op == "model-attest"]
    assert len(attests) == 2 and svc.pag.verify().ok


def test_weight_fingerprint_value_addressed():
    import numpy as np
    sd = {"layer.w": np.ones((4, 4), dtype=np.float32),
          "layer.b": np.zeros(4, dtype=np.float32)}
    fp = weight_fingerprint(sd)
    assert fp.startswith("wfp:") and fp == weight_fingerprint(dict(reversed(sd.items())))
    sd2 = {k: v.copy() for k, v in sd.items()}
    sd2["layer.w"][0, 0] = 2.0
    assert weight_fingerprint(sd2) != fp


# --- durability: the chain survives a restart ------------------------------- #

def test_pag_durable_save_load_roundtrip(tmp_path):
    log = ProvenanceLog(signing_key=KEY)
    for i in range(3):
        log.append(i, "write", f"M{i:04d}", "episodic")
    p = tmp_path / "pag.json"
    log.save(p)
    restored = ProvenanceLog.load(p, signing_key=KEY)
    assert restored.head() == log.head() and len(restored) == 3 and restored.verify().ok


def test_pag_load_refuses_a_tampered_file(tmp_path):
    log = ProvenanceLog()
    for i in range(2):
        log.append(i, "write", f"M{i:04d}")
    p = tmp_path / "pag.json"
    log.save(p)
    import json
    d = json.loads(p.read_text())
    d["entries"][0]["detail"] = "history, rewritten on disk"
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError):
        ProvenanceLog.load(p)


def test_service_pag_persists_and_restores_continuing_the_chain(tmp_path):
    svc = MemoryService(policy=MemoryPolicy(), embedder=HashingEmbedder(dim=64))
    svc.remember("Ada moved to Lisbon", day=1, subject="ada", attribute="loc", value="Lisbon")
    p = tmp_path / "pag.json"
    svc.save_pag(p)
    head, length = svc.pag.head(), len(svc.pag)

    # a fresh instance (a "restart") adopts the verified chain and continues it
    svc2 = MemoryService(policy=MemoryPolicy(), embedder=HashingEmbedder(dim=64))
    svc2.restore_pag(p)
    assert svc2.pag.head() == head and len(svc2.pag) == length and svc2.pag.verify().ok
    svc2.remember("Ada moved to Madrid", day=5, subject="ada", attribute="loc", value="Madrid")
    assert len(svc2.pag) == length + 1 and svc2.pag.verify().ok  # appended, still intact
