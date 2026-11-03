"""Fact extraction: schema-locked, abstaining, and measured -- no model, no network.

Supersession keys on (subject, attribute), so these tests pin the property the memory
eval depends on: facts extracted from free text land on the same keys as the
structured triples the eval was built on, and statements that assert no current fact
(negation, hypotheticals, questions, other people's facts, past values) yield none.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_memory import MemoryPolicy, MemoryService
from agent_memory.embeddings import HashingEmbedder
from agent_memory.extractor import (
    DEFAULT_MODEL,
    FACT_SCHEMA,
    FALLBACK_BETA,
    NO_FACT,
    NONE_ATTRIBUTE,
    LLMExtractor,
    RuleExtractor,
    StructuredExtractor,
)
from agent_memory.service import load_interactions

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
from run_eval import corpus_cases, load_cases, score_extraction  # noqa: E402

FULL = MemoryPolicy(use_recency=True, use_consolidation=True, use_supersession=True)
HELD_OUT = load_cases()


# --------------------------- rule extractor ---------------------------------- #
def test_rule_extractor_recovers_every_corpus_label():
    # Wiring check: the generator's phrasings are covered, so this must be exact.
    for r in load_interactions():
        assert RuleExtractor().extract(r) == StructuredExtractor().extract(r), r["text"]


def test_rule_extractor_on_held_out_never_extracts_a_wrong_or_false_fact():
    sc = score_extraction(RuleExtractor(), HELD_OUT)
    totals = {k: sum(c.get(k, 0) for c in sc["by_category"].values())
              for k in ("wrong", "false")}
    assert totals == {"wrong": 0, "false": 0}, sc["by_category"]
    assert sc["precision"] == 1.0
    # recall on unseen phrasings is genuinely partial; guard it from regressing
    assert sc["recall"] >= 0.86, sc


@pytest.mark.parametrize("case", [c for c in HELD_OUT if c["attribute"] is None],
                         ids=lambda c: f"{c['category']}:{c['text']}")
def test_rule_extractor_abstains(case):
    assert RuleExtractor().extract(case) == NO_FACT


# A fact pattern matches inside each of these, but the clause asserts nothing current.
# (Kept out of eval/extraction_cases.jsonl: they were written against the code.)
@pytest.mark.parametrize("text", [
    "I'm not sure my manager is Bob Tran.",              # negation
    "It's not like I'm working on Project Delta.",        # negation
    "If I'm working on Project Zephyr next year, I'll need a laptop.",  # hypothetical
    "Maybe my manager is Lena Ortiz now.",                # hypothetical
    "I'm working on Project Atlas, or I was until last week.",  # past
    "My manager is Alice Reyes, or was before the reorg.",       # past
])
def test_rule_extractor_skips_clauses_that_assert_nothing_current(text):
    assert RuleExtractor().extract({"text": text}) == NO_FACT


@pytest.mark.parametrize("text, attribute", [
    ("My manager is Alice Reyes.", "current_manager"),
    ("My boss is Alice Reyes.", "current_manager"),
    ("I report to Alice Reyes.", "current_manager"),
    ("Heads up, my manager is now Alice Reyes after the reorg.", "current_manager"),
])
def test_synonyms_land_on_one_key(text, attribute):
    _, got, value = RuleExtractor().extract({"text": text})
    assert (got, value) == (attribute, "Alice Reyes")


def test_two_different_facts_in_one_statement_abstain():
    text = "My manager is Alice Reyes and I live in Denver."
    assert RuleExtractor().extract({"text": text}) == NO_FACT


# --------------------------- service integration ----------------------------- #
def _service(records, extractor=None):
    svc = MemoryService(policy=FULL, embedder=HashingEmbedder())
    svc.ingest_records(records, extractor)
    now = max(r["ts_day"] for r in records) + 1
    svc.consolidate(now)
    return svc


def _active_facts(svc) -> dict:
    return {(it.subject, it.attribute): it.value for it in svc.semantic.all()
            if it.superseded_by is None and it.attribute}


def test_supersession_works_across_synonyms_from_text():
    records = [
        {"id": "a", "ts_day": 1, "text": "My manager is Alice Reyes."},
        {"id": "b", "ts_day": 5, "text": "My boss is now Bob Tran."},
    ]
    assert _active_facts(_service(records, RuleExtractor())) == {
        ("user", "current_manager"): "Bob Tran"}


def test_ingest_with_an_extractor_ignores_the_record_labels():
    rec = {"id": "a", "ts_day": 1, "text": "My boss is Priya Shah.",
           "subject": "user", "attribute": "wrong_label", "value": "nobody"}
    svc = MemoryService(policy=FULL, embedder=HashingEmbedder())
    svc.ingest_records([rec], RuleExtractor())
    item = svc.episodic.items()[0]
    assert (item.attribute, item.value) == ("current_manager", "Priya Shah")


def test_text_extraction_consolidates_to_the_same_facts_as_the_labels():
    records = load_interactions()
    assert _active_facts(_service(records, RuleExtractor())) == _active_facts(_service(records))


def test_scoring_counts_the_corpus_as_fully_recovered():
    sc = score_extraction(RuleExtractor(), corpus_cases(load_interactions()))
    assert (sc["precision"], sc["recall"], sc["false"]) == (1.0, 1.0, 0)


# --------------------------- LLM extractor (fake client) --------------------- #
class FakeClient:
    """Stands in for anthropic.Anthropic(): records requests, replays canned replies."""

    def __init__(self, reply=None, stop_reason="end_turn", raises=None):
        self.calls: list[dict] = []
        self._reply, self._stop, self._raises = reply, stop_reason, raises
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self._raises is not None:
            raise self._raises
        text = self._reply if isinstance(self._reply, str) else json.dumps(self._reply)
        return SimpleNamespace(stop_reason=self._stop,
                               content=[SimpleNamespace(type="text", text=text)])


def _llm(**kwargs):
    client = FakeClient(**kwargs)
    return LLMExtractor(client=client), client


TEXT = {"text": "My boss is now Bob Tran."}


def test_llm_valid_reply_becomes_a_fact():
    ex, _ = _llm(reply={"attribute": "current_manager", "subject": "user", "value": "Bob Tran"})
    assert ex.extract(TEXT) == ("user", "current_manager", "Bob Tran")


def test_llm_request_is_schema_locked_with_fallbacks():
    ex, client = _llm(reply={"attribute": NONE_ATTRIBUTE, "subject": "user", "value": ""})
    ex.extract(TEXT)
    req = client.calls[0]
    assert req["model"] == DEFAULT_MODEL
    assert req["fallbacks"] == "default" and req["betas"] == [FALLBACK_BETA]
    fmt = req["output_config"]["format"]
    assert fmt["type"] == "json_schema"
    assert fmt["schema"]["properties"]["attribute"]["enum"] == [*FACT_SCHEMA, NONE_ATTRIBUTE]
    assert fmt["schema"]["additionalProperties"] is False
    assert all(attr in req["system"] for attr in FACT_SCHEMA)
    assert req["messages"] == [{"role": "user", "content": TEXT["text"]}]


def _reply(attribute: str, value: str, subject: str = "user") -> dict:
    return {"attribute": attribute, "subject": subject, "value": value}


@pytest.mark.parametrize("reply, why", [
    (_reply(NONE_ATTRIBUTE, ""), "model says no fact"),
    (_reply("favourite_colour", "Bob Tran"), "slot not in schema"),
    (_reply("current_manager", "Bob Tran", subject="other"), "someone else's fact"),
    (_reply("current_manager", "Alice Reyes"), "value not in text"),
    (_reply("current_manager", "   "), "blank value"),
    ("not json at all", "malformed output"),
    (["current_manager"], "wrong JSON shape"),
])
def test_llm_rejects_unusable_replies(reply, why):
    ex, _ = _llm(reply=reply)
    assert ex.extract(TEXT) == NO_FACT, why


@pytest.mark.parametrize("stop_reason", ["refusal", "max_tokens"])
def test_llm_refusal_or_truncation_is_no_fact(stop_reason):
    reply = {"attribute": "current_manager", "subject": "user", "value": "Bob Tran"}
    ex, _ = _llm(reply=reply, stop_reason=stop_reason)
    assert ex.extract(TEXT) == NO_FACT


def test_llm_empty_statement_makes_no_request():
    ex, client = _llm(reply={})
    assert ex.extract({"text": "   "}) == NO_FACT and client.calls == []


def test_llm_transient_error_is_no_fact_but_config_errors_raise():
    anthropic = pytest.importorskip("anthropic")
    httpx = pytest.importorskip("httpx2")
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    ex, _ = _llm(raises=anthropic.APIConnectionError(request=request))
    assert ex.extract(TEXT) == NO_FACT
    bad = anthropic.BadRequestError("bad request", response=httpx.Response(400, request=request),
                                    body=None)
    ex, _ = _llm(raises=bad)
    with pytest.raises(anthropic.BadRequestError):
        ex.extract(TEXT)
    ex, _ = _llm(raises=ValueError("a bug, not an outage"))
    with pytest.raises(ValueError):
        ex.extract(TEXT)


# --------------------------- LLM extractor, open vocabulary (LoCoMo) --------- #
OPEN_TEXT = {"text": "Caroline: I'm a school counselor now."}


def _open(**kwargs):
    client = FakeClient(**kwargs)
    return LLMExtractor(client=client, schema=None), client


def test_open_vocabulary_keeps_the_speaker_as_subject():
    ex, client = _open(reply=_reply("occupation", "school counselor", subject="Caroline"))
    assert ex.extract(OPEN_TEXT) == ("Caroline", "occupation", "school counselor")
    attr_schema = client.calls[0]["output_config"]["format"]["schema"]["properties"]["attribute"]
    assert "enum" not in attr_schema  # no fixed slot list in this mode


@pytest.mark.parametrize("reply, why", [
    (_reply("occupation", "school counselor", subject="Melanie"), "subject never named"),
    (_reply(NONE_ATTRIBUTE, "school counselor", subject="Caroline"), "model says no fact"),
    (_reply("Job Title!", "school counselor", subject="Caroline"), "not a snake_case slot"),
    (_reply("occupation", "teacher", subject="Caroline"), "value not in text"),
])
def test_open_vocabulary_rejects_unusable_replies(reply, why):
    ex, _ = _open(reply=reply)
    assert ex.extract(OPEN_TEXT) == NO_FACT, why
