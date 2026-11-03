"""Fact extraction: turn a raw statement into one (subject, attribute, value) fact.

Supersession only works if every statement of a fact lands on the SAME key: "my
manager is Alice" and "my boss is now Bob" must both become `current_manager`, or the
stale value survives as a separate fact and the memory service quietly returns it. So
extraction here is **schema-locked**: an extractor may only emit an attribute from
FACT_SCHEMA, and anything it cannot place there is *not* a fact -- the statement is
kept as an episodic memory instead of being guessed into the wrong slot.

Three extractors share one interface, `extract(record) -> (subject, attribute, value)`
(all None when the statement asserts no current first-person fact):

  StructuredExtractor  passthrough of fields an upstream agent already emitted (oracle)
  RuleExtractor        deterministic patterns from the text alone -- no network, no key
  LLMExtractor         Claude with a JSON-schema-constrained output over FACT_SCHEMA,
                       validated so a value must appear verbatim in the statement

Both text extractors ABSTAIN rather than guess: questions, negation ("no longer"),
hypotheticals ("if", "might"), past tense ("was", "used to") and other people's facts
("Sam's manager") yield no fact. `eval/run_eval.py` measures how often each one is right.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Protocol

Fact = tuple[str | None, str | None, str | None]
NO_FACT: Fact = (None, None, None)
SUBJECT = "user"  # facts are about the person talking to the assistant

# The fact keys the memory service consolidates on. Descriptions double as the LLM's
# instructions, so they say what belongs in each slot.
FACT_SCHEMA: dict[str, str] = {
    "name": "the user's own full name",
    "employee_id": "the user's employee ID",
    "home_city": "the city the user lives in",
    "hire_date": "the date the user was hired (YYYY-MM-DD)",
    "timezone": "the user's time zone",
    "current_manager": "the person the user currently reports to",
    "current_project": "the project the user is currently working on",
    "job_title": "the user's current job title",
    "office_location": "where the user's office is (or that they are fully remote)",
    "primary_metric": "the main metric the user reports on",
    "preferred_ide": "the editor / IDE the user currently uses",
    "sprint_goal": "the user's goal for the current sprint",
}


class Extractor(Protocol):
    def extract(self, record: dict) -> Fact: ...


class StructuredExtractor:
    """Passthrough: the interaction record already carries the structured triple."""

    def extract(self, record: dict) -> Fact:
        return record.get("subject"), record.get("attribute"), record.get("value")


# --------------------------------------------------------------------------- #
# Rule-based extraction (deterministic, offline)
# --------------------------------------------------------------------------- #
# Value grammars. Keyword prefixes are case-insensitive; values are case-sensitive, so
# a name has to look like a name ("Bob Tran") and stops at the first lowercase word.
_NAME = r"[A-Z][a-z]+(?: [A-Z][a-z'-]+){1,2}"
_PROJECT = r"Project [A-Z][a-z]+"
_TITLE = r"(?:[A-Z][a-z]+ ){0,2}[A-Z][a-z]+"
_CITY = r"[A-Z][a-z]+(?: [A-Z][a-z]+)?"
_DATE = r"\d{4}-\d{2}-\d{2}"
_EMPLOYEE_ID = r"E-\d{4,6}"
_TIMEZONE = r"[A-Z][a-z]+ Time"
_IDE = r"[A-Z][A-Za-z+#]*(?: [A-Z][A-Za-z+#]*)?"
_FREE = r"[^.;!?]+"


@dataclass(frozen=True)
class _Rule:
    attribute: str
    pattern: re.Pattern[str]
    requires: re.Pattern[str] | None = None  # a keyword the whole statement must contain


def _rule(attribute: str, prefix: str, value: str, requires: str | None = None) -> _Rule:
    return _Rule(attribute, re.compile(rf"(?i:{prefix})(?P<v>{value})"),
                 re.compile(requires, re.IGNORECASE) if requires else None)


_MANAGER_NOUN = r"(?:manager|boss|line manager|supervisor)"
_RULES: tuple[_Rule, ...] = (
    _rule("current_manager", rf"\bmy (?:new |current )?{_MANAGER_NOUN} is (?:now )?", _NAME),
    _rule("current_manager", r"\bI (?:now )?report to ", _NAME),
    _rule("current_manager", r"\bI(?:'m| am) (?:now )?reporting to ", _NAME),
    _rule("current_project", r"\bI(?:'m| am) (?:currently |now )?working on ", _PROJECT),
    _rule("current_project", r"\bI (?:currently |now )?work on ", _PROJECT),
    _rule("current_project", r"\bI(?:'ve| have) moved (?:over )?to ", _PROJECT),
    _rule("current_project", r"\bI(?:'m| am) (?:currently |now )?on ", _PROJECT),
    _rule("current_project", r"\bmy (?:current |new )?project is (?:now )?", _PROJECT),
    _rule("job_title", r"\bmy (?:job )?title (?:is (?:now )?|changed to )", _TITLE),
    _rule("job_title", r"\bmy (?:current |new )?role is (?:now )?", _TITLE),
    _rule("job_title", r"\bI was promoted to ", _TITLE),
    _rule("office_location", r"\bmy office (?:is (?:now )?|(?:has )?moved to )", _FREE),
    _rule("office_location", r"\bI(?:'m| am) (?:now )?", r"fully remote"),
    _rule("primary_metric",
          r"\b(?:the |my |our )?(?:headline |primary |main |key )?(?:metric|kpi) "
          r"I (?:report on|track) is ", _FREE),
    _rule("primary_metric", r"\bswitched the (?:headline |primary |main |key )?(?:metric|kpi) to ",
          _FREE),
    _rule("primary_metric", r"\bmy (?:headline |primary |main |key )?(?:metric|kpi) is (?:now )?",
          _FREE),
    _rule("preferred_ide", r"\bI use ", rf"{_IDE}(?= as my (?:editor|ide)\b)"),
    _rule("preferred_ide", r"\bmy (?:preferred |main )?(?:editor|ide) is (?:now )?", _IDE),
    _rule("preferred_ide", r"\bI(?:'m| am) (?:now )?on ", _IDE, requires=r"\b(?:editors?|ides?)\b"),
    _rule("preferred_ide", r"\bI(?:'ve| have) switched to ", _IDE,
          requires=r"\b(?:editors?|ides?)\b"),
    _rule("sprint_goal", r"\b(?:my |the )?goal is (?:now )?to ", _FREE, requires=r"\bsprint\b"),
    _rule("name", r"\bmy name is ", _NAME),
    _rule("employee_id", r"\bmy employee (?:id|number|no\.?) is ", _EMPLOYEE_ID),
    _rule("home_city", r"\bI live in ", _CITY),
    _rule("hire_date", r"\bI was hired on ", _DATE),
    _rule("hire_date", r"\bI (?:started|joined) (?:here |the company )?on ", _DATE),
    _rule("timezone", r"\bI(?:'m| am) in ", _TIMEZONE),
    _rule("timezone", r"\bmy time ?zone is ", _TIMEZONE),
    _rule("timezone", r"\bI work in ", _TIMEZONE),
)

# A clause carrying any of these asserts no CURRENT first-person fact.
_NEGATION = re.compile(r"\b(?:not|no|never|no longer|nor)\b|n't\b", re.IGNORECASE)
_HYPOTHETICAL = re.compile(
    r"\b(?:if|would|might|may|could|maybe|perhaps|will|someday|hopefully|hope|wish|"
    r"going to|plan to|planning to|considering|thinking about)\b|'d\b|"
    r"\bnext (?:week|month|quarter|year)\b", re.IGNORECASE)
_PAST = re.compile(r"\b(?:was(?! (?:hired|promoted)\b)|were|used to|previously|formerly|before)\b",
                   re.IGNORECASE)
# Clause boundaries: ";", or ", [so] " before a new first-person clause.
_CLAUSE_SPLIT = re.compile(r";\s*|,\s+(?=(?:so\s+|and\s+)?(?:my|i|i'm|i've)\b)", re.IGNORECASE)
_TRAILING_FILLER = re.compile(r"\s+(?:now|these days|for now|again|after the reorg)\s*$",
                              re.IGNORECASE)


class RuleExtractor:
    """Deterministic, schema-locked extraction from the statement text alone.

    Each clause is matched against first-person patterns for the schema's attributes;
    a clause that is negated, hypothetical or in the past tense asserts nothing current
    and is skipped. More than one distinct fact in a statement, or a question, abstains."""

    def extract(self, record: dict) -> Fact:
        text = record.get("text") or ""
        if text.rstrip().endswith("?"):
            return NO_FACT
        found: set[tuple[str, str]] = set()
        for clause in _CLAUSE_SPLIT.split(text):
            if _NEGATION.search(clause) or _HYPOTHETICAL.search(clause) or _PAST.search(clause):
                continue
            for rule in _RULES:
                if rule.requires is not None and not rule.requires.search(text):
                    continue
                m = rule.pattern.search(clause)
                if m is None:
                    continue
                value = _TRAILING_FILLER.sub("", m.group("v")).strip(" ,")
                if value:
                    found.add((rule.attribute, value))
        if len(found) != 1:
            return NO_FACT  # nothing, or two different facts we can't tell apart
        attribute, value = found.pop()
        return SUBJECT, attribute, value


# --------------------------------------------------------------------------- #
# LLM extraction (Claude, schema-constrained)
# --------------------------------------------------------------------------- #
NONE_ATTRIBUTE = "none"
_ABSTAIN_RULES = (
    'Answer attribute "none" for questions, negations ("no longer", "not"), hypotheticals '
    "or plans, and values that are no longer true.\n"
    "Copy the value exactly as written in the message."
)


def _output_schema(schema: dict[str, str] | None) -> dict[str, Any]:
    if schema is not None:
        attribute: dict[str, Any] = {"type": "string", "enum": [*schema, NONE_ATTRIBUTE]}
        subject: dict[str, Any] = {"type": "string", "enum": [SUBJECT, "other"]}
    else:
        attribute, subject = {"type": "string"}, {"type": "string"}
    return {
        "type": "object",
        "properties": {"attribute": attribute, "subject": subject, "value": {"type": "string"}},
        "required": ["attribute", "subject", "value"],
        "additionalProperties": False,
    }


def _system_prompt(schema: dict[str, str] | None) -> str:
    if schema is not None:
        return (
            "You extract at most one fact from one message a user sent to their work "
            "assistant. Only extract a fact the user states as currently true about "
            'themselves; for a fact about someone else answer subject "other".\n'
            + _ABSTAIN_RULES + "\nAttributes:\n"
            + "\n".join(f"- {k}: {v}" for k, v in schema.items())
        )
    return (
        "You extract at most one durable fact about a person from one message. subject is "
        "that person's name exactly as written in the message; attribute is a short "
        "snake_case name for what the fact describes.\n" + _ABSTAIN_RULES
    )


_OPEN_ATTRIBUTE = re.compile(r"^[a-z][a-z0-9_]{1,40}$")
FALLBACK_BETA = "server-side-fallback-2026-07-01"
DEFAULT_MODEL = "claude-opus-5-5"


def _normalize(s: str) -> str:
    return " ".join(s.split()).casefold()


class LLMExtractor:
    """Claude-backed extraction, constrained by a JSON schema and validated.

    By default the attribute must come from `schema` (FACT_SCHEMA) and the fact must be
    about the user, so supersession keys stay canonical. `schema=None` is an open-
    vocabulary mode for corpora whose facts no fixed schema covers (the LoCoMo runner):
    any snake_case attribute, with the subject a name that appears in the statement.
    Either way the value must appear verbatim in the statement, so the model cannot
    invent one. A refusal, a truncated or malformed response, or a transient API failure
    (rate limit, overload, connection) yields no fact; a configuration error (bad request,
    auth) is raised, not silently swallowed.

    Pass `client` to inject one (tests use a fake); by default the Anthropic client is
    built lazily from the environment's credentials."""

    def __init__(self, model: str = DEFAULT_MODEL, client: Any | None = None,
                 max_tokens: int = 2048, schema: dict[str, str] | None = FACT_SCHEMA):
        self.model = model
        self.max_tokens = max_tokens
        self.schema = schema
        self._client = client

    def _get_client(self) -> Any:
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic()
        return self._client

    def _request(self, text: str) -> Any:
        return self._get_client().beta.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            betas=[FALLBACK_BETA],
            fallbacks="default",  # a declined request re-runs on Anthropic's recommended model
            system=_system_prompt(self.schema),
            messages=[{"role": "user", "content": text}],
            output_config={"effort": "low",
                           "format": {"type": "json_schema",
                                      "schema": _output_schema(self.schema)}},
        )

    def extract(self, record: dict) -> Fact:
        text = record.get("text") or ""
        if not text.strip():
            return NO_FACT
        try:
            response = self._request(text)
        except Exception as exc:
            if _is_transient(exc):
                return NO_FACT
            raise
        if getattr(response, "stop_reason", None) in ("refusal", "max_tokens"):
            return NO_FACT
        body = next((b.text for b in response.content if getattr(b, "type", None) == "text"), "")
        try:
            data = json.loads(body)
        except (json.JSONDecodeError, TypeError):
            return NO_FACT
        return self._validate(data, text)

    def _validate(self, data: Any, text: str) -> Fact:
        if not isinstance(data, dict):
            return NO_FACT
        attribute, subject, value = data.get("attribute"), data.get("subject"), data.get("value")
        if not all(isinstance(x, str) for x in (attribute, subject, value)):
            return NO_FACT
        if self.schema is not None:
            if attribute not in self.schema or subject != SUBJECT:
                return NO_FACT
        else:
            subject = subject.strip()
            if (attribute == NONE_ATTRIBUTE or not _OPEN_ATTRIBUTE.match(attribute)
                    or not subject or _normalize(subject) not in _normalize(text)):
                return NO_FACT  # an unusable slot, or a subject the statement never names
        value = value.strip().rstrip(".")
        if not value or _normalize(value) not in _normalize(text):
            return NO_FACT  # not grounded in the statement: the model made it up
        return subject, attribute, value


def _is_transient(exc: Exception) -> bool:
    """Rate limits, overload / 5xx and connection failures: skip this statement."""
    try:
        import anthropic
    except ImportError:  # pragma: no cover - the llm extra is not installed
        return False
    if isinstance(exc, (anthropic.RateLimitError, anthropic.APIConnectionError)):
        return True
    return isinstance(exc, anthropic.APIStatusError) and exc.status_code >= 500
