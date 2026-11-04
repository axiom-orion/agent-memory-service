"""Scoring for fact extraction: how often an extractor is right, wrong, or silent.

Each case is a statement plus its gold fact, or no fact (`attribute` None) when the
statement asserts no current first-person fact. Every case lands in exactly one bucket:

  correct      gold fact, extracted with the same subject, attribute and value
  wrong        gold fact, extracted with another subject, attribute or value
  missed       gold fact, nothing extracted
  false        no gold fact, but one was extracted (a "false fact")
  quiet        no gold fact, nothing extracted (the right answer)
  unavailable  the extractor could not run (an outage); left out of both rates

  precision = correct / (correct + wrong + false)    of the facts extracted, how many are right
  recall    = correct / (correct + wrong + missed)   of the facts stated, how many came out right

Precision is None when nothing was extracted at all: an extractor that always abstains
has no precision, not a perfect one. The gold subject defaults to SUBJECT ("user").
Values compare case-insensitively after trimming whitespace ("Bob Tran" == "bob tran ").
Both a wrong and a false fact land on a supersession key and can retire a correct value,
which is why precision -- not recall -- is the number that must stay at 1.000.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from .config import settings
from .extractor import SUBJECT, ExtractionUnavailable, Extractor

FACT_CATEGORY = "fact"  # corpus statements that carry a labelled fact
NOISE_CATEGORY = "noise"  # corpus statements that carry none
OUTCOMES = ("correct", "wrong", "missed", "false", "quiet", "unavailable")


def load_cases(path: Path | None = None) -> list[dict]:
    """An extraction case set: eval/extraction_dev.jsonl (the dev set) by default."""
    path = path or settings.extraction_dev_path
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def adversarial_cases() -> list[dict]:
    """The reviewers' probes (eval/extraction_adversarial.jsonl), with gold labels."""
    return load_cases(settings.extraction_adversarial_path)


def corpus_cases(records: list[dict]) -> list[dict]:
    """The generated corpus as extraction cases: each record's own labels are the gold."""
    return [{"id": r["id"], "category": FACT_CATEGORY if r.get("attribute") else NOISE_CATEGORY,
             "text": r["text"], "subject": r.get("subject"),
             "attribute": r.get("attribute"), "value": r.get("value")}
            for r in records]


def same_value(a: str | None, b: str | None) -> bool:
    return a is not None and b is not None and a.strip().casefold() == b.strip().casefold()


def outcome(extractor: Extractor, case: dict) -> str:
    """Which bucket (see the module docstring) one case falls in."""
    try:
        subject, attribute, value = extractor.extract(case)
    except ExtractionUnavailable:
        return "unavailable"
    if case["attribute"] is None:
        return "quiet" if attribute is None else "false"
    if attribute is None:
        return "missed"
    if (subject == (case.get("subject") or SUBJECT) and attribute == case["attribute"]
            and same_value(value, case["value"])):
        return "correct"
    return "wrong"


def score_extraction(extractor: Extractor, cases: list[dict]) -> dict:
    """Outcome counts per category and overall, with precision and recall."""
    by_category: dict[str, dict[str, int]] = defaultdict(lambda: dict.fromkeys(OUTCOMES, 0))
    for case in cases:
        by_category[case["category"]][outcome(extractor, case)] += 1
    total = {k: sum(cat[k] for cat in by_category.values()) for k in OUTCOMES}
    extracted = total["correct"] + total["wrong"] + total["false"]
    stated = total["correct"] + total["wrong"] + total["missed"]
    return {
        "precision": round(total["correct"] / extracted, 3) if extracted else None,
        "recall": round(total["correct"] / stated, 3) if stated else None,
        **total,
        "negatives": total["false"] + total["quiet"],
        "by_category": {name: dict(cat) for name, cat in by_category.items()},
    }
