"""Memory-policy ablation.

Each rung adds one capability, holding the corpus fixed:

  flat-vector            dump every statement in a vector store, cosine top-k
  +recency               blend recency + importance into the score
  +consolidation         distil episodic -> semantic, dedupe repeats
  +supersession (full)   keep only the latest value per fact key; suppress stale

Scored on the gold query set ("what is my current <X>?"):

  acc@1      top-1 retrieved memory holds the CURRENT value          (higher better)
  found@5    current value appears anywhere in top-5                 (higher better)
  stale@5    a superseded/old value appears in top-5                 (LOWER  better)
  ctx_tokens mean tokens across the top-5 retrieved memories (cost)  (lower  better)

Then extraction, the step the ablation above takes as given. The ablation feeds
the service the structured triples the generator emitted; the "from text" row instead
extracts every fact from the statement text (RuleExtractor) and runs the same policy.
Extraction itself is scored on the corpus and on a held-out set of statements written
in phrasings the generator never uses, including ones that must yield NO fact
(negation, hypotheticals, questions, other people's facts, past values):

  precision  extracted facts that are exactly right (attribute + value)  (higher better)
  recall     facts that should be extracted and are                      (higher better)
  false      statements that assert no current fact but yielded one      (LOWER  better)

  python eval/run_eval.py                    # full eval (deterministic given the seed)
  python eval/run_eval.py --llm              # + the Claude extractor (needs credentials)
  python eval/run_eval.py --embedder hashing # offline, no model download; prints only
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent_memory.config import MemoryPolicy  # noqa: E402
from agent_memory.extractor import Extractor, LLMExtractor, RuleExtractor  # noqa: E402
from agent_memory.service import MemoryService, load_interactions, load_queries  # noqa: E402

K = 5
EVAL_DIR = Path(__file__).resolve().parent
CASES_PATH = EVAL_DIR / "extraction_cases.jsonl"

POLICIES = [
    ("flat-vector", MemoryPolicy(use_recency=False, use_consolidation=False,
                                 use_supersession=False)),
    ("+recency", MemoryPolicy(use_recency=True, use_consolidation=False,
                              use_supersession=False)),
    ("+consolidation", MemoryPolicy(use_recency=True, use_consolidation=True,
                                    use_supersession=False)),
    ("+supersession (full)", MemoryPolicy(use_recency=True, use_consolidation=True,
                                          use_supersession=True)),
]


def evaluate(records: list[dict], queries: list[dict], policy: MemoryPolicy,
             extractor: Extractor | None = None, embedder=None) -> dict:
    now = max(r["ts_day"] for r in records) + 1
    svc = MemoryService(policy=policy, embedder=embedder)
    svc.ingest_records(records, extractor)
    svc.consolidate(now)
    svc.forget(now)

    agg: dict[str, list[float]] = defaultdict(list)
    by_cat: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for q in queries:
        hits = svc.recall(q["query"], k=K, now_day=now)
        values = [h.value for h in hits]
        gold, priors = q["gold_value"], set(q["prior_values"])
        acc1 = 1.0 if (values and values[0] == gold) else 0.0
        found = 1.0 if gold in values else 0.0
        stale = 1.0 if any(v in priors for v in values if v is not None) else 0.0
        ctx = float(sum(h.token_estimate() for h in hits))
        for m, v in (("acc@1", acc1), ("found@5", found),
                     ("stale@5", stale), ("ctx_tokens", ctx)):
            agg[m].append(v)
            by_cat[q["category"]][m].append(v)

    def mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    return {
        "overall": {m: round(mean(v), 3) for m, v in agg.items()},
        "by_category": {c: {m: round(mean(v), 3) for m, v in d.items()}
                        for c, d in by_cat.items()},
    }


def overall_table(results):
    rows = ["| memory policy | acc@1 ↑ | found@5 ↑ | stale@5 ↓ | ctx tokens ↓ |",
            "|---|---|---|---|---|"]
    for name, res in results:
        o = res["overall"]
        rows.append(f"| {name} | {o['acc@1']:.3f} | {o['found@5']:.3f} | "
                    f"{o['stale@5']:.3f} | {o['ctx_tokens']:.1f} |")
    return "\n".join(rows)


def changing_table(results):
    rows = ["| memory policy | acc@1 ↑ | stale@5 ↓ |", "|---|---|---|"]
    for name, res in results:
        c = res["by_category"].get("changing", {})
        rows.append(f"| {name} | {c.get('acc@1', 0):.3f} | {c.get('stale@5', 0):.3f} |")
    return "\n".join(rows)


# --------------------------------------------------------------------------- #
# Extraction scoring
# --------------------------------------------------------------------------- #
def load_cases(path: Path = CASES_PATH) -> list[dict]:
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def corpus_cases(records: list[dict]) -> list[dict]:
    """The corpus as extraction cases: each statement's labelled triple is the gold."""
    return [{"id": r["id"], "category": "fact" if r.get("attribute") else "noise",
             "text": r["text"], "attribute": r.get("attribute"), "value": r.get("value")}
            for r in records]


def _same_value(a: str | None, b: str | None) -> bool:
    return a is not None and b is not None and a.strip().casefold() == b.strip().casefold()


def score_extraction(extractor: Extractor, cases: list[dict]) -> dict:
    """Per-category and overall extraction counts, precision, recall and false facts."""
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for c in cases:
        _, attribute, value = extractor.extract(c)
        cat = counts[c["category"]]
        cat["n"] += 1
        if c["attribute"] is None:
            cat["false"] += attribute is not None
        elif attribute is None:
            cat["missed"] += 1
        elif attribute == c["attribute"] and _same_value(value, c["value"]):
            cat["correct"] += 1
        else:
            cat["wrong"] += 1
    total: dict[str, int] = defaultdict(int)
    for cat in counts.values():
        for k, v in cat.items():
            total[k] += v
    extracted = total["correct"] + total["wrong"] + total["false"]
    positives = total["correct"] + total["wrong"] + total["missed"]
    negatives = sum(cat["n"] for name, cat in counts.items()
                    if all(c["attribute"] is None for c in cases if c["category"] == name))
    return {
        "precision": round(total["correct"] / extracted, 3) if extracted else 1.0,
        "recall": round(total["correct"] / positives, 3) if positives else 1.0,
        "false": total["false"], "negatives": negatives,
        "by_category": {name: dict(cat) for name, cat in counts.items()},
    }


def extraction_table(rows: list[tuple[str, dict]]) -> str:
    out = ["| extractor · statements | precision ↑ | recall ↑ | false facts ↓ |",
           "|---|---|---|---|"]
    for name, sc in rows:
        out.append(f"| {name} | {sc['precision']:.3f} | {sc['recall']:.3f} | "
                   f"{sc['false']} / {sc['negatives']} |")
    return "\n".join(out)


def category_table(name: str, sc: dict) -> str:
    out = [f"| {name} · category | n | correct | missed | wrong | false facts |",
           "|---|---|---|---|---|---|"]
    for cat, c in sc["by_category"].items():
        out.append(f"| {cat} | {c.get('n', 0)} | {c.get('correct', 0)} | {c.get('missed', 0)} "
                   f"| {c.get('wrong', 0)} | {c.get('false', 0)} |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--embedder", choices=["minilm", "hashing"], default="minilm",
                    help="hashing: no model download; results are printed, not written")
    ap.add_argument("--llm", action="store_true",
                    help="also score the Claude extractor (needs Anthropic credentials)")
    args = ap.parse_args()
    embedder = None
    if args.embedder == "hashing":
        from agent_memory.embeddings import HashingEmbedder
        embedder = HashingEmbedder()

    records = load_interactions()
    queries = load_queries()
    results = [(name, evaluate(records, queries, pol, embedder=embedder))
               for name, pol in POLICIES]
    # The same full policy, but every fact extracted from the statement text.
    full_policy = POLICIES[-1][1]
    results.append(("+supersession, facts extracted from text",
                    evaluate(records, queries, full_policy, RuleExtractor(), embedder)))
    for name, res in results:
        o = res["overall"]
        print(f"  [{name}] acc@1={o['acc@1']:.3f} stale@5={o['stale@5']:.3f} "
              f"ctx={o['ctx_tokens']:.1f}", file=sys.stderr)

    cases = load_cases()
    extractors: list[tuple[str, Extractor]] = [("rule", RuleExtractor())]
    if args.llm:
        extractors.append(("llm", LLMExtractor()))
    extraction: dict[str, dict] = {}
    rows: list[tuple[str, dict]] = []
    for ex_name, ex in extractors:
        for set_name, set_cases in (("corpus", corpus_cases(records)), ("held-out", cases)):
            sc = score_extraction(ex, set_cases)
            extraction[f"{ex_name} · {set_name}"] = sc
            rows.append((f"{ex_name} · {set_name}", sc))
            print(f"  [extract {ex_name} · {set_name}] precision={sc['precision']:.3f} "
                  f"recall={sc['recall']:.3f} false={sc['false']}", file=sys.stderr)

    n_change = sum(q["category"] == "changing" for q in queries)
    md = [
        "# Memory-policy evaluation\n",
        f"{len(records)} interactions · {len(queries)} gold queries "
        f"({n_change} changing facts, {len(queries) - n_change} stable). "
        "↑ higher is better, ↓ lower is better.\n",
        "## Overall\n", overall_table(results), "",
        "## Changing facts only (the supersession cases)\n", changing_table(results), "",
        "## Extraction\n",
        f"Corpus = the {len(records)} generated statements (the rule extractor's patterns "
        "cover the generator's phrasings, so this row checks wiring, not generalization). "
        f"Held-out = {len(cases)} statements in phrasings the generator never uses, "
        f"{sum(c['attribute'] is None for c in cases)} of which must yield no fact.\n",
        extraction_table(rows), "",
        *[category_table(name, sc) + "\n" for name, sc in rows if name.endswith("held-out")],
        "_Generated by `eval/run_eval.py`; deterministic given the seeded sessions._",
    ]
    print("\n" + overall_table(results) + "\n\n" + changing_table(results)
          + "\n\n" + extraction_table(rows))
    if embedder is not None:
        print("\n(--embedder hashing: results printed only; results.md is MiniLM's)",
              file=sys.stderr)
        return
    out = EVAL_DIR / "results.md"
    out.write_text("\n".join(md) + "\n", encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(
        {**{n: r for n, r in results}, "extraction": extraction}, indent=2), encoding="utf-8")
    print(f"\nwrote {out} and results.json", file=sys.stderr)


if __name__ == "__main__":
    main()
