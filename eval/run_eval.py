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
Extraction itself is scored on three sets, none of them an independent held-out set:

  corpus       the generator's own statements (checks wiring)
  dev          paraphrases and must-abstain statements, written alongside the rules
  adversarial  the probes two hostile reviewers used to break an earlier version

Definitions live in agent_memory/extraction_eval.py:

  precision    of the facts extracted, the share with the right subject, slot and value (↑)
  recall       of the facts stated, the share extracted right                        (↑)
  false facts  statements that assert no current fact but yielded one               (↓)

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
from agent_memory.extraction_eval import (  # noqa: E402
    adversarial_cases,
    corpus_cases,
    load_cases,
    score_extraction,
)
from agent_memory.extractor import Extractor, LLMExtractor, RuleExtractor  # noqa: E402
from agent_memory.service import MemoryService, load_interactions, load_queries  # noqa: E402

K = 5
EVAL_DIR = Path(__file__).resolve().parent

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
# Extraction tables
# --------------------------------------------------------------------------- #
def _rate(x: float | None) -> str:
    return "—" if x is None else f"{x:.3f}"  # None: nothing to divide by (see extraction_eval)


def extraction_table(rows: list[tuple[str, dict]]) -> str:
    out = ["| extractor · statements | precision ↑ | recall ↑ | wrong values ↓ "
           "| false facts ↓ |", "|---|---|---|---|---|"]
    for name, sc in rows:
        out.append(f"| {name} | {_rate(sc['precision'])} | {_rate(sc['recall'])} | "
                   f"{sc['wrong']} | {sc['false']} / {sc['negatives']} |")
    return "\n".join(out)


def category_table(name: str, sc: dict) -> str:
    out = [f"| {name} · category | correct | wrong | missed | false facts | quiet |",
           "|---|---|---|---|---|---|"]
    for cat, c in sc["by_category"].items():
        out.append(f"| {cat} | {c['correct']} | {c['wrong']} | {c['missed']} | {c['false']} "
                   f"| {c['quiet']} |")
    return "\n".join(out)


def write_results(results: list[tuple[str, dict]], rows: list[tuple[str, dict]],
                  extraction: dict[str, dict], records: list[dict], queries: list[dict],
                  sets: dict[str, list[dict]]) -> Path:
    """Write eval/results.md and eval/results.json; returns the .md path."""
    n_change = sum(q["category"] == "changing" for q in queries)
    md = [
        "# Memory-policy evaluation\n",
        f"{len(records)} interactions · {len(queries)} gold queries "
        f"({n_change} changing facts, {len(queries) - n_change} stable). "
        "↑ higher is better, ↓ lower is better.\n",
        "## Overall\n", overall_table(results), "",
        "## Changing facts only (the supersession cases)\n", changing_table(results), "",
        "## Extraction\n",
        "None of these sets is independent of the rules, so none estimates accuracy on "
        f"real traffic. Corpus = the {len(records)} generated statements (checks wiring). "
        f"Dev = {len(sets['dev'])} paraphrases and must-abstain statements written alongside "
        f"the rules. Adversarial = {len(sets['adversarial'])} probes two hostile reviewers "
        "used to break an earlier version, with gold labels; the fixes were written against "
        "them, so this set shows those failures are closed, not that new ones are absent. "
        "A wrong value is the failure that matters: it retires a correct one.\n",
        extraction_table(rows), "",
        *[category_table(name, sc) + "\n" for name, sc in rows],
        "_Generated by `eval/run_eval.py`; deterministic given the seeded sessions._",
    ]
    out = EVAL_DIR / "results.md"
    out.write_text("\n".join(md) + "\n", encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(
        {**{n: r for n, r in results}, "extraction": extraction}, indent=2), encoding="utf-8")
    return out


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

    sets = {"corpus": corpus_cases(records), "dev": load_cases(),
            "adversarial": adversarial_cases()}
    extractors: list[tuple[str, Extractor]] = [("rule", RuleExtractor())]
    if args.llm:
        extractors.append(("llm", LLMExtractor()))
    extraction: dict[str, dict] = {}
    rows: list[tuple[str, dict]] = []
    for ex_name, ex in extractors:
        for set_name, set_cases in sets.items():
            sc = score_extraction(ex, set_cases)
            extraction[f"{ex_name} · {set_name}"] = sc
            rows.append((f"{ex_name} · {set_name}", sc))
            print(f"  [extract {ex_name} · {set_name}] precision={_rate(sc['precision'])} "
                  f"recall={_rate(sc['recall'])} wrong={sc['wrong']} false={sc['false']} "
                  f"unavailable={sc['unavailable']}", file=sys.stderr)

    print("\n" + overall_table(results) + "\n\n" + changing_table(results)
          + "\n\n" + extraction_table(rows))
    if embedder is not None:
        print("\n(--embedder hashing: results printed only; results.md is MiniLM's)",
              file=sys.stderr)
        return
    out = write_results(results, rows, extraction, records, queries, sets)
    print(f"\nwrote {out} and results.json", file=sys.stderr)


if __name__ == "__main__":
    main()
