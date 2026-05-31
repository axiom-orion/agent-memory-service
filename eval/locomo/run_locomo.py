"""LoCoMo-10 evaluation runner.

  # deterministic, no API key — recall@k / MRR of gold evidence turns
  python eval/locomo/run_locomo.py --mode retrieval --data data/locomo/locomo10.json

  # end-to-end answer F1 (LoCoMo's official token-F1); needs ANTHROPIC_API_KEY
  python eval/locomo/run_locomo.py --mode qa --data data/locomo/locomo10.json
  #   add --extract to route turns through LLM fact extraction -> consolidation ->
  #   supersession (the knowledge-update / "current value" variant; many LLM calls)

Categories: 1 multi-hop, 2 temporal, 3 open-domain, 4 single-hop, 5 adversarial.
Retrieval recall is scored on categories with gold evidence (1/2/3/4); category 5
(adversarial / unanswerable) is excluded from recall by construction.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from agent_memory.config import MemoryPolicy  # noqa: E402
from agent_memory.service import MemoryService  # noqa: E402

from .adapter import ingest_sample  # noqa: E402
from .loader import CATEGORY_NAMES, LocomoSample, load_locomo  # noqa: E402
from .metrics import locomo_f1, recall_at_k, reciprocal_rank  # noqa: E402

K_VALUES = (1, 5, 10, 20)


def _now_day(sample: LocomoSample) -> int:
    return max((t.session for t in sample.turns), default=0) + 1


def run_retrieval(samples: list[LocomoSample]) -> dict:
    agg: dict[str, list[float]] = defaultdict(list)
    per_cat: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    n_scored = 0
    for sample in samples:
        svc = MemoryService(policy=MemoryPolicy(
            use_recency=False, use_consolidation=False, use_supersession=False))
        ingest_sample(svc, sample)
        now = _now_day(sample)
        for qa in sample.qa:
            if qa.category == 5 or not qa.evidence:
                continue
            hits = svc.recall(qa.question, k=max(K_VALUES), now_day=now)
            ranked = [h.id for h in hits]
            n_scored += 1
            for k in K_VALUES:
                r = recall_at_k(ranked, qa.evidence, k)
                if r is not None:
                    agg[f"recall@{k}"].append(r)
                    per_cat[qa.category][f"recall@{k}"].append(r)
            rr = reciprocal_rank(ranked, qa.evidence)
            agg["mrr"].append(rr)
            per_cat[qa.category]["mrr"].append(rr)

    def mean(xs):
        return round(sum(xs) / len(xs), 3) if xs else 0.0

    return {
        "mode": "retrieval",
        "n_scored": n_scored,
        "overall": {m: mean(v) for m, v in agg.items()},
        "by_category": {
            CATEGORY_NAMES[c]: {m: mean(v) for m, v in d.items()} | {"n": len(d["mrr"])}
            for c, d in sorted(per_cat.items())},
    }


def run_qa(samples: list[LocomoSample], extract: bool, k: int) -> dict:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("qa mode needs ANTHROPIC_API_KEY (answer generation). "
                         "Use --mode retrieval for the key-free metric.")
    import anthropic
    client = anthropic.Anthropic()
    model = os.environ.get("LOCOMO_QA_MODEL", "claude-sonnet-4-20250514")
    sys_prompt = (
        "Answer the question using ONLY the conversation memories provided. "
        "Be concise: reply with the shortest exact answer (a name, date, place, or "
        "short phrase). If the memories do not contain the answer, reply exactly "
        "'No information available'.")
    policy = MemoryPolicy() if extract else MemoryPolicy(
        use_recency=True, use_consolidation=False, use_supersession=False)

    per_cat: dict[int, list[float]] = defaultdict(list)
    agg: list[float] = []
    for sample in samples:
        svc = MemoryService(policy=policy)
        if extract:
            from agent_memory.extractor import LLMExtractor
            ex = LLMExtractor()
            for t in sample.turns:
                subj, attr, val = ex.extract({"text": f"{t.speaker}: {t.text}"})
                svc.remember(f"{t.speaker}: {t.text}", day=t.session,
                             subject=subj, attribute=attr, value=val)
            svc.consolidate(_now_day(sample))
        else:
            ingest_sample(svc, sample)
        now = _now_day(sample)
        for qa in sample.qa:
            if qa.category == 5:
                continue
            hits = svc.recall(qa.question, k=k, now_day=now)
            context = "\n".join(f"- {h.content}" for h in hits)
            msg = client.messages.create(
                model=model, max_tokens=100, system=sys_prompt,
                messages=[{"role": "user",
                           "content": f"Memories:\n{context}\n\nQuestion: {qa.question}"}])
            answer = "".join(b.text for b in msg.content if b.type == "text").strip()
            f1 = locomo_f1(answer, qa.answer)
            agg.append(f1)
            per_cat[qa.category].append(f1)

    def mean(xs):
        return round(sum(xs) / len(xs), 3) if xs else 0.0

    return {
        "mode": "qa", "metric": "locomo_token_f1", "extract": extract,
        "overall_f1": mean(agg), "n": len(agg),
        "by_category_f1": {CATEGORY_NAMES[c]: mean(v) for c, v in sorted(per_cat.items())},
    }


def render_retrieval_md(res: dict) -> str:
    o = res["overall"]
    rows = ["# LoCoMo-10 retrieval evaluation\n",
            f"Memory-service retriever (dense, `all-MiniLM-L6-v2`) vs. the gold "
            f"evidence turns of the public LoCoMo-10 set. {res['n_scored']} questions "
            "scored (categories 1/2/3/4 with gold evidence; category 5 adversarial "
            "excluded). Deterministic, no API key.\n",
            "## Overall\n",
            "| metric | value |", "|---|---|",
            f"| recall@1 | {o.get('recall@1', 0):.3f} |",
            f"| recall@5 | {o.get('recall@5', 0):.3f} |",
            f"| recall@10 | {o.get('recall@10', 0):.3f} |",
            f"| recall@20 | {o.get('recall@20', 0):.3f} |",
            f"| MRR (cutoff 10) | {o.get('mrr', 0):.3f} |", "",
            "## By question category\n",
            "| category | recall@5 | recall@10 | recall@20 | n |", "|---|---|---|---|---|"]
    for cat, d in res["by_category"].items():
        rows.append(f"| {cat} | {d.get('recall@5', 0):.3f} | {d.get('recall@10', 0):.3f} "
                    f"| {d.get('recall@20', 0):.3f} | {d['n']} |")
    rows += ["", "_Generated by `eval/locomo/run_locomo.py --mode retrieval`. "
             "Retrieval recall isolates the embedding/retrieval layer; the memory "
             "lifecycle (consolidation/supersession) is evaluated by `--mode qa "
             "--extract`._"]
    return "\n".join(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["retrieval", "qa"], default="retrieval")
    ap.add_argument("--data", default="data/locomo/locomo10.json")
    ap.add_argument("--extract", action="store_true",
                    help="(qa) LLM-extract facts -> consolidate -> supersession")
    ap.add_argument("--k", type=int, default=10, help="(qa) memories retrieved per question")
    ap.add_argument("--limit", type=int, default=0, help="cap samples (debug)")
    args = ap.parse_args()

    samples = load_locomo(args.data)
    if args.limit:
        samples = samples[: args.limit]
    print(f"LoCoMo: {len(samples)} conversations, "
          f"{sum(len(s.turns) for s in samples)} turns", file=sys.stderr)

    out_dir = Path(__file__).resolve().parent
    if args.mode == "retrieval":
        res = run_retrieval(samples)
        (out_dir / "results.md").write_text(render_retrieval_md(res) + "\n")
        (out_dir / "results.json").write_text(json.dumps(res, indent=2))
        print(render_retrieval_md(res))
    else:
        res = run_qa(samples, args.extract, args.k)
        (out_dir / "results_qa.json").write_text(json.dumps(res, indent=2))
        print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
