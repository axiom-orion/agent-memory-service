"""Deterministic generator for a synthetic assistant-interaction history.

Models the canonical agent-memory scenario: over many sessions a work assistant
is told facts about the user/project. Some facts are stable (name, home city);
some are *updated* over time (manager, current project, job title). The eval then
asks for the CURRENT value of each fact -- which is exactly where a naive
"append everything to a vector store" memory fails, because stale mentions are
just as retrievable as current ones.

Outputs (deterministic; seeded):
  data/sessions/interactions.jsonl  -- timestamped episodic statements + noise
  eval/queries.jsonl                -- gold queries (current value, prior values)

Run: python data/generate_sessions.py
"""
from __future__ import annotations

import json
import random
from pathlib import Path

SEED = 23
random.seed(SEED)

ROOT = Path(__file__).resolve().parents[1]
SESS = ROOT / "data" / "sessions"
EVAL = ROOT / "eval"

# Day-0 baseline; each statement carries an integer day offset as its timestamp.
SUBJECT = "user"

# --------------------------------------------------------------------------- #
# Facts. Stable facts have one value. Changing facts have an ordered list of
# (day, value) updates -- the last is the "current" truth at query time.
# --------------------------------------------------------------------------- #
STABLE = {
    "name": ("What is my name?", "Jordan Ellison"),
    "employee_id": ("What is my employee ID?", "E-44817"),
    "home_city": ("What city do I live in?", "Denver"),
    "hire_date": ("When was I hired?", "2019-06-03"),
    "timezone": ("What time zone am I in?", "Mountain Time"),
}

CHANGING = {
    "current_manager": ("Who is my current manager?", [
        (2, "Alice Reyes"), (40, "Bob Tran")]),
    "current_project": ("What project am I currently on?", [
        (3, "Project Atlas"), (28, "Project Borealis"), (62, "Project Cygnus")]),
    "job_title": ("What is my current job title?", [
        (1, "Senior Analyst"), (55, "Staff Analyst")]),
    "office_location": ("Where is my office?", [
        (4, "Floor 3, Building A"), (70, "fully remote")]),
    "primary_metric": ("What is the primary metric I report on?", [
        (6, "weekly active users"), (33, "net revenue retention")]),
    "preferred_ide": ("What is my preferred IDE?", [
        (2, "VS Code"), (48, "PyCharm")]),
    "sprint_goal": ("What is my current sprint goal?", [
        (10, "ship the ingestion API"), (24, "cut p99 latency below 200ms"),
        (38, "instrument retention dashboards"), (66, "harden the eval harness")]),
}

# Natural-language phrasings for statements (initial vs. update wording).
INIT_PHRASING = {
    "current_manager": "My manager is {v}.",
    "current_project": "I'm working on {v}.",
    "job_title": "My job title is {v}.",
    "office_location": "My office is {v}.",
    "primary_metric": "The metric I report on is {v}.",
    "preferred_ide": "I use {v} as my editor.",
    "sprint_goal": "This sprint my goal is to {v}.",
}
UPDATE_PHRASING = {
    "current_manager": "Heads up, my manager is now {v} after the reorg.",
    "current_project": "I've moved over to {v} now.",
    "job_title": "I was promoted; my title is now {v}.",
    "office_location": "Update: I'm now {v}.",
    "primary_metric": "We switched the headline metric to {v}.",
    "preferred_ide": "I've switched editors; I'm on {v} now.",
    "sprint_goal": "New sprint: the goal is now to {v}.",
}
STABLE_PHRASING = {
    "name": "My name is {v}.",
    "employee_id": "My employee ID is {v}.",
    "home_city": "I live in {v}.",
    "hire_date": "I was hired on {v}.",
    "timezone": "I'm in {v}.",
}

NOISE = [
    "Can you summarize that PDF on Q2 forecasts?",
    "The coffee machine on floor 3 is broken again.",
    "Remind me to email the vendor next week.",
    "What's a good regex for parsing ISO dates?",
    "Draft a thank-you note to the offsite organizers.",
    "I had three meetings back-to-back today, exhausting.",
    "Can you convert this CSV to a markdown table?",
    "What's the weather looking like for the team picnic?",
    "Help me brainstorm names for the new dashboard.",
    "Summarize the postmortem from last Friday's incident.",
]

# --------------------------------------------------------------------------- #
interactions: list[dict] = []
_c = 0


def add(day: int, text: str, attribute: str | None, value: str | None,
        importance: float) -> None:
    global _c
    _c += 1
    interactions.append(dict(
        id=f"M{_c:04d}", ts_day=day, text=text, type="episodic",
        subject=SUBJECT if attribute else None, attribute=attribute, value=value,
        importance=round(importance, 2),
        provenance=f"session:{day}:turn:{random.randint(1, 9)}"))


def build() -> None:
    # stable facts: stated once or twice early
    for attr, (_, val) in STABLE.items():
        add(random.randint(0, 5), STABLE_PHRASING[attr].format(v=val), attr, val, 0.7)
        if random.random() < 0.4:
            add(random.randint(6, 20), STABLE_PHRASING[attr].format(v=val),
                attr, val, 0.6)
    # changing facts: initial value (sometimes repeated), then updates
    for attr, (_, updates) in CHANGING.items():
        for i, (day, val) in enumerate(updates):
            phrasing = INIT_PHRASING if i == 0 else UPDATE_PHRASING
            add(day, phrasing[attr].format(v=val), attr, val, 0.8)
            # initial values often get repeated before being superseded
            if i == 0 and len(updates) > 1 and random.random() < 0.7:
                rep_day = random.randint(day + 1, updates[1][0] - 1) \
                    if updates[1][0] > day + 1 else day
                add(rep_day, INIT_PHRASING[attr].format(v=val), attr, val, 0.6)
    # noise events sprinkled across the timeline
    for txt in NOISE:
        add(random.randint(0, 75), txt, None, None, round(random.uniform(0.1, 0.3), 2))

    interactions.sort(key=lambda m: m["ts_day"])


def gold_queries() -> list[dict]:
    qs: list[dict] = []
    qc = 0
    for attr, (q, updates) in CHANGING.items():
        qc += 1
        current = updates[-1][1]
        priors = [v for _, v in updates[:-1]]
        qs.append(dict(id=f"Q{qc:03d}", query=q, subject=SUBJECT, attribute=attr,
                       gold_value=current, prior_values=priors, category="changing"))
    for attr, (q, val) in STABLE.items():
        qc += 1
        qs.append(dict(id=f"Q{qc:03d}", query=q, subject=SUBJECT, attribute=attr,
                       gold_value=val, prior_values=[], category="stable"))
    return qs


def main() -> None:
    build()
    queries = gold_queries()
    SESS.mkdir(parents=True, exist_ok=True)
    EVAL.mkdir(parents=True, exist_ok=True)
    with (SESS / "interactions.jsonl").open("w") as fh:
        for m in interactions:
            fh.write(json.dumps(m) + "\n")
    with (EVAL / "queries.jsonl").open("w") as fh:
        for q in queries:
            fh.write(json.dumps(q) + "\n")

    n_fact = sum(1 for m in interactions if m["attribute"])
    print(f"interactions      : {len(interactions)} "
          f"({n_fact} fact statements, {len(interactions) - n_fact} noise)")
    print(f"timeline (days)   : {interactions[0]['ts_day']}..{interactions[-1]['ts_day']}")
    print(f"gold queries      : {len(queries)} "
          f"({sum(q['category']=='changing' for q in queries)} changing, "
          f"{sum(q['category']=='stable' for q in queries)} stable)")
    upd = sum(len(u) - 1 for _, (_, u) in CHANGING.items())
    print(f"fact updates       : {upd} (the supersession cases)")


if __name__ == "__main__":
    main()
