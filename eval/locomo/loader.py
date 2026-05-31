"""Load the public LoCoMo-10 dataset (not redistributed here; see `make locomo-data`)."""
from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

_SESSION = re.compile(r"session_(\d+)$")

CATEGORY_NAMES = {1: "multi-hop", 2: "temporal", 3: "open-domain",
                  4: "single-hop", 5: "adversarial"}


@dataclass(slots=True)
class Turn:
    dia_id: str
    speaker: str
    text: str
    session: int


@dataclass(slots=True)
class QA:
    question: str
    answer: str
    evidence: set[str]
    category: int


@dataclass(slots=True)
class LocomoSample:
    sample_id: str
    turns: list[Turn]
    qa: list[QA] = field(default_factory=list)


def _evidence(raw) -> set[str]:
    if isinstance(raw, str):
        try:
            raw = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            raw = []
    if not isinstance(raw, list):
        return set()
    return {e for e in raw if isinstance(e, str) and e.startswith("D")}


def load_locomo(path: str | Path) -> list[LocomoSample]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download LoCoMo-10 first: `make locomo-data` "
            "(clones snap-research/locomo). The dataset is not redistributed in this repo.")
    data = json.loads(path.read_text())
    samples: list[LocomoSample] = []
    for s in data:
        conv = s["conversation"]
        turns: list[Turn] = []
        session_keys = sorted(
            (k for k in conv if _SESSION.fullmatch(k)),
            key=lambda k: int(_SESSION.fullmatch(k).group(1)))
        for idx, sk in enumerate(session_keys):
            for t in conv[sk]:
                if "text" in t and "dia_id" in t:
                    turns.append(Turn(t["dia_id"], t.get("speaker", ""), t["text"], idx))
        qa = [QA(q["question"], str(q.get("answer", "")),
                 _evidence(q.get("evidence")), int(q["category"]))
              for q in s.get("qa", [])]
        samples.append(LocomoSample(s.get("sample_id", str(len(samples))), turns, qa))
    return samples
