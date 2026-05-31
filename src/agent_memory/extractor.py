"""Fact extraction: turn raw interaction text into (subject, attribute, value).

The eval uses StructuredExtractor (the upstream agent already emitted structured
fields, so this is a passthrough) to keep results deterministic. In production an
LLM populates the triple from free text; LLMExtractor sketches that path and is
not exercised by the eval or CI (no network/key required)."""
from __future__ import annotations

import json
import os


class StructuredExtractor:
    """Passthrough: the interaction record already carries the structured triple."""

    def extract(self, record: dict) -> tuple[str | None, str | None, str | None]:
        return record.get("subject"), record.get("attribute"), record.get("value")


class LLMExtractor:  # pragma: no cover - optional production path
    """Anthropic-backed extractor. Returns (subject, attribute, value) or all-None.

    Requires ANTHROPIC_API_KEY and the `anthropic` package; degrades to None so a
    caller can fall back to leaving the statement as a non-fact episodic memory."""

    SYSTEM = ("Extract a single (subject, attribute, value) fact from the user "
              "message if one is asserted. Reply with compact JSON "
              '{"subject":..,"attribute":..,"value":..} or {} if none.')

    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        self.model = model

    def extract(self, record: dict) -> tuple[str | None, str | None, str | None]:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            return None, None, None
        try:
            import anthropic
            client = anthropic.Anthropic()
            msg = client.messages.create(
                model=self.model, max_tokens=200,
                system=self.SYSTEM,
                messages=[{"role": "user", "content": record["text"]}])
            text = "".join(b.text for b in msg.content if b.type == "text")
            data = json.loads(text)
            return data.get("subject"), data.get("attribute"), data.get("value")
        except Exception:
            return None, None, None
