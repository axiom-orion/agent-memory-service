"""LoCoMo benchmark integration for the agent-memory-service.

LoCoMo (Maharana et al., 2024, "Evaluating Very Long-Term Conversational Memory
of LLM Agents") is the field-standard long-conversation memory benchmark, reported
on by Mem0, Zep, and Letta. This package evaluates the memory service against the
public LoCoMo-10 set in two modes:

  retrieval  -- recall@k / MRR of the gold *evidence* turns (deterministic, no LLM)
  qa         -- end-to-end answer F1 using LoCoMo's official token-F1 metric
                (answer generation requires an Anthropic API key; scoring is
                 deterministic and matches LoCoMo's evaluation.py exactly)
"""
from .loader import LocomoSample, load_locomo
from .metrics import locomo_f1, recall_at_k, reciprocal_rank

__all__ = ["LocomoSample", "load_locomo", "locomo_f1", "recall_at_k", "reciprocal_rank"]
