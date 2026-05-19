"""Memory-evolution strategies (Section 4.2 / Fig 4).

The evolution function F_M takes a session transcript and produces the items
to add to long-term memory. We support three strategies:

- raw_history     : verbatim concatenation
- verbal_reflection: one extra LLM call producing a free-form reflection
- refined_experience: one extra LLM call producing 1-5 'If X, then Y' rules

`evolve(strategy, transcript)` returns a list[str] of memory items.
"""

from __future__ import annotations

import json
from typing import Literal

from deepseek_client import call_deepseek


EvolutionStrategy = Literal["raw_history", "verbal_reflection", "refined_experience"]


def evolve(strategy: EvolutionStrategy, transcript_str: str) -> list[str]:
    if strategy == "raw_history":
        return [transcript_str.strip()] if transcript_str.strip() else []
    if strategy == "verbal_reflection":
        return _verbal_reflection(transcript_str)
    if strategy == "refined_experience":
        return _refined_experience(transcript_str)
    raise ValueError(f"unknown evolution strategy: {strategy}")


def _verbal_reflection(transcript: str) -> list[str]:
    prompt = (
        "Below is a session transcript between a web agent and its tools. "
        "Write a SHORT reflection (3-6 sentences) about what happened and "
        "what should be remembered for future sessions. Do not invent facts; "
        "summarise only what appears in the transcript.\n\n"
        "TRANSCRIPT:\n" + transcript
    )
    try:
        content, _ = call_deepseek(
            prompt,
            system="You write concise reflective notes about agent sessions.",
            temperature=0.3,
            max_tokens=400,
        )
    except Exception as e:
        return [f"[reflection failed: {e}]"]
    return [content.strip()]


def _refined_experience(transcript: str) -> list[str]:
    prompt = (
        "Below is a session transcript between a web agent and its tools. "
        "Extract at most 5 REUSABLE rules in the form 'If X, then Y.' Output "
        "STRICT JSON: {\"rules\": [\"...\", \"...\"]}. Do not include other "
        "text. Rules should be procedural and apply to future sessions.\n\n"
        "TRANSCRIPT:\n" + transcript
    )
    try:
        content, _ = call_deepseek(
            prompt,
            system="You distil procedural rules from agent transcripts.",
            temperature=0.2,
            max_tokens=400,
            json_mode=True,
        )
    except Exception as e:
        return [f"[refinement failed: {e}]"]
    try:
        data = json.loads(content)
        rules = data.get("rules") or []
        rules = [str(r).strip() for r in rules if str(r).strip()]
        return rules[:5]
    except Exception:
        # Fall back: split lines
        return [ln.strip("-* \t") for ln in content.splitlines() if ln.strip()][:5]
