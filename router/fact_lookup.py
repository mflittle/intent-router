"""Stage 0b: deterministic fact lookup.

The sibling of Stage 0a (template_prefilter), for a different flavor of
"no model needed": a request that's a known factual question with a
known, registrable answer -- "who is the compliance officer", "what's
today's date" -- rather than a transformation to perform.

Same allowlist discipline as Stage 0a, and for the same reason: an exact
(normalized) match against a registry, never a free-text heuristic
guessing "this looks like a simple factual question". A guess here could
silently answer something that actually needed judgment or an up-to-date
source; an allowlist can't drift like that. Falls through to Stage 1
(the classifier) on anything not registered -- the default is always
"let the classifier see it", never "assume it's a known fact".

Starts empty. Grows only as real, recurring factual questions justify a
registered entry.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass(frozen=True)
class FactMatch:
    question: str
    answer: str


_FACT_REGISTRY: dict[str, Callable[[], str]] = {}


def _normalize(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[?.!]+$", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def register_fact(question: str, aliases: Optional[list[str]] = None):
    """Decorator to register a Stage 0b fact handler under `question` and
    any `aliases` (all normalized the same way before matching).
    """
    def _wrap(fn: Callable[[], str]):
        for phrasing in [question, *(aliases or [])]:
            _FACT_REGISTRY[_normalize(phrasing)] = fn
        return fn
    return _wrap


def match_fact(request_text: str) -> Optional[FactMatch]:
    """Return a FactMatch if `request_text` exactly matches a registered
    (normalized) question or alias, else None (falls through to Stage 1).
    """
    key = _normalize(request_text)
    handler = _FACT_REGISTRY.get(key)
    if handler is None:
        return None
    return FactMatch(question=request_text, answer=handler())


def registered_questions() -> list[str]:
    return sorted(_FACT_REGISTRY.keys())
