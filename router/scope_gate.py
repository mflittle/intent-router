"""Stage 0 (scope gate): deterministic out-of-scope / non-actionable
detection. Runs first, before Stage 0a/0b/0c, on every segment.

Catches requests -- or request fragments -- that ask for something this
system was never built to do: a physical-world action ("get lunch",
"walk the dog"), or anything else outside the digital/informational
tasks the intent categories cover. Without this stage, an out-of-scope
segment falls to IntentCategory.OTHER in the classifier, which the tier
policy treats as a cautious default -> HIGH, frontier-tier,
human-in-the-loop. That's the worst possible outcome for a nonsensical
request: the most expensive tier gets spent on something no tier should
ever attempt. This stage exists to catch the obvious cases before any
model call happens at all, and to give the user a direct "not something
this can solve" notice instead.

Unlike Stage 0a/0b, this is a SUBSTRING match against a curated denylist,
not an exact-match allowlist -- an out-of-scope fragment is usually
embedded inside a longer, otherwise-legitimate sentence ("summarize this
brief, send it to the judge, and get lunch"), not the whole segment.
That's a real trade-off: substring matching is inherently looser than
exact match, so the denylist stays small and deliberately curated
(obvious, unambiguous cases only) to keep false positives rare. A false
negative here just falls through to the classifier -- no worse than
today's OTHER -> cautious-HIGH behavior. A false positive would reject
something legitimate, which is the failure mode to guard against, so
entries are added conservatively, the same discipline as Stage 0a/0b.

Genuinely ambiguous requests are NOT this stage's job -- they should
still reach the classifier and the existing OTHER -> cautious-HIGH
fallback, which is the right backstop for "maybe legitimate, unclear."
This stage is only for the unambiguous cases.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ScopeGateMatch:
    matched_phrase: str
    reason: str


_DENYLIST: dict[str, str] = {}


def _normalize(text: str) -> str:
    text = text.lower().strip()
    return re.sub(r"\s+", " ", text)


def register_out_of_scope_phrase(
    phrase: str,
    reason: str = "physical-world action, outside this system's scope",
) -> None:
    """Register a phrase (normalized, matched as a substring) that marks
    any segment containing it as out of scope.
    """
    _DENYLIST[_normalize(phrase)] = reason


def match_scope_gate(segment_text: str) -> Optional[ScopeGateMatch]:
    normalized = _normalize(segment_text)
    for phrase, reason in _DENYLIST.items():
        if phrase in normalized:
            return ScopeGateMatch(matched_phrase=phrase, reason=reason)
    return None


def registered_phrases() -> list[str]:
    return sorted(_DENYLIST.keys())


# Starter denylist: unambiguous physical-world / non-actionable requests.
# Illustrative and deliberately small -- curate and extend it the same
# way the template and fact-lookup registries grow, only as real
# recurring cases justify adding an entry, never broadened speculatively.
for _phrase in [
    "get lunch",
    "get me lunch",
    "grab lunch",
    "walk the dog",
    "pick up my kids",
    "pick up the kids",
    "drive to",
    "call 911",
    "make me coffee",
    "walk to the store",
]:
    register_out_of_scope_phrase(_phrase)
