"""Stage -1: deterministic request segmentation.

Splits a compound request ("summarize this filing and draft a note to
external counsel") into independent segments *before* any of the routing
stages run, so each sub-intent gets its own classification and tier --
one blended decision for a multi-intent request would misroute at least
one of the intents.

This is deliberately rule-based, not model-based: detecting that a
request contains multiple asks doesn't itself require judgment when the
separators are unambiguous, and keeping it deterministic means "no AI
needed" can apply to the segmentation step too, not just to what happens
after it.

Conservative on purpose, same principle as the other deterministic
stages: a false negative here (a compound request treated as one segment)
just means that segment's signals get OR'd together and it possibly
routes a tier higher than strictly necessary -- safe. A false positive
(splitting a single coherent request into unrelated pieces) could
misrepresent what was actually asked. So this only splits on strong,
unambiguous separators, never on a bare "and" ("review and approve" is
one action, not two).
"""
from __future__ import annotations

import re

# Explicit sequencing connectors -- deliberately narrower than "and" alone.
_CONNECTOR_PATTERN = re.compile(
    r"\s*(?:\band also\b|\band then\b|\balso,\s|\badditionally,\s)\s*",
    re.IGNORECASE,
)

_LIST_MARKER_PATTERN = re.compile(r"^\s*(?:[-*]|\d+[.)])\s+")


def segment_request(text: str) -> list[str]:
    """Split `text` into independent segments using only unambiguous
    separators. Returns [text] unchanged (stripped) when no such
    separator is found, so single-intent callers see no behavior change.
    """
    stripped = text.strip()
    if not stripped:
        return [stripped]

    # Multi-line input: treat each non-empty line as its own segment,
    # stripping any leading list marker ("-", "*", "1.").
    lines = [line for line in stripped.splitlines() if line.strip()]
    if len(lines) > 1:
        return [_LIST_MARKER_PATTERN.sub("", line).strip() for line in lines]

    # Single line: split on semicolons or explicit sequencing connectors.
    parts = re.split(r";", stripped)
    if len(parts) == 1:
        parts = _CONNECTOR_PATTERN.split(stripped)

    segments = [p.strip() for p in parts if p.strip()]
    return segments if segments else [stripped]
