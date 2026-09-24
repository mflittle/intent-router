"""Stage -1: deterministic request segmentation.

Splits a compound request ("summarize this filing and draft a note to
external counsel") into independent segments *before* any of the routing
stages run, so each sub-intent gets its own classification and tier --
one blended decision for a multi-intent request would misroute at least
one of the intents, and an out-of-scope fragment inside an otherwise
legitimate request would cause the whole request to be rejected instead
of just the out-of-scope part (see the comma-list case below).

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
misrepresent what was actually asked. So this only splits on strong
separators, never on a bare "and" with no other signal ("review and
approve" stays one action, not two).

One deliberately accepted exception to that conservatism: a serial comma
list ("summarize this brief, send it to the judge, and get lunch") is
split on 2+ commas. Two or more commas is treated as a strong-enough
signal of a genuine list (as opposed to one incidental comma used for
some other grammatical reason -- "well, let's get started" stays one
segment). This is a real, known trade-off, not a free improvement: a
sentence that happens to use 2+ commas for a different reason ("summarize
this, in plain English, for the board") will get incorrectly split into
fragments. That risk was judged worth taking specifically because the
alternative failure mode is worse in a common real case -- an
out-of-scope fragment anywhere in an unsplit compound request causes the
*entire* request to be rejected by the scope gate, even when most of it
was legitimate.
"""
from __future__ import annotations

import re

# Explicit sequencing connectors -- deliberately narrower than "and" alone.
_CONNECTOR_PATTERN = re.compile(
    r"\s*(?:\band also\b|\band then\b|\balso,\s|\badditionally,\s)\s*",
    re.IGNORECASE,
)

_LIST_MARKER_PATTERN = re.compile(r"^\s*(?:[-*]|\d+[.)])\s+")
_LEADING_CONNECTOR_WORD = re.compile(r"^(?:and|or)\s+", re.IGNORECASE)


def segment_request(text: str) -> list[str]:
    """Split `text` into independent segments using only unambiguous
    separators (plus the comma-list exception above). Returns [text]
    unchanged (stripped) when no such separator is found, so
    single-intent callers see no behavior change.
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

    # Serial comma list ("A, B, and C" or "A, B, C") -- only when neither
    # of the above already found a split, and only with 2+ commas (the
    # conservative threshold explained in the module docstring).
    if len(parts) == 1 and stripped.count(",") >= 2:
        parts = [_LEADING_CONNECTOR_WORD.sub("", p.strip()).strip() for p in stripped.split(",")]

    segments = [p.strip() for p in parts if p.strip()]
    return segments if segments else [stripped]
