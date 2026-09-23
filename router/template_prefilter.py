"""Stage 0: deterministic template pre-filter.

This runs before any model call. It is an allowlist of known-safe,
mechanically templatable operations -- not a heuristic that tries to detect
"simple" requests. Anything that isn't an exact registry match falls
through to the classifier; the default is always "go to Stage 1", never
"assume trivial".

False negatives here (a genuinely trivial task falls through) just cost one
extra classifier call. False positives (something non-trivial gets treated
as templatable) would mean a real task skips model review entirely, which
is the one failure mode this stage must never allow. Hence: allowlist, not
heuristic.

Starts empty. Grows only as real, recurring trivial operations justify
adding an entry -- consistent with this project's own bounded-output-space
argument for why small, fixed sets carry less risk than open-ended ones.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional


@dataclass(frozen=True)
class TemplateMatch:
    operation: str
    output: str


_TEMPLATE_REGISTRY: dict[str, Callable[[dict], str]] = {}


def register_template(operation: str):
    """Decorator to register a Stage 0 template handler."""
    def _wrap(fn: Callable[[dict], str]):
        _TEMPLATE_REGISTRY[operation] = fn
        return fn
    return _wrap


def match_template(request: dict) -> Optional[TemplateMatch]:
    """Return a TemplateMatch if `request` exactly matches a registered
    template operation, else None (falls through to Stage 1).

    `request` must carry an explicit "operation" key set by the caller --
    this stage never infers "this looks templatable" from free text.
    """
    operation = request.get("operation")
    if operation is None or operation not in _TEMPLATE_REGISTRY:
        return None
    handler = _TEMPLATE_REGISTRY[operation]
    return TemplateMatch(operation=operation, output=handler(request))


def registered_operations() -> list[str]:
    return sorted(_TEMPLATE_REGISTRY.keys())
