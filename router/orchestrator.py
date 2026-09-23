"""Top-level orchestration: Stage -1 -> (Stage 0a / 0b / Stage 1+2) per
segment, with structured logging of every routing decision. This is the
reusable entry point a consuming project calls.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Optional

from .classifier import Classifier
from .fact_lookup import match_fact
from .schema import IntentCategory, MultiRouteResult, RouteDecision, Signals, Tier
from .segmentation import segment_request
from .template_prefilter import match_template
from .tier_policy import TIER_TO_ROUTE, decide_tier

logger = logging.getLogger("intent_router")

ROUTE_TEMPLATE = "template-no-model-call"
ROUTE_FACT_LOOKUP = "deterministic-lookup-no-model-call"

_TIER_ORDER = [Tier.TRIVIAL, Tier.LOW, Tier.MODERATE, Tier.HIGH]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def route_request(
    text: str,
    classifier: Classifier,
    request_extra: Optional[dict] = None,
) -> MultiRouteResult:
    """Route a (possibly compound) request end to end.

    Segments `text` deterministically (Stage -1), then routes each
    segment independently through Stage 0a/0b/1/2. A single-intent
    request still returns a one-element `segments` tuple, so callers
    always handle the same shape.

    `request_extra` is merged into each segment's request dict -- use it
    to carry an explicit "operation" key when the caller already knows a
    segment is a templatable transform (Stage 0a needs that key; it never
    infers "templatable" from free text).
    """
    segments = segment_request(text)
    decisions = tuple(
        _route_segment(segment, classifier, request_extra) for segment in segments
    )
    overall_tier = max(decisions, key=lambda d: _TIER_ORDER.index(d.tier)).tier if decisions else Tier.LOW

    result = MultiRouteResult(segments=decisions, overall_tier=overall_tier, timestamp=_now_iso())
    _log_result(result)
    return result


def route(request: dict, classifier: Classifier) -> RouteDecision:
    """Convenience wrapper for a caller that already knows it has exactly
    one segment (e.g. a template operation call). Skips segmentation.
    """
    decision = _route_segment(request["text"], classifier, {k: v for k, v in request.items() if k != "text"})
    _log_decision(decision)
    return decision


def _route_segment(segment_text: str, classifier: Classifier, request_extra: Optional[dict]) -> RouteDecision:
    request = {"text": segment_text, **(request_extra or {})}

    template_match = match_template(request)
    if template_match is not None:
        return RouteDecision(
            segment_text=segment_text,
            intent_category=IntentCategory.OTHER,  # not applicable -- template bypassed classification
            signals=Signals(),
            confidence=1.0,
            tier=Tier.TRIVIAL,
            tier_reason=f"matched registered template operation '{template_match.operation}'",
            route=ROUTE_TEMPLATE,
            model_version="none",
            stage="template",
            timestamp=_now_iso(),
        )

    fact_match = match_fact(segment_text)
    if fact_match is not None:
        return RouteDecision(
            segment_text=segment_text,
            intent_category=IntentCategory.QA_LOOKUP,
            signals=Signals(),
            confidence=1.0,
            tier=Tier.TRIVIAL,
            tier_reason="matched registered fact lookup entry",
            route=ROUTE_FACT_LOOKUP,
            model_version="none",
            stage="fact_lookup",
            timestamp=_now_iso(),
        )

    classifier_output = classifier.classify(segment_text)
    tier, reason = decide_tier(classifier_output)

    return RouteDecision(
        segment_text=segment_text,
        intent_category=classifier_output.intent_category,
        signals=classifier_output.signals,
        confidence=classifier_output.confidence,
        tier=tier,
        tier_reason=reason,
        route=TIER_TO_ROUTE[tier],
        model_version=classifier_output.model_version,
        stage="classifier",
        timestamp=_now_iso(),
    )


def _log_decision(decision: RouteDecision) -> None:
    logger.info(json.dumps(_decision_dict(decision)))


def _log_result(result: MultiRouteResult) -> None:
    logger.info(json.dumps({
        "timestamp": result.timestamp,
        "overall_tier": result.overall_tier.value,
        "segment_count": len(result.segments),
        "segments": [_decision_dict(d) for d in result.segments],
    }))


def _decision_dict(decision: RouteDecision) -> dict:
    """Structured, inspectable record of one routing decision -- the
    "monitored" piece of the governance wrapper. Minimum viable version:
    one JSON object per decision, logged as a line. Swap for a real sink
    (table, log aggregator) without changing callers.
    """
    return {
        "timestamp": decision.timestamp,
        "stage": decision.stage,
        "segment_text": decision.segment_text,
        "intent_category": decision.intent_category.value,
        "tier": decision.tier.value,
        "tier_reason": decision.tier_reason,
        "route": decision.route,
        "confidence": decision.confidence,
        "model_version": decision.model_version,
        "signals": {
            "external_facing": decision.signals.external_facing,
            "legal_compliance_exposure": decision.signals.legal_compliance_exposure,
            "irreversible_or_costly_error": decision.signals.irreversible_or_costly_error,
            "multi_step_reasoning": decision.signals.multi_step_reasoning,
            "domain_expertise_required": decision.signals.domain_expertise_required,
        },
    }
