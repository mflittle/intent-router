"""Top-level orchestration: Stage 0 -> Stage 1 -> Stage 2, with structured
logging of every routing decision. This is the reusable entry point a
consuming project calls.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from .classifier import Classifier
from .schema import IntentCategory, RouteDecision, Signals, Tier
from .template_prefilter import match_template
from .tier_policy import TIER_TO_ROUTE, decide_tier

logger = logging.getLogger("intent_router")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def route(request: dict, classifier: Classifier) -> RouteDecision:
    """Route a request through the full pipeline.

    `request` must carry "text" (the free-text task description) and may
    carry "operation" to be eligible for the Stage 0 template pre-filter.
    """
    template_match = match_template(request)
    if template_match is not None:
        decision = RouteDecision(
            intent_category=IntentCategory.OTHER,  # not applicable -- template bypassed classification
            signals=Signals(),
            confidence=1.0,
            tier=Tier.TRIVIAL,
            tier_reason=f"matched registered template operation '{template_match.operation}'",
            route=TIER_TO_ROUTE[Tier.TRIVIAL],
            model_version="none",
            stage="template",
            timestamp=_now_iso(),
        )
        _log_decision(decision)
        return decision

    classifier_output = classifier.classify(request["text"])
    tier, reason = decide_tier(classifier_output)

    decision = RouteDecision(
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
    _log_decision(decision)
    return decision


def _log_decision(decision: RouteDecision) -> None:
    """Structured, inspectable log of every routing decision -- the
    "monitored" piece of the governance wrapper. Minimum viable version:
    one JSON line per decision. Swap for a real sink (table, log
    aggregator) without changing callers.
    """
    logger.info(json.dumps({
        "timestamp": decision.timestamp,
        "stage": decision.stage,
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
    }))
