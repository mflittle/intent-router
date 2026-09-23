"""Top-level orchestration: Stage -1 -> (Stage 0a / 0b / 0c / Stage 1+2)
per segment, with a durable, structured log of every routing decision.
This is the reusable entry point a consuming project calls.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from . import decision_log
from .classifier import Classifier
from .fact_lookup import match_fact
from .knowledge_resolver import KNOWLEDGE_RESOLVER_CONFIDENCE, KnowledgeResolver, NullKnowledgeResolver
from .schema import IntentCategory, MultiRouteResult, RouteDecision, Signals, Tier
from .segmentation import segment_request
from .template_prefilter import match_template
from .tier_policy import TIER_TO_ROUTE, decide_tier

logger = logging.getLogger("intent_router")

ROUTE_TEMPLATE = "template-no-model-call"
ROUTE_FACT_LOOKUP = "deterministic-lookup-no-model-call"
ROUTE_KNOWLEDGE_RESOLVER = "rag-grounded-low-tier"

_TIER_ORDER = [Tier.TRIVIAL, Tier.LOW, Tier.MODERATE, Tier.HIGH]

_DEFAULT_KNOWLEDGE_RESOLVER = NullKnowledgeResolver()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def route_request(
    text: str,
    classifier: Classifier,
    request_extra: Optional[dict] = None,
    knowledge_resolver: KnowledgeResolver = _DEFAULT_KNOWLEDGE_RESOLVER,
    log_path: Optional[Path] = None,
) -> MultiRouteResult:
    """Route a (possibly compound) request end to end.

    Segments `text` deterministically (Stage -1), then routes each
    segment independently through Stage 0a/0b/0c/1/2. A single-intent
    request still returns a one-element `segments` tuple, so callers
    always handle the same shape. All segments share one `request_id`,
    so a compound request's sub-decisions can be reassembled from the
    decision log afterward.

    `request_extra` is merged into each segment's request dict -- use it
    to carry an explicit "operation" key when the caller already knows a
    segment is a templatable transform (Stage 0a needs that key; it never
    infers "templatable" from free text).
    """
    request_id = str(uuid.uuid4())
    segments = segment_request(text)
    decisions = tuple(
        _route_segment(segment, classifier, request_extra, knowledge_resolver, request_id, log_path)
        for segment in segments
    )
    overall_tier = max(decisions, key=lambda d: _TIER_ORDER.index(d.tier)).tier if decisions else Tier.LOW

    result = MultiRouteResult(
        request_id=request_id, segments=decisions, overall_tier=overall_tier, timestamp=_now_iso()
    )
    logger.info("routed request %s: %d segment(s), overall_tier=%s", request_id, len(decisions), overall_tier.value)
    return result


def route(
    request: dict,
    classifier: Classifier,
    knowledge_resolver: KnowledgeResolver = _DEFAULT_KNOWLEDGE_RESOLVER,
    log_path: Optional[Path] = None,
) -> RouteDecision:
    """Convenience wrapper for a caller that already knows it has exactly
    one segment (e.g. a template operation call). Skips segmentation.
    """
    request_id = str(uuid.uuid4())
    request_extra = {k: v for k, v in request.items() if k != "text"}
    return _route_segment(request["text"], classifier, request_extra, knowledge_resolver, request_id, log_path)


def _route_segment(
    segment_text: str,
    classifier: Classifier,
    request_extra: Optional[dict],
    knowledge_resolver: KnowledgeResolver,
    request_id: str,
    log_path: Optional[Path],
) -> RouteDecision:
    request = {"text": segment_text, **(request_extra or {})}

    template_match = match_template(request)
    if template_match is not None:
        decision = RouteDecision(
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
            request_id=request_id,
        )
        return _finalize(decision, log_path)

    fact_match = match_fact(segment_text)
    if fact_match is not None:
        decision = RouteDecision(
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
            request_id=request_id,
        )
        return _finalize(decision, log_path)

    knowledge_answer = knowledge_resolver.resolve(segment_text)
    if knowledge_answer is not None:
        # Not a zero-model-call bypass -- retrieval is ~deterministic but
        # synthesis still needs a (cheap) model, so this is LOW tier, not
        # TRIVIAL. See knowledge_resolver.py for the reasoning.
        decision = RouteDecision(
            segment_text=segment_text,
            intent_category=IntentCategory.QA_LOOKUP,
            signals=Signals(),
            confidence=KNOWLEDGE_RESOLVER_CONFIDENCE,
            tier=Tier.LOW,
            tier_reason="grounded in internal knowledge base retrieval",
            route=ROUTE_KNOWLEDGE_RESOLVER,
            model_version="none",
            stage="knowledge_resolver",
            timestamp=_now_iso(),
            request_id=request_id,
        )
        return _finalize(decision, log_path)

    classifier_output = classifier.classify(segment_text)
    tier, reason = decide_tier(classifier_output)

    decision = RouteDecision(
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
        request_id=request_id,
    )
    return _finalize(decision, log_path)


def _finalize(decision: RouteDecision, log_path: Optional[Path]) -> RouteDecision:
    """Durable log write (the actual governance/tuning artifact) plus a
    real-time logger line for local visibility.
    """
    decision_log.append_decision(decision, decision.request_id, log_path)
    logger.info(
        "stage=%s tier=%s intent=%s route=%s",
        decision.stage, decision.tier.value, decision.intent_category.value, decision.route,
    )
    return decision
