"""Shared data contracts for the intent classification + routing pipeline.

These types are the interface between the pipeline stages:
  Stage -1 (segmentation)      -> splits a request into independent segments
  Stage 0a (template_prefilter)-> per-segment: templatable transform? bypass
  Stage 0b (fact_lookup)       -> per-segment: known fact? bypass
  Stage 1 (classifier)         -> per-segment: produces a ClassifierOutput
  Stage 2 (tier_policy)        -> per-segment: signals -> tier

Keeping this in one module means the router, the classifier, and the tests
all agree on one shape.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class IntentCategory(str, Enum):
    CONTENT_GENERATION = "content_generation"
    CODE_GENERATION = "code_generation"
    DATA_ANALYSIS = "data_analysis"
    SUMMARIZATION = "summarization"
    QA_LOOKUP = "qa_lookup"
    DECISION_SUPPORT = "decision_support"
    OTHER = "other"


class Tier(str, Enum):
    TRIVIAL = "trivial"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


@dataclass(frozen=True)
class Signals:
    """Boolean risk signals extracted by the classifier.

    These are judgments about the *task*, not a tier assignment. Stage 2
    turns these into a tier deterministically, so the escalation policy can
    change without touching the classifier.
    """
    external_facing: bool = False
    legal_compliance_exposure: bool = False
    irreversible_or_costly_error: bool = False
    multi_step_reasoning: bool = False
    domain_expertise_required: bool = False

    def hard_signal_count(self) -> int:
        """Count of the three signals that drive escalation on their own."""
        return sum([
            self.external_facing,
            self.legal_compliance_exposure,
            self.irreversible_or_costly_error,
        ])


@dataclass(frozen=True)
class ClassifierOutput:
    """Stage 1 output: what the LLM (or a stub) decided about a request."""
    intent_category: IntentCategory
    signals: Signals
    confidence: float
    model_version: str


@dataclass(frozen=True)
class RouteDecision:
    """Final output of routing ONE segment, and the audit record for it."""
    segment_text: str
    intent_category: IntentCategory
    signals: Signals
    confidence: float
    tier: Tier
    tier_reason: str
    route: str
    model_version: str
    stage: str  # "template" | "fact_lookup" | "classifier"
    timestamp: str


@dataclass(frozen=True)
class MultiRouteResult:
    """Final output of routing a full (possibly compound) request.

    `segments` holds one RouteDecision per independently-routed sub-intent
    -- a single-intent request still produces a one-element list, so
    callers always deal with the same shape. `overall_tier` is the max
    across segments, for callers that just want "what's the most
    capability this request needs anywhere in it".
    """
    segments: tuple[RouteDecision, ...]
    overall_tier: Tier
    timestamp: str
