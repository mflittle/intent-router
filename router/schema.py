"""Shared data contracts for the intent classification + routing pipeline.

These types are the interface between the pipeline stages:
  Stage -1 (segmentation)      -> splits a request into independent segments
  Stage 0  (scope_gate)        -> per-segment: out of scope? reject, no execution
  Stage 0a (template_prefilter)-> per-segment: templatable transform? bypass
  Stage 0b (fact_lookup)       -> per-segment: known fact? bypass
  Stage 0c (knowledge_resolver)-> per-segment: internal knowledge match? LOW tier
  Stage 1  (classifier)        -> per-segment: produces a ClassifierOutput
  Stage 2  (tier_policy)       -> per-segment: signals -> tier
  Stage 3  (execution)         -> per-segment: tier -> actually perform the task

Keeping this in one module means the router, the classifier, the
executors, and the tests all agree on one shape.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


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
    """Final output of routing ONE segment, and the audit record for it.

    `output` carries the actual answer for stages that compute one as
    part of deciding the route -- template_prefilter's transform result,
    fact_lookup's/knowledge_resolver's answer, or scope_gate's rejection
    message. It's None for "classifier" stage decisions, since Stage 1
    only classifies; Stage 3 (execution) fills in the real answer for
    those separately, once a tier has been decided.
    """
    segment_text: str
    intent_category: IntentCategory
    signals: Signals
    confidence: float
    tier: Tier
    tier_reason: str
    route: str
    model_version: str
    stage: str  # "scope_gate" | "template" | "fact_lookup" | "knowledge_resolver" | "classifier"
    timestamp: str
    request_id: str  # ties segments of one compound request together in the decision log
    output: Optional[str] = None


@dataclass(frozen=True)
class MultiRouteResult:
    """Final output of routing a full (possibly compound) request.

    `segments` holds one RouteDecision per independently-routed sub-intent
    -- a single-intent request still produces a one-element list, so
    callers always deal with the same shape. `overall_tier` is the max
    across segments, for callers that just want "what's the most
    capability this request needs anywhere in it".
    """
    request_id: str
    segments: tuple[RouteDecision, ...]
    overall_tier: Tier
    timestamp: str

@dataclass(frozen=True)
class ExecutionResult:
    """Stage 3 output: the actual outcome of carrying out ONE routed
    segment, pairing back to its RouteDecision by `request_id` +
    `segment_text`.

    `output` is None when `human_review_required` is True -- the HIGH
    tier's human-in-the-loop gate means execution is deliberately NOT
    performed automatically (see router/execution.py). It's also
    possible for a TRIVIAL decision to already carry its answer on the
    RouteDecision itself (Stage 0's output field); in that case Stage 3
    passes it through rather than invoking a model at all.
    """
    segment_text: str
    tier: Tier
    output: Optional[str]
    executor_model_id: str
    human_review_required: bool
    timestamp: str


@dataclass(frozen=True)
class ExecutionSummary:
    """Final output of executing a full (possibly compound) request --
    one ExecutionResult per segment of the corresponding MultiRouteResult.
    """
    request_id: str
    results: tuple[ExecutionResult, ...]
    timestamp: str
