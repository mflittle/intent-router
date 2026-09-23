"""Shared data contracts for the intent classification + routing pipeline.

These types are the interface between the three pipeline stages:
  Stage 0 (template_prefilter) -> bypasses the rest entirely
  Stage 1 (classifier)         -> produces a ClassifierOutput
  Stage 2 (tier_policy)        -> consumes ClassifierOutput, produces a tier

Keeping this in one module means the router, the classifier, and the tests
all agree on one shape.
"""
from __future__ import annotations

from dataclasses import dataclass
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
    """Final output of the full pipeline, and the audit record for it."""
    intent_category: IntentCategory
    signals: Signals
    confidence: float
    tier: Tier
    tier_reason: str
    route: str
    model_version: str
    stage: str  # "template" | "classifier"
    timestamp: str
