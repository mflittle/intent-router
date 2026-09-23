"""Stage 1: intent + signal classifier.

This module defines the classifier *interface* the rest of the pipeline
depends on. StubClassifier below lets Stage 0 and Stage 2 be built and
tested end-to-end without a live model call. A real implementation
(GLM on Bedrock, behind the governance wrapper: pinned model version,
invoke-only IAM role, output validated against the bounded label set)
plugs in later by implementing the same `classify` method -- the router
never changes.
"""
from __future__ import annotations

from typing import Protocol

from .schema import ClassifierOutput, IntentCategory, Signals


class Classifier(Protocol):
    def classify(self, request_text: str) -> ClassifierOutput:
        ...


class StubClassifier:
    """Deterministic keyword-based classifier for development and tests.

    Not a real model call -- just enough logic to exercise the router
    end-to-end and validate Stage 2's tier policy before introducing an
    actual LLM (and its failure modes) into the pipeline.
    """

    MODEL_VERSION = "stub-v0"

    def classify(self, request_text: str) -> ClassifierOutput:
        text = request_text.lower()

        category = IntentCategory.OTHER
        if any(w in text for w in ("email", "draft", "slide", "write")):
            category = IntentCategory.CONTENT_GENERATION
        elif any(w in text for w in ("code", "function", "script", "bug")):
            category = IntentCategory.CODE_GENERATION
        elif any(w in text for w in ("analyze", "data", "trend", "chart")):
            category = IntentCategory.DATA_ANALYSIS
        elif any(w in text for w in ("summarize", "summary", "tl;dr")):
            category = IntentCategory.SUMMARIZATION
        elif any(w in text for w in ("what is", "who is", "when did", "?")):
            category = IntentCategory.QA_LOOKUP
        elif any(w in text for w in ("should we", "recommend", "decide")):
            category = IntentCategory.DECISION_SUPPORT

        signals = Signals(
            external_facing=any(w in text for w in ("customer", "client", "regulator", "public", "external")),
            legal_compliance_exposure=any(w in text for w in ("legal", "compliance", "regulat", "contract")),
            irreversible_or_costly_error=any(w in text for w in ("send", "publish", "submit", "file with")),
            multi_step_reasoning=any(w in text for w in ("analyze", "synthesize", "across", "multiple")),
            domain_expertise_required=any(w in text for w in ("specializ", "expert", "technical")),
        )

        return ClassifierOutput(
            intent_category=category,
            signals=signals,
            confidence=0.9,
            model_version=self.MODEL_VERSION,
        )
