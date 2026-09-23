"""Stage 2: deterministic tier policy.

This is the only place that decides what tier a task routes to. It is a
pure function of the classifier's output -- no model call, no randomness --
so the escalation policy is auditable and can change without touching the
classifier itself. This separation of judgment (Stage 1) from policy
(Stage 2) is the governance pattern this project is meant to demonstrate.
"""
from __future__ import annotations

from .schema import ClassifierOutput, IntentCategory, Tier

# Policy knob, not a discovered truth: below this confidence, escalate one
# tier from whatever the signal-based rules produced. Untuned placeholder --
# revisit once real classification runs give us something to tune against.
CONFIDENCE_ESCALATION_THRESHOLD = 0.6

_TIER_ORDER = [Tier.TRIVIAL, Tier.LOW, Tier.MODERATE, Tier.HIGH]


def _escalate(tier: Tier) -> Tier:
    idx = _TIER_ORDER.index(tier)
    return _TIER_ORDER[min(idx + 1, len(_TIER_ORDER) - 1)]


def decide_tier(output: ClassifierOutput) -> tuple[Tier, str]:
    """Map a classifier output to a tier and a human-readable reason.

    Locked rules:
      - intent_category == other                  -> high (cautious default)
      - 2+ hard signals present                    -> high
      - 1 hard signal AND multi_step_reasoning      -> high
      - exactly 1 hard signal, no others            -> moderate
      - 0 hard signals                              -> low
      - low-confidence classification escalates one tier, never down

    "Hard signals" are external_facing, legal_compliance_exposure, and
    irreversible_or_costly_error -- see Signals.hard_signal_count().
    """
    s = output.signals
    hard_count = s.hard_signal_count()

    if output.intent_category == IntentCategory.OTHER:
        tier, reason = Tier.HIGH, "unclassifiable intent (other) -- cautious default"
    elif hard_count >= 2:
        tier, reason = Tier.HIGH, f"{hard_count} hard signals present"
    elif hard_count == 1 and s.multi_step_reasoning:
        tier, reason = Tier.HIGH, "hard signal combined with multi-step reasoning"
    elif hard_count == 1:
        tier, reason = Tier.MODERATE, "exactly one hard signal present"
    else:
        tier, reason = Tier.LOW, "no hard signals present"

    if output.confidence < CONFIDENCE_ESCALATION_THRESHOLD:
        escalated = _escalate(tier)
        if escalated != tier:
            reason = f"{reason}; escalated from {tier.value} on low confidence ({output.confidence:.2f})"
            tier = escalated

    return tier, reason


TIER_TO_ROUTE = {
    Tier.TRIVIAL: "template-no-model-call",
    Tier.LOW: "open-weight-classifier-tier",
    Tier.MODERATE: "mid-tier-model",
    Tier.HIGH: "frontier-tier-human-in-loop",
}
