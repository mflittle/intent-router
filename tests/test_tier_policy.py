from router.schema import ClassifierOutput, IntentCategory, Signals, Tier
from router.tier_policy import decide_tier


def _output(intent_category=IntentCategory.CONTENT_GENERATION, signals=None, confidence=0.9):
    return ClassifierOutput(
        intent_category=intent_category,
        signals=signals or Signals(),
        confidence=confidence,
        model_version="test",
    )


def test_no_signals_routes_low():
    tier, _ = decide_tier(_output())
    assert tier == Tier.LOW


def test_one_hard_signal_routes_moderate():
    tier, _ = decide_tier(_output(signals=Signals(external_facing=True)))
    assert tier == Tier.MODERATE


def test_two_hard_signals_route_high():
    tier, _ = decide_tier(_output(signals=Signals(external_facing=True, legal_compliance_exposure=True)))
    assert tier == Tier.HIGH


def test_hard_signal_with_multistep_routes_high():
    tier, _ = decide_tier(_output(signals=Signals(external_facing=True, multi_step_reasoning=True)))
    assert tier == Tier.HIGH


def test_other_category_routes_high():
    tier, _ = decide_tier(_output(intent_category=IntentCategory.OTHER))
    assert tier == Tier.HIGH


def test_low_confidence_escalates_one_tier():
    tier, reason = decide_tier(_output(confidence=0.4))
    assert tier == Tier.MODERATE
    assert "escalated" in reason


def test_low_confidence_never_escalates_past_high():
    tier, _ = decide_tier(_output(intent_category=IntentCategory.OTHER, confidence=0.1))
    assert tier == Tier.HIGH


def test_high_confidence_does_not_escalate():
    tier, reason = decide_tier(_output(confidence=0.95))
    assert tier == Tier.LOW
    assert "escalated" not in reason
