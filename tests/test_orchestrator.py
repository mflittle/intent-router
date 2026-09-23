from router.classifier import StubClassifier
from router.orchestrator import route
from router.schema import Tier
from router.template_prefilter import register_template


def test_template_match_bypasses_classifier():
    @register_template("noop_test_op")
    def _handler(request):
        return request["text"]

    decision = route({"operation": "noop_test_op", "text": "hi"}, StubClassifier())
    assert decision.tier == Tier.TRIVIAL
    assert decision.stage == "template"
    assert decision.model_version == "none"


def test_routine_internal_email_routes_low_or_moderate():
    decision = route(
        {"text": "draft a routine internal status update email to my team"},
        StubClassifier(),
    )
    assert decision.stage == "classifier"
    assert decision.tier in (Tier.LOW, Tier.MODERATE)


def test_external_legal_multi_account_task_routes_high():
    decision = route(
        {"text": "summarize this exam finding for external counsel and the regulator, covering multiple accounts"},
        StubClassifier(),
    )
    assert decision.tier == Tier.HIGH
