from router.classifier import StubClassifier
from router.fact_lookup import register_fact
from router.orchestrator import route, route_request
from router.schema import Tier
from router.template_prefilter import register_template


def test_template_match_bypasses_classifier(tmp_path):
    @register_template("noop_test_op")
    def _handler(request):
        return request["text"]

    decision = route({"operation": "noop_test_op", "text": "hi"}, StubClassifier(), log_path=tmp_path / "d.jsonl")
    assert decision.tier == Tier.TRIVIAL
    assert decision.stage == "template"
    assert decision.model_version == "none"


def test_routine_internal_email_routes_low_or_moderate(tmp_path):
    decision = route(
        {"text": "draft a routine internal status update email to my team"},
        StubClassifier(),
        log_path=tmp_path / "d.jsonl",
    )
    assert decision.stage == "classifier"
    assert decision.tier in (Tier.LOW, Tier.MODERATE)


def test_external_legal_multi_account_task_routes_high(tmp_path):
    decision = route(
        {"text": "summarize this exam finding for external counsel and the regulator, covering multiple accounts"},
        StubClassifier(),
        log_path=tmp_path / "d.jsonl",
    )
    assert decision.tier == Tier.HIGH


def test_single_intent_request_returns_one_segment(tmp_path):
    result = route_request(
        "draft a routine internal status update email", StubClassifier(), log_path=tmp_path / "d.jsonl"
    )
    assert len(result.segments) == 1
    assert result.overall_tier == result.segments[0].tier


def test_fact_lookup_segment_bypasses_classifier(tmp_path):
    @register_fact("who is the test director")
    def _answer():
        return "Jane Test"

    result = route_request("Who is the test director?", StubClassifier(), log_path=tmp_path / "d.jsonl")
    assert len(result.segments) == 1
    assert result.segments[0].stage == "fact_lookup"
    assert result.segments[0].tier == Tier.TRIVIAL
    assert result.segments[0].model_version == "none"


def test_compound_request_routes_each_segment_independently(tmp_path):
    @register_fact("who is the test compliance officer")
    def _answer():
        return "John Test"

    text = "who is the test compliance officer and also summarize this exam finding for external counsel and the regulator"
    result = route_request(text, StubClassifier(), log_path=tmp_path / "d.jsonl")

    assert len(result.segments) == 2
    fact_segment, classifier_segment = result.segments
    assert fact_segment.stage == "fact_lookup"
    assert fact_segment.tier == Tier.TRIVIAL
    assert classifier_segment.stage == "classifier"
    assert classifier_segment.tier == Tier.HIGH
    # overall tier reflects the riskiest segment, not the first or last
    assert result.overall_tier == Tier.HIGH


def test_overall_tier_is_low_when_no_segment_escalates(tmp_path):
    # Both segments classify cleanly (no hard signals) -- neither falls to
    # OTHER, which is the case that deliberately escalates to HIGH.
    result = route_request(
        "draft a routine internal email; draft a quick internal slide summary",
        StubClassifier(),
        log_path=tmp_path / "d.jsonl",
    )
    assert len(result.segments) == 2
    assert result.overall_tier == Tier.LOW


def test_unclassifiable_segment_escalates_overall_tier(tmp_path):
    # A segment the stub classifier can't categorize falls to OTHER, which
    # is a deliberate cautious-default -> HIGH, and that should dominate
    # the overall tier even when the other segment is low-risk.
    result = route_request(
        "draft a routine internal email; update the tracker", StubClassifier(), log_path=tmp_path / "d.jsonl"
    )
    assert len(result.segments) == 2
    assert result.overall_tier == Tier.HIGH
