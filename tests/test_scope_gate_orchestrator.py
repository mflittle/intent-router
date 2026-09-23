from router.classifier import StubClassifier
from router.orchestrator import route_request
from router.schema import Tier


def test_out_of_scope_request_is_rejected_before_any_model_tier(tmp_path):
    # This is the motivating example: without the scope gate, this falls
    # to IntentCategory.OTHER and the cautious-default HIGH tier --
    # spending the most expensive tier on a nonsensical request. The
    # scope gate should catch it first and reject it, at TRIVIAL, with
    # no model call at all.
    result = route_request(
        "summarize this brief, send it to the judge, and get lunch",
        StubClassifier(),
        log_path=tmp_path / "d.jsonl",
    )
    assert len(result.segments) == 1  # bare "and" doesn't split
    decision = result.segments[0]
    assert decision.stage == "scope_gate"
    assert decision.tier == Tier.TRIVIAL
    assert decision.model_version == "none"
    assert decision.route == "reject-out-of-scope-notify-user"
    assert result.overall_tier == Tier.TRIVIAL


def test_scope_gate_does_not_affect_legitimate_requests(tmp_path):
    result = route_request(
        "draft a routine internal status update email",
        StubClassifier(),
        log_path=tmp_path / "d.jsonl",
    )
    assert result.segments[0].stage == "classifier"
