from router.orchestrator import route_request
from router.schema import Tier


class _FailingClassifier:
    """Stand-in for a classifier that raises -- network error, or
    exhausted retries on invalid output (see BedrockGLMClassifier).
    """

    model_id = "failing-test-model"

    def classify(self, request_text: str):
        raise RuntimeError("simulated invocation failure")


def test_classifier_failure_routes_high_not_crash(tmp_path):
    result = route_request(
        "draft a routine internal status email",
        _FailingClassifier(),
        log_path=tmp_path / "d.jsonl",
    )

    assert len(result.segments) == 1
    decision = result.segments[0]
    assert decision.tier == Tier.HIGH
    assert decision.stage == "classifier"
    assert "failing-test-model" in decision.model_version
    assert result.overall_tier == Tier.HIGH


def test_classifier_failure_does_not_affect_other_segments(tmp_path):
    result = route_request(
        "draft a routine internal email; draft another quick internal summary",
        _FailingClassifier(),
        log_path=tmp_path / "d.jsonl",
    )
    # both segments hit the same failing classifier -- both should fail
    # safe to HIGH independently, not crash the whole request
    assert len(result.segments) == 2
    assert all(d.tier == Tier.HIGH for d in result.segments)
