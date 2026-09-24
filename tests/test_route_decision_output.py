from router.classifier import StubClassifier
from router.fact_lookup import register_fact
from router.orchestrator import route
from router.template_prefilter import register_template


def test_template_decision_carries_transform_output(tmp_path):
    @register_template("uppercase_output_test")
    def _handler(request):
        return request["text"].upper()

    decision = route(
        {"operation": "uppercase_output_test", "text": "hello"},
        StubClassifier(),
        log_path=tmp_path / "d.jsonl",
    )
    assert decision.output == "HELLO"


def test_fact_lookup_decision_carries_answer(tmp_path):
    @register_fact("who is the output test director")
    def _answer():
        return "Output Test Person"

    decision = route(
        {"text": "who is the output test director"},
        StubClassifier(),
        log_path=tmp_path / "d.jsonl",
    )
    assert decision.output == "Output Test Person"


def test_scope_gate_decision_carries_rejection_message(tmp_path):
    decision = route(
        {"text": "get lunch"},
        StubClassifier(),
        log_path=tmp_path / "d.jsonl",
    )
    assert decision.output is not None
    assert "get lunch" in decision.output


def test_classifier_decision_has_no_output(tmp_path):
    decision = route(
        {"text": "draft a routine internal status update email"},
        StubClassifier(),
        log_path=tmp_path / "d.jsonl",
    )
    assert decision.output is None
