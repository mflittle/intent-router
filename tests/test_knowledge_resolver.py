from router.classifier import StubClassifier
from router.knowledge_resolver import KnowledgeResolver
from router.orchestrator import route_request
from router.schema import Tier


class _StubResolver:
    """Test double standing in for a real RAG-backed resolver."""

    def resolve(self, question_text: str):
        if "internal knowledge base test question" in question_text.lower():
            return "answer from the stub knowledge base"
        return None


def test_null_resolver_falls_through_to_classifier(tmp_path):
    result = route_request(
        "who is the test director",
        StubClassifier(),
        log_path=tmp_path / "decisions.jsonl",
    )
    assert result.segments[0].stage == "classifier"


def test_knowledge_resolver_match_routes_low_not_trivial(tmp_path):
    result = route_request(
        "what is the internal knowledge base test question answer",
        StubClassifier(),
        knowledge_resolver=_StubResolver(),
        log_path=tmp_path / "decisions.jsonl",
    )
    decision = result.segments[0]
    assert decision.stage == "knowledge_resolver"
    # Grounded retrieval still needs a model call to synthesize -- LOW,
    # not TRIVIAL like the zero-model-call Stage 0a/0b bypasses.
    assert decision.tier == Tier.LOW
    assert decision.model_version == "none"
