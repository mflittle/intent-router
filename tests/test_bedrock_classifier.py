import json

import pytest

from router.bedrock_classifier import BedrockGLMClassifier, ClassifierInvocationError
from router.schema import IntentCategory


class _FakeBedrockClient:
    """Test double for the Bedrock Converse API client."""

    def __init__(self, responses):
        # responses: list of raw text strings to return on successive calls
        self._responses = list(responses)
        self.calls = []

    def converse(self, **kwargs):
        self.calls.append(kwargs)
        text = self._responses.pop(0)
        return {"output": {"message": {"content": [{"text": text}]}}}


VALID_RESPONSE = json.dumps({
    "intent_category": "content_generation",
    "signals": {
        "external_facing": False,
        "legal_compliance_exposure": False,
        "irreversible_or_costly_error": False,
        "multi_step_reasoning": False,
        "domain_expertise_required": False,
    },
    "confidence": 0.92,
})


def test_valid_response_parses_correctly():
    client = _FakeBedrockClient([VALID_RESPONSE])
    classifier = BedrockGLMClassifier(model_id="test-model", region="us-east-1", client=client)

    output = classifier.classify("draft an email")

    assert output.intent_category == IntentCategory.CONTENT_GENERATION
    assert output.confidence == 0.92
    assert output.model_version == "test-model"
    assert len(client.calls) == 1


def test_response_wrapped_in_markdown_fence_is_handled():
    fenced = f"```json\n{VALID_RESPONSE}\n```"
    client = _FakeBedrockClient([fenced])
    classifier = BedrockGLMClassifier(model_id="test-model", region="us-east-1", client=client)

    output = classifier.classify("draft an email")
    assert output.intent_category == IntentCategory.CONTENT_GENERATION


def test_invalid_json_retries_then_succeeds():
    client = _FakeBedrockClient(["not json at all", VALID_RESPONSE])
    classifier = BedrockGLMClassifier(model_id="test-model", region="us-east-1", client=client, max_retries=1)

    output = classifier.classify("draft an email")
    assert output.intent_category == IntentCategory.CONTENT_GENERATION
    assert len(client.calls) == 2
    # the retry prompt should reference the prior failure
    assert "invalid" in client.calls[1]["messages"][0]["content"][0]["text"].lower()


def test_invalid_category_exhausts_retries_and_raises():
    bad_category = json.dumps({
        "intent_category": "not_a_real_category",
        "signals": {f: False for f in (
            "external_facing", "legal_compliance_exposure", "irreversible_or_costly_error",
            "multi_step_reasoning", "domain_expertise_required",
        )},
        "confidence": 0.5,
    })
    client = _FakeBedrockClient([bad_category, bad_category])
    classifier = BedrockGLMClassifier(model_id="test-model", region="us-east-1", client=client, max_retries=1)

    with pytest.raises(ClassifierInvocationError):
        classifier.classify("draft an email")
    assert len(client.calls) == 2


def test_missing_signal_field_is_rejected():
    incomplete = json.dumps({
        "intent_category": "content_generation",
        "signals": {"external_facing": False},  # missing the other four
        "confidence": 0.9,
    })
    client = _FakeBedrockClient([incomplete, incomplete])
    classifier = BedrockGLMClassifier(model_id="test-model", region="us-east-1", client=client, max_retries=1)

    with pytest.raises(ClassifierInvocationError):
        classifier.classify("draft an email")


def test_out_of_range_confidence_is_rejected():
    bad_confidence = json.dumps({
        "intent_category": "content_generation",
        "signals": {f: False for f in (
            "external_facing", "legal_compliance_exposure", "irreversible_or_costly_error",
            "multi_step_reasoning", "domain_expertise_required",
        )},
        "confidence": 1.5,
    })
    client = _FakeBedrockClient([bad_confidence, bad_confidence])
    classifier = BedrockGLMClassifier(model_id="test-model", region="us-east-1", client=client, max_retries=1)

    with pytest.raises(ClassifierInvocationError):
        classifier.classify("draft an email")
