"""Stage 1 (live): GLM-on-Bedrock-backed classifier.

Replaces StubClassifier once real AWS/Bedrock access is wired in. Same
`Classifier` interface (a `classify(text) -> ClassifierOutput` method),
so nothing else in the pipeline changes when this is swapped in.

Requires boto3 and valid AWS credentials with bedrock:InvokeModel (or
Converse) permission on the model configured in config/settings.py --
see CLASSIFIER_IAM_ROLE_ARN there for the intended least-privilege scope.
This module does not attempt to construct AWS credentials itself; it
uses whatever boto3's default credential chain finds (environment,
profile, instance/task role), consistent with running inside a properly
scoped IAM role rather than embedded secrets.

Enforces the bounded output space: GLM's response is parsed and
validated against the fixed IntentCategory set and the Signals schema.
An invalid or unparseable response is retried once with a corrective
prompt; if it still doesn't validate, this raises rather than silently
guessing or passing through unvalidated model output -- this is what
makes "bounded output space carries less misuse risk" an enforced
property of the system instead of an assertion about it. The orchestrator
is responsible for deciding what happens on that failure (see
orchestrator.py's fail-safe: a classifier exception routes the segment
to the same cautious-HIGH default as an unclassifiable OTHER result,
rather than crashing the pipeline).
"""
from __future__ import annotations

import json
import re
from typing import Any, Optional, Protocol

from .schema import ClassifierOutput, IntentCategory, Signals

_VALID_CATEGORIES = {c.value for c in IntentCategory}
_SIGNAL_FIELDS = (
    "external_facing",
    "legal_compliance_exposure",
    "irreversible_or_costly_error",
    "multi_step_reasoning",
    "domain_expertise_required",
)

_SYSTEM_PROMPT = f"""You are an intent classifier. Given a single task request, \
respond with ONLY a JSON object (no markdown fences, no prose) matching this \
exact shape:

{{
  "intent_category": one of {sorted(_VALID_CATEGORIES)},
  "signals": {{
    "external_facing": true or false,
    "legal_compliance_exposure": true or false,
    "irreversible_or_costly_error": true or false,
    "multi_step_reasoning": true or false,
    "domain_expertise_required": true or false
  }},
  "confidence": a number between 0.0 and 1.0 reflecting your confidence \
in this classification
}}

Definitions:
- external_facing: output goes to a customer, regulator, external partner, or public channel
- legal_compliance_exposure: touches legal, regulatory, compliance, or contractual language
- irreversible_or_costly_error: mistakes are hard to undo or expensive (sent communication, published content, financial action)
- multi_step_reasoning: requires chaining steps, synthesis across sources, or non-trivial inference
- domain_expertise_required: needs specialized knowledge beyond general-purpose competence

If the request doesn't clearly fit any category, use "other". Respond with \
the JSON object only."""


class _BedrockClient(Protocol):
    def converse(self, **kwargs: Any) -> dict:
        ...


class ClassifierOutputValidationError(Exception):
    """Raised when a GLM response can't be parsed into a valid ClassifierOutput."""


class ClassifierInvocationError(Exception):
    """Raised when the live classifier fails after exhausting retries."""


class BedrockGLMClassifier:
    """Live classifier backed by GLM on AWS Bedrock.

    `client` is injectable for testing (anything with a `.converse()`
    method matching the Bedrock Converse API shape) -- defaults to a
    lazily-constructed real boto3 client so importing this module never
    requires boto3 or AWS credentials to be present unless it's actually used.
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        region: Optional[str] = None,
        client: Optional[_BedrockClient] = None,
        max_retries: Optional[int] = None,
    ):
        from config.settings import BEDROCK_MODEL_ID, BEDROCK_REGION, CLASSIFIER_MAX_RETRIES

        self.model_id = model_id or BEDROCK_MODEL_ID
        self.region = region or BEDROCK_REGION
        self.max_retries = CLASSIFIER_MAX_RETRIES if max_retries is None else max_retries
        self._client = client  # lazily constructed in _get_client() if None

    def _get_client(self) -> _BedrockClient:
        if self._client is None:
            import boto3  # imported lazily so tests never need boto3 installed

            self._client = boto3.client("bedrock-runtime", region_name=self.region)
        return self._client

    def classify(self, request_text: str) -> ClassifierOutput:
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            raw = self._invoke(request_text, is_retry=attempt > 0, previous_error=last_error)
            try:
                return _parse_and_validate(raw, model_version=self.model_id)
            except ClassifierOutputValidationError as e:
                last_error = e
                continue
        raise ClassifierInvocationError(
            f"GLM ({self.model_id}) returned invalid output after {self.max_retries + 1} attempt(s): {last_error}"
        )

    def _invoke(self, request_text: str, is_retry: bool, previous_error: Optional[Exception]) -> str:
        client = self._get_client()
        user_text = request_text
        if is_retry:
            user_text = (
                f"{request_text}\n\n"
                f"(Your previous response was invalid: {previous_error}. "
                f"Respond with ONLY the JSON object described in the system prompt.)"
            )
        response = client.converse(
            modelId=self.model_id,
            system=[{"text": _SYSTEM_PROMPT}],
            messages=[{"role": "user", "content": [{"text": user_text}]}],
        )
        return response["output"]["message"]["content"][0]["text"]


def _parse_and_validate(raw_text: str, model_version: str) -> ClassifierOutput:
    text = raw_text.strip()
    # Defensive: strip markdown code fences if the model added them despite instructions.
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ClassifierOutputValidationError(f"not valid JSON: {e}") from e

    category_raw = data.get("intent_category")
    if category_raw not in _VALID_CATEGORIES:
        raise ClassifierOutputValidationError(
            f"intent_category '{category_raw}' not in allowed set {sorted(_VALID_CATEGORIES)}"
        )

    signals_raw = data.get("signals")
    if not isinstance(signals_raw, dict):
        raise ClassifierOutputValidationError("'signals' missing or not an object")
    missing = [f for f in _SIGNAL_FIELDS if f not in signals_raw]
    if missing:
        raise ClassifierOutputValidationError(f"'signals' missing fields: {missing}")
    for f in _SIGNAL_FIELDS:
        if not isinstance(signals_raw[f], bool):
            raise ClassifierOutputValidationError(f"signals.{f} must be a boolean")

    confidence = data.get("confidence")
    if not isinstance(confidence, (int, float)) or not (0.0 <= float(confidence) <= 1.0):
        raise ClassifierOutputValidationError(f"'confidence' must be a number in [0.0, 1.0], got {confidence!r}")

    return ClassifierOutput(
        intent_category=IntentCategory(category_raw),
        signals=Signals(**{f: signals_raw[f] for f in _SIGNAL_FIELDS}),
        confidence=float(confidence),
        model_version=model_version,
    )
