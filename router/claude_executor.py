"""MODERATE/HIGH-tier executor: Claude on Bedrock.

Governance rationale (see README): keeping MODERATE and HIGH on one
governed vendor (rather than adding a third, open-weight-only vendor
relationship) keeps the sharpest, most defensible line in this project's
governance story at open-weight vs. vendor-governed -- not spread across
three separate vendor relationships to govern.

INVOCATION PATH -- corrected after live testing (2026-09-25), previously
wrong: this module originally used the `anthropic` package's
AnthropicBedrockMantle client against a `bedrock-mantle` endpoint,
following Anthropic's own docs, which state a bare model ID
(e.g. "anthropic.claude-haiku-4-5") is sufficient. Live testing against
a real account proved that guidance incomplete/wrong for this account:
a direct AWS CLI call with the bare model ID against boto3's
`bedrock-runtime` Converse API failed with

    ValidationException: Invocation of model ID
    anthropic.claude-haiku-4-5-20251001-v1:0 with on-demand throughput
    isn't supported. Retry your request with the ID or ARN of an
    inference profile that contains this model.

and succeeded only once the model ID was swapped for its cross-region
INFERENCE PROFILE ID (e.g. "us.anthropic.claude-haiku-4-5-20251001-v1:0").
This module now uses boto3's `bedrock-runtime` client directly, the same
pattern as GLMBedrockExecutor and BedrockGLMClassifier, rather than a
separate Anthropic-SDK invocation path -- one consistent invocation
pattern across all three executors instead of a second, unverified one.
`config/settings.py`'s MODERATE_MODEL_ID / HIGH_MODEL_ID must be
inference profile IDs, not bare model IDs -- see that file's comments.
"""
from __future__ import annotations

from typing import Any, Optional, Protocol


class _BedrockClient(Protocol):
    def converse(self, **kwargs: Any) -> dict:
        ...


_SYSTEM_PROMPT = "You are a helpful, precise assistant. Carry out the requested task directly."


class ClaudeBedrockExecutor:
    """MODERATE/HIGH-tier task executor, Claude on Bedrock via the
    Converse API.

    `client` is injectable for testing -- defaults to a lazily-constructed
    real boto3 `bedrock-runtime` client, so importing this module never
    requires boto3 or AWS credentials unless it's actually used.

    `model_id` MUST be an inference profile ID (e.g.
    "us.anthropic.claude-haiku-4-5-20251001-v1:0"), not a bare model ID --
    see module docstring.
    """

    def __init__(
        self,
        model_id: str,
        region: Optional[str] = None,
        client: Optional[_BedrockClient] = None,
        max_tokens: int = 1024,
    ):
        from config.settings import BEDROCK_REGION

        self.model_id = model_id
        self.region = region or BEDROCK_REGION
        self.max_tokens = max_tokens
        self._client = client

    def _get_client(self) -> _BedrockClient:
        if self._client is None:
            import boto3

            self._client = boto3.client("bedrock-runtime", region_name=self.region)
        return self._client

    def execute(self, task_text: str) -> str:
        client = self._get_client()
        response = client.converse(
            modelId=self.model_id,
            system=[{"text": _SYSTEM_PROMPT}],
            messages=[{"role": "user", "content": [{"text": task_text}]}],
            inferenceConfig={"maxTokens": self.max_tokens},
        )
        return response["output"]["message"]["content"][0]["text"]
