"""MODERATE/HIGH-tier executor: Claude on Bedrock.

Governance rationale (see README): keeping MODERATE and HIGH on one
governed vendor (rather than adding a third, open-weight-only vendor
relationship) keeps the sharpest, most defensible line in this project's
governance story at open-weight vs. vendor-governed -- not spread across
three separate vendor relationships to govern.

IMPORTANT, stated plainly rather than glossed over: this uses a
DIFFERENT invocation path than GLMBedrockExecutor / BedrockGLMClassifier.
Those use boto3's `bedrock-runtime` client and the Converse API directly.
Current Anthropic documentation (platform.claude.com) states that Claude
Sonnet 5 and Claude Haiku 4.5 on Bedrock go through the Anthropic
Messages API via a separate `bedrock-mantle` endpoint, using the
`anthropic` Python SDK's Bedrock integration rather than boto3's Converse
call -- a real difference in Bedrock's API surface across model
providers, not a mistake carried over from the GLM code. One AWS
model-card page suggested boto3 Converse IS also supported for Haiku via
an inference-profile ID, which conflicts with that guidance -- this
module follows Anthropic's own docs as the more authoritative source for
"how to call Claude", but THIS HAS NOT BEEN VERIFIED AGAINST A LIVE
ENDPOINT. Confirm with scripts/smoke_test_claude_executor.py before
relying on it.
"""
from __future__ import annotations

from typing import Any, Optional, Protocol


class _AnthropicBedrockClient(Protocol):
    class _Messages(Protocol):
        def create(self, **kwargs: Any) -> Any:
            ...

    messages: "_AnthropicBedrockClient._Messages"


class ClaudeBedrockExecutor:
    """MODERATE/HIGH-tier task executor, Claude on Bedrock via the
    Anthropic Messages API (AnthropicBedrockMantle client).

    `client` is injectable for testing -- defaults to a lazily-constructed
    real `anthropic.AnthropicBedrockMantle` client, so importing this
    module never requires the `anthropic` package or AWS credentials
    unless it's actually used.
    """

    def __init__(
        self,
        model_id: str,
        region: Optional[str] = None,
        client: Optional[_AnthropicBedrockClient] = None,
        max_tokens: int = 1024,
    ):
        from config.settings import BEDROCK_REGION

        self.model_id = model_id
        self.region = region or BEDROCK_REGION
        self.max_tokens = max_tokens
        self._client = client

    def _get_client(self) -> _AnthropicBedrockClient:
        if self._client is None:
            from anthropic import AnthropicBedrockMantle  # lazy import, see module docstring

            self._client = AnthropicBedrockMantle(aws_region=self.region)
        return self._client

    def execute(self, task_text: str) -> str:
        client = self._get_client()
        message = client.messages.create(
            model=self.model_id,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": task_text}],
        )
        return next(block.text for block in message.content if block.type == "text")
