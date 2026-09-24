"""LOW-tier executor: GLM on Bedrock, performing the task directly
(open-ended generation) rather than classifying it.

Reuses the same model as Stage 1's live classifier (BedrockGLMClassifier)
-- consistent with the LOW tier's whole premise, an open-weight model
appropriate for low-stakes, bounded-risk work -- but this is a distinct
class because the job is different: no strict output schema to enforce,
just "do the task and return the result".
"""
from __future__ import annotations

from typing import Any, Optional, Protocol


class _BedrockClient(Protocol):
    def converse(self, **kwargs: Any) -> dict:
        ...


_SYSTEM_PROMPT = "You are a helpful assistant. Carry out the requested task directly and concisely."


class GLMBedrockExecutor:
    """LOW-tier task executor, GLM on Bedrock via the Converse API.

    `client` is injectable for testing, same pattern as
    BedrockGLMClassifier -- defaults to a lazily-constructed real boto3
    client so importing this module never requires boto3/AWS credentials
    unless it's actually used.
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        region: Optional[str] = None,
        client: Optional[_BedrockClient] = None,
    ):
        from config.settings import BEDROCK_MODEL_ID, BEDROCK_REGION

        self.model_id = model_id or BEDROCK_MODEL_ID
        self.region = region or BEDROCK_REGION
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
        )
        return response["output"]["message"]["content"][0]["text"]
