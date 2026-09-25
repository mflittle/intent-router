from router.claude_executor import ClaudeBedrockExecutor


class _FakeBedrockClient:
    def __init__(self, text):
        self._text = text
        self.calls = []

    def converse(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [{"text": self._text}],
                }
            }
        }


def test_execute_returns_generated_text():
    client = _FakeBedrockClient("Here is the summary for counsel.")
    executor = ClaudeBedrockExecutor(
        model_id="us.anthropic.claude-test-v1:0", region="us-east-1", client=client
    )

    output = executor.execute("summarize this exam finding")

    assert output == "Here is the summary for counsel."
    call = client.calls[0]
    assert call["modelId"] == "us.anthropic.claude-test-v1:0"
    assert call["messages"][0]["content"] == [{"text": "summarize this exam finding"}]


def test_execute_passes_max_tokens_in_inference_config():
    client = _FakeBedrockClient("ok")
    executor = ClaudeBedrockExecutor(
        model_id="us.anthropic.claude-test-v1:0",
        region="us-east-1",
        client=client,
        max_tokens=256,
    )

    executor.execute("do the thing")

    assert client.calls[0]["inferenceConfig"]["maxTokens"] == 256
