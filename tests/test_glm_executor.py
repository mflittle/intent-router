from router.glm_executor import GLMBedrockExecutor


class _FakeBedrockClient:
    def __init__(self, text):
        self._text = text
        self.calls = []

    def converse(self, **kwargs):
        self.calls.append(kwargs)
        return {"output": {"message": {"content": [{"text": self._text}]}}}


def test_execute_returns_generated_text():
    client = _FakeBedrockClient("Here is the drafted email.")
    executor = GLMBedrockExecutor(model_id="test-glm", region="us-east-1", client=client)

    output = executor.execute("draft a routine internal status email")

    assert output == "Here is the drafted email."
    assert client.calls[0]["modelId"] == "test-glm"
    assert client.calls[0]["messages"][0]["content"][0]["text"] == "draft a routine internal status email"
