from router.claude_executor import ClaudeBedrockExecutor


class _FakeTextBlock:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class _FakeMessage:
    def __init__(self, text):
        self.content = [_FakeTextBlock(text)]


class _FakeMessages:
    def __init__(self, text):
        self._text = text
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeMessage(self._text)


class _FakeAnthropicBedrockClient:
    def __init__(self, text):
        self.messages = _FakeMessages(text)


def test_execute_returns_generated_text():
    client = _FakeAnthropicBedrockClient("Here is the summary for counsel.")
    executor = ClaudeBedrockExecutor(model_id="anthropic.claude-test", region="us-east-1", client=client)

    output = executor.execute("summarize this exam finding")

    assert output == "Here is the summary for counsel."
    assert client.messages.calls[0]["model"] == "anthropic.claude-test"
    assert client.messages.calls[0]["messages"][0]["content"] == "summarize this exam finding"
