import json

from router.classifier import StubClassifier
from router.orchestrator import route_request


def test_route_request_writes_durable_log_entries(tmp_path):
    log_path = tmp_path / "decisions.jsonl"
    result = route_request(
        "draft a routine internal status email",
        StubClassifier(),
        log_path=log_path,
    )

    assert log_path.exists()
    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 1

    record = json.loads(lines[0])
    assert record["request_id"] == result.request_id
    assert record["stage"] == "classifier"
    assert "signals" in record and "confidence" in record


def test_compound_request_shares_one_request_id_across_segments(tmp_path):
    log_path = tmp_path / "decisions.jsonl"
    result = route_request(
        "draft a routine internal email; draft a quick internal slide summary",
        StubClassifier(),
        log_path=log_path,
    )

    assert len(result.segments) == 2
    assert len(set(d.request_id for d in result.segments)) == 1
    assert result.segments[0].request_id == result.request_id

    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 2
    request_ids = {json.loads(line)["request_id"] for line in lines}
    assert request_ids == {result.request_id}
