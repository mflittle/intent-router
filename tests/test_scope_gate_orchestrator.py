from router.classifier import StubClassifier
from router.orchestrator import route_request
from router.schema import Tier


def test_compound_out_of_scope_request_rejects_only_the_bad_segment(tmp_path):
    # This is the motivating example. With comma-list segmentation, this
    # now splits into three independent segments instead of being
    # rejected as one undifferentiated block -- so the legitimate part
    # ("summarize this brief") still gets a real classification instead
    # of being thrown away because "get lunch" was tacked onto the end.
    result = route_request(
        "summarize this brief, send it to the judge, and get lunch",
        StubClassifier(),
        log_path=tmp_path / "d.jsonl",
    )
    assert len(result.segments) == 3
    by_text = {d.segment_text: d for d in result.segments}

    assert by_text["get lunch"].stage == "scope_gate"
    assert by_text["get lunch"].tier == Tier.TRIVIAL
    assert by_text["get lunch"].model_version == "none"
    assert by_text["get lunch"].route == "reject-out-of-scope-notify-user"

    # The legitimate first segment gets classified normally -- it's no
    # longer collateral damage from the third segment being nonsensical.
    assert by_text["summarize this brief"].stage == "classifier"
    assert by_text["summarize this brief"].tier == Tier.LOW

    # "send it to the judge" doesn't match any StubClassifier keyword, so
    # it falls to OTHER -> the existing cautious-HIGH default. That's a
    # separate, correct mechanism (ambiguous, not obviously out of scope)
    # from the scope gate (obviously out of scope) -- both exist for a
    # reason, and this segment exercises the ambiguous-case backstop.
    assert by_text["send it to the judge"].stage == "classifier"
    assert by_text["send it to the judge"].tier == Tier.HIGH


def test_scope_gate_does_not_affect_legitimate_requests(tmp_path):
    result = route_request(
        "draft a routine internal status update email",
        StubClassifier(),
        log_path=tmp_path / "d.jsonl",
    )
    assert result.segments[0].stage == "classifier"
