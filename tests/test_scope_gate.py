from router.scope_gate import match_scope_gate, register_out_of_scope_phrase


def test_denylisted_phrase_matches_standalone():
    match = match_scope_gate("get lunch")
    assert match is not None
    assert match.matched_phrase == "get lunch"


def test_denylisted_phrase_matches_as_substring():
    match = match_scope_gate("summarize this brief, send it to the judge, and get lunch")
    assert match is not None
    assert match.matched_phrase == "get lunch"


def test_legitimate_request_does_not_match():
    assert match_scope_gate("summarize this brief for the client") is None


def test_custom_registered_phrase_matches():
    register_out_of_scope_phrase("feed the cat", reason="test entry")
    match = match_scope_gate("please feed the cat before you leave")
    assert match is not None
    assert match.reason == "test entry"
