from router.segmentation import segment_request


def test_single_sentence_stays_one_segment():
    assert segment_request("draft an email to my team") == ["draft an email to my team"]


def test_bare_and_does_not_split():
    text = "review and approve the attached memo"
    assert segment_request(text) == [text]


def test_explicit_connector_splits():
    segments = segment_request("summarize this filing and also draft a note to counsel")
    assert segments == ["summarize this filing", "draft a note to counsel"]


def test_semicolon_splits():
    segments = segment_request("pull the Q3 numbers; email them to Cheryl")
    assert segments == ["pull the Q3 numbers", "email them to Cheryl"]


def test_multiline_list_splits_per_line():
    text = "- summarize the filing\n- draft a note to counsel\n- update the tracker"
    segments = segment_request(text)
    assert segments == ["summarize the filing", "draft a note to counsel", "update the tracker"]


def test_empty_text_returns_single_empty_segment():
    assert segment_request("   ") == [""]


def test_oxford_comma_list_splits_into_three():
    segments = segment_request("summarize this brief, send it to the judge, and get lunch")
    assert segments == ["summarize this brief", "send it to the judge", "get lunch"]


def test_comma_list_without_trailing_and_splits():
    segments = segment_request("pull the numbers, format them, post them")
    assert segments == ["pull the numbers", "format them", "post them"]


def test_single_comma_does_not_split():
    # Below the 2-comma threshold -- stays conservative.
    text = "well, let's get started on this"
    assert segment_request(text) == [text]


def test_two_incidental_commas_still_split_known_tradeoff():
    # Documents the accepted trade-off explicitly: this is ONE task with
    # two comma-set-off clauses, not a three-item list, but the 2+ comma
    # heuristic can't tell the difference and splits it anyway. This test
    # exists so that behavior is a visible, intentional choice rather
    # than a silent surprise if it's ever "fixed" without noticing the
    # cost on the other side (see segmentation.py's module docstring).
    segments = segment_request("summarize this, in plain English, for the board")
    assert segments == ["summarize this", "in plain English", "for the board"]
