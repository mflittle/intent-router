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
