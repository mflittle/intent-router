from router.fact_lookup import match_fact, register_fact, registered_questions


def test_unregistered_question_falls_through():
    assert match_fact("who is the CEO of a company that was never registered") is None


def test_registered_question_matches_exactly():
    @register_fact("who is the test vp", aliases=["who's the test vp"])
    def _answer():
        return "Test Person"

    match = match_fact("Who is the test VP?")
    assert match is not None
    assert match.answer == "Test Person"


def test_registered_alias_matches():
    @register_fact("what is the test policy number", aliases=["test policy number"])
    def _answer():
        return "TP-123"

    match = match_fact("test policy number")
    assert match is not None
    assert match.answer == "TP-123"
    assert "what is the test policy number" in registered_questions()
