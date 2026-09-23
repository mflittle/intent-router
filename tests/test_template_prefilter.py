from router.template_prefilter import match_template, register_template, registered_operations


def test_no_operation_falls_through():
    assert match_template({"text": "draft an email"}) is None


def test_unregistered_operation_falls_through():
    assert match_template({"operation": "not_registered_anywhere"}) is None


def test_registered_operation_matches():
    @register_template("uppercase_test_only")
    def _handler(request):
        return request["text"].upper()

    match = match_template({"operation": "uppercase_test_only", "text": "hello"})
    assert match is not None
    assert match.output == "HELLO"
    assert "uppercase_test_only" in registered_operations()
