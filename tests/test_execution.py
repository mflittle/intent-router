from router.classifier import StubClassifier
from router.execution import execute_request
from router.orchestrator import route_request
from router.schema import Tier


class _PoisonExecutor:
    """Fails the test if invoked -- used to prove a tier's executor is
    never called when it shouldn't be (TRIVIAL, and HIGH without force).
    """

    model_id = "poison"

    def execute(self, task_text: str) -> str:
        raise AssertionError(f"executor should not have been invoked for: {task_text!r}")


class _EchoExecutor:
    model_id = "echo-test-model"

    def execute(self, task_text: str) -> str:
        return f"echoed: {task_text}"


def _route(text, tmp_path):
    return route_request(text, StubClassifier(), log_path=tmp_path / "d.jsonl")


def test_trivial_segment_passes_through_output_without_executor(tmp_path):
    result = _route("get lunch", tmp_path)
    assert result.segments[0].tier == Tier.TRIVIAL

    summary = execute_request(result, executors={Tier.LOW: _PoisonExecutor(), Tier.MODERATE: _PoisonExecutor(), Tier.HIGH: _PoisonExecutor()})

    assert len(summary.results) == 1
    exec_result = summary.results[0]
    assert exec_result.output == result.segments[0].output
    assert exec_result.executor_model_id == "none"
    assert not exec_result.human_review_required


def test_low_tier_invokes_low_executor(tmp_path):
    result = _route("draft a routine internal status update email", tmp_path)
    assert result.segments[0].tier == Tier.LOW

    summary = execute_request(result, executors={Tier.LOW: _EchoExecutor(), Tier.MODERATE: _PoisonExecutor(), Tier.HIGH: _PoisonExecutor()})

    exec_result = summary.results[0]
    assert exec_result.output == "echoed: draft a routine internal status update email"
    assert exec_result.executor_model_id == "echo-test-model"


def test_high_tier_is_not_auto_executed_by_default(tmp_path):
    result = _route(
        "summarize this exam finding for external counsel and the regulator, covering multiple accounts",
        tmp_path,
    )
    assert result.segments[0].tier == Tier.HIGH

    summary = execute_request(result, executors={Tier.LOW: _PoisonExecutor(), Tier.MODERATE: _PoisonExecutor(), Tier.HIGH: _PoisonExecutor()})

    exec_result = summary.results[0]
    assert exec_result.human_review_required is True
    assert exec_result.output is None
    assert exec_result.executor_model_id == "poison"  # identifies which model WOULD run, without running it


def test_high_tier_executes_when_forced(tmp_path):
    result = _route(
        "summarize this exam finding for external counsel and the regulator, covering multiple accounts",
        tmp_path,
    )
    assert result.segments[0].tier == Tier.HIGH

    summary = execute_request(
        result,
        executors={Tier.LOW: _PoisonExecutor(), Tier.MODERATE: _PoisonExecutor(), Tier.HIGH: _EchoExecutor()},
        force_execute_high=True,
    )

    exec_result = summary.results[0]
    assert exec_result.human_review_required is False
    assert exec_result.output.startswith("echoed:")


def test_missing_executor_for_tier_raises(tmp_path):
    result = _route("draft a routine internal status update email", tmp_path)
    assert result.segments[0].tier == Tier.LOW

    try:
        execute_request(result, executors={Tier.MODERATE: _EchoExecutor(), Tier.HIGH: _EchoExecutor()})
        assert False, "expected ValueError for missing LOW executor"
    except ValueError as e:
        assert "low" in str(e)
