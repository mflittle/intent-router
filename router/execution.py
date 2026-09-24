"""Stage 3: task execution.

Takes an already-routed MultiRouteResult (from orchestrator.route_request)
and actually performs each segment's task by invoking the
tier-appropriate executor -- completing the classify -> route -> execute
loop. Kept deliberately separate from routing: a caller can inspect the
routing decision (intent, signals, tier, tier_reason) and choose whether
to execute at all before doing so, which matters specifically for HIGH
tier's human-in-the-loop gate below.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Optional

from .executor import Executor
from .schema import ExecutionResult, ExecutionSummary, MultiRouteResult, RouteDecision, Tier


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_executors() -> Dict[Tier, Executor]:
    """Build the standard LOW/MODERATE/HIGH executors from
    config/settings.py's tier->model bindings. Convenience factory for
    scripts; tests should inject their own fakes instead.
    """
    from config.settings import HIGH_MODEL_ID, MODERATE_MODEL_ID
    from .claude_executor import ClaudeBedrockExecutor
    from .glm_executor import GLMBedrockExecutor

    return {
        Tier.LOW: GLMBedrockExecutor(),
        Tier.MODERATE: ClaudeBedrockExecutor(model_id=MODERATE_MODEL_ID),
        Tier.HIGH: ClaudeBedrockExecutor(model_id=HIGH_MODEL_ID),
    }


def execute_request(
    result: MultiRouteResult,
    executors: Dict[Tier, Executor],
    force_execute_high: bool = False,
) -> ExecutionSummary:
    """Execute every segment of an already-routed request.

    `executors` maps Tier -> Executor for LOW/MODERATE/HIGH (TRIVIAL never
    needs one -- its answer, or rejection message, is already on the
    RouteDecision from Stage 0). HIGH tier is NOT auto-executed unless
    force_execute_high=True -- the human-in-the-loop gate is enforced
    here, not just labeled in a route string.
    """
    results = tuple(
        _execute_segment(decision, executors, force_execute_high) for decision in result.segments
    )
    return ExecutionSummary(request_id=result.request_id, results=results, timestamp=_now_iso())


def _execute_segment(
    decision: RouteDecision,
    executors: Dict[Tier, Executor],
    force_execute_high: bool,
) -> ExecutionResult:
    if decision.tier == Tier.TRIVIAL:
        # Stage 0 already computed the answer (or rejection message) as
        # part of deciding the route -- nothing left to execute.
        return ExecutionResult(
            segment_text=decision.segment_text,
            tier=decision.tier,
            output=decision.output,
            executor_model_id="none",
            human_review_required=False,
            timestamp=_now_iso(),
        )

    if decision.tier == Tier.HIGH and not force_execute_high:
        high_executor = executors.get(Tier.HIGH)
        return ExecutionResult(
            segment_text=decision.segment_text,
            tier=decision.tier,
            output=None,
            executor_model_id=getattr(high_executor, "model_id", "unassigned"),
            human_review_required=True,
            timestamp=_now_iso(),
        )

    executor = executors.get(decision.tier)
    if executor is None:
        raise ValueError(f"No executor configured for tier '{decision.tier.value}'")

    output = executor.execute(decision.segment_text)
    return ExecutionResult(
        segment_text=decision.segment_text,
        tier=decision.tier,
        output=output,
        executor_model_id=getattr(executor, "model_id", "unknown"),
        human_review_required=False,
        timestamp=_now_iso(),
    )
