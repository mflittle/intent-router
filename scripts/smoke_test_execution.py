#!/usr/bin/env python3
"""End-to-end smoke test: classify -> route -> EXECUTE, against real
GLM and Claude models on Bedrock -- one request per tier (LOW/MODERATE
always run; HIGH only runs its executor with --force-high).

THIS IS WHERE THE UNVERIFIED CLAUDE-ON-BEDROCK PATH GETS TESTED FOR
REAL. router/claude_executor.py uses the Anthropic Messages API via the
AnthropicBedrockMantle client (per Anthropic's own docs), NOT boto3's
Converse API -- a genuinely different invocation path than GLM's, and
one that conflicting documentation made it worth verifying rather than
assuming. If this fails specifically on the MODERATE or --force-high
HIGH request (not the LOW one), that's the signal something about the
Claude-on-Bedrock path needs adjusting -- check the error and
router/claude_executor.py's docstring for context.

Run locally, after `aws configure`:

    pip install -r requirements.txt
    python3 scripts/smoke_test_execution.py                # LOW + MODERATE executed, HIGH gated
    python3 scripts/smoke_test_execution.py --force-high    # also executes the HIGH request for real

Makes real, billed API calls across up to three different models --
deliberately excluded from pytest.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from router.bedrock_classifier import BedrockGLMClassifier
from router.execution import default_executors, execute_request
from router.orchestrator import route_request
from router.schema import Tier

# One request chosen to land in each tier, based on prior live runs
# against StubClassifier/GLM (see tests/test_orchestrator.py and the
# earlier pipeline smoke test).
SAMPLE_REQUESTS = [
    ("LOW", "draft a one-sentence routine internal status update email to my team"),
    ("MODERATE", "draft a one-sentence email to a single external vendor confirming a meeting time"),
    ("HIGH", "summarize in one sentence: this exam finding covers a mutual fund rep's trading activity across multiple accounts, going to external counsel and the regulator"),
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force-high", action="store_true",
        help="Actually invoke the HIGH-tier executor (real Claude Sonnet call) instead of leaving it gated.",
    )
    args = parser.parse_args()

    classifier = BedrockGLMClassifier()
    executors = default_executors()
    print(f"LOW      -> {executors[Tier.LOW].model_id}")
    print(f"MODERATE -> {executors[Tier.MODERATE].model_id}")
    print(f"HIGH     -> {executors[Tier.HIGH].model_id}\n")

    for expected_tier, text in SAMPLE_REQUESTS:
        print(f"Request ({expected_tier} expected): {text!r}")
        result = route_request(text, classifier, log_path=Path("logs/execution_smoke_test.jsonl"))
        decision = result.segments[0]
        print(f"  routed: tier={decision.tier.value} reason={decision.tier_reason}")
        if decision.tier.value.upper() != expected_tier:
            print(f"  NOTE: expected {expected_tier} but got {decision.tier.value} -- GLM's live judgment differed, not necessarily wrong")

        try:
            summary = execute_request(result, executors, force_execute_high=args.force_high)
        except Exception as e:
            print(f"  EXECUTION FAILED: {e}")
            print(f"  (if this is the {decision.tier.value.upper()} request, check the Claude-on-Bedrock path -- see this script's docstring)")
            print()
            continue

        exec_result = summary.results[0]
        if exec_result.human_review_required:
            print(f"  HIGH tier -- human review required, NOT auto-executed (model would be: {exec_result.executor_model_id})")
            print(f"  Re-run with --force-high to actually execute it and verify the Claude Sonnet path.")
        else:
            print(f"  executed by {exec_result.executor_model_id}:")
            print(f"    {exec_result.output}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
