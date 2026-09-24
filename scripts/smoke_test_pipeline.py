#!/usr/bin/env python3
"""End-to-end smoke test: the full pipeline (segmentation -> scope gate
-> template/fact/knowledge -> live GLM classifier -> tier policy) against
real requests, using the live Bedrock-backed classifier.

Unlike scripts/smoke_test_bedrock.py (which only exercises Stage 1), this
confirms Stage 2's tier policy actually produces the right routing
decision when fed real GLM output, not just StubClassifier's keyword
heuristic -- the earlier test suite validated the tier policy logic in
isolation, but never against a live model's actual signal judgments.

Run locally, after `aws configure`:

    python3 scripts/smoke_test_pipeline.py

Makes real, billed API calls -- deliberately excluded from pytest.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from router.bedrock_classifier import BedrockGLMClassifier
from router.orchestrator import route_request

SAMPLE_REQUESTS = [
    "draft a routine internal status update email to my team",
    "summarize this exam finding for external counsel and the regulator, covering multiple customer accounts",
    "what is 2 plus 2",
    "summarize this brief, send it to the judge, and get lunch",  # scope gate should catch this
    "draft a routine internal email; summarize this exam finding for external counsel and the regulator",  # multi-intent
]


def main() -> int:
    classifier = BedrockGLMClassifier()
    print(f"Testing full pipeline against {classifier.model_id} ...\n")

    for text in SAMPLE_REQUESTS:
        print(f"Request: {text!r}")
        result = route_request(text, classifier, log_path=Path("logs/pipeline_smoke_test.jsonl"))
        for i, decision in enumerate(result.segments):
            label = f"  segment {i+1}/{len(result.segments)}" if len(result.segments) > 1 else "  "
            print(f"{label} stage={decision.stage} tier={decision.tier.value} route={decision.route}")
            print(f"      intent={decision.intent_category.value} reason={decision.tier_reason}")
        print(f"  overall_tier = {result.overall_tier.value}\n")

    print("Full-pipeline smoke test complete. Decisions logged to logs/pipeline_smoke_test.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
