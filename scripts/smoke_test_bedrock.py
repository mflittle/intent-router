#!/usr/bin/env python3
"""Smoke test: confirms BedrockGLMClassifier can actually reach GLM on
Bedrock with your real AWS credentials, end to end.

Run this yourself, locally, where `aws configure` has already set up
credentials boto3's default credential chain can find:

    pip install -r requirements.txt
    python3 scripts/smoke_test_bedrock.py

This deliberately does NOT run as part of the pytest suite -- it makes a
real, billed API call and requires real AWS credentials, neither of
which belong in an automated test run.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from router.bedrock_classifier import BedrockGLMClassifier, ClassifierInvocationError

SAMPLE_REQUESTS = [
    "draft a routine internal status update email to my team",
    "summarize this exam finding for external counsel and the regulator, covering multiple customer accounts",
    "what is 2 plus 2",
]


def main() -> int:
    classifier = BedrockGLMClassifier()  # real boto3 client, config.settings defaults
    print(f"Testing {classifier.model_id} in {classifier.region} ...\n")

    exit_code = 0
    for text in SAMPLE_REQUESTS:
        print(f"Request: {text!r}")
        try:
            output = classifier.classify(text)
            print(f"  intent_category = {output.intent_category.value}")
            print(f"  signals         = {output.signals}")
            print(f"  confidence      = {output.confidence}")
            print(f"  model_version   = {output.model_version}")
        except ClassifierInvocationError as e:
            print(f"  FAILED: {e}")
            exit_code = 1
        except Exception as e:  # credentials, network, permissions, etc.
            print(f"  ERROR (not a classifier validation failure -- check credentials/IAM/region): {e}")
            exit_code = 1
        print()

    if exit_code == 0:
        print("All requests classified successfully -- Bedrock access confirmed.")
    else:
        print("One or more requests failed -- see errors above.")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
