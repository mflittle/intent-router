"""Governance-relevant configuration for the live (non-stub) classifier.

Placeholder values only -- filled in when the Bedrock-backed GLM
classifier replaces StubClassifier. Kept as named constants, not inline
literals, so the "sandboxed / versioned / monitored" governance claims
have a single, auditable place to point to in the repo.
"""
from __future__ import annotations

# Versioned: pin an exact Bedrock model version string, never "latest".
# Model version drift was a named risk of adopting open-weight models
# without a governance perimeter -- this constant is how the pipeline
# guards against it.
BEDROCK_MODEL_ID = "REPLACE_WITH_PINNED_GLM_MODEL_VERSION"

# Sandboxed: the IAM role the classifier assumes should be scoped to
# bedrock:InvokeModel on BEDROCK_MODEL_ID only -- no broader network or
# data access than the classifier strictly needs.
CLASSIFIER_IAM_ROLE_ARN = "REPLACE_WITH_SCOPED_INVOKE_ONLY_ROLE_ARN"

# Monitored: where structured routing-decision logs land. The orchestrator
# logs via the standard `logging` module under the "intent_router" name;
# point a handler at this destination (CloudWatch, a file, a table) rather
# than changing orchestrator.py.
DECISION_LOG_DESTINATION = "REPLACE_WITH_LOG_SINK"
