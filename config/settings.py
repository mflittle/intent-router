"""Governance-relevant configuration for the live (non-stub) classifier.

Kept as named constants, not inline literals, so the "sandboxed /
versioned / monitored" governance claims have a single, auditable place
to point to in the repo.
"""
from __future__ import annotations

# Versioned: a pinned Bedrock model ID, never a floating "latest" alias.
# Model version drift was a named risk of adopting open-weight models
# without a governance perimeter -- this constant is how the pipeline
# guards against it. Verified against AWS's own Bedrock model card
# (docs.aws.amazon.com/bedrock/latest/userguide/model-card-zai-glm-5.html)
# rather than assumed -- confirm this is still current before relying on
# it, since Bedrock model IDs and availability do change.
BEDROCK_MODEL_ID = "zai.glm-5"

# GLM 5 on Bedrock is available "In-Region" only (no cross-region
# inference) -- pick the region your actual AWS/Bedrock access is
# provisioned in. us-east-1 is Bedrock's most broadly-featured region as
# of this writing; confirm this matches your org's actual setup.
BEDROCK_REGION = "us-east-1"

# Sandboxed: the IAM role the classifier assumes should be scoped to
# bedrock:InvokeModel (or Converse) on BEDROCK_MODEL_ID only -- no
# broader network or data access than the classifier strictly needs.
# FINRA-specific -- fill in once provisioned.
CLASSIFIER_IAM_ROLE_ARN = "REPLACE_WITH_SCOPED_INVOKE_ONLY_ROLE_ARN"

# Monitored: the durable decision log lives in router/decision_log.py
# (default logs/decisions.jsonl, overridable via INTENT_ROUTER_LOG_PATH
# or the log_path parameter on route()/route_request()). Point this at
# a real sink (CloudWatch, a table) for production use.
DECISION_LOG_DESTINATION = "REPLACE_WITH_PRODUCTION_LOG_SINK"

# How many times the live classifier retries on an invalid/unparseable
# response before giving up and letting the orchestrator's fail-safe
# (cautious HIGH-tier default) take over. See router/bedrock_classifier.py.
CLASSIFIER_MAX_RETRIES = 1
