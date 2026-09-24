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

# --- Stage 3 (execution): tier -> model bindings ---
#
# LOW stays on GLM (router/glm_executor.py, boto3 Converse API) -- same
# model as the classifier, consistent with the LOW tier's premise.
#
# MODERATE and HIGH stay on one governed vendor (Claude) rather than
# adding a third vendor relationship -- see README for the reasoning.
# They're deliberately different sizes within that one vendor: Haiku for
# MODERATE, Sonnet for HIGH. Both go through router/claude_executor.py,
# which uses the Anthropic Messages API via Bedrock (NOT boto3 Converse
# -- see that module's docstring for why, and the fact that this is
# UNVERIFIED against a live endpoint as of this writing).
MODERATE_MODEL_ID = "anthropic.claude-haiku-4-5"
HIGH_MODEL_ID = "anthropic.claude-sonnet-5"

# HIGH tier's human-in-the-loop is an enforced gate, not just a label:
# router/execution.py does not auto-invoke HIGH_MODEL_ID by default. See
# execute_request()'s force_execute parameter for the explicit override.
