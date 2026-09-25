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
# via boto3's bedrock-runtime Converse API -- same pattern as GLM.
#
# IMPORTANT: these MUST be cross-region INFERENCE PROFILE IDs, not bare
# model IDs. Verified live (2026-09-25) via AWS CLI: a bare model ID
# ("anthropic.claude-haiku-4-5-20251001-v1:0") fails with
# "on-demand throughput isn't supported"; the "us." inference-profile
# form succeeds. This is a genuine Bedrock requirement for these models
# on Converse -- confirmed against a live endpoint, not assumed from docs
# (Anthropic's own docs claimed bare IDs were sufficient; they were wrong
# for this account/path).
MODERATE_MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"

# HIGH_MODEL_ID -- resolved 2026-09-25 by testing candidates live, not
# guessing. Claude Sonnet 5 and Opus 4.8 (the newest flagships) both
# returned AccessDeniedException "not available for this account" on
# this account regardless of model-ID form -- genuine account-tier
# entitlement, not something IAM/Marketplace permissions fix (the error
# itself points to AWS Sales, not a console setting). Claude Sonnet 4.6
# separately failed on an AWS Marketplace subscription error, distinct
# from the entitlement gate above. Claude Opus 4.5 and Sonnet 4.5 both
# invoked successfully -- Opus 4.5 chosen for HIGH: a genuine frontier
# model, live on this account today, reserved for real high-stakes work
# behind the human-in-the-loop gate below.
HIGH_MODEL_ID = "us.anthropic.claude-opus-4-5-20251101-v1:0"

# HIGH tier's human-in-the-loop is an enforced gate, not just a label:
# router/execution.py does not auto-invoke HIGH_MODEL_ID by default. See
# execute_request()'s force_execute parameter for the explicit override.
