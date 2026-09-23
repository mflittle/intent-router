"""Persistent, append-only record of every routing decision.

This is the actual governance/tuning artifact. Python's `logging` module
alone (as used elsewhere in the orchestrator for real-time visibility)
has no sink configured by default -- nothing durable is written unless a
handler is attached, so relying on it alone would mean the "monitored"
claim in this project's governance story isn't backed by anything. This
module is the durable sink: one JSON line per decision, append-only,
ready to be queried for audits or used as eval/tuning data later (e.g.
"how often does low confidence trigger escalation", "what's the real
distribution of tiers seen in production" to re-check the confidence
threshold and the escalation rules against real traffic).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from .schema import RouteDecision

DEFAULT_LOG_PATH = Path(os.environ.get("INTENT_ROUTER_LOG_PATH", "logs/decisions.jsonl"))


def decision_to_dict(decision: RouteDecision, request_id: str) -> dict:
    return {
        "request_id": request_id,
        "timestamp": decision.timestamp,
        "stage": decision.stage,
        "segment_text": decision.segment_text,
        "intent_category": decision.intent_category.value,
        "tier": decision.tier.value,
        "tier_reason": decision.tier_reason,
        "route": decision.route,
        "confidence": decision.confidence,
        "model_version": decision.model_version,
        "signals": {
            "external_facing": decision.signals.external_facing,
            "legal_compliance_exposure": decision.signals.legal_compliance_exposure,
            "irreversible_or_costly_error": decision.signals.irreversible_or_costly_error,
            "multi_step_reasoning": decision.signals.multi_step_reasoning,
            "domain_expertise_required": decision.signals.domain_expertise_required,
        },
    }


def append_decision(decision: RouteDecision, request_id: str, log_path: Optional[Path] = None) -> dict:
    """Append one decision as a JSON line. Returns the record written, so
    callers (and tests) can assert on it without re-reading the file.
    """
    path = log_path or DEFAULT_LOG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    record = decision_to_dict(decision, request_id)
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")
    return record
