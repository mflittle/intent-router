# intent-router

A reusable intent classification and model-tier routing component.

## Thesis

"AI ROI" is usually measured as one aggregate line (total spend vs. total
time saved), which is the wrong granularity. The real failure mode isn't
that agentic tools are bad for low-stakes tasks (drafting an email,
building a slide) -- it's that nobody classified the *intent* first to
know whether a frontier model was warranted at all. Most task volume in a
typical workflow doesn't need frontier-model capability, and routing it
there anyway is where the ROI story actually breaks down.

This project is the practical embodiment of that argument: a small,
reusable component that decides, before any expensive model call, what a
request actually needs.

## Architecture

Three stages, each independently testable:

```
request -> [Stage 0: template pre-filter] -- trivial? --> handled directly, no model call
                    |  no match
                    v
           [Stage 1: classifier] -- intent category + risk signals
                    |
                    v
           [Stage 2: tier policy] -- deterministic signals -> tier
                    |
                    v
           route to model tier, with a structured decision log
```

**Stage 0 -- `router/template_prefilter.py`**
A deterministic allowlist of known-safe, mechanically templatable
operations. Not a heuristic for "this looks simple" -- an exact-match
registry, because false positives here (treating a real task as trivial)
mean it skips model review entirely. Starts empty; grows only as real,
recurring trivial operations justify an entry.

**Stage 1 -- `router/classifier.py`**
Classifies intent into a small, fixed category set and extracts five
boolean risk signals (`external_facing`, `legal_compliance_exposure`,
`irreversible_or_costly_error`, `multi_step_reasoning`,
`domain_expertise_required`). Deliberately does **not** output a tier --
see below. Ships with `StubClassifier`, a keyword-based stand-in that
lets the router be built and tested before any live model call. The
intended production classifier is an open-weight model (GLM, chosen for
its permissive MIT license and agentic-planning strength) run on AWS
Bedrock, behind a governance wrapper:

- **Sandboxed** -- least-privilege IAM role scoped to `InvokeModel` on
  one pinned model only, no broader access.
- **Versioned** -- an exact, pinned Bedrock model version string, never
  `latest`; every decision log records which version handled it.
- **Monitored** -- every classification is logged (input, output,
  confidence, model version, timestamp).
- **Enforced output space** -- classifier output is validated against
  the fixed label set on the way out; anything outside it is
  rejected/retried rather than passed through. This is what makes the
  "bounded output space carries less misuse risk" argument a property of
  the system rather than an assertion about it -- see the note on
  alignment-stripping ("abliteration") research below.

Config placeholders for the live classifier live in `config/settings.py`.

**Stage 2 -- `router/tier_policy.py`**
A pure function: `ClassifierOutput -> (Tier, reason)`. No model call, no
randomness. Judgment (Stage 1) and policy (Stage 2) are deliberately
separate -- the escalation rules can change without retraining or
re-prompting the classifier, and a tier assignment is an auditable
function rather than an LLM's unexplained call.

Locked escalation rules:

| Condition | Tier |
|---|---|
| `intent_category == other` | `high` (cautious default) |
| 2+ hard signals present | `high` |
| 1 hard signal + `multi_step_reasoning` | `high` |
| exactly 1 hard signal | `moderate` |
| 0 hard signals | `low` |
| trivial (Stage 0 match) | `trivial` -- no model call at all |

"Hard signals" are `external_facing`, `legal_compliance_exposure`, and
`irreversible_or_costly_error`. Low-confidence classifications escalate
one tier (never down) -- see `CONFIDENCE_ESCALATION_THRESHOLD` in
`tier_policy.py`, an intentionally named, untuned config constant rather
than a hardcoded "discovered" number.

## Why open-weight models need a governance perimeter, not just cost math

The common pitch for open-weight models is "cheaper." The more complete
story is a tradeoff: self-hosting shifts savings into an operational
security and stability burden the enterprise now owns -- patching
cadence, model version drift, no vendor SLA, no managed guardrails layer.
Separately, published research (Qi et al. 2024, on "abliteration") shows
that open-weight safety alignment can be stripped quickly and cheaply --
this is a documented technique, not speculation. The resolution isn't
"don't use open-weight models," it's that they should sit behind the same
governance perimeter as anything else: sandboxed, versioned, monitored,
not adopted just because they're free.

The reason this project uses an open-weight model specifically for the
*classifier* component -- and not, say, for open-ended content generation
-- is the bounded-output-space distinction: a small, fixed label set
carries far less misuse risk than open-ended generation, which is exactly
the profile where an open-weight model's risk/governance-overhead
tradeoff is favorable.

## Status

Stages 0, 1 (stub), and 2 are implemented and tested. The live
Bedrock/GLM-backed classifier (replacing `StubClassifier`) and its
governance wrapper are the next milestone -- see `config/settings.py`.

## Running tests

```
pip install -r requirements.txt
pytest
```
