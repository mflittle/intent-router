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

Five stages, each independently testable. A request is segmented first,
then every segment runs the deterministic-first pipeline independently --
a compound request never gets one blended (and possibly wrong) decision.

```
request -> [Stage -1: segmentation] -- split into independent segments
                    |
                    v  (per segment)
           [Stage 0: scope gate] -- out of scope / non-actionable? --> rejected, notify user, no model call
                    |  in scope
                    v
           [Stage 0a: template pre-filter] -- templatable transform? --> handled directly, no model call
                    |  no match
                    v
           [Stage 0b: fact lookup] -- known fact? --> handled directly, no model call
                    |  no match
                    v
           [Stage 0c: knowledge resolver] -- internal knowledge base match? --> LOW tier, grounded answer
                    |  no match
                    v
           [Stage 1: classifier] -- intent category + risk signals
                    |
                    v
           [Stage 2: tier policy] -- deterministic signals -> tier
                    |
                    v
           route to model tier, with a durable, structured decision log
                    |
                    v
           [Stage 3: execution] -- actually perform the task at the routed tier
                    (TRIVIAL: passes through Stage 0's already-computed output
                     HIGH: gated -- not auto-executed without explicit override)

overall result = one RouteDecision (+ ExecutionResult once executed) per segment
```

**Stage -1 -- `router/segmentation.py`**
Deterministic, rule-based splitting of a compound request ("summarize
this filing and also draft a note to external counsel") into independent
segments, before any routing happens. Conservative by design: it only
splits on unambiguous separators (newlines, semicolons, explicit
connectors like "and also" / "and then") and never on a bare "and" --
"review and approve the memo" stays one segment. A false negative here
(a compound request treated as one segment) just means its signals get
combined and it may route a tier higher than strictly necessary -- safe.
A false positive (splitting one coherent request into unrelated pieces)
would misrepresent what was actually asked, which this stage is built to
avoid. Splitting itself needs no model call, which matters for the
project's own thesis: "no AI needed" can apply to detecting multiple
intents, not just to what happens after.

One deliberate exception to that conservatism: a serial comma list
("summarize this brief, send it to the judge, and get lunch") splits on
2+ commas. This is a real, accepted trade-off, not a free improvement --
a sentence using 2+ commas for some other grammatical reason
("summarize this, in plain English, for the board") gets incorrectly
split too (see `test_two_incidental_commas_still_split_known_tradeoff`
in `tests/test_segmentation.py`, which exists specifically so that cost
stays visible rather than being a silent surprise). It was judged worth
it because the alternative is worse in a common real case: without it,
an out-of-scope fragment (Stage 0's scope gate) anywhere in an unsplit
compound request causes the *entire* request to be rejected, even when
most of it was legitimate.

**Stage 0 -- `router/scope_gate.py`**
Runs first, before any other Stage 0 check. A deterministic denylist for
requests -- or fragments of a request -- that ask for something this
system was never built to do: a physical-world action ("get lunch",
"walk the dog"), or anything clearly outside its intent categories.
Without this stage, that kind of request falls to `IntentCategory.OTHER`
in the classifier, which Stage 2 treats as a cautious default -> `high`,
frontier-tier, human-in-the-loop -- the worst possible outcome for a
nonsensical request, since it spends the most expensive tier on
something no tier should attempt. Unlike Stage 0a/0b, this is a
substring match (an out-of-scope fragment is usually embedded in an
otherwise-legitimate sentence, not the whole segment), so the denylist
stays small and curated to keep false positives rare -- a false negative
here just falls through to the existing `OTHER` -> cautious-`high`
backstop, which remains the right catch-all for genuinely ambiguous
requests this stage doesn't (and shouldn't try to) catch.

**Stage 0a -- `router/template_prefilter.py`**
A deterministic allowlist of known-safe, mechanically templatable
operations. Not a heuristic for "this looks simple" -- an exact-match
registry, because false positives here (treating a real task as trivial)
mean it skips model review entirely. Starts empty; grows only as real,
recurring trivial operations justify an entry. Requires the caller to
supply an explicit `operation` key -- it never infers "templatable" from
free text.

**Stage 0b -- `router/fact_lookup.py`**
The other flavor of "no model needed": a known factual question with a
registrable answer ("who is the compliance officer"), as opposed to a
transformation to perform. Same allowlist discipline as Stage 0a -- an
exact, normalized match against a registry, never a heuristic guess that
a question "looks simple". Starts empty for the same reason.

**Stage 0c -- `router/knowledge_resolver.py`**
Extension point for grounding QA-style questions in an internal
knowledge base (a RAG pipeline) when Stage 0b's exact-match registry
misses -- e.g. a VP name that wasn't hand-registered. Deliberately
stubbed (`NullKnowledgeResolver`) rather than built out; wiring in a
real vector store is its own project. Important distinction from Stage
0a/0b: a knowledge-resolver match is NOT a zero-model-call bypass --
retrieval is close to deterministic, but turning retrieved context into
an answer still needs a (usually cheap) model call, so it routes `low`,
not `trivial`. Also deliberately does not fall back to live internet
search when it misses -- that's a materially different risk profile
(external data egress, unverified sources) and would need its own
explicit signal/gating rather than being silently bolted onto a `low`
tier's default behavior.

**Stage 1 -- `router/classifier.py` (dev) / `router/bedrock_classifier.py` (live)**
Classifies intent into a small, fixed category set and extracts five
boolean risk signals (`external_facing`, `legal_compliance_exposure`,
`irreversible_or_costly_error`, `multi_step_reasoning`,
`domain_expertise_required`). Deliberately does **not** output a tier --
see below.

`StubClassifier` (`router/classifier.py`) is a keyword-based stand-in
that let the router be built and tested before any live model call.

`BedrockGLMClassifier` (`router/bedrock_classifier.py`) is the live
implementation: GLM 5 on AWS Bedrock (model ID `zai.glm-5`, verified
against AWS's own Bedrock model card, not assumed -- Bedrock model IDs
and availability do change, so re-check before relying on this),
chosen as an open-weight model for its permissive MIT license and
agentic-planning strength, behind a governance wrapper:

- **Sandboxed** -- least-privilege IAM role scoped to `InvokeModel`/
  `Converse` on one pinned model only, no broader access. ARN placeholder
  in `config/settings.py`, FINRA-specific, fill in once provisioned.
- **Versioned** -- an exact, pinned Bedrock model ID, never a floating
  `latest` alias (`BEDROCK_MODEL_ID` in `config/settings.py`); every
  decision log entry records which version handled it.
- **Monitored** -- every classification is durably logged via
  `router/decision_log.py` (input, output, confidence, model version,
  timestamp, and a `request_id` tying multi-intent segments together).
- **Enforced output space** -- the response is parsed and validated
  against the fixed `IntentCategory` set and the `Signals` schema on the
  way out. An invalid or unparseable response is retried once with a
  corrective prompt (`CLASSIFIER_MAX_RETRIES` in `config/settings.py`);
  if it still doesn't validate, `BedrockGLMClassifier.classify()` raises
  rather than silently guessing or passing through unvalidated model
  output. This is what makes "bounded output space carries less misuse
  risk" an enforced property of the system rather than an assertion
  about it -- see the note on alignment-stripping ("abliteration")
  research below.
- **Fails safe, not closed** -- the orchestrator wraps every classifier
  call (`_classify_with_failsafe` in `orchestrator.py`). If the live
  classifier raises -- a network error, or exhausted retries on invalid
  output -- the segment is routed the same way an unclassifiable `OTHER`
  result is: cautious `HIGH` tier, human-in-the-loop, rather than
  crashing the pipeline or silently defaulting to something cheaper.

**A note on testing this without live AWS access:** `BedrockGLMClassifier`
accepts an injectable `client` (anything with a Bedrock Converse-API-shaped
`.converse()` method), so `tests/test_bedrock_classifier.py` verifies the
parsing/validation/retry logic entirely against a fake client -- `boto3`
is imported lazily and is never required unless a real client is actually
constructed. Wiring this up against real Bedrock requires valid AWS
credentials with the IAM permissions above, provisioned in your own
environment -- that hasn't been done yet.

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

Stages -1 through 2 (segmentation, scope gate, 0a/0b/0c, classifier, tier
policy) are implemented and tested. `BedrockGLMClassifier` (the live
Stage 1) has been verified against real Bedrock via
`scripts/smoke_test_bedrock.py` and `scripts/smoke_test_pipeline.py`.

**Stage 3 -- `router/execution.py`, `router/glm_executor.py`,
`router/claude_executor.py`** (new): completes the loop from
classify/route to actually performing the task.

- `Executor` (`router/executor.py`) is the interface -- deliberately
  separate from `Classifier`, since classification and execution are
  different jobs even when LOW tier happens to use the same model (GLM)
  for both.
- LOW tier executes on GLM (`GLMBedrockExecutor`), same boto3 Converse
  API path as the classifier.
- MODERATE and HIGH execute on Claude, kept on one governed vendor
  rather than adding a third -- Haiku for MODERATE, Sonnet for HIGH
  (`MODERATE_MODEL_ID`/`HIGH_MODEL_ID` in `config/settings.py`).
- **Important, unverified as of this writing:** `ClaudeBedrockExecutor`
  uses the Anthropic Messages API via the `anthropic` package's
  `AnthropicBedrockMantle` client, hitting a `bedrock-mantle` endpoint --
  NOT boto3's Converse API, which is what GLM uses. This is per current
  Anthropic documentation, but conflicts with one AWS model-card page
  that suggested Converse is also supported for Claude Haiku via an
  inference-profile ID. Verify with `scripts/smoke_test_execution.py`
  before relying on this -- see that script's docstring and
  `router/claude_executor.py`'s docstring for the full reasoning.
- **HIGH tier's human-in-the-loop is an enforced gate, not just a route
  label**: `execute_request()` does not auto-invoke the HIGH executor by
  default -- it returns a result with `human_review_required=True` and
  `output=None`, naming which model *would* run without running it. A
  caller passes `force_execute_high=True` to actually execute it,
  intended to happen only after a human has reviewed the routing
  decision (intent, signals, tier_reason) and decided to proceed.

61 unit tests pass, all without `boto3` or `anthropic` installed --
both are imported lazily and every executor/classifier accepts an
injectable fake client for testing, so the parsing/validation/dispatch
logic is fully covered without a live network call or either dependency
present.

Live-verified end to end (a personal AWS account, not FINRA's):
`scripts/smoke_test_bedrock.py` (Stage 1 only), `scripts/smoke_test_pipeline.py`
(Stages -1 through 2 end to end), and `scripts/smoke_test_execution.py`
(the full loop, Stage 3 included) -- all three make real, billed API
calls and are excluded from the pytest suite for that reason.

Multi-intent requests are handled: `router.route_request(text,
classifier)` segments the input and returns a `MultiRouteResult` with
one `RouteDecision` per segment plus an `overall_tier` (the max across
segments). `router.route(request, classifier)` remains available as a
single-segment convenience wrapper for callers that already know they
have exactly one intent (e.g. a pre-tagged template operation call).

Every routing decision is written durably to an append-only JSONL log
(`router/decision_log.py`, default `logs/decisions.jsonl`), keyed by a
`request_id` shared across all segments of one compound request -- this
is the actual data the tier-policy thresholds and the classifier itself
should eventually be tuned against, not just an audit trail.

## Running tests

```
pip install -r requirements.txt
pytest
```

## Verifying live Bedrock access

`pytest` never makes a real network call -- `tests/test_bedrock_classifier.py`
exercises `BedrockGLMClassifier` entirely against a fake, injected client.
To confirm real AWS/Bedrock access works end to end (after `aws configure`
locally with credentials scoped to `bedrock:InvokeModel`/`Converse` on
`zai.glm-5` -- see `config/settings.py`):

```
python3 scripts/smoke_test_bedrock.py
```

This makes real, billed API calls and is intentionally excluded from the
automated test suite for that reason.
