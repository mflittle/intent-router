"""Stage 0c (optional): retrieval-backed knowledge resolution.

Extension point for grounding QA-style questions in an internal
knowledge base (a RAG pipeline) when Stage 0b's exact-match registry
misses -- e.g. "who is the VP of X" when X isn't one of the handful of
names someone bothered to hand-register. Deliberately NOT implemented
here: wiring in a vector store and embedding model is its own project,
and committing to one inside a routing demo would be scope creep in the
same way building the live GLM classifier before validating Stage 2
would have been. NullKnowledgeResolver is the stub, same pattern as
StubClassifier -- swap it for a real implementation without touching the
router.

Two things worth being explicit about, because they're easy to blur
together with "no AI needed":

1. A knowledge-resolver match is NOT a zero-model-call bypass like Stage
   0a/0b. Retrieval (nearest-neighbor lookup) is close to deterministic,
   but turning retrieved context into an answer still needs a model
   call, even a cheap one. So a match here routes as a LOW tier decision
   -- logged and tiered like any other LOW-tier task, just grounded in
   retrieved context instead of open-ended generation, not skipped
   entirely.

2. Live internet search, for when even the internal knowledge base
   misses, is a materially different decision from RAG over your own
   documents: it means data leaving a controlled environment and an
   answer grounded in an unverified external source. This project does
   NOT wire that in as a default fallback when the knowledge resolver
   also misses -- that would silently hand a "low" open-weight tier a
   capability with a very different risk profile than what the tier was
   designed for. If/when it's wanted, it should be its own explicitly
   gated capability with its own signal (e.g. a future
   `requires_external_source` signal that forces at least MODERATE),
   not something bolted onto Stage 0c.
"""
from __future__ import annotations

from typing import Optional, Protocol

# Placeholder confidence for a knowledge-resolver match. A real RAG
# implementation should return its own retrieval confidence instead of
# this constant -- kept here, named, so it's visibly a stand-in.
KNOWLEDGE_RESOLVER_CONFIDENCE = 0.75


class KnowledgeResolver(Protocol):
    def resolve(self, question_text: str) -> Optional[str]:
        """Return a grounded answer if the internal knowledge base has a
        confident match, else None (falls through to the classifier).
        """
        ...


class NullKnowledgeResolver:
    """Default resolver: no knowledge base wired up. Always falls through."""

    def resolve(self, question_text: str) -> Optional[str]:
        return None
