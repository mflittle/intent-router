"""Stage 3: task execution -- the `Executor` interface.

Everything before this stage (Stages -1 through 2) only decides WHAT
should happen to a request; nothing so far has actually performed the
task. This module defines the interface every tier-specific executor
implements, deliberately separate from `Classifier` -- classification
and execution are different jobs even when, for the LOW tier, they
happen to run on the same underlying model (GLM).
"""
from __future__ import annotations

from typing import Protocol


class Executor(Protocol):
    model_id: str

    def execute(self, task_text: str) -> str:
        """Actually perform the task and return the generated output."""
        ...
