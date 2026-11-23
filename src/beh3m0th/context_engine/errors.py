"""Context-engine failures."""

from __future__ import annotations


class ContextBudgetExceededError(Exception):
    """Mandatory context does not fit in the available input budget."""

    def __init__(self, budget: int, mandatory_tokens: int, message: str | None = None) -> None:
        self.budget = budget
        self.mandatory_tokens = mandatory_tokens
        detail = message or (
            f"Mandatory context uses {mandatory_tokens} tokens, "
            f"which exceeds the available budget of {budget}"
        )
        super().__init__(detail)
