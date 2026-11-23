"""Failures raised by the trajectory evaluation pipeline."""


class EvalError(Exception):
    """A golden file, judge payload, or eval mode broke the suite contract."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
