"""Display models shared by the offline demos and the Streamlit pages."""

from __future__ import annotations

from pydantic import Field

from beh3m0th.core.schemas import StrictModel


class DemoResult(StrictModel):
    """One page result: a summary, a table, and optional trajectory lines."""

    title: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    rows: list[dict[str, str]] = Field(default_factory=list)
    steps: list[str] = Field(default_factory=list)
    detail: str = ""
