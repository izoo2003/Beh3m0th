"""Offline demos used by the Streamlit pages."""

from beh3m0th.ui.demos_core import assemble_context, home_status, scorecard
from beh3m0th.ui.demos_retrieval import answer_question, query_graph, search_chunks
from beh3m0th.ui.demos_workflow import (
    call_tool_demo,
    recall_bugfix,
    run_abstaining_objective,
    run_seeded_fix,
)
from beh3m0th.ui.models import DemoResult

__all__ = [
    "DemoResult",
    "answer_question",
    "assemble_context",
    "call_tool_demo",
    "home_status",
    "query_graph",
    "recall_bugfix",
    "run_abstaining_objective",
    "run_seeded_fix",
    "scorecard",
    "search_chunks",
]
