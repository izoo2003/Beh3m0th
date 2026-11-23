"""Streamlit pages for every finished Beh3m0th phase."""

from __future__ import annotations

import asyncio
from pathlib import Path

import streamlit as st

from beh3m0th.config import Settings
from beh3m0th.context_engine.errors import ContextBudgetExceededError
from beh3m0th.ui.demos_core import assemble_context, home_status, scorecard
from beh3m0th.ui.demos_retrieval import answer_question, query_graph, search_chunks
from beh3m0th.ui.demos_workflow import (
    call_tool_demo,
    recall_bugfix,
    run_abstaining_objective,
    run_seeded_fix,
)
from beh3m0th.ui.models import DemoResult

_ROOT = Path("data/ui")


def main() -> None:
    """Open the multipage dashboard. ``streamlit run`` calls this."""
    st.set_page_config(page_title="Beh3m0th", layout="wide")
    _sidebar()
    page = st.navigation(
        [
            st.Page(page_home, title="Home", default=True),
            st.Page(page_context, title="Context"),
            st.Page(page_search, title="Hybrid search"),
            st.Page(page_graph, title="Graph"),
            st.Page(page_answers, title="Grounded answers"),
            st.Page(page_tools, title="Tools"),
            st.Page(page_memory, title="Memory"),
            st.Page(page_agents, title="Agents"),
            st.Page(page_scorecard, title="Scorecard"),
        ]
    )
    page.run()


def page_home() -> None:
    result = home_status()
    st.header(result.title)
    st.write(
        "Try each phase from the sidebar. Every button runs on this machine "
        "with sample data, and none of them call a paid or free-tier model."
    )
    _show(result)


def page_context() -> None:
    st.header("Context budget")
    st.write("The assembler keeps the system block and gives leftover room to the evidence.")
    system = st.text_area("System", value="You are a careful engineering assistant.")
    notes = st.text_area("Notes", value="The pool size is 4.")
    evidence = st.text_area("Evidence", value="Retries fail when the pool is exhausted.")
    left, right = st.columns(2)
    if left.button("Assemble", type="primary"):
        _show(assemble_context(system, notes, evidence))
    if right.button("Force an overflow"):
        try:
            assemble_context(
                "token " * 800,
                notes,
                evidence,
                model_window=900,
                reserved_output=100,
                safety_margin=100,
            )
        except ContextBudgetExceededError as exc:
            st.error(str(exc))


def page_search() -> None:
    st.header("Hybrid search")
    st.write("BM25 and a local vector index share three sample notes. ONNX is optional.")
    query = st.text_input("Query", value="authentication retries pool")
    use_onnx = st.checkbox("Use the local ONNX embedder (downloads on first use)")
    if st.button("Search", type="primary"):
        if not query.strip():
            st.error("Type a query first.")
        else:
            _show(asyncio.run(search_chunks(query, _ROOT / "search", use_onnx=use_onnx)))


def page_graph() -> None:
    st.header("Graph")
    st.write("A short Python snippet is turned into a graph. The query walks two hops.")
    if st.button("Run the two-hop query", type="primary"):
        _show(asyncio.run(query_graph(_ROOT / "graph")))


def page_answers() -> None:
    st.header("Grounded answers")
    st.write("A cited note is answered. An off-topic note is refused.")
    left, right = st.columns(2)
    if left.button("Answer from the note", type="primary"):
        _show(asyncio.run(answer_question(answerable=True)))
    if right.button("Ask something the note cannot answer"):
        _show(asyncio.run(answer_question(answerable=False)))


def page_tools() -> None:
    st.header("Tools")
    st.write("The researcher may list and read. A write is rejected and the file stays put.")
    workspace = _ROOT / "tools"
    list_col, read_col, write_col = st.columns(3)
    if list_col.button("List the workspace"):
        _show(asyncio.run(call_tool_demo(workspace, "list")))
    if read_col.button("Read note.txt"):
        _show(asyncio.run(call_tool_demo(workspace, "read")))
    if write_col.button("Researcher tries to write", type="primary"):
        _show(asyncio.run(call_tool_demo(workspace, "write")))


def page_memory() -> None:
    st.header("Memory")
    st.write("Session 1 stores a bug fix. Session 2 asks for it and gets a ranked hit.")
    if st.button("Recall the bug fix", type="primary"):
        _show(asyncio.run(recall_bugfix(_ROOT / "memory")))


def page_agents() -> None:
    st.header("Agents")
    st.write("The seeded widget bug goes through every role. An uncited objective abstains.")
    left, right = st.columns(2)
    if left.button("Fix the seeded bug", type="primary"):
        _show(asyncio.run(run_seeded_fix(_ROOT / "agents")))
    if right.button("Run with nothing to cite"):
        _show(asyncio.run(run_abstaining_objective(_ROOT / "abstain")))


def page_scorecard() -> None:
    st.header("Scorecard")
    st.write("The golden trajectories are scored offline and compared with the saved baseline.")
    if st.button("Run the scorecard", type="primary"):
        _show(scorecard())


def _sidebar() -> None:
    settings = Settings.from_env()
    present = bool(settings.google_api_key or settings.groq_api_key or settings.openrouter_api_key)
    st.sidebar.header("Beh3m0th")
    st.sidebar.write("Offline demos are the default. Live model calls stay off.")
    st.sidebar.write(f"Free-tier key: {'present' if present else 'missing'}")
    st.sidebar.write(f"Profile: {settings.profile}")


def _show(result: DemoResult) -> None:
    st.subheader(result.title)
    if result.summary in {"passed", "ANSWERED", "COMPLETED", "allowed"}:
        st.success(result.summary)
    elif result.summary in {"ABSTAINED", "failed"}:
        st.warning(result.summary)
    else:
        st.write(result.summary)
    if result.rows:
        st.dataframe(result.rows, width="stretch")
    if result.detail:
        st.code(result.detail)
    if result.steps:
        st.write("Trajectory")
        for step in result.steps:
            st.text(step)


if __name__ == "__main__":
    main()
