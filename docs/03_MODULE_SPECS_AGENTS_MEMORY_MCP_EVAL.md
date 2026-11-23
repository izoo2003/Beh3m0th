# Beh3m0th Engineering Specs: Projects 6, 7, 8 & 9

This document provides Cursor with exact engineering specs, protocols, schemas, and zero-cost-friendly design patterns for Beh3m0th's **Custom MCP Tool Server**, **Three-Layer Memory**, **Typed Multi-Agent Orchestrator**, and **Trajectory Evaluation Pipeline**.

---

## Project 6: Custom MCP Tool Server (`src/beh3m0th/mcp_server/`)

### Goal
Build an **SDK-independent** Model Context Protocol (MCP) server from scratch to prove deep understanding of how AI tools communicate, validate schemas, enforce permissions, and execute untrusted code safely.

### Core Requirements
1. **Raw JSON-RPC 2.0 Implementation (No Official MCP SDK):**
   - Implement JSON-RPC 2.0 manually (`jsonrpc: "2.0"`, `id`, `method`, `params`, `result`, `error` with standard codes `-32700`, `-32600`, `-32601`, `-32602`, `-32603`, plus custom auth error `-32001`).
   - Support core MCP lifecycle & tool methods:
     - `initialize` (protocol version & server capabilities negotiation)
     - `tools/list` (returns auto-generated JSON Schemas from Pydantic input models)
     - `tools/call` (validates arguments, checks RBAC permissions, runs tool, returns MCP content blocks)
     - `ping`
   - Support dual transports:
     - **`stdio`** (newline-delimited JSON-RPC over stdin/stdout—can even be plugged into Cursor/Claude Desktop directly!)
     - **`HTTP + SSE`** (FastAPI endpoints `/mcp/rpc` and `/mcp/sse` for internal agent calls)
2. **Strict Pydantic Validation:**
   - Every tool defines a Pydantic `BaseModel` input schema and output schema.
   - Tool registry auto-exports `input_model.model_json_schema()` for `tools/list` and validates incoming `params.arguments` via `model_validate`, returning JSON-RPC `-32602 Invalid Params` with field-level errors on failure.
3. **Permission Management (Per-Agent RBAC):**
   - Define capability scopes: `READ_FS`, `SEARCH_CODE`, `WRITE_FS`, `RUN_SANDBOX_CODE`, `RUN_TESTS`.
   - Every `tools/call` carries caller role/scoped token:
     - `Researcher`: `READ_FS | SEARCH_CODE`
     - `Coder`: `READ_FS | SEARCH_CODE | WRITE_FS | RUN_SANDBOX_CODE | RUN_TESTS`
     - `Reviewer`: `READ_FS | SEARCH_CODE | RUN_TESTS`
   - Enforce strict workspace path jail (`Path.resolve()` checks rejecting any `..` symlink/path escape outside `MCP_ALLOWED_WORKSPACE`).
4. **Sandboxed Execution ($0 Local Docker + Safe Fallback):**
   - Primary backend (`docker`): launches an ephemeral container from `docker/sandbox.Dockerfile` (`python:3.12-slim`) with:
     - `--network=none` (zero network access)
     - `--read-only` root filesystem + small `/tmp` tmpfs
     - `--memory=256m`, `--cpus=0.5`, `--pids-limit=64`
     - Hard timeout (e.g., 15s) with guaranteed cleanup (`finally: container.kill()`).
   - Fallback backend (`subprocess_restricted`, for laptops without Docker running): executes in an isolated temp directory with stripped environment variables and strict `timeout` (explicitly logged as `SANDBOX_MODE=fallback`).
5. **Custom MCP Client (`client.py`):**
   - Thin async client used by Project 8 agents to discover and invoke tools over HTTP or stdio cleanly.

### Starter Tools to Implement
- `read_file` (`READ_FS`)
- `list_directory` (`READ_FS`)
- `search_files_grep` (`SEARCH_CODE`)
- `write_file_patch` (`WRITE_FS`)
- `run_python_sandbox` (`RUN_SANDBOX_CODE`)
- `run_pytest_sandbox` (`RUN_TESTS`)

---

## Project 7: Three-Layer Agent Memory (`src/beh3m0th/memory/`)

### Goal
Give Beh3m0th persistent, structured long-term memory across sessions using **$0 embedded storage** (SQLite + in-memory/disk TTL) by default, with plug-and-play upgrade to PostgreSQL+`pgvector` and Redis via repository interfaces.

### Core Requirements
1. **Layer 1 — Working Memory (`working.py`):**
   - Stores active session scratchpad: current objective, plan steps, active file diffs, recent turns, intermediate hypotheses.
   - `WorkingMemoryProtocol` implemented by `InMemoryTTLWorkingStore` (Profile A, $0, persists snapshots to SQLite) and `RedisWorkingStore` (Profile B).
2. **Layer 2 — Episodic Memory (`episodic.py`):**
   - Stores timestamped records of completed tasks, debugging episodes, tool failures, and reviewer feedback:
     `(id, project_id, session_id, summary, outcome, tags, importance_score, embedding, created_at, last_accessed_at)`.
   - Profile A stores embeddings as compact float32 blobs in SQLite and computes vectorized cosine similarity via `numpy` (sub-10ms up to 50k memories, $0!). Profile B uses PostgreSQL `pgvector` HNSW index.
3. **Layer 3 — Semantic Memory (`semantic.py`):**
   - Stores distilled, timeless facts, repo conventions, and user preferences (e.g., *"Beh3m0th uses SQLAlchemy 2.0 async and prohibits sync DB calls"*).
   - Supports **upsert & conflict resolution**: before inserting a new semantic fact, check cosine similarity $> 0.88$ against existing facts; if similar, merge or supersede the old fact with `supersedes_id` and `updated_at`.
4. **Importance Scoring & Composite Retrieval Ranking:**
   - On write: compute raw importance $I \in [0, 1]$ via deterministic heuristics (e.g., bug fix verified by tests = $0.85$, conflict resolution = $0.90$, routine lookup = $0.35$) with optional `FAST_MODEL` refinement to save free-tier calls.
   - On recall: rank candidate memories using a weighted composite of semantic similarity, importance, and exponential time decay:
     $$S(m, q) = w_{\text{rel}} \cdot \cos(\mathbf{e}_q, \mathbf{e}_m) + w_{\text{imp}} \cdot I_m + w_{\text{rec}} \cdot e^{-\lambda \Delta t_m}$$
     where $\Delta t_m$ is hours since `last_accessed_at` and $\lambda$ is the decay constant (e.g., $\lambda = 0.005$).
5. **Cross-Session Persistence & Consolidation (`consolidator.py`):**
   - At the end of every workflow run, run a single batched consolidation call (`FAST_MODEL`) to:
     1. Summarize Working Memory into 1+ structured **Episodic** entries.
     2. Extract reusable long-term lessons into **Semantic** memory (with deduplication/superseding).
     3. Persist across sessions scoped by `(user_id, project_id)`.

---

## Project 8: Typed Multi-Agent System (`src/beh3m0th/agents/`)

### Goal
Coordinate specialized agents (**Supervisor**, **Researcher**, **Coder**, **Reviewer**) using strongly typed state transitions, consensus voting, and deterministic conflict resolution—designed to be **token-frugal** so complete multi-agent runs work smoothly on free API tiers.

### Core Requirements
1. **Strictly Typed State Passing (`state.py`):**
   - Define a versioned Pydantic `WorkflowState` (`ConfigDict(extra="forbid")`):
     - `run_id`, `session_id`, `project_id`, `objective`
     - `plan: list[TaskStep]`
     - `research_findings: list[ResearchArtifact]` (typed citations + confidence from Project 4)
     - `code_proposals: list[CodeProposal]` (file paths, unified diffs, sandbox test outputs)
     - `review_ballots: list[ReviewBallot]` (reviewer persona ID, `APPROVE | REQUEST_CHANGES | REJECT`, severity-tagged findings, confidence)
     - `conflicts: list[ConflictRecord]`
     - `iteration_count: int`, `max_iterations: int` (default `3` to cap latency & free-tier calls)
     - `status: Literal["PLANNING", "RESEARCHING", "CODING", "REVIEWING", "RESOLVING_CONFLICT", "COMPLETED", "ABSTAINED", "FAILED"]`
   - Every agent implements: `async def run(self, state: WorkflowState) -> AgentOutput`.
   - Every LLM call inside every agent MUST pass through **Project 1 (`ContextAssembler`)**.
2. **Specialized Agent Roles:**
   - **Supervisor (`supervisor.py`):** Decomposes objective into typed tasks, dispatches to Researcher/Coder/Reviewer, enforces iteration/token budgets, arbitrates conflicts, and produces the final structured report.
   - **Researcher (`researcher.py`):** Uses Project 4 (`SelfCorrectingRAG`), Project 7 (`Memory`), and read-only Project 6 MCP tools to produce grounded `ResearchArtifact`s. Never writes files.
   - **Coder (`coder.py`):** Consumes verified `ResearchArtifact`s, applies patches via Project 6 MCP (`write_file_patch`), executes tests in sandbox (`run_pytest_sandbox`), and emits a `CodeProposal`.
   - **Reviewer (`reviewer.py`):** Inspects `CodeProposal` against requirements, citations, edge cases, and sandbox test logs; runs read-only checks and emits structured `ReviewBallot`s.
3. **Consensus Voting (`consensus.py`):**
   - Before accepting a `CodeProposal` (or high-stakes answer), run a 3-perspective Review Panel:
     1. *Deterministic Test & Static Gate* ($0 — checks sandbox pytest exit code + syntax/ruff checks -> produces a typed ballot with 0 LLM tokens!)
     2. *Correctness & Edge-Case Reviewer* (`REASONING_MODEL`)
     3. *Security & Grounding Reviewer* (`JUDGE_MODEL` or distinct prompt persona)
   - Compute weighted consensus score:
     $$C = \frac{\sum_{i=1}^{N} w_i \cdot v_i \cdot c_i}{\sum_{i=1}^{N} w_i \cdot c_i}$$
     where $v_i \in \{+1, 0, -1\}$ is ballot verdict (`APPROVE`, `REQUEST_CHANGES`, `REJECT`), $w_i$ is weight, and $c_i \in (0, 1]$ is confidence.
   - Pass only if $C \ge \tau_{\text{consensus}}$ AND zero `CRITICAL` severity blockers exist.
4. **Conflict Resolution (`conflict.py`):**
   - If Coder and Reviewer deadlock (e.g., `REQUEST_CHANGES` for $\ge 2$ rounds on the same issue, or split Reviewer panel):
     1. Record a structured `ConflictRecord` with Coder's defense and Reviewer's objection + cited evidence.
     2. Escalate to **Supervisor Arbitration**: Supervisor inspects raw sandbox test logs + RAG citations and issues a binding `ArbitrationDecision` (`ACCEPT_CODER`, `ENFORCE_REVIEWER`, `REQUEST_NEW_RESEARCH`, or `ESCALATE_ABSTAIN`).

---

## Project 9: Agent Trajectory Evaluation Pipeline (`src/beh3m0th/evaluation/`)

### Goal
Evaluate the **entire multi-agent workflow trajectory**—not just the final answer—and automatically detect regressions in GitHub Actions CI for **$0**.

### Core Requirements
1. **Step-by-Step Trajectory Capture (`tracer.py`):**
   - Record every event as an ordered `TrajectoryStep` stored in SQLite/Postgres and exportable as `.jsonl`:
     - `step_index`, `timestamp`, `agent_name`, `step_type` (`CONTEXT_ASSEMBLE`, `RETRIEVAL`, `MEMORY_RECALL`, `MCP_TOOL_CALL`, `AGENT_DECISION`, `CONSENSUS_VOTE`, `CONFLICT_ARBITRATION`, `FINAL_OUTPUT`)
     - `input_summary`, `output_summary`, `tokens_used`, `latency_ms`, `metadata`.
2. **Golden Workflow Comparison (`golden.py`):**
   - Define golden test cases in `evals/golden_workflows/*.yaml` specifying:
     - `task_prompt`, `fixture_repo_or_docs`
     - `expected_agent_sequence` / required milestones
     - `expected_tools_called` (required & forbidden tools)
     - `expected_citations` / `expected_abstention: bool`
     - `max_steps`, `max_tokens`.
   - Compute deterministic ($0) trajectory metrics:
     - **Tool Precision & Recall**
     - **Milestone Coverage** (fraction of golden milestones achieved in valid order)
     - **Efficiency Ratio** ($\min(1.0, \text{golden\_steps} / \text{actual\_steps})$).
3. **Step-by-Step LLM-as-a-Judge (`judge.py`):**
   - Uses `JUDGE_MODEL` (e.g., free Groq Llama-3.3-70B, or deterministic stub in offline CI mode) with strict Pydantic output schemas to grade trajectories on 5 rubrics (0.0 to 1.0 + rationale):
     1. `retrieval_relevance_and_grounding`
     2. `context_and_memory_utilization`
     3. `tool_usage_correctness`
     4. `multi_agent_coordination_and_review`
     5. `final_answer_correctness`
4. **Regression Detection (`regression.py`):**
   - Compare current run aggregate metrics against `evals/baselines/main_baseline.json`.
   - Fail (exit code `1`) if:
     - Any core metric drops by more than threshold $\delta$ (e.g., $\Delta < -0.05$), OR
     - Hallucination rate rises above cap, OR
     - Abstention accuracy on unanswerable traps fails.
5. **Zero-Cost CI Pipeline (`.github/workflows/agent-eval-ci.yml`):**
   - On every PR: run `ruff`, `mypy`, unit tests, and **deterministic golden trajectory checks** (0 API keys needed, $0).
   - On manual dispatch or labeled PRs (`run-live-evals`): run full LLM-as-a-Judge eval suite using free-tier repository secrets (`GEMINI_API_KEY` / `GROQ_API_KEY`) and print a Markdown scorecard summary.