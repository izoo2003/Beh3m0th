# Beh3m0th — Phased Implementation Roadmap (Cursor Build Guide)

**Golden Rule for Building with Cursor:** Never ask Cursor to generate all 8 projects at once. Build **one vertical slice at a time**, run `uv run pytest tests/unit` ($0, offline), commit to Git, and move to the next phase. Always attach the relevant blueprint files with `@docs/...` in Cursor Composer.

---

## Phase 0: Beh3m0th Skeleton, $0 LLM Gateway & Tracer
**Goal:** Bootable `beh3m0th` repo, $0-first config, cached & rate-limited LLM client (supporting Gemini Free / Groq Free / Ollama / FakeMock), and `TrajectoryTracer`.

### Suggested Cursor Prompt
> `@docs/00_PROJECT_VISION_AND_ARCHITECTURE.md @docs/01_TECH_STACK_AND_SETUP.md`
> Bootstrap Phase 0 of Beh3m0th:
> 1. Create `pyproject.toml` using `uv` with the lightweight dependencies (`fastembed`, `aiosqlite`, `qdrant-client`, `networkx`, `litellm`, `pydantic`, `tiktoken`, `structlog`, `typer`, `rich`).
> 2. Create `.env.example`, `src/beh3m0th/config.py` (defaulting to `STORAGE_PROFILE=embedded` and $0 free models), and `docker/sandbox.Dockerfile`.
> 3. Build `src/beh3m0th/llm/client.py`: async `LLMClient` protocol + `LiteLLMClient` with SQLite prompt response cache, async RPM rate limiter, exponential backoff, fallback model support, and structured Pydantic output parsing; plus a `FakeLLMClient` for offline unit tests.
> 4. Build `src/beh3m0th/evaluation/tracer.py` (`TrajectoryStep` + `TrajectoryTracer` exporting JSONL) so every future module logs structured steps from Day 1.
> 5. Add unit tests in `tests/unit/test_phase0_llm_and_config.py`.

### Verification Gate
- `uv run pytest tests/unit/test_phase0_llm_and_config.py` passes offline ($0).
- Cache test proves identical prompts hit SQLite cache and make 0 duplicate API calls.

---

## Phase 1: Project 1 — Context Assembler Engine
**Goal:** Deterministic, tested token budgeting and overflow-proof context assembly.

### Suggested Cursor Prompt
> `@docs/02_MODULE_SPECS_RETRIEVAL_AND_CONTEXT.md`
> Implement Project 1 (`src/beh3m0th/context_engine/`). Build `tokenizer.py`, `models.py`, and `assembler.py` with dynamic token budgeting ($W_{\text{model}} - R_{\text{output}} - M_{\text{safety}}$), priority buckets (P0-P4), water-filling surplus reallocation, zero-cost deterministic trimming/overflow prevention, and `ContextAssemblyReport` logging to `TrajectoryTracer`. Add unit tests in `tests/unit/test_context_assembler.py`.

### Verification Gate
- Assembled prompt NEVER exceeds available token budget.
- Surplus reallocation shifts unused memory tokens to retrieval evidence.

---

## Phase 2: Project 2 — Hybrid Retrieval & Reranking System ($0 Local CPU)
**Goal:** Ingest sample docs + Python code, run code-aware BM25 + Embedded Qdrant Vector RRF + metadata filtering + `fastembed` ONNX Cross-Encoder reranking.

### Suggested Cursor Prompt
> `@docs/01_TECH_STACK_AND_SETUP.md @docs/02_MODULE_SPECS_RETRIEVAL_AND_CONTEXT.md`
> Implement ingestion (`src/beh3m0th/retrieval/ingestion/`) for Markdown and Python AST chunks, and Project 2 (`src/beh3m0th/retrieval/hybrid/`): `fastembed` ONNX embedder (`BAAI/bge-small-en-v1.5`), code-aware BM25 index, Qdrant store (supporting local embedded `path=` by default), metadata pre-filtering, Reciprocal Rank Fusion, `fastembed` Cross-Encoder reranker (`Xenova/ms-marco-MiniLM-L-6-v2`), and latency telemetry. Add unit tests in `tests/unit/test_hybrid_retrieval.py`.

### Verification Gate
- Runs completely locally with $0 API cost and zero Docker containers required.
- Hybrid + reranked search beats BM25-only and dense-only on mixed symbol + conceptual test queries.

---

## Phase 3: Project 3 — GraphRAG Knowledge Pipeline
**Goal:** Zero-token Python AST entity/edge extraction + cached LLM doc extraction, NetworkX (and optional Neo4j) GraphStore, Leiden community summaries, and 1–3 hop traversal.

### Suggested Cursor Prompt
> `@docs/02_MODULE_SPECS_RETRIEVAL_AND_CONTEXT.md`
> Implement Project 3 (`src/beh3m0th/retrieval/graphrag/`). Build deterministic Python `ast` code extractor + cached LLM prose extractor, `GraphStoreProtocol` with `NetworkXGraphStore` (JSON persisted, default $0) and `Neo4jGraphStore` adapter, entity deduplication, Leiden community detection + summarizer (`igraph`/`leidenalg`), and `GraphRAGPipeline` supporting `local` (1-3 hop) and `global` (community) queries. Add unit tests in `tests/unit/test_graphrag.py`.

### Verification Gate
- Multi-hop query (`ServiceA -> calls -> AuthModule -> reads -> PolicyDoc`) returns the connected subgraph and source chunk IDs.

---

## Phase 4: Project 4 — Self-Correcting Agentic RAG
**Goal:** Wrap Projects 2 & 3 in iterative query rewriting, citation grounding, hallucination verification, and confidence-based abstention.

### Suggested Cursor Prompt
> `@docs/02_MODULE_SPECS_RETRIEVAL_AND_CONTEXT.md`
> Implement Project 4 (`src/beh3m0th/retrieval/agentic_rag/`). Build `rewriter.py`, `verifier.py`, and `engine.py` combining `HybridRetriever` and `GraphRAGPipeline` with bounded multi-step retrieval (`max_steps=3`), zero-cost reranker relevance gating, strict citation ID checking, claim-level hallucination scoring, and confidence-based abstention. Route all LLM prompts through `ContextAssembler` (Project 1). Add unit tests in `tests/unit/test_agentic_rag.py` including an unanswerable trap that MUST return `status="ABSTAINED"`.

### Verification Gate
- Valid query returns claims grounded in real chunk IDs.
- Unanswerable query reliably abstains with confidence $< \tau_{\text{abstain}}$.

---

## Phase 5: Project 6 — Custom SDK-Independent MCP Tool Server
**Goal:** Hand-built JSON-RPC 2.0 MCP server (`stdio` + `HTTP/SSE`) with Pydantic schemas, per-agent RBAC, path-traversal protection, and Docker sandbox execution.

### Suggested Cursor Prompt
> `@docs/03_MODULE_SPECS_AGENTS_MEMORY_MCP_EVAL.md`
> Implement Project 6 (`src/beh3m0th/mcp_server/`) WITHOUT any official MCP SDK. Build raw JSON-RPC 2.0 dispatcher, Pydantic tool registry (`initialize`, `tools/list`, `tools/call`, `ping`), per-agent RBAC permission checker with `Path.resolve()` jail, Docker sandbox runner (with `subprocess_restricted` fallback if Docker is unavailable), the 6 starter tools, `stdio` + FastAPI `HTTP/SSE` transports, and async `MCPClient`. Add security & protocol tests in `tests/unit/test_mcp_server.py`.

### Verification Gate
- `Researcher` role is rejected with JSON-RPC `-32001` on `write_file_patch` and `run_python_sandbox`.
- Path traversal (`../../etc/passwd`) is blocked with `-32001`.

---

## Phase 6: Project 7 — Three-Layer Agent Memory
**Goal:** Working (in-memory/disk TTL), Episodic (SQLite + numpy cosine / optional pgvector), and Semantic memory with importance scoring, decay ranking, and cross-session consolidation.

### Suggested Cursor Prompt
> `@docs/03_MODULE_SPECS_AGENTS_MEMORY_MCP_EVAL.md`
> Implement Project 7 (`src/beh3m0th/memory/`). Build `WorkingMemoryStore`, SQLite async `EpisodicMemoryStore` and `SemanticMemoryStore` (using `fastembed` vectors + numpy cosine search behind clean repository protocols so Postgres+pgvector plugs in later), composite relevance+importance+recency scoring, semantic deduplication/superseding, and `MemoryConsolidator`. Add cross-session persistence tests in `tests/unit/test_three_layer_memory.py`.

### Verification Gate
- Session 1 consolidates a discovered bug fix (Episodic) and repo rule (Semantic) -> Session 2 recalls both and ranks them by composite score.

---

## Phase 7: Project 8 — Typed Multi-Agent System
**Goal:** Wire Supervisor, Researcher, Coder, and Reviewer together with typed `WorkflowState`, Consensus Voting, Conflict Resolution, and a Rich CLI.

### Suggested Cursor Prompt
> `@docs/03_MODULE_SPECS_AGENTS_MEMORY_MCP_EVAL.md`
> Implement Project 8 (`src/beh3m0th/agents/`) and CLI (`src/beh3m0th/cli/app.py`). Build strict Pydantic `WorkflowState`, `ResearcherAgent`, `CoderAgent`, `ReviewerAgent`, weighted `ConsensusEngine` (including a $0 deterministic test-gate ballot), `ConflictResolver` with Supervisor arbitration, and `SupervisorOrchestrator`. Route every LLM call through `ContextAssembler` (Project 1) and record all steps via `TrajectoryTracer`. Add end-to-end tests in `tests/unit/test_multi_agent_workflow.py` (with `FakeLLMClient`) and a live smoke script using free Gemini/Groq/Ollama.

### Verification Gate
- Seeded mini-repo bug -> Researcher cites root cause -> Coder patches & passes sandboxed `pytest` -> Reviewer consensus approves -> Supervisor returns final report & consolidates memory.
- Forced disagreement triggers `ConflictRecord` and Supervisor `ArbitrationDecision`.

---

## Phase 8: Project 9 — Trajectory Evaluation Pipeline, FastAPI SSE & Free CI
**Goal:** Golden workflow comparison, LLM-as-a-judge, baseline regression gate, FastAPI SSE server, and $0 GitHub Actions CI.

### Suggested Cursor Prompt
> `@docs/03_MODULE_SPECS_AGENTS_MEMORY_MCP_EVAL.md`
> Implement Project 9 (`src/beh3m0th/evaluation/`), `evals/run_eval_suite.py`, 5 golden workflows in `evals/golden_workflows/` (multi-hop GraphRAG, code bug-fix + sandbox test, conflict arbitration, memory recall across sessions, and unanswerable abstention trap), baseline regression checker, `.github/workflows/agent-eval-ci.yml` (runs deterministic mode on every PR for $0, live judge mode on manual dispatch), and a thin FastAPI SSE server (`src/beh3m0th/api/main.py`).

### Verification Gate
- `uv run python evals/run_eval_suite.py --mode deterministic` exits `0` offline ($0).
- `uv run python evals/run_eval_suite.py --mode live` runs against your free Gemini/Groq/Ollama key and outputs a full scorecard.

---

## Phase 9 (Later, Post-Prototype): Free Cloud Deployment -> Paid API Upgrade
1. **Deploy Prototype for $0:** Push to GitHub -> deploy FastAPI backend to Render/Koyeb/HF Spaces free tier, swap `.env` to Supabase/Neon Free Postgres (`pgvector`), Qdrant Cloud Free (1GB), Neo4j AuraDB Free, Upstash Redis Free.
2. **After Salary (Paid Upgrade — 0 Code Changes):** Drop paid `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `GEMINI_API_KEY` into `.env`, update `FAST_MODEL`, `REASONING_MODEL`, `JUDGE_MODEL`, and set `LLM_MAX_RPM=500`. Beh3m0th instantly scales to frontier models!