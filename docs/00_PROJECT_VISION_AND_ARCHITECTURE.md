# Beh3m0th — Master Architecture & Unified Blueprint

## 1. Project Concept
**Beh3m0th** is an **Autonomous Research & Code Engineering Multi-Agent Platform** engineered to run **100% free ($0)** during prototyping and early deployment, while remaining **1-env-var upgradeable** to enterprise paid LLM APIs and managed cloud databases later.

Given a complex engineering request (e.g., *"Investigate why authentication retries fail under load across our docs and repo, propose an architectural fix, implement the patch, and verify it with tests"*), **Beh3m0th** orchestrates specialized typed agents (**Researcher**, **Coder**, **Reviewer**) backed by **Hybrid RAG**, **GraphRAG**, **Self-Correcting Retrieval**, **Three-Layer Memory**, **Dynamic Context Budgeting**, a **Custom SDK-Independent MCP Tool Server**, and an automated **Trajectory Evaluation Pipeline**.

This single flagship system proves mastery across all 8 target capabilities without feeling bolted together—every subsystem solves a real bottleneck in the workflow.

---

## 2. How the 8 Projects Map into Beh3m0th

| Source Project | Beh3m0th Module | Role in the Unified System |
| :--- | :--- | :--- |
| **Project 1: Context Assembler Engine** | `src/beh3m0th/context_engine/` | Runs before **every** LLM call across all agents. Allocates token budgets across system prompts, 3-layer memory, RAG evidence, and MCP tool definitions—critical for fitting free-tier token/TPM limits and preventing overflow. |
| **Project 2: Hybrid Retrieval & Reranking** | `src/beh3m0th/retrieval/hybrid/` | Powers fast, accurate document & code chunk search using BM25 + dense vectors (Reciprocal Rank Fusion), metadata filtering, and local CPU Cross-Encoder reranking ($0). |
| **Project 3: GraphRAG Knowledge Pipeline** | `src/beh3m0th/retrieval/graphrag/` | Extracts entities, code symbols, and relationships into a Knowledge Graph (local NetworkX or Neo4j Community), runs Leiden community summaries, and executes multi-hop traversal. |
| **Project 4: Self-Correcting Agentic RAG** | `src/beh3m0th/retrieval/agentic_rag/` | Wraps Hybrid + GraphRAG in a self-correcting loop: query rewriting, multi-step retrieval, inline citation grounding, hallucination verification, and confidence-based abstention. |
| **Project 6: Custom MCP Tool Server** | `src/beh3m0th/mcp_server/` | A hand-built, SDK-independent **JSON-RPC 2.0** tool server (`stdio` & `HTTP/SSE`) with Pydantic validation, per-agent RBAC permissions, and Docker-sandboxed execution for code/test tools. |
| **Project 7: Three-Layer Agent Memory** | `src/beh3m0th/memory/` | Provides **Working** (active task), **Episodic** (past runs/fixes with importance scoring), and **Semantic** (distilled repo conventions & facts) memory persisted across sessions. |
| **Project 8: Typed Multi-Agent System** | `src/beh3m0th/agents/` | Coordinates **Supervisor**, **Researcher**, **Coder**, and **Reviewer** agents using strictly typed Pydantic state envelopes, consensus voting, and structured conflict resolution. |
| **Project 9: Agent Trajectory Evaluation** | `src/beh3m0th/evaluation/` | Records full step-by-step execution trajectories, compares them against golden workflows, scores steps with LLM-as-a-judge (or deterministic rubrics), detects regressions, and gates PRs in CI. |

*(Note: Clean interfaces keep the architecture extensible so optional modules—like durable checkpointing or a security gateway—can be plugged in later without refactoring.)*

---

## 3. End-to-End System Architecture

```text
+-------------------------------------------------------------------------+
|                     USER / CLI / API / WEB DASHBOARD                    |
|                (Typer Rich CLI + FastAPI SSE Stream)                    |
+------------------------------------+------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|               TYPED MULTI-AGENT ORCHESTRATOR (Project 8)                |
|  +-------------------------------------------------------------------+  |
|  |  SUPERVISOR AGENT (Task Plan, Routing, Consensus & Arbitration)   |  |
|  +----------+----------------------+----------------------+----------+  |
|             |                      |                      |             |
|             v                      v                      v             |
|     +---------------+      +---------------+      +---------------+     |
|     |  RESEARCHER   | ---> |     CODER     | ---> |   REVIEWER    |     |
|     | (Read-Only)   |      | (Write+Exec)  |      | (Audit+Vote)  |     |
|     +-------+-------+      +-------+-------+      +-------+-------+     |
+-------------|----------------------|----------------------|-------------+
              |                      |                      |
              |  Every LLM call passes through:             |
              v                      v                      v
+-------------------------------------------------------------------------+
|                   CONTEXT ASSEMBLER ENGINE (Project 1)                  |
|  [Token Budgeter] -> [Priority Allocator] -> [Overflow Guard / Trimmer] |
+---------+--------------------------+--------------------------+---------+
          |                          |                          |
          |          +---------------+---------------+          |
          |          |  ZERO-COST LLM GATEWAY        |          |
          |          |  (Cache + Rate Limiter +      |          |
          |          |   Free API / Ollama / Paid)   |          |
          |          +---------------+---------------+          |
          v                          v                          v
+--------------------+    +--------------------+    +---------------------+
| THREE-LAYER MEMORY |    | SELF-CORRECTING    |    | CUSTOM MCP SERVER   |
|     (Project 7)    |    | AGENTIC RAG (P4)   |    |    (Project 6)      |
| - Working (Mem/Red)|    | - Query Rewriter   |    | - Raw JSON-RPC 2.0  |
| - Episodic (SQL/PG)|    | - Citation Ground  |    | - Pydantic Schemas  |
| - Semantic (SQL/PG)|    | - Hallucination    |    | - RBAC Permissions  |
| - Importance Score |    |   & Abstention     |    | - Docker Sandbox    |
+--------------------+    +----+----------+----+    +---------------------+
                               |          |
                      +--------+          +--------+
                      v                            v
           +--------------------+       +--------------------+
           | HYBRID RETRIEVAL   |       | GRAPHRAG PIPELINE  |
           |    (Project 2)     |       |    (Project 3)     |
           | - BM25 + Vector    |       | - Entity/Rel Extr. |
           | - Metadata Filter  |       | - NetworkX / Neo4j |
           | - CPU Cross-Enc.   |       | - Multi-Hop Query  |
           |   Reranker         |       | - Community Summ.  |
           +--------------------+       +--------------------+
                                     |
                                     v (Async Event Bus / Trajectory Logger)
+-------------------------------------------------------------------------+
|            AGENT TRAJECTORY EVALUATION PIPELINE (Project 9)             |
|  [Step Logger] -> [Golden Diff] -> [LLM-as-a-Judge] -> [CI Regression]  |
+-------------------------------------------------------------------------+
```

---

## 4. Target Repository Directory Structure

Cursor must strictly follow this modular layout. No circular imports are permitted between top-level packages under `src/beh3m0th/`.

```text
beh3m0th/
├── .cursorrules                     # Copy from CURSOR_RULES.md
├── .github/
│   └── workflows/
│       └── agent-eval-ci.yml        # Project 9 CI Regression Pipeline ($0 GitHub Actions)
├── docker/
│   ├── docker-compose.yml           # Optional local Postgres+pgvector, Qdrant, Neo4j, Redis
│   └── sandbox.Dockerfile           # Project 6 isolated code execution image
├── docs/
│   ├── 00_PROJECT_VISION_AND_ARCHITECTURE.md
│   ├── 01_TECH_STACK_AND_SETUP.md
│   ├── 02_MODULE_SPECS_RETRIEVAL_AND_CONTEXT.md
│   ├── 03_MODULE_SPECS_AGENTS_MEMORY_MCP_EVAL.md
│   ├── 04_IMPLEMENTATION_ROADMAP.md
│   └── 05_CURSOR_WORKFLOW_CHEATSHEET.md
├── evals/
│   ├── golden_workflows/            # YAML/JSON golden trajectories
│   ├── baselines/                   # Stored baseline metrics for regression
│   └── run_eval_suite.py            # CLI entrypoint for CI & local evals
├── src/
│   └── beh3m0th/
│       ├── __init__.py
│       ├── config.py                # Pydantic Settings (ZERO-COST default, swap via .env)
│       ├── llm/                     # Unified LLM Gateway + Disk/Redis Cache + Rate Limiter
│       ├── schemas/                 # Shared Pydantic v2 models & state envelopes
│       ├── context_engine/          # Project 1: Dynamic token budgeting & assembly
│       ├── retrieval/
│       │   ├── ingestion/           # Chunkers (Markdown, AST code)
│       │   ├── hybrid/              # Project 2: BM25 + Vector + CPU Reranker
│       │   ├── graphrag/            # Project 3: GraphStore (NetworkX/Neo4j) + Leiden
│       │   └── agentic_rag/         # Project 4: Rewriting, Grounding, Verifier
│       ├── mcp_server/              # Project 6: Raw JSON-RPC 2.0 + Sandbox
│       ├── memory/                  # Project 7: Working, Episodic, Semantic
│       ├── agents/                  # Project 8: Supervisor, Researcher, Coder, Reviewer
│       ├── evaluation/              # Project 9: Tracer, Golden Comparator, Judge
│       ├── cli/                     # Typer + Rich terminal interface
│       └── api/                     # FastAPI server + SSE streaming endpoints
├── tests/
│   ├── unit/                        # 100% offline, $0, zero API keys required
│   ├── integration/
│   └── fixtures/
├── pyproject.toml                   # Managed with `uv`
└── README.md
```

---

## 5. Core Architectural Principles

1. **$0 Prototype First, Seamless Paid Upgrade Later:**
   - Every external dependency (LLM provider, Vector DB, Graph DB, Memory DB) sits behind a clean Python `Protocol`/interface.
   - Day 1 runs for **$0** (free LLM tiers / Ollama + local CPU embeddings/reranker + embedded or local Docker storage).
   - Post-salary upgrade to paid Claude/GPT/Gemini APIs or managed cloud DBs requires **zero code changes**—only editing `.env`.
2. **Aggressive Cost & Rate-Limit Defense:**
   - Because free API tiers have RPM/RPD rate limits, `src/beh3m0th/llm/` includes an automatic **SQLite/Redis LLM Response Cache** (identical prompts return instantly for 0 tokens) and an **Async Token-Bucket Rate Limiter with exponential backoff**.
3. **Zero Framework Lock-In Where Mastery Matters:**
   - Do **not** use high-level black-box wrappers (LangChain, CrewAI, official `mcp` SDK) for the 8 core mechanisms. Build them cleanly on `Pydantic v2`, `asyncio`, `httpx`, and lightweight libraries.
4. **Typed Contracts & Observability Everywhere:**
   - Agents exchange strictly typed Pydantic `WorkflowState` objects and emit `TrajectoryStep` events on every action so Project 9 works end-to-end.