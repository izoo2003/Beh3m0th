# Beh3m0th — $0-First Tech Stack & Upgrade Path

This guide is specifically engineered so you can build, test, and deploy **Beh3m0th** for **$0/month right now**, while keeping clean interfaces so that once your salary comes in, upgrading to paid frontier APIs takes **60 seconds in `.env`**.

---

## 1. Zero-Cost LLM Strategy (Pick Any Free Option — Mix & Match!)

Because all LLM calls flow through `src/beh3m0th/llm/client.py` (`LiteLLM`), you can combine free cloud tiers and/or local Ollama models with **zero credit card required**:

| Option | Cost | Best Models (Free) | Best For |
| :--- | :--- | :--- | :--- |
| **1. Google AI Studio (Gemini Free Tier)** | **$0** | `gemini/gemini-2.5-flash`, `gemini/gemini-2.5-flash-lite` | Huge context window, fast JSON/structured outputs, great default `FAST_MODEL` & `REASONING_MODEL`. |
| **2. Groq Free Tier** | **$0** | `groq/llama-3.3-70b-versatile`, `groq/qwen-qwq-32b` | Blazing-fast inference; great as an independent `JUDGE_MODEL` (Project 4 & 9) to avoid self-bias. |
| **3. OpenRouter Free Models** | **$0** | Any `:free` model (e.g., `openrouter/deepseek/deepseek-r1:free`, `openrouter/qwen/qwen-2.5-coder-32b-instruct:free`) | Great backup when another free tier hits daily quota. |
| **4. Ollama (100% Local Offline)** | **$0** | `ollama/qwen2.5-coder:7b` (or `3b` on 8GB RAM), `ollama/llama3.2:3b` | Zero rate limits, works offline, unlimited retries during heavy dev loops. |
| **5. Deterministic Mock Mode** | **$0** | Built-in `FakeLLMClient` + recorded JSON cassettes | Instant unit tests & GitHub Actions CI without spending a single API call. |

### Free-Tier Survival Features (Built Into `src/beh3m0th/llm/`)
1. **Prompt & Embedding Disk Cache:** Hash `(model, messages, schema)` into a local SQLite cache (`data/llm_cache.db`). Re-running tests or tweaking downstream code costs **0 API calls**.
2. **Token-Bucket Rate Limiter + Retry Backoff:** Respects free-tier RPM limits gracefully (e.g., 10–15 RPM) with automatic jittered retry on HTTP `429`, plus optional automatic fallback from Provider A -> Provider B -> local Ollama.
3. **AST-First Graph Extraction (Project 3):** Extract code entities/edges (`Class`, `Function`, `IMPORTS`, `CALLS`) deterministically with Python `ast` / `tree-sitter` = **0 LLM tokens spent** on code ingestion! Only call `FAST_MODEL` on prose Markdown docs.

---

## 2. Two Storage Profiles (Start Ultra-Light, Scale Anytime)

Every storage layer sits behind a clean Python `Protocol` so you can pick the profile that fits your laptop's RAM right now:

| Subsystem | **Profile A: Ultra-Light Embedded ($0, No Heavy Docker RAM)** *(Recommended to start!)* | **Profile B: Local Docker Production Mirror ($0)** | **Future Free Cloud Deployment ($0)** |
| :--- | :--- | :--- | :--- |
| **Vector + Hybrid (P2)** | **Qdrant Embedded Local Mode** (`QdrantClient(path="./data/qdrant")`) + `rank-bm25` | `qdrant/qdrant` Docker container | **Qdrant Cloud Free Tier** (1 GB forever free) |
| **Embeddings (P2 & P7)** | **`fastembed`** (`BAAI/bge-small-en-v1.5` — ONNX CPU, ~130MB RAM, no PyTorch needed!) | Same (`fastembed` local CPU = $0) | Same (or OpenAI/Gemini embeddings later) |
| **Reranker (P2)** | **`fastembed` TextCrossEncoder** (`Xenova/ms-marco-MiniLM-L-6-v2` — ONNX CPU, ~90MB!) | Same | Same (or Cohere Rerank later) |
| **Knowledge Graph (P3)** | **NetworkX + JSON persistence** + `igraph`/`leidenalg` | **Neo4j 5 Community** Docker container | **Neo4j AuraDB Free Tier** |
| **Working Memory (P7)** | **In-Memory / Disk TTL Store** (`diskcache` / SQLite) | **Redis 7 Alpine** Docker container | **Upstash Redis Free Tier** |
| **Episodic, Semantic & Traces (P7 & P9)** | **SQLite (`aiosqlite`)** + numpy cosine similarity (handles 50k+ memories effortlessly!) | **PostgreSQL 16 + `pgvector`** Docker container | **Neon** or **Supabase** Free Tier (`pgvector` built-in) |
| **Sandboxed Code Exec (P6)** | **Docker Sandbox** (`python:3.12-slim` with `--network=none`), with restricted subprocess fallback if Docker daemon is off | **Docker Sandbox** (strict resource limits) | Dedicated worker / Fly.io / Oracle Free VM |

> **Why `fastembed` instead of heavy `sentence-transformers` + PyTorch?**
> Installing PyTorch pulls ~2.5 GB of dependencies and eats RAM. `fastembed` uses ONNX Runtime, installs in seconds, runs **both** dense embeddings (`BAAI/bge-small-en-v1.5`) and Cross-Encoder reranking (`Xenova/ms-marco-MiniLM-L-6-v2`) super fast on a normal laptop CPU for **$0**!

---

## 3. Core Tech Stack Summary

- **Language & Package Manager:** Python 3.12+ managed with **`uv`** (free, ultra-fast).
- **Schemas & Config:** **Pydantic v2** + `pydantic-settings`.
- **LLM Gateway:** **LiteLLM** (`litellm`) behind `src/beh3m0th/llm/` with built-in caching + rate-limit fallback.
- **Token Budgeting (P1):** **`tiktoken`** (offline, deterministic, $0).
- **Code Parsing (P2 & P3):** Python stdlib **`ast`** (zero dependencies) + optional `tree-sitter` later for JS/TS.
- **API & UI:** **Typer + Rich** CLI first (zero frontend overhead) -> **FastAPI + SSE** -> optional **Streamlit** or **Next.js** dashboard hosted free on Vercel / Hugging Face Spaces.
- **Testing & CI (P9):** `pytest`, `pytest-asyncio`, `ruff`, `mypy`, and **GitHub Actions** (free for public repos + 2,000 free mins/mo on private repos).

---

## 4. Lightweight `pyproject.toml` Blueprint

```toml
[project]
name = "beh3m0th"
version = "0.1.0"
description = "Beh3m0th — Zero-cost-first Autonomous Multi-Agent Engineering Platform"
requires-python = ">=3.12"
dependencies = [
    "pydantic>=2.9",
    "pydantic-settings>=2.5",
    "fastapi>=0.115",
    "uvicorn[standard]>=0.32",
    "sse-starlette>=2.1",
    "httpx>=0.27",
    "litellm>=1.50",
    "tiktoken>=0.8",
    "sqlalchemy[asyncio]>=2.0",
    "aiosqlite>=0.20",
    "qdrant-client>=1.12",
    "rank-bm25>=0.2.2",
    "fastembed>=0.4",
    "numpy>=2.0",
    "networkx>=3.4",
    "igraph>=0.11",
    "leidenalg>=0.10",
    "docker>=7.1",
    "typer>=0.12",
    "rich>=13.9",
    "structlog>=24.4",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
# Install these only when switching to Profile B (Docker Postgres/Neo4j/Redis)
prod-backends = [
    "asyncpg>=0.30",
    "pgvector>=0.3",
    "alembic>=1.13",
    "neo4j>=5.25",
    "redis>=5.2",
]
dev = [
    "pytest>=8.3",
    "pytest-asyncio>=0.24",
    "pytest-cov>=5.0",
    "ruff>=0.7",
    "mypy>=1.12",
    "respx>=0.21",
]
```

---

## 5. Environment Configuration (`.env.example`)

Notice how **Day 1 ($0)** works out of the box, and **Future Paid Upgrade** is just commenting/uncommenting 3 lines:

```bash
# ============================================================
# BEH3M0TH CONFIG — DEFAULT: 100% FREE ($0) PROTOTYPE PROFILE
# ============================================================
BEH3M0TH_ENV=prototype

# --- FREE LLM KEYS (Fill any 1+ free key, or use Ollama locally) ---
GEMINI_API_KEY=                  # Free at aistudio.google.com
GROQ_API_KEY=                    # Free at console.groq.com
OPENROUTER_API_KEY=              # Free models at openrouter.ai
OLLAMA_BASE_URL=http://localhost:11434

# --- ACTIVE MODEL ROUTING ($0 Free Tier Defaults) ---
FAST_MODEL=gemini/gemini-2.5-flash-lite
REASONING_MODEL=gemini/gemini-2.5-flash
JUDGE_MODEL=groq/llama-3.3-70b-versatile
FALLBACK_MODEL=ollama/qwen2.5-coder:7b

# --- FUTURE PAID UPGRADE (Swap later after salary—zero code changes!) ---
# OPENAI_API_KEY=
# ANTHROPIC_API_KEY=
# FAST_MODEL=gemini/gemini-2.5-flash
# REASONING_MODEL=anthropic/claude-sonnet-4-5
# JUDGE_MODEL=openai/gpt-4o

# --- FREE-TIER RATE LIMIT & CACHE PROTECTION ---
LLM_CACHE_ENABLED=true
LLM_CACHE_PATH=./data/llm_cache.db
LLM_MAX_RPM=12                   # Prevents hitting free-tier 429 limits

# --- LOCAL CPU EMBEDDINGS & RERANKER ($0 via fastembed ONNX) ---
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
RERANKER_MODEL=Xenova/ms-marco-MiniLM-L-6-v2

# --- STORAGE PROFILE: "embedded" ($0, no Docker) OR "docker" (Postgres/Neo4j/Redis) ---
STORAGE_PROFILE=embedded
DATA_DIR=./data

# Profile A (Embedded $0 Defaults)
SQLITE_URL=sqlite+aiosqlite:///./data/beh3m0th.db
QDRANT_LOCAL_PATH=./data/qdrant
GRAPH_LOCAL_PATH=./data/knowledge_graph.json

# Profile B (Optional Docker / Cloud Free Tiers later)
# POSTGRES_URL=postgresql+asyncpg://beh3m0th:beh3m0th@localhost:5432/beh3m0th
# QDRANT_URL=http://localhost:6333
# NEO4J_URI=bolt://localhost:7687
# NEO4J_USER=neo4j
# NEO4J_PASSWORD=beh3m0th_dev
# REDIS_URL=redis://localhost:6379/0

# --- PROJECT 6: MCP SANDBOX LIMITS ---
MCP_SANDBOX_BACKEND=docker       # "docker" or "subprocess_restricted" fallback
MCP_SANDBOX_IMAGE=beh3m0th-sandbox:latest
MCP_SANDBOX_TIMEOUT_SEC=15
MCP_SANDBOX_MEM_LIMIT=256m
MCP_ALLOWED_WORKSPACE=/tmp/beh3m0th_workspace
```