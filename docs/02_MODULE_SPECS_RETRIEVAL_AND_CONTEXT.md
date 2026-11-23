# Beh3m0th Engineering Specs: Projects 1, 2, 3 & 4

This document provides Cursor with exact architectural contracts, mathematical formulas, schemas, and zero-cost design rules for Beh3m0th's **Context Engine** and **Knowledge/RAG Subsystem**.

---

## Project 1: Context Assembler Engine (`src/beh3m0th/context_engine/`)

### Goal
Ensure every LLM call across Supervisor, Researcher, Coder, and Reviewer stays strictly within its model context window (and free-tier token-per-minute caps) while maximizing signal-to-noise ratio.

### Core Requirements
1. **Dynamic Token Budgeting:**
   - Compute available input budget dynamically before assembly:
     $$B_{\text{avail}} = W_{\text{model}} - R_{\text{output}} - M_{\text{safety}}$$
     where $W_{\text{model}}$ is the configured model/tier context budget, $R_{\text{output}}$ is reserved completion tokens, and $M_{\text{safety}}$ is a safety margin (e.g., 256 tokens for message framing overhead).
2. **Priority-Based Context Injection & Allocation Buckets:**
   Every context candidate is wrapped in a `ContextItem(id, bucket, priority, content, token_count, metadata, truncatable)`:
   - **Priority 0 (Mandatory / Non-Droppable):** System prompt, agent persona, active user objective.
   - **Priority 1 (High):** MCP tool definitions required for the current agent's role + active Working Memory state.
   - **Priority 2 (Core Evidence):** Grounded retrieval chunks (HybridRAG + GraphRAG summaries) sorted by rerank score.
   - **Priority 3 (Long-Term Memory):** Top-scored Episodic & Semantic memory entries.
   - **Priority 4 (Auxiliary):** Older conversation turns / historical tool outputs.
3. **Dynamic Reallocation (Water-Filling Algorithm):**
   - Start with default bucket ratios (e.g., System/Tools 20%, Working Memory 15%, Retrieval 45%, Long-Term Memory 20%).
   - If a bucket uses less than its allocated share, redistribute the unused surplus tokens proportionally to higher-priority buckets that still have pending items.
4. **Context Overflow Prevention (Zero-Cost First):**
   - Count exact tokens offline using `tiktoken` (cached per `ContextItem`).
   - If mandatory items exceed $B_{\text{avail}}$, raise a typed `ContextBudgetExceededError`.
   - For droppable/truncatable items: apply deterministic priority eviction first, then extractive sentence/line trimming ($0), and only call `FAST_MODEL` summarization if explicitly enabled.
   - Emit a `ContextAssemblyReport` (tokens used per bucket, items included, items dropped, overflow actions) logged directly to the Project 9 Trajectory Tracer.

### Key Files to Build
- `tokenizer.py` — `TokenCounter` wrapping `tiktoken` with fallback estimator.
- `models.py` — `ContextBucket`, `ContextItem`, `BudgetPolicy`, `AssembledContext`, `ContextAssemblyReport`.
- `assembler.py` — `ContextAssembler.assemble(items, policy) -> AssembledContext`.

---

## Project 2: Hybrid Retrieval & Reranking System (`src/beh3m0th/retrieval/hybrid/`)

### Goal
Deliver high-precision, low-latency retrieval across codebases and technical documentation at **$0 inference cost** using local CPU embeddings, BM25, Qdrant, and a local ONNX Cross-Encoder reranker.

### Core Requirements
1. **Document & Code Ingestion (`src/beh3m0th/retrieval/ingestion/`):**
   - Support Markdown/text (recursive header-aware chunking with overlap) and Python source code (stdlib `ast` function & class-level chunking—zero heavy deps).
   - Attach rich metadata to every chunk: `doc_id`, `source_path`, `language`, `section_header`, `symbol_name`, `created_at`, `tags`, `line_start`, `line_end`.
2. **BM25 + Dense Vector Hybrid Search:**
   - **Dense Vector Search:** Local `fastembed` (`BAAI/bge-small-en-v1.5`, 384-dim) stored in **Qdrant** (supports `QdrantClient(path="./data/qdrant")` embedded mode for Profile A or Docker Qdrant URL for Profile B).
   - **Lexical BM25 Search:** Persisted `BM25Okapi` index (`rank-bm25`) with code-aware tokenization (splits `camelCase` and `snake_case` identifiers so symbol search works cleanly).
   - Run both searches concurrently via `asyncio.gather` and fuse using **Reciprocal Rank Fusion (RRF)**:
     $$\text{RRF}(d) = \sum_{r \in \{\text{dense}, \text{bm25}\}} \frac{w_r}{k + \text{rank}_r(d)}$$
     with default $k = 60$ and tunable weights $w_{\text{dense}}, w_{\text{bm25}}$.
3. **Metadata Filtering:**
   - Support structured `MetadataFilter` (exact match, list `IN`, date range, path prefix) applied **prior** to or during retrieval (pre-filtering in Qdrant + filtered BM25 candidate mask) so top-K slots are never wasted on irrelevant files.
4. **Local CPU Cross-Encoder Reranking ($0):**
   - Take the top $N$ fused candidates (e.g., $N = 20$) and batch-score `(query, chunk.text)` pairs through `fastembed`'s `TextCrossEncoder` (`Xenova/ms-marco-MiniLM-L-6-v2`).
   - Return the top $K$ reranked chunks (e.g., $K = 6$) with normalized confidence scores.
5. **Latency & Accuracy Optimization:**
   - Local SQLite/in-memory LRU cache for query embeddings and repeated `(query_hash, filter_hash)` queries.
   - Offload CPU-bound embedding/reranking via `asyncio.to_thread`.
   - Record per-stage latency (`embed_ms`, `bm25_ms`, `vector_ms`, `rerank_ms`, `total_ms`).

### Key Files to Build
- `schemas.py` — `DocumentChunk`, `MetadataFilter`, `RetrievedChunk`, `RetrievalMetrics`.
- `embedder.py` — Lazy-loaded `fastembed` ONNX embedding wrapper.
- `bm25_index.py` — Code-aware persisted BM25 index with metadata pre-filtering.
- `vector_store.py` — Qdrant wrapper seamlessly supporting `path=` (embedded) and `url=` (Docker/Cloud).
- `reranker.py` — Batched `fastembed` ONNX Cross-Encoder wrapper.
- `retriever.py` — `HybridRetriever.search(query, filters, top_k)` combining RRF + reranking + telemetry.

---

## Project 3: GraphRAG Knowledge Pipeline (`src/beh3m0th/retrieval/graphrag/`)

### Goal
Connect scattered facts across documents and code modules to answer multi-hop relational questions (*"Which services break if we alter TokenValidator?"*) and global architectural questions (*"Summarize the main subsystems and failure modes"*), while keeping LLM extraction cost minimal.

### Core Requirements
1. **Entity & Relationship Extraction (Cost-Optimized Hybrid):**
   - **For Code Files (0 LLM tokens!):** Deterministic extraction using Python `ast` (`File`, `Class`, `Function`, `Module` nodes; `DEFINES`, `IMPORTS`, `CALLS`, `INHERITS` edges).
   - **For Prose/Docs:** Batched `FAST_MODEL` structured extraction (cached via SQLite LLM cache) extracting typed entities (`Concept`, `Service`, `Component`, `API`, `Policy`, `Error`) and directed relationships (`DEPENDS_ON`, `CAUSES`, `MITIGATES`, `CONFIGURES`) with source chunk provenance (`chunk_id`).
2. **Knowledge Graph Construction & Deduplication (Dual Backend):**
   - Define a clean `GraphStoreProtocol`:
     - **`NetworkXGraphStore` (Profile A — $0, zero Docker):** Persists graph to `./data/knowledge_graph.json`, supports fast 1–3 hop BFS/DFS traversal.
     - **`Neo4jGraphStore` (Profile B — Docker / AuraDB Free):** Uses async `neo4j` driver with idempotent `MERGE` and Cypher multi-hop traversal.
   - Perform entity resolution (normalized canonical name + alias matching + optional embedding cosine similarity $> 0.90$ merge) to prevent duplicate nodes.
3. **Community Detection & Hierarchical Summaries:**
   - Convert graph adjacency into `igraph` and run the **Leiden algorithm** (`leidenalg`) to partition dense clusters into communities.
   - Generate a **Community Summary** for each cluster (using `FAST_MODEL` or deterministic structural summary in ultra-low-quota mode) and index summaries in Qdrant for global search.
4. **Multi-Hop Graph Traversal (Local & Global Search):**
   - **Local Multi-Hop Search:** Match query terms/entities to anchor nodes -> execute bounded `1..3` hop traversal collecting connecting paths and linked `chunk_id`s -> rank subgraphs by edge weight and node degree.
   - **Global Community Search:** Match query against `Community` summary embeddings to answer corpus-wide architectural questions.

### Key Files to Build
- `schemas.py` — `Entity`, `Relationship`, `GraphExtractionResult`, `SubgraphResult`, `CommunitySummary`.
- `extractor.py` — Zero-token AST code extractor + cached LLM prose extractor.
- `graph_store.py` — `GraphStoreProtocol`, `NetworkXGraphStore` (default), and `Neo4jGraphStore` (optional Profile B).
- `community.py` — Leiden clustering + community summarizer.
- `pipeline.py` — `GraphRAGPipeline` exposing `ingest()` and `query(mode="local" | "global" | "hybrid")`.

---

## Project 4: Self-Correcting Agentic RAG (`src/beh3m0th/retrieval/agentic_rag/`)

### Goal
Turn raw retrieval into a reliable, evidence-backed research engine that catches bad queries, repairs missing context, grounds every claim in citations, detects hallucinations, and refuses to guess when evidence is weak.

### Core Requirements
1. **Query Rewriting & Decomposition:**
   - Analyze incoming query: decompose complex multi-part queries into 2–3 atomic sub-queries (kept compact to save free-tier tokens); rewrite ambiguous queries with domain terms.
   - Route each sub-query to `HybridRetriever`, `GraphRAGPipeline`, or both.
2. **Multi-Step Iterative Retrieval Loop (Bounded by `max_steps = 3`):**
   - After Step 1 retrieval, grade relevance using **reranker scores first** ($0 heuristic gate before spending an LLM call!).
   - If average top-K rerank score or coverage is below threshold $\tau_{\text{rel}}$ (e.g., $0.60$), generate a targeted query rewrite focusing on the missing gap and re-retrieve.
3. **Citation Grounding:**
   - Prompt the synthesizer to emit structured claims where every factual statement includes explicit bracketed citations (e.g., `[chunk:doc42_p3]` or `[src/auth.py:45-68]`).
   - Run a deterministic zero-cost post-check verifying that 100% of cited IDs exist in the retrieved evidence set.
4. **Hallucination Detection (Claim-Level Verifier):**
   - Run an independent verification pass (`JUDGE_MODEL`) breaking the draft answer into atomic claims and labeling each as `SUPPORTED`, `PARTIALLY_SUPPORTED`, or `UNSUPPORTED` against its cited chunks.
   - Compute the Groundedness Score:
     $$G = \frac{\text{Number of SUPPORTED claims}}{\text{Total factual claims}}$$
   - If $G < \tau_{\text{ground}}$ (e.g., $0.80$) and retries remain, re-retrieve for unsupported claims or strip unsupported statements.
5. **Confidence-Based Abstention:**
   - Combine retrieval rerank score, graph coverage, and groundedness score $G$ into final confidence.
   - If final confidence $< \tau_{\text{abstain}}$ (e.g., $0.55$) after retries, **abstain explicitly**: return `status="ABSTAINED"` detailing what *was* found, what evidence is missing, and targeted follow-up questions rather than hallucinating.

### Key Files to Build
- `schemas.py` — `SubQuery`, `Citation`, `ClaimVerification`, `AgenticRAGResponse` (`ANSWERED` | `ABSTAINED`).
- `rewriter.py` — Query decomposition and gap-driven query rewriting.
- `verifier.py` — Deterministic citation ID validator + claim-level hallucination detector.
- `engine.py` — `SelfCorrectingRAG` orchestrating the bounded retrieve -> grade -> synthesize -> verify -> abstain loop via `ContextAssembler`.