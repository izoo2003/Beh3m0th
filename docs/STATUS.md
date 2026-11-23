# Beh3m0th phase status

Update this file only after that phase's verification gate passes (`pytest` and `ruff check .`). Do not start the next phase while the current one is unchecked.

- [x] **Phase 0 — Skeleton, $0 LLM gateway, and storage.** `tests/unit/test_phase0.py` passes offline (10 tests). Schema validation, cache hits, Google/Groq/OpenRouter model fallbacks, token-bucket waiting, and SQLite/Qdrant initialization are covered. `ruff check .` is clean. Unit tests stay offline; live keys belong in `.env` only.
- [x] **Phase 1 — Context Assembler.** Assembled prompts never exceed the token budget. Unused memory budget is reallocated to retrieval evidence. `tests/unit/test_context_assembler.py` passes offline (9 tests). `ruff check .` is clean.
- [x] **Phase 2 — Hybrid retrieval.** Local BM25 + embedded Qdrant + fastembed rerank, $0, no Docker. Hybrid search beats BM25-only and dense-only on the fixture queries. `tests/unit/test_hybrid_retrieval.py` passes offline (8 tests). `ruff check .` is clean.
- [x] **Phase 3 — GraphRAG.** Multi-hop query returns the connected subgraph and source chunk IDs. NetworkX is the default store. `tests/unit/test_graphrag.py` passes offline (12 tests). `ruff check .` is clean.
- [x] **Phase 4 — Self-correcting RAG.** Answerable queries cite real chunk IDs. Unanswerable queries return `status="ABSTAINED"`. `tests/unit/test_agentic_rag.py` passes offline (9 tests). `ruff check .` is clean.
- [x] **Phase 5 — MCP tool server.** Researcher is rejected with JSON-RPC `-32001` on write and exec tools. Path traversal, including a symlink or junction escape, is blocked. `tests/unit/test_mcp_server.py` passes offline (12 tests). `ruff check .` is clean.
- [x] **Phase 6 — Three-layer memory.** Session 2 recalls the episodic bug fix and the semantic repo rule from session 1, ranked by composite score. `tests/unit/test_three_layer_memory.py` passes offline (8 tests). `ruff check .` is clean.
- [x] **Phase 7 — Typed multi-agent system.** Seeded bug flows Researcher → Coder → Reviewer → Supervisor. Forced disagreement produces a `ConflictRecord` and an arbitration decision. `tests/unit/test_multi_agent_workflow.py` passes offline (4 tests). `ruff check .` is clean.
- [x] **Phase 8 — Trajectory evaluation, FastAPI SSE, and CI.** `evals/run_eval_suite.py --mode deterministic` exits 0 offline. Golden metrics, regression exits, the judge stub, and the SSE run stream are covered. `tests/unit/test_evals.py` passes offline (9 tests). `ruff check .` is clean.
- [ ] **Phase 9 — Free cloud deployment, then paid upgrade.** Not started. Prototype deployment and the `.env`-only paid swap come after Phase 8.
