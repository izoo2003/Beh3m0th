# Beh3m0th — Cursor Workflow Cheatsheet ($0 Edition)

This cheatsheet shows you **exactly how to set up your repo in Cursor**, avoid burning free-tier API/Cursor limits, and ship all 8 subsystems cleanly without Cursor hallucinating or breaking earlier modules.

---

## 1. First 10 Minutes: Repo Setup

Run these commands once on your machine:

```bash
mkdir beh3m0th && cd beh3m0th
git init
mkdir -p docs src/beh3m0th tests/unit tests/integration docker evals/golden_workflows evals/baselines data
```

1. Copy `CURSOR_RULES.md` to `.cursorrules` at the root of `beh3m0th/`.
2. Copy the five `00_*.md` – `05_*.md` files into `docs/`.
3. Add `data/`, `.env`, and `__pycache__/` to `.gitignore`.
4. Open the `beh3m0th/` folder in **Cursor**.
5. Grab **at least ONE free API key** (no credit card needed):
   - **Google AI Studio (`GEMINI_API_KEY`)** -> fastest free setup for `FAST_MODEL` & `REASONING_MODEL`.
   - **Groq (`GROQ_API_KEY`)** -> free `llama-3.3-70b-versatile` for `JUDGE_MODEL`.
   - *(Optional)* Install **Ollama** locally (`ollama pull qwen2.5-coder:7b` or `qwen2.5-coder:3b` for 8GB RAM laptops) for unlimited offline runs.

---

## 2. How to Prompt Cursor Without Losing Control

### Always Use the 3-Step Loop per Phase
1. **Plan First (Ask/Chat Mode):**
   > `@docs/04_IMPLEMENTATION_ROADMAP.md @docs/02_MODULE_SPECS_RETRIEVAL_AND_CONTEXT.md`
   > Before writing any code, list the exact files, classes, and Pydantic models you will create for **Phase X**, and confirm how it integrates with `ContextAssembler` and `TrajectoryTracer`.
2. **Implement Vertical Slice (Agent/Composer Mode):**
   - Once the plan looks right, tell Cursor: *"Approved. Implement Phase X now and write offline unit tests in `tests/unit/`."*
3. **Verify & Commit:**
   ```bash
   uv run ruff check .
   uv run pytest tests/unit
   git add . && git commit -m "feat: complete phase X"
   ```
   **Never start Phase $X+1$ with failing tests or uncommitted work!**

---

## 3. Anti-Bloat & Cost-Saving Habits in Cursor

- **Start a Fresh Cursor Chat Every Phase:** Long chat histories confuse Cursor and waste context tokens. Start a clean Composer session for each phase and tag only the 1–2 relevant `@docs/...` files + the specific interfaces it needs.
- **Test Offline First (`FakeLLMClient`):** Do 90% of your debugging with `FakeLLMClient` and unit tests ($0, instant). Only flip on live free Gemini/Groq keys once a phase's unit tests pass 100%.
- **Keep `LLM_CACHE_ENABLED=true`:** Whenever you run live demos or golden evaluations, Beh3m0th caches LLM responses in `data/llm_cache.db`. Re-running the same workflow while fixing a bug in Reviewer or CLI costs **0 API calls**.
- **Pause if Cursor Suggests Paid Tools:** If Cursor ever suggests `OpenAIEmbeddings`, `Pinecone`, `LangChain`, or `mcp` SDK, reply:
  > *"Violation of `.cursorrules`. Use our $0 local `fastembed` / embedded Qdrant / raw JSON-RPC architecture from `@docs/01_TECH_STACK_AND_SETUP.md`."*

---

## 4. Quick Debugging Prompts for Cursor

- **When a test fails:**
  > Here is the exact `pytest` traceback. Fix the root cause without weakening assertions or bypassing Pydantic validation.
- **When checking architectural compliance between phases:**
  > Audit `src/beh3m0th/` against `.cursorrules`: verify no module bypasses `ContextAssembler`, no module imports `langchain` or `mcp`, and every agent/retrieval action emits a `TrajectoryStep`.
- **When you get your salary later and want to upgrade to paid APIs / cloud DBs:**
  > Add the Profile B adapter tests for PostgreSQL+`pgvector` and Neo4j using `docker/docker-compose.yml`, keeping Profile A (`embedded`) intact as the offline test default.

---

## 5. Portfolio / Demo Showcase Checklist (For Interviews & GitHub README)

Even while running on **$0**, Beh3m0th will look like a senior-level AI systems engineering project if you record these 4 quick terminal/UI demos:
1. **Context Budget Proof (P1):** Show a `ContextAssemblyReport` table proving token allocation + surplus water-filling without overflow.
2. **Hybrid + GraphRAG + Abstention Proof (P2/P3/P4):** Ask a multi-hop codebase question (shows citations + graph path), then ask an unanswerable trick question (shows `ABSTAINED` with groundedness score instead of hallucinating).
3. **Custom MCP Security Proof (P6):** Show `Researcher` getting `-32001 Permission Denied` on `write_file_patch`, and `Coder` safely executing `pytest` inside the `--network=none` Docker sandbox.
4. **Multi-Agent + Eval Scorecard Proof (P7/P8/P9):** Run Session 1 -> fix a bug -> Reviewer consensus + Memory consolidation -> Run Session 2 showing episodic recall -> run `evals/run_eval_suite.py` printing the Rich regression table.