"""Environment settings for the $0 embedded profile."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def load_dotenv(path: Path | None = None) -> None:
    """Load KEY=VALUE lines into the environment without overriding existing variables."""
    env_path = path if path is not None else Path(".env")
    if not env_path.is_file():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


# Google AI Studio — free lite models (low capacity).
DEFAULT_GOOGLE_MODEL = "gemini-3.1-flash-lite"
DEFAULT_GOOGLE_FALLBACK_1 = "gemini-3.5-flash-lite"
DEFAULT_GOOGLE_FALLBACK_2 = "gemini-2.5-flash-lite"

# Groq free coding models — 20B base, 27B medium fallback (120B left out on purpose).
DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"
DEFAULT_GROQ_FALLBACK_1 = "qwen/qwen3.8-27b"
DEFAULT_GROQ_FALLBACK_2 = ""

# OpenRouter :free variants — small then mid-size only.
DEFAULT_OPENROUTER_MODEL = "liquid/lfm-3-1.7b:free"
DEFAULT_OPENROUTER_FALLBACK_1 = "google/gemma-4-26b-a4b-it:free"
DEFAULT_OPENROUTER_FALLBACK_2 = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"


def _env(name: str, default: str) -> str:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return value.strip()


@dataclass(frozen=True, slots=True)
class Settings:
    """Constructor-friendly snapshot of the Profile A configuration."""

    profile: str
    google_api_key: str
    google_model: str
    google_fallback_model_1: str
    google_fallback_model_2: str
    groq_api_key: str
    groq_model: str
    groq_fallback_model_1: str
    groq_fallback_model_2: str
    openrouter_api_key: str
    openrouter_model: str
    openrouter_fallback_model_1: str
    openrouter_fallback_model_2: str
    llm_cache_path: Path
    llm_max_rpm: float
    sqlite_path: Path
    qdrant_local_path: Path
    mcp_sandbox_backend: str = "subprocess_restricted"
    mcp_sandbox_image: str = "beh3m0th-sandbox:latest"
    mcp_sandbox_timeout_sec: float = 15.0
    mcp_sandbox_mem_limit: str = "256m"
    mcp_allowed_workspace: Path = field(default_factory=lambda: Path("./data/workspace"))

    @classmethod
    def from_env(cls, *, env_file: Path | None = None) -> Settings:
        load_dotenv(env_file)
        google = _env("GOOGLE_API_KEY", "")
        if not google:
            google = _env("GEMINI_API_KEY", "")
        profile = _env("PROFILE", "")
        if not profile:
            profile = _env("STORAGE_PROFILE", "embedded")
        rpm = float(_env("LLM_MAX_RPM", "12"))
        if rpm <= 0:
            raise ValueError("LLM_MAX_RPM must be positive")
        backend = _env("MCP_SANDBOX_BACKEND", "subprocess_restricted")
        if backend not in {"docker", "subprocess_restricted"}:
            raise ValueError("MCP_SANDBOX_BACKEND must be docker or subprocess_restricted")
        timeout = float(_env("MCP_SANDBOX_TIMEOUT_SEC", "15"))
        if timeout <= 0:
            raise ValueError("MCP_SANDBOX_TIMEOUT_SEC must be positive")
        return cls(
            profile=profile,
            google_api_key=google,
            google_model=_env("GOOGLE_MODEL", DEFAULT_GOOGLE_MODEL),
            google_fallback_model_1=_env(
                "GOOGLE_FALLBACK_MODEL_1",
                DEFAULT_GOOGLE_FALLBACK_1,
            ),
            google_fallback_model_2=_env(
                "GOOGLE_FALLBACK_MODEL_2",
                DEFAULT_GOOGLE_FALLBACK_2,
            ),
            groq_api_key=_env("GROQ_API_KEY", ""),
            groq_model=_env("GROQ_MODEL", DEFAULT_GROQ_MODEL),
            groq_fallback_model_1=_env("GROQ_FALLBACK_MODEL_1", DEFAULT_GROQ_FALLBACK_1),
            groq_fallback_model_2=_env("GROQ_FALLBACK_MODEL_2", DEFAULT_GROQ_FALLBACK_2),
            openrouter_api_key=_env("OPENROUTER_API_KEY", ""),
            openrouter_model=_env("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL),
            openrouter_fallback_model_1=_env(
                "OPENROUTER_FALLBACK_MODEL_1",
                DEFAULT_OPENROUTER_FALLBACK_1,
            ),
            openrouter_fallback_model_2=_env(
                "OPENROUTER_FALLBACK_MODEL_2",
                DEFAULT_OPENROUTER_FALLBACK_2,
            ),
            llm_cache_path=Path(_env("LLM_CACHE_PATH", "./data/llm_cache.db")),
            llm_max_rpm=rpm,
            sqlite_path=Path(_env("SQLITE_PATH", "./data/beh3m0th.db")),
            qdrant_local_path=Path(_env("QDRANT_LOCAL_PATH", "./data/qdrant")),
            mcp_sandbox_backend=backend,
            mcp_sandbox_image=_env("MCP_SANDBOX_IMAGE", "beh3m0th-sandbox:latest"),
            mcp_sandbox_timeout_sec=timeout,
            mcp_sandbox_mem_limit=_env("MCP_SANDBOX_MEM_LIMIT", "256m"),
            mcp_allowed_workspace=Path(_env("MCP_ALLOWED_WORKSPACE", "./data/workspace")),
        )
