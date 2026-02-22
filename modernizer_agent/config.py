"""
Configuration module for the Modernization Agent.

All tunable constants and paths are defined here. Values can be overridden
via environment variables where noted.
"""

import os
from pathlib import Path


# ---------------------------------------------------------------------------
# Ollama / LLM Settings
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_GENERATE_ENDPOINT: str = f"{OLLAMA_BASE_URL}/api/generate"
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen3:8b")

# Maximum tokens the model may generate per request.
LLM_MAX_TOKENS: int = 2048

# How many times to retry when the LLM returns non-JSON output.
LLM_JSON_RETRIES: int = 3

# Timeout (seconds) for a single Ollama HTTP request.
LLM_REQUEST_TIMEOUT: int = 300

# ---------------------------------------------------------------------------
# Agent Control-Loop Settings
# ---------------------------------------------------------------------------
# Maximum retry attempts for a single plan item before reverting.
MAX_RETRIES_PER_ITEM: int = 3

# Maximum total iterations the agent will run (global safety cap).
MAX_TOTAL_ITERATIONS: int = 50

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Root of *this* package — used to locate bundled assets like prompts.
PACKAGE_DIR: Path = Path(__file__).resolve().parent

SYSTEM_PROMPT_PATH: Path = PACKAGE_DIR / "prompts" / "system_prompt.txt"

# Default location for the SQLite memory database.
DATABASE_PATH: Path = PACKAGE_DIR / "database" / "memory.db"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

# File to write structured JSON logs into (in addition to stderr).
LOG_FILE: str | None = os.getenv("LOG_FILE", None)

# ---------------------------------------------------------------------------
# Tool Defaults
# ---------------------------------------------------------------------------
# Git branch prefix the agent creates when working on a repo.
GIT_BRANCH_PREFIX: str = "modernize/"

# Linter command — ruff is the default; override to "flake8" if preferred.
LINTER_COMMAND: str = os.getenv("LINTER_COMMAND", "ruff check")

# Fallback linter if the primary is not installed.
LINTER_FALLBACK: str = "flake8"

# Test runner command.
TEST_COMMAND: str = os.getenv("TEST_COMMAND", "python3 -m pytest")
