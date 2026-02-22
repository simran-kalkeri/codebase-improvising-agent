# Modernizer Agent

Modernizer Agent is a local, iterative code-modernization assistant built with LangGraph and a local Ollama model. It scans a target repository, creates a file-by-file plan, proposes diffs, asks for explicit approval, verifies changes (lint + tests), and commits only verified updates.

## Current Setup (This Repo)

Repository structure:
- `README.md` (repo root)
- `modernizer_agent/pyproject.toml` (project metadata + console script)
- `modernizer_agent/requirements.txt` (minimal runtime deps)
- `modernizer_agent/...` (Python package source code)

Entry points:
- Module run: `python -m modernizer_agent.main`
- Installed console script: `modernizer-agent`

## Prerequisites

- Python 3.10+
- `git` in PATH
- Ollama running locally (default `http://localhost:11434`)
- A pulled Ollama model (default `qwen3:8b`)

Optional:
- `ruff` or `flake8` for linting
- `pytest` for tests

## Install

From repo root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r modernizer_agent\requirements.txt
python -m pip install -e .\modernizer_agent
```

Optional dev tools:

```powershell
python -m pip install -e .\modernizer_agent[dev]
```

## Run

From repo root (or any environment where package is installed):

```powershell
python -m modernizer_agent.main --repo <path-to-target-repo> --goal "<modernization-goal>"
```

Or via console script:

```powershell
modernizer-agent --repo <path-to-target-repo> --goal "<modernization-goal>"
```

Example:

```powershell
modernizer-agent --repo C:\work\legacy-service --goal "Add type hints and remove deprecated syntax"
```

Useful flags:
- `--max-iterations <n>`: cap total loop iterations.
- `--dry-run`: generate/print plan and diffs without writing or committing.

## Workflow Summary

1. Planner scans code files and requests a JSON plan from the model.
2. Executor generates a full-file rewrite for one plan item and computes a unified diff.
3. You choose `Apply`, `Skip`, or `Quit`.
4. If applied, verifier runs lint and tests.
5. On pass, the agent commits; on failure, it reverts the file and retries (up to limit).

Safety behavior:
- One file per plan item.
- Explicit approval before write.
- Verification before commit.
- Git branch auto-created with prefix `modernize/`.

## Interactive Controls

- `A` / `Apply`: apply change, verify, commit on success.
- `S` / `Skip`: skip this plan item.
- `Q` / `Quit`: stop immediately.

Before execution starts, the plan is shown and confirmed with `Y/n`.

## Configuration

Config lives in `modernizer_agent/config.py`.

Environment variables:
- `OLLAMA_BASE_URL` (default: `http://localhost:11434`)
- `OLLAMA_MODEL` (default: `qwen3:8b`)
- `LOG_LEVEL` (default: `INFO`)
- `LOG_FILE` (optional JSON log file path)
- `LINTER_COMMAND` (default: `ruff check`)
- `TEST_COMMAND` (default: `python3 -m pytest`)

Notes:
- If primary linter is unavailable, fallback is `flake8`.
- If pytest is missing, tests are skipped as non-fatal.
- Pytest exit code `5` ("no tests collected") is treated as pass.

## Components

- `modernizer_agent/main.py`: CLI bootstrap and banner.
- `modernizer_agent/agent/controller.py`: LangGraph orchestration.
- `modernizer_agent/agent/planner.py`: repo scan + plan generation.
- `modernizer_agent/agent/executor.py`: change generation + diff/apply.
- `modernizer_agent/agent/verifier.py`: lint/test result aggregation.
- `modernizer_agent/agent/memory.py`: SQLite error/fix memory.
- `modernizer_agent/tools/*.py`: safe file/git/test wrappers.
- `modernizer_agent/llm/ollama_client.py`: Ollama JSON client.
- `modernizer_agent/prompts/system_prompt.txt`: system prompt.

Runtime memory DB default:
- `modernizer_agent/database/memory.db` (created automatically).

## Limitations

- Planning uses truncated file previews.
- Semantic change quality depends on model behavior.
- Verification defaults are Python-centric unless commands are overridden.
- No sandbox/isolation for lint/test command execution.
