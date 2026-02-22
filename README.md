# Modernizer Agent

Modernizer Agent is a local, iterative code-modernization assistant. It scans a target repository, creates a file-by-file modernization plan, proposes concrete edits, asks for user approval, verifies each accepted change (lint + tests), and commits successful updates on a dedicated branch.

The system is designed to be conservative:
- one plan item per file
- explicit user approval before applying each diff
- automatic verification before commit
- retry with memory hints on failures

## What It Does

Given a modernization goal such as:
- "Add type hints to public APIs"
- "Replace deprecated Python APIs"
- "Refactor legacy modules incrementally"

the agent executes this loop:
1. Read repository code files and build a summary.
2. Ask the LLM for an ordered JSON plan (`file + change` pairs).
3. For each plan item, ask the LLM for a full-file rewrite and compute a unified diff.
4. Show the diff and prompt you to `Apply`, `Skip`, or `Quit`.
5. If applied, run linter and tests.
6. If verification passes, commit the change; if not, revert and retry (up to configured limits).

## Repository Layout

- `modernizer_agent/main.py`: CLI entry point and startup flow.
- `modernizer_agent/config.py`: central config constants and env overrides.
- `modernizer_agent/agent/controller.py`: main orchestration loop.
- `modernizer_agent/agent/planner.py`: repo scan + plan generation.
- `modernizer_agent/agent/executor.py`: per-file change generation and apply logic.
- `modernizer_agent/agent/verifier.py`: lint/test verification abstraction.
- `modernizer_agent/agent/memory.py`: SQLite memory store for error/fix history.
- `modernizer_agent/llm/ollama_client.py`: Ollama API wrapper with JSON enforcement.
- `modernizer_agent/tools/`: file, git, and test/lint tool wrappers.
- `modernizer_agent/prompts/system_prompt.txt`: bundled system prompt.
- `modernizer_agent/database/memory.db`: runtime SQLite memory database.

## Architecture Overview

### 1) Planner
- Lists files under the target repo.
- Filters to code extensions (`.py`, `.js`, `.ts`, `.java`, `.go`, etc.).
- Reads a short preview per file.
- Prompts the model to return JSON with small, single-file plan items.

### 2) Executor
- Reads the target file content.
- Prompts the model for the full updated file content plus explanation.
- Produces a unified diff against original content.
- Applies the change only after explicit user approval.

### 3) Verifier
- Runs lint and test commands for the target repository.
- Default linter strategy: primary (`ruff check`) then fallback (`flake8`).
- Default test command: `python3 -m pytest`.
- Treats "no tests collected" as pass.

### 4) Memory
- Stores prior attempts in SQLite (`error_text`, `applied_fix`, `success`, `file_path`).
- On retries, surfaces similar past outcomes to the model as hints.
- Uses normalized error signatures plus keyword fallback matching.

### 5) Controller
- Coordinates full lifecycle: plan -> propose -> approve -> verify -> commit/retry.
- Creates or checks out a `modernize/*` branch before changes.
- Enforces retry cap per item and optional total iteration cap.

## Prerequisites

- Python 3.10+ recommended
- `git` installed and available in PATH
- Ollama running locally (default: `http://localhost:11434`)
- A pulled Ollama model (default: `qwen3:8b`)

Optional but recommended:
- `ruff` (or `flake8`) for linting
- `pytest` for tests

## Setup

From project root:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r modernizer_agent/requirements.txt
```

Install optional tooling in the environment as needed:

```bash
pip install ruff pytest
```

## Running the Agent

Use the module entrypoint:

```bash
python -m modernizer_agent.main --repo <path-to-target-repo> --goal "<modernization-goal>"
```

Example:

```bash
python -m modernizer_agent.main --repo C:\work\legacy-service --goal "Add type hints and remove deprecated syntax"
```

Useful flags:
- `--max-iterations <n>`: global cap for control-loop iterations.
- `--dry-run`: generate and display plan/proposals without writing or committing.

## Interactive Controls

During execution, each proposed file change prompts:
- `A` or `Apply`: write file, run verification, and commit if successful.
- `S` or `Skip`: ignore this plan item and continue.
- `Q` or `Quit`: stop the run immediately.

Before item execution, the controller asks for plan confirmation (`Y/n`).

## Configuration

Most operational settings are defined in `modernizer_agent/config.py` and several support environment variable overrides.

Key env vars:
- `OLLAMA_BASE_URL` (default `http://localhost:11434`)
- `OLLAMA_MODEL` (default `qwen3:8b`)
- `LOG_LEVEL` (default `INFO`)
- `LOG_FILE` (optional; enables JSON log file output)
- `LINTER_COMMAND` (default `ruff check`)
- `TEST_COMMAND` (default `python3 -m pytest`)

Other safety/performance defaults are currently code constants, including:
- max LLM JSON retries
- LLM request timeout
- per-item retry count
- global max iterations

## Output and Logging

- Human-readable terminal output for plans, diffs, and run summaries.
- Structured JSON logging on stderr by default.
- Optional JSON log file if `LOG_FILE` is set.

Typical logged fields include:
- `action`, `tool`, `iteration`, `result`
- `file`, `plan_item`, `commit_hash`, `error`

## Safety Model and Boundaries

- File operations validate paths against repo root to prevent path traversal.
- Only one file is targeted per plan item.
- Changes are explicit and user-gated before write.
- Verification runs before commit.
- Failed verification triggers local file revert to original content.

Note: the repo also includes a `revert_last_commit` git helper; it is not part of the default failure path used by the controller.

## Limitations

- LLM responses are constrained to JSON, but semantic quality still depends on model behavior.
- Planning uses truncated file previews, which may miss deep cross-file context.
- Multi-language support exists at planning level, but verification defaults are Python-centric unless overridden.
- No sandboxing/isolation is built into execution of lint/test commands.

## Development Notes

- Keep prompts and parsing strict to minimize malformed model outputs.
- Add tests around planner/executor parsing and verifier behavior for regressions.
- If extending tools, preserve path validation and structured logging conventions.
