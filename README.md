# Modernizer Agent

A Python-based agent framework for planning, executing, verifying, and storing memory for modernization tasks.

## Structure

- `modernizer_agent/agent`: orchestration components (planner, executor, verifier, controller, memory)
- `modernizer_agent/llm`: LLM client integrations
- `modernizer_agent/tools`: utility tools for file, git, test, and search operations
- `modernizer_agent/database`: local runtime data

## Setup

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r modernizer_agent/requirements.txt
```

## Run

```bash
python -m modernizer_agent.main
```
