"""
Executor module for the Modernization Agent.

Takes a single PlanItem, reads the target file, asks the LLM for a
code change, and produces a **RecommendedChange** — a structured diff
that must be approved by the user before being applied.

The executor does NOT write files automatically.

Usage:
    from modernizer_agent.agent.executor import Executor
    executor = Executor(llm=client, repo_path="/path/to/repo")
    rec = executor.generate_change(plan_item)
    if user_approves:
        executor.apply_change(rec)
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from pathlib import Path

from modernizer_agent.agent.memory import MemoryStore
from modernizer_agent.agent.planner import PlanItem
from modernizer_agent.llm.ollama_client import OllamaClient
from modernizer_agent.tools.file_tools import read_file, write_file
from modernizer_agent.utils.logger import get_logger

log = get_logger("modernizer_agent.agent.executor")


@dataclass
class RecommendedChange:
    """A proposed change that awaits user approval."""
    plan_item: PlanItem
    file_path: str
    original_content: str
    proposed_content: str
    diff: str
    explanation: str

    @property
    def has_changes(self) -> bool:
        """True if the proposed content differs from the original."""
        return self.original_content != self.proposed_content


class Executor:
    """Generates recommended code changes (does not auto-apply)."""

    def __init__(
        self,
        llm: OllamaClient,
        repo_path: str,
        memory: MemoryStore | None = None,
    ) -> None:
        self.llm = llm
        self.repo_path = repo_path
        self.memory = memory

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_change(
        self,
        plan_item: PlanItem,
        memory_hints: str = "",
    ) -> RecommendedChange:
        """Read the file, ask the LLM for a change, and return a diff.

        This does NOT modify the file system.
        """
        abs_path = str(Path(self.repo_path) / plan_item.file)

        # Step 1: read current file content.
        try:
            original = read_file(abs_path, self.repo_path)
        except Exception as e:
            log.error(f"Cannot read {plan_item.file}: {e}", extra={
                "action": "read", "tool": "executor", "error": str(e),
            })
            return RecommendedChange(
                plan_item=plan_item,
                file_path=abs_path,
                original_content="",
                proposed_content="",
                diff="",
                explanation=f"Error: could not read file — {e}",
            )

        # Step 2: ask the LLM for the change.
        prompt = self._build_change_prompt(plan_item, original, memory_hints)
        system = (
            "You are a precise code editor. Respond ONLY with valid JSON. "
            "No markdown, no explanation text outside JSON."
        )

        response = self.llm.generate(prompt=prompt, system=system)
        proposed, explanation = self._parse_change_response(response, original)

        # Step 3: compute diff.
        diff = self._compute_diff(
            original, proposed,
            from_label=f"a/{plan_item.file}",
            to_label=f"b/{plan_item.file}",
        )

        log.info(
            f"Generated change for {plan_item.file}",
            extra={
                "action": "generate_change",
                "tool": "executor",
                "file": plan_item.file,
                "result": "has_changes" if original != proposed else "no_changes",
            },
        )

        return RecommendedChange(
            plan_item=plan_item,
            file_path=abs_path,
            original_content=original,
            proposed_content=proposed,
            diff=diff,
            explanation=explanation,
        )

    def apply_change(self, change: RecommendedChange) -> None:
        """Write the proposed content to disk (called after user approval)."""
        if not change.has_changes:
            log.info("No changes to apply", extra={
                "action": "apply", "tool": "executor",
            })
            return

        write_file(change.file_path, change.proposed_content, self.repo_path)
        log.info(
            f"Applied change to {change.plan_item.file}",
            extra={
                "action": "apply",
                "tool": "executor",
                "file": change.plan_item.file,
            },
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_change_prompt(
        self,
        plan_item: PlanItem,
        current_content: str,
        memory_hints: str,
    ) -> str:
        """Build the prompt that asks the LLM to modify a file."""
        hints_section = ""
        if memory_hints:
            hints_section = f"""
MEMORY HINTS (learned from previous attempts):
{memory_hints}
"""

        return f"""Modify the following file according to the change description.

FILE: {plan_item.file}
CHANGE: {plan_item.change}
{hints_section}
CURRENT FILE CONTENT:
```
{current_content}
```

Respond with a JSON object in this exact format:
{{
  "action": "write_file",
  "arguments": {{
    "content": "<the complete modified file content>",
    "explanation": "<brief explanation of what was changed and why>"
  }}
}}

Rules:
- Return the COMPLETE file content, not just the changed parts.
- Make minimal, focused changes — only what the change description asks for.
- Preserve existing formatting and style where possible.
- Do not add unrelated changes.
"""

    def _parse_change_response(
        self, response: dict, original: str
    ) -> tuple[str, str]:
        """Extract proposed content and explanation from the LLM response.

        Returns (proposed_content, explanation).
        Falls back to original content if parsing fails.
        """
        # Expected: {"action": "write_file", "arguments": {"content": ..., "explanation": ...}}
        args = response.get("arguments", {})
        if not args:
            # Try flat format: {"content": ..., "explanation": ...}
            args = response

        content = args.get("content", "")
        explanation = args.get("explanation", "No explanation provided.")

        if not content or not isinstance(content, str):
            log.warning("LLM returned empty/invalid content, keeping original", extra={
                "action": "parse_change", "tool": "executor",
            })
            return original, "LLM did not produce valid content."

        return content, explanation

    @staticmethod
    def _compute_diff(
        original: str,
        proposed: str,
        from_label: str = "original",
        to_label: str = "proposed",
    ) -> str:
        """Compute a unified diff between original and proposed content."""
        diff_lines = difflib.unified_diff(
            original.splitlines(keepends=True),
            proposed.splitlines(keepends=True),
            fromfile=from_label,
            tofile=to_label,
        )
        return "".join(diff_lines)
