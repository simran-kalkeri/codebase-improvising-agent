"""
Planner module for the Modernization Agent.

Scans the target repository, builds a file-level summary, and asks the
LLM to produce a structured modernization plan — a list of individual
changes, each scoped to a single file.

Usage:
    from modernizer_agent.agent.planner import Planner
    planner = Planner(llm=client, repo_path="/path/to/repo")
    plan = planner.generate_plan(goal="Add type hints")
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from modernizer_agent.llm.ollama_client import OllamaClient
from modernizer_agent.tools.file_tools import list_files, read_file
from modernizer_agent.utils.logger import get_logger

log = get_logger("modernizer_agent.agent.planner")

# Maximum characters of a file to include in the repo summary sent to the LLM.
_FILE_PREVIEW_LIMIT = 500

# File extensions we consider for modernization.
_CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".go",
    ".rb", ".rs", ".c", ".cpp", ".h", ".hpp", ".cs",
}


@dataclass
class PlanItem:
    """A single planned change."""
    file: str
    change: str

    def to_dict(self) -> dict:
        return {"file": self.file, "change": self.change}

    @classmethod
    def from_dict(cls, d: dict) -> PlanItem:
        return cls(file=d["file"], change=d["change"])


@dataclass
class ModernizationPlan:
    """The full plan produced by the Planner."""
    goal: str
    items: list[PlanItem]

    def __len__(self) -> int:
        return len(self.items)

    def summary(self) -> str:
        lines = [f"Modernization Plan — {self.goal}", f"Total items: {len(self.items)}"]
        for i, item in enumerate(self.items, 1):
            lines.append(f"  {i}. [{item.file}] {item.change}")
        return "\n".join(lines)


class Planner:
    """Scans a repo and asks the LLM to produce a modernization plan."""

    def __init__(self, llm: OllamaClient, repo_path: str) -> None:
        self.llm = llm
        self.repo_path = repo_path

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_plan(self, goal: str) -> ModernizationPlan:
        """Scan the repo and return a structured modernization plan.

        Steps:
        1. List all code files in the repository.
        2. Read a preview of each file.
        3. Send the summary + goal to the LLM.
        4. Parse the LLM response into a list of PlanItems.
        """
        log.info("Scanning repository for plan generation", extra={
            "action": "plan_start", "tool": "planner",
        })

        # Step 1+2: build repo summary
        repo_summary = self._build_repo_summary()

        if not repo_summary:
            log.warning("No code files found in repository", extra={
                "action": "plan_abort", "tool": "planner",
            })
            return ModernizationPlan(goal=goal, items=[])

        # Step 3: ask the LLM
        prompt = self._build_plan_prompt(goal, repo_summary)
        system = (
            "You are a code modernization planner. "
            "Respond ONLY with valid JSON. No markdown, no explanation."
        )

        response = self.llm.generate(prompt=prompt, system=system)

        # Step 4: parse response
        plan = self._parse_plan_response(response, goal)

        log.info(f"Plan generated with {len(plan)} items", extra={
            "action": "plan_complete", "tool": "planner", "result": len(plan),
        })
        return plan

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_repo_summary(self) -> str:
        """Create a text summary of every code file in the repo."""
        try:
            files = list_files(self.repo_path, self.repo_path)
        except Exception as e:
            log.error(f"Failed to list files: {e}", extra={
                "action": "list_files", "tool": "planner", "error": str(e),
            })
            return ""

        # Filter to code files only.
        code_files = [
            f for f in files
            if Path(f).suffix in _CODE_EXTENSIONS
        ]

        if not code_files:
            return ""

        sections: list[str] = []
        for rel_path in code_files:
            abs_path = str(Path(self.repo_path) / rel_path)
            try:
                content = read_file(abs_path, self.repo_path)
                preview = content[:_FILE_PREVIEW_LIMIT]
                if len(content) > _FILE_PREVIEW_LIMIT:
                    preview += f"\n... ({len(content)} chars total)"
            except Exception:
                preview = "(could not read file)"

            sections.append(f"--- {rel_path} ---\n{preview}")

        return "\n\n".join(sections)

    def _build_plan_prompt(self, goal: str, repo_summary: str) -> str:
        """Build the prompt that asks the LLM to create a plan."""
        return f"""Analyze this repository and create a modernization plan.

GOAL: {goal}

REPOSITORY FILES:
{repo_summary}

Create a plan as a JSON object with this exact format:
{{
  "action": "plan",
  "arguments": {{
    "items": [
      {{"file": "path/to/file.py", "change": "Description of the specific change to make"}},
      ...
    ]
  }}
}}

Rules:
- Each item must target exactly ONE file.
- Changes must be small and incremental.
- Order items by dependency (independent changes first).
- Be specific about what to change (not vague like "improve code").
- Only include changes relevant to the stated goal.
"""

    def _parse_plan_response(
        self, response: dict, goal: str
    ) -> ModernizationPlan:
        """Parse the LLM's JSON response into a ModernizationPlan."""
        items: list[PlanItem] = []

        # Handle the expected format: {"action": "plan", "arguments": {"items": [...]}}
        if response.get("action") == "plan":
            raw_items = response.get("arguments", {}).get("items", [])
        # Also handle direct {"items": [...]} format
        elif "items" in response:
            raw_items = response["items"]
        else:
            log.warning(
                "Unexpected plan format from LLM, wrapping as single-item plan",
                extra={"action": "parse_plan", "tool": "planner", "error": str(response)},
            )
            raw_items = []

        for raw in raw_items:
            if isinstance(raw, dict) and "file" in raw and "change" in raw:
                items.append(PlanItem(
                    file=raw["file"],
                    change=raw["change"],
                ))
            else:
                log.warning(f"Skipping malformed plan item: {raw}", extra={
                    "action": "parse_plan", "tool": "planner",
                })

        return ModernizationPlan(goal=goal, items=items)
