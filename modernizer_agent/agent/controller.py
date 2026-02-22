"""
Controller module for the Modernization Agent.

Orchestrates the full modernization loop:
    plan → execute → present to user → apply if approved → verify → commit/retry/revert

The controller enforces all safety constraints:
    - One file at a time.
    - 3 retry limit per plan item.
    - Memory lookup before every retry.
    - Test before commit.
    - Log every action.

Usage:
    from modernizer_agent.agent.controller import Controller
    ctrl = Controller(repo_path="/path/to/repo", goal="Add type hints")
    ctrl.run()
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from modernizer_agent.agent.executor import Executor, RecommendedChange
from modernizer_agent.agent.memory import MemoryStore
from modernizer_agent.agent.planner import ModernizationPlan, PlanItem, Planner
from modernizer_agent.agent.verifier import Verifier
from modernizer_agent.config import DATABASE_PATH, MAX_RETRIES_PER_ITEM
from modernizer_agent.llm.ollama_client import OllamaClient
from modernizer_agent.tools.git_tools import (
    commit,
    create_branch,
    is_git_repo,
)
from modernizer_agent.utils.logger import get_logger

log = get_logger("modernizer_agent.agent.controller")

# ANSI colours for terminal output.
_GREEN = "\033[92m"
_RED = "\033[91m"
_YELLOW = "\033[93m"
_CYAN = "\033[96m"
_BOLD = "\033[1m"
_RESET = "\033[0m"


class ControllerState(TypedDict, total=False):
    """LangGraph state payload for orchestration."""
    idx: int
    total: int
    current_item: PlanItem
    completed: int
    skipped: int
    failed: int
    stop: bool
    quit: bool


class Controller:
    """Main orchestration loop for the modernization agent."""

    def __init__(
        self,
        repo_path: str,
        goal: str,
        max_retries: int = MAX_RETRIES_PER_ITEM,
        dry_run: bool = False,
        max_iterations: int | None = None,
    ) -> None:
        self.repo_path = repo_path
        self.goal = goal
        self.max_retries = max_retries
        self.dry_run = dry_run
        self.max_iterations = max_iterations

        # --- Initialise components ---
        self.llm = OllamaClient()
        self.memory = MemoryStore(db_path=DATABASE_PATH)
        self.planner = Planner(llm=self.llm, repo_path=repo_path)
        self.executor = Executor(llm=self.llm, repo_path=repo_path, memory=self.memory)
        self.verifier = Verifier(repo_path=repo_path)

        self._iteration = 0
        self._plan: ModernizationPlan | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Execute the full modernization loop using LangGraph orchestration."""
        log.info("Controller starting", extra={
            "action": "start", "tool": "controller", "iteration": 0,
        })

        graph = self._build_graph()
        graph.invoke({
            "idx": 0,
            "completed": 0,
            "skipped": 0,
            "failed": 0,
            "stop": False,
            "quit": False,
        })

    def _build_graph(self):
        """Build and compile the LangGraph workflow."""
        workflow = StateGraph(ControllerState)

        workflow.add_node("setup", self._setup_node)
        workflow.add_node("next_item", self._next_item_node)
        workflow.add_node("process_item", self._process_item_node)
        workflow.add_node("summary", self._summary_node)

        workflow.add_edge(START, "setup")
        workflow.add_conditional_edges(
            "setup",
            self._route_after_setup,
            {
                "next_item": "next_item",
                "end": END,
            },
        )
        workflow.add_conditional_edges(
            "next_item",
            self._route_after_next_item,
            {
                "process_item": "process_item",
                "summary": "summary",
            },
        )
        workflow.add_conditional_edges(
            "process_item",
            self._route_after_process_item,
            {
                "next_item": "next_item",
                "summary": "summary",
            },
        )
        workflow.add_edge("summary", END)

        return workflow.compile()

    def _setup_node(self, _: ControllerState) -> ControllerState:
        """Pre-flight checks, planning, user confirmation, and branch setup."""
        if not self._preflight():
            self.memory.close()
            return {"stop": True}

        print(f"\n{_CYAN}{_BOLD}▶ Generating modernization plan...{_RESET}\n")
        self._plan = self.planner.generate_plan(self.goal)

        if self._plan is None or len(self._plan) == 0:
            print(f"{_YELLOW}No changes to make. The repository may already be up to date.{_RESET}")
            self.memory.close()
            return {"stop": True}

        self._display_plan(self._plan)
        if not self._confirm("Proceed with this plan?"):
            print("Aborted by user.")
            self.memory.close()
            return {"stop": True}

        if not self.dry_run:
            branch_name = self.goal.lower().replace(" ", "-")[:40]
            branch = create_branch(branch_name, self.repo_path)
            print(f"{_GREEN}Created branch: {branch}{_RESET}\n")

        return {
            "idx": 0,
            "total": len(self._plan),
            "completed": 0,
            "skipped": 0,
            "failed": 0,
            "stop": False,
            "quit": False,
        }

    def _next_item_node(self, state: ControllerState) -> ControllerState:
        """Select the next plan item or stop when finished/capped."""
        if self._plan is None:
            return {"stop": True}

        idx = state.get("idx", 0)
        total = state.get("total", len(self._plan))

        if idx >= total:
            return {"stop": True}

        self._iteration += 1
        if self.max_iterations and self._iteration > self.max_iterations:
            print(f"\n{_YELLOW}Reached max iterations ({self.max_iterations}). Stopping.{_RESET}")
            return {"stop": True}

        item = self._plan.items[idx]
        print(f"\n{_CYAN}{_BOLD}━━━ Item {idx + 1}/{total} ━━━{_RESET}")
        print(f"  File:   {item.file}")
        print(f"  Change: {item.change}\n")

        return {"current_item": item, "stop": False}

    def _process_item_node(self, state: ControllerState) -> ControllerState:
        """Execute one plan item and update counters/state."""
        item = state.get("current_item")
        idx = state.get("idx", 0)
        total = state.get("total", 0)

        if item is None:
            return {"stop": True}

        result = self._process_item(item, idx + 1, total)
        completed = state.get("completed", 0)
        skipped = state.get("skipped", 0)
        failed = state.get("failed", 0)

        updates: ControllerState = {"idx": idx + 1}

        if result == "completed":
            updates["completed"] = completed + 1
        elif result == "skipped":
            updates["skipped"] = skipped + 1
        elif result == "failed":
            updates["failed"] = failed + 1
        elif result == "quit":
            updates["stop"] = True
            updates["quit"] = True
            updates["idx"] = idx

        return updates

    def _summary_node(self, state: ControllerState) -> ControllerState:
        """Print summary and release resources."""
        if self._plan is None:
            self.memory.close()
            return {}

        self._print_summary(
            completed=state.get("completed", 0),
            skipped=state.get("skipped", 0),
            failed=state.get("failed", 0),
            total=len(self._plan),
        )
        self.memory.close()
        return {}

    @staticmethod
    def _route_after_setup(state: ControllerState) -> str:
        """Route after setup."""
        if state.get("stop"):
            return "end"
        return "next_item"

    @staticmethod
    def _route_after_next_item(state: ControllerState) -> str:
        """Route from item selection to either processing or summary."""
        if state.get("stop"):
            return "summary"
        return "process_item"

    @staticmethod
    def _route_after_process_item(state: ControllerState) -> str:
        """Route after processing an item."""
        if state.get("stop"):
            return "summary"
        return "next_item"

    # ------------------------------------------------------------------
    # Item processing (the inner loop)
    # ------------------------------------------------------------------

    def _process_item(
        self, item: PlanItem, idx: int, total: int
    ) -> str:
        """Process a single plan item with retry logic.

        Returns: "completed", "skipped", "failed", or "quit".
        """
        for attempt in range(1, self.max_retries + 1):
            self._iteration += 1 if attempt > 1 else 0

            if attempt > 1:
                print(f"\n  {_YELLOW}Retry {attempt}/{self.max_retries}{_RESET}")

            # Get memory hints for this error (if retrying).
            memory_hints = ""
            if attempt > 1:
                memory_hints = self._get_memory_hints(item)

            # --- Generate recommended change ---
            log.info(f"Generating change for {item.file}", extra={
                "action": "generate", "tool": "controller",
                "iteration": self._iteration, "plan_item": item.change,
            })
            change = self.executor.generate_change(item, memory_hints)

            if not change.has_changes:
                print(f"  {_YELLOW}No changes generated for this item.{_RESET}")
                return "skipped"

            # --- Display diff to user ---
            self._display_change(change)

            # --- Ask for user decision ---
            decision = self._ask_decision()

            if decision == "skip":
                return "skipped"
            elif decision == "quit":
                return "quit"
            elif decision == "apply":
                # Apply, verify, and commit.
                result = self._apply_and_verify(change, attempt)
                if result == "success":
                    return "completed"
                elif result == "retry" and attempt < self.max_retries:
                    continue
                else:
                    # Max retries exhausted.
                    print(f"  {_RED}Max retries reached. Skipping this item.{_RESET}")
                    return "failed"

        return "failed"

    def _apply_and_verify(
        self, change: RecommendedChange, attempt: int
    ) -> str:
        """Apply a change, run verification, commit or revert.

        Returns: "success" or "retry".
        """
        if self.dry_run:
            print(f"  {_YELLOW}[DRY RUN] Would apply change to {change.plan_item.file}{_RESET}")
            return "success"

        # Apply the change.
        self.executor.apply_change(change)
        print(f"  {_CYAN}Applied change. Running verification...{_RESET}")

        # Verify.
        result = self.verifier.verify()
        print(f"  Verification: {result.summary}")

        log.info("Verification result", extra={
            "action": "verify", "tool": "controller",
            "iteration": self._iteration,
            "result": "pass" if result.all_passed else "fail",
        })

        if result.all_passed:
            # Commit.
            msg = f"modernize: {change.plan_item.change} ({change.plan_item.file})"
            commit_hash = commit(msg, self.repo_path)
            print(f"  {_GREEN}✔ Committed: {commit_hash}{_RESET}")

            # Store success in memory.
            self.memory.store_fix(
                error_text="",
                applied_fix=change.plan_item.change,
                success=True,
                file_path=change.plan_item.file,
            )

            log.info("Change committed", extra={
                "action": "commit", "tool": "controller",
                "iteration": self._iteration,
                "commit_hash": commit_hash,
            })
            return "success"
        else:
            # Revert the applied change.
            print(f"  {_RED}✘ Verification failed. Reverting...{_RESET}")
            self._revert_file(change)

            # Store failure in memory.
            self.memory.store_fix(
                error_text=result.error_text,
                applied_fix=change.plan_item.change,
                success=False,
                file_path=change.plan_item.file,
            )

            log.info("Change reverted after failure", extra={
                "action": "revert", "tool": "controller",
                "iteration": self._iteration,
                "error": result.error_text[:200],
            })
            return "retry"

    # ------------------------------------------------------------------
    # User interaction helpers
    # ------------------------------------------------------------------

    def _display_plan(self, plan: ModernizationPlan) -> None:
        """Print the plan in a readable format."""
        print(f"{_BOLD}Modernization Plan — {plan.goal}{_RESET}")
        print(f"Total changes: {len(plan)}\n")
        for i, item in enumerate(plan.items, 1):
            print(f"  {_CYAN}{i}.{_RESET} [{item.file}]")
            print(f"     {item.change}")
        print()

    def _display_change(self, change: RecommendedChange) -> None:
        """Print the recommended change with a coloured diff."""
        print(f"\n  {_BOLD}Explanation:{_RESET} {change.explanation}\n")

        if change.diff:
            print(f"  {_BOLD}Diff:{_RESET}")
            for line in change.diff.splitlines():
                if line.startswith("+") and not line.startswith("+++"):
                    print(f"  {_GREEN}{line}{_RESET}")
                elif line.startswith("-") and not line.startswith("---"):
                    print(f"  {_RED}{line}{_RESET}")
                else:
                    print(f"  {line}")
            print()
        else:
            print(f"  {_YELLOW}(no diff available){_RESET}\n")

    @staticmethod
    def _ask_decision() -> str:
        """Present [A]pply / [S]kip / [Q]uit and return the choice."""
        while True:
            try:
                choice = input(
                    f"  {_BOLD}[A]pply  [S]kip  [Q]uit → {_RESET}"
                ).strip().lower()
            except (EOFError, KeyboardInterrupt):
                print()
                return "quit"

            if choice in ("a", "apply"):
                return "apply"
            elif choice in ("s", "skip"):
                return "skip"
            elif choice in ("q", "quit"):
                return "quit"
            else:
                print(f"  {_YELLOW}Please enter A, S, or Q.{_RESET}")

    @staticmethod
    def _confirm(question: str) -> bool:
        """Ask a yes/no question. Returns True for yes."""
        try:
            answer = input(f"{_BOLD}{question} [Y/n] {_RESET}").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return False
        return answer in ("", "y", "yes")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _preflight(self) -> bool:
        """Run pre-flight checks before starting the loop."""
        # Check repo exists.
        if not Path(self.repo_path).is_dir():
            print(f"{_RED}Error: Repository path does not exist: {self.repo_path}{_RESET}")
            return False

        # Check it's a git repo.
        if not is_git_repo(self.repo_path):
            print(f"{_RED}Error: {self.repo_path} is not a git repository.{_RESET}")
            print(f"Initialise with: git -C {self.repo_path} init")
            return False

        return True

    def _revert_file(self, change: RecommendedChange) -> None:
        """Revert a file to its original content."""
        from modernizer_agent.tools.file_tools import write_file
        write_file(change.file_path, change.original_content, self.repo_path)

    def _get_memory_hints(self, item: PlanItem) -> str:
        """Query memory for relevant past fixes and format as hints."""
        records = self.memory.query_similar(item.change)
        if not records:
            return ""

        hints: list[str] = []
        for r in records:
            status = "WORKED" if r.success else "FAILED"
            hints.append(f"- [{status}] {r.applied_fix}")

        return "\n".join(hints)

    @staticmethod
    def _print_summary(completed: int, skipped: int, failed: int, total: int) -> None:
        """Print a final summary of the run."""
        print(f"\n{_BOLD}{'━' * 50}{_RESET}")
        print(f"{_BOLD}Run Complete{_RESET}")
        print(f"  {_GREEN}Completed : {completed}/{total}{_RESET}")
        print(f"  {_YELLOW}Skipped   : {skipped}/{total}{_RESET}")
        print(f"  {_RED}Failed    : {failed}/{total}{_RESET}")
        print(f"{_BOLD}{'━' * 50}{_RESET}\n")
