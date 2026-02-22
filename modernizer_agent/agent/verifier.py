"""
Verifier module for the Modernization Agent.

Runs the linter and test suite after a change has been applied, and
returns a structured VerificationResult the controller can act on.

Usage:
    from modernizer_agent.agent.verifier import Verifier
    verifier = Verifier(repo_path="/path/to/repo")
    result = verifier.verify()
"""

from __future__ import annotations

from dataclasses import dataclass

from modernizer_agent.tools.test_tools import ToolResult, run_linter, run_tests
from modernizer_agent.utils.logger import get_logger

log = get_logger("modernizer_agent.agent.verifier")


@dataclass
class VerificationResult:
    """Outcome of running linter + tests after a change."""
    lint_passed: bool
    lint_output: str
    tests_passed: bool
    test_output: str

    @property
    def all_passed(self) -> bool:
        """True if both linter and tests passed."""
        return self.lint_passed and self.tests_passed

    @property
    def summary(self) -> str:
        """One-line human-readable summary."""
        lint = "✔ lint" if self.lint_passed else "✘ lint"
        tests = "✔ tests" if self.tests_passed else "✘ tests"
        return f"{lint}  |  {tests}"

    @property
    def error_text(self) -> str:
        """Combined error output for memory storage (empty if all passed)."""
        parts: list[str] = []
        if not self.lint_passed:
            parts.append(f"LINT ERRORS:\n{self.lint_output}")
        if not self.tests_passed:
            parts.append(f"TEST ERRORS:\n{self.test_output}")
        return "\n\n".join(parts)


class Verifier:
    """Runs linter and tests against the repository."""

    def __init__(self, repo_path: str) -> None:
        self.repo_path = repo_path

    def verify(self, run_lint: bool = True, run_test: bool = True) -> VerificationResult:
        """Run verification checks and return a VerificationResult.

        Parameters
        ----------
        run_lint : bool
            Whether to run the linter (default True).
        run_test : bool
            Whether to run the test suite (default True).
        """
        lint_result = ToolResult(success=True, output="skipped", return_code=0)
        test_result = ToolResult(success=True, output="skipped", return_code=0)

        # --- Lint ---
        if run_lint:
            log.info("Running linter", extra={
                "action": "lint", "tool": "verifier",
            })
            lint_result = run_linter(self.repo_path)

        # --- Tests ---
        if run_test:
            log.info("Running tests", extra={
                "action": "test", "tool": "verifier",
            })
            test_result = run_tests(self.repo_path)

        result = VerificationResult(
            lint_passed=lint_result.success,
            lint_output=lint_result.output,
            tests_passed=test_result.success,
            test_output=test_result.output,
        )

        log.info(
            f"Verification: {result.summary}",
            extra={
                "action": "verify_complete",
                "tool": "verifier",
                "result": "pass" if result.all_passed else "fail",
            },
        )

        return result
