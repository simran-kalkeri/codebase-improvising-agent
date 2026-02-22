"""
Test and linting tools for the Modernization Agent.

Wraps pytest and ruff/flake8 via subprocess to provide structured
pass/fail results the agent can act on.

Handles missing tools gracefully:
  - Linter: tries ruff → flake8 → skips with warning
  - Tests:  tries the configured command → skips if no tests found
"""

import shutil
import subprocess
from dataclasses import dataclass

from modernizer_agent.config import LINTER_COMMAND, LINTER_FALLBACK, TEST_COMMAND
from modernizer_agent.utils.logger import get_logger

log = get_logger("modernizer_agent.tools.test")


@dataclass
class ToolResult:
    """Structured result from a test or lint run."""
    success: bool
    output: str
    return_code: int


def _is_pytest_available() -> bool:
    """Check if pytest is actually importable (not just if python3 exists)."""
    try:
        result = subprocess.run(
            ["python3", "-m", "pytest", "--version"],
            capture_output=True, text=True, timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def run_tests(repo_path: str, test_command: str = TEST_COMMAND) -> ToolResult:
    """Run the test suite in *repo_path*.

    If there are no test files, returns success (nothing to break).
    If the test runner is not installed, returns success with a warning.
    """
    # Check if pytest module is actually available.
    if "pytest" in test_command and not _is_pytest_available():
        msg = "pytest is not installed — skipping tests. Install with: pip3 install pytest"
        log.warning(msg, extra={"tool": "tests", "action": "run", "result": "skipped"})
        return ToolResult(success=True, output=msg, return_code=0)

    # For non-pytest commands, check the binary exists.
    runner = test_command.split()[0]
    if not shutil.which(runner):
        msg = f"Test runner '{runner}' not found — skipping tests."
        log.warning(msg, extra={"tool": "tests", "action": "run", "result": "skipped"})
        return ToolResult(success=True, output=msg, return_code=0)

    result = _run_tool(
        command=test_command,
        cwd=repo_path,
        tool_name="tests",
    )

    # pytest returns exit code 5 when no tests are collected — that's not a failure.
    if result.return_code == 5:
        log.info("No tests found — treating as pass", extra={
            "tool": "tests", "action": "run", "result": "no_tests",
        })
        return ToolResult(
            success=True,
            output="No tests found (this is OK).",
            return_code=0,
        )

    return result


def run_linter(repo_path: str, linter_command: str = LINTER_COMMAND) -> ToolResult:
    """Run the linter on *repo_path*.

    Tries the primary linter command first. If not installed, tries
    the fallback. If neither is available, skips with a warning.
    """
    # Try primary linter.
    primary_bin = linter_command.split()[0]
    if shutil.which(primary_bin):
        return _run_tool(
            command=linter_command,
            cwd=repo_path,
            tool_name="linter",
        )

    # Try fallback linter.
    if shutil.which(LINTER_FALLBACK):
        log.info(
            f"Primary linter '{primary_bin}' not found, using fallback '{LINTER_FALLBACK}'",
            extra={"tool": "linter", "action": "fallback"},
        )
        return _run_tool(
            command=LINTER_FALLBACK,
            cwd=repo_path,
            tool_name="linter",
        )

    # Neither available — skip gracefully.
    msg = (
        f"No linter available (tried '{primary_bin}' and '{LINTER_FALLBACK}'). "
        f"Install one with: pip3 install ruff"
    )
    log.warning(msg, extra={"tool": "linter", "action": "run", "result": "skipped"})
    return ToolResult(success=True, output=msg, return_code=0)


def _run_tool(command: str, cwd: str, tool_name: str) -> ToolResult:
    """Run *command* as a shell process in *cwd* and return a ToolResult."""
    try:
        result = subprocess.run(
            command.split(),
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except FileNotFoundError:
        msg = f"{tool_name} command not found: {command.split()[0]}"
        log.error(msg, extra={"tool": tool_name, "action": "run", "result": "error"})
        return ToolResult(success=False, output=msg, return_code=-1)
    except subprocess.TimeoutExpired:
        msg = f"{tool_name} timed out after 120s"
        log.error(msg, extra={"tool": tool_name, "action": "run", "result": "timeout"})
        return ToolResult(success=False, output=msg, return_code=-1)

    combined_output = result.stdout + result.stderr
    success = result.returncode == 0

    log.info(
        f"{tool_name} {'passed' if success else 'failed'} (exit {result.returncode})",
        extra={
            "tool": tool_name,
            "action": "run",
            "result": "pass" if success else "fail",
        },
    )

    return ToolResult(
        success=success,
        output=combined_output.strip(),
        return_code=result.returncode,
    )
