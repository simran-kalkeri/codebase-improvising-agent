"""
Git tools for the Modernization Agent.

Wraps common git operations (branch, commit, revert) via subprocess.
All commands are executed with *cwd* set to the target repository.
"""

import subprocess
from typing import Any

from modernizer_agent.config import GIT_BRANCH_PREFIX
from modernizer_agent.utils.logger import get_logger

log = get_logger("modernizer_agent.tools.git")


class GitToolError(Exception):
    """Raised when a git operation fails."""


def _run_git(args: list[str], cwd: str) -> str:
    """Execute ``git <args>`` in *cwd* and return stdout.

    Raises GitToolError on non-zero exit code.
    """
    cmd = ["git"] + args
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except FileNotFoundError:
        raise GitToolError("git is not installed or not on PATH")
    except subprocess.TimeoutExpired:
        raise GitToolError(f"git command timed out: {' '.join(cmd)}")

    if result.returncode != 0:
        raise GitToolError(
            f"git {' '.join(args)} failed (exit {result.returncode}): "
            f"{result.stderr.strip()}"
        )

    return result.stdout.strip()


def is_git_repo(repo_path: str) -> bool:
    """Return True if *repo_path* is inside a git repository."""
    try:
        _run_git(["rev-parse", "--is-inside-work-tree"], cwd=repo_path)
        return True
    except GitToolError:
        return False


def create_branch(branch_name: str, repo_path: str) -> str:
    """Create and switch to a new branch.

    The branch name is prefixed with ``GIT_BRANCH_PREFIX`` (e.g.
    ``modernize/add-type-hints``).  If the branch already exists the
    agent simply checks it out.

    Returns the full branch name.
    """
    full_name = f"{GIT_BRANCH_PREFIX}{branch_name}"

    # Check if branch exists already.
    try:
        _run_git(["checkout", full_name], cwd=repo_path)
        log.info(
            f"Checked out existing branch {full_name}",
            extra={"tool": "git", "action": "checkout"},
        )
    except GitToolError:
        _run_git(["checkout", "-b", full_name], cwd=repo_path)
        log.info(
            f"Created and switched to branch {full_name}",
            extra={"tool": "git", "action": "create_branch"},
        )

    return full_name


def commit(message: str, repo_path: str) -> str:
    """Stage all changes and create a commit.

    Returns the short commit hash.
    """
    _run_git(["add", "-A"], cwd=repo_path)

    # Check if there is anything to commit.
    status = _run_git(["status", "--porcelain"], cwd=repo_path)
    if not status:
        log.info("Nothing to commit", extra={"tool": "git", "action": "commit"})
        return ""

    _run_git(["commit", "-m", message], cwd=repo_path)
    commit_hash = _run_git(["rev-parse", "--short", "HEAD"], cwd=repo_path)

    log.info(
        f"Committed: {commit_hash} — {message}",
        extra={
            "tool": "git",
            "action": "commit",
            "commit_hash": commit_hash,
        },
    )
    return commit_hash


def revert_last_commit(repo_path: str) -> str:
    """Revert the most recent commit (keeps the working tree clean).

    Returns the hash of the reverted commit.
    """
    reverted_hash = _run_git(["rev-parse", "--short", "HEAD"], cwd=repo_path)
    _run_git(["reset", "--hard", "HEAD~1"], cwd=repo_path)

    log.info(
        f"Reverted commit {reverted_hash}",
        extra={
            "tool": "git",
            "action": "revert",
            "commit_hash": reverted_hash,
        },
    )
    return reverted_hash


def get_current_branch(repo_path: str) -> str:
    """Return the name of the currently checked-out branch."""
    return _run_git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_path)


def get_diff(repo_path: str) -> str:
    """Return the unstaged diff of the working tree."""
    return _run_git(["diff"], cwd=repo_path)
