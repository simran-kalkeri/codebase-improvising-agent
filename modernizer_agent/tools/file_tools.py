from __future__ import annotations

"""
File system tools for the Modernization Agent.

Provides safe wrappers for listing, reading, and writing files
within a target repository. All paths are validated to prevent
operations outside the repo boundary.
"""

import os
from pathlib import Path

from modernizer_agent.utils.logger import get_logger

log = get_logger("modernizer_agent.tools.file")


class FileToolError(Exception):
    """Raised when a file operation fails."""


def _validate_path(file_path: str, repo_root: str) -> Path:
    """Resolve *file_path* and ensure it lives inside *repo_root*.

    Prevents path-traversal attacks (e.g. ``../../etc/passwd``).
    """
    resolved = Path(file_path).resolve()
    root = Path(repo_root).resolve()
    if not str(resolved).startswith(str(root)):
        raise FileToolError(
            f"Path escapes repository boundary: {file_path} "
            f"(resolved to {resolved}, repo root is {root})"
        )
    return resolved


def list_files(directory: str, repo_root: str) -> list[str]:
    """Return relative paths of all files under *directory* (recursive).

    Skips hidden directories (e.g. ``.git``, ``__pycache__``).
    """
    dir_path = _validate_path(directory, repo_root)
    if not dir_path.is_dir():
        raise FileToolError(f"Not a directory: {directory}")

    result: list[str] = []
    root = Path(repo_root).resolve()

    for dirpath, dirnames, filenames in os.walk(dir_path):
        # Filter out hidden / cache directories in-place.
        dirnames[:] = [
            d for d in dirnames
            if not d.startswith(".") and d != "__pycache__"
        ]
        for fname in filenames:
            if fname.startswith("."):
                continue
            full = Path(dirpath) / fname
            result.append(str(full.relative_to(root)))

    log.info(
        f"Listed {len(result)} files in {directory}",
        extra={"tool": "list_files", "action": "list", "result": len(result)},
    )
    return sorted(result)


def read_file(file_path: str, repo_root: str) -> str:
    """Read and return the full text content of *file_path*."""
    resolved = _validate_path(file_path, repo_root)
    if not resolved.is_file():
        raise FileToolError(f"Not a file: {file_path}")

    content = resolved.read_text(encoding="utf-8")
    log.info(
        f"Read {len(content)} chars from {file_path}",
        extra={"tool": "read_file", "action": "read", "file": file_path},
    )
    return content


def write_file(file_path: str, content: str, repo_root: str) -> str:
    """Write *content* to *file_path*, creating parent dirs if needed.

    Returns the absolute path of the written file.
    """
    resolved = _validate_path(file_path, repo_root)

    # Create parent directories if they don't exist.
    resolved.parent.mkdir(parents=True, exist_ok=True)

    resolved.write_text(content, encoding="utf-8")
    log.info(
        f"Wrote {len(content)} chars to {file_path}",
        extra={"tool": "write_file", "action": "write", "file": file_path},
    )
    return str(resolved)
