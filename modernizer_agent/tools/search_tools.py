"""
Code search tools for the Modernization Agent.

Tries ripgrep (``rg``) first for speed; falls back to a pure-Python
recursive grep if ``rg`` is not installed.
"""

import os
import re
import subprocess
from dataclasses import dataclass

from modernizer_agent.utils.logger import get_logger

log = get_logger("modernizer_agent.tools.search")


@dataclass
class SearchMatch:
    """A single search result."""
    file: str
    line_number: int
    line_content: str


def search_code(
    pattern: str,
    search_path: str,
    max_results: int = 50,
) -> list[SearchMatch]:
    """Search for *pattern* (regex) under *search_path*.

    Attempts ripgrep first.  If ``rg`` is not available, falls back to
    a pure-Python implementation.

    Returns at most *max_results* matches.
    """
    try:
        return _search_ripgrep(pattern, search_path, max_results)
    except FileNotFoundError:
        log.info(
            "ripgrep not found, falling back to Python search",
            extra={"tool": "search_code", "action": "fallback"},
        )
        return _search_python(pattern, search_path, max_results)


# ------------------------------------------------------------------
# Ripgrep implementation
# ------------------------------------------------------------------

def _search_ripgrep(
    pattern: str,
    search_path: str,
    max_results: int,
) -> list[SearchMatch]:
    """Search using ``rg`` (ripgrep) with JSON output."""
    cmd = [
        "rg",
        "--line-number",       # include line numbers
        "--no-heading",        # one result per line
        "--color", "never",    # no ANSI codes
        "--max-count", str(max_results),
        "--glob", "!.git",     # skip .git directory
        pattern,
        search_path,
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=30,
    )

    matches: list[SearchMatch] = []
    for line in result.stdout.splitlines():
        # ripgrep output format: <file>:<line_no>:<content>
        parts = line.split(":", 2)
        if len(parts) >= 3:
            matches.append(SearchMatch(
                file=parts[0],
                line_number=int(parts[1]),
                line_content=parts[2].strip(),
            ))

        if len(matches) >= max_results:
            break

    log.info(
        f"ripgrep found {len(matches)} matches for '{pattern}'",
        extra={"tool": "search_code", "action": "search", "result": len(matches)},
    )
    return matches


# ------------------------------------------------------------------
# Pure-Python fallback
# ------------------------------------------------------------------

_SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv"}
_BINARY_EXTENSIONS = {
    ".pyc", ".pyo", ".so", ".o", ".a", ".dylib", ".dll",
    ".exe", ".bin", ".png", ".jpg", ".jpeg", ".gif", ".ico",
    ".zip", ".tar", ".gz", ".bz2", ".whl",
}


def _search_python(
    pattern: str,
    search_path: str,
    max_results: int,
) -> list[SearchMatch]:
    """Pure-Python recursive regex search (fallback)."""
    compiled = re.compile(pattern)
    matches: list[SearchMatch] = []

    for dirpath, dirnames, filenames in os.walk(search_path):
        # Skip irrelevant directories in-place.
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]

        for fname in filenames:
            # Skip binary files by extension.
            if any(fname.endswith(ext) for ext in _BINARY_EXTENSIONS):
                continue

            filepath = os.path.join(dirpath, fname)
            try:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as fh:
                    for line_no, line in enumerate(fh, start=1):
                        if compiled.search(line):
                            matches.append(SearchMatch(
                                file=filepath,
                                line_number=line_no,
                                line_content=line.strip(),
                            ))
                            if len(matches) >= max_results:
                                break
            except (OSError, PermissionError):
                continue

            if len(matches) >= max_results:
                break

        if len(matches) >= max_results:
            break

    log.info(
        f"Python search found {len(matches)} matches for '{pattern}'",
        extra={"tool": "search_code", "action": "search", "result": len(matches)},
    )
    return matches
