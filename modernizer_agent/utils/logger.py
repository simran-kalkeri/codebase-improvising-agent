from __future__ import annotations

"""
Structured logging module for the Modernization Agent.

Produces JSON-formatted log records so every action, tool call, test result,
and memory update can be parsed programmatically.

Usage:
    from modernizer_agent.utils.logger import get_logger
    log = get_logger(__name__)
    log.info("Starting iteration", extra={"iteration": 1, "action": "plan"})
"""

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any

from modernizer_agent.config import LOG_FILE, LOG_LEVEL


class _JSONFormatter(logging.Formatter):
    """Formats every log record as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Merge any structured fields passed via `extra={...}`.
        for key in (
            "iteration",
            "action",
            "tool",
            "result",
            "commit_hash",
            "memory_update",
            "error",
            "file",
            "plan_item",
        ):
            value = getattr(record, key, None)
            if value is not None:
                entry[key] = value

        return json.dumps(entry, default=str)


def get_logger(name: str) -> logging.Logger:
    """Return a logger configured with JSON output on stderr (and optionally a file).

    Calling this multiple times with the same *name* returns the same logger
    instance (standard ``logging`` behaviour), so handlers are only attached
    once.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers on repeated calls.
    if logger.handlers:
        return logger

    logger.setLevel(LOG_LEVEL)

    # --- stderr handler ---
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(_JSONFormatter())
    logger.addHandler(stderr_handler)

    # --- optional file handler ---
    if LOG_FILE:
        file_handler = logging.FileHandler(LOG_FILE)
        file_handler.setFormatter(_JSONFormatter())
        logger.addHandler(file_handler)

    # Don't propagate to the root logger (avoids duplicate output).
    logger.propagate = False

    return logger
