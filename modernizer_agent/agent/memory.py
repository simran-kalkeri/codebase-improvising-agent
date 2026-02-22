"""
Memory layer for the Modernization Agent (SQLite-backed).

Stores error/fix pairs so the agent can learn from past attempts:
  - Before retrying a fix, query for similar past errors.
  - Reuse strategies that worked; avoid strategies that failed.

No model fine-tuning — this is pure pattern matching on error signatures.

Usage:
    from modernizer_agent.agent.memory import MemoryStore
    mem = MemoryStore()
    mem.store_fix("SyntaxError: unexpected indent", "fix indentation", True)
    hints = mem.query_similar("SyntaxError: unexpected indent")
"""

import hashlib
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from modernizer_agent.config import DATABASE_PATH
from modernizer_agent.utils.logger import get_logger

log = get_logger("modernizer_agent.agent.memory")


@dataclass
class MemoryRecord:
    """A single error/fix record from the memory store."""
    id: int
    error_signature: str
    error_text: str
    applied_fix: str
    file_path: str
    success: bool
    timestamp: str


class MemoryStore:
    """SQLite-backed memory for error/fix pattern storage and retrieval.

    The database is created automatically on first use.
    """

    def __init__(self, db_path: str | Path = DATABASE_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._init_db()

    # ------------------------------------------------------------------
    # Database setup
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        """Create the memory table if it does not exist."""
        conn = self._get_conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS memory (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                error_signature TEXT    NOT NULL,
                error_text      TEXT    NOT NULL DEFAULT '',
                applied_fix     TEXT    NOT NULL,
                file_path       TEXT    NOT NULL DEFAULT '',
                success         INTEGER NOT NULL DEFAULT 0,
                timestamp       TEXT    NOT NULL
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_error_sig
            ON memory (error_signature)
        """)
        conn.commit()
        log.info("Memory database initialised", extra={"tool": "memory", "action": "init"})

    def _get_conn(self) -> sqlite3.Connection:
        """Return a (lazily opened) connection to the SQLite database."""
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
        return self._conn

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def store_fix(
        self,
        error_text: str,
        applied_fix: str,
        success: bool,
        file_path: str = "",
    ) -> int:
        """Record an error/fix attempt.

        Parameters
        ----------
        error_text : str
            The raw error message or traceback.
        applied_fix : str
            Description or content of the fix that was applied.
        success : bool
            Whether the fix resolved the error.
        file_path : str, optional
            The file that was being modified.

        Returns
        -------
        int
            The row id of the inserted record.
        """
        signature = self._make_signature(error_text)
        now = datetime.now(timezone.utc).isoformat()

        conn = self._get_conn()
        cursor = conn.execute(
            """
            INSERT INTO memory (error_signature, error_text, applied_fix,
                                file_path, success, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (signature, error_text, applied_fix, file_path, int(success), now),
        )
        conn.commit()
        row_id = cursor.lastrowid

        log.info(
            f"Stored fix (success={success})",
            extra={
                "tool": "memory",
                "action": "store",
                "memory_update": {
                    "id": row_id,
                    "success": success,
                    "file": file_path,
                },
            },
        )
        return row_id  # type: ignore[return-value]

    def query_similar(
        self,
        error_text: str,
        limit: int = 5,
    ) -> list[MemoryRecord]:
        """Find past fixes for errors similar to *error_text*.

        Matching strategy:
        1. Exact signature match (fastest).
        2. Keyword overlap search (fallback).

        Results are ordered: successful fixes first, then most recent.
        """
        # --- Strategy 1: exact signature ---
        signature = self._make_signature(error_text)
        conn = self._get_conn()
        rows = conn.execute(
            """
            SELECT * FROM memory
            WHERE error_signature = ?
            ORDER BY success DESC, timestamp DESC
            LIMIT ?
            """,
            (signature, limit),
        ).fetchall()

        if rows:
            records = [self._row_to_record(r) for r in rows]
            log.info(
                f"Found {len(records)} exact matches in memory",
                extra={"tool": "memory", "action": "query", "result": len(records)},
            )
            return records

        # --- Strategy 2: keyword overlap ---
        keywords = self._extract_keywords(error_text)
        if not keywords:
            return []

        # Build a LIKE query for each keyword.
        conditions = " OR ".join(["error_text LIKE ?"] * len(keywords))
        params = [f"%{kw}%" for kw in keywords]
        params.append(str(limit))  # type: ignore[arg-type]

        rows = conn.execute(
            f"""
            SELECT * FROM memory
            WHERE {conditions}
            ORDER BY success DESC, timestamp DESC
            LIMIT ?
            """,
            params,
        ).fetchall()

        records = [self._row_to_record(r) for r in rows]
        log.info(
            f"Found {len(records)} keyword matches in memory",
            extra={"tool": "memory", "action": "query", "result": len(records)},
        )
        return records

    def get_stats(self) -> dict:
        """Return summary statistics about the memory store."""
        conn = self._get_conn()
        total = conn.execute("SELECT COUNT(*) FROM memory").fetchone()[0]
        successes = conn.execute(
            "SELECT COUNT(*) FROM memory WHERE success = 1"
        ).fetchone()[0]
        failures = total - successes
        return {
            "total_records": total,
            "successful_fixes": successes,
            "failed_fixes": failures,
        }

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_signature(error_text: str) -> str:
        """Create a normalised hash signature for an error message.

        Strips line numbers and file paths so that the same logical
        error in different locations produces the same signature.
        """
        # Normalise: lowercase, strip whitespace, remove line numbers
        import re
        normalised = error_text.lower().strip()
        normalised = re.sub(r"line \d+", "line N", normalised)
        normalised = re.sub(r'file "[^"]*"', 'file "X"', normalised)
        normalised = re.sub(r"\s+", " ", normalised)
        return hashlib.sha256(normalised.encode()).hexdigest()[:16]

    @staticmethod
    def _extract_keywords(error_text: str) -> list[str]:
        """Pull meaningful keywords from an error message for fuzzy matching."""
        import re
        # Remove common noise words.
        noise = {
            "the", "a", "an", "in", "on", "at", "to", "for", "of", "is",
            "was", "error", "file", "line", "from", "import", "not", "and",
        }
        words = re.findall(r"[a-zA-Z_]\w+", error_text)
        keywords = [w.lower() for w in words if w.lower() not in noise and len(w) > 2]
        # Deduplicate while preserving order.
        seen: set[str] = set()
        unique: list[str] = []
        for kw in keywords:
            if kw not in seen:
                seen.add(kw)
                unique.append(kw)
        return unique[:10]  # Cap at 10 keywords

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> MemoryRecord:
        """Convert a database row to a MemoryRecord dataclass."""
        return MemoryRecord(
            id=row["id"],
            error_signature=row["error_signature"],
            error_text=row["error_text"],
            applied_fix=row["applied_fix"],
            file_path=row["file_path"],
            success=bool(row["success"]),
            timestamp=row["timestamp"],
        )
