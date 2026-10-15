"""SQLite helpers: connection + audit-event writer.

The `audit_events` table lives in the same .db file as the LangGraph
SQLite checkpointer tables — different schemas, same file.

`init_schema()` is idempotent and runs on the first connection, so students
don't need a separate setup step.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

import aiosqlite

from common.schemas import AuditEntry


SCHEMA_FILE = Path(__file__).resolve().parent.parent / "audit" / "schema.sql"


def db_path() -> str:
    """Return the SQLite file path (override with HITL_DB_PATH env var)."""
    return os.environ.get("HITL_DB_PATH", "hitl_audit.db")


async def _ensure_schema(conn: aiosqlite.Connection) -> None:
    async with conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='audit_events'"
    ) as cur:
        exists = await cur.fetchone() is not None
    if exists:
        async with conn.execute("PRAGMA table_info(audit_events)") as cur:
            columns = {row[1] for row in await cur.fetchall()}
        if "event_key" not in columns:
            await conn.execute("ALTER TABLE audit_events ADD COLUMN event_key TEXT")
    await conn.executescript(SCHEMA_FILE.read_text())
    await conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS audit_events_dedupe_idx "
        "ON audit_events (thread_id, event_key) WHERE event_key IS NOT NULL"
    )
    await conn.commit()


@asynccontextmanager
async def db_conn() -> AsyncIterator[aiosqlite.Connection]:
    """Open an aiosqlite connection, applying schema if first use."""
    conn = await aiosqlite.connect(db_path())
    conn.row_factory = aiosqlite.Row
    try:
        await _ensure_schema(conn)
        yield conn
    finally:
        await conn.close()


async def write_audit_event(
    *,
    thread_id: str,
    pr_url: str,
    entry: AuditEntry,
    event_key: str | None = None,
) -> None:
    """Append one structured audit row.

    `thread_id` and `pr_url` are session-context columns (used for grouping
    and filtering); all other fields come from the `AuditEntry` so they map
    1-to-1 with first-class SQL columns.
    """
    async with db_conn() as conn:
        await conn.execute(
            """
            INSERT OR IGNORE INTO audit_events (
                timestamp, thread_id, pr_url,
                agent_id, action, confidence, risk_level,
                reviewer_id, decision, reason, execution_time_ms, event_key
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry.timestamp.isoformat(), thread_id, pr_url,
                entry.agent_id, entry.action, entry.confidence, entry.risk_level,
                entry.reviewer_id, entry.decision, entry.reason, entry.execution_time_ms,
                event_key,
            ),
        )
        await conn.commit()


async def replay_events(thread_id: str) -> list[dict[str, Any]]:
    """Return every event for a thread, ordered by time. Used by audit/replay.py."""
    async with db_conn() as conn:
        async with conn.execute(
            """
            SELECT id, timestamp, agent_id, action, confidence, risk_level,
                   reviewer_id, decision, reason, execution_time_ms
              FROM audit_events
             WHERE thread_id = ?
             ORDER BY id
            """,
            (thread_id,),
        ) as cur:
            rows = await cur.fetchall()
    return [dict(r) for r in rows]


async def list_review_sessions(limit: int = 25) -> list[dict[str, Any]]:
    """Return recent sessions with a severity-aware worst-risk calculation."""
    async with db_conn() as conn:
        async with conn.execute(
            """
            SELECT thread_id,
                   pr_url,
                   MIN(timestamp) AS started,
                   MAX(timestamp) AS last_event,
                   CASE MAX(CASE risk_level
                       WHEN 'high' THEN 3 WHEN 'med' THEN 2 ELSE 1 END)
                       WHEN 3 THEN 'high' WHEN 2 THEN 'med' ELSE 'low' END AS worst_risk,
                   COUNT(*) AS events
              FROM audit_events
             GROUP BY thread_id, pr_url
             ORDER BY MAX(timestamp) DESC
             LIMIT ?
            """,
            (limit,),
        ) as cur:
            rows = await cur.fetchall()
    return [dict(row) for row in rows]
