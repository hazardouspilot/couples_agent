"""
Local SQLite storage: couple accounts, sessions, exercise records, safety
events, and the memory-document library (see memory_store.py for the
higher-level document API).

Ported from counselling_agent v0.4 (agent/db.py). What changed for couples:
  - `users` is now `couples`: one login per couple (SPEC: "one couple login"),
    holding both partners' display names. Every `user_id` is now `couple_id`.
  - memory_documents.scope ('personal' | 'reference') is now `doc_scope`
    ('couple' | 'partner_private' | 'reference'), and SQLite itself rejects
    any other value or an inconsistent couple_id (see the CHECK constraints).

Unchanged data-minimisation rules from v0.4:
  - No raw chat transcripts are stored. The curator folds a session into
    memory documents and the transcript is discarded. (In the LangGraph
    build the checkpointer DOES hold messages, so its thread is deleted
    after curation; that lives in the graph code, not here.)
  - Safety events store category + timestamp only: never the message, and
    in this build never which partner said it either. In a joint session a
    record of "partner A triggered an abuse flag" is itself risky data.
"""

import os
import sqlite3
from contextlib import contextmanager

from app.memory.timeutil import now_iso


def _db_path() -> str:
    # Read on every call (not once at import) so tests can point it at a
    # temp file just by setting the env var. See tests/conftest.py.
    return os.environ.get("COUPLES_AGENT_DB", "couples_agent.db")


SCHEMA = """
CREATE TABLE IF NOT EXISTS couples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    partner_a_name TEXT NOT NULL,
    partner_b_name TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    couple_id INTEGER NOT NULL REFERENCES couples(id),
    started_at TEXT NOT NULL,
    ended_at TEXT
);

CREATE TABLE IF NOT EXISTS exercise_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    couple_id INTEGER NOT NULL REFERENCES couples(id),
    session_id INTEGER NOT NULL REFERENCES sessions(id),
    exercise_key TEXT NOT NULL,
    data_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS safety_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    couple_id INTEGER NOT NULL REFERENCES couples(id),
    session_id INTEGER NOT NULL REFERENCES sessions(id),
    category TEXT NOT NULL,
    created_at TEXT NOT NULL
);

-- doc_scope decides who may ever see a document:
--   couple          : this couple's shared memory; both partners can read it
--   partner_private : one partner only (v2; never returned in v1)
--   reference       : shared theory notes, not tied to any couple
-- The second CHECK ties couple_id to scope: reference docs have no owner,
-- every other doc must have one. A bug that writes a couple's note as
-- 'reference' (making it visible to everyone) fails at INSERT time.
CREATE TABLE IF NOT EXISTS memory_documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    couple_id INTEGER REFERENCES couples(id),
    doc_scope TEXT NOT NULL
        CHECK (doc_scope IN ('couple', 'partner_private', 'reference')),
    doc_type TEXT NOT NULL,        -- profile_overview, agreements, ...
    doc_key TEXT NOT NULL,         -- stable id; groups versions of a living doc
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    metadata_json TEXT,            -- optional structured extras
    version INTEGER NOT NULL DEFAULT 1,
    superseded INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    CHECK ((doc_scope = 'reference') = (couple_id IS NULL))
);

CREATE INDEX IF NOT EXISTS idx_memdocs_lookup
    ON memory_documents(couple_id, doc_key, superseded);

-- FTS5 is SQLite's built-in full-text search: a virtual table that keeps
-- an inverted index over title + content and supports MATCH queries,
-- ranking (bm25) and snippet(). UNINDEXED columns are stored alongside
-- for filtering but not searched. `superseded` is deliberately NOT copied
-- here: search joins back to memory_documents, the single source of truth.
CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
    title, content,
    doc_id UNINDEXED,
    couple_id UNINDEXED,
    doc_scope UNINDEXED,
    doc_type UNINDEXED
);
"""


@contextmanager
def get_conn():
    """One short-lived connection per operation. Commits only if the
    `with` block finishes without an exception; otherwise close() discards
    the uncommitted changes, so a failed multi-step write leaves nothing
    half-done."""
    conn = sqlite3.connect(_db_path())
    conn.row_factory = sqlite3.Row  # rows behave like dicts: row["column"]
    conn.execute("PRAGMA foreign_keys = ON")  # SQLite leaves FKs off by default
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_conn() as conn:
        conn.executescript(SCHEMA)


# --- couples (accounts) ---

def create_couple(username: str, password_hash: str,
                  partner_a_name: str, partner_b_name: str) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO couples (username, password_hash, partner_a_name, "
            "partner_b_name, created_at) VALUES (?, ?, ?, ?, ?)",
            (username, password_hash, partner_a_name, partner_b_name, now_iso()),
        )
        return cur.lastrowid


def get_couple_by_username(username: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM couples WHERE username = ?", (username,)
        ).fetchone()
        return dict(row) if row else None


def get_partner_names(couple_id: int) -> dict[str, str] | None:
    """{'partner_a': 'Jordan', 'partner_b': 'Riley'}, the shape the graph
    state's `partner_names` field expects."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT partner_a_name, partner_b_name FROM couples WHERE id = ?",
            (couple_id,),
        ).fetchone()
    if not row:
        return None
    return {"partner_a": row["partner_a_name"], "partner_b": row["partner_b_name"]}


# --- sessions ---

def start_session(couple_id: int) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO sessions (couple_id, started_at) VALUES (?, ?)",
            (couple_id, now_iso()),
        )
        return cur.lastrowid


def end_session(session_id: int):
    with get_conn() as conn:
        conn.execute(
            "UPDATE sessions SET ended_at = ? WHERE id = ?", (now_iso(), session_id)
        )


def count_completed_sessions(couple_id: int) -> int:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM sessions "
            "WHERE couple_id = ? AND ended_at IS NOT NULL",
            (couple_id,),
        ).fetchone()
        return row["n"] if row else 0


def get_last_session_before(couple_id: int, exclude_session_id: int) -> dict | None:
    """The couple's most recent session other than the given one (normally
    the one just started). Used to compute "time since we last talked".
    None for a first-ever session.

    `id DESC` breaks ties: two sessions started within the same clock tick
    would otherwise come back in arbitrary order (v0.4 had no tiebreak)."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM sessions WHERE couple_id = ? AND id != ? "
            "ORDER BY started_at DESC, id DESC LIMIT 1",
            (couple_id, exclude_session_id),
        ).fetchone()
        return dict(row) if row else None


def get_first_session_started_at(couple_id: int) -> str | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT started_at FROM sessions WHERE couple_id = ? "
            "ORDER BY started_at ASC, id ASC LIMIT 1",
            (couple_id,),
        ).fetchone()
        return row["started_at"] if row else None


# --- exercise records ---

def save_exercise_record(couple_id: int, session_id: int, exercise_key: str,
                         data_json: str):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO exercise_records (couple_id, session_id, exercise_key, "
            "data_json, created_at) VALUES (?, ?, ?, ?, ?)",
            (couple_id, session_id, exercise_key, data_json, now_iso()),
        )


def list_exercise_records(couple_id: int, limit: int = 20) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM exercise_records WHERE couple_id = ? "
            "ORDER BY created_at DESC, id DESC LIMIT ?",
            (couple_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]


# --- safety events ---

def log_safety_event(couple_id: int, session_id: int, category: str):
    """Category + timestamp only. Deliberately no message text and no
    speaker. There is also deliberately no getter: these are never shown
    in the app (not even via /memory)."""
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO safety_events (couple_id, session_id, category, created_at) "
            "VALUES (?, ?, ?, ?)",
            (couple_id, session_id, category, now_iso()),
        )


# --- account deletion (/forget) ---

def delete_couple_data(couple_id: int):
    """Delete everything tied to a couple, all document versions included.
    Irreversible. One connection, so it's all-or-nothing (see get_conn).
    Children are deleted before parents because foreign keys are on."""
    with get_conn() as conn:
        conn.execute(
            "DELETE FROM memory_fts WHERE doc_id IN "
            "(SELECT id FROM memory_documents WHERE couple_id = ?)",
            (couple_id,),
        )
        conn.execute("DELETE FROM memory_documents WHERE couple_id = ?", (couple_id,))
        conn.execute("DELETE FROM safety_events WHERE couple_id = ?", (couple_id,))
        conn.execute("DELETE FROM exercise_records WHERE couple_id = ?", (couple_id,))
        conn.execute("DELETE FROM sessions WHERE couple_id = ?", (couple_id,))
        conn.execute("DELETE FROM couples WHERE id = ?", (couple_id,))
