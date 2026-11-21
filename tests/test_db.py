"""Tests for app/memory/db.py.

Ported from v0.4's test_auth_and_db.py (storage half; the auth half moves
to test_auth.py in the next step) and the session-gap helpers from
test_temporal_wiring.py. New tests cover the couples changes: partner
names, the doc_scope CHECK constraints, and /forget not touching another
couple's data.
"""

import json
import sqlite3

import pytest

from app.memory import db

# `tmp_db` (tests/conftest.py) gives every test a fresh empty database.
# Fixtures that depend on it, like `couple` below, get it too.


@pytest.fixture
def couple(tmp_db) -> int:
    return db.create_couple("jordan-and-riley", "hash", "Jordan", "Riley")


@pytest.fixture
def other_couple(tmp_db) -> int:
    return db.create_couple("sam-and-alex", "hash", "Sam", "Alex")


def _insert_doc(couple_id, doc_scope, content="note"):
    """Raw insert, so these tests don't depend on memory_store (step 4)."""
    with db.get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO memory_documents (couple_id, doc_scope, doc_type, doc_key, "
            "title, content, created_at, updated_at) "
            "VALUES (?, ?, 'session_note', 'k', 't', ?, 'x', 'x')",
            (couple_id, doc_scope, content),
        )
        conn.execute(
            "INSERT INTO memory_fts (doc_id, couple_id, doc_scope, doc_type, title, "
            "content) VALUES (?, ?, ?, 'session_note', 't', ?)",
            (cur.lastrowid, couple_id, doc_scope, content),
        )


def _rows(table, couple_id):
    with db.get_conn() as conn:
        return conn.execute(
            f"SELECT * FROM {table} WHERE couple_id = ?", (couple_id,)
        ).fetchall()


# --- couples ---

def test_couple_round_trip_with_partner_names(couple):
    row = db.get_couple_by_username("jordan-and-riley")
    assert row["id"] == couple
    assert db.get_partner_names(couple) == {"partner_a": "Jordan", "partner_b": "Riley"}


def test_unknown_couple_returns_none(tmp_db):
    assert db.get_couple_by_username("nobody") is None
    assert db.get_partner_names(999) is None


# --- sessions (incl. v0.4 test_temporal_wiring session-gap tests) ---

def test_new_couple_has_no_completed_sessions(couple):
    assert db.count_completed_sessions(couple) == 0


def test_ending_a_session_increments_completed_count(couple):
    session_id = db.start_session(couple)
    assert db.count_completed_sessions(couple) == 0  # not ended yet
    db.end_session(session_id)
    assert db.count_completed_sessions(couple) == 1


def test_first_session_has_no_previous(couple):
    session_id = db.start_session(couple)
    assert db.get_last_session_before(couple, session_id) is None


def test_second_session_finds_the_first(couple):
    s1 = db.start_session(couple)
    db.end_session(s1)
    s2 = db.start_session(couple)
    assert db.get_last_session_before(couple, s2)["id"] == s1


def test_finds_most_recent_not_just_any_previous(couple):
    s1 = db.start_session(couple)
    db.end_session(s1)
    s2 = db.start_session(couple)
    db.end_session(s2)
    s3 = db.start_session(couple)
    assert db.get_last_session_before(couple, s3)["id"] == s2


def test_first_session_started_at_is_earliest(couple):
    s1 = db.start_session(couple)
    db.end_session(s1)
    db.start_session(couple)
    with db.get_conn() as conn:
        row = conn.execute("SELECT started_at FROM sessions WHERE id = ?", (s1,)).fetchone()
    assert db.get_first_session_started_at(couple) == row["started_at"]


def test_sessions_are_isolated_per_couple(couple, other_couple):
    s_other = db.start_session(other_couple)
    db.end_session(s_other)
    s_mine = db.start_session(couple)
    assert db.get_last_session_before(couple, s_mine) is None
    assert db.count_completed_sessions(couple) == 0


# --- exercise records / safety events ---

def test_exercise_record_round_trip(couple):
    session_id = db.start_session(couple)
    data = {"partner_a:facts": "late home", "partner_b:facts": "dinner went cold"}
    db.save_exercise_record(couple, session_id, "conflict-debrief", json.dumps(data))
    records = db.list_exercise_records(couple)
    assert len(records) == 1
    assert json.loads(records[0]["data_json"]) == data


def test_safety_event_logged_as_category_only(couple):
    session_id = db.start_session(couple)
    db.log_safety_event(couple, session_id, "abuse")
    rows = _rows("safety_events", couple)
    assert len(rows) == 1
    assert rows[0]["category"] == "abuse"
    # The schema itself has nowhere to put message text or a speaker.
    assert set(rows[0].keys()) == {"id", "couple_id", "session_id", "category", "created_at"}


# --- doc_scope constraints (new for couples) ---

def test_doc_scope_rejects_unknown_value(couple):
    with pytest.raises(sqlite3.IntegrityError):
        _insert_doc(couple, "personal")  # v0.4's old scope name


def test_reference_doc_cannot_belong_to_a_couple(couple):
    with pytest.raises(sqlite3.IntegrityError):
        _insert_doc(couple, "reference")


def test_couple_doc_must_have_a_couple(tmp_db):
    with pytest.raises(sqlite3.IntegrityError):
        _insert_doc(None, "couple")


def test_failed_block_rolls_back_earlier_writes(couple):
    # A valid write followed by an invalid one in the same `with` block:
    # get_conn never reaches commit(), so the valid write is discarded too.
    with pytest.raises(sqlite3.IntegrityError), db.get_conn() as conn:
        conn.execute("INSERT INTO sessions (couple_id, started_at) VALUES (?, 'x')", (couple,))
        conn.execute("INSERT INTO sessions (couple_id, started_at) VALUES (999, 'x')")  # bad FK
    assert _rows("sessions", couple) == []


# --- /forget ---

def test_delete_couple_data_removes_everything(couple):
    session_id = db.start_session(couple)
    db.save_exercise_record(couple, session_id, "conflict-debrief", "{}")
    db.log_safety_event(couple, session_id, "self_harm")
    _insert_doc(couple, "couple", "some memory")

    db.delete_couple_data(couple)

    assert db.get_couple_by_username("jordan-and-riley") is None
    for table in ("sessions", "exercise_records", "safety_events",
                  "memory_documents", "memory_fts"):
        assert _rows(table, couple) == [], f"{table} should be empty after delete"


def test_delete_couple_data_leaves_other_couples_and_reference_docs(couple, other_couple):
    _insert_doc(couple, "couple", "ours")
    _insert_doc(other_couple, "couple", "theirs")
    _insert_doc(None, "reference", "shared theory")

    db.delete_couple_data(couple)

    assert len(_rows("memory_documents", other_couple)) == 1
    with db.get_conn() as conn:
        refs = conn.execute(
            "SELECT COUNT(*) FROM memory_documents WHERE doc_scope = 'reference'"
        ).fetchone()[0]
    assert refs == 1
