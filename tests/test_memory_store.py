"""Tests for app/memory/memory_store.py: versioning, couple isolation,
scope filtering, search. The first block is v0.4's test_memory_store.py
adapted to couples (personal doc types swapped for couple ones); the rest
are new."""

import pytest

from app.memory import db
from app.memory import memory_store as ms
from app.memory.memory_store import JOINT_SESSION_SCOPES

COUPLE_ONLY = {ms.SCOPE_COUPLE}


@pytest.fixture
def couple_a(tmp_db) -> int:
    return db.create_couple("couple-a", "hash", "Jordan", "Riley")


@pytest.fixture
def couple_b(tmp_db) -> int:
    return db.create_couple("couple-b", "hash", "Sam", "Alex")


# --- ported from v0.4 ---

def test_living_document_update_supersedes_previous_version(couple_a):
    v1 = ms.upsert_document(couple_a, "shared_goals", "shared_goals", "Goals",
                            "Get married next year.")
    v2 = ms.upsert_document(couple_a, "shared_goals", "shared_goals", "Goals",
                            "Get married next year. Venue shortlisted.")
    assert (v1["version"], v2["version"]) == (1, 2)

    latest = ms.get_latest_by_key(couple_a, "shared_goals")
    assert latest["id"] == v2["id"]
    assert "Venue shortlisted" in latest["content"]

    history = ms.get_document_history(couple_a, "shared_goals")
    assert [h["version"] for h in history] == [1, 2]


def test_append_only_documents_do_not_supersede(couple_a):
    ms.upsert_document(couple_a, "session_note", "session_note:1", "S1", "Talked about chores.")
    ms.upsert_document(couple_a, "session_note", "session_note:2", "S2", "Talked about money.")
    assert len(ms.list_documents(couple_a, COUPLE_ONLY, doc_type="session_note")) == 2


def test_list_documents_excludes_superseded_versions(couple_a):
    ms.upsert_document(couple_a, "agreements", "agreements", "Agreements", "v1 content")
    ms.upsert_document(couple_a, "agreements", "agreements", "Agreements", "v2 content")
    docs = ms.list_documents(couple_a, COUPLE_ONLY, doc_type="agreements")
    assert [d["content"] for d in docs] == ["v2 content"]


def test_search_is_isolated_per_couple(couple_a, couple_b):
    ms.upsert_document(couple_a, "shared_goals", "shared_goals", "Goals", "Buy a house in Cairns.")
    ms.upsert_document(couple_b, "shared_goals", "shared_goals", "Goals", "Travel to Townsville.")
    assert len(ms.search("house Cairns", couple_a, COUPLE_ONLY)) == 1
    assert len(ms.search("house Cairns", couple_b, COUPLE_ONLY)) == 0


def test_search_includes_reference_docs_when_allowed(couple_a):
    ms.seed_reference_documents()
    results = ms.search("catastrophizing", couple_a, JOINT_SESSION_SCOPES)
    assert any(r["doc_scope"] == "reference" for r in results)


def test_search_can_exclude_reference_docs(couple_a):
    ms.seed_reference_documents()
    assert ms.search("catastrophizing", couple_a, COUPLE_ONLY) == []


def test_search_malformed_query_returns_empty_not_error(couple_a):
    assert ms.search('"unterminated', couple_a, JOINT_SESSION_SCOPES) == []


def test_couples_reference_notes_are_searchable(couple_a):
    ms.seed_reference_documents()
    results = ms.search("stonewalling", couple_a, JOINT_SESSION_SCOPES)
    assert any("four horsemen" in r["title"] for r in results)


def test_seed_reference_documents_is_idempotent(tmp_db):
    ms.seed_reference_documents()
    count_1 = len(ms.list_documents(None, {ms.SCOPE_REFERENCE}))
    ms.seed_reference_documents()
    count_2 = len(ms.list_documents(None, {ms.SCOPE_REFERENCE}))
    assert count_1 == count_2 == len(ms.REFERENCE_DOCS)


def test_get_document_returns_none_for_missing_id(tmp_db):
    assert ms.get_document(999999) is None


# --- new: scopes ---

def test_partner_private_never_returned_in_joint_session(couple_a):
    ms.upsert_document(couple_a, "session_note", "private:1", "Solo reflection",
                       "A private worry about money.", doc_scope=ms.SCOPE_PARTNER_PRIVATE)
    ms.upsert_document(couple_a, "session_note", "note:1", "Joint session",
                       "We talked about money together.")

    found = ms.search("money", couple_a, JOINT_SESSION_SCOPES)
    listed = ms.list_documents(couple_a, JOINT_SESSION_SCOPES)
    assert [r["title"] for r in found] == ["Joint session"]
    assert [d["title"] for d in listed] == ["Joint session"]


def test_no_allowed_scopes_returns_nothing(couple_a):
    ms.upsert_document(couple_a, "session_note", "note:1", "Note", "money")
    assert ms.list_documents(couple_a, set()) == []
    assert ms.search("money", couple_a, set()) == []


def test_unknown_allowed_scope_raises(couple_a):
    with pytest.raises(ValueError):
        ms.list_documents(couple_a, {"personal"})


@pytest.mark.parametrize(
    ("doc_type", "doc_scope"),
    [
        ("not_a_type", "couple"),
        ("session_note", "personal"),     # v0.4's old scope name
        ("reference", "couple"),          # reference type must be reference scope
        ("session_note", "reference"),    # and vice versa
    ],
)
def test_upsert_rejects_bad_type_or_scope(couple_a, doc_type, doc_scope):
    with pytest.raises(ValueError):
        ms.upsert_document(couple_a, doc_type, "k", "t", "c", doc_scope=doc_scope)


def test_living_versions_are_per_couple(couple_a, couple_b):
    ms.upsert_document(couple_a, "agreements", "agreements", "A", "ours")
    theirs = ms.upsert_document(couple_b, "agreements", "agreements", "A", "theirs")
    assert theirs["version"] == 1  # not superseding couple A's doc
    assert ms.get_latest_by_key(couple_a, "agreements")["content"] == "ours"


# --- new: search robustness ---

def test_search_query_with_apostrophe_still_matches(couple_a):
    # Raw "Jordan's" is invalid FTS5 syntax; v0.4 returned [] here.
    ms.upsert_document(couple_a, "conflict_patterns", "conflict_patterns", "Patterns",
                       "When Jordan's family visits, we fall into pursue-withdraw.")
    results = ms.search("Jordan's family", couple_a, COUPLE_ONLY)
    assert len(results) == 1
    assert "[family]" in results[0]["snippet"]


def test_search_fts_syntax_still_works(couple_a):
    ms.upsert_document(couple_a, "session_note", "n:1", "N1", "We argued about chores.")
    ms.upsert_document(couple_a, "session_note", "n:2", "N2", "Budget planning went well.")
    assert len(ms.search("chores OR budget", couple_a, COUPLE_ONLY)) == 2


def test_search_ignores_superseded_versions(couple_a):
    ms.upsert_document(couple_a, "agreements", "agreements", "A", "No phones at dinner.")
    ms.upsert_document(couple_a, "agreements", "agreements", "A", "Walk together on Sundays.")
    assert ms.search("phones", couple_a, COUPLE_ONLY) == []
    assert len(ms.search("Sundays", couple_a, COUPLE_ONLY)) == 1


def test_metadata_round_trip(couple_a):
    doc = ms.upsert_document(couple_a, "assessment", "assessment:das:1", "DAS",
                             "Logged externally.", metadata={"total": 112})
    assert ms.get_document(doc["id"])["metadata"] == {"total": 112}
