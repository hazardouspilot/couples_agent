"""Tests for app/memory/retrieval.py, the tool-facing layer where scope is
enforced. Ported: v0.4's test_memory_tools.py and the memory-tool half of
test_temporal_wiring.py. New: couples scope rules and input robustness."""

import pytest

from app.memory import db, retrieval
from app.memory import memory_store as ms


@pytest.fixture
def couple_a(tmp_db) -> int:
    ms.seed_reference_documents()
    return db.create_couple("couple-a", "hash", "Jordan", "Riley")


@pytest.fixture
def couple_b(tmp_db) -> int:
    return db.create_couple("couple-b", "hash", "Sam", "Alex")


@pytest.fixture
def doc(couple_a) -> dict:
    return ms.upsert_document(
        couple_a, "conflict_patterns", "conflict_patterns", "Our cycle",
        "When plans change last minute, Jordan pursues and Riley withdraws.",
    )


# --- ported from v0.4 test_memory_tools.py ---

def test_search_memory_returns_snippet(couple_a, doc):
    result = retrieval.dispatch(couple_a, "search_memory", {"query": "plans"})
    assert len(result["results"]) == 1
    assert "[plans]" in result["results"][0]["snippet"]


def test_search_memory_requires_nonempty_query(couple_a):
    assert "error" in retrieval.dispatch(couple_a, "search_memory", {"query": "  "})


def test_get_memory_document_returns_full_content(couple_a, doc):
    result = retrieval.dispatch(couple_a, "get_memory_document", {"doc_id": doc["id"]})
    assert result["content"] == doc["content"]


def test_get_memory_document_blocks_other_couples_documents(couple_b, doc):
    result = retrieval.dispatch(couple_b, "get_memory_document", {"doc_id": doc["id"]})
    assert "error" in result


def test_list_memory_documents_scoped_to_couple(couple_b, doc):
    result = retrieval.dispatch(couple_b, "list_memory_documents", {})
    assert "Our cycle" not in [d["title"] for d in result["documents"]]


def test_unknown_tool_returns_error(couple_a):
    assert "error" in retrieval.dispatch(couple_a, "delete_everything", {})


# --- ported from v0.4 test_temporal_wiring.py (memory tool fields) ---

def test_search_results_include_relative_time_for_couple_docs(couple_a, doc):
    result = retrieval.search_memory(couple_a, "withdraws")
    assert result["results"][0]["relative_time"] == "just now"


def test_reference_docs_have_no_relative_time(couple_a):
    result = retrieval.search_memory(couple_a, "catastrophizing")
    refs = [r for r in result["results"] if r["doc_type"] == "reference"]
    assert refs and refs[0]["relative_time"] is None


def test_get_memory_document_reports_version_history(couple_a, doc):
    ms.upsert_document(couple_a, "conflict_patterns", "conflict_patterns", "Our cycle",
                       "Pursue-withdraw; we now call it 'the chase'.")
    result = retrieval.get_memory_document(couple_a, doc["id"])
    assert result["version_count"] == 2
    assert result["first_recorded_relative"] == "just now"


def test_list_memory_documents_includes_relative_time(couple_a, doc):
    docs = retrieval.list_memory_documents(couple_a)["documents"]
    mine = [d for d in docs if d["doc_type"] == "conflict_patterns"]
    assert mine and mine[0]["relative_time"] is not None


# --- new: scope enforcement ---

def test_partner_private_invisible_everywhere(couple_a):
    private = ms.upsert_document(couple_a, "session_note", "private:1", "Solo note",
                                 "secret worry", doc_scope=ms.SCOPE_PARTNER_PRIVATE)
    assert retrieval.search_memory(couple_a, "secret worry")["results"] == []
    listed = retrieval.list_memory_documents(couple_a)["documents"]
    assert private["id"] not in [d["doc_id"] for d in listed]
    # Even the owning couple can't fetch it by id in a joint session.
    assert "error" in retrieval.get_memory_document(couple_a, private["id"])


def test_forbidden_and_missing_give_identical_error(couple_b, doc):
    forbidden = retrieval.get_memory_document(couple_b, doc["id"])
    missing_id = doc["id"] + 1000
    missing = retrieval.get_memory_document(couple_b, missing_id)
    assert forbidden["error"] == f"No document with id {doc['id']}."
    assert missing["error"] == f"No document with id {missing_id}."


def test_any_couple_can_read_reference_docs(couple_b):
    ms.seed_reference_documents()
    refs = ms.list_documents(None, {ms.SCOPE_REFERENCE})
    result = retrieval.get_memory_document(couple_b, refs[0]["id"])
    assert "content" in result


# --- new: robustness to model-generated arguments ---

def test_doc_id_as_string_is_accepted(couple_a, doc):
    assert retrieval.get_memory_document(couple_a, str(doc["id"]))["doc_id"] == doc["id"]


def test_non_numeric_doc_id_is_an_error_not_a_crash(couple_a):
    assert "error" in retrieval.get_memory_document(couple_a, "the agreements doc")


def test_unknown_doc_type_is_an_error(couple_a):
    assert "error" in retrieval.search_memory(couple_a, "x", doc_type="career_history")
    assert "error" in retrieval.list_memory_documents(couple_a, doc_type="career_history")


def test_unexpected_argument_is_an_error_not_a_crash(couple_a):
    result = retrieval.dispatch(couple_a, "search_memory", {"query": "x", "couple_id": 2})
    assert "error" in result


def test_fetching_old_version_points_to_latest(couple_a, doc):
    v2 = ms.upsert_document(couple_a, "conflict_patterns", "conflict_patterns",
                            "Our cycle", "updated")
    result = retrieval.get_memory_document(couple_a, doc["id"])
    assert result["superseded"] is True
    assert result["latest_doc_id"] == v2["id"]


def test_long_documents_are_truncated(couple_a):
    big = ms.upsert_document(couple_a, "session_note", "n:big", "Big", "x" * 5000)
    result = retrieval.get_memory_document(couple_a, big["id"])
    assert result["truncated"] is True
    assert len(result["content"]) < 5000
