"""Tests for the curator's code half: proposal schema, session material,
apply_proposal. Ported from counselling_agent v0.4 tests/test_curator.py
(formatting and apply tests); the model loop is tested in step 3."""

from typing import get_args

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from pydantic import ValidationError

from app.curator import (
    DocumentUpdate,
    MemoryProposal,
    UpdatableType,
    apply_proposal,
    format_session_material,
)
from app.memory import db
from app.memory import memory_store as ms
from app.prompts import speaker_message
from tests.fakes import tool_call

NAMES = {"partner_a": "Jordan", "partner_b": "Riley"}


def test_updatable_types_are_living_types_minus_overview():
    assert set(get_args(UpdatableType)) == ms.LIVING_TYPES - {"profile_overview"}


def test_proposal_rejects_unknown_doc_type():
    with pytest.raises(ValidationError):
        DocumentUpdate(doc_type="session_note", title="x", content="y")


# --- format_session_material ---

def test_material_has_both_partners_and_replies():
    text = format_session_material([
        speaker_message("we argued about chores", "partner_a", NAMES),
        AIMessage(content="What did you each need?"),
        speaker_message("to feel helped", "partner_b", NAMES),
    ])
    assert text.splitlines() == ["Jordan: we argued about chores",
                                 "Companion: What did you each need?",
                                 "Riley: to feel helped"]


def test_material_skips_tool_traffic():
    text = format_session_material([
        speaker_message("what did we agree?", "partner_a", NAMES),
        tool_call("search_memory", {"query": "agree"}),
        ToolMessage(content='{"results": ["SECRET-ish memory"]}', tool_call_id="call_1"),
        AIMessage(content="You agreed to walk on Sundays."),
    ])
    assert "SECRET-ish" not in text and "search_memory" not in text
    assert "walk on Sundays" in text


def test_material_drops_flagged_turn_and_its_reply():
    text = format_session_material([
        speaker_message("we had a rough week", "partner_a", NAMES),
        speaker_message("I want to kill myself", "partner_b", NAMES),
        AIMessage(content="Thank you for telling us. CRISIS REPLY"),
        speaker_message("thanks, can we talk about chores", "partner_b", NAMES),
        AIMessage(content="Of course."),
    ])
    assert "kill myself" not in text and "CRISIS REPLY" not in text
    assert "rough week" in text and "Of course." in text


def test_material_includes_exercise_records():
    records = [{"exercise_key": "conflict-debrief", "data": {"agreement": "Pause at 10pm"}}]
    text = format_session_material([], records)
    assert "conflict-debrief" in text and "Pause at 10pm" in text


def test_empty_session_is_empty_string():
    assert format_session_material([]) == ""


# --- apply_proposal ---

@pytest.fixture
def ids(tmp_db):
    couple = db.create_couple("couple-a", "hash", "Jordan", "Riley")
    return couple, db.start_session(couple)


def test_apply_writes_documents_and_returns_notes(ids):
    couple, session = ids
    proposal = MemoryProposal(
        profile_overview="Working on evening routines.",
        session_note="Talked about chores; agreed to try a rota.",
        document_updates=[DocumentUpdate(doc_type="agreements", title="Agreements",
                                         content="Chores rota on Sundays.",
                                         reason="new agreement")],
    )
    notes = apply_proposal(couple, session, proposal)

    assert notes == ["Updated your profile overview.", "Logged a note for this session.",
                     "Updated agreements (new agreement)."]
    assert ms.get_latest_by_key(couple, "profile_overview")["content"] == \
        "Working on evening routines."
    agreements = ms.get_latest_by_key(couple, "agreements")
    assert agreements["content"] == "Chores rota on Sundays."
    assert agreements["doc_scope"] == "couple"


def test_apply_supersedes_living_document(ids):
    couple, session = ids
    for text in ("v1 agreements", "v2 agreements"):
        apply_proposal(couple, session, MemoryProposal(
            profile_overview="", session_note="",
            document_updates=[DocumentUpdate(doc_type="agreements", title="A", content=text)]))
    history = ms.get_document_history(couple, "agreements")
    assert [h["content"] for h in history] == ["v1 agreements", "v2 agreements"]
    assert ms.get_latest_by_key(couple, "agreements")["content"] == "v2 agreements"


def test_apply_skips_blank_fields(ids):
    couple, session = ids
    notes = apply_proposal(couple, session, MemoryProposal(
        profile_overview="  ", session_note="",
        document_updates=[DocumentUpdate(doc_type="appreciations", title="x", content=" ")]))
    assert notes == []
    assert ms.list_documents(couple, {"couple"}) == []
