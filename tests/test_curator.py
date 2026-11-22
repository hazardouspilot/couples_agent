"""Tests for the curator's code half: proposal schema, session material,
apply_proposal. Ported from counselling_agent v0.4 tests/test_curator.py
(formatting and apply tests); the model loop is tested in step 3."""

from typing import get_args

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from pydantic import ValidationError

from app.curator import (
    SAFETY_FLAG_NOTE,
    DocumentUpdate,
    MemoryProposal,
    UpdatableType,
    apply_proposal,
    curate_session,
    format_session_material,
)
from app.memory import db
from app.memory import memory_store as ms
from app.prompts import speaker_message
from tests.fakes import FakeToolModel, tool_call

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


def test_material_drops_turn_flagged_by_id_even_if_regex_misses_it():
    # A classifier-only flag: the regex can't re-derive it, so the gate's
    # recorded message ID is what keeps it out of long-term memory.
    flagged = speaker_message("wrote a few letters last night, just in case", "partner_b", NAMES)
    flagged.id = "msg-7"
    text = format_session_material([
        speaker_message("we had a rough week", "partner_a", NAMES),
        flagged,
        AIMessage(content="I'm really glad you said that. HANDOFF"),
    ], flagged_ids=["msg-7"])
    assert "letters" not in text and "HANDOFF" not in text
    assert "rough week" in text


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


def _proposal(**kw):
    return MemoryProposal(**{"profile_overview": "Overview.", "session_note": "Note.", **kw})


def test_curate_session_writes_memory_and_returns_notes(ids):
    couple, session = ids
    ms.upsert_document(couple, "agreements", "agreements", "Agreements", "Walk on Sundays.")
    model = FakeToolModel(structured_responses=[_proposal(document_updates=[
        DocumentUpdate(doc_type="agreements", title="Agreements",
                       content="Walk on Sundays. Chores rota.")])])
    messages = [speaker_message("let's try a chores rota", "partner_a", NAMES),
                AIMessage(content="That sounds like a good experiment.")]

    notes = curate_session(model, couple, session, messages,
                           now_text="Saturday, 21 November 2026", temporal_context="First session.")

    assert "Updated agreements." in notes
    assert ms.get_latest_by_key(couple, "agreements")["content"] == "Walk on Sundays. Chores rota."
    call = model.calls[0]
    assert call["kwargs"] == {"schema": "MemoryProposal", "method": "json_schema"}
    prompt = call["messages"][1].content
    assert "Walk on Sundays." in prompt                 # existing docs fetched by code
    assert "Jordan: let's try a chores rota" in prompt   # the session
    assert "TODAY: Saturday, 21 November 2026. First session." in prompt


def test_curate_session_tells_model_about_a_safety_flag_but_not_the_content(ids):
    couple, session = ids
    model = FakeToolModel(structured_responses=[_proposal()])
    messages = [speaker_message("I want to kill myself", "partner_b", NAMES),
                AIMessage(content="Thank you for telling us."),
                speaker_message("we also planned our trip", "partner_a", NAMES)]
    curate_session(model, couple, session, messages, safety_flagged=True)
    prompt = model.calls[0]["messages"][1].content
    assert SAFETY_FLAG_NOTE in prompt
    assert "kill myself" not in prompt and "planned our trip" in prompt


def test_empty_session_makes_no_model_call(ids):
    couple, session = ids
    model = FakeToolModel()
    assert curate_session(model, couple, session, []) == []
    assert model.calls == []


def test_curator_failure_never_raises(ids):
    couple, session = ids
    model = FakeToolModel(structured_responses=[ConnectionError("API down")])
    messages = [speaker_message("hello", "partner_a", NAMES)]
    assert curate_session(model, couple, session, messages) == []
    assert ms.list_documents(couple, {"couple"}) == []


def test_apply_skips_blank_fields(ids):
    couple, session = ids
    notes = apply_proposal(couple, session, MemoryProposal(
        profile_overview="  ", session_note="",
        document_updates=[DocumentUpdate(doc_type="appreciations", title="x", content=" ")]))
    assert notes == []
    assert ms.list_documents(couple, {"couple"}) == []
