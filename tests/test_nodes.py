"""Tests for the non-model nodes: load_context, safety_gate, router,
crisis, output_check. Each node is called directly with a state dict and
its returned partial update is checked; graph wiring is tested in
test_graph.py."""

import pytest
from langchain_core.messages import AIMessage, RemoveMessage

from app.memory import db
from app.memory import memory_store as ms
from app.nodes.context import compute_temporal_context, load_context
from app.nodes.crisis import CRISIS_NOTE, crisis
from app.nodes.router import route_after_router, router
from app.nodes.safety import (
    NEUTRAL_FALLBACK,
    neutrality_flags,
    output_check,
    route_after_safety,
    safety_gate,
)
from app.prompts import speaker_message
from app.resources.crisis_resources import JOINT_SESSION_INTRO
from app.state import new_session_state

NAMES = {"partner_a": "Jordan", "partner_b": "Riley"}


@pytest.fixture
def couple(tmp_db) -> int:
    return db.create_couple("couple-a", "hash", "Jordan", "Riley")


def _state(couple_id, session_id, *messages):
    state = new_session_state(couple_id, session_id, NAMES)
    state.update(messages=list(messages), speaker="partner_a")
    return state


def _safety_rows(couple_id):
    with db.get_conn() as conn:
        return [dict(r) for r in conn.execute(
            "SELECT category FROM safety_events WHERE couple_id = ?", (couple_id,))]


# --- load_context (two tests ported from v0.4 test_temporal_wiring) ---

def test_compute_temporal_context_first_session(couple):
    s = db.start_session(couple)
    now_text, ctx, gap = compute_temporal_context(couple, s)
    assert gap is None and now_text
    assert "first session" in ctx.lower()


def test_compute_temporal_context_returning_couple(couple):
    s1 = db.start_session(couple)
    db.end_session(s1)
    s2 = db.start_session(couple)
    _, ctx, gap = compute_temporal_context(couple, s2)
    assert gap == "just now"
    assert "session #2" in ctx


def test_load_context_first_turn_loads_overview(couple):
    ms.upsert_document(couple, "profile_overview", "profile_overview", "Overview",
                       "Engaged; planning a wedding.")
    update = load_context(_state(couple, db.start_session(couple)))
    assert update["profile_overview"] == "Engaged; planning a wedding."
    assert "first session" in update["temporal_context"]


def test_load_context_later_turns_only_refresh_time_and_reset(couple):
    state = _state(couple, db.start_session(couple))
    state.update(temporal_context="already set", tool_rounds=3, display=["old"])
    update = load_context(state)
    assert "temporal_context" not in update and "profile_overview" not in update
    assert update["tool_rounds"] == 0 and update["display"] == []
    assert update["now_text"]


# --- safety_gate ---

def test_safety_gate_clear_message(couple):
    s = db.start_session(couple)
    update = safety_gate(_state(couple, s, speaker_message("we argued about chores",
                                                           "partner_a", NAMES)))
    assert update["safety"]["triggered"] is False
    assert _safety_rows(couple) == []


def test_safety_gate_flags_and_logs_category_only(couple):
    s = db.start_session(couple)
    update = safety_gate(_state(couple, s, speaker_message("he hits me", "partner_b", NAMES)))
    assert update["safety"]["primary_category"] == "abuse"
    assert _safety_rows(couple) == [{"category": "abuse"}]


def test_safety_gate_checks_latest_human_message_only(couple):
    s = db.start_session(couple)
    old = speaker_message("I want to kill myself", "partner_a", NAMES)
    state = _state(couple, s, old, AIMessage(content="..."),
                   speaker_message("thanks, feeling better", "partner_a", NAMES))
    assert safety_gate(state)["safety"]["triggered"] is False


def test_route_after_safety():
    assert route_after_safety({"safety": {"triggered": True}}) == "crisis"
    assert route_after_safety({"safety": {"triggered": False}}) == "router"


# --- router ---

def test_router_defaults_to_converse():
    update = router({"active_exercise": None})
    assert update == {"route": "converse"}
    assert route_after_router(update) == "converse"


def test_router_keeps_active_exercise():
    assert router({"active_exercise": "conflict-debrief"}) == {"route": "exercise"}


# --- crisis ---

def test_crisis_abuse_in_joint_mode_uses_neutral_text():
    update = crisis({"safety": {"primary_category": "abuse"}, "joint_mode": True})
    assert JOINT_SESSION_INTRO in update["display"][0]
    assert update["messages"][0].content == CRISIS_NOTE


def test_crisis_self_harm_uses_category_text():
    update = crisis({"safety": {"primary_category": "self_harm"}, "joint_mode": True})
    assert "13 11 14" in update["display"][0]
    assert JOINT_SESSION_INTRO not in update["display"][0]


# --- output_check ---

@pytest.mark.parametrize(
    ("reply", "flag"),
    [
        ("Honestly, it sounds like he's a narcissist.", "diagnosis"),
        ("That sounds like a toxic relationship.", "diagnosis"),
        ("I think it's her fault this time.", "side_taking"),
        ("Jordan is right about the budget.", "side_taking"),
        ("riley was wrong to say that", "side_taking"),
    ],
)
def test_neutrality_flags(reply, flag):
    assert flag in neutrality_flags(reply, NAMES)


@pytest.mark.parametrize(
    "reply",
    [
        "It sounds like you both want to feel heard. Jordan, what was that like for you?",
        "You're each describing the same evening from different sides.",
        "What would feel right to try next time?",
    ],
)
def test_neutral_replies_pass(reply):
    assert neutrality_flags(reply, NAMES) == []


def test_output_check_passes_good_reply_unchanged(couple):
    s = db.start_session(couple)
    state = _state(couple, s, AIMessage(content="What did you each need then?", id="r1"))
    assert output_check(state) == {}


def test_output_check_replaces_side_taking_reply(couple):
    s = db.start_session(couple)
    state = _state(couple, s, AIMessage(content="Riley is right here.", id="r1"))
    update = output_check(state)
    removed, fallback = update["messages"]
    assert isinstance(removed, RemoveMessage) and removed.id == "r1"
    assert fallback.content == NEUTRAL_FALLBACK
    assert update["output_flags"] == ["side_taking"]


def test_output_check_replaces_empty_reply(couple):
    s = db.start_session(couple)
    update = output_check(_state(couple, s, AIMessage(content="", id="r1")))
    assert update["output_flags"] == ["empty_reply"]


def test_output_check_adds_resources_but_keeps_reply(couple):
    s = db.start_session(couple)
    reply = "If either of you ever feels suicidal, please reach out for support."
    update = output_check(_state(couple, s, AIMessage(content=reply, id="r1")))
    assert "messages" not in update  # reply kept
    assert "13 11 14" in update["display"][0]
    assert update["output_flags"] == ["safety:self_harm"]
    assert _safety_rows(couple) == [{"category": "self_harm"}]
