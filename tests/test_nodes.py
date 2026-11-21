"""Tests for the non-model nodes: load_context, safety_gate, router,
crisis, output_check. Each node is called directly with a state dict and
its returned partial update is checked; graph wiring is tested in
test_graph.py."""

import pytest
from langchain_core.messages import AIMessage, RemoveMessage

from app.memory import db
from app.memory import memory_store as ms
from app.nodes.context import compute_temporal_context, load_context
from app.nodes.crisis import CRISIS_REPLY_FALLBACK, make_crisis_node, route_after_crisis
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
from tests.fakes import FakeToolModel

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

def _crisis_state(category, *messages):
    state = new_session_state(1, 1, NAMES)
    state.update(messages=list(messages) or [speaker_message("...", "partner_a", NAMES)],
                 speaker="partner_a", safety={"triggered": True, "categories": [category],
                                              "primary_category": category})
    return state


def test_crisis_abuse_in_joint_mode_is_code_only_and_ends_session():
    model = FakeToolModel(responses=[])  # any model call would raise (and be caught)
    update = make_crisis_node(model)(_crisis_state("abuse"))
    assert model.calls == []  # the model never sees the disclosure
    assert JOINT_SESSION_INTRO in update["display"][0]
    assert update["session_ended"] is True
    assert update["safety_flags"] == ["abuse"]
    assert "messages" not in update
    assert route_after_crisis({**update}) == "end"


def test_crisis_self_harm_shows_resources_then_model_replies_without_tools():
    model = FakeToolModel(responses=[AIMessage(content="I'm really glad you said that.")])
    update = make_crisis_node(model)(_crisis_state("self_harm"))

    assert "13 11 14" in update["display"][0]
    assert JOINT_SESSION_INTRO not in update["display"][0]
    assert update["messages"][0].content == "I'm really glad you said that."
    assert update["safety_flags"] == ["self_harm"]
    call = model.calls[0]
    assert call["kwargs"]["tool_choice"] == {"type": "none"}
    assert "self-harm or suicide" in call["messages"][0].content  # the safety note
    assert route_after_crisis({"session_ended": False}) == "output_check"


def test_crisis_reply_falls_back_if_the_model_fails():
    model = FakeToolModel(responses=[])  # raises when called
    update = make_crisis_node(model)(_crisis_state("harm_to_others"))
    assert update["messages"][0].content == CRISIS_REPLY_FALLBACK
    assert "000" in update["display"][0]  # resources never depend on the model


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


def _crisis_turn(couple_id, session_id, reply_text):
    state = _state(couple_id, session_id, AIMessage(content=reply_text, id="r1"))
    state["safety"] = {"triggered": True, "categories": ["self_harm"],
                       "primary_category": "self_harm"}
    return state


def test_output_check_crisis_reply_mentioning_suicide_is_left_alone(couple):
    # Resources were already shown by `crisis`; don't show or log them twice.
    s = db.start_session(couple)
    reply = "Thoughts of suicide are really heavy to carry. Please reach out to Lifeline."
    assert output_check(_crisis_turn(couple, s, reply)) == {}
    assert _safety_rows(couple) == []


def test_output_check_bad_crisis_reply_gets_the_crisis_fallback(couple):
    s = db.start_session(couple)
    update = output_check(_crisis_turn(couple, s, ""))
    assert update["messages"][1].content == CRISIS_REPLY_FALLBACK
