"""Tests for app/state.py. The reducer tests double as a demonstration of
what `add_messages` does, using it directly rather than through a graph."""

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph.message import add_messages

from app.state import CoupleState, new_session_state

NAMES = {"partner_a": "Jordan", "partner_b": "Riley"}


def test_add_messages_appends():
    old = [HumanMessage(content="Jordan: hi", id="1")]
    merged = add_messages(old, [AIMessage(content="Hello both", id="2")])
    assert [m.content for m in merged] == ["Jordan: hi", "Hello both"]


def test_add_messages_replaces_message_with_same_id():
    # This is how a node could rewrite a reply in place (output_check uses it).
    old = [AIMessage(content="draft", id="r1")]
    merged = add_messages(old, [AIMessage(content="final", id="r1")])
    assert [m.content for m in merged] == ["final"]


def test_new_session_state_covers_every_field_except_per_turn_inputs():
    state = new_session_state(1, 7, NAMES)
    expected = set(CoupleState.__annotations__) - {"messages", "speaker"}
    assert set(state) == expected
    assert state["joint_mode"] is True
    assert state["exercise_answers"] == {}
