"""Tests for the converse node and its conditional edge, with a fake model."""

from langchain_core.messages import AIMessage, SystemMessage

from app.nodes.converse import MAX_TOOL_ROUNDS, make_converse_node, route_after_converse
from app.prompts import LOOKUPS_DONE_NOTE, speaker_message
from app.state import new_session_state
from tests.fakes import FakeToolModel, tool_call

NAMES = {"partner_a": "Jordan", "partner_b": "Riley"}


def _state(*messages, speaker="partner_a", tool_rounds=0):
    state = new_session_state(1, 1, NAMES)
    state.update(messages=list(messages), speaker=speaker, tool_rounds=tool_rounds,
                 now_text="Saturday, 21 November 2026, 14:00")
    return state


def test_sends_system_prompt_first_with_names_and_speaker():
    model = FakeToolModel(responses=[AIMessage(content="Hi both")])
    node = make_converse_node(model)
    node(_state(speaker_message("hello", "partner_b", NAMES), speaker="partner_b"))

    sent = model.calls[0]["messages"]
    assert isinstance(sent[0], SystemMessage)
    assert "Jordan and Riley" in sent[0].content
    assert "Riley is typing this turn" in sent[0].content
    assert sent[1].content == "Riley: hello"


def test_tools_are_bound_and_model_may_choose():
    model = FakeToolModel(responses=[AIMessage(content="ok")])
    make_converse_node(model)(_state(speaker_message("hi", "partner_a", NAMES)))
    kwargs = model.calls[0]["kwargs"]
    assert kwargs["tools"] == ["search_memory", "get_memory_document", "list_memory_documents"]
    assert kwargs["tool_choice"] is None  # model decides


def test_text_reply_is_appended_and_does_not_count_a_round():
    model = FakeToolModel(responses=[AIMessage(content="Thanks for sharing")])
    update = make_converse_node(model)(_state(speaker_message("hi", "partner_a", NAMES)))
    assert [m.content for m in update["messages"]] == ["Thanks for sharing"]
    assert update["tool_rounds"] == 0


def test_tool_call_counts_a_round():
    model = FakeToolModel(responses=[tool_call("search_memory", {"query": "money"})])
    update = make_converse_node(model)(_state(speaker_message("money again", "partner_a", NAMES)))
    assert update["tool_rounds"] == 1


def test_at_round_cap_tools_are_switched_off():
    model = FakeToolModel(responses=[AIMessage(content="Here's what I found")])
    state = _state(speaker_message("hi", "partner_a", NAMES), tool_rounds=MAX_TOOL_ROUNDS)
    make_converse_node(model)(state)
    assert model.calls[0]["kwargs"]["tool_choice"] == {"type": "none"}
    # And the model is told why, or it replies with nothing (found live).
    assert LOOKUPS_DONE_NOTE in model.calls[0]["messages"][0].content


def test_lookups_note_absent_below_cap():
    model = FakeToolModel(responses=[AIMessage(content="ok")])
    make_converse_node(model)(_state(speaker_message("hi", "partner_a", NAMES)))
    assert LOOKUPS_DONE_NOTE not in model.calls[0]["messages"][0].content


def test_turn_streak_reaches_the_prompt():
    msgs = []
    for _ in range(3):
        msgs += [speaker_message("more", "partner_a", NAMES), AIMessage(content="mm")]
    model = FakeToolModel(responses=[AIMessage(content="ok")])
    make_converse_node(model)(_state(*msgs))
    assert "Warmly invite Riley" in model.calls[0]["messages"][0].content


def test_route_after_converse():
    asks_for_tool = tool_call("search_memory", {"query": "x"})
    assert route_after_converse({"messages": [asks_for_tool]}) == "tools"
    assert route_after_converse({"messages": [AIMessage(content="done")]}) == "output_check"
