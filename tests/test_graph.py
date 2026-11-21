"""End-to-end graph tests with a fake model: no API key, no network.

These prove the properties the design depends on: a flagged message never
reaches the model, the tool loop works and is capped, bad replies are
replaced, and the checkpointer carries state between turns and can be
deleted."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.graph import (
    build_graph,
    delete_session_thread,
    make_checkpointer,
    sweep_orphan_threads,
    thread_config,
)
from app.memory import db
from app.memory import memory_store as ms
from app.nodes.converse import EARLIER_SAFETY_NOTE, MAX_TOOL_ROUNDS
from app.nodes.safety import NEUTRAL_FALLBACK
from app.prompts import speaker_message
from app.state import new_session_state
from tests.fakes import FakeToolModel, tool_call

NAMES = {"partner_a": "Jordan", "partner_b": "Riley"}


@pytest.fixture
def session(tmp_db, tmp_path):
    couple = db.create_couple("couple-a", "hash", "Jordan", "Riley")
    session_id = db.start_session(couple)
    checkpointer = make_checkpointer(str(tmp_path / "checkpoints.db"))
    return {"couple": couple, "session": session_id, "cp": checkpointer}


def _first_turn(s, text, speaker="partner_a"):
    """Input for a session's first invoke: full session state + message."""
    return {**new_session_state(s["couple"], s["session"], NAMES),
            "messages": [speaker_message(text, speaker, NAMES)], "speaker": speaker}


def _next_turn(text, speaker="partner_b"):
    """Later invokes only pass what's new; the checkpointer supplies the rest."""
    return {"messages": [speaker_message(text, speaker, NAMES)], "speaker": speaker}


def test_normal_turn_reaches_model_and_back(session):
    model = FakeToolModel(responses=[AIMessage(content="What did you each need?")])
    graph = build_graph(model, session["cp"])
    out = graph.invoke(_first_turn(session, "we argued about chores"),
                       thread_config(session["session"]))
    assert out["messages"][-1].content == "What did you each need?"
    assert out["safety"]["triggered"] is False
    assert out["display"] == []


def test_flagged_message_skips_normal_chat_and_gets_a_handoff_reply(session):
    model = FakeToolModel(responses=[AIMessage(content="I'm so glad you told us.")])
    graph = build_graph(model, session["cp"])
    out = graph.invoke(_first_turn(session, "I want to kill myself"),
                       thread_config(session["session"]))
    assert "13 11 14" in out["display"][0]
    assert out["messages"][-1].content == "I'm so glad you told us."
    # One model call, from `crisis`, with tools off: never the converse path.
    assert len(model.calls) == 1
    assert model.calls[0]["kwargs"]["tool_choice"] == {"type": "none"}
    assert out["safety_flags"] == ["self_harm"]


def test_abuse_in_joint_session_never_reaches_the_model_and_ends(session):
    model = FakeToolModel(responses=[])  # any model call would raise
    graph = build_graph(model, session["cp"])
    out = graph.invoke(_first_turn(session, "he hits me"), thread_config(session["session"]))
    assert model.calls == []
    assert out["session_ended"] is True
    assert "1800 737 732" in out["display"][0]
    assert out["messages"][-1].content == "Jordan: he hits me"  # no reply added


def test_later_turns_keep_the_safety_posture(session):
    model = FakeToolModel(responses=[AIMessage(content="I'm here."), AIMessage(content="ok")])
    graph = build_graph(model, session["cp"])
    config = thread_config(session["session"])
    graph.invoke(_first_turn(session, "I want to kill myself"), config)
    graph.invoke(_next_turn("can we keep talking?"), config)
    assert EARLIER_SAFETY_NOTE in model.calls[1]["messages"][0].content


def test_tool_loop_runs_tools_then_answers(session):
    ms.upsert_document(session["couple"], "agreements", "agreements", "Agreements",
                       "No phones at dinner.")
    model = FakeToolModel(responses=[
        tool_call("search_memory", {"query": "phones"}),
        AIMessage(content="Last time you agreed: no phones at dinner."),
    ])
    graph = build_graph(model, session["cp"])
    out = graph.invoke(_first_turn(session, "phones at dinner again"),
                       thread_config(session["session"]))

    tool_msgs = [m for m in out["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == 1 and "No [phones] at dinner" in tool_msgs[0].content
    # The model's second call saw the tool result.
    assert isinstance(model.calls[1]["messages"][-1], ToolMessage)
    assert out["messages"][-1].content.startswith("Last time")


def test_tool_rounds_are_capped(session):
    calls = [tool_call("list_memory_documents", {}, f"c{i}") for i in range(MAX_TOOL_ROUNDS)]
    model = FakeToolModel(responses=[*calls, AIMessage(content="Here's where we are.")])
    graph = build_graph(model, session["cp"])
    out = graph.invoke(_first_turn(session, "what do you remember?"),
                       thread_config(session["session"]))
    assert len(model.calls) == MAX_TOOL_ROUNDS + 1
    assert model.calls[-1]["kwargs"]["tool_choice"] == {"type": "none"}
    assert out["messages"][-1].content == "Here's where we are."


def test_side_taking_reply_is_replaced_in_history(session):
    model = FakeToolModel(responses=[AIMessage(content="Honestly, Jordan is right.")])
    graph = build_graph(model, session["cp"])
    out = graph.invoke(_first_turn(session, "who's right?"), thread_config(session["session"]))
    contents = [m.content for m in out["messages"]]
    assert "Honestly, Jordan is right." not in contents
    assert contents[-1] == NEUTRAL_FALLBACK
    assert out["output_flags"] == ["side_taking"]


def test_checkpointer_carries_state_between_turns(session):
    model = FakeToolModel(responses=[AIMessage(content="one"), AIMessage(content="two")])
    graph = build_graph(model, session["cp"])
    config = thread_config(session["session"])
    first = graph.invoke(_first_turn(session, "hi from Jordan"), config)
    second = graph.invoke(_next_turn("hi from Riley"), config)

    humans = [m.content for m in second["messages"] if isinstance(m, HumanMessage)]
    assert humans == ["Jordan: hi from Jordan", "Riley: hi from Riley"]
    assert second["couple_id"] == session["couple"]  # not re-sent; restored
    assert second["temporal_context"] == first["temporal_context"]  # computed once
    assert second["speaker"] == "partner_b"


def test_per_turn_fields_reset_after_a_crisis_turn(session):
    model = FakeToolModel(responses=[AIMessage(content="I'm here."),
                                     AIMessage(content="I'm here with you both.")])
    graph = build_graph(model, session["cp"])
    config = thread_config(session["session"])
    graph.invoke(_first_turn(session, "I want to kill myself"), config)
    out = graph.invoke(_next_turn("ok, can we keep talking?"), config)
    assert out["display"] == [] and out["safety"]["triggered"] is False
    assert out["safety_flags"] == ["self_harm"]  # session-long: NOT reset


def test_delete_session_thread_removes_message_history(session):
    model = FakeToolModel(responses=[AIMessage(content="hello")])
    graph = build_graph(model, session["cp"])
    config = thread_config(session["session"])
    graph.invoke(_first_turn(session, "a private detail"), config)
    assert graph.get_state(config).values["messages"]

    delete_session_thread(session["cp"], session["session"])
    assert graph.get_state(config).values == {}


def test_sweep_orphan_threads_keeps_only_active(session):
    model = FakeToolModel(responses=[AIMessage(content="a"), AIMessage(content="b")])
    graph = build_graph(model, session["cp"])
    other = db.start_session(session["couple"])
    graph.invoke(_first_turn(session, "x"), thread_config(session["session"]))
    graph.invoke({**_first_turn(session, "y"), "session_id": other}, thread_config(other))

    deleted = sweep_orphan_threads(session["cp"], keep_session_ids={other})
    assert deleted == [f"session-{session['session']}"]
    assert graph.get_state(thread_config(other)).values["messages"]


def test_graph_has_the_designed_shape(session):
    graph = build_graph(FakeToolModel(), session["cp"])
    nodes = set(graph.get_graph().nodes) - {"__start__", "__end__"}
    assert nodes == {"load_context", "safety_gate", "crisis", "router",
                     "converse", "tools", "output_check"}
