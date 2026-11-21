"""Tests for the conflict-debrief exercise: step table, a full run through
the graph with a fake model, safety mid-exercise, and the CLI handover."""

import json

import pytest
from langchain_core.messages import AIMessage

import cli
from app.graph import build_graph, make_checkpointer, thread_config
from app.memory import db
from app.nodes.exercise import (
    BOTH,
    EXERCISE_KEY,
    NOT_NOW,
    REFLECTION_FALLBACK,
    STEPS,
    expected_speaker,
)
from app.prompts import speaker_message
from app.state import new_session_state
from tests.fakes import FakeToolModel

NAMES = {"partner_a": "Jordan", "partner_b": "Riley"}


def test_step_table_shape():
    assert len(STEPS) == 13
    assert STEPS[0][:2] == ("facts", "partner_a") and STEPS[1][:2] == ("facts", "partner_b")
    assert STEPS[-1][:2] == ("agreement", BOTH)
    # Every one-partner step is asked of both partners, A then B.
    solo = [(k, w) for k, w, _ in STEPS if w != BOTH]
    assert solo[::2] == [(k, "partner_a") for k, _ in solo[::2]]
    assert solo[1::2] == [(k, "partner_b") for k, _ in solo[::2]]


@pytest.fixture
def s(tmp_db, tmp_path):
    couple = db.create_couple("couple-a", "hash", "Jordan", "Riley")
    session = db.start_session(couple)
    return {"couple": couple, "session": session, "config": thread_config(session),
            "cp": make_checkpointer(str(tmp_path / "cp.db"))}


def _start(s):
    return {**new_session_state(s["couple"], s["session"], NAMES),
            "messages": [speaker_message("Let's do a conflict debrief.", "partner_a", NAMES)],
            "speaker": "partner_a", "active_exercise": EXERCISE_KEY, "exercise_step": -1}


def _answer(state, text):
    who = expected_speaker(state) or state["speaker"]
    return {"messages": [speaker_message(text, who, NAMES)], "speaker": who}


def _records(couple):
    return [json.loads(r["data_json"]) for r in db.list_exercise_records(couple)]


def test_full_debrief_runs_code_questions_then_one_model_reflection(s):
    model = FakeToolModel(responses=[AIMessage(content="You both worked hard on that.")])
    graph = build_graph(model, s["cp"])
    state = graph.invoke(_start(s), s["config"])
    assert "Jordan: what happened?" in state["messages"][-1].content
    assert expected_speaker(state) == "partner_a"

    for i in range(len(STEPS)):
        state = graph.invoke(_answer(state, f"answer {i}"), s["config"])

    # No model call for the 13 questions; exactly one for the reflection.
    assert len(model.calls) == 1
    call = model.calls[0]
    assert call["kwargs"]["tool_choice"] == {"type": "none"}
    assert "facts (Jordan): answer 0" in call["messages"][0].content
    assert "agreement (together): answer 12" in call["messages"][0].content
    assert state["messages"][-1].content == "You both worked hard on that."
    # Saved by code, then the exercise is cleared.
    (record,) = _records(s["couple"])
    assert len(record) == len(STEPS) and record["facts (Riley)"] == "answer 1"
    assert state["active_exercise"] is None and state["exercise_answers"] == {}


def test_safety_flag_mid_exercise_stops_it_and_saves_nothing(s):
    model = FakeToolModel(responses=[AIMessage(content="I'm glad you said that.")])
    graph = build_graph(model, s["cp"])
    state = graph.invoke(_start(s), s["config"])
    state = graph.invoke(_answer(state, "we argued about money"), s["config"])
    state = graph.invoke(_answer(state, "I want to kill myself"), s["config"])

    assert "13 11 14" in state["display"][0]
    assert state["active_exercise"] is None
    assert _records(s["couple"]) == []


def test_no_exercise_after_a_safety_flag(s):
    model = FakeToolModel(responses=[AIMessage(content="I'm here.")])
    graph = build_graph(model, s["cp"])
    first = {**_start(s), "active_exercise": None,
             "messages": [speaker_message("I want to kill myself", "partner_a", NAMES)]}
    graph.invoke(first, s["config"])
    state = graph.invoke({**_answer({"speaker": "partner_a"}, "can we do the debrief?"),
                          "active_exercise": EXERCISE_KEY, "exercise_step": -1}, s["config"])
    assert state["messages"][-1].content == NOT_NOW
    assert state["active_exercise"] is None


def test_failed_reflection_still_saves_the_answers(s):
    graph = build_graph(FakeToolModel(responses=[]), s["cp"])  # model call raises
    state = graph.invoke(_start(s), s["config"])
    for i in range(len(STEPS)):
        state = graph.invoke(_answer(state, f"a{i}"), s["config"])
    assert state["messages"][-1].content == REFLECTION_FALLBACK
    assert len(_records(s["couple"])) == 1


# --- CLI ---

def _run(monkeypatch, couple_id, cp, model, lines):
    feed = iter(lines)
    monkeypatch.setattr("builtins.input", lambda prompt="": next(feed))
    return cli.run_session(couple_id, "couple-a", model, cp)


def test_cli_debrief_hands_over_and_stop_leaves(monkeypatch, capsys, s):
    model = FakeToolModel(responses=[AIMessage(content="Back to chatting.")])
    _run(monkeypatch, s["couple"], s["cp"], model,
         ["/debrief", "we argued about dishes", "/stop", "ok what now?", "/quit"])
    out = capsys.readouterr().out
    assert "Jordan: what happened?" in out
    assert "[Over to Riley.]" in out and "Riley: same for you" in out
    assert "Stopped the debrief" in out
    assert "Back to chatting." in out  # after /stop, a normal converse turn
