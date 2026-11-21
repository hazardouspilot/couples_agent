"""Tests for cli.py: speaker parsing, and whole scripted sessions with a
fake or offline model (input() is replaced by a list of lines)."""

import pytest
from langchain_core.messages import AIMessage

import cli
from app.curator import MemoryProposal
from app.graph import make_checkpointer, thread_config
from app.memory import db
from app.memory import memory_store as ms
from app.model import OFFLINE_REPLY, OfflineChatModel
from app.resources.crisis_resources import JOINT_SESSION_INTRO
from tests.fakes import FakeToolModel

NAMES = {"partner_a": "Jordan", "partner_b": "Riley"}


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("/a", ("partner_a", "")),
        ("/b I felt left out", ("partner_b", "I felt left out")),
        ("Riley: I felt left out", ("partner_b", "I felt left out")),
        ("jordan:   ok", ("partner_a", "ok")),
        ("just talking", ("partner_b", "just talking")),          # current speaker kept
        ("Note: buy milk", ("partner_b", "Note: buy milk")),      # not a partner name
        ("/about", ("partner_b", "/about")),                      # not /a
    ],
)
def test_parse_turn(line, expected):
    assert cli.parse_turn(line, "partner_b", NAMES) == expected


@pytest.fixture
def couple(tmp_db, tmp_path):
    couple_id = db.create_couple("couple-a", "hash", "Jordan", "Riley")
    return couple_id, make_checkpointer(str(tmp_path / "cp.db"))


def _run(monkeypatch, couple, model, lines):
    feed = iter(lines)
    monkeypatch.setattr("builtins.input", lambda prompt="": next(feed))
    couple_id, cp = couple
    return cli.run_session(couple_id, "couple-a", model, cp)


def test_scripted_session_replies_and_cleans_up(monkeypatch, capsys, couple):
    model = FakeToolModel(responses=[AIMessage(content="What did you each need?")])
    result = _run(monkeypatch, couple, model, ["/b", "Jordan: we argued about chores", "/quit"])
    out = capsys.readouterr().out

    assert result == "quit"
    assert "What did you each need?" in out
    assert model.calls[0]["messages"][-1].content == "Jordan: we argued about chores"
    # Session closed and its message history deleted.
    couple_id, cp = couple
    with db.get_conn() as conn:
        (session,) = [dict(r) for r in conn.execute(
            "SELECT * FROM sessions WHERE couple_id = ?", (couple_id,))]
    assert session["ended_at"] is not None
    assert list(cp.list(thread_config(session["id"]))) == []


def _session_row(couple_id):
    with db.get_conn() as conn:
        return dict(conn.execute("SELECT * FROM sessions WHERE couple_id = ?",
                                 (couple_id,)).fetchone())


def test_quit_runs_curator_shows_notes_then_deletes_thread(monkeypatch, capsys, couple):
    model = FakeToolModel(
        responses=[AIMessage(content="A rota sounds good.")],
        structured_responses=[MemoryProposal(profile_overview="Trying a chores rota.",
                                             session_note="Agreed a rota.")])
    _run(monkeypatch, couple, model, ["let's try a chores rota", "/quit"])
    out = capsys.readouterr().out
    couple_id, cp = couple

    assert "Updated your profile overview." in out
    assert ms.get_latest_by_key(couple_id, "profile_overview")["content"] == \
        "Trying a chores rota."
    # The curator saw the session, and the thread was deleted afterwards.
    assert "let's try a chores rota" in model.calls[-1]["messages"][1].content
    assert list(cp.list(thread_config(_session_row(couple_id)["id"]))) == []


def test_ctrl_c_skips_curator_but_still_deletes_thread(monkeypatch, capsys, couple):
    def lines():
        yield "hello"
        raise KeyboardInterrupt

    feed = lines()
    monkeypatch.setattr("builtins.input", lambda prompt="": next(feed))
    couple_id, cp = couple
    model = FakeToolModel(responses=[AIMessage(content="hi")])
    cli.run_session(couple_id, "couple-a", model, cp)

    assert len(model.calls) == 1  # the turn only; no curator call
    assert list(cp.list(thread_config(_session_row(couple_id)["id"]))) == []


def test_no_curator_when_nothing_was_said(monkeypatch, capsys, couple):
    model = FakeToolModel()
    _run(monkeypatch, couple, model, ["/memory", "/quit"])
    assert model.calls == []


def test_crisis_turn_prints_resources_then_the_reply(monkeypatch, capsys, couple):
    model = FakeToolModel(responses=[AIMessage(content="Thank you for telling us.")])
    _run(monkeypatch, couple, model, ["I want to kill myself", "/quit"])
    out = capsys.readouterr().out
    assert out.index("13 11 14") < out.index("Thank you for telling us.")
    assert "SAFETY HANDOFF" not in out  # the safety note is for the model only


def test_abuse_flag_ends_the_session(monkeypatch, capsys, couple):
    # "never sent" would raise StopIteration if the loop asked for it.
    model = FakeToolModel()
    result = _run(monkeypatch, couple, model, ["he hits me"])
    out = capsys.readouterr().out
    assert result == "logout"
    assert JOINT_SESSION_INTRO in out and "session has ended" in out
    assert model.calls == []  # neither the chat model nor the curator ran
    couple_id, cp = couple
    assert ms.list_documents(couple_id, {"couple"}) == []  # nothing written to memory
    assert list(cp.list(thread_config(_session_row(couple_id)["id"]))) == []


def test_offline_mode_still_runs_the_safety_check(monkeypatch, capsys, couple):
    _run(monkeypatch, couple, OfflineChatModel(), ["hello", "I want to kill myself", "/quit"])
    out = capsys.readouterr().out
    assert OFFLINE_REPLY in out
    assert "13 11 14" in out


def test_memory_command_shows_couple_docs_only(monkeypatch, capsys, couple):
    couple_id, _ = couple
    ms.seed_reference_documents()
    ms.upsert_document(couple_id, "agreements", "agreements", "Our agreements", "Walk Sundays.")
    _run(monkeypatch, couple, FakeToolModel(), ["/memory", "/quit"])
    out = capsys.readouterr().out
    assert "Our agreements" in out
    assert "Cognitive distortions" not in out


def test_api_error_is_reported_and_session_continues(monkeypatch, capsys, couple):
    class Exploding(FakeToolModel):
        def _generate(self, *args, **kwargs):
            raise ConnectionError("network down")

    _run(monkeypatch, couple, Exploding(), ["hello", "/who", "/quit"])
    captured = capsys.readouterr()
    assert "network down" in captured.err
    assert "Speaking now: Jordan" in captured.out  # loop kept going
