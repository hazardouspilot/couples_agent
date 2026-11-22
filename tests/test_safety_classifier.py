"""The LLM safety classifier: union with the regex, fails soft, and a flag
from it takes the crisis path in the real graph. Fake model, no network."""

from langchain_core.messages import AIMessage

from app.curator import MemoryProposal, curate_session
from app.graph import build_graph, make_checkpointer, thread_config
from app.memory import db
from app.nodes.safety import check_message, check_with_classifier
from app.nodes.safety_classifier import SafetyVerdict, make_llm_classifier
from app.prompts import speaker_message
from app.state import new_session_state
from tests.fakes import FakeToolModel

NAMES = {"partner_a": "Sam", "partner_b": "Alex"}
INDIRECT = "wrote a few letters last night, just in case"  # regex misses this


def _classifier(*verdicts):
    model = FakeToolModel(structured_responses=list(verdicts))
    return model, make_llm_classifier(model)


def test_classifier_maps_verdicts_to_categories():
    _, classify = _classifier(SafetyVerdict(category="self_harm"), SafetyVerdict(category="none"))
    assert classify(INDIRECT) == ["self_harm"]
    assert classify("we argued about the dishes") == []


def test_classifier_fails_soft_on_api_error():
    _, classify = _classifier(RuntimeError("API down"))
    assert classify(INDIRECT) == []


def test_regex_hit_skips_the_classifier():
    model, classify = _classifier()  # any call would fail: no scripted responses
    result = check_with_classifier("I want to kill myself", classify)
    assert result.categories == ["self_harm"] and model.calls == []


def test_classifier_adds_a_flag_the_regex_missed():
    _, classify = _classifier(SafetyVerdict(category="self_harm"))
    assert not check_message(INDIRECT).triggered
    assert check_with_classifier(INDIRECT, classify).categories == ["self_harm"]


def test_classifier_flag_takes_the_crisis_path_with_code_resources(tmp_db, tmp_path):
    couple = db.create_couple("c", "h", "Sam", "Alex")
    session = db.start_session(couple)
    _, classify = _classifier(SafetyVerdict(category="self_harm"))
    model = FakeToolModel(responses=[AIMessage(content="I'm really glad you said that.")])
    graph = build_graph(model, make_checkpointer(str(tmp_path / "cp.db")), classify)
    out = graph.invoke({**new_session_state(couple, session, NAMES),
                        "messages": [speaker_message(INDIRECT, "partner_a", NAMES)],
                        "speaker": "partner_a"}, thread_config(session))
    assert "13 11 14" in out["display"][0]  # Lifeline, shown by code
    assert out["safety_flags"] == ["self_harm"]
    assert model.calls[0]["kwargs"]["tool_choice"] == {"type": "none"}  # crisis node, not converse


def test_classifier_only_flag_never_reaches_the_curator(tmp_db, tmp_path):
    """The gap found in the walkthrough: the curator used to re-run only the
    regex, so a disclosure only the classifier caught went into memory."""
    couple = db.create_couple("c", "h", "Sam", "Alex")
    session = db.start_session(couple)
    _, classify = _classifier(SafetyVerdict(category="none"), SafetyVerdict(category="self_harm"))
    model = FakeToolModel(responses=[AIMessage(content="What happened?"),
                                     AIMessage(content="I'm really glad you said that.")])
    graph = build_graph(model, make_checkpointer(str(tmp_path / "cp.db")), classify)
    config = thread_config(session)
    graph.invoke({**new_session_state(couple, session, NAMES),
                  "messages": [speaker_message("we argued about the bins", "partner_a", NAMES)],
                  "speaker": "partner_a"}, config)
    out = graph.invoke({"messages": [speaker_message(INDIRECT, "partner_b", NAMES)],
                        "speaker": "partner_b"}, config)

    flagged_id = out["messages"][-2].id  # Alex's message, before the handoff reply
    assert out["safety_flagged_ids"] == [flagged_id]

    curator_model = FakeToolModel(structured_responses=[
        MemoryProposal(profile_overview="", session_note="Talked about the bins.")])
    curate_session(curator_model, couple, session, out["messages"],
                   safety_flagged=True, flagged_ids=out["safety_flagged_ids"])
    seen = curator_model.calls[0]["messages"][-1].content
    assert "bins" in seen
    assert "letters" not in seen and "glad you said that" not in seen
