"""Metrics: the callback sees nodes, model calls and tool calls; the cost
maths is right; labels carry no message text."""

from langchain_core.messages import AIMessage
from prometheus_client import REGISTRY, generate_latest

from app.graph import build_graph, make_checkpointer, thread_config
from app.memory import db
from app.observability import MetricsCallback, estimate_cost_aud, turn_route
from app.prompts import speaker_message
from app.state import new_session_state
from tests.fakes import FakeToolModel, tool_call

NAMES = {"partner_a": "Sam", "partner_b": "Alex"}


def _value(name, **labels):
    return REGISTRY.get_sample_value(name, labels) or 0.0


def test_callback_records_nodes_llm_calls_and_tools(tmp_db, tmp_path):
    couple = db.create_couple("c", "h", "Sam", "Alex")
    session = db.start_session(couple)
    model = FakeToolModel(responses=[tool_call("search_memory", {"query": "zzzq"}),
                                     AIMessage(content="What happened?")])
    graph = build_graph(model, make_checkpointer(str(tmp_path / "cp.db")))
    before = {
        "gate": _value("agent_node_latency_seconds_count", node="safety_gate"),
        "llm": _value("llm_requests_total", outcome="ok"),
        "tool": _value("tool_calls_total", tool="search_memory"),
        "empty": _value("retrieval_empty_total"),
    }
    config = {**thread_config(session), "callbacks": [MetricsCallback()]}
    graph.invoke({**new_session_state(couple, session, NAMES),
                  "messages": [speaker_message("a private detail", "partner_a", NAMES)],
                  "speaker": "partner_a"}, config)

    assert _value("agent_node_latency_seconds_count", node="safety_gate") == before["gate"] + 1
    assert _value("llm_requests_total", outcome="ok") == before["llm"] + 2
    assert _value("tool_calls_total", tool="search_memory") == before["tool"] + 1
    assert _value("retrieval_empty_total") == before["empty"] + 1
    assert b"private detail" not in generate_latest()  # labels never hold message text


def test_cost_estimate():
    # 1M input + 1M output on Sonnet 5.5 = $2 + $10 = $12 USD
    assert estimate_cost_aud("claude-sonnet-5-5", 1_000_000, 1_000_000) == 12 * 1.55
    assert estimate_cost_aud("unknown-model", 1000, 1000) == 0.0


def test_turn_route():
    assert turn_route({"load_context", "safety_gate", "crisis"}) == "crisis"
    assert turn_route({"router", "exercise", "output_check"}) == "exercise"
    assert turn_route({"router", "converse", "tools"}) == "converse"
