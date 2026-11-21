"""Tests for the @tool wrappers (app/tools/memory.py), including running
them through LangGraph's ToolNode the way the graph will."""

import json
from typing import Annotated, TypedDict, get_args

from langchain_core.messages import AnyMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from app.memory import db
from app.memory import memory_store as ms
from app.tools.memory import MEMORY_TOOLS, DocType, search_memory
from tests.fakes import tool_call


def test_doc_type_literal_matches_memory_store():
    assert set(get_args(DocType)) == set(ms.ALL_TYPES)


def test_model_facing_schema_hides_couple_id():
    for t in MEMORY_TOOLS:
        # tool_call_schema is what's sent to the model; InjectedState args are removed.
        props = t.tool_call_schema.model_json_schema().get("properties", {})
        assert "couple_id" not in props, t.name


def test_schema_comes_from_type_hints_and_docstring():
    props = search_memory.tool_call_schema.model_json_schema()["properties"]
    assert "query" in props and "doc_type" in props
    assert "keyword" in search_memory.description


def test_tool_node_injects_couple_id_from_state(tmp_db):
    mine = db.create_couple("couple-a", "hash", "Jordan", "Riley")
    other = db.create_couple("couple-b", "hash", "Sam", "Alex")
    ms.upsert_document(mine, "agreements", "agreements", "Agreements", "No phones at dinner.")
    ms.upsert_document(other, "agreements", "agreements", "Agreements", "No phones in bed.")

    # ToolNode reads tool_calls off the last AIMessage, runs them, and
    # returns ToolMessages. couple_id comes from the state, not the model.
    # ToolNode needs LangGraph's runtime, so run it in a one-node graph.
    class MiniState(TypedDict):
        messages: Annotated[list[AnyMessage], add_messages]
        couple_id: int

    graph = StateGraph(MiniState)
    graph.add_node("tools", ToolNode(MEMORY_TOOLS))
    graph.add_edge(START, "tools")
    graph.add_edge("tools", END)
    out = graph.compile().invoke({
        "messages": [tool_call("search_memory", {"query": "phones"})],
        "couple_id": mine,
    })
    msg = out["messages"][-1]
    assert isinstance(msg, ToolMessage) and msg.tool_call_id == "call_1"
    results = json.loads(msg.content)["results"]
    assert [r["snippet"] for r in results] == ["No [phones] at dinner."]
