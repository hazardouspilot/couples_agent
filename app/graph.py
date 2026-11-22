r"""
The per-turn graph: wiring of the nodes, plus the checkpointer.

    START -> load_context -> safety_gate --(flagged)--> crisis --(IPV)--> END
                                         |                     \--------> output_check
                                         \--(clear)---> router -> converse <-> tools
                                                          |          \-> output_check -> END
                                                          \-> exercise --(last answer)--> reflect
                                                                       \-> output_check

The graph runs ONCE PER TURN. END ends the turn, not the session; the
checkpointer carries the state to the next turn.

LANGGRAPH CONCEPT: StateGraph
StateGraph(CoupleState) creates a graph whose nodes all share CoupleState.
add_node registers a function under a name; add_edge says "always go from
A to B"; compile() checks the wiring (no dangling nodes, every route
mapped) and returns a runnable you call with .invoke() / .stream().

LANGGRAPH CONCEPT: CONDITIONAL EDGES
add_conditional_edges(source, fn, mapping): after `source` runs, LangGraph
calls fn(state); fn returns a key, and the mapping turns the key into the
next node. The mapping also documents every possible branch, so the
compiled graph (and a diagram of it) shows all paths. The safety gate is
the only way past START, so no path reaches the model without it.

LANGGRAPH CONCEPT: ToolNode
A prebuilt node that reads the tool_calls on the last AIMessage, runs the
matching tools (injecting InjectedState arguments from the state), and
appends one ToolMessage per call.

LANGGRAPH CONCEPT: CHECKPOINTER
compile(checkpointer=...) makes LangGraph save the full state after every
node, keyed by the `thread_id` in the call's config. A later .invoke() with
the same thread_id resumes from that saved state. That's what makes
multi-turn chat work, and what lets the exercise remember which question
it's on between turns. graph.update_state(config, {...}) edits a thread's
saved state from outside the graph (the CLI's /stop uses it). It stores
message text, so the session's thread is deleted
after the session ends (data minimisation: no raw transcripts kept).
"""

import os
import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from app.nodes.context import load_context
from app.nodes.converse import make_converse_node, route_after_converse
from app.nodes.crisis import make_crisis_node, route_after_crisis
from app.nodes.exercise import exercise, make_reflect_node, route_after_exercise
from app.nodes.router import route_after_router, router
from app.nodes.safety import make_safety_gate, output_check, route_after_safety
from app.state import CoupleState
from app.tools.memory import MEMORY_TOOLS


def build_graph(model, checkpointer=None, classifier=None):
    """Wire and compile the per-turn graph around a chat model (real or fake).
    classifier: optional LLM safety classifier (text -> categories) that
    backs up the regex in safety_gate; None = regex only."""
    g = StateGraph(CoupleState)

    g.add_node("load_context", load_context)
    g.add_node("safety_gate", make_safety_gate(classifier))
    g.add_node("crisis", make_crisis_node(model))
    g.add_node("router", router)
    g.add_node("converse", make_converse_node(model))
    g.add_node("exercise", exercise)
    g.add_node("reflect", make_reflect_node(model))
    g.add_node("tools", ToolNode(MEMORY_TOOLS))
    g.add_node("output_check", output_check)

    g.add_edge(START, "load_context")
    g.add_edge("load_context", "safety_gate")
    g.add_conditional_edges("safety_gate", route_after_safety,
                            {"crisis": "crisis", "router": "router"})
    g.add_conditional_edges("crisis", route_after_crisis,
                            {"end": END, "output_check": "output_check"})
    g.add_conditional_edges("router", route_after_router,
                            {"converse": "converse", "exercise": "exercise"})
    # Exercise: a code-written question goes straight to output_check; after
    # the last answer, the model writes a reflection first.
    g.add_conditional_edges("exercise", route_after_exercise,
                            {"reflect": "reflect", "output_check": "output_check"})
    g.add_edge("reflect", "output_check")
    g.add_conditional_edges("converse", route_after_converse,
                            {"tools": "tools", "output_check": "output_check"})
    g.add_edge("tools", "converse")  # the loop back: tool results -> model
    g.add_edge("output_check", END)

    return g.compile(checkpointer=checkpointer)


# --- checkpointer and threads ---

def _checkpoint_path() -> str:
    return os.environ.get("COUPLES_AGENT_CHECKPOINTS", "checkpoints.db")


def make_checkpointer(path: str | None = None) -> SqliteSaver:
    """SqliteSaver over its own file, separate from the memory DB, so graph
    checkpoints (which hold message text) are easy to reason about and
    delete. check_same_thread=False because LangGraph may use the
    connection from worker threads."""
    conn = sqlite3.connect(path or _checkpoint_path(), check_same_thread=False)
    return SqliteSaver(conn)


def thread_id_for(session_id: int) -> str:
    return f"session-{session_id}"


def thread_config(session_id: int) -> dict:
    """The config every invoke for a session passes: one thread per session."""
    return {"configurable": {"thread_id": thread_id_for(session_id)}}


def delete_session_thread(checkpointer: SqliteSaver, session_id: int):
    """Delete every checkpoint for a session (its message history)."""
    checkpointer.delete_thread(thread_id_for(session_id))


def sweep_orphan_threads(checkpointer: SqliteSaver, keep_session_ids: set[int] = frozenset()):
    """Delete leftover threads, e.g. from a session that crashed before
    /quit. Run at startup, before any session is active."""
    keep = {thread_id_for(s) for s in keep_session_ids}
    thread_ids = {c.config["configurable"]["thread_id"] for c in checkpointer.list(None)}
    for tid in thread_ids - keep:
        checkpointer.delete_thread(tid)
    return sorted(thread_ids - keep)
