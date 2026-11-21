r"""
The per-turn graph: wiring of the nodes, plus the checkpointer.

    START -> load_context -> safety_gate --(flagged)--> crisis -> END
                                         \--(clear)---> router -> converse <-> tools
                                                                     \-> output_check -> END

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
the same thread_id resumes from that saved state. Needed for multi-turn
chat now, and for interrupt() (pausing mid-graph for input) this
afternoon. It stores message text, so the session's thread is deleted
after the session ends (data minimisation: no raw transcripts kept).
"""

import os
import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from app.nodes.context import load_context
from app.nodes.converse import make_converse_node, route_after_converse
from app.nodes.crisis import crisis
from app.nodes.router import route_after_router, router
from app.nodes.safety import output_check, route_after_safety, safety_gate
from app.state import CoupleState
from app.tools.memory import MEMORY_TOOLS


def build_graph(model, checkpointer=None):
    """Wire and compile the per-turn graph around a chat model (real or fake)."""
    g = StateGraph(CoupleState)

    g.add_node("load_context", load_context)
    g.add_node("safety_gate", safety_gate)
    g.add_node("crisis", crisis)
    g.add_node("router", router)
    g.add_node("converse", make_converse_node(model))
    g.add_node("tools", ToolNode(MEMORY_TOOLS))
    g.add_node("output_check", output_check)

    g.add_edge(START, "load_context")
    g.add_edge("load_context", "safety_gate")
    g.add_conditional_edges("safety_gate", route_after_safety,
                            {"crisis": "crisis", "router": "router"})
    g.add_edge("crisis", END)
    # Only "converse" for now; Sat afternoon adds "exercise".
    g.add_conditional_edges("router", route_after_router, {"converse": "converse"})
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
