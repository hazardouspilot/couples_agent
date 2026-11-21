"""
The `converse` node: the chat model, with the memory tools bound, writes
the next message. Plus the edge function that decides where to go after it.

This is where v0.4's hand-written tool loop (client.chat) went. In v0.4 one
Python `for` loop called the API, ran tools and called again. In the graph,
that loop is two nodes and an edge:

    converse --(reply has tool_calls?)--> tools --> converse   (loop)
             --(no tool_calls)----------> output_check

LANGCHAIN CONCEPT: bind_tools
`model.bind_tools(tools)` returns the same model with the tool schemas
attached to every request. The model may then answer with text, or with an
AIMessage whose `.tool_calls` lists the tools it wants run.
"""

from langchain_core.messages import AIMessage, SystemMessage

from app.prompts import build_system_prompt, turn_balance_note
from app.state import CoupleState
from app.tools.memory import MEMORY_TOOLS

MAX_TOOL_ROUNDS = 4  # as in v0.4


def make_converse_node(model):
    """Build the node around a given chat model (the real ChatAnthropic in
    the CLI, a fake in tests). A factory, so the graph doesn't import or
    construct the real model itself."""
    with_tools = model.bind_tools(MEMORY_TOOLS)
    # Past the round cap: same tools attached (the API requires tool
    # definitions whenever the history contains tool calls) but
    # tool_choice "none", so the model must answer in text.
    tools_off = model.bind_tools(MEMORY_TOOLS, tool_choice={"type": "none"})

    def converse(state: CoupleState) -> dict:
        names = state["partner_names"]
        system = build_system_prompt(
            partner_names=names,
            speaker=state.get("speaker"),
            now_text=state.get("now_text"),
            temporal_context=state.get("temporal_context"),
            profile_overview=state.get("profile_overview"),
            turn_note=turn_balance_note(state["messages"], names),
            active_exercise=state.get("active_exercise"),
        )
        messages = [SystemMessage(content=system), *state["messages"]]

        rounds = state.get("tool_rounds", 0)
        runnable = with_tools if rounds < MAX_TOOL_ROUNDS else tools_off
        reply: AIMessage = runnable.invoke(messages)

        # Partial update: append the reply (add_messages reducer) and count
        # the round if the model asked for tools.
        return {"messages": [reply], "tool_rounds": rounds + (1 if reply.tool_calls else 0)}

    return converse


def route_after_converse(state: CoupleState) -> str:
    """LANGGRAPH CONCEPT: CONDITIONAL EDGE.
    A plain function that reads the state and returns the NAME of the next
    node. graph.py registers it with add_conditional_edges. Here: if the
    model's last message asked for tools, run them; otherwise check the
    reply before it's shown."""
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return "output_check"
