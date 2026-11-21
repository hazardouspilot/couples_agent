"""
CoupleState: the shared data every node in the graph reads and updates.

LANGGRAPH CONCEPT: STATE
A LangGraph graph is a set of nodes (plain Python functions) that all
share one state object. A node receives the current state and returns a
PARTIAL update: a dict containing only the keys it wants to change.
LangGraph merges that update into the state, then moves to the next node.
Nodes never mutate the state directly.

LANGGRAPH CONCEPT: REDUCERS
How an update is merged is decided per key:
  - Default: OVERWRITE. If a node returns {"route": "converse"}, the old
    route is replaced.
  - With a reducer: the key is declared as Annotated[type, reducer_fn], and
    LangGraph calls reducer_fn(old_value, update) to get the new value.
`messages` uses the built-in `add_messages` reducer, which APPENDS new
messages (and replaces one in place if the update has the same message id).
That's why a node can return {"messages": [reply]} and the conversation
grows instead of being wiped.

With a checkpointer (graph.py), this whole state is saved after every node
under the session's thread_id, so the next turn starts from it.
"""

from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

Speaker = Literal["partner_a", "partner_b"]
Route = Literal["crisis", "exercise", "converse"]


class SafetyState(TypedDict):
    """Output of the safety regex as stored in state: categories only,
    never the matched text (see SafetyResult.as_state)."""
    triggered: bool
    categories: list[str]
    primary_category: str | None


class CoupleState(TypedDict):
    # Conversation so far. Reducer: append (see module docstring).
    messages: Annotated[list[AnyMessage], add_messages]

    # Set once per session by the CLI (all overwrite on update).
    couple_id: int
    session_id: int
    partner_names: dict[str, str]        # {"partner_a": "Jordan", "partner_b": "Riley"}
    joint_mode: bool                     # True in v1; drives the IPV display rules

    # Set every turn by the CLI.
    speaker: Speaker                     # who typed this turn

    # Set by load_context. All computed in Python (timeutil), never by the model.
    now_text: str | None                 # current date/time, refreshed every turn
    temporal_context: str | None         # "session #4, last one 12 days ago" (once)
    profile_overview: str | None         # the one memory doc injected up front (once)

    # Set per turn by nodes.
    safety: SafetyState | None           # safety_gate
    route: Route | None                  # router
    tool_rounds: int                     # converse; reset to 0 each turn by load_context

    # Exercise (Sat afternoon). String keys like "partner_a:facts", not
    # tuples: the checkpointer serialises state, and tuple keys don't survive.
    active_exercise: str | None
    exercise_answers: dict[str, str]


def new_session_state(couple_id: int, session_id: int,
                      partner_names: dict[str, str]) -> dict:
    """Everything except messages/speaker, for the first invoke of a
    session. Later turns only pass the new message and speaker; the
    checkpointer supplies the rest."""
    return {
        "couple_id": couple_id,
        "session_id": session_id,
        "partner_names": partner_names,
        "joint_mode": True,
        "now_text": None,
        "temporal_context": None,
        "profile_overview": None,
        "safety": None,
        "route": None,
        "tool_rounds": 0,
        "active_exercise": None,
        "exercise_answers": {},
    }
