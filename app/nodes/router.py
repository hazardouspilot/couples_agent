"""
`router`: decides the route for a message that passed the safety gate.

Deterministic on purpose: no model call decides control flow. An active
exercise keeps the turn in the exercise (the CLI's /debrief sets
active_exercise to start one); otherwise it's conversation. Local commands
like /memory never reach the graph; the CLI handles them.
"""

from app.state import CoupleState


def router(state: CoupleState) -> dict:
    return {"route": "exercise" if state.get("active_exercise") else "converse"}


def route_after_router(state: CoupleState) -> str:
    """Conditional edge: the next node's name is the route just chosen."""
    return state["route"]
