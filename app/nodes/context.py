"""
`load_context`: the first node of every turn. Pure Python, no model call.

Every turn: refreshes the current time and resets the per-turn fields.
First turn of a session only: works out the session number, the time since
the last session, and loads the one always-injected memory document
(profile_overview). After that the checkpointer carries them forward.

Port of v0.4's main.compute_temporal_context. All date maths stays in
timeutil; the model only ever receives the finished sentence.
"""

from app.memory import db, memory_store, timeutil
from app.state import CoupleState


def compute_temporal_context(couple_id: int, session_id: int) -> tuple[str, str, str | None]:
    """(now_text, temporal_context, greeting_gap). greeting_gap is the bare
    "3 weeks ago" phrase for the CLI greeting, or None on a first session."""
    now = timeutil.now_iso()
    now_text = timeutil.format_local(now)
    previous = db.get_last_session_before(couple_id, session_id)
    session_number = db.count_completed_sessions(couple_id) + 1

    if previous:
        gap = timeutil.humanize_delta(previous["started_at"], reference_iso=now)
        prev_date = timeutil.format_date_local(previous["started_at"])
        return (
            now_text,
            f"This is session #{session_number}. Their previous session started "
            f"{gap} ({prev_date}).",
            gap,
        )
    return now_text, "This is the couple's first session.", None


def load_context(state: CoupleState) -> dict:
    update = {
        "now_text": timeutil.format_local(timeutil.now_iso()),
        # Per-turn fields start clean, so nothing from the last turn leaks in.
        "safety": None,
        "route": None,
        "tool_rounds": 0,
        "display": [],
        "output_flags": [],
    }
    if state.get("temporal_context") is None:  # first turn of this session
        _, temporal_context, _ = compute_temporal_context(state["couple_id"], state["session_id"])
        overview = memory_store.get_latest_by_key(state["couple_id"], "profile_overview")
        update["temporal_context"] = temporal_context
        update["profile_overview"] = overview["content"] if overview else None
    return update
