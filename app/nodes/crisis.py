"""
`crisis`: where a flagged message goes instead of the model.

Minimal version (Sat midday). Code puts the verified resources in
`display` for the CLI to print; no model call, so this works even if the
API is down. A short code-written note is added to the conversation so
that on the next turn the model knows resources were shown and follows its
SAFETY HANDOFF rules, rather than carrying on as if nothing happened.

Sat afternoon adds: the model's caring follow-up reply, and the full IPV
protocol (stopping mediation for the rest of the session on `abuse`).
"""

from langchain_core.messages import AIMessage

from app.resources.crisis_resources import format_crisis_message, format_joint_session_message
from app.state import CoupleState

# Deliberately generic: it never names the category, the partner, or what
# was said, because both partners read the conversation.
CRISIS_NOTE = (
    "(The app showed support services here and paused the conversation. "
    "Follow the SAFETY HANDOFF rules from now on.)"
)


def crisis(state: CoupleState) -> dict:
    category = (state.get("safety") or {}).get("primary_category") or "general"
    if category == "abuse" and state.get("joint_mode", True):
        text = format_joint_session_message()  # IPV protocol: neutral, identical text
    else:
        text = format_crisis_message(category)
    return {"display": [text], "messages": [AIMessage(content=CRISIS_NOTE, name="safety_system")]}
