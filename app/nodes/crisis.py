"""
`crisis`: where a flagged message goes instead of the normal chat path.

Code decides everything that must be right; the model only adds warmth:

  self_harm / harm_to_others (or abuse outside joint mode, v2):
    1. CODE puts the verified resources in `display` (printed first).
    2. The MODEL writes a short caring reply, under the SAFETY HANDOFF
       rules, with a safety note in its prompt and tools switched off.
       If the call fails for any reason, a fixed reply is used instead:
       the crisis path never depends on the API being up.
    3. The reply then goes through output_check like any other.

  abuse in a JOINT session (IPV / coercive-control protocol, SPEC):
    No model call at all: a model might quote the disclosure or name who
    said it, and the partner who poses a risk may be reading. Code shows
    the neutral joint-session text and sets `session_ended`; the CLI then
    closes the session and skips the curator, so nothing about it is
    written to memory both partners can read.
"""

from langchain_core.messages import AIMessage, SystemMessage

from app.prompts import build_system_prompt
from app.resources.crisis_resources import format_crisis_message, format_joint_session_message
from app.state import CoupleState
from app.tools.memory import MEMORY_TOOLS

# What the model is told (never shown to the couple). Generic on purpose:
# the category, not the words, and never which partner.
_CONCERN = {
    "self_harm": "possible thoughts of self-harm or suicide",
    "harm_to_others": "possible risk of someone being harmed",
    "abuse": "possible abuse or feeling unsafe",
}
CRISIS_SAFETY_NOTE = (
    "The safety system just detected {concern} in the latest message, and the app "
    "has already shown the couple a list of support services. Follow the SAFETY "
    "HANDOFF rules now: reply in text, briefly and warmly. Don't continue any "
    "exercise or earlier topic."
)

# Used if the model call fails, and by output_check if the model's crisis
# reply is empty or breaks a rule. Fails closed, to a known-safe message.
CRISIS_REPLY_FALLBACK = (
    "Thank you for saying something about this; it matters. The services above are "
    "free and confidential, and reaching out to one of them, or to someone you trust, "
    "is a really good next step. I'm still here if you'd like to keep talking."
)


def make_crisis_node(model):
    # Tools stay attached (the API needs their definitions if the history
    # has tool calls) but are switched off: no memory lookups in a crisis.
    replier = model.bind_tools(MEMORY_TOOLS, tool_choice={"type": "none"})

    def crisis(state: CoupleState) -> dict:
        category = (state.get("safety") or {}).get("primary_category") or "general"

        if category == "abuse" and state.get("joint_mode", True):
            return {"display": [format_joint_session_message()],
                    "safety_flags": [category], "session_ended": True}

        note = CRISIS_SAFETY_NOTE.format(concern=_CONCERN.get(category, "a possible risk"))
        system = build_system_prompt(partner_names=state["partner_names"],
                                     speaker=state.get("speaker"),
                                     now_text=state.get("now_text"),
                                     safety_note=note)
        try:
            reply = replier.invoke([SystemMessage(content=system), *state["messages"]])
        except Exception:  # API down, refusal error, anything: still answer
            reply = AIMessage(content=CRISIS_REPLY_FALLBACK)

        return {"display": [format_crisis_message(category)],
                "messages": [reply], "safety_flags": [category]}

    return crisis


def route_after_crisis(state: CoupleState) -> str:
    """Conditional edge: the IPV protocol ends the turn (and the session);
    otherwise the model's crisis reply is checked like any other."""
    return "end" if state.get("session_ended") else "output_check"
