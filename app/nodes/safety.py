"""
Deterministic safety layer: the regex check that runs on every message
before the model sees it (safety_gate) and on every model reply
(output_check). The regex part (check_message) is pure Python so it can be
tested and evaluated on its own; the graph nodes are at the bottom.

Ported unchanged from counselling_agent v0.4 (agent/safety.py) as the
BASELINE for Sunday's safety eval. The couples-specific patterns (coercive
control: "checks my phone", "controls the money", ...) are deliberately
added on Sunday, after the first eval run, so the before/after recall on
`abuse` is a real measured difference.

Why regex and not another LLM call:
  1. Reliability: it can't fail from an API outage, rate limit, or the
     model "staying in character". It always runs, in microseconds.
  2. Auditability: you can read every pattern and know exactly what will
     and won't trigger.
The trade-off: keyword matching is blunt. It produces false positives and
misses indirect phrasing, so it's a FLOOR, not a full solution. The system
prompt asks the model to recognise risk too (the second layer), and an
optional LLM classifier is a Sunday experiment.

Bias: towards OVER-triggering. In v0.4 a false positive cost one extra
resources message and the chat carried on; a false negative could cost far
more. In joint couples sessions the cost depends on the category: any flag
shows crisis resources to BOTH partners mid-conversation, and an `abuse`
flag also stops mediation for the rest of the session (IPV protocol). So
precision on `abuse` matters more here than it did in v0.4.
"""

import re
from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage

from app.memory import db
from app.resources.crisis_resources import format_crisis_message, format_joint_session_message

# Highest priority first: decides which resource message is shown when
# several categories match. Self-harm is the most time-critical.
CATEGORY_PRIORITY = ("self_harm", "harm_to_others", "abuse")


@dataclass
class SafetyResult:
    triggered: bool
    categories: list[str] = field(default_factory=list)
    # The matched phrases are pieces of the user's message. Useful for
    # debugging and evals, but never stored: see as_state().
    matched: list[str] = field(default_factory=list)

    @property
    def primary_category(self) -> str:
        for c in CATEGORY_PRIORITY:
            if c in self.categories:
                return c
        return self.categories[0] if self.categories else "general"

    def as_state(self) -> dict:
        """What goes into graph state (and so into checkpoints and traces):
        categories only, never the matched text."""
        return {
            "triggered": self.triggered,
            "categories": list(self.categories),
            "primary_category": self.primary_category if self.triggered else None,
        }


# (regex, category). \b = word boundary; matching is case-insensitive.
# Patterns favour recall over precision on purpose.
_PATTERNS = [
    # --- Suicide / self-harm ---
    (r"\bsuicid(e|al)\b", "self_harm"),
    (r"\bkill myself\b", "self_harm"),
    (r"\bend(ing)? my life\b", "self_harm"),
    (r"\bwant(ed)? to die\b", "self_harm"),
    (r"\bdon'?t want to (be alive|live|exist)\b", "self_harm"),
    (r"\bno reason to live\b", "self_harm"),
    (r"\bbetter off (dead|without me)\b", "self_harm"),
    (r"\bself[\s-]?harm\b", "self_harm"),
    (r"\bhurt(ing)? myself\b", "self_harm"),
    (r"\bcutt?ing myself\b", "self_harm"),
    (r"\boverdos(e|ing)\b", "self_harm"),
    (r"\bgoing to (kill|hurt) myself\b", "self_harm"),
    (r"\bplan(ning)? to (kill|hurt) myself\b", "self_harm"),
    (r"\bsuicide (plan|note)\b", "self_harm"),
    # --- Harm to others ---
    (r"\b(kill|hurt|attack|shoot|stab) (him|her|them|someone|somebody)\b", "harm_to_others"),
    (r"\bwant(ed)? to (kill|hurt) (him|her|them|somebody|someone)\b", "harm_to_others"),
    (r"\bmake (him|her|them) pay\b", "harm_to_others"),
    (r"\bkill (people|everyone)\b", "harm_to_others"),
    # --- Abuse / domestic violence ---
    (r"\bhe hits me\b|\bshe hits me\b|\bthey hit me\b", "abuse"),
    (r"\b(being|getting) abused\b", "abuse"),
    (r"\bdomestic violence\b", "abuse"),
    (r"\bafraid (of|for) my (life|safety)\b", "abuse"),
    (r"\bhe threatened (to kill|me)\b|\bshe threatened (to kill|me)\b", "abuse"),
    (r"\bsexual(ly)? assault(ed)?\b", "abuse"),
    (r"\brape(d)?\b", "abuse"),
]

_COMPILED = [(re.compile(p, re.IGNORECASE), cat) for p, cat in _PATTERNS]


def check_message(text: str | None) -> SafetyResult:
    """Scan one message for crisis indicators. Check `.triggered`."""
    if not text:
        return SafetyResult(triggered=False)

    categories, matched = [], []
    for pattern, category in _COMPILED:
        m = pattern.search(text)
        if m:
            matched.append(m.group(0))
            if category not in categories:  # de-dupe, keep first-seen order
                categories.append(category)

    return SafetyResult(triggered=bool(categories), categories=categories, matched=matched)


# ======================================================================
# Graph nodes
# ======================================================================

def safety_gate(state) -> dict:
    """Runs the regex on this turn's message, before anything else can
    reach the model. Logs the category only (never the text or speaker)."""
    last_human = next(
        (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None
    )
    result = check_message(last_human.content if last_human else None)
    if result.triggered:
        db.log_safety_event(state["couple_id"], state["session_id"], result.primary_category)
    return {"safety": result.as_state()}


def route_after_safety(state) -> str:
    """Conditional edge: a flagged message goes to `crisis`, never on to
    the router or the model."""
    return "crisis" if state["safety"]["triggered"] else "router"


# --- output_check: the backstop on the model's own replies ---
# v0.4 described an output-side check but never called it. Two parts:
#   1. Neutrality rules: side-taking or diagnostic labels. The reply is
#      REPLACED with a neutral fallback, because both partners would read it.
#   2. The crisis regex on the reply: if it fires, resources are ADDED by
#      code (the reply is kept; we never remove a safety message).
# Like the input regex, this is blunt and recall-biased: a false positive
# costs one gentler, generic reply.

_LABEL_PATTERNS = [
    (r"\bnarcissis\w*", "diagnosis"),
    (r"\bgaslight\w*", "diagnosis"),
    (r"\bco-?dependen\w*", "diagnosis"),
    (r"\b(socio|psycho)path\w*", "diagnosis"),
    (r"\btoxic\b", "diagnosis"),
    (r"\bpersonality disorder\b", "diagnosis"),
    (r"\b(has|have|with) (bpd|npd|adhd|bipolar|depression)\b", "diagnosis"),
    (r"\b(you're|you are|he's|he is|she's|she is) (being )?(abusive|manipulative)\b", "diagnosis"),
    (r"\b(it's|it is|that's|that is) (all )?(his|her|your) fault\b", "side_taking"),
    (r"\b(you're|you are|he's|he is|she's|she is) to blame\b", "side_taking"),
]
_COMPILED_LABELS = [(re.compile(p, re.IGNORECASE), cat) for p, cat in _LABEL_PATTERNS]

NEUTRAL_FALLBACK = (
    "I want to be careful to stay on both your sides here, so let me step back. "
    "Could you each tell me, in a sentence or two, what felt most important to "
    "you in what just happened?"
)
EMPTY_REPLY_FALLBACK = (
    "I'm not able to respond to that one, sorry. Would you like to come at it "
    "from a different angle, or talk about something else?"
)


def neutrality_flags(text: str, partner_names: dict[str, str]) -> list[str]:
    """Names of the neutrality rules a reply breaks ([] if none). Partner
    names are added per call, e.g. 'Jordan is right' / 'Riley is wrong'."""
    flags = [cat for pat, cat in _COMPILED_LABELS if pat.search(text or "")]
    names = "|".join(re.escape(n) for n in partner_names.values())
    if re.search(rf"\b({names}) (is|was) (right|wrong|at fault|to blame)\b", text or "",
                 re.IGNORECASE):
        flags.append("side_taking")
    return sorted(set(flags))


def output_check(state) -> dict:
    reply = state["messages"][-1]
    text = reply.text if isinstance(reply, AIMessage) else ""
    update: dict = {}

    flags = neutrality_flags(text, state["partner_names"])
    if not text.strip():
        flags.append("empty_reply")  # e.g. a refusal; the turn still needs an answer
    if flags:
        fallback = EMPTY_REPLY_FALLBACK if flags == ["empty_reply"] else NEUTRAL_FALLBACK
        # RemoveMessage tells the add_messages reducer to delete that message;
        # the fallback is then appended in its place.
        update["messages"] = [RemoveMessage(id=reply.id), AIMessage(content=fallback)]
        update["output_flags"] = flags
        text = fallback

    result = check_message(text)
    if result.triggered:
        category = result.primary_category
        db.log_safety_event(state["couple_id"], state["session_id"], category)
        resources = (format_joint_session_message()
                     if category == "abuse" and state.get("joint_mode", True)
                     else format_crisis_message(category))
        update["display"] = [*state.get("display", []), resources]
        update["output_flags"] = [*update.get("output_flags", []), f"safety:{category}"]
    return update
