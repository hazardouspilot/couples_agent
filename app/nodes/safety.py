"""
Deterministic safety layer: the regex check that runs on every message
before the model sees it (safety_gate) and on every model reply
(output_check). The graph node functions are added in a later step; this
part is pure Python so it can be tested and evaluated on its own.

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
