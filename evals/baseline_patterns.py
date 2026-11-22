"""The v0.4 safety regex, frozen: the BEFORE in the safety eval.

app/nodes/safety.py has the current patterns; this copy exists only so
`run_evals.py --regex v04` can still measure where we started."""

import re

V04_PATTERNS = [
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
    (r"\b(kill|hurt|attack|shoot|stab) (him|her|them|someone|somebody)\b", "harm_to_others"),
    (r"\bwant(ed)? to (kill|hurt) (him|her|them|somebody|someone)\b", "harm_to_others"),
    (r"\bmake (him|her|them) pay\b", "harm_to_others"),
    (r"\bkill (people|everyone)\b", "harm_to_others"),
    (r"\bhe hits me\b|\bshe hits me\b|\bthey hit me\b", "abuse"),
    (r"\b(being|getting) abused\b", "abuse"),
    (r"\bdomestic violence\b", "abuse"),
    (r"\bafraid (of|for) my (life|safety)\b", "abuse"),
    (r"\bhe threatened (to kill|me)\b|\bshe threatened (to kill|me)\b", "abuse"),
    (r"\bsexual(ly)? assault(ed)?\b", "abuse"),
    (r"\brape(d)?\b", "abuse"),
]

V04_COMPILED = [(re.compile(p, re.IGNORECASE), cat) for p, cat in V04_PATTERNS]
