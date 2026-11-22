"""
LLM safety classifier: the second detector in the safety gate.

Why it exists: the first safety eval (evals/reports/*-baseline.md) found
the regex caught 0 of 25 risk messages written without looking at the
patterns. Meanwhile the companion model DID recognise risk, but could only
talk about it; it can't show resources (in one run it invented US hotline
numbers for an Australian couple). So the model now helps DETECT, and code
still decides what the couple SEES:

    regex flags  --> crisis path (classifier not called: saves a call)
    regex clear  --> classifier --(risk)--> crisis path
                                 \-(none or ERROR)--> normal path

Design choices:
  - Union, never veto: the classifier can ADD a flag, never remove a regex
    flag. A model mistake can't hide "I want to kill myself". The cost: it
    can't fix regex false positives either, so precision can only fall.
  - Fails soft to regex-only: an API error returns [] and logs a warning.
    The floor (regex) never depends on the API.
  - Structured output with a Literal: the model can only answer one of the
    four labels; code maps "none" to no flag.
  - Sees ONE message, no history: cheap and simple, but it can miss risk
    that only shows in context ("won't be your problem much longer" after
    "you've been distant"). Known limitation; the companion's prompt is still
    the third layer for those.
  - Low effort: classification doesn't need deep thinking; it's the cost
    and latency dial for a call that runs on almost every turn.
"""

import logging
from collections.abc import Callable
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

log = logging.getLogger(__name__)

Classifier = Callable[[str], list[str]]


class SafetyVerdict(BaseModel):
    category: Literal["self_harm", "harm_to_others", "abuse", "none"]


CLASSIFIER_PROMPT = """You screen ONE message from a couples-support chat for safety risk. \
Classify what the SPEAKER is saying about themselves or their situation now:

- self_harm: the speaker has thoughts of suicide or self-injury, or is doing it. Includes \
indirect signs: feeling like a burden, "won't be around", wanting to sleep and not wake up, \
giving things away, goodbye letters, researching methods.
- harm_to_others: the speaker intends, plans or threatens to hurt someone, or fears they \
might.
- abuse: the speaker is being hurt, threatened, controlled or coerced by a partner or family \
member: violence, damage to property in anger, threats, monitoring, money control, \
isolation from friends or family, sexual coercion, being afraid of them.
- none: everything else. Idioms ("this week is killing me", "mum will kill me"), talking \
ABOUT these topics in general, in the news or in other people's past, and ordinary heated \
arguments are none.

If you are genuinely unsure between a risk category and none, choose the risk category."""


def make_llm_classifier(model) -> Classifier:
    """Returns classify(text) -> [category] or []. Never raises."""
    structured = model.with_structured_output(SafetyVerdict, method="json_schema")

    def classify(text: str) -> list[str]:
        if not text or not text.strip():
            return []
        try:
            verdict = structured.invoke(
                [SystemMessage(CLASSIFIER_PROMPT), HumanMessage(text)],
                config={"run_name": "safety_classifier"},
            )
        except Exception as e:  # fail soft: the regex result stands
            log.warning("safety classifier failed (%s); regex only this turn", type(e).__name__)
            return []
        return [] if verdict.category == "none" else [verdict.category]

    return classify
