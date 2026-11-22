"""
The conflict-debrief exercise: a fixed sequence of questions that
alternates between the partners, one question per turn.

Design: a STATE MACHINE ACROSS TURNS. `exercise_step` in the state says
which question is waiting for an answer. Each answer arrives as an ordinary
turn, so it goes START -> load_context -> safety_gate -> router -> exercise,
passing the safety gate like any other message with no extra code.

(Considered and rejected: LangGraph's interrupt()/Command(resume=...),
which pauses a graph mid-node and resumes it later. A resumed graph
continues from where it paused, so it would skip the safety gate at START
and need a second gate inside the exercise. More to build, more to get
wrong.)

CODE writes the questions (fixed, own-words text; no model call per step:
fast, free, predictable). The MODEL writes only the closing reflection.
The answers are saved as an exercise_record when the last one arrives,
before the model is called, so a failed reflection loses nothing.

Steps: SPEC "Guided exercise (v1): Conflict debrief", plus the two steps
from the two reference books (the pattern, and the four horsemen).
"""

import json

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.memory import db
from app.prompts import build_system_prompt
from app.state import CoupleState
from app.tools.memory import MEMORY_TOOLS

EXERCISE_KEY = "conflict_debrief"
EXERCISE_TITLE = "conflict debrief"

A, B, BOTH = "partner_a", "partner_b", "together"

# (answer key, who answers, question). {name} is the person asked.
_EACH = [
    ("felt", "{name}: what did you feel at the time, and how strongly, from 0 to 100?"),
    ("needed", "{name}: what did you need in that moment? The need underneath the "
               "reaction, not what the other person should have done."),
    ("my_part", "{name}: what was your part in how it went? Even a small part counts."),
]
STEPS: list[tuple[str, str, str]] = [
    ("facts", A, "{name}: what happened? Just the facts, the way a camera would have "
                 "seen it, without guessing what anyone meant."),
    ("facts", B, "{name}: same for you. What happened, as facts only?"),
    *[(key, who, q) for key, q in _EACH for who in (A, B)],
    ("pattern", BOTH, "Together: which pattern were you two caught in? For example, one "
                      "of you pushing to talk while the other pulls back. Put it in your "
                      "own words, as something you're both caught in."),
    ("horsemen", BOTH, "Together: did criticism, contempt, defensiveness or stonewalling "
                       "show up? If so, what might the gentler version have sounded like?"),
    *[("request", who, "{name}: one specific request for next time. Something the other "
                       "person could actually do.") for who in (A, B)],
    ("agreement", BOTH, "Last one, together: what's one thing you'll both try next time? "
                        "I'll keep it as one of your agreements."),
]

INTRO = (
    "Let's do a conflict debrief. It's a structured way to look back at one "
    "disagreement together, taking turns, so each of you gets heard. There are "
    f"{len(STEPS)} short questions; type /stop at any time to leave it.\n\n"
    "Pick one specific, recent disagreement, then:"
)
NOT_NOW = (
    "Let's leave the structured exercises for another session. I'm still here if "
    "you'd like to keep talking."
)
REFLECTION_FALLBACK = (
    "Thank you both for working through that so carefully. Your answers are saved, "
    "and your agreement will be kept with your other agreements."
)
REFLECTION_NOTE = (
    "The couple has just finished the conflict debrief; their answers are below. "
    "Write a short closing reflection (under 150 words): reflect each partner's view "
    "evenly, describe the pattern as something they're both caught in, and affirm "
    "their agreement. Don't start a new topic.\n\nANSWERS:\n{answers}"
)


def _ask(index: int, names: dict[str, str]) -> str:
    _, who, question = STEPS[index]
    return question.format(name=names.get(who, ""))


def expected_speaker(state: dict) -> str | None:
    """Who the current question is for (the CLI switches to them), or None
    for a 'together' step or when no exercise is running."""
    step = state.get("exercise_step", -1)
    if not state.get("active_exercise") or not 0 <= step < len(STEPS):
        return None
    who = STEPS[step][1]
    return None if who == BOTH else who


def _answer_label(key: str, who: str, names: dict[str, str]) -> str:
    return f"{key} ({'together' if who == BOTH else names[who]})"


def exercise(state: CoupleState) -> dict:
    names = state["partner_names"]
    step = state.get("exercise_step", -1)

    # Starting (-1 = just started by /debrief).
    if step < 0:
        if state.get("safety_flags"):  # not after a safety flag this session
            return {"messages": [AIMessage(content=NOT_NOW)], "active_exercise": None}
        return {"messages": [AIMessage(content=f"{INTRO}\n\n{_ask(0, names)}")],
                "exercise_step": 0, "exercise_answers": {}}

    # Record the answer to the current step. Strip the "Name: " prefix.
    last = next(m for m in reversed(state["messages"]) if isinstance(m, HumanMessage))
    prefix = f"{names.get(last.name, '')}: "
    text = last.text[len(prefix):] if last.text.startswith(prefix) else last.text
    key, who, _ = STEPS[step]
    answers = {**state.get("exercise_answers", {}), _answer_label(key, who, names): text}

    if step + 1 < len(STEPS):
        reply = f"Thank you.\n\n{_ask(step + 1, names)}"
        return {"messages": [AIMessage(content=reply)],
                "exercise_step": step + 1, "exercise_answers": answers}

    # Finished: CODE saves the record before the model reflects.
    db.save_exercise_record(state["couple_id"], state["session_id"], EXERCISE_KEY,
                            json.dumps(answers))
    return {"exercise_step": len(STEPS), "exercise_answers": answers}


def route_after_exercise(state: CoupleState) -> str:
    """Conditional edge: after the last answer the model writes the
    reflection; otherwise the code-written question goes to output_check."""
    return "reflect" if state.get("exercise_step", -1) >= len(STEPS) else "output_check"


def make_reflect_node(model):
    # Tools attached but off (history may hold tool calls), as in crisis.
    replier = model.bind_tools(MEMORY_TOOLS, tool_choice={"type": "none"})

    def reflect(state: CoupleState) -> dict:
        answers = "\n".join(f"- {k}: {v}" for k, v in state["exercise_answers"].items())
        system = build_system_prompt(partner_names=state["partner_names"],
                                     speaker=state.get("speaker"),
                                     now_text=state.get("now_text"),
                                     task_note=REFLECTION_NOTE.format(answers=answers))
        try:
            reply = replier.invoke([SystemMessage(content=system), *state["messages"]])
        except Exception:
            reply = AIMessage(content=REFLECTION_FALLBACK)
        # The exercise is over either way.
        return {"messages": [reply], "active_exercise": None, "exercise_step": -1,
                "exercise_answers": {}}

    return reflect
