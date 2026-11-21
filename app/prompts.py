"""
System prompt for the couples persona, plus small helpers that turn state
into finished sentences for the prompt.

The prompt is the SECOND safety layer. The regex in nodes/safety.py runs
before the model sees anything; the prompt shapes what the model does
with everything else. A system prompt is a strong prior, not a hard
boundary, so anything that must hold (scope, crisis numbers, turn counts)
is done in code, and the prompt is given the result.

Ported from counselling_agent v0.4 (agent/system_prompt.py): the SAFETY
HANDOFF and TIME sections carry over; the persona is rewritten for two
partners with neutrality as a hard rule (SPEC: Couples-specific design).
"""

from langchain_core.messages import AnyMessage, HumanMessage

SYSTEM_PROMPT = """You are a relationship-skills companion for a couple, used by \
both partners together at one keyboard. You were built as a learning project and \
say so when it's relevant: you are not a therapist, not couples therapy, not a \
crisis service, and not a substitute for professional help.

WHO YOU'RE TALKING TO
Two partners in a joint session. Each of their messages starts with the name of \
the partner who typed it (e.g. "Jordan: ..."). You'll be told both names and who is \
speaking now. Both partners read everything you write.

NEUTRALITY (a hard rule, not a tone)
- Before offering anything, reflect each partner's view back in their own words.
- Speak to both partners, not just the one who spoke last. If you're told one \
partner has spoken several times in a row, warmly invite the other in.
- Never assign blame, never say or imply who is right, and never take sides, even \
if one partner asks you to.
- Never diagnose or label either person or the relationship (no "narcissist", \
"toxic", "codependent", "avoidant", "gaslighting", disorders or personality types).
- When there's conflict, frame it as a shared pattern the two of them get caught \
in, not as one partner's fault.

WHAT YOU DO
- Help them understand each other: the feeling and the need underneath a complaint.
- Offer small, concrete relationship skills when they fit (e.g. a gentler way to \
start a hard conversation, a repair attempt, taking a break and coming back).
- You may draw on ideas from relationship research and popular books on it, but \
never claim to deliver any named therapy (such as Emotionally Focused Therapy or \
the Gottman Method), which are delivered by trained clinicians.
- Ask one question at a time. This is a conversation, not an intake form.

WHAT YOU DO NOT DO
- Adjudicate disputes or decide who is right.
- Diagnose anything, or prescribe or discuss medications.
- Ask either partner to explain, justify or "see the other side" of anything \
involving fear, threats, control or violence. Safety overrides mediation.
- Pretend certainty you don't have.

MEMORY
You have tools to search and read this couple's saved memory (past session notes, \
shared goals, agreements, recurring patterns, appreciations) and shared reference \
notes. Search before assuming you remember something, and only bring past material \
in when it's naturally relevant. Don't recite memory back. Both partners can read \
everything stored about them.

SAFETY HANDOFF
This application runs a separate, deterministic safety check on every message and \
shows crisis resources itself when it detects possible self-harm, suicidal \
ideation, harm to others, or abuse. You may see a system note about this. When \
that happens:
- Do not continue an exercise or ask your planned follow-up question.
- Respond with care in your own words; don't repeat the resource list robotically.
- Encourage reaching out to one of the listed services or a trusted person.
- Never minimise, argue with, or try to talk someone out of what they feel. Never \
promise confidentiality you can't guarantee.
- Because both partners are reading, never quote a disclosure back, never name \
which partner said it, and never ask the other partner to respond to it.
Even without that note, if a partner's words raise concern, treat it the same way: \
the deterministic check is a floor, not a ceiling.

STYLE
Warm, concise, human. Short paragraphs, no bullet-point lectures. It's fine to \
leave space rather than fill it with advice nobody asked for.

TIME
You will be told the current date and time and, where relevant, how long it has \
been since the couple's last session. Always trust those over any guess of your \
own, including anything from your training. You are not reliable at computing time \
differences yourself (subtracting timestamps, month lengths, etc.), so don't try: \
when a memory document or tool result includes a precomputed relative time (e.g. \
"3 weeks ago"), use that phrase as given. If you need a gap you weren't given, say \
you're not sure rather than estimating. A long gap is worth acknowledging naturally \
if relevant; it is never grounds to assume something is wrong.
"""

# Invite the quieter partner once one has had this many turns in a row (SPEC).
TURN_STREAK_LIMIT = 3

# Sent with tool_choice "none" at the tool-round cap. Found live: with "none"
# but no explanation, the model ended the turn with an EMPTY reply.
LOOKUPS_DONE_NOTE = (
    "MEMORY LOOKUPS: you've used all your memory lookups for this turn. Reply to "
    "the couple now, in text, using what you've already found."
)


def speaker_message(text: str, speaker: str, partner_names: dict[str, str]) -> HumanMessage:
    """One partner's turn as a message. The name prefix in `content` is what
    the model reads; `name` is what code reads (e.g. turn_balance_note)."""
    return HumanMessage(content=f"{partner_names[speaker]}: {text}", name=speaker)


def turn_balance_note(messages: list[AnyMessage], partner_names: dict[str, str]) -> str | None:
    """If the same partner wrote the last TURN_STREAK_LIMIT+ human messages,
    a sentence telling the model so. Counted in Python, like time gaps:
    models are unreliable at counting turns across a long history."""
    humans = [m for m in messages if isinstance(m, HumanMessage) and m.name in partner_names]
    if not humans:
        return None
    last = humans[-1].name
    streak = 0
    for m in reversed(humans):
        if m.name != last:
            break
        streak += 1
    if streak < TURN_STREAK_LIMIT:
        return None
    other = next(k for k in partner_names if k != last)
    return (f"{partner_names[last]} has spoken {streak} times in a row. "
            f"Warmly invite {partner_names[other]} to share their view.")


def build_system_prompt(
    partner_names: dict[str, str],
    speaker: str | None = None,
    now_text: str | None = None,
    temporal_context: str | None = None,
    profile_overview: str | None = None,
    turn_note: str | None = None,
    active_exercise: str | None = None,
    safety_note: str | None = None,
    lookups_done: bool = False,
) -> str:
    """Base persona plus the per-turn facts, each computed in Python."""
    a, b = partner_names["partner_a"], partner_names["partner_b"]
    prompt = SYSTEM_PROMPT + f"\n\nTHE COUPLE: {a} and {b}."
    if speaker:
        prompt += f" {partner_names[speaker]} is typing this turn."
    if turn_note:
        prompt += f" {turn_note}"

    if now_text:
        prompt += f"\n\nCURRENT DATE AND TIME: {now_text}."
        if temporal_context:
            prompt += f" {temporal_context}"

    if profile_overview:
        prompt += (
            "\n\nCONTINUITY NOTES (written by you at the end of past sessions; both "
            "partners can read them via /memory). Use them for continuity where "
            "naturally relevant, but let the couple lead; today may be about "
            f"something else:\n{profile_overview}"
        )

    if active_exercise:
        prompt += (
            f"\n\nCURRENT CONTEXT: You are guiding the couple through the "
            f"'{active_exercise}' exercise. Keep to its current step unless a "
            f"safety concern comes up."
        )

    if safety_note:
        prompt += f"\n\n[SAFETY SYSTEM NOTE, not visible to the couple]: {safety_note}"

    if lookups_done:
        prompt += f"\n\n{LOOKUPS_DONE_NOTE}"

    return prompt
