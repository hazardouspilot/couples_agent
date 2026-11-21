"""Tests for app/prompts.py. The time tests are ported from v0.4's
test_temporal_wiring.py (TestSystemPromptTemporalInjection)."""

from langchain_core.messages import AIMessage

from app.prompts import build_system_prompt, speaker_message, turn_balance_note

NAMES = {"partner_a": "Jordan", "partner_b": "Riley"}


def _turns(*speakers):
    """Human messages from the given speakers, with an AI reply after each."""
    msgs = []
    for s in speakers:
        msgs += [speaker_message("something", s, NAMES), AIMessage(content="reply")]
    return msgs


# --- ported from v0.4 ---

def test_includes_current_time_when_provided():
    prompt = build_system_prompt(NAMES, now_text="Saturday, 21 November 2026, 13:00")
    assert "CURRENT DATE AND TIME" in prompt
    assert "Saturday, 21 November 2026" in prompt


def test_includes_temporal_context_when_provided():
    prompt = build_system_prompt(
        NAMES, now_text="Saturday, 21 November 2026, 13:00",
        temporal_context="This is session #4. The previous session started 12 days ago.",
    )
    assert "session #4" in prompt and "12 days ago" in prompt


def test_omits_time_block_when_not_provided():
    assert "CURRENT DATE AND TIME" not in build_system_prompt(NAMES)


def test_instructs_model_not_to_compute_time_itself():
    assert "not reliable at computing time" in build_system_prompt(NAMES).lower()


# --- new: couples ---

def test_prompt_names_both_partners_and_current_speaker():
    prompt = build_system_prompt(NAMES, speaker="partner_b")
    assert "Jordan and Riley" in prompt
    assert "Riley is typing this turn" in prompt


def test_prompt_states_neutrality_rules():
    prompt = build_system_prompt(NAMES)
    assert "NEUTRALITY" in prompt
    assert "never say or imply who is right" in prompt.lower()


def test_safety_note_and_exercise_are_appended_only_when_given():
    base = build_system_prompt(NAMES)
    assert "SAFETY SYSTEM NOTE" not in base and "CURRENT CONTEXT" not in base
    full = build_system_prompt(NAMES, active_exercise="conflict-debrief",
                               safety_note="flagged: self_harm")
    assert "conflict-debrief" in full and "flagged: self_harm" in full


def test_speaker_message_prefix_and_name():
    msg = speaker_message("I felt ignored", "partner_a", NAMES)
    assert msg.content == "Jordan: I felt ignored"
    assert msg.name == "partner_a"


def test_no_turn_note_below_streak_limit():
    assert turn_balance_note(_turns("partner_a", "partner_a"), NAMES) is None
    assert turn_balance_note(_turns("partner_a", "partner_b", "partner_a"), NAMES) is None
    assert turn_balance_note([], NAMES) is None


def test_turn_note_after_three_in_a_row_invites_the_other_partner():
    note = turn_balance_note(_turns("partner_b", "partner_a", "partner_a", "partner_a"), NAMES)
    assert note == ("Jordan has spoken 3 times in a row. "
                    "Warmly invite Riley to share their view.")
