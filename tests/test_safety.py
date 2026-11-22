"""Tests for the deterministic safety regex (app/nodes/safety.py).

Ported from v0.4's test_safety.py: same cases, pytest style. These are a
regression net so pattern edits can't silently break the phrasings that
matter most. Precision/recall across many phrasings is the job of the
Sunday safety eval (evals/datasets/safety.jsonl), not of these tests.
"""

import pytest

from app.nodes.safety import check_message

# --- ported from v0.4 ---

@pytest.mark.parametrize(
    "text",
    [
        "I've been thinking about suicide a lot lately",
        "sometimes I just want to kill myself",
        "I don't want to be alive anymore",
        "I've been planning to kill myself this week",
        "honestly I feel like I have no reason to live",
    ],
)
def test_suicidal_ideation_triggers(text):
    result = check_message(text)
    assert result.triggered
    assert result.primary_category == "self_harm"


@pytest.mark.parametrize(
    ("text", "category"),
    [
        ("I've been cutting myself when things get bad", "self_harm"),
        ("I just want to hurt him for what he did", "harm_to_others"),
        ("he hits me when he's been drinking", "abuse"),
        ("I think I'm experiencing domestic violence", "abuse"),
    ],
)
def test_category_triggers(text, category):
    result = check_message(text)
    assert result.triggered
    assert result.primary_category == category


@pytest.mark.parametrize(
    "text",
    [
        "I had a rough day at work today",
        # "X is killing my ___" is a known false-positive shape; the
        # self-harm patterns avoid it by requiring "myself".
        "my exam is killing my motivation this week",
        "I feel really anxious about my presentation tomorrow",
        "my friend and I had an argument",
        "I'm just stressed about deadlines",
    ],
)
def test_benign_messages_do_not_trigger(text):
    assert not check_message(text).triggered


def test_empty_and_none_safe():
    assert not check_message("").triggered
    assert not check_message(None).triggered


def test_case_insensitivity():
    assert check_message("I WANT TO KILL MYSELF").triggered


# --- new ---

def test_self_harm_outranks_other_categories():
    result = check_message("he hits me and I want to kill myself")
    assert set(result.categories) == {"abuse", "self_harm"}
    assert result.primary_category == "self_harm"


def test_state_form_never_contains_message_text():
    result = check_message("he hits me when he's been drinking")
    state = result.as_state()
    assert state == {"triggered": True, "categories": ["abuse"], "primary_category": "abuse"}
    assert "hits me" not in str(state)


def test_state_form_when_clear():
    assert check_message("we argued about chores").as_state() == {
        "triggered": False, "categories": [], "primary_category": None,
    }


# --- Saturday's two strict xfails, fixed by Sunday's pattern revision ---

def test_hurt_feelings_is_not_harm_to_others():
    assert not check_message("I didn't mean to hurt her feelings").triggered
    assert check_message("I want to hurt her").triggered  # the real threat still flags


def test_better_off_without_me_is_flagged():
    assert check_message(
        "Some nights I think everyone would be better off if I wasn't here."
    ).primary_category == "self_harm"


# --- Sunday: discussing a topic is not disclosing it ---

@pytest.mark.parametrize("text", [
    "my friend's dad died by suicide last year",
    "we watched a doco on domestic violence",
    "my sister used to self harm as a teenager",
    "that episode where the character overdosed was hard to watch",
])
def test_discussing_a_topic_does_not_trigger(text):
    assert not check_message(text).triggered


@pytest.mark.parametrize(("text", "category"), [
    ("honestly been thinking about killing myself", "self_harm"),
    ("i've started cutting again", "self_harm"),
    ("i walk on eggshells all the time", "abuse"),
    ("she grabbed me and wouldn't let go", "abuse"),
])
def test_sunday_word_forms_and_couples_phrasing(text, category):
    assert check_message(text).primary_category == category
