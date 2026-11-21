"""Tests for app/resources/crisis_resources.py. v0.4 had no tests for this
file; these check the properties the safety design depends on."""

from datetime import date

import pytest

from app.resources import crisis_resources as cr

CATEGORIES = ["self_harm", "harm_to_others", "abuse", "general"]


def test_every_resource_has_required_fields():
    for r in cr.RESOURCES:
        assert {"name", "phone", "hours", "for"} <= r.keys(), r


def test_verified_on_is_a_real_date():
    date.fromisoformat(cr.VERIFIED_ON)


@pytest.mark.parametrize("category", CATEGORIES)
def test_crisis_message_always_has_000_and_every_number(category):
    msg = cr.format_crisis_message(category)
    assert "000" in msg
    for r in cr.RESOURCES:
        assert r["phone"] in msg


def test_joint_message_has_000_and_every_number():
    msg = cr.format_joint_session_message()
    assert "000" in msg
    for r in cr.RESOURCES:
        assert r["phone"] in msg


def test_joint_message_never_refers_to_what_was_said():
    # IPV protocol: the text must not single out a disclosure or a person.
    msg = cr.format_joint_session_message().lower()
    for phrase in ("you're describing", "what you said", "you told", "abuse", "your partner"):
        assert phrase not in msg


def test_joint_message_frames_individual_support_as_standard():
    assert "standard practice" in cr.format_joint_session_message()


def test_no_university_specific_text():
    text = cr.format_crisis_message() + cr.format_joint_session_message()
    assert "JCU" not in text and "university" not in text.lower()
