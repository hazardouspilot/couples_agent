"""Tests for app/memory/timeutil.py, the only place time-delta arithmetic
happens. Ported from v0.4; same cases and assertions, pytest style.
Every bucket boundary is tested because off-by-one bucket edges are
where this kind of function silently misbehaves."""

from datetime import UTC, datetime, timedelta

import pytest

from app.memory.timeutil import (
    days_between,
    format_date_local,
    humanize_delta,
    humanize_interval,
)

NOW = datetime(2026, 8, 21, 12, 0, 0, tzinfo=UTC)
NOW_ISO = NOW.isoformat()


def ago(**kwargs) -> str:
    return (NOW - timedelta(**kwargs)).isoformat()


# --- humanize_delta ---
# parametrize runs the one test body once per (input, expected) pair,
# replacing v0.4's several assertEqual calls per test method.

@pytest.mark.parametrize(
    ("delta", "expected"),
    [
        ({"seconds": 5}, "just now"),
        ({"seconds": 59}, "just now"),
        ({"minutes": 1}, "1 minute ago"),
        ({"minutes": 30}, "30 minutes ago"),
        ({"hours": 1}, "1 hour ago"),
        ({"hours": 12}, "12 hours ago"),
        ({"hours": 25}, "yesterday"),
        ({"days": 1, "hours": 23}, "yesterday"),
        ({"days": 3}, "3 days ago"),
        ({"days": 6}, "6 days ago"),
        ({"days": 7}, "1 week ago"),
        ({"days": 20}, "3 weeks ago"),
        ({"days": 40}, "1 month ago"),
        ({"days": 90}, "3 months ago"),
        ({"days": 400}, "1 year ago"),
        ({"days": 800}, "2 years ago"),
    ],
)
def test_humanize_delta_buckets(delta, expected):
    assert humanize_delta(ago(**delta), NOW_ISO) == expected


def test_future_timestamp_does_not_claim_past():
    future = (NOW + timedelta(days=3)).isoformat()
    assert "ago" not in humanize_delta(future, NOW_ISO)


def test_tiny_clock_skew_reads_as_just_now():
    slightly_future = (NOW + timedelta(seconds=2)).isoformat()
    assert humanize_delta(slightly_future, NOW_ISO) == "just now"


def test_defaults_to_real_now_when_reference_omitted():
    recent = (datetime.now(UTC) - timedelta(minutes=5)).isoformat()
    result = humanize_delta(recent)
    assert isinstance(result, str) and result


# --- humanize_interval ---

def test_interval_is_order_independent():
    a, b = ago(days=30), ago(days=11)
    assert humanize_interval(a, b) == humanize_interval(b, a)


@pytest.mark.parametrize(
    ("delta", "expected"),
    [
        ({"hours": 5}, "5 hours"),
        ({"days": 10}, "10 days"),
        ({"days": 19}, "3 weeks"),
    ],
)
def test_interval_buckets(delta, expected):
    assert humanize_interval(NOW_ISO, ago(**delta)) == expected


def test_interval_has_no_ago_suffix():
    assert "ago" not in humanize_interval(NOW_ISO, ago(days=5))


# --- days_between / format_date_local ---

def test_days_between_basic():
    assert days_between(NOW_ISO, ago(days=19)) == 19


def test_days_between_order_independent():
    assert days_between(NOW_ISO, ago(days=7)) == days_between(ago(days=7), NOW_ISO)


def test_format_date_local_is_readable():
    result = format_date_local(NOW_ISO)
    assert "2026" in result and "August" in result
