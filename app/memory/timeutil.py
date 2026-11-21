"""
Time formatting: the single place "now" comes from, and the only place
that turns two timestamps into a phrase like "3 weeks ago".

Ported unchanged from counselling_agent v0.4 (agent/timeutil.py).

Why this is its own module rather than handing the model raw ISO
timestamps: LLMs are unreliable at date arithmetic (subtracting two
timestamps, month lengths, "how long ago"), and it gets worse the more
context sits between the timestamp and the question. So the rule
everywhere memory touches time is: **Python computes every delta**. The
model only ever receives a finished phrase ("3 weeks ago"), plus the raw
timestamp where exact precision matters. In the LangGraph version, the
`load_context` node calls into this module and puts the results in graph
state; tool results carry `relative_time` computed here.

Storage is always UTC. Display converts to the local system timezone,
a known simplification for a CLI that runs on the couple's own machine.
"""

from datetime import UTC, datetime


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def parse_iso(iso_timestamp: str) -> datetime:
    dt = datetime.fromisoformat(iso_timestamp)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


def humanize_delta(iso_timestamp: str, reference_iso: str | None = None) -> str:
    """'3 weeks ago', 'yesterday', 'just now', etc., relative to
    reference_iso (defaults to now)."""
    then = parse_iso(iso_timestamp)
    now = parse_iso(reference_iso) if reference_iso else datetime.now(UTC)
    seconds = (now - then).total_seconds()

    if seconds < 0:
        # Clock skew or a future timestamp: don't claim a confident "ago"
        # for something that hasn't happened yet.
        return "just now" if seconds > -5 else "in the future"

    if seconds < 60:
        return "just now"
    minutes = seconds / 60
    if minutes < 60:
        n = round(minutes)
        return "1 minute ago" if n == 1 else f"{n} minutes ago"
    hours = minutes / 60
    if hours < 24:
        n = round(hours)
        return "1 hour ago" if n == 1 else f"{n} hours ago"
    days = hours / 24
    if days < 2:
        return "yesterday"
    if days < 7:
        return f"{round(days)} days ago"
    weeks = days / 7
    if days < 35:
        n = round(weeks)
        return "1 week ago" if n == 1 else f"{n} weeks ago"
    months = days / 30.44
    if days < 365:
        n = round(months)
        return "1 month ago" if n == 1 else f"{n} months ago"
    years = days / 365.25
    n = round(years)
    return "1 year ago" if n == 1 else f"{n} years ago"


def humanize_interval(iso_a: str, iso_b: str) -> str:
    """Duration between two timestamps (order-independent), with no 'ago'
    suffix, e.g. '19 days', '6 weeks'. For gaps where neither end is now."""
    a, b = parse_iso(iso_a), parse_iso(iso_b)
    seconds = abs((b - a).total_seconds())
    days = seconds / 86400

    if days < 1:
        hours = seconds / 3600
        n = max(round(hours), 1)
        return "1 hour" if n == 1 else f"{n} hours"
    if days < 14:
        n = round(days)
        return "1 day" if n == 1 else f"{n} days"
    if days < 60:
        n = round(days / 7)
        return "1 week" if n == 1 else f"{n} weeks"
    if days < 730:
        n = round(days / 30.44)
        return "1 month" if n == 1 else f"{n} months"
    n = round(days / 365.25)
    return "1 year" if n == 1 else f"{n} years"


def days_between(iso_a: str, iso_b: str) -> int:
    a, b = parse_iso(iso_a), parse_iso(iso_b)
    return round(abs((b - a).total_seconds()) / 86400)


def format_local(iso_timestamp: str, fmt: str = "%A, %d %B %Y, %H:%M") -> str:
    """Absolute timestamp in the local timezone. Used for the 'current date
    and time' line the model receives every turn."""
    return parse_iso(iso_timestamp).astimezone().strftime(fmt)


def format_date_local(iso_timestamp: str) -> str:
    return parse_iso(iso_timestamp).astimezone().strftime("%d %B %Y")
