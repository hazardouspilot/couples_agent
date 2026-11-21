"""
Local couple-account authentication.

Ported from counselling_agent v0.4 (agent/auth.py). The account is now a
couple (one shared login, per SPEC), so register() also takes both
partners' display names.

Passwords are hashed with argon2id (argon2-cffi), the current OWASP
default. hash() generates a random salt and encodes salt + cost
parameters into the stored string, so there is no separate salt column,
and verify() re-derives from that string with a constant-time compare.
This is local, single-machine auth (no tokens or sessions); the hashing
part is what you'd keep if this sat behind a real API.
"""

import re

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerifyMismatchError

from app.memory import db

_hasher = PasswordHasher()

USERNAME_RE = re.compile(r"^[a-zA-Z0-9_.-]{3,32}$")
MAX_NAME_LEN = 40


class AuthError(Exception):
    pass


def validate_username(username: str):
    if not USERNAME_RE.match(username or ""):
        raise AuthError(
            "Username must be 3-32 characters: letters, numbers, '.', '_', '-' only."
        )


def validate_password(password: str):
    if not password or len(password) < 8:
        raise AuthError("Password must be at least 8 characters.")


def validate_partner_names(partner_a_name: str, partner_b_name: str) -> tuple[str, str]:
    """Both names required, and they must differ (ignoring case): the CLI
    lets a turn start with a name prefix like "Jordan:", so two identical
    names would make the speaker ambiguous. Returns the stripped names."""
    a, b = (partner_a_name or "").strip(), (partner_b_name or "").strip()
    if not a or not b:
        raise AuthError("Both partners' names are required.")
    if len(a) > MAX_NAME_LEN or len(b) > MAX_NAME_LEN:
        raise AuthError(f"Names must be at most {MAX_NAME_LEN} characters.")
    if a.casefold() == b.casefold():
        raise AuthError("Partners' names must be different so turns can be told apart.")
    return a, b


def register(username: str, password: str, partner_a_name: str, partner_b_name: str) -> int:
    """Create a couple account. Returns couple_id."""
    validate_username(username)
    validate_password(password)
    a, b = validate_partner_names(partner_a_name, partner_b_name)
    if db.get_couple_by_username(username):
        raise AuthError(f"Username '{username}' is already taken.")
    return db.create_couple(username, _hasher.hash(password), a, b)


def login(username: str, password: str) -> int:
    """Returns couple_id, or raises AuthError."""
    couple = db.get_couple_by_username(username)
    if not couple:
        # Same message as a wrong password: don't reveal which part was wrong.
        raise AuthError("Incorrect username or password.")
    try:
        _hasher.verify(couple["password_hash"], password)
    except (VerifyMismatchError, InvalidHashError) as e:
        raise AuthError("Incorrect username or password.") from e
    return couple["id"]
