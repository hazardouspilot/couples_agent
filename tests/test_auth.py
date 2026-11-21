"""Tests for app/memory/auth.py. The first seven are v0.4's auth tests
(from test_auth_and_db.py) adapted to couple accounts; the rest cover
partner-name validation, which is new."""

import pytest

from app.memory import auth, db

PW = "correcthorsebattery"


def _register(username="jordan-and-riley", password=PW, a="Jordan", b="Riley"):
    return auth.register(username, password, a, b)


# --- ported from v0.4 ---

def test_register_then_login(tmp_db):
    couple_id = _register()
    assert isinstance(couple_id, int)
    assert auth.login("jordan-and-riley", PW) == couple_id


def test_login_wrong_password_fails(tmp_db):
    _register()
    with pytest.raises(auth.AuthError):
        auth.login("jordan-and-riley", "wrongpassword")


def test_login_unknown_user_fails(tmp_db):
    with pytest.raises(auth.AuthError):
        auth.login("nobody", "whatever123")


def test_wrong_password_and_unknown_user_give_same_message(tmp_db):
    _register()
    with pytest.raises(auth.AuthError) as wrong_pw:
        auth.login("jordan-and-riley", "wrongpassword")
    with pytest.raises(auth.AuthError) as unknown:
        auth.login("nobody", "whatever123")
    assert str(wrong_pw.value) == str(unknown.value)


def test_duplicate_username_rejected(tmp_db):
    _register()
    with pytest.raises(auth.AuthError):
        _register(password="differentpassword")


def test_short_password_rejected(tmp_db):
    with pytest.raises(auth.AuthError):
        _register(password="short")


def test_invalid_username_rejected(tmp_db):
    with pytest.raises(auth.AuthError):
        _register(username="h!")


def test_password_hash_is_not_plaintext(tmp_db):
    _register()
    stored = db.get_couple_by_username("jordan-and-riley")["password_hash"]
    assert stored != PW
    assert stored.startswith("$argon2id$")


# --- new: partner names ---

def test_partner_names_stored_stripped(tmp_db):
    couple_id = _register(a="  Jordan ", b="Riley  ")
    assert db.get_partner_names(couple_id) == {"partner_a": "Jordan", "partner_b": "Riley"}


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("", "Riley"),         # missing
        ("Jordan", "   "),        # whitespace only
        ("Jordan", "jordan"),      # same name ignoring case: speaker would be ambiguous
        ("H" * 41, "Riley"),   # too long
    ],
)
def test_invalid_partner_names_rejected(tmp_db, a, b):
    with pytest.raises(auth.AuthError):
        _register(a=a, b=b)
    assert db.get_couple_by_username("jordan-and-riley") is None  # nothing written
