"""Shared pytest fixtures.

v0.4 repeated the same setUp/tearDown in every test class: make a temp
file, point the DB env var at it, init the schema, delete it afterwards.
Here that's one fixture. `monkeypatch.setenv` undoes itself after each
test, and `tmp_path` is a per-test directory pytest cleans up, so no test
can ever touch the real couples_agent.db.
"""

import pytest


@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    """Point the app at a fresh, empty SQLite file for one test."""
    path = tmp_path / "test.db"
    monkeypatch.setenv("COUPLES_AGENT_DB", str(path))
    # Imported lazily: app.memory.db is created in step 2. Until then,
    # tests that don't request this fixture still run.
    from app.memory import db

    db.init_db()
    return path
