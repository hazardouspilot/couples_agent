"""Checks the chat model's settings. Constructs ChatAnthropic with a dummy
key; nothing is sent to the API."""

from app.model import DEFAULT_MODEL, make_chat_model


def test_defaults(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)
    monkeypatch.delenv("ANTHROPIC_EFFORT", raising=False)
    model = make_chat_model()
    assert model.model == DEFAULT_MODEL == "claude-opus-5-5"
    assert model.temperature is None  # current models reject sampling params
    assert model.output_config == {"effort": "medium"}
    assert model.model_kwargs == {"fallbacks": "default"}


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")
    monkeypatch.setenv("ANTHROPIC_EFFORT", "low")
    model = make_chat_model()
    assert model.model == "claude-sonnet-5-5"
    assert model.output_config == {"effort": "low"}
