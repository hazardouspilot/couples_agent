"""
Builds the chat model. Kept in one place so the graph can be given a fake
model in tests (dependency injection) and the real one in the CLI.

Settings, checked against the current Claude API reference:
  - Model id from ANTHROPIC_MODEL; default claude-opus-5-5 (current Opus).
  - No temperature: current models reject sampling parameters.
  - Thinking is always on for Opus 5.5; `effort` is the depth/cost dial.
    Its default on this model is "medium"; we set it explicitly so it's
    visible, and ANTHROPIC_EFFORT can lower it ("low") as a cost lever.
  - Server-side refusal fallback: if a safety classifier declines a
    request, the API re-runs it on a fallback model inside the same call.
"""

import os

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

DEFAULT_MODEL = "claude-opus-5-5"
DEFAULT_EFFORT = "medium"
MAX_TOKENS = 16000


def make_chat_model():
    """ChatAnthropic is LangChain's adapter for Claude: it implements the
    shared chat-model interface (.invoke / .stream / .bind_tools) and
    converts LangChain message objects to and from the Anthropic API."""
    from langchain_anthropic import ChatAnthropic  # imported lazily: tests never need it

    return ChatAnthropic(
        model=os.environ.get("ANTHROPIC_MODEL", DEFAULT_MODEL),
        max_tokens=MAX_TOKENS,
        output_config={"effort": os.environ.get("ANTHROPIC_EFFORT", DEFAULT_EFFORT)},
        betas=["server-side-fallback-2026-07-01"],
        model_kwargs={"fallbacks": "default"},
    )


OFFLINE_REPLY = (
    "[No ANTHROPIC_API_KEY configured, so I can't reply. Add one to .env to chat. "
    "Commands like /memory and /crisis still work, and the safety check still runs "
    "on everything you type.]"
)


class OfflineChatModel(BaseChatModel):
    """Stand-in used when no API key is set. The graph still runs (so the
    safety gate still checks every message), but the reply is a fixed notice."""

    @property
    def _llm_type(self) -> str:
        return "offline"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=OFFLINE_REPLY))])

    def bind_tools(self, tools, **kwargs):
        return self


def get_model():
    """The real model if a key is configured, else the offline stand-in."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return make_chat_model()
    return OfflineChatModel()
