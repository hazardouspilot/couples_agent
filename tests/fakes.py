"""A fake chat model for offline tests.

LangChain's built-in fake chat models don't implement bind_tools, so this
one does: it returns scripted AIMessages in order and records each call's
messages and bound kwargs (tools, tool_choice) for assertions. Because it
implements the same BaseChatModel interface as ChatAnthropic, nodes can't
tell the difference. That's the point of LangChain's shared interface.
"""

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field


class FakeToolModel(BaseChatModel):
    responses: list[AIMessage] = Field(default_factory=list)
    calls: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "fake-tool-model"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.calls.append({"messages": list(messages), "kwargs": kwargs})
        if not self.responses:
            raise AssertionError("FakeToolModel ran out of scripted responses")
        reply = self.responses.pop(0).model_copy()
        return ChatResult(generations=[ChatGeneration(message=reply)])

    def bind_tools(self, tools, tool_choice=None, **kwargs):
        # .bind() returns a wrapper that passes these kwargs into _generate.
        return self.bind(tools=[t.name for t in tools], tool_choice=tool_choice, **kwargs)


def tool_call(name: str, args: dict, call_id: str = "call_1") -> AIMessage:
    """An AIMessage asking for one tool call, as the real model would send."""
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])
