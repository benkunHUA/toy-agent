from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Protocol

from openai.types.chat import ChatCompletionChunk, ChatCompletionMessageParam, ChatCompletionToolParam

from src.llm.types import Timeouts


class Provider(Protocol):
  name: str

  def stream_chat(
    self,
    messages: list[ChatCompletionMessageParam],
    *,
    tools: list[ChatCompletionToolParam] | None = None,
    tool_choice: Any | None = None,
    model: str,
    timeouts: Timeouts,
  ) -> Iterator[ChatCompletionChunk]: ...

