from __future__ import annotations

import os
from collections.abc import Iterator
from functools import lru_cache
from typing import Any

from openai import OpenAI
from openai.types.chat import ChatCompletionChunk, ChatCompletionMessageParam, ChatCompletionToolParam


@lru_cache(maxsize=1)
def get_client() -> OpenAI:
  return OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_API_BASE_URL"),
  )


def query(
  messages: list[ChatCompletionMessageParam],
  *,
  tools: list[ChatCompletionToolParam] | None = None,
  tool_choice: Any | None = None,
  model: str = "deepseek-v4-pro",
) -> Iterator[ChatCompletionChunk]:
  client = get_client()
  kwargs: dict[str, Any] = {"model": model, "messages": messages, "stream": True}
  if tools:
    kwargs["tools"] = tools
  if tool_choice is not None:
    kwargs["tool_choice"] = tool_choice
  stream = client.chat.completions.create(**kwargs)
  for chunk in stream:
    yield chunk

