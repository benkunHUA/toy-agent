from __future__ import annotations

import os
from collections.abc import Iterator
from functools import lru_cache
from typing import Any

import httpx
from openai import OpenAI
from openai.types.chat import ChatCompletionChunk, ChatCompletionMessageParam, ChatCompletionToolParam

from src.llm.types import Timeouts


def _to_httpx_timeout(timeouts: Timeouts) -> httpx.Timeout:
  return httpx.Timeout(
    timeout=timeouts.total,
    connect=timeouts.connect,
    read=timeouts.read,
    write=timeouts.write,
  )


@lru_cache(maxsize=16)
def _get_client(api_key: str | None, base_url: str | None, timeout: float | None) -> OpenAI:
  return OpenAI(
    api_key=api_key,
    base_url=base_url,
    timeout=timeout,
    max_retries=0,
  )


class OpenAICompatProvider:
  def __init__(self, name: str, api_key_env: str, base_url_env: str | None = None):
    self.name = name
    self._api_key_env = api_key_env
    self._base_url_env = base_url_env

  def _get_credentials(self) -> tuple[str | None, str | None]:
    api_key = os.getenv(self._api_key_env)
    base_url = os.getenv(self._base_url_env) if self._base_url_env else None
    return api_key, base_url

  def stream_chat(
    self,
    messages: list[ChatCompletionMessageParam],
    *,
    tools: list[ChatCompletionToolParam] | None = None,
    tool_choice: Any | None = None,
    model: str,
    timeouts: Timeouts,
  ) -> Iterator[ChatCompletionChunk]:
    api_key, base_url = self._get_credentials()
    if not api_key:
      raise RuntimeError(f"缺少环境变量: {self._api_key_env}")

    client = _get_client(api_key, base_url, timeouts.total)
    kwargs: dict[str, Any] = {"model": model, "messages": messages, "stream": True, "timeout": _to_httpx_timeout(timeouts)}
    if tools:
      kwargs["tools"] = tools
    if tool_choice is not None:
      kwargs["tool_choice"] = tool_choice

    stream = client.chat.completions.create(**kwargs)
    for chunk in stream:
      yield chunk
