from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from openai.types.chat import ChatCompletionChunk, ChatCompletionMessageParam

from src.llm import call_model
from src.tools import execute_tool_call, openai_tools

def query_engine(user_query: str) -> Iterator[str]:
  system_prompt = "你是一个专业的助手，你的任务是回答用户的问题。"
  messages: list[ChatCompletionMessageParam] = [{"role": "system", "content": system_prompt}]
  messages.append({"role": "user", "content": user_query})

  tools = openai_tools()
  max_rounds = 8

  for _ in range(max_rounds):
    content_parts: list[str] = []
    tool_call_acc: dict[int, dict[str, Any]] = {}
    last_chunk: ChatCompletionChunk | None = None

    for chunk in call_model(messages, tools=tools):
      last_chunk = chunk
      delta = chunk.choices[0].delta
      content = getattr(delta, "content", None)
      if content:
        content_parts.append(content)
        yield content

      tool_calls_delta = getattr(delta, "tool_calls", None)
      if not tool_calls_delta:
        continue

      for tc in tool_calls_delta:
        index = tc.index
        acc = tool_call_acc.setdefault(index, {"id": None, "name": None, "arguments": ""})
        if tc.id:
          acc["id"] = tc.id
        fn = getattr(tc, "function", None)
        if fn is None:
          continue
        if getattr(fn, "name", None):
          acc["name"] = fn.name
        if getattr(fn, "arguments", None):
          acc["arguments"] += fn.arguments

    if last_chunk is None:
      raise RuntimeError("模型未返回任何内容")

    tool_calls = []
    for i in sorted(tool_call_acc.keys()):
      acc = tool_call_acc[i]
      if not acc.get("id") or not acc.get("name"):
        continue
      tool_calls.append(
        {
          "id": acc["id"],
          "type": "function",
          "function": {"name": acc["name"], "arguments": acc["arguments"]},
        }
      )

    assistant_content = "".join(content_parts) if content_parts else None

    if not tool_calls:
      if assistant_content is None:
        raise RuntimeError("模型未生成 content，也未触发工具调用")
      messages.append({"role": "assistant", "content": assistant_content})
      return

    messages.append({"role": "assistant", "content": assistant_content, "tool_calls": tool_calls})
    for tc in tool_calls:
      result = execute_tool_call(tc["function"]["name"], tc["function"]["arguments"])
      messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})

  raise RuntimeError("工具调用轮次超出上限，可能陷入循环")
