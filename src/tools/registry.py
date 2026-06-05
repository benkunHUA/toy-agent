from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from openai.types.chat import ChatCompletionToolParam

from src.tools.calculate import tool as calculate_tool
from src.tools.get_current_time import tool as get_current_time_tool
from src.tools.tool_types import Tool


def tools() -> list[Tool]:
  return [get_current_time_tool, calculate_tool]


def openai_tools() -> list[ChatCompletionToolParam]:
  return [
    {
      "type": "function",
      "function": {
        "name": t.name,
        "description": t.description,
        "parameters": t.parameters,
      },
    }
    for t in tools()
  ]


def tool_handlers() -> dict[str, Callable[..., Any]]:
  return {t.name: t.handler for t in tools()}


def _json_dumps(value: Any) -> str:
  if isinstance(value, str):
    return value
  return json.dumps(value, ensure_ascii=False)


def execute_tool_call(name: str, arguments_json: str) -> str:
  handler = tool_handlers().get(name)
  if handler is None:
    return _json_dumps({"error": f"未找到工具: {name}"})
  try:
    arguments = json.loads(arguments_json) if arguments_json else {}
  except json.JSONDecodeError as e:
    return _json_dumps({"error": f"工具参数不是合法 JSON: {e}"})
  if not isinstance(arguments, dict):
    return _json_dumps({"error": "工具参数必须是 JSON object"})
  try:
    return _json_dumps(handler(**arguments))
  except Exception as e:
    return _json_dumps({"error": f"工具执行失败: {e}"})

