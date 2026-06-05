from __future__ import annotations

import ast

from src.tools.tool_types import Tool


def _safe_calculate(expression: str) -> float:
  tree = ast.parse(expression, mode="eval")
  allowed_nodes = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Mod,
    ast.Pow,
    ast.USub,
    ast.UAdd,
    ast.Constant,
    ast.Load,
    ast.FloorDiv,
  )
  for node in ast.walk(tree):
    if not isinstance(node, allowed_nodes):
      raise ValueError("不支持的表达式")
    if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
      raise ValueError("仅支持数字常量")
  value = eval(compile(tree, "<expression>", "eval"), {"__builtins__": {}}, {})
  if not isinstance(value, (int, float)):
    raise ValueError("计算结果不是数字")
  return float(value)


def calculate(expression: str) -> str:
  return str(_safe_calculate(expression))


tool = Tool(
  name="calculate",
  description="安全地计算一个算术表达式（仅支持数字与 + - * / // % ** 和括号）",
  parameters={
    "type": "object",
    "properties": {"expression": {"type": "string"}},
    "required": ["expression"],
  },
  handler=calculate,
)

