from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from src.tools.tool_types import Tool


def get_current_time() -> str:
  dt = datetime.now(tz=ZoneInfo("Asia/Shanghai"))
  return dt.isoformat()


tool = Tool(
  name="get_current_time",
  description="获取当前上海时间（ISO 8601 字符串）",
  parameters={"type": "object", "properties": {}, "required": []},
  handler=get_current_time,
)

