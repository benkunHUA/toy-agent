from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Tool:
  name: str
  description: str
  parameters: dict[str, Any]
  handler: Callable[..., Any]

