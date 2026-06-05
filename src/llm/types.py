from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


ProviderName = Literal["deepseek", "openai"]


@dataclass(frozen=True)
class Timeouts:
  total: float | None = 60.0
  connect: float | None = 5.0
  read: float | None = 60.0
  write: float | None = 10.0


@dataclass(frozen=True)
class RetryPolicy:
  max_retries: int = 2
  base_delay_s: float = 0.5
  max_delay_s: float = 4.0
  jitter_s: float = 0.2
  retry_if_no_output: bool = True


@dataclass(frozen=True)
class CallModelOptions:
  provider: ProviderName | None = None
  fallback_providers: list[ProviderName] = field(default_factory=list)
  timeouts: Timeouts = field(default_factory=Timeouts)
  retry: RetryPolicy = field(default_factory=RetryPolicy)
  request_id: str | None = None

