from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RetryState:
  attempt: int
  max_attempts: int


def backoff_delay_s(base_delay_s: float, max_delay_s: float, jitter_s: float, attempt: int) -> float:
  delay = min(max_delay_s, base_delay_s * (2 ** max(0, attempt - 1)))
  return max(0.0, delay + random.uniform(0.0, jitter_s))


def status_code_of_error(err: BaseException) -> int | None:
  value = getattr(err, "status_code", None)
  if isinstance(value, int):
    return value
  return None


def is_retryable_error(err: BaseException) -> bool:
  code = status_code_of_error(err)
  if code in (408, 409, 429, 500, 502, 503, 504):
    return True
  name = type(err).__name__.lower()
  if "timeout" in name:
    return True
  if "ratelimit" in name:
    return True
  if "connection" in name:
    return True
  return False


def retry_sleep(policy: Any, attempt: int) -> None:
  time.sleep(backoff_delay_s(policy.base_delay_s, policy.max_delay_s, policy.jitter_s, attempt))

