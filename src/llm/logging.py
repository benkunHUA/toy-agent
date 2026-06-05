from __future__ import annotations

import logging
from typing import Any


def get_logger() -> logging.Logger:
  logger = logging.getLogger("llm")
  if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
      fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
      datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
  return logger


def log_event(logger: logging.Logger, event: str, **fields: Any) -> None:
  payload = " ".join(f"{k}={fields[k]!r}" for k in sorted(fields.keys()))
  logger.info("%s %s", event, payload)

