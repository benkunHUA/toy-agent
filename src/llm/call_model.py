from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from time import perf_counter
from uuid import uuid4

from openai.types.chat import ChatCompletionChunk, ChatCompletionMessageParam, ChatCompletionToolParam

from src.llm.logging import get_logger, log_event
from src.llm.registry import get_provider
from src.llm.retry import is_retryable_error, retry_sleep, status_code_of_error
from src.llm.router import route
from src.llm.types import CallModelOptions


def call_model(
  messages: list[ChatCompletionMessageParam],
  *,
  tools: list[ChatCompletionToolParam] | None = None,
  tool_choice: Any | None = None,
  model: str = "deepseek-v4-pro",
  options: CallModelOptions | None = None,
) -> Iterator[ChatCompletionChunk]:
  opts = options or CallModelOptions()
  request_id = opts.request_id or uuid4().hex
  logger = get_logger()
  last_error: BaseException | None = None

  for candidate in route(model, opts):
    provider = get_provider(candidate.provider)
    max_attempts = opts.retry.max_retries + 1

    for attempt in range(1, max_attempts + 1):
      started_at = perf_counter()
      log_event(
        logger,
        "call_model_start",
        request_id=request_id,
        provider=candidate.provider,
        model=candidate.model,
        attempt=attempt,
      )
      try:
        stream = provider.stream_chat(
          messages,
          tools=tools,
          tool_choice=tool_choice,
          model=candidate.model,
          timeouts=opts.timeouts,
        )
        first_chunk = next(stream)
      except StopIteration as e:
        raise RuntimeError("模型未返回任何内容") from e
      except Exception as e:
        last_error = e
        log_event(
          logger,
          "call_model_error",
          request_id=request_id,
          provider=candidate.provider,
          model=candidate.model,
          attempt=attempt,
          error_type=type(e).__name__,
          status_code=status_code_of_error(e),
        )
        can_retry = (
          opts.retry.retry_if_no_output
          and attempt < max_attempts
          and is_retryable_error(e)
        )
        if can_retry:
          retry_sleep(opts.retry, attempt)
          continue
        break

      first_latency_ms = int((perf_counter() - started_at) * 1000)
      log_event(
        logger,
        "call_model_first_chunk",
        request_id=request_id,
        provider=candidate.provider,
        model=candidate.model,
        attempt=attempt,
        latency_ms=first_latency_ms,
      )

      yield first_chunk

      try:
        for chunk in stream:
          yield chunk
      except Exception as e:
        last_error = e
        log_event(
          logger,
          "call_model_stream_error",
          request_id=request_id,
          provider=candidate.provider,
          model=candidate.model,
          attempt=attempt,
          error_type=type(e).__name__,
          status_code=status_code_of_error(e),
        )
        raise
      finally:
        total_ms = int((perf_counter() - started_at) * 1000)
        log_event(
          logger,
          "call_model_done",
          request_id=request_id,
          provider=candidate.provider,
          model=candidate.model,
          attempt=attempt,
          total_ms=total_ms,
        )

      return

  raise RuntimeError(f"call_model 失败: {type(last_error).__name__ if last_error else 'unknown'}")
