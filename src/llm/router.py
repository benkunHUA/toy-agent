from __future__ import annotations

from dataclasses import dataclass

from src.llm.types import CallModelOptions, ProviderName


@dataclass(frozen=True)
class ProviderCandidate:
  provider: ProviderName
  model: str


def route(model: str, options: CallModelOptions) -> list[ProviderCandidate]:
  if options.provider is not None:
    candidates = [ProviderCandidate(provider=options.provider, model=model)]
  else:
    if model.startswith("deepseek-"):
      candidates = [ProviderCandidate(provider="deepseek", model=model)]
    elif model.startswith("gpt-") or model.startswith("o1-") or model.startswith("o3-"):
      candidates = [ProviderCandidate(provider="openai", model=model)]
    else:
      candidates = [ProviderCandidate(provider="deepseek", model=model)]

  for p in options.fallback_providers:
    if all(c.provider != p for c in candidates):
      candidates.append(ProviderCandidate(provider=p, model=model))

  return candidates

