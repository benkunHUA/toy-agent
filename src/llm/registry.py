from __future__ import annotations

from functools import lru_cache

from src.llm.providers.base import Provider
from src.llm.providers.openai_compat import OpenAICompatProvider
from src.llm.types import ProviderName


@lru_cache(maxsize=1)
def providers() -> dict[ProviderName, Provider]:
  return {
    "deepseek": OpenAICompatProvider(
      name="deepseek",
      api_key_env="DEEPSEEK_API_KEY",
      base_url_env="DEEPSEEK_API_BASE_URL",
    ),
    "openai": OpenAICompatProvider(
      name="openai",
      api_key_env="OPENAI_API_KEY",
      base_url_env=None,
    ),
  }


def get_provider(name: ProviderName) -> Provider:
  return providers()[name]

