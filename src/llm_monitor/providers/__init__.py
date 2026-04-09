"""Provider registry for llm-monitor."""

from __future__ import annotations

from typing import TYPE_CHECKING

from llm_monitor.providers.base import Provider

# Module-level registry: provider name -> provider class
PROVIDERS: dict[str, type[Provider]] = {}


def register_provider(cls: type[Provider]) -> type[Provider]:
    """Class decorator that registers a provider in the global registry.

    Uses ``cls.__new__(cls)`` to instantiate without calling ``__init__``
    so we can read ``name()`` for the registry key.
    """
    instance = cls.__new__(cls)
    provider_name = instance.name()
    PROVIDERS[provider_name] = cls
    return cls


def get_enabled_providers(config: dict) -> list[type[Provider]]:
    """Return provider classes that are enabled in the configuration.

    A provider is enabled when ``providers.<name>.enabled`` is truthy
    (defaults to True if the key is absent).
    """
    enabled: list[type[Provider]] = []
    providers_cfg = config.get("providers", {})
    for name, cls in PROVIDERS.items():
        section = providers_cfg.get(name, {})
        if section.get("enabled", True):
            enabled.append(cls)
    return enabled


# Import concrete providers to trigger registration
from llm_monitor.providers.claude import ClaudeProvider  # noqa: E402, F401
from llm_monitor.providers.grok import GrokProvider  # noqa: E402, F401
from llm_monitor.providers.openai import OpenAIProvider  # noqa: E402, F401
