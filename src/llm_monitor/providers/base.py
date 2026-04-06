"""Abstract base class for LLM usage providers."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod

import httpx

from llm_monitor.models import CredentialError, ProviderStatus, SecretStr
from llm_monitor.security import run_key_command


class Provider(ABC):
    """Base class that every usage provider must implement."""

    @abstractmethod
    def name(self) -> str:
        """Short identifier used in config keys and cache paths (e.g. 'claude')."""

    @abstractmethod
    def display_name(self) -> str:
        """Human-readable provider name (e.g. 'Anthropic Claude')."""

    @abstractmethod
    def is_configured(self) -> bool:
        """Return True when credentials / config are present enough to attempt a fetch."""

    @abstractmethod
    async def fetch_usage(self, client: httpx.AsyncClient) -> ProviderStatus:
        """Fetch current usage from the provider API.

        Must never raise — all errors are returned inside ProviderStatus.errors.
        """

    @abstractmethod
    def auth_instructions(self) -> str:
        """User-facing message explaining how to set up this provider."""

    @property
    def allowed_hosts(self) -> list[str]:
        """Hostnames that this provider is allowed to contact."""
        return []

    # ------------------------------------------------------------------
    # Credential resolution (4-tier)
    # ------------------------------------------------------------------

    def _default_env_var(self) -> str | None:
        """Override to provide a default environment variable name for the API key."""
        return None

    def resolve_credential(self, config: dict) -> SecretStr | None:
        """Resolve a credential through the 4-tier chain.

        Resolution order:
        1. key_command (from config) — hard fail on error (raises CredentialError)
        2. Environment variable (from config ``env_var`` or ``_default_env_var()``)
        3. Keyring lookup (service = ``llm-monitor/<provider_name>``)
        4. Returns None

        If key_command is configured and fails, a CredentialError is raised —
        we do NOT fall through to later tiers.
        """
        provider_cfg = config.get("providers", {}).get(self.name(), {})

        # Tier 1: key_command
        key_cmd = provider_cfg.get("key_command")
        if key_cmd:
            # Hard fail — raises CredentialError on any problem
            return run_key_command(key_cmd)

        # Tier 2: environment variable
        env_var = provider_cfg.get("env_var") or self._default_env_var()
        if env_var:
            value = os.environ.get(env_var)
            if value:
                return SecretStr(value)

        # Tier 3: keyring
        try:
            import keyring as kr

            service = f"llm-monitor/{self.name()}"
            secret = kr.get_password(service, "api_key")
            if secret:
                return SecretStr(secret)
        except Exception:
            # keyring not available or failed — fall through
            pass

        # Tier 4: nothing found
        return None
