"""Tests for the Provider base class and registry."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import httpx
import pytest

from clawmeter.models import CredentialError, ProviderStatus, SecretStr
from clawmeter.providers import PROVIDERS, get_enabled_providers, register_provider
from clawmeter.providers.base import Provider


# ---------------------------------------------------------------------------
# Concrete test provider
# ---------------------------------------------------------------------------


class StubProvider(Provider):
    """Minimal concrete provider for testing the base class."""

    def __init__(self, config: dict | None = None) -> None:
        self._config = config or {}

    def name(self) -> str:
        return "test_provider"

    def display_name(self) -> str:
        return "Stub Provider"

    def is_configured(self) -> bool:
        return True

    async def fetch_usage(self, client: httpx.AsyncClient) -> ProviderStatus:
        from datetime import datetime, timezone

        return ProviderStatus(
            provider_name=self.name(),
            provider_display=self.display_name(),
            timestamp=datetime.now(timezone.utc),
            cached=False,
            cache_age_seconds=0,
        )

    def auth_instructions(self) -> str:
        return "No setup needed for stub provider."

    def _default_env_var(self) -> str | None:
        return "TEST_PROVIDER_API_KEY"


# ---------------------------------------------------------------------------
# resolve_credential tests
# ---------------------------------------------------------------------------


class TestResolveCredential:
    def test_key_command_success(self):
        """Tier 1: key_command returns a SecretStr when command succeeds."""
        config = {
            "providers": {
                "test_provider": {"key_command": "echo my-secret-key"},
            },
        }
        provider = StubProvider(config)
        result = provider.resolve_credential(config)
        assert isinstance(result, SecretStr)
        assert result.get_secret_value() == "my-secret-key"

    def test_key_command_hard_fail(self):
        """Tier 1: key_command failure raises CredentialError — does NOT fall through."""
        config = {
            "providers": {
                "test_provider": {"key_command": "false"},
            },
        }
        provider = StubProvider(config)
        with pytest.raises(CredentialError, match="key_command failed"):
            provider.resolve_credential(config)

    def test_key_command_timeout(self):
        """Tier 1: key_command timeout raises CredentialError."""
        import sys

        config = {
            "providers": {
                "test_provider": {
                    "key_command": f"{sys.executable} -c \"import time; time.sleep(30)\"",
                },
            },
        }
        provider = StubProvider(config)
        with pytest.raises(CredentialError, match="timed out"):
            provider.resolve_credential(config)

    def test_env_var(self, monkeypatch):
        """Tier 2: falls through to env var when no key_command is set."""
        monkeypatch.setenv("TEST_PROVIDER_API_KEY", "env-secret-123")
        config = {"providers": {"test_provider": {}}}
        provider = StubProvider(config)
        result = provider.resolve_credential(config)
        assert isinstance(result, SecretStr)
        assert result.get_secret_value() == "env-secret-123"

    def test_env_var_from_config(self, monkeypatch):
        """Tier 2: config ``env_var`` overrides _default_env_var."""
        monkeypatch.setenv("CUSTOM_VAR", "custom-env-secret")
        monkeypatch.delenv("TEST_PROVIDER_API_KEY", raising=False)
        config = {
            "providers": {
                "test_provider": {"env_var": "CUSTOM_VAR"},
            },
        }
        provider = StubProvider(config)
        result = provider.resolve_credential(config)
        assert isinstance(result, SecretStr)
        assert result.get_secret_value() == "custom-env-secret"

    def test_keyring_mock(self, monkeypatch):
        """Tier 3: falls through to keyring when no env var is set."""
        monkeypatch.delenv("TEST_PROVIDER_API_KEY", raising=False)

        mock_kr = MagicMock()
        mock_kr.get_password.return_value = "keyring-secret-456"

        with patch.dict("sys.modules", {"keyring": mock_kr}):
            config = {"providers": {"test_provider": {}}}
            provider = StubProvider(config)
            result = provider.resolve_credential(config)
            assert isinstance(result, SecretStr)
            assert result.get_secret_value() == "keyring-secret-456"
            mock_kr.get_password.assert_any_call(
                "clawmeter/test_provider", "api_key"
            )

    def test_no_credential(self, monkeypatch):
        """Tier 4: returns None when nothing is configured."""
        monkeypatch.delenv("TEST_PROVIDER_API_KEY", raising=False)

        # Mock keyring to return None
        mock_kr = MagicMock()
        mock_kr.get_password.return_value = None

        with patch.dict("sys.modules", {"keyring": mock_kr}):
            config = {"providers": {"test_provider": {}}}
            provider = StubProvider(config)
            result = provider.resolve_credential(config)
            assert result is None


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------


class TestRegisterProvider:
    def test_decorator_registers_class(self):
        """@register_provider adds the class to PROVIDERS dict."""

        @register_provider
        class DummyProvider(Provider):
            def name(self) -> str:
                return "dummy_test"

            def display_name(self) -> str:
                return "Dummy"

            def is_configured(self) -> bool:
                return False

            async def fetch_usage(self, client: httpx.AsyncClient) -> ProviderStatus:
                raise NotImplementedError

            def auth_instructions(self) -> str:
                return ""

        assert "dummy_test" in PROVIDERS
        assert PROVIDERS["dummy_test"] is DummyProvider

        # Clean up
        del PROVIDERS["dummy_test"]

    def test_claude_registered(self):
        """ClaudeProvider should be registered at import time."""
        assert "claude" in PROVIDERS


class TestGetEnabledProviders:
    def test_returns_enabled_providers(self):
        """Providers with enabled=True (or absent) are returned."""
        config = {
            "providers": {
                "claude": {"enabled": True},
            },
        }
        result = get_enabled_providers(config)
        names = [cls.__new__(cls).name() for cls in result]
        assert "claude" in names

    def test_filters_disabled_providers(self):
        """Providers with enabled=False are excluded."""
        config = {
            "providers": {
                "claude": {"enabled": False},
            },
        }
        result = get_enabled_providers(config)
        names = [cls.__new__(cls).name() for cls in result]
        assert "claude" not in names

    def test_default_enabled_when_key_absent(self):
        """A provider is enabled by default if its config section doesn't have 'enabled'."""
        config = {"providers": {"claude": {}}}
        result = get_enabled_providers(config)
        names = [cls.__new__(cls).name() for cls in result]
        assert "claude" in names
