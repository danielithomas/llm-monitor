"""Tests for the Ollama provider — cloud usage monitoring (alpha)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
import respx

from llm_monitor.models import ProviderStatus
from llm_monitor.providers.ollama import OllamaProvider, CLOUD_API_BASE

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
HOST_URL = "http://localhost:11434"
CLOUD_USAGE_URL = f"{CLOUD_API_BASE}/api/account/usage"
API_KEY = "ollama-test-key-abc123"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_fixture(name: str) -> dict:
    return json.loads((FIXTURES_DIR / name).read_text())


def _make_cloud_config(
    alpha: bool = True,
    cloud_enabled: bool = True,
    api_key_env: str = "OLLAMA_API_KEY",
) -> dict:
    return {
        "general": {
            "enable_alpha_features": alpha,
        },
        "providers": {
            "ollama": {
                "enabled": True,
                "host": HOST_URL,
                "cloud_enabled": cloud_enabled,
                "api_key_env": api_key_env,
            },
        },
    }


def _make_cloud_provider(
    monkeypatch,
    alpha: bool = True,
    cloud_enabled: bool = True,
    api_key: str = API_KEY,
) -> OllamaProvider:
    if api_key:
        monkeypatch.setenv("OLLAMA_API_KEY", api_key)
    config = _make_cloud_config(alpha=alpha, cloud_enabled=cloud_enabled)
    return OllamaProvider(config)


def _mock_local_host() -> None:
    """Mock the local host endpoints so they don't interfere."""
    respx.get(f"{HOST_URL}/api/tags").respond(200, json={"models": []})
    respx.get(f"{HOST_URL}/api/ps").respond(200, json={"models": []})


# ---------------------------------------------------------------------------
# Alpha feature gating
# ---------------------------------------------------------------------------


class TestAlphaGating:
    @respx.mock
    @pytest.mark.asyncio
    async def test_cloud_disabled_when_alpha_off(self, monkeypatch):
        """Cloud usage is not fetched when alpha features are disabled."""
        provider = _make_cloud_provider(monkeypatch, alpha=False)
        _mock_local_host()
        # Don't mock cloud endpoint — it should not be called

        async with httpx.AsyncClient() as client:
            status = await provider.fetch_usage(client)

        assert "cloud" not in status.extras

    @respx.mock
    @pytest.mark.asyncio
    async def test_cloud_disabled_when_cloud_enabled_false(self, monkeypatch):
        """Cloud usage not fetched when cloud_enabled is false."""
        provider = _make_cloud_provider(
            monkeypatch, alpha=True, cloud_enabled=False
        )
        _mock_local_host()

        async with httpx.AsyncClient() as client:
            status = await provider.fetch_usage(client)

        assert "cloud" not in status.extras

    @respx.mock
    @pytest.mark.asyncio
    async def test_cloud_enabled_when_both_flags_set(self, monkeypatch):
        """Cloud usage is fetched when both alpha and cloud_enabled are true."""
        provider = _make_cloud_provider(monkeypatch)
        _mock_local_host()
        respx.get(CLOUD_USAGE_URL).respond(
            200, json=_load_fixture("ollama_cloud_usage.json")
        )

        async with httpx.AsyncClient() as client:
            status = await provider.fetch_usage(client)

        assert "cloud" in status.extras
        assert status.extras["cloud"]["alpha"] is True

    @respx.mock
    @pytest.mark.asyncio
    async def test_alpha_warning_emitted(self, monkeypatch, capfd):
        """Alpha feature warning is printed to stderr."""
        import llm_monitor.providers.ollama as ollama_mod
        monkeypatch.setattr(ollama_mod, "_alpha_warning_emitted", False)

        provider = _make_cloud_provider(monkeypatch)
        _mock_local_host()
        respx.get(CLOUD_USAGE_URL).respond(404)

        async with httpx.AsyncClient() as client:
            await provider.fetch_usage(client)

        captured = capfd.readouterr()
        assert "alpha features are enabled" in captured.err.lower()


# ---------------------------------------------------------------------------
# Cloud usage parsing
# ---------------------------------------------------------------------------


class TestCloudUsageParsing:
    @respx.mock
    @pytest.mark.asyncio
    async def test_session_window_parsed(self, monkeypatch):
        """Cloud session usage window is correctly parsed."""
        provider = _make_cloud_provider(monkeypatch)
        _mock_local_host()
        respx.get(CLOUD_USAGE_URL).respond(
            200, json=_load_fixture("ollama_cloud_usage.json")
        )

        async with httpx.AsyncClient() as client:
            status = await provider.fetch_usage(client)

        session = next(
            (w for w in status.windows if w.name == "Cloud Session"), None
        )
        assert session is not None
        assert session.utilisation == 4.0
        assert session.unit == "percent"
        assert session.resets_at is not None

    @respx.mock
    @pytest.mark.asyncio
    async def test_weekly_window_parsed(self, monkeypatch):
        """Cloud weekly usage window is correctly parsed."""
        provider = _make_cloud_provider(monkeypatch)
        _mock_local_host()
        respx.get(CLOUD_USAGE_URL).respond(
            200, json=_load_fixture("ollama_cloud_usage.json")
        )

        async with httpx.AsyncClient() as client:
            status = await provider.fetch_usage(client)

        weekly = next(
            (w for w in status.windows if w.name == "Cloud Weekly"), None
        )
        assert weekly is not None
        assert weekly.utilisation == 14.3
        assert weekly.unit == "percent"

    @respx.mock
    @pytest.mark.asyncio
    async def test_plan_in_cloud_extras(self, monkeypatch):
        """Cloud plan type is reported in extras."""
        provider = _make_cloud_provider(monkeypatch)
        _mock_local_host()
        respx.get(CLOUD_USAGE_URL).respond(
            200, json=_load_fixture("ollama_cloud_usage.json")
        )

        async with httpx.AsyncClient() as client:
            status = await provider.fetch_usage(client)

        assert status.extras["cloud"]["plan"] == "pro"

    @respx.mock
    @pytest.mark.asyncio
    async def test_cloud_usage_status_thresholds(self, monkeypatch):
        """Cloud usage windows use compute_status for thresholds."""
        provider = _make_cloud_provider(monkeypatch)
        _mock_local_host()

        usage = _load_fixture("ollama_cloud_usage.json")
        usage["session"]["used_percent"] = 85.0
        respx.get(CLOUD_USAGE_URL).respond(200, json=usage)

        async with httpx.AsyncClient() as client:
            status = await provider.fetch_usage(client)

        session = next(w for w in status.windows if w.name == "Cloud Session")
        assert session.status == "warning"


# ---------------------------------------------------------------------------
# Cloud API key resolution
# ---------------------------------------------------------------------------


class TestCloudApiKey:
    @respx.mock
    @pytest.mark.asyncio
    async def test_missing_api_key_error(self, monkeypatch):
        """Missing API key produces error, no cloud data."""
        monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
        config = _make_cloud_config()
        provider = OllamaProvider(config)
        _mock_local_host()

        async with httpx.AsyncClient() as client:
            status = await provider.fetch_usage(client)

        assert any("API key not found" in e for e in status.errors)
        assert "cloud" not in status.extras

    @respx.mock
    @pytest.mark.asyncio
    async def test_custom_env_var(self, monkeypatch):
        """Custom api_key_env is respected."""
        monkeypatch.setenv("MY_OLLAMA_KEY", API_KEY)
        config = _make_cloud_config(api_key_env="MY_OLLAMA_KEY")
        provider = OllamaProvider(config)
        _mock_local_host()
        respx.get(CLOUD_USAGE_URL).respond(
            200, json=_load_fixture("ollama_cloud_usage.json")
        )

        async with httpx.AsyncClient() as client:
            status = await provider.fetch_usage(client)

        assert "cloud" in status.extras


# ---------------------------------------------------------------------------
# Cloud endpoint failures (graceful)
# ---------------------------------------------------------------------------


class TestCloudEndpointFailures:
    @respx.mock
    @pytest.mark.asyncio
    async def test_404_no_usage_endpoint(self, monkeypatch):
        """404 from /api/account/usage is expected (no stable API yet)."""
        provider = _make_cloud_provider(monkeypatch)
        _mock_local_host()
        respx.get(CLOUD_USAGE_URL).respond(404)

        async with httpx.AsyncClient() as client:
            status = await provider.fetch_usage(client)

        # Cloud section present but with no_usage_endpoint status
        assert status.extras["cloud"]["status"] == "no_usage_endpoint"
        # No cloud windows created
        cloud_windows = [
            w for w in status.windows if w.name.startswith("Cloud")
        ]
        assert len(cloud_windows) == 0

    @respx.mock
    @pytest.mark.asyncio
    async def test_401_auth_failure(self, monkeypatch):
        """401 from cloud endpoint reports auth error."""
        provider = _make_cloud_provider(monkeypatch)
        _mock_local_host()
        respx.get(CLOUD_USAGE_URL).respond(401)

        async with httpx.AsyncClient() as client:
            status = await provider.fetch_usage(client)

        assert any("401" in e for e in status.errors)

    @respx.mock
    @pytest.mark.asyncio
    async def test_429_backoff(self, monkeypatch):
        """429 from cloud endpoint reports rate limit."""
        provider = _make_cloud_provider(monkeypatch)
        _mock_local_host()
        respx.get(CLOUD_USAGE_URL).respond(429)

        async with httpx.AsyncClient() as client:
            status = await provider.fetch_usage(client)

        assert any("429" in e for e in status.errors)

    @respx.mock
    @pytest.mark.asyncio
    async def test_network_error(self, monkeypatch):
        """Network error to cloud is reported but local data unaffected."""
        provider = _make_cloud_provider(monkeypatch)
        _mock_local_host()
        respx.get(CLOUD_USAGE_URL).mock(
            side_effect=httpx.ConnectError("Connection refused")
        )

        async with httpx.AsyncClient() as client:
            status = await provider.fetch_usage(client)

        assert any("ollama.com" in e.lower() for e in status.errors)
        # Local data should still be present
        names = {w.name for w in status.windows}
        assert "Models Available" in names
