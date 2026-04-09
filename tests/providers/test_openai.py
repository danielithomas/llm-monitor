"""Tests for the OpenAI provider."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import respx

from llm_monitor.models import ProviderStatus
from llm_monitor.providers.openai import OpenAIProvider, API_BASE

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"

ADMIN_KEY = "sk-admin-test-key-value-abcdef1234567890"
USAGE_URL = f"{API_BASE}/organization/usage/completions"
COSTS_URL = f"{API_BASE}/organization/costs"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_fixture(name: str) -> dict:
    return json.loads((FIXTURES_DIR / name).read_text())


def _make_config(
    enabled: bool = True,
    admin_key_env: str = "OPENAI_ADMIN_KEY",
) -> dict:
    return {
        "providers": {
            "openai": {
                "enabled": enabled,
                "admin_key_env": admin_key_env,
            },
        },
    }


def _make_provider(
    monkeypatch,
    admin_key: str = ADMIN_KEY,
    admin_key_env: str = "OPENAI_ADMIN_KEY",
) -> OpenAIProvider:
    """Create an OpenAIProvider with env-based credentials."""
    if admin_key:
        monkeypatch.setenv(admin_key_env, admin_key)
    config = _make_config(admin_key_env=admin_key_env)
    return OpenAIProvider(config)


def _mock_all_endpoints(
    usage: dict | None = None,
    costs: dict | None = None,
) -> None:
    """Set up respx mocks for all OpenAI endpoints."""
    respx.get(USAGE_URL).respond(
        200, json=usage or _load_fixture("openai_usage_completions.json")
    )
    respx.get(COSTS_URL).respond(
        200, json=costs or _load_fixture("openai_costs.json")
    )


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


class TestMetadata:
    def test_name(self):
        provider = OpenAIProvider(_make_config())
        assert provider.name() == "openai"

    def test_display_name(self):
        provider = OpenAIProvider(_make_config())
        assert provider.display_name() == "OpenAI"

    def test_auth_instructions(self):
        provider = OpenAIProvider(_make_config())
        instructions = provider.auth_instructions()
        assert "Admin API Key" in instructions
        assert "platform.openai.com" in instructions
        assert "OPENAI_ADMIN_KEY" in instructions

    def test_allowed_hosts(self):
        provider = OpenAIProvider(_make_config())
        assert provider.allowed_hosts == ["api.openai.com"]


# ---------------------------------------------------------------------------
# Credential resolution
# ---------------------------------------------------------------------------


class TestCredentialResolution:
    def test_admin_key_from_env(self, monkeypatch):
        monkeypatch.setenv("OPENAI_ADMIN_KEY", ADMIN_KEY)
        provider = OpenAIProvider(_make_config())
        key = provider._resolve_admin_key()
        assert key is not None
        assert key.get_secret_value() == ADMIN_KEY

    def test_admin_key_custom_env_var(self, monkeypatch):
        monkeypatch.setenv("MY_OPENAI_KEY", ADMIN_KEY)
        config = _make_config(admin_key_env="MY_OPENAI_KEY")
        provider = OpenAIProvider(config)
        key = provider._resolve_admin_key()
        assert key is not None
        assert key.get_secret_value() == ADMIN_KEY

    def test_admin_key_not_found(self, monkeypatch):
        monkeypatch.delenv("OPENAI_ADMIN_KEY", raising=False)
        provider = OpenAIProvider(_make_config())
        key = provider._resolve_admin_key()
        assert key is None

    def test_admin_key_repr_masked(self, monkeypatch):
        monkeypatch.setenv("OPENAI_ADMIN_KEY", ADMIN_KEY)
        provider = OpenAIProvider(_make_config())
        key = provider._resolve_admin_key()
        assert "sk-admin" not in repr(key)
        assert "***" in repr(key)


# ---------------------------------------------------------------------------
# is_configured
# ---------------------------------------------------------------------------


class TestIsConfigured:
    def test_configured_with_admin_key(self, monkeypatch):
        monkeypatch.setenv("OPENAI_ADMIN_KEY", ADMIN_KEY)
        provider = OpenAIProvider(_make_config())
        assert provider.is_configured() is True

    def test_not_configured_without_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_ADMIN_KEY", raising=False)
        provider = OpenAIProvider(_make_config())
        assert provider.is_configured() is False


# ---------------------------------------------------------------------------
# fetch_usage — full response
# ---------------------------------------------------------------------------


class TestFetchUsage:
    @respx.mock
    @pytest.mark.asyncio
    async def test_full_response(self, monkeypatch):
        provider = _make_provider(monkeypatch)
        _mock_all_endpoints()

        async with httpx.AsyncClient() as client:
            status = await provider.fetch_usage(client)

        assert isinstance(status, ProviderStatus)
        assert status.provider_name == "openai"
        assert status.provider_display == "OpenAI"
        assert status.errors == []

    @respx.mock
    @pytest.mark.asyncio
    async def test_spend_mtd_window(self, monkeypatch):
        """Costs across all buckets sum to a single Spend (MTD) window."""
        provider = _make_provider(monkeypatch)
        _mock_all_endpoints()

        async with httpx.AsyncClient() as client:
            status = await provider.fetch_usage(client)

        spend_windows = [w for w in status.windows if w.name == "Spend (MTD)"]
        assert len(spend_windows) == 1

        spend = spend_windows[0]
        assert spend.unit == "usd"
        # 1.25 + 0.45 + 0.98 = 2.68
        assert spend.raw_value == pytest.approx(2.68, abs=0.01)
        assert spend.status == "normal"

    @respx.mock
    @pytest.mark.asyncio
    async def test_per_model_usage(self, monkeypatch):
        """Usage endpoint populates per-model token counts."""
        provider = _make_provider(monkeypatch)
        _mock_all_endpoints()

        async with httpx.AsyncClient() as client:
            status = await provider.fetch_usage(client)

        models = {m.model: m for m in status.model_usage}
        assert "gpt-4o-2024-08-06" in models
        assert "gpt-4.1-mini-2025-04-14" in models

        gpt4o = models["gpt-4o-2024-08-06"]
        # 85000 + 56201 = 141201 (across two buckets)
        assert gpt4o.input_tokens == 141201
        # 12000 + 8500 = 20500
        assert gpt4o.output_tokens == 20500
        assert gpt4o.total_tokens == 141201 + 20500
        # 250 + 180 = 430
        assert gpt4o.request_count == 430

        mini = models["gpt-4.1-mini-2025-04-14"]
        assert mini.input_tokens == 56000
        assert mini.output_tokens == 9756
        assert mini.request_count == 220

    @respx.mock
    @pytest.mark.asyncio
    async def test_per_model_costs(self, monkeypatch):
        """Costs endpoint populates per-model cost in USD."""
        provider = _make_provider(monkeypatch)
        _mock_all_endpoints()

        async with httpx.AsyncClient() as client:
            status = await provider.fetch_usage(client)

        models = {m.model: m for m in status.model_usage}

        gpt4o = models["gpt-4o-2024-08-06"]
        # 1.25 + 0.98 = 2.23
        assert gpt4o.cost == pytest.approx(2.23, abs=0.01)
        assert gpt4o.period == "mtd"

        mini = models["gpt-4.1-mini-2025-04-14"]
        assert mini.cost == pytest.approx(0.45, abs=0.01)

    @respx.mock
    @pytest.mark.asyncio
    async def test_models_used_extras(self, monkeypatch):
        """Extras dict includes sorted models_used list."""
        provider = _make_provider(monkeypatch)
        _mock_all_endpoints()

        async with httpx.AsyncClient() as client:
            status = await provider.fetch_usage(client)

        assert "models_used" in status.extras
        assert status.extras["models_used"] == [
            "gpt-4.1-mini-2025-04-14",
            "gpt-4o-2024-08-06",
        ]

    @respx.mock
    @pytest.mark.asyncio
    async def test_top_model_spend_extras(self, monkeypatch):
        """Extras dict includes top_model_spend."""
        provider = _make_provider(monkeypatch)
        _mock_all_endpoints()

        async with httpx.AsyncClient() as client:
            status = await provider.fetch_usage(client)

        assert "top_model_spend" in status.extras
        top = status.extras["top_model_spend"]
        assert top["model"] == "gpt-4o-2024-08-06"
        assert top["cost"] == pytest.approx(2.23, abs=0.01)


# ---------------------------------------------------------------------------
# Token + cost merge
# ---------------------------------------------------------------------------


class TestModelMerge:
    @respx.mock
    @pytest.mark.asyncio
    async def test_cost_only_model(self, monkeypatch):
        """A model in costs but not usage still gets a ModelUsage entry."""
        provider = _make_provider(monkeypatch)

        # Usage has only gpt-4o, costs has gpt-4o + text-embedding
        usage = {
            "object": "page",
            "data": [{
                "object": "bucket",
                "start_time": 1736553600,
                "end_time": 1736640000,
                "results": [{
                    "object": "organization.usage.completions.result",
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "input_cached_tokens": 0,
                    "input_audio_tokens": 0,
                    "output_audio_tokens": 0,
                    "num_model_requests": 1,
                    "model": "gpt-4o-2024-08-06",
                }],
            }],
            "has_more": False,
            "next_page": None,
        }
        costs = {
            "object": "page",
            "data": [{
                "object": "bucket",
                "start_time": 1736553600,
                "end_time": 1736640000,
                "results": [
                    {
                        "object": "organization.costs.result",
                        "amount": {"value": 0.01, "currency": "usd"},
                        "line_item": "gpt-4o-2024-08-06",
                        "project_id": None,
                    },
                    {
                        "object": "organization.costs.result",
                        "amount": {"value": 0.005, "currency": "usd"},
                        "line_item": "text-embedding-3-small",
                        "project_id": None,
                    },
                ],
            }],
            "has_more": False,
            "next_page": None,
        }

        _mock_all_endpoints(usage=usage, costs=costs)

        async with httpx.AsyncClient() as client:
            status = await provider.fetch_usage(client)

        models = {m.model: m for m in status.model_usage}
        assert "text-embedding-3-small" in models
        embedding = models["text-embedding-3-small"]
        assert embedding.cost == pytest.approx(0.005)
        assert embedding.input_tokens is None
        assert embedding.output_tokens is None

    @respx.mock
    @pytest.mark.asyncio
    async def test_usage_only_model(self, monkeypatch):
        """A model in usage but not costs gets None cost."""
        provider = _make_provider(monkeypatch)

        usage = {
            "object": "page",
            "data": [{
                "object": "bucket",
                "start_time": 1736553600,
                "end_time": 1736640000,
                "results": [{
                    "object": "organization.usage.completions.result",
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "input_cached_tokens": 0,
                    "input_audio_tokens": 0,
                    "output_audio_tokens": 0,
                    "num_model_requests": 5,
                    "model": "gpt-4o-2024-08-06",
                }],
            }],
            "has_more": False,
            "next_page": None,
        }
        costs = {
            "object": "page",
            "data": [],
            "has_more": False,
            "next_page": None,
        }

        _mock_all_endpoints(usage=usage, costs=costs)

        async with httpx.AsyncClient() as client:
            status = await provider.fetch_usage(client)

        models = {m.model: m for m in status.model_usage}
        gpt4o = models["gpt-4o-2024-08-06"]
        assert gpt4o.input_tokens == 100
        assert gpt4o.cost is None


# ---------------------------------------------------------------------------
# Multi-bucket aggregation
# ---------------------------------------------------------------------------


class TestMultiBucketAggregation:
    @respx.mock
    @pytest.mark.asyncio
    async def test_tokens_summed_across_buckets(self, monkeypatch):
        """Token counts are summed across multiple time buckets."""
        provider = _make_provider(monkeypatch)
        _mock_all_endpoints()

        async with httpx.AsyncClient() as client:
            status = await provider.fetch_usage(client)

        models = {m.model: m for m in status.model_usage}
        gpt4o = models["gpt-4o-2024-08-06"]
        # Bucket 1: 85000 input, bucket 2: 56201 input
        assert gpt4o.input_tokens == 85000 + 56201

    @respx.mock
    @pytest.mark.asyncio
    async def test_costs_summed_across_buckets(self, monkeypatch):
        """Costs are summed across multiple time buckets for total spend."""
        provider = _make_provider(monkeypatch)
        _mock_all_endpoints()

        async with httpx.AsyncClient() as client:
            status = await provider.fetch_usage(client)

        spend = next(w for w in status.windows if w.name == "Spend (MTD)")
        # 1.25 + 0.45 + 0.98 = 2.68
        assert spend.raw_value == pytest.approx(2.68, abs=0.01)

    @respx.mock
    @pytest.mark.asyncio
    async def test_empty_buckets(self, monkeypatch):
        """Empty data arrays produce zero spend and no models."""
        provider = _make_provider(monkeypatch)
        empty = {
            "object": "page",
            "data": [],
            "has_more": False,
            "next_page": None,
        }
        _mock_all_endpoints(usage=empty, costs=empty)

        async with httpx.AsyncClient() as client:
            status = await provider.fetch_usage(client)

        assert status.errors == []
        spend = next(w for w in status.windows if w.name == "Spend (MTD)")
        assert spend.raw_value == pytest.approx(0.0)
        assert status.model_usage == []
        assert "models_used" not in status.extras


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    @respx.mock
    @pytest.mark.asyncio
    async def test_missing_admin_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_ADMIN_KEY", raising=False)
        provider = OpenAIProvider(_make_config())

        async with httpx.AsyncClient() as client:
            status = await provider.fetch_usage(client)

        assert len(status.errors) == 1
        assert "admin key not found" in status.errors[0]

    @respx.mock
    @pytest.mark.asyncio
    async def test_401_unauthorized(self, monkeypatch):
        provider = _make_provider(monkeypatch)
        respx.get(USAGE_URL).respond(401)
        respx.get(COSTS_URL).respond(401)

        async with httpx.AsyncClient() as client:
            status = await provider.fetch_usage(client)

        assert any("401" in e for e in status.errors)
        assert any("invalid or revoked" in e for e in status.errors)

    @respx.mock
    @pytest.mark.asyncio
    async def test_403_forbidden(self, monkeypatch):
        provider = _make_provider(monkeypatch)
        respx.get(USAGE_URL).respond(403)
        respx.get(COSTS_URL).respond(403)

        async with httpx.AsyncClient() as client:
            status = await provider.fetch_usage(client)

        assert any("403" in e for e in status.errors)
        assert any("api.usage.read" in e for e in status.errors)

    @respx.mock
    @pytest.mark.asyncio
    async def test_429_backoff(self, monkeypatch):
        provider = _make_provider(monkeypatch)
        respx.get(USAGE_URL).respond(429)
        respx.get(COSTS_URL).respond(429)

        async with httpx.AsyncClient() as client:
            status = await provider.fetch_usage(client)

        assert any("429" in e for e in status.errors)
        assert status.extras.get("_backoff") is True

    @respx.mock
    @pytest.mark.asyncio
    async def test_500_server_error(self, monkeypatch):
        provider = _make_provider(monkeypatch)
        respx.get(USAGE_URL).respond(500, text="Internal Server Error")
        respx.get(COSTS_URL).respond(500, text="Internal Server Error")

        async with httpx.AsyncClient() as client:
            status = await provider.fetch_usage(client)

        assert any("500" in e for e in status.errors)

    @respx.mock
    @pytest.mark.asyncio
    async def test_network_error(self, monkeypatch):
        provider = _make_provider(monkeypatch)
        respx.get(USAGE_URL).mock(side_effect=httpx.ConnectError("DNS failed"))
        respx.get(COSTS_URL).mock(side_effect=httpx.ConnectError("DNS failed"))

        async with httpx.AsyncClient() as client:
            status = await provider.fetch_usage(client)

        assert any("Cannot reach api.openai.com" in e for e in status.errors)

    @respx.mock
    @pytest.mark.asyncio
    async def test_timeout_error(self, monkeypatch):
        provider = _make_provider(monkeypatch)
        respx.get(USAGE_URL).mock(
            side_effect=httpx.ReadTimeout("Read timed out")
        )
        respx.get(COSTS_URL).mock(
            side_effect=httpx.ReadTimeout("Read timed out")
        )

        async with httpx.AsyncClient() as client:
            status = await provider.fetch_usage(client)

        assert any("timed out" in e for e in status.errors)

    @respx.mock
    @pytest.mark.asyncio
    async def test_partial_failure(self, monkeypatch):
        """Usage succeeds but costs fails — still returns usage data."""
        provider = _make_provider(monkeypatch)
        respx.get(USAGE_URL).respond(
            200,
            json=_load_fixture("openai_usage_completions.json"),
        )
        respx.get(COSTS_URL).respond(500, text="Error")

        async with httpx.AsyncClient() as client:
            status = await provider.fetch_usage(client)

        # Should have errors from costs endpoint
        assert len(status.errors) > 0
        # But model_usage should still be populated from usage endpoint
        assert len(status.model_usage) > 0
        # Spend window should still be created (with 0.0)
        spend = next(
            (w for w in status.windows if w.name == "Spend (MTD)"), None
        )
        assert spend is None  # No costs data → no spend window

    @respx.mock
    @pytest.mark.asyncio
    async def test_admin_key_command_failure(self, monkeypatch):
        """key_command failure raises CredentialError → error status."""
        monkeypatch.delenv("OPENAI_ADMIN_KEY", raising=False)
        config = {
            "providers": {
                "openai": {
                    "enabled": True,
                    "admin_key_command": "false",  # exits non-zero
                },
            },
        }
        provider = OpenAIProvider(config)

        async with httpx.AsyncClient() as client:
            status = await provider.fetch_usage(client)

        assert len(status.errors) == 1
        assert "command failed" in status.errors[0].lower()
