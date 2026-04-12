"""Tests for the xAI Grok provider."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import respx

from clawmeter.models import ProviderStatus
from clawmeter.providers.grok import GrokProvider, MANAGEMENT_API_BASE

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
TEAM_ID = "team-test-123"
BASE_URL = f"{MANAGEMENT_API_BASE}/billing/teams/{TEAM_ID}"

MGMT_KEY = "xai-mgmt-test-key-value-abcdef1234567890"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_fixture(name: str) -> dict:
    return json.loads((FIXTURES_DIR / name).read_text())


def _make_config(
    enabled: bool = True,
    team_id: str = TEAM_ID,
    management_key_env: str = "XAI_MANAGEMENT_KEY",
) -> dict:
    return {
        "providers": {
            "grok": {
                "enabled": enabled,
                "team_id": team_id,
                "management_key_env": management_key_env,
            },
        },
    }


def _make_provider(
    monkeypatch,
    team_id: str = TEAM_ID,
    mgmt_key: str = MGMT_KEY,
    mgmt_key_env: str = "XAI_MANAGEMENT_KEY",
) -> GrokProvider:
    """Create a GrokProvider with env-based credentials."""
    if mgmt_key:
        monkeypatch.setenv(mgmt_key_env, mgmt_key)
    config = _make_config(team_id=team_id, management_key_env=mgmt_key_env)
    return GrokProvider(config)


def _mock_all_endpoints(
    invoice: dict | None = None,
    limits: dict | None = None,
    balance: dict | None = None,
    analytics: dict | None = None,
) -> None:
    """Set up respx mocks for all management API endpoints."""
    respx.get(f"{BASE_URL}/postpaid/invoice/preview").respond(
        200, json=invoice or _load_fixture("grok_invoice_preview.json")
    )
    respx.get(f"{BASE_URL}/postpaid/spending-limits").respond(
        200, json=limits or _load_fixture("grok_spending_limits.json")
    )
    respx.get(f"{BASE_URL}/prepaid/balance").respond(
        200, json=balance or _load_fixture("grok_prepaid_balance.json")
    )
    respx.post(f"{BASE_URL}/usage").respond(
        200, json=analytics or _load_fixture("grok_usage_analytics.json")
    )


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


class TestMetadata:
    def test_name(self):
        provider = GrokProvider(_make_config())
        assert provider.name() == "grok"

    def test_display_name(self):
        provider = GrokProvider(_make_config())
        assert provider.display_name() == "xAI Grok"

    def test_auth_instructions(self):
        provider = GrokProvider(_make_config())
        instructions = provider.auth_instructions()
        assert "Management Key" in instructions
        assert "console.x.ai" in instructions
        assert "XAI_MANAGEMENT_KEY" in instructions

    def test_allowed_hosts(self):
        provider = GrokProvider(_make_config())
        assert "management-api.x.ai" in provider.allowed_hosts
        assert "api.x.ai" in provider.allowed_hosts


# ---------------------------------------------------------------------------
# Credential resolution
# ---------------------------------------------------------------------------


class TestCredentialResolution:
    def test_management_key_from_env(self, monkeypatch):
        monkeypatch.setenv("XAI_MANAGEMENT_KEY", MGMT_KEY)
        provider = GrokProvider(_make_config())
        key = provider._resolve_management_key()
        assert key is not None
        assert key.get_secret_value() == MGMT_KEY

    def test_management_key_custom_env_var(self, monkeypatch):
        monkeypatch.setenv("MY_CUSTOM_XAI_KEY", MGMT_KEY)
        config = _make_config(management_key_env="MY_CUSTOM_XAI_KEY")
        provider = GrokProvider(config)
        key = provider._resolve_management_key()
        assert key is not None
        assert key.get_secret_value() == MGMT_KEY

    def test_management_key_not_set(self, monkeypatch):
        monkeypatch.delenv("XAI_MANAGEMENT_KEY", raising=False)
        provider = GrokProvider(_make_config())
        key = provider._resolve_management_key()
        assert key is None

    def test_team_id_from_config(self):
        provider = GrokProvider(_make_config(team_id="my-team"))
        assert provider._team_id == "my-team"

    def test_team_id_from_env(self, monkeypatch):
        monkeypatch.setenv("XAI_TEAM_ID", "env-team-id")
        config = _make_config(team_id="")
        provider = GrokProvider(config)
        assert provider._team_id == "env-team-id"

    def test_team_id_config_takes_precedence(self, monkeypatch):
        monkeypatch.setenv("XAI_TEAM_ID", "env-team-id")
        config = _make_config(team_id="config-team-id")
        provider = GrokProvider(config)
        assert provider._team_id == "config-team-id"


# ---------------------------------------------------------------------------
# is_configured
# ---------------------------------------------------------------------------


class TestIsConfigured:
    def test_configured_with_key_and_team(self, monkeypatch):
        provider = _make_provider(monkeypatch)
        assert provider.is_configured() is True

    def test_not_configured_without_key(self, monkeypatch):
        monkeypatch.delenv("XAI_MANAGEMENT_KEY", raising=False)
        provider = GrokProvider(_make_config())
        assert provider.is_configured() is False

    def test_not_configured_without_team_id(self, monkeypatch):
        monkeypatch.setenv("XAI_MANAGEMENT_KEY", MGMT_KEY)
        monkeypatch.delenv("XAI_TEAM_ID", raising=False)
        provider = GrokProvider(_make_config(team_id=""))
        assert provider.is_configured() is False


# ---------------------------------------------------------------------------
# fetch_usage — successful responses
# ---------------------------------------------------------------------------


class TestFetchUsage:
    @respx.mock
    @pytest.mark.asyncio
    async def test_full_response_all_windows(self, monkeypatch):
        """200 from all endpoints produces 3 UsageWindow objects."""
        provider = _make_provider(monkeypatch)
        _mock_all_endpoints()

        async with httpx.AsyncClient() as client:
            status = await provider.fetch_usage(client)

        assert isinstance(status, ProviderStatus)
        assert len(status.errors) == 0
        assert len(status.windows) == 3
        names = {w.name for w in status.windows}
        assert "Spend (MTD)" in names
        assert "Spend vs Limit" in names
        assert "Prepaid Balance" in names

    @respx.mock
    @pytest.mark.asyncio
    async def test_spend_mtd_value(self, monkeypatch):
        """Invoice preview totalWithCorr is correctly parsed to USD."""
        provider = _make_provider(monkeypatch)
        _mock_all_endpoints()

        async with httpx.AsyncClient() as client:
            status = await provider.fetch_usage(client)

        mtd = next(w for w in status.windows if w.name == "Spend (MTD)")
        # totalWithCorr.val = "145000" cents = $1450.00
        assert mtd.raw_value == 1450.00
        assert mtd.unit == "usd"

    @respx.mock
    @pytest.mark.asyncio
    async def test_spend_vs_limit_percentage(self, monkeypatch):
        """Spend vs Limit utilisation is correctly calculated."""
        provider = _make_provider(monkeypatch)
        _mock_all_endpoints()

        async with httpx.AsyncClient() as client:
            status = await provider.fetch_usage(client)

        limit_w = next(w for w in status.windows if w.name == "Spend vs Limit")
        # $1450.00 / $5000.00 = 29%
        assert limit_w.utilisation == pytest.approx(29.0, rel=0.01)
        assert limit_w.raw_value == 1450.00
        assert limit_w.raw_limit == 5000.00
        assert limit_w.unit == "percent"

    @respx.mock
    @pytest.mark.asyncio
    async def test_prepaid_balance_value(self, monkeypatch):
        """Prepaid balance total is correctly parsed to USD."""
        provider = _make_provider(monkeypatch)
        _mock_all_endpoints()

        async with httpx.AsyncClient() as client:
            status = await provider.fetch_usage(client)

        balance = next(
            w for w in status.windows if w.name == "Prepaid Balance"
        )
        # total.val = "7500" cents = $75.00
        assert balance.raw_value == 75.00
        assert balance.unit == "usd"

    @respx.mock
    @pytest.mark.asyncio
    async def test_model_usage_from_invoice(self, monkeypatch):
        """Per-model line items are parsed into ModelUsage entries."""
        provider = _make_provider(monkeypatch)
        _mock_all_endpoints()

        async with httpx.AsyncClient() as client:
            status = await provider.fetch_usage(client)

        assert len(status.model_usage) >= 2
        models = {m.model for m in status.model_usage}
        assert "grok-3" in models
        assert "grok-3-mini" in models

        grok3 = next(m for m in status.model_usage if m.model == "grok-3")
        # grok-3: input 450.00 + output 750.00 = 1200.00 cents = $12.00
        assert grok3.cost == pytest.approx(12.00, rel=0.01)
        assert grok3.period == "mtd"

    @respx.mock
    @pytest.mark.asyncio
    async def test_billing_cycle_in_extras(self, monkeypatch):
        """Billing cycle year/month stored in extras."""
        provider = _make_provider(monkeypatch)
        _mock_all_endpoints()

        async with httpx.AsyncClient() as client:
            status = await provider.fetch_usage(client)

        assert status.extras.get("billing_cycle") == {
            "year": 2026,
            "month": 4,
        }

    @respx.mock
    @pytest.mark.asyncio
    async def test_models_used_in_extras(self, monkeypatch):
        """Models used list stored in extras."""
        provider = _make_provider(monkeypatch)
        _mock_all_endpoints()

        async with httpx.AsyncClient() as client:
            status = await provider.fetch_usage(client)

        assert status.extras.get("models_used") == [
            "grok-3",
            "grok-3-mini",
        ]

    @respx.mock
    @pytest.mark.asyncio
    async def test_spend_vs_limit_status_thresholds(self, monkeypatch):
        """Spend vs Limit uses compute_status for threshold detection."""
        provider = _make_provider(monkeypatch)

        # Create a scenario where spend is 85% of limit (warning)
        invoice = _load_fixture("grok_invoice_preview.json")
        invoice["coreInvoice"]["totalWithCorr"]["val"] = "425000"  # $4250

        _mock_all_endpoints(invoice=invoice)

        async with httpx.AsyncClient() as client:
            status = await provider.fetch_usage(client)

        limit_w = next(w for w in status.windows if w.name == "Spend vs Limit")
        # $4250 / $5000 = 85% -> warning
        assert limit_w.utilisation == pytest.approx(85.0, rel=0.01)
        assert limit_w.status == "warning"


# ---------------------------------------------------------------------------
# fetch_usage — invoice line parsing edge cases
# ---------------------------------------------------------------------------


class TestInvoiceParsing:
    @respx.mock
    @pytest.mark.asyncio
    async def test_empty_invoice_lines(self, monkeypatch):
        """Empty lines array still produces spend window from total.

        Model usage may come from analytics even when invoice lines are empty.
        """
        provider = _make_provider(monkeypatch)

        invoice = _load_fixture("grok_invoice_preview.json")
        invoice["coreInvoice"]["lines"] = []
        analytics = {"timeSeries": [], "limitReached": False}

        _mock_all_endpoints(invoice=invoice, analytics=analytics)

        async with httpx.AsyncClient() as client:
            status = await provider.fetch_usage(client)

        mtd = next(
            (w for w in status.windows if w.name == "Spend (MTD)"), None
        )
        assert mtd is not None
        assert mtd.raw_value == 1450.00
        assert len(status.model_usage) == 0

    @respx.mock
    @pytest.mark.asyncio
    async def test_zero_spending_limit(self, monkeypatch):
        """Zero spending limit does not produce Spend vs Limit window."""
        provider = _make_provider(monkeypatch)

        limits = _load_fixture("grok_spending_limits.json")
        limits["spendingLimits"]["effectiveHardSl"]["val"] = "0"

        _mock_all_endpoints(limits=limits)

        async with httpx.AsyncClient() as client:
            status = await provider.fetch_usage(client)

        names = {w.name for w in status.windows}
        assert "Spend vs Limit" not in names


# ---------------------------------------------------------------------------
# fetch_usage — usage analytics parsing
# ---------------------------------------------------------------------------


class TestUsageAnalytics:
    @respx.mock
    @pytest.mark.asyncio
    async def test_analytics_sums_daily_values(self, monkeypatch):
        """Usage analytics sums daily data points per model."""
        provider = _make_provider(monkeypatch)

        analytics = _load_fixture("grok_usage_analytics.json")
        _mock_all_endpoints(analytics=analytics)

        async with httpx.AsyncClient() as client:
            status = await provider.fetch_usage(client)

        # grok-3 daily values: 3.25 + 5.10 + 2.80 + 4.50 + 1.20 + 0.0 + 6.30 = 23.15
        # grok-3-mini: 0.50 + 0.75 + 0.30 + 0.60 + 0.40 + 0.0 + 0.95 = 3.50
        # But invoice data already provides cost, so analytics doesn't override it
        grok3 = next(m for m in status.model_usage if m.model == "grok-3")
        # Invoice cost takes precedence (12.00)
        assert grok3.cost == pytest.approx(12.00, rel=0.01)

    @respx.mock
    @pytest.mark.asyncio
    async def test_analytics_new_model_not_in_invoice(self, monkeypatch):
        """Models in analytics but not in invoice get new entries."""
        provider = _make_provider(monkeypatch)

        analytics = {
            "timeSeries": [
                {
                    "group": ["grok-3"],
                    "groupLabels": ["grok-3"],
                    "dataPoints": [
                        {"timestamp": "2026-04-01T00:00:00Z", "values": [5.0]}
                    ],
                },
                {
                    "group": ["grok-2"],
                    "groupLabels": ["grok-2"],
                    "dataPoints": [
                        {"timestamp": "2026-04-01T00:00:00Z", "values": [1.50]}
                    ],
                },
            ],
            "limitReached": False,
        }
        _mock_all_endpoints(analytics=analytics)

        async with httpx.AsyncClient() as client:
            status = await provider.fetch_usage(client)

        models = {m.model for m in status.model_usage}
        assert "grok-2" in models
        grok2 = next(m for m in status.model_usage if m.model == "grok-2")
        assert grok2.cost == pytest.approx(1.50, rel=0.01)

    @respx.mock
    @pytest.mark.asyncio
    async def test_empty_analytics_response(self, monkeypatch):
        """Empty analytics time series is handled gracefully."""
        provider = _make_provider(monkeypatch)

        analytics = {"timeSeries": [], "limitReached": False}
        _mock_all_endpoints(analytics=analytics)

        async with httpx.AsyncClient() as client:
            status = await provider.fetch_usage(client)

        assert len(status.errors) == 0
        # Model usage still comes from invoice
        assert len(status.model_usage) >= 2


# ---------------------------------------------------------------------------
# fetch_usage — error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_missing_management_key(self, monkeypatch):
        """Missing management key returns error without HTTP calls."""
        monkeypatch.delenv("XAI_MANAGEMENT_KEY", raising=False)
        provider = GrokProvider(_make_config())

        async with httpx.AsyncClient() as client:
            status = await provider.fetch_usage(client)

        assert len(status.errors) == 1
        assert "management key not found" in status.errors[0].lower()

    @pytest.mark.asyncio
    async def test_missing_team_id(self, monkeypatch):
        """Missing team ID returns error without HTTP calls."""
        monkeypatch.setenv("XAI_MANAGEMENT_KEY", MGMT_KEY)
        monkeypatch.delenv("XAI_TEAM_ID", raising=False)
        provider = GrokProvider(_make_config(team_id=""))

        async with httpx.AsyncClient() as client:
            status = await provider.fetch_usage(client)

        assert len(status.errors) == 1
        assert "team id" in status.errors[0].lower()

    @respx.mock
    @pytest.mark.asyncio
    async def test_401_auth_failure(self, monkeypatch):
        """401 from invoice preview reports auth error."""
        provider = _make_provider(monkeypatch)

        respx.get(f"{BASE_URL}/postpaid/invoice/preview").respond(401)
        respx.get(f"{BASE_URL}/postpaid/spending-limits").respond(401)
        respx.get(f"{BASE_URL}/prepaid/balance").respond(401)
        respx.post(f"{BASE_URL}/usage").respond(401)

        async with httpx.AsyncClient() as client:
            status = await provider.fetch_usage(client)

        assert len(status.errors) > 0
        assert any("401" in e for e in status.errors)

    @respx.mock
    @pytest.mark.asyncio
    async def test_403_access_denied(self, monkeypatch):
        """403 reports access denied error."""
        provider = _make_provider(monkeypatch)

        respx.get(f"{BASE_URL}/postpaid/invoice/preview").respond(403)
        respx.get(f"{BASE_URL}/postpaid/spending-limits").respond(200, json=_load_fixture("grok_spending_limits.json"))
        respx.get(f"{BASE_URL}/prepaid/balance").respond(200, json=_load_fixture("grok_prepaid_balance.json"))
        respx.post(f"{BASE_URL}/usage").respond(200, json=_load_fixture("grok_usage_analytics.json"))

        async with httpx.AsyncClient() as client:
            status = await provider.fetch_usage(client)

        assert any("403" in e for e in status.errors)

    @respx.mock
    @pytest.mark.asyncio
    async def test_429_backoff(self, monkeypatch):
        """429 response sets _backoff flag in extras."""
        provider = _make_provider(monkeypatch)

        respx.get(f"{BASE_URL}/postpaid/invoice/preview").respond(429)
        respx.get(f"{BASE_URL}/postpaid/spending-limits").respond(200, json=_load_fixture("grok_spending_limits.json"))
        respx.get(f"{BASE_URL}/prepaid/balance").respond(200, json=_load_fixture("grok_prepaid_balance.json"))
        respx.post(f"{BASE_URL}/usage").respond(200, json=_load_fixture("grok_usage_analytics.json"))

        async with httpx.AsyncClient() as client:
            status = await provider.fetch_usage(client)

        assert any("429" in e for e in status.errors)
        assert status.extras.get("_backoff") is True

    @respx.mock
    @pytest.mark.asyncio
    async def test_500_error(self, monkeypatch):
        """Non-200 status codes produce errors."""
        provider = _make_provider(monkeypatch)

        respx.get(f"{BASE_URL}/postpaid/invoice/preview").respond(
            500, text="Internal Server Error"
        )
        respx.get(f"{BASE_URL}/postpaid/spending-limits").respond(200, json=_load_fixture("grok_spending_limits.json"))
        respx.get(f"{BASE_URL}/prepaid/balance").respond(200, json=_load_fixture("grok_prepaid_balance.json"))
        respx.post(f"{BASE_URL}/usage").respond(200, json=_load_fixture("grok_usage_analytics.json"))

        async with httpx.AsyncClient() as client:
            status = await provider.fetch_usage(client)

        assert any("500" in e for e in status.errors)

    @respx.mock
    @pytest.mark.asyncio
    async def test_network_error(self, monkeypatch):
        """Connection error produces error in ProviderStatus."""
        provider = _make_provider(monkeypatch)

        respx.get(f"{BASE_URL}/postpaid/invoice/preview").mock(
            side_effect=httpx.ConnectError("Connection refused")
        )
        respx.get(f"{BASE_URL}/postpaid/spending-limits").mock(
            side_effect=httpx.ConnectError("Connection refused")
        )
        respx.get(f"{BASE_URL}/prepaid/balance").mock(
            side_effect=httpx.ConnectError("Connection refused")
        )
        respx.post(f"{BASE_URL}/usage").mock(
            side_effect=httpx.ConnectError("Connection refused")
        )

        async with httpx.AsyncClient() as client:
            status = await provider.fetch_usage(client)

        assert len(status.errors) > 0
        assert any("Cannot reach" in e for e in status.errors)

    @respx.mock
    @pytest.mark.asyncio
    async def test_partial_failure_still_returns_data(self, monkeypatch):
        """One endpoint failing doesn't block others."""
        provider = _make_provider(monkeypatch)

        # Invoice fails, but balance and limits succeed
        respx.get(f"{BASE_URL}/postpaid/invoice/preview").respond(500, text="error")
        respx.get(f"{BASE_URL}/postpaid/spending-limits").respond(
            200, json=_load_fixture("grok_spending_limits.json")
        )
        respx.get(f"{BASE_URL}/prepaid/balance").respond(
            200, json=_load_fixture("grok_prepaid_balance.json")
        )
        respx.post(f"{BASE_URL}/usage").respond(
            200, json=_load_fixture("grok_usage_analytics.json")
        )

        async with httpx.AsyncClient() as client:
            status = await provider.fetch_usage(client)

        # Should have errors but also some windows
        assert len(status.errors) > 0
        # Prepaid balance should still be present
        names = {w.name for w in status.windows}
        assert "Prepaid Balance" in names

    @respx.mock
    @pytest.mark.asyncio
    async def test_timeout_error(self, monkeypatch):
        """Timeout produces error in ProviderStatus."""
        provider = _make_provider(monkeypatch)

        respx.get(f"{BASE_URL}/postpaid/invoice/preview").mock(
            side_effect=httpx.ReadTimeout("Read timed out")
        )
        respx.get(f"{BASE_URL}/postpaid/spending-limits").respond(
            200, json=_load_fixture("grok_spending_limits.json")
        )
        respx.get(f"{BASE_URL}/prepaid/balance").respond(
            200, json=_load_fixture("grok_prepaid_balance.json")
        )
        respx.post(f"{BASE_URL}/usage").respond(
            200, json=_load_fixture("grok_usage_analytics.json")
        )

        async with httpx.AsyncClient() as client:
            status = await provider.fetch_usage(client)

        assert any("timed out" in e for e in status.errors)


# ---------------------------------------------------------------------------
# USD cents conversion
# ---------------------------------------------------------------------------


class TestCentsConversion:
    def test_string_cents_to_usd(self):
        assert GrokProvider._cents_to_usd("14500") == 145.00

    def test_int_cents_to_usd(self):
        assert GrokProvider._cents_to_usd(14500) == 145.00

    def test_zero_cents(self):
        assert GrokProvider._cents_to_usd("0") == 0.0

    def test_none_returns_none(self):
        assert GrokProvider._cents_to_usd(None) is None

    def test_invalid_string_returns_none(self):
        assert GrokProvider._cents_to_usd("not-a-number") is None
