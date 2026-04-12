"""Tests for the JSON output formatter."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from clawmeter.formatters.json_fmt import format_json, format_resets_in_human
from clawmeter.models import ProviderStatus, SecretStr, UsageWindow

# Fixed "now" used by all resets_in_human tests so results are deterministic.
_FROZEN_NOW = datetime(2026, 4, 5, 10, 0, 0, tzinfo=timezone.utc)


def _patch_now():
    """Patch datetime.now inside the json_fmt module to return _FROZEN_NOW."""
    real_datetime = datetime

    class FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):  # noqa: ANN001
            return _FROZEN_NOW

    return patch("clawmeter.formatters.json_fmt.datetime", FrozenDatetime)


# ---------------------------------------------------------------------------
# format_resets_in_human
# ---------------------------------------------------------------------------


class TestFormatResetsInHuman:
    """Tests for the resets_in_human helper."""

    def test_none_returns_none(self) -> None:
        assert format_resets_in_human(None) is None

    def test_two_hours(self) -> None:
        future = _FROZEN_NOW + timedelta(hours=2, minutes=15)
        with _patch_now():
            result = format_resets_in_human(future)
        assert result is not None
        assert "2h" in result
        assert "15m" in result

    def test_two_days(self) -> None:
        future = _FROZEN_NOW + timedelta(days=2, hours=13)
        with _patch_now():
            result = format_resets_in_human(future)
        assert result is not None
        assert "2d" in result
        assert "13h" in result
        # Should only have two units, no minutes
        assert "m" not in result

    def test_forty_five_minutes(self) -> None:
        future = _FROZEN_NOW + timedelta(minutes=45)
        with _patch_now():
            result = format_resets_in_human(future)
        assert result == "45m"

    def test_past_or_expired(self) -> None:
        past = _FROZEN_NOW - timedelta(minutes=5)
        with _patch_now():
            result = format_resets_in_human(past)
        assert result == "< 1m"

    def test_less_than_one_minute(self) -> None:
        almost_now = _FROZEN_NOW + timedelta(seconds=30)
        with _patch_now():
            result = format_resets_in_human(almost_now)
        assert result == "< 1m"

    def test_exactly_one_hour(self) -> None:
        future = _FROZEN_NOW + timedelta(hours=1)
        with _patch_now():
            result = format_resets_in_human(future)
        assert result is not None
        assert "1h" in result


# ---------------------------------------------------------------------------
# format_json
# ---------------------------------------------------------------------------


def _make_status(
    resets_at: datetime | None = None,
    extras: dict | None = None,
) -> ProviderStatus:
    window = UsageWindow(
        name="Session (5h)",
        utilisation=42.0,
        resets_at=resets_at,
        status="normal",
        unit="percent",
    )
    return ProviderStatus(
        provider_name="claude",
        provider_display="Anthropic Claude",
        timestamp=datetime(2026, 4, 5, 10, 30, 0, tzinfo=timezone.utc),
        cached=False,
        cache_age_seconds=0,
        windows=[window],
        extras=extras or {},
    )


class TestFormatJson:
    """Tests for the JSON formatter."""

    def test_output_is_valid_json(self) -> None:
        status = _make_status()
        output = format_json([status], "0.1.0")
        parsed = json.loads(output)
        assert isinstance(parsed, dict)

    def test_schema_has_required_top_level_keys(self) -> None:
        status = _make_status()
        output = format_json([status], "0.1.0")
        parsed = json.loads(output)
        assert "timestamp" in parsed
        assert "version" in parsed
        assert "providers" in parsed
        assert isinstance(parsed["providers"], list)

    def test_version_matches_input(self) -> None:
        status = _make_status()
        output = format_json([status], "0.1.0")
        parsed = json.loads(output)
        assert parsed["version"] == "0.1.0"

    def test_provider_schema(self) -> None:
        status = _make_status(
            resets_at=datetime(2026, 4, 5, 15, 0, 0, tzinfo=timezone.utc)
        )
        output = format_json([status], "0.1.0")
        parsed = json.loads(output)
        provider = parsed["providers"][0]
        assert provider["provider_name"] == "claude"
        assert provider["provider_display"] == "Anthropic Claude"
        assert "timestamp" in provider
        assert provider["cached"] is False
        assert provider["cache_age_seconds"] == 0
        assert isinstance(provider["windows"], list)
        assert isinstance(provider["extras"], dict)
        assert isinstance(provider["errors"], list)

    def test_window_schema(self) -> None:
        resets_at = datetime(2026, 4, 5, 15, 0, 0, tzinfo=timezone.utc)
        status = _make_status(resets_at=resets_at)
        output = format_json([status], "0.1.0")
        parsed = json.loads(output)
        window = parsed["providers"][0]["windows"][0]
        assert window["name"] == "Session (5h)"
        assert window["utilisation"] == 42.0
        assert window["resets_at"] is not None
        assert window["status"] == "normal"
        assert window["unit"] == "percent"
        assert "raw_value" in window
        assert "raw_limit" in window

    def test_uses_utilisation_not_utilization(self) -> None:
        """Field name must be 'utilisation' (British spelling)."""
        status = _make_status()
        output = format_json([status], "0.1.0")
        assert "utilisation" in output
        assert '"utilization"' not in output

    def test_resets_in_human_present_when_resets_at_set(self) -> None:
        future = datetime.now(timezone.utc) + timedelta(hours=2, minutes=15)
        status = _make_status(resets_at=future)
        output = format_json([status], "0.1.0")
        parsed = json.loads(output)
        window = parsed["providers"][0]["windows"][0]
        assert window["resets_in_human"] is not None
        assert isinstance(window["resets_in_human"], str)

    def test_resets_in_human_null_when_resets_at_none(self) -> None:
        status = _make_status(resets_at=None)
        output = format_json([status], "0.1.0")
        parsed = json.loads(output)
        window = parsed["providers"][0]["windows"][0]
        assert window["resets_at"] is None
        assert window["resets_in_human"] is None

    def test_no_secrets_in_output(self) -> None:
        """Ensure SecretStr values and known secret patterns are absent."""
        status = _make_status(extras={
            "api_key": SecretStr("sk-ant-secret-key-12345"),
            "token": SecretStr("Bearer super-secret-token"),
            "safe_key": "this-is-fine",
        })
        output = format_json([status], "0.1.0")
        assert "sk-ant" not in output
        assert "Bearer" not in output
        assert "super-secret" not in output
        # The safe key should still not leak secret values
        parsed = json.loads(output)
        extras = parsed["providers"][0]["extras"]
        assert "api_key" not in extras
        assert "token" not in extras
        assert extras.get("safe_key") == "this-is-fine"

    def test_multiple_providers(self) -> None:
        s1 = _make_status()
        s2 = ProviderStatus(
            provider_name="openai",
            provider_display="OpenAI",
            timestamp=datetime(2026, 4, 5, 10, 30, 0, tzinfo=timezone.utc),
            cached=True,
            cache_age_seconds=120,
            windows=[],
        )
        output = format_json([s1, s2], "0.1.0")
        parsed = json.loads(output)
        assert len(parsed["providers"]) == 2
        assert parsed["providers"][0]["provider_name"] == "claude"
        assert parsed["providers"][1]["provider_name"] == "openai"

    def test_empty_providers_list(self) -> None:
        output = format_json([], "0.1.0")
        parsed = json.loads(output)
        assert parsed["providers"] == []
