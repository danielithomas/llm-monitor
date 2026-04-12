"""Tests for the Rich table formatter."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from clawmeter.formatters.table_fmt import format_table
from clawmeter.models import ProviderStatus, UsageWindow


def _make_status(
    utilisation: float = 42.0,
    status: str = "normal",
    window_name: str = "Session (5h)",
    display: str = "Anthropic Claude",
    resets_at: datetime | None = None,
) -> ProviderStatus:
    if resets_at is None:
        resets_at = datetime.now(timezone.utc) + timedelta(hours=2, minutes=15)
    window = UsageWindow(
        name=window_name,
        utilisation=utilisation,
        resets_at=resets_at,
        status=status,
        unit="percent",
    )
    return ProviderStatus(
        provider_name="claude",
        provider_display=display,
        timestamp=datetime(2026, 4, 5, 10, 30, 0, tzinfo=timezone.utc),
        cached=False,
        cache_age_seconds=0,
        windows=[window],
    )


class TestFormatTableColour:
    """Tests for colour / ANSI behaviour."""

    def test_colour_true_contains_ansi_codes(self) -> None:
        status = _make_status()
        output = format_table([status], colour=True)
        assert "\x1b[" in output

    def test_colour_false_no_ansi_codes(self) -> None:
        status = _make_status()
        output = format_table([status], colour=False)
        assert "\x1b[" not in output


class TestFormatTableContent:
    """Tests for content in the rendered table."""

    def test_contains_provider_display_name(self) -> None:
        status = _make_status(display="Anthropic Claude")
        output = format_table([status], colour=False)
        assert "Anthropic Claude" in output

    def test_contains_window_name(self) -> None:
        status = _make_status(window_name="Session (5h)")
        output = format_table([status], colour=False)
        assert "Session (5h)" in output

    def test_contains_percentage(self) -> None:
        status = _make_status(utilisation=42.0)
        output = format_table([status], colour=False)
        assert "42%" in output

    def test_contains_reset_time(self) -> None:
        future = datetime.now(timezone.utc) + timedelta(hours=2, minutes=15)
        status = _make_status(resets_at=future)
        output = format_table([status], colour=False)
        assert "resets in" in output

    def test_multiple_providers(self) -> None:
        s1 = _make_status(display="Anthropic Claude")
        s2 = ProviderStatus(
            provider_name="openai",
            provider_display="OpenAI",
            timestamp=datetime(2026, 4, 5, 10, 30, 0, tzinfo=timezone.utc),
            cached=True,
            cache_age_seconds=180,
            windows=[
                UsageWindow(
                    name="Rate Limit",
                    utilisation=22.0,
                    resets_at=datetime.now(timezone.utc) + timedelta(hours=18),
                    status="normal",
                    unit="percent",
                )
            ],
        )
        output = format_table([s1, s2], colour=False)
        assert "Anthropic Claude" in output
        assert "OpenAI" in output
        assert "Session (5h)" in output
        assert "Rate Limit" in output

    def test_high_utilisation_percentage(self) -> None:
        status = _make_status(utilisation=95.0, status="critical")
        output = format_table([status], colour=False)
        assert "95%" in output

    def test_exceeded_utilisation(self) -> None:
        status = _make_status(utilisation=110.0, status="exceeded")
        output = format_table([status], colour=False)
        assert "110%" in output

    def test_empty_providers_produces_output(self) -> None:
        output = format_table([], colour=False)
        assert "LLM Monitor" in output

    def test_header_contains_clawmeter(self) -> None:
        status = _make_status()
        output = format_table([status], colour=False)
        assert "LLM Monitor" in output

    def test_usd_window_shows_dollar_value(self) -> None:
        window = UsageWindow(
            name="Spend (MTD)",
            utilisation=0.0,
            resets_at=None,
            status="normal",
            unit="usd",
            raw_value=1450.00,
        )
        status = ProviderStatus(
            provider_name="grok",
            provider_display="xAI Grok",
            timestamp=datetime(2026, 4, 5, 10, 30, 0, tzinfo=timezone.utc),
            cached=False,
            cache_age_seconds=0,
            windows=[window],
        )
        output = format_table([status], colour=False)
        assert "$1,450.00" in output
        assert "0%" not in output

    def test_usd_window_no_progress_bar(self) -> None:
        window = UsageWindow(
            name="Prepaid Balance",
            utilisation=0.0,
            resets_at=None,
            status="normal",
            unit="usd",
            raw_value=75.00,
        )
        status = ProviderStatus(
            provider_name="grok",
            provider_display="xAI Grok",
            timestamp=datetime(2026, 4, 5, 10, 30, 0, tzinfo=timezone.utc),
            cached=False,
            cache_age_seconds=0,
            windows=[window],
        )
        output = format_table([status], colour=False)
        assert "$75.00" in output
        # Should not contain filled bar characters
        assert "\u2588" not in output
        assert "#" not in output

    def test_mixed_usd_and_percent_windows(self) -> None:
        windows = [
            UsageWindow(
                name="Spend (MTD)",
                utilisation=0.0,
                resets_at=None,
                status="normal",
                unit="usd",
                raw_value=1450.00,
            ),
            UsageWindow(
                name="Spend vs Limit",
                utilisation=29.0,
                resets_at=None,
                status="normal",
                unit="percent",
                raw_value=1450.00,
                raw_limit=5000.00,
            ),
        ]
        status = ProviderStatus(
            provider_name="grok",
            provider_display="xAI Grok",
            timestamp=datetime(2026, 4, 5, 10, 30, 0, tzinfo=timezone.utc),
            cached=False,
            cache_age_seconds=0,
            windows=windows,
        )
        output = format_table([status], colour=False)
        assert "$1,450.00" in output
        assert "29%" in output

    def test_cached_provider_shows_cache_info(self) -> None:
        status = ProviderStatus(
            provider_name="claude",
            provider_display="Anthropic Claude",
            timestamp=datetime(2026, 4, 5, 10, 30, 0, tzinfo=timezone.utc),
            cached=True,
            cache_age_seconds=180,
            windows=[],
        )
        output = format_table([status], colour=False)
        assert "cached 3m ago" in output
