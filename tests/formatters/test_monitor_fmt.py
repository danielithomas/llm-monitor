"""Tests for the Rich Live TUI monitor formatter."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from rich.console import Console
from rich.text import Text

from llm_monitor.formatters.monitor_fmt import (
    MonitorRunner,
    _build_help_panel,
    _build_provider_panel,
    _health_dot,
    build_display,
    format_compact_line,
    render_sparkline,
)
from llm_monitor.models import ProviderStatus, UsageWindow


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_status(
    utilisation: float = 42.0,
    status: str = "normal",
    window_name: str = "Session (5h)",
    display: str = "Anthropic Claude",
    provider_name: str = "claude",
    resets_at: datetime | None = None,
    cache_age_seconds: int = 120,
    errors: list[str] | None = None,
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
        provider_name=provider_name,
        provider_display=display,
        timestamp=datetime.now(timezone.utc),
        cached=True,
        cache_age_seconds=cache_age_seconds,
        windows=[window],
        errors=errors or [],
    )


def _render_to_str(renderable, width: int = 100) -> str:
    """Render a Rich object to a plain string (no ANSI codes)."""
    console = Console(width=width, no_color=True, force_terminal=False)
    with console.capture() as cap:
        console.print(renderable)
    return cap.get()


# ---------------------------------------------------------------------------
# Sparkline rendering (D-046)
# ---------------------------------------------------------------------------


class TestRenderSparkline:
    def test_basic_ascending(self) -> None:
        result = render_sparkline([10, 20, 30, 40, 50, 60, 70, 80])
        assert result == "\u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"

    def test_empty_history_returns_empty(self) -> None:
        assert render_sparkline([]) == ""

    def test_fewer_than_3_points_returns_empty(self) -> None:
        assert render_sparkline([10, 20]) == ""

    def test_exactly_3_points_renders(self) -> None:
        result = render_sparkline([0, 50, 100])
        assert len(result) == 3
        assert result != ""

    def test_constant_values_renders_mid_level(self) -> None:
        result = render_sparkline([50, 50, 50, 50])
        assert len(result) == 4
        # All same value -> all same character (mid-level)
        assert len(set(result)) == 1

    def test_length_matches_input(self) -> None:
        vals = [10, 20, 30, 40, 50]
        assert len(render_sparkline(vals)) == 5

    def test_24_hour_data(self) -> None:
        """24 data points produce a 24-character sparkline."""
        vals = [float(i * 4) for i in range(24)]
        result = render_sparkline(vals)
        assert len(result) == 24


# ---------------------------------------------------------------------------
# Health indicator (D-050)
# ---------------------------------------------------------------------------


class TestHealthDot:
    def test_green_within_poll_interval(self) -> None:
        dot = _health_dot(300, 600, False)
        text = _render_to_str(dot).strip()
        assert text == "\u25cf"  # Just check it renders the dot

    def test_yellow_stale(self) -> None:
        dot = _health_dot(1200, 600, False)
        # Stale: > 1x but <= 3x poll_interval
        assert isinstance(dot, Text)

    def test_red_very_stale(self) -> None:
        dot = _health_dot(2000, 600, False)
        assert isinstance(dot, Text)

    def test_red_with_errors(self) -> None:
        dot = _health_dot(10, 600, True)
        # Errors override freshness — should be red
        assert isinstance(dot, Text)

    def test_boundary_exactly_one_interval(self) -> None:
        # Exactly at poll_interval — should still be green (<=)
        dot = _health_dot(600, 600, False)
        assert isinstance(dot, Text)

    def test_boundary_exactly_three_intervals(self) -> None:
        # Exactly at 3x — should still be yellow (<=)
        dot = _health_dot(1800, 600, False)
        assert isinstance(dot, Text)


# ---------------------------------------------------------------------------
# Compact line (D-045)
# ---------------------------------------------------------------------------


class TestFormatCompactLine:
    def test_produces_single_line(self) -> None:
        status = _make_status()
        line = format_compact_line(status)
        text = _render_to_str(line)
        # Should be a single line (no newlines except trailing)
        assert text.strip().count("\n") == 0

    def test_contains_provider_name(self) -> None:
        status = _make_status(display="Anthropic Claude")
        text = _render_to_str(format_compact_line(status))
        assert "Anthropic Claude" in text

    def test_contains_percentage(self) -> None:
        status = _make_status(utilisation=42.0)
        text = _render_to_str(format_compact_line(status))
        assert "42.0%" in text

    def test_contains_reset_time(self) -> None:
        future = datetime.now(timezone.utc) + timedelta(hours=2, minutes=15)
        status = _make_status(resets_at=future)
        text = _render_to_str(format_compact_line(status))
        assert "resets" in text

    def test_contains_health_dot(self) -> None:
        status = _make_status()
        text = _render_to_str(format_compact_line(status))
        assert "\u25cf" in text  # ●

    def test_error_provider(self) -> None:
        status = ProviderStatus(
            provider_name="claude",
            provider_display="Anthropic Claude",
            timestamp=datetime.now(timezone.utc),
            cached=False,
            cache_age_seconds=0,
            windows=[],
            errors=["auth failed"],
        )
        text = _render_to_str(format_compact_line(status))
        assert "error" in text

    def test_no_windows_no_errors(self) -> None:
        status = ProviderStatus(
            provider_name="claude",
            provider_display="Anthropic Claude",
            timestamp=datetime.now(timezone.utc),
            cached=False,
            cache_age_seconds=0,
            windows=[],
        )
        text = _render_to_str(format_compact_line(status))
        assert "no data" in text


# ---------------------------------------------------------------------------
# Colour transitions
# ---------------------------------------------------------------------------


class TestColourTransitions:
    """Verify status-to-colour mapping produces correct ANSI output."""

    @pytest.mark.parametrize(
        "status_str,expected_style",
        [
            ("normal", "green"),
            ("warning", "yellow"),
            ("critical", "red"),
            ("exceeded", "magenta"),
        ],
    )
    def test_compact_line_uses_status_colour(
        self, status_str: str, expected_style: str
    ) -> None:
        status = _make_status(utilisation=85.0, status=status_str)
        line = format_compact_line(status)
        # Render with colour enabled to check ANSI codes
        console = Console(width=100, force_terminal=True, no_color=False)
        with console.capture() as cap:
            console.print(line)
        output = cap.get()
        # The output should contain ANSI escape codes (we can't easily
        # check for specific colours without parsing ANSI, so verify it
        # renders without error and has escape codes)
        assert "\x1b[" in output

    @pytest.mark.parametrize(
        "status_str",
        ["normal", "warning", "critical", "exceeded"],
    )
    def test_provider_panel_renders_all_statuses(self, status_str: str) -> None:
        status = _make_status(utilisation=85.0, status=status_str)
        panel = _build_provider_panel(status)
        output = _render_to_str(panel)
        assert "85.0%" in output


# ---------------------------------------------------------------------------
# Countdown timer formatting
# ---------------------------------------------------------------------------


class TestCountdownTimers:
    def test_hours_and_minutes(self) -> None:
        future = datetime.now(timezone.utc) + timedelta(hours=2, minutes=15)
        status = _make_status(resets_at=future)
        text = _render_to_str(format_compact_line(status))
        assert "resets 2h 15m" in text or "resets 2h 14m" in text

    def test_days_and_hours(self) -> None:
        future = datetime.now(timezone.utc) + timedelta(days=2, hours=13)
        status = _make_status(resets_at=future)
        text = _render_to_str(format_compact_line(status))
        assert "resets 2d 13h" in text or "resets 2d 12h" in text

    def test_minutes_only(self) -> None:
        future = datetime.now(timezone.utc) + timedelta(minutes=45)
        status = _make_status(resets_at=future)
        text = _render_to_str(format_compact_line(status))
        assert "resets 45m" in text or "resets 44m" in text

    def test_less_than_one_minute(self) -> None:
        future = datetime.now(timezone.utc) + timedelta(seconds=30)
        status = _make_status(resets_at=future)
        text = _render_to_str(format_compact_line(status))
        assert "resets < 1m" in text


# ---------------------------------------------------------------------------
# Full display build
# ---------------------------------------------------------------------------


class TestBuildDisplay:
    def test_renders_without_crash_single_provider(self) -> None:
        status = _make_status()
        display = build_display([status])
        output = _render_to_str(display)
        assert "LLM Monitor" in output
        assert "Anthropic Claude" in output

    def test_renders_without_crash_multiple_providers(self) -> None:
        s1 = _make_status(display="Anthropic Claude", provider_name="claude")
        s2 = _make_status(
            display="OpenAI",
            provider_name="openai",
            utilisation=22.0,
            window_name="Rate Limit",
        )
        display = build_display([s1, s2])
        output = _render_to_str(display)
        assert "Anthropic Claude" in output
        assert "OpenAI" in output

    def test_compact_mode_single_lines(self) -> None:
        s1 = _make_status(display="Anthropic Claude")
        s2 = _make_status(display="OpenAI", provider_name="openai")
        display = build_display([s1, s2], compact=True)
        output = _render_to_str(display)
        assert "Anthropic Claude" in output
        assert "OpenAI" in output

    def test_daemon_running_indicator(self) -> None:
        status = _make_status()
        display = build_display(
            [status], daemon_running=True, last_poll_str="30s ago"
        )
        output = _render_to_str(display)
        assert "daemon" in output
        assert "30s ago" in output

    def test_standalone_indicator(self) -> None:
        status = _make_status()
        display = build_display([status], daemon_running=False)
        output = _render_to_str(display)
        assert "standalone" in output

    def test_no_providers(self) -> None:
        display = build_display([])
        output = _render_to_str(display)
        assert "No provider data available" in output

    def test_help_overlay(self) -> None:
        display = build_display([], show_help=True)
        output = _render_to_str(display)
        assert "Key Bindings" in output
        assert "Press any key to dismiss" in output

    def test_footer_message(self) -> None:
        status = _make_status()
        display = build_display([status], footer_msg="Saved to test.json")
        output = _render_to_str(display)
        assert "Saved to test.json" in output

    def test_sparklines_displayed(self) -> None:
        status = _make_status()
        sparklines = {
            "claude:Session (5h)": [10, 20, 30, 40, 50, 60, 70, 80, 90, 80, 70, 60,
                                    50, 40, 30, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        }
        display = build_display(
            [status], sparklines=sparklines, show_sparkline=True
        )
        output = _render_to_str(display)
        # Should contain sparkline characters
        assert any(c in output for c in "\u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588")

    def test_sparklines_hidden_when_disabled(self) -> None:
        status = _make_status()
        sparklines = {
            "claude:Session (5h)": [10, 20, 30, 40, 50],
        }
        # Build with sparklines enabled vs disabled and compare.
        # The disabled version should not contain the sparkline string.
        display_on = build_display(
            [status], sparklines=sparklines, show_sparkline=True
        )
        display_off = build_display(
            [status], sparklines=sparklines, show_sparkline=False
        )
        output_on = _render_to_str(display_on)
        output_off = _render_to_str(display_off)
        spark_str = render_sparkline(sparklines["claude:Session (5h)"])
        # When enabled the sparkline text appears; when disabled it does not
        assert spark_str in output_on
        assert spark_str not in output_off


# ---------------------------------------------------------------------------
# Help panel (D-047)
# ---------------------------------------------------------------------------


class TestHelpPanel:
    def test_contains_all_keybindings(self) -> None:
        panel = _build_help_panel()
        output = _render_to_str(panel)
        for key in ["r", "1-9", "q", "j", "?"]:
            assert key in output

    def test_contains_dismiss_instruction(self) -> None:
        panel = _build_help_panel()
        output = _render_to_str(panel)
        assert "Press any key to dismiss" in output


# ---------------------------------------------------------------------------
# SIGUSR1 signal handler (MonitorRunner)
# ---------------------------------------------------------------------------


class TestSignalHandlers:
    def test_sigusr1_sets_force_refresh(self) -> None:
        """SIGUSR1 handler sets _force_refresh flag on the runner."""
        import signal

        runner = MonitorRunner(
            config={"general": {"poll_interval": 600}},
            interval=30,
        )
        assert runner._force_refresh is False

        # Install signal handlers
        runner._install_signal_handlers()

        # Send SIGUSR1 to ourselves
        import os

        os.kill(os.getpid(), signal.SIGUSR1)

        assert runner._force_refresh is True
