"""Rich Live TUI monitor for clawmeter.

Provides a persistent auto-refreshing terminal dashboard using Rich Live.
See SPEC.md Section 4.2.5 for the full monitor specification.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
import termios
import tty
from datetime import datetime, timezone
from typing import Any, Callable

from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from clawmeter.formatters.json_fmt import format_json, format_resets_in_human
from clawmeter.models import ProviderStatus, UsageWindow

# Shared status-to-colour mapping (matches table_fmt.py).
STATUS_COLOURS: dict[str, str] = {
    "normal": "green",
    "warning": "yellow",
    "critical": "red",
    "exceeded": "magenta",
}

# Sparkline block characters (8 levels, low to high).
_SPARK_CHARS = "▁▂▃▄▅▆▇█"

# Minimum data points for a meaningful sparkline (D-046).
_SPARK_MIN_POINTS = 3

# Full-width progress bar for normal mode.
_BAR_WIDTH = 20

# Narrow bar for compact mode (D-045).
_COMPACT_BAR_WIDTH = 10


# ======================================================================
# Sparkline renderer (D-046)
# ======================================================================


def render_sparkline(values: list[float]) -> str:
    """Render a list of numeric values as a Unicode sparkline string.

    Uses ``▁▂▃▄▅▆▇█`` mapped linearly across the min-max range.
    Returns an empty string if fewer than *_SPARK_MIN_POINTS* values.
    """
    if len(values) < _SPARK_MIN_POINTS:
        return ""

    lo = min(values)
    hi = max(values)
    span = hi - lo

    chars: list[str] = []
    for v in values:
        if span == 0:
            idx = 4  # mid-level when all values are identical
        else:
            idx = int((v - lo) / span * 7)
            idx = max(0, min(7, idx))
        chars.append(_SPARK_CHARS[idx])

    return "".join(chars)


# ======================================================================
# Health indicator (D-050)
# ======================================================================


def _health_dot(cache_age_seconds: int, poll_interval: int, has_errors: bool) -> Text:
    """Return a coloured health dot based on data staleness.

    Green  ``●`` = data age <= 1x poll_interval (healthy)
    Yellow ``●`` = data age <= 3x poll_interval (stale)
    Red    ``●`` = data age > 3x poll_interval OR errors
    """
    if has_errors or cache_age_seconds > 3 * poll_interval:
        return Text("●", style="red")
    if cache_age_seconds > poll_interval:
        return Text("●", style="yellow")
    return Text("●", style="green")


# ======================================================================
# Progress bar builder
# ======================================================================


def _build_bar(utilisation: float, status: str, width: int = _BAR_WIDTH) -> Text:
    """Build a Unicode progress bar with status colouring."""
    filled = max(0, min(width, round(utilisation / 100 * width)))
    empty = width - filled
    bar_str = "\u2588" * filled + "\u2591" * empty
    style = STATUS_COLOURS.get(status, "white")
    return Text(bar_str, style=style)


# ======================================================================
# Compact line renderer (D-045)
# ======================================================================


def format_compact_line(
    status: ProviderStatus,
    poll_interval: int = 600,
) -> Text:
    """Render a single provider as one compact text line.

    Format: ``● <name>  <bar> <pct>%  resets <time>``
    """
    dot = _health_dot(
        status.cache_age_seconds, poll_interval, bool(status.errors)
    )

    line = Text()
    line.append_text(dot)
    line.append(f" {status.provider_display:<16}", style="bold")

    # Show the first (primary) window
    if status.windows:
        w = status.windows[0]
        val = w.raw_value or 0.0
        if w.unit == "usd":
            bar = Text(" " * _COMPACT_BAR_WIDTH)
            val_str = f" ${val:,.2f}"
            line.append_text(bar)
            line.append(val_str, style=STATUS_COLOURS.get(w.status, "white"))
        elif w.unit == "credits":
            bar = Text(" " * _COMPACT_BAR_WIDTH)
            line.append_text(bar)
            line.append(f" ${val:,.2f}", style=STATUS_COLOURS.get(w.status, "white"))
        elif w.unit == "count":
            bar = Text(" " * _COMPACT_BAR_WIDTH)
            line.append_text(bar)
            line.append(f" {int(val)}", style=STATUS_COLOURS.get(w.status, "white"))
        elif w.unit == "mb":
            bar = Text(" " * _COMPACT_BAR_WIDTH)
            line.append_text(bar)
            line.append(f" {val:,.0f} MB", style=STATUS_COLOURS.get(w.status, "white"))
        else:
            bar = _build_bar(w.utilisation, w.status, width=_COMPACT_BAR_WIDTH)
            pct = f" {w.utilisation:5.1f}%"
            line.append_text(bar)
            line.append(pct, style=STATUS_COLOURS.get(w.status, "white"))
        human = format_resets_in_human(w.resets_at)
        if human:
            line.append(f"  resets {human}", style="dim")
    elif status.errors:
        line.append("  error", style="red")
    else:
        line.append("  no data", style="dim")

    return line


# ======================================================================
# Full panel renderer
# ======================================================================


def _build_provider_panel(
    status: ProviderStatus,
    poll_interval: int = 600,
    sparklines: dict[str, list[float]] | None = None,
    show_sparkline: bool = True,
) -> Panel:
    """Build a Rich Panel for a single provider with all its windows."""
    has_errors = bool(status.errors)
    dot = _health_dot(status.cache_age_seconds, poll_interval, has_errors)

    # Title with health dot
    title = Text()
    title.append_text(dot)
    title.append(f" {status.provider_display}", style="bold")

    # Data age
    age = status.cache_age_seconds
    if age < 60:
        age_str = f"{age}s ago"
    else:
        age_str = f"{age // 60}m ago"

    table = Table(
        show_header=False,
        show_edge=False,
        box=None,
        pad_edge=False,
        expand=True,
    )
    table.add_column("Name", min_width=18, no_wrap=True)
    table.add_column("Bar", min_width=_BAR_WIDTH, no_wrap=True)
    table.add_column("Pct", min_width=7, no_wrap=True)
    table.add_column("Reset", no_wrap=True)

    for window in status.windows:
        name = Text(f"  {window.name}")
        style = STATUS_COLOURS.get(window.status, "white")
        val = window.raw_value or 0.0
        if window.unit == "usd":
            bar = Text(" " * _BAR_WIDTH)
            pct = Text(f"${val:,.2f}", style=style)
        elif window.unit == "credits":
            bar = Text(" " * _BAR_WIDTH)
            pct = Text(f"${val:,.2f}", style=style)
        elif window.unit == "count":
            bar = Text(" " * _BAR_WIDTH)
            pct = Text(f"{int(val):>5}", style=style)
        elif window.unit == "mb":
            bar = Text(" " * _BAR_WIDTH)
            pct = Text(f"{val:,.0f} MB", style=style)
        else:
            bar = _build_bar(window.utilisation, window.status)
            pct = Text(f"{window.utilisation:5.1f}%", style=style)

        reset_parts = Text()
        human = format_resets_in_human(window.resets_at)
        if human:
            reset_parts.append(f"resets {human}", style="dim")

        # Sparkline suffix
        if show_sparkline and sparklines:
            key = f"{status.provider_name}:{window.name}"
            data = sparklines.get(key, [])
            spark = render_sparkline(data)
            if spark:
                reset_parts.append("  ")
                reset_parts.append(spark, style=style)

        table.add_row(name, bar, pct, reset_parts)

    # Show errors if any
    for err in status.errors:
        table.add_row(
            Text("  error", style="red"),
            Text(""),
            Text(""),
            Text(err, style="red dim"),
        )

    return Panel(
        table,
        title=title,
        subtitle=Text(age_str, style="dim"),
        subtitle_align="right",
        border_style="dim",
        expand=True,
    )


# ======================================================================
# Help overlay (D-047)
# ======================================================================


_HELP_TEXT = """\
[bold]Key Bindings[/bold]

  [bold]r[/bold]     Force refresh all providers
  [bold]1-9[/bold]   Force refresh provider by index
  [bold]q[/bold]     Quit
  [bold]j[/bold]     Dump current state as JSON to file
  [bold]?[/bold]     Show/dismiss this help

[dim]Press any key to dismiss[/dim]"""


def _build_help_panel() -> Panel:
    """Build the help overlay panel (D-047)."""
    return Panel(
        _HELP_TEXT,
        title="[bold]Help[/bold]",
        border_style="bright_blue",
        expand=False,
        width=50,
        padding=(1, 3),
    )


# ======================================================================
# Header bar
# ======================================================================


def _build_header(
    daemon_running: bool,
    last_poll_str: str | None,
    mode: str,
    footer_msg: str | None = None,
) -> Text:
    """Build the top header line with time, daemon status, and mode."""
    now = datetime.now(timezone.utc).astimezone()
    header = Text()
    header.append("LLM Monitor", style="bold")
    header.append(f"  {now.strftime('%d %b %Y, %H:%M:%S %Z')}", style="dim")
    header.append("  │  ", style="dim")

    if daemon_running:
        header.append("●", style="green")
        header.append(" daemon", style="dim")
        if last_poll_str:
            header.append(f" (last poll {last_poll_str})", style="dim")
    else:
        header.append("○", style="dim")
        header.append(f" standalone", style="dim")

    if mode == "compact":
        header.append("  │  ", style="dim")
        header.append("compact", style="dim")

    if footer_msg:
        header.append("\n")
        header.append(footer_msg, style="bright_cyan")

    return header


# ======================================================================
# Full layout builder
# ======================================================================


def build_display(
    statuses: list[ProviderStatus],
    *,
    compact: bool = False,
    daemon_running: bool = False,
    last_poll_str: str | None = None,
    poll_interval: int = 600,
    sparklines: dict[str, list[float]] | None = None,
    show_sparkline: bool = True,
    show_help: bool = False,
    footer_msg: str | None = None,
) -> Group:
    """Build the complete TUI display as a Rich renderable.

    Parameters
    ----------
    statuses:
        Current provider statuses to display.
    compact:
        When True, render one line per provider (D-045).
    daemon_running:
        Whether the daemon is detected as running.
    last_poll_str:
        Human-readable string of last daemon poll time.
    poll_interval:
        Global poll interval in seconds (for health indicators).
    sparklines:
        Dict mapping ``"provider:window_name"`` to lists of hourly
        utilisation values for sparkline rendering.
    show_sparkline:
        Whether to render sparklines (from config).
    show_help:
        When True, show the help overlay instead of provider data.
    footer_msg:
        Optional transient message to show in the header (e.g., JSON dump path).

    Returns
    -------
    Group
        A Rich Group renderable suitable for ``Live.update()``.
    """
    mode = "compact" if compact else "normal"
    header = _build_header(daemon_running, last_poll_str, mode, footer_msg)

    if show_help:
        return Group(header, Text(""), _build_help_panel())

    if not statuses:
        return Group(
            header,
            Text(""),
            Text("  No provider data available.", style="dim"),
        )

    if compact:
        lines = [header, Text("")]
        for status in statuses:
            lines.append(format_compact_line(status, poll_interval))
        return Group(*lines)

    # Full panel mode
    panels = [header, Text("")]
    for status in statuses:
        panels.append(_build_provider_panel(
            status,
            poll_interval=poll_interval,
            sparklines=sparklines,
            show_sparkline=show_sparkline,
        ))

    return Group(*panels)


# ======================================================================
# Keyboard input (non-blocking, raw terminal)
# ======================================================================


def _read_key(fd: int) -> str | None:
    """Non-blocking read of a single key from the terminal.

    Returns None if no key is available.
    """
    import select

    r, _, _ = select.select([fd], [], [], 0)
    if r:
        return os.read(fd, 1).decode("utf-8", errors="ignore")
    return None


# ======================================================================
# MonitorRunner — main loop
# ======================================================================


class MonitorRunner:
    """Manages the Rich Live monitor loop.

    Handles refresh cycles, key input, signal handling, and terminal
    state restoration.
    """

    def __init__(
        self,
        *,
        config: dict,
        provider_filter: list[str] | None = None,
        compact: bool = False,
        interval: int = 30,
        colour: bool = True,
        fetch_fn: Callable[..., Any] | None = None,
    ):
        self.config = config
        self.provider_filter = provider_filter
        self.compact = compact
        self.interval = interval
        self.colour = colour
        self.fetch_fn = fetch_fn

        self.poll_interval = config.get("general", {}).get("poll_interval", 600)
        self.show_sparkline = config.get("monitor", {}).get("show_sparkline", True)
        if config.get("monitor", {}).get("compact", False):
            self.compact = True

        self.statuses: list[ProviderStatus] = []
        self.sparklines: dict[str, list[float]] = {}
        self.daemon_running = False
        self.last_poll_str: str | None = None
        self.show_help = False
        self.footer_msg: str | None = None
        self._footer_clear_time: float | None = None
        self._force_refresh = False
        self._running = True
        self._old_termios: list | None = None

    def _set_footer(self, msg: str, duration: float = 3.0) -> None:
        """Set a transient footer message that auto-clears."""
        import time

        self.footer_msg = msg
        self._footer_clear_time = time.monotonic() + duration

    def _check_footer_expiry(self) -> None:
        """Clear footer message if its display duration has elapsed."""
        import time

        if self._footer_clear_time and time.monotonic() >= self._footer_clear_time:
            self.footer_msg = None
            self._footer_clear_time = None

    def _fetch_sparkline_data(self) -> dict[str, list[float]]:
        """Load sparkline data from the history database."""
        from clawmeter.history import HistoryStore

        result: dict[str, list[float]] = {}
        try:
            store = HistoryStore()
            store.open()
            try:
                now = datetime.now(timezone.utc)
                from_dt = now.replace(hour=now.hour, minute=0, second=0, microsecond=0)
                from_dt = from_dt.replace(
                    hour=0, minute=0, second=0
                ) if (now - from_dt).total_seconds() < 0 else from_dt

                # Get last 24h of hourly data
                from datetime import timedelta

                from_dt = now - timedelta(hours=24)
                rows = store.aggregate_samples(
                    granularity="hourly",
                    from_dt=from_dt,
                    to_dt=now,
                )

                for row in rows:
                    key = f"{row['provider']}:{row['window_name']}"
                    result.setdefault(key, []).append(row["utilisation"])
            finally:
                store.close()
        except Exception:
            pass  # Sparklines are best-effort

        return result

    def _detect_daemon(self) -> None:
        """Check whether the daemon is running and set state accordingly."""
        from clawmeter.daemon import is_daemon_running

        running, _ = is_daemon_running(self.config)
        self.daemon_running = running

        if running:
            from clawmeter.history import HistoryStore

            try:
                store = HistoryStore()
                store.open()
                try:
                    last_poll = store.get_last_poll_time()
                    if last_poll:
                        age = int(
                            (datetime.now(timezone.utc) - last_poll).total_seconds()
                        )
                        self.last_poll_str = (
                            f"{age}s ago" if age < 60 else f"{age // 60}m ago"
                        )
                    else:
                        self.last_poll_str = None
                finally:
                    store.close()
            except Exception:
                self.last_poll_str = None

    def _refresh_data(self) -> None:
        """Fetch fresh data from daemon DB or providers directly."""
        self._detect_daemon()

        if self.daemon_running:
            from clawmeter.history import HistoryStore

            store = HistoryStore()
            store.open()
            try:
                self.statuses = store.get_latest_statuses(self.provider_filter)
            finally:
                store.close()
        elif self.fetch_fn is not None:
            self.statuses = self.fetch_fn()

        # Refresh sparkline data
        if self.show_sparkline:
            self.sparklines = self._fetch_sparkline_data()

    def _handle_key(self, key: str) -> None:
        """Process a single keypress."""
        if key == "q":
            self._running = False
        elif key == "?":
            self.show_help = not self.show_help
        elif key == "r":
            self._force_refresh = True
        elif key == "j":
            self._dump_json()
        elif key.isdigit() and key != "0":
            idx = int(key) - 1
            # Force refresh — the full refresh will re-fetch all providers;
            # provider-specific refresh would require per-provider fetch
            # which is complex. For now, 1-9 triggers a full refresh.
            if idx < len(self.statuses):
                self._force_refresh = True
        elif self.show_help:
            # Any key dismisses help
            self.show_help = False

    def _dump_json(self) -> None:
        """Dump current state as JSON to a file in CWD (D-048)."""
        import clawmeter

        now = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"clawmeter-{now}.json"
        try:
            content = format_json(self.statuses, version=clawmeter.__version__)
            with open(filename, "w") as f:
                f.write(content)
                f.write("\n")
            self._set_footer(f"Saved to {filename}")
        except OSError as exc:
            self._set_footer(f"Error: {exc}")

    def _build_renderable(self) -> Group:
        """Build the current display state."""
        return build_display(
            self.statuses,
            compact=self.compact,
            daemon_running=self.daemon_running,
            last_poll_str=self.last_poll_str,
            poll_interval=self.poll_interval,
            sparklines=self.sparklines,
            show_sparkline=self.show_sparkline,
            show_help=self.show_help,
            footer_msg=self.footer_msg,
        )

    def _install_signal_handlers(self) -> None:
        """Install SIGUSR1 (force refresh) and SIGHUP (reload config) handlers."""
        def _on_usr1(signum: int, frame: Any) -> None:
            self._force_refresh = True

        def _on_hup(signum: int, frame: Any) -> None:
            from clawmeter.config import load_config

            try:
                self.config = load_config()
                self.poll_interval = self.config.get("general", {}).get(
                    "poll_interval", 600
                )
                self.show_sparkline = self.config.get("monitor", {}).get(
                    "show_sparkline", True
                )
                self._set_footer("Config reloaded")
            except Exception as exc:
                self._set_footer(f"Config reload failed: {exc}")

        signal.signal(signal.SIGUSR1, _on_usr1)
        signal.signal(signal.SIGHUP, _on_hup)

    def run(self) -> None:
        """Run the monitor loop until the user quits.

        Manages terminal raw mode for key input, Rich Live display,
        and clean terminal restoration via atexit.
        """
        import atexit
        import time

        console = Console(force_terminal=self.colour)
        fd = sys.stdin.fileno()

        # Save terminal state and enter raw mode for key reading
        self._old_termios = termios.tcgetattr(fd)

        def _restore_terminal() -> None:
            if self._old_termios is not None:
                termios.tcsetattr(fd, termios.TCSADRAIN, self._old_termios)

        atexit.register(_restore_terminal)

        self._install_signal_handlers()

        # Initial data fetch
        self._refresh_data()

        try:
            tty.setcbreak(fd)  # cbreak mode: single-char input, no echo

            with Live(
                self._build_renderable(),
                console=console,
                screen=True,
                refresh_per_second=2,
            ) as live:
                last_refresh = time.monotonic()

                while self._running:
                    # Check for key input
                    key = _read_key(fd)
                    if key:
                        self._handle_key(key)
                        if not self._running:
                            break
                        live.update(self._build_renderable())

                    # Check for forced or scheduled refresh
                    elapsed = time.monotonic() - last_refresh
                    if self._force_refresh or elapsed >= self.interval:
                        self._force_refresh = False
                        self._refresh_data()
                        last_refresh = time.monotonic()
                        live.update(self._build_renderable())

                    # Clear expired footer messages
                    self._check_footer_expiry()
                    if self.footer_msg is None and self._footer_clear_time is None:
                        pass  # No update needed just for footer clear
                    elif self.footer_msg is None:
                        live.update(self._build_renderable())

                    # Small sleep to avoid busy-waiting while staying responsive
                    time.sleep(0.05)

        except KeyboardInterrupt:
            pass  # Clean exit on Ctrl+C
        finally:
            _restore_terminal()
            atexit.unregister(_restore_terminal)
