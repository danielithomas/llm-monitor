"""Rich table formatter for llm-monitor."""

from __future__ import annotations

from datetime import datetime, timezone

from rich.console import Console
from rich.table import Table
from rich.text import Text

from llm_monitor.formatters.json_fmt import format_resets_in_human
from llm_monitor.models import ProviderStatus, UsageWindow

# Mapping from status string to Rich colour name.
_STATUS_COLOURS: dict[str, str] = {
    "normal": "green",
    "warning": "yellow",
    "critical": "red",
    "exceeded": "magenta",
}

_BAR_WIDTH = 20


def _build_bar(utilisation: float, status: str, colour: bool) -> Text:
    """Build a text-based progress bar for a usage window."""
    filled = max(0, min(_BAR_WIDTH, round(utilisation / 100 * _BAR_WIDTH)))
    empty = _BAR_WIDTH - filled

    bar_str = "\u2588" * filled + "\u2591" * empty

    if colour:
        style = _STATUS_COLOURS.get(status, "white")
        return Text(bar_str, style=style)
    # Plain mode: use ASCII characters
    bar_str = "#" * filled + "-" * empty
    return Text(bar_str)


def _format_cache_info(status: ProviderStatus) -> str:
    """Format the cache info string for a provider header."""
    if not status.cached:
        return "fresh"
    age = status.cache_age_seconds
    if age < 60:
        return f"cached {age}s ago"
    minutes = age // 60
    return f"cached {minutes}m ago"


def _format_value_and_reset(window: UsageWindow) -> str:
    """Format the value and reset-time suffix for a window row.

    For percentage windows, shows ``42%``.  For USD windows, shows
    ``$1,450.00`` from ``raw_value``.
    """
    if window.unit == "usd":
        val = window.raw_value or 0.0
        formatted = f"${val:,.2f}"
    else:
        formatted = f"{window.utilisation:.0f}%"
    human = format_resets_in_human(window.resets_at)
    if human is not None:
        return f"{formatted}    resets in {human}"
    return formatted


def format_table(
    statuses: list[ProviderStatus],
    colour: bool = True,
) -> str:
    """Render provider statuses as a Rich table string.

    Parameters
    ----------
    statuses:
        List of provider status objects to render.
    colour:
        When True, output includes ANSI colour codes and Unicode
        box-drawing characters.  When False, output is plain ASCII
        with no escape codes.  The caller is responsible for
        deciding this based on TTY detection, ``$NO_COLOR``, and
        ``$TERM``.

    Returns
    -------
    str
        The rendered table as a string.
    """
    now = datetime.now(timezone.utc).astimezone()
    header_time = now.strftime("%d %b %Y, %H:%M %Z")

    table = Table(
        show_header=False,
        show_edge=True,
        title=f"LLM Monitor{' ' * 30}{header_time}",
        title_style="bold" if colour else "",
        box=None if not colour else None,
        pad_edge=True,
        expand=True,
    )

    table.add_column("Name", min_width=20, no_wrap=True, overflow="ellipsis")
    table.add_column("Bar", min_width=20, no_wrap=True)
    table.add_column("Info", no_wrap=True)

    for idx, status in enumerate(statuses):
        cache_info = _format_cache_info(status)

        # Provider header row
        provider_label = status.provider_display
        if colour:
            header_text = Text(provider_label, style="bold")
        else:
            header_text = Text(provider_label)

        cache_text = Text(cache_info, style="dim" if colour else "")

        table.add_row(header_text, Text(""), cache_text)

        # Window rows
        for window in status.windows:
            name_text = Text(f"  {window.name}")
            if window.unit == "usd":
                bar = Text(" " * _BAR_WIDTH)
            else:
                bar = _build_bar(window.utilisation, window.status, colour)
            info = _format_value_and_reset(window)

            if colour:
                style = _STATUS_COLOURS.get(window.status, "white")
                info_text = Text(info, style=style)
            else:
                info_text = Text(info)

            table.add_row(name_text, bar, info_text)

        # Blank separator between providers (except last)
        if idx < len(statuses) - 1:
            table.add_row(Text(""), Text(""), Text(""))

    console = Console(
        force_terminal=colour,
        no_color=not colour,
        width=80,
    )
    with console.capture() as capture:
        console.print(table)

    return capture.get()
