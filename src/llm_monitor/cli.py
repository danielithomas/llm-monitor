"""Click CLI entry point for llm-monitor.

All data goes to stdout, all messages go to stderr.
See SPEC.md Section 4 for the full CLI specification.
"""

from __future__ import annotations

import asyncio
import os
import sys
from datetime import datetime, timedelta, timezone

import click

import llm_monitor
from llm_monitor.cache import ProviderCache
from llm_monitor.config import get_cache_dir, get_pid_file, get_state_file, load_config
from llm_monitor.core import determine_exit_code, fetch_all
from llm_monitor.daemon import (
    DaemonRunner,
    daemonise,
    is_daemon_running,
    read_state,
)
from llm_monitor.formatters.json_fmt import format_json
from llm_monitor.formatters.table_fmt import format_table
from llm_monitor.history import HistoryStore
from llm_monitor.providers import PROVIDERS, get_enabled_providers
from llm_monitor.security import is_container_mode


def _resolve_colour(no_colour: bool, colour: str | None) -> bool:
    """Determine whether colour output should be used.

    Precedence (highest to lowest):
    1. --no-colour flag (always disables)
    2. $NO_COLOR env var (if set to any value, disables)
    3. $LLM_MONITOR_NO_COLOR env var
    4. $TERM=dumb (disables)
    5. TTY detection (auto)
    6. --colour=always (force enable even when piped)
    """
    if no_colour:
        return False
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("LLM_MONITOR_NO_COLOR") is not None:
        return False
    if os.environ.get("TERM") == "dumb":
        return False
    if sys.stdout.isatty():
        return True
    if colour == "always":
        return True
    return False


def _open_history(config: dict, no_history: bool) -> HistoryStore | None:
    """Open a HistoryStore if history is enabled, otherwise return None."""
    if no_history:
        return None
    if not config.get("history", {}).get("enabled", True):
        return None
    store = HistoryStore()
    store.open()
    retention = config.get("history", {}).get("retention_days", 90)
    store.prune(retention)
    return store


def _format_size(size_bytes: int) -> str:
    """Format bytes as human-readable size."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


def _stdin_is_tty() -> bool:
    """Check if stdin is a TTY. Extracted for testability."""
    return sys.stdin.isatty()


def _load_config_or_exit(config_path: str | None = None) -> dict:
    """Load config or exit with error."""
    try:
        return load_config(config_path)
    except ValueError as exc:
        click.echo(
            f"Error: {exc}\n"
            "Fix: Check your config file syntax (TOML format).",
            err=True,
        )
        sys.exit(1)


# ======================================================================
# Main CLI group with default-to-status behaviour
# ======================================================================


class _MainGroup(click.Group):
    """Group that delegates unknown subcommands and bare flags to 'status'."""

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        # If no args or first arg is a flag (not a known subcommand),
        # inject 'status' so the main fetch path runs by default.
        if not args or (args[0].startswith("-") and args[0] not in ("-h", "--help", "-V", "--version")):
            args = ["status"] + list(args)
        elif args[0] not in self.commands and args[0] not in ("-h", "--help", "-V", "--version"):
            # Unknown positional — insert 'status' and let it handle the error
            args = ["status"] + list(args)
        return super().parse_args(ctx, args)


@click.group(cls=_MainGroup, context_settings={"help_option_names": ["-h", "--help"]},
             invoke_without_command=True)
@click.option("--version", "-V", is_flag=True, default=False, help="Print version and exit.")
@click.pass_context
def cli(ctx: click.Context, version: bool) -> None:
    """Monitor LLM service usage across providers."""
    if version:
        click.echo(f"llm-monitor {llm_monitor.__version__}")
        ctx.exit(0)


# ======================================================================
# status — default command (fetch and display)
# ======================================================================

@cli.command()
@click.option("--now", is_flag=True, default=False, help="Display table output.")
@click.option(
    "--provider", "-p", default=None,
    help="Comma-separated list of providers to query.",
)
@click.option("--fresh", "-f", is_flag=True, default=False, help="Bypass cache.")
@click.option("--verbose", "-v", is_flag=True, default=False, help="Verbose logging to stderr.")
@click.option("--quiet", "-q", is_flag=True, default=False, help="Suppress non-error stderr output.")
@click.option("--config", "-c", "config_path", default=None, help="Config file path override.")
@click.option("--clear-cache", is_flag=True, default=False, help="Delete cache and exit.")
@click.option("--list-providers", is_flag=True, default=False, help="List providers and exit.")
@click.option("--no-colour", is_flag=True, default=False, help="Disable colour output.")
@click.option(
    "--colour", default=None,
    help="Force colour output (use --colour=always).",
)
@click.option(
    "--no-history", is_flag=True, default=False,
    help="Disable history recording for this invocation.",
)
@click.option(
    "--report", is_flag=True, default=False,
    help="Display usage report from history. Alias for 'history report'.",
)
@click.option("--days", default=None, type=int, help="Report: number of days.")
@click.option("--from", "from_date", default=None, help="Report: start date (YYYY-MM-DD).")
@click.option("--to", "to_date", default=None, help="Report: end date (YYYY-MM-DD).")
@click.option(
    "--format", "output_format", default=None,
    type=click.Choice(["json", "table", "csv"]),
    help="Report: output format.",
)
@click.option(
    "--granularity", default=None,
    type=click.Choice(["raw", "hourly", "daily"]),
    help="Report: aggregation granularity.",
)
@click.option("--models", is_flag=True, default=False, help="Report: include per-model breakdown.")
@click.option(
    "--window", "window_filter", default=None,
    help="Report: filter to specific window name.",
)
def status(
    now: bool,
    provider: str | None,
    fresh: bool,
    verbose: bool,
    quiet: bool,
    config_path: str | None,
    clear_cache: bool,
    list_providers: bool,
    no_colour: bool,
    colour: str | None,
    no_history: bool,
    report: bool,
    days: int | None,
    from_date: str | None,
    to_date: str | None,
    output_format: str | None,
    granularity: str | None,
    models: bool,
    window_filter: str | None,
) -> None:
    """Fetch and display current LLM usage (default command)."""
    # Mutual exclusivity: --quiet and --verbose
    if quiet and verbose:
        click.echo(
            "Error: --quiet and --verbose are mutually exclusive.\n"
            "Fix: Use only one of --quiet or --verbose.",
            err=True,
        )
        sys.exit(1)

    # Load config
    try:
        config = load_config(config_path)
    except ValueError as exc:
        click.echo(
            f"Error: {exc}\n"
            "Fix: Check your config file syntax (TOML format).",
            err=True,
        )
        sys.exit(1)

    # If --report, delegate to report logic
    if report:
        _run_report(
            config=config,
            provider_filter=provider,
            days=days,
            from_date=from_date,
            to_date=to_date,
            output_format=output_format or "table",
            granularity=granularity or "daily",
            include_models=models,
            window_filter=window_filter,
            no_colour=no_colour,
            colour=colour,
        )
        return

    cache_dir = get_cache_dir()
    cache = ProviderCache(cache_dir)

    # --clear-cache
    if clear_cache:
        cache.clear_all()
        click.echo("Cache cleared.", err=True)
        return

    # --list-providers
    if list_providers:
        providers_cfg = config.get("providers", {})
        for name, cls in PROVIDERS.items():
            section = providers_cfg.get(name, {})
            enabled = section.get("enabled", True)
            enabled_str = "enabled" if enabled else "disabled"
            try:
                instance = cls(config)
                configured = instance.is_configured()
            except Exception:
                configured = False
            configured_str = "configured" if configured else "not configured"
            click.echo(f"{name}: {enabled_str}, {configured_str}")
        return

    # Daemon detection: if daemon is running and --fresh not set,
    # read from the history DB instead of fetching directly.
    if not fresh:
        running, daemon_pid = is_daemon_running(config)
        if running:
            store = HistoryStore()
            store.open()
            try:
                provider_filter_list = (
                    [p.strip() for p in provider.split(",")]
                    if provider else None
                )
                statuses = store.get_latest_statuses(provider_filter_list)
                last_poll = store.get_last_poll_time()
            finally:
                store.close()

            if statuses:
                # Report daemon mode to stderr
                if not quiet:
                    if last_poll:
                        ago = int(
                            (datetime.now(timezone.utc) - last_poll).total_seconds()
                        )
                        if ago < 60:
                            ago_str = f"{ago}s ago"
                        else:
                            ago_str = f"{ago // 60}m ago"
                    else:
                        ago_str = "unknown"
                    click.echo(
                        f"Reading from daemon (last poll {ago_str})", err=True
                    )

                use_colour = _resolve_colour(no_colour, colour)
                if now:
                    output = format_table(statuses, colour=use_colour)
                else:
                    output = format_json(
                        statuses, version=llm_monitor.__version__
                    )
                click.echo(output)

                exit_code = determine_exit_code(statuses)
                if exit_code != 0:
                    sys.exit(exit_code)
                return

    # Get enabled providers (filtered by --provider if given)
    if provider:
        requested = [p.strip() for p in provider.split(",")]
        provider_classes = []
        for name in requested:
            if name not in PROVIDERS:
                available = ", ".join(PROVIDERS.keys())
                click.echo(
                    f"Error: Unknown provider '{name}'.\n"
                    f"Available providers: {available}\n"
                    f"Fix: Use --provider with one of: {available}",
                    err=True,
                )
                sys.exit(1)
            provider_classes.append(PROVIDERS[name])
    else:
        provider_classes = get_enabled_providers(config)

    if not provider_classes:
        click.echo(
            "Error: No providers enabled or available.\n"
            "At least one provider must be enabled in the config file.\n"
            "Fix: Set 'enabled = true' under a [providers.<name>] section in "
            f"{get_cache_dir().parent / 'config.toml'}",
            err=True,
        )
        sys.exit(1)

    # Instantiate providers
    provider_instances = []
    for cls in provider_classes:
        try:
            instance = cls(config)
            provider_instances.append(instance)
        except Exception as exc:
            click.echo(
                f"Error: Failed to initialise provider '{cls.__name__}': {exc}",
                err=True,
            )
            sys.exit(1)

    # Fetch usage
    statuses = asyncio.run(fetch_all(provider_instances, cache, config, fresh=fresh))

    # Record to history
    history = _open_history(config, no_history)
    if history is not None:
        try:
            for s in statuses:
                history.record(s)
        finally:
            history.close()

    # Determine colour
    use_colour = _resolve_colour(no_colour, colour)

    # Format output
    if now:
        output = format_table(statuses, colour=use_colour)
    else:
        output = format_json(statuses, version=llm_monitor.__version__)

    click.echo(output)

    # Exit with appropriate code
    exit_code = determine_exit_code(statuses)
    if exit_code != 0:
        sys.exit(exit_code)


# ======================================================================
# history subcommand group
# ======================================================================

@cli.group()
def history() -> None:
    """History database commands (stats, purge, export, report)."""


@history.command(name="stats")
@click.option("--config", "-c", "config_path", default=None, help="Config file path override.")
def history_stats(config_path: str | None) -> None:
    """Display history database summary."""
    config = _load_config_or_exit(config_path)
    store = HistoryStore()
    store.open()
    try:
        info = store.stats()
        retention = config.get("history", {}).get("retention_days", 90)
        prune_count = store.prune_count(retention)

        providers_str = ", ".join(info["providers"]) if info["providers"] else "none"
        click.echo(f"History Database: {info['db_path']}")
        click.echo(f"  Size:       {_format_size(info['db_size'])}")
        click.echo(f"  Samples:    {info['sample_count']:,}")
        click.echo(f"  Models:     {info['model_count']:,}")
        click.echo(f"  Providers:  {providers_str}")
        click.echo(f"  Oldest:     {info['oldest'] or 'n/a'}")
        click.echo(f"  Newest:     {info['newest'] or 'n/a'}")
        click.echo(
            f"  Retention:  {retention} days "
            f"(next prune removes {prune_count:,} records)"
        )
    finally:
        store.close()


@history.command(name="purge")
@click.option("--confirm", is_flag=True, default=False, help="Skip interactive confirmation.")
@click.option("--config", "-c", "config_path", default=None, help="Config file path override.")
def history_purge(confirm: bool, config_path: str | None) -> None:
    """Permanently delete all history data."""
    _load_config_or_exit(config_path)
    store = HistoryStore()
    store.open()
    try:
        info = store.stats()

        if not confirm:
            # Check if stdin is a TTY
            if not _stdin_is_tty():
                click.echo(
                    "Error: history purge requires interactive confirmation.\n"
                    "Fix: Use --confirm to bypass: llm-monitor history purge --confirm",
                    err=True,
                )
                sys.exit(1)

            click.echo("", err=True)
            click.echo("WARNING: This will permanently delete all usage history.", err=True)
            click.echo(f"  Database: {info['db_path']}", err=True)
            click.echo(
                f"  Records:  {info['sample_count']:,} samples across "
                f"{len(info['providers'])} providers",
                err=True,
            )
            click.echo(f"  Oldest:   {(info['oldest'] or 'n/a')[:10]}", err=True)
            click.echo(f"  Size:     {_format_size(info['db_size'])}", err=True)
            click.echo("", err=True)
            click.echo("This action cannot be undone.", err=True)
            click.echo("", err=True)

            try:
                response = input("Type 'purge' to confirm: ")
            except EOFError:
                click.echo("Aborted. History was not modified.", err=True)
                return

            if response != "purge":
                click.echo("Aborted. History was not modified.", err=True)
                return

        store.purge()
        click.echo("History purged successfully.", err=True)
    finally:
        store.close()


@history.command(name="export")
@click.option(
    "--format", "export_format", required=True,
    type=click.Choice(["sql", "jsonl", "csv"]),
    help="Export format.",
)
@click.option("--config", "-c", "config_path", default=None, help="Config file path override.")
def history_export(export_format: str, config_path: str | None) -> None:
    """Export all history data for backup or migration."""
    _load_config_or_exit(config_path)
    store = HistoryStore()
    store.open()
    try:
        if export_format == "sql":
            output = store.export_sql()
        elif export_format == "jsonl":
            output = store.export_jsonl()
        elif export_format == "csv":
            output = store.export_csv()
        else:
            output = ""
        click.echo(output, nl=False)
    finally:
        store.close()


@history.command(name="report")
@click.option("--days", default=7, type=int, help="Number of days to report on.")
@click.option("--from", "from_date", default=None, help="Start date (YYYY-MM-DD).")
@click.option("--to", "to_date", default=None, help="End date (YYYY-MM-DD).")
@click.option(
    "--format", "output_format", default="table",
    type=click.Choice(["table", "json", "csv"]),
    help="Output format.",
)
@click.option(
    "--provider", "-p", default=None,
    help="Filter to specific provider(s).",
)
@click.option("--window", "window_filter", default=None, help="Filter to specific window name.")
@click.option(
    "--granularity", default="daily",
    type=click.Choice(["raw", "hourly", "daily"]),
    help="Aggregation granularity.",
)
@click.option("--models", is_flag=True, default=False, help="Include per-model breakdown.")
@click.option("--no-colour", is_flag=True, default=False, help="Disable colour output.")
@click.option(
    "--colour", default=None,
    help="Force colour output (use --colour=always).",
)
@click.option("--config", "-c", "config_path", default=None, help="Config file path override.")
def history_report(
    days: int,
    from_date: str | None,
    to_date: str | None,
    output_format: str,
    provider: str | None,
    window_filter: str | None,
    granularity: str,
    models: bool,
    no_colour: bool,
    colour: str | None,
    config_path: str | None,
) -> None:
    """Display usage report from history data."""
    config = _load_config_or_exit(config_path)
    _run_report(
        config=config,
        provider_filter=provider,
        days=days,
        from_date=from_date,
        to_date=to_date,
        output_format=output_format,
        granularity=granularity,
        include_models=models,
        window_filter=window_filter,
        no_colour=no_colour,
        colour=colour,
    )


# ======================================================================
# Report implementation
# ======================================================================


def _run_report(
    *,
    config: dict,
    provider_filter: str | None,
    days: int | None,
    from_date: str | None,
    to_date: str | None,
    output_format: str,
    granularity: str,
    include_models: bool,
    window_filter: str | None,
    no_colour: bool,
    colour: str | None,
) -> None:
    """Shared report logic for --report flag and history report subcommand."""
    store = HistoryStore()
    store.open()
    try:
        # Resolve date range
        now = datetime.now(timezone.utc)
        if to_date:
            to_dt = datetime.fromisoformat(to_date).replace(tzinfo=timezone.utc)
        else:
            to_dt = now

        if from_date:
            from_dt = datetime.fromisoformat(from_date).replace(tzinfo=timezone.utc)
        else:
            report_days = days if days is not None else 7
            from_dt = to_dt - timedelta(days=report_days)

        # Query
        samples = store.query_samples(
            provider=provider_filter,
            window=window_filter,
            from_dt=from_dt,
            to_dt=to_dt,
        )

        model_rows = []
        if include_models:
            model_rows = store.query_model_usage(
                provider=provider_filter,
                from_dt=from_dt,
                to_dt=to_dt,
            )

        # Aggregate
        aggregated = store.aggregate_samples(samples, granularity)
        model_aggregated = []
        if include_models:
            model_aggregated = store.aggregate_model_usage(model_rows, granularity)

        # Format output
        if output_format == "json":
            output = _format_report_json(
                aggregated, model_aggregated, from_dt, to_dt, granularity
            )
        elif output_format == "csv":
            output = _format_report_csv(aggregated, model_aggregated)
        else:
            use_colour = _resolve_colour(no_colour, colour)
            output = _format_report_table(
                aggregated, model_aggregated, from_dt, to_dt, use_colour
            )

        click.echo(output)
    finally:
        store.close()


def _format_report_json(
    samples: list[dict],
    model_usage: list[dict],
    from_dt: datetime,
    to_dt: datetime,
    granularity: str,
) -> str:
    """Format report as JSON."""
    import json
    report = {
        "report": {
            "from": from_dt.isoformat(),
            "to": to_dt.isoformat(),
            "granularity": granularity,
            "sample_count": len(samples),
        },
        "usage": samples,
    }
    if model_usage:
        report["model_usage"] = model_usage
    return json.dumps(report, indent=2)


def _format_report_csv(
    samples: list[dict],
    model_usage: list[dict],
) -> str:
    """Format report as CSV."""
    import csv
    import io

    output = io.StringIO()
    if samples:
        keys = list(samples[0].keys())
        writer = csv.DictWriter(output, fieldnames=keys)
        writer.writeheader()
        for row in samples:
            writer.writerow(row)

    if model_usage:
        output.write("\n")
        keys = list(model_usage[0].keys())
        writer = csv.DictWriter(output, fieldnames=keys)
        writer.writeheader()
        for row in model_usage:
            writer.writerow(row)

    return output.getvalue()


def _format_report_table(
    samples: list[dict],
    model_usage: list[dict],
    from_dt: datetime,
    to_dt: datetime,
    use_colour: bool,
) -> str:
    """Format report as a Rich table."""
    if not samples:
        return "No history data found for the specified period."

    from_str = from_dt.strftime("%d %b %Y")
    to_str = to_dt.strftime("%d %b %Y")

    lines: list[str] = []
    lines.append(f"LLM Usage Report                     {from_str} - {to_str}")
    lines.append("=" * 60)

    # Group by provider
    providers: dict[str, list[dict]] = {}
    for s in samples:
        providers.setdefault(s["provider"], []).append(s)

    for provider_name, provider_samples in providers.items():
        lines.append("")
        lines.append(f"  {provider_name}")
        lines.append("-" * 60)

        # Group by window
        windows: dict[str, list[dict]] = {}
        for s in provider_samples:
            windows.setdefault(s["window_name"], []).append(s)

        for window_name, window_samples in windows.items():
            utils = [
                s["utilisation"] for s in window_samples
                if s["utilisation"] is not None
            ]
            if utils:
                avg = sum(utils) / len(utils)
                peak = max(utils)
                exceeded = sum(
                    1 for s in window_samples
                    if s.get("status") == "exceeded"
                )
                lines.append(
                    f"  {window_name:<20} avg {avg:.0f}%   "
                    f"peak {peak:.0f}%   exceeded {exceeded}x"
                )

                # Sparkline
                sparkline = _make_sparkline(utils)
                lines.append(f"  {'':<20} {sparkline}")
            else:
                lines.append(f"  {window_name:<20} no data")

    # Model usage section
    if model_usage:
        lines.append("")
        lines.append("  Per-Model Breakdown")
        lines.append("-" * 60)
        for mu in model_usage:
            parts = [f"  {mu['model']:<24}"]
            if mu.get("total_tokens") is not None:
                parts.append(f"{mu['total_tokens']:,} tokens")
            if mu.get("cost") is not None:
                parts.append(f"${mu['cost']:.2f}")
            if mu.get("request_count") is not None:
                parts.append(f"{mu['request_count']} requests")
            lines.append("  ".join(parts))

    lines.append("")
    lines.append("=" * 60)
    total_samples = sum(s.get("sample_count", 1) for s in samples)
    lines.append(
        f"  Period: {(to_dt - from_dt).days} days | "
        f"Samples: {total_samples:,}"
    )

    return "\n".join(lines)


def _make_sparkline(values: list[float]) -> str:
    """Generate a Unicode sparkline from a list of values."""
    if not values:
        return ""
    blocks = " \u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"
    lo = min(values)
    hi = max(values)
    spread = hi - lo if hi != lo else 1
    chars = []
    for v in values:
        idx = int((v - lo) / spread * (len(blocks) - 1))
        chars.append(blocks[idx])
    return "".join(chars)


# ======================================================================
# daemon subcommand group
# ======================================================================


@cli.group(name="daemon")
def daemon_group() -> None:
    """Daemon management commands (start, stop, status, run, install, uninstall)."""


@daemon_group.command(name="start")
@click.option("--config", "-c", "config_path", default=None, help="Config file path override.")
def daemon_start(config_path: str | None) -> None:
    """Start the daemon as a background process."""
    if is_container_mode():
        click.echo(
            "Error: 'daemon start' is not available in container mode.\n"
            "Fix: Use 'daemon run' instead (foreground).",
            err=True,
        )
        sys.exit(1)

    config = _load_config_or_exit(config_path)
    running, existing_pid = is_daemon_running(config)
    if running:
        click.echo(
            f"Error: Daemon is already running (PID {existing_pid}).\n"
            "Fix: Use 'daemon stop' first, or 'daemon status' for details.",
            err=True,
        )
        sys.exit(1)

    child_pid = daemonise(config)
    click.echo(f"Daemon started (PID {child_pid})", err=True)


@daemon_group.command(name="run")
@click.option("--config", "-c", "config_path", default=None, help="Config file path override.")
def daemon_run(config_path: str | None) -> None:
    """Run daemon in the foreground (for systemd/Docker)."""
    config = _load_config_or_exit(config_path)
    running, existing_pid = is_daemon_running(config)
    if running:
        click.echo(
            f"Error: Daemon is already running (PID {existing_pid}).\n"
            "Fix: Use 'daemon stop' first.",
            err=True,
        )
        sys.exit(1)

    runner = DaemonRunner(config, foreground=True)
    runner.run()


@daemon_group.command(name="stop")
@click.option("--config", "-c", "config_path", default=None, help="Config file path override.")
def daemon_stop(config_path: str | None) -> None:
    """Stop the running daemon."""
    import signal as sig
    import time as t

    config = _load_config_or_exit(config_path)
    running, pid = is_daemon_running(config)
    if not running:
        click.echo("Daemon is not running.", err=True)
        return

    # Send SIGTERM
    try:
        os.kill(pid, sig.SIGTERM)
    except ProcessLookupError:
        click.echo("Daemon process already exited.", err=True)
        from llm_monitor.daemon import remove_pid_file
        remove_pid_file(get_pid_file(config))
        return

    # Wait up to 5 seconds for clean shutdown
    for _ in range(25):
        t.sleep(0.2)
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            click.echo("Daemon stopped.", err=True)
            return

    # SIGKILL as last resort
    try:
        os.kill(pid, sig.SIGKILL)
    except ProcessLookupError:
        pass

    from llm_monitor.daemon import remove_pid_file
    remove_pid_file(get_pid_file(config))
    click.echo("Daemon killed.", err=True)


@daemon_group.command(name="status")
@click.option("--config", "-c", "config_path", default=None, help="Config file path override.")
@click.option("--quiet", "-q", is_flag=True, default=False, help="Exit 0/1 silently (for health checks).")
def daemon_status(config_path: str | None, quiet: bool) -> None:
    """Show daemon status."""
    config = _load_config_or_exit(config_path)
    running, pid = is_daemon_running(config)

    if not running:
        if quiet:
            sys.exit(1)
        click.echo("Daemon: stopped")
        sys.exit(1)

    if quiet:
        sys.exit(0)

    # Read state file for details
    state_path = get_state_file(config)
    state = read_state(state_path)

    # Calculate uptime
    uptime_str = ""
    if state and state.get("started_at"):
        try:
            started = datetime.fromisoformat(state["started_at"])
            delta = datetime.now(timezone.utc) - started
            hours = int(delta.total_seconds() // 3600)
            minutes = int((delta.total_seconds() % 3600) // 60)
            if hours > 0:
                uptime_str = f"{hours}h {minutes}m"
            else:
                uptime_str = f"{minutes}m"
        except (ValueError, TypeError):
            uptime_str = "unknown"

    click.echo(f"Daemon: running (PID {pid}, uptime {uptime_str})")

    # Per-provider status
    if state and state.get("providers"):
        now = datetime.now(timezone.utc)
        for pname, pinfo in state["providers"].items():
            last_poll_str = ""
            if pinfo.get("last_poll"):
                try:
                    last_dt = datetime.fromisoformat(pinfo["last_poll"])
                    ago_secs = int((now - last_dt).total_seconds())
                    if ago_secs < 60:
                        last_poll_str = f"last poll {ago_secs}s ago"
                    else:
                        last_poll_str = f"last poll {ago_secs // 60}m ago"
                except (ValueError, TypeError):
                    last_poll_str = "last poll unknown"

            next_poll_str = ""
            if pinfo.get("next_poll"):
                try:
                    next_dt = datetime.fromisoformat(pinfo["next_poll"])
                    until_secs = int((next_dt - now).total_seconds())
                    if until_secs <= 0:
                        next_poll_str = "next now"
                    elif until_secs < 60:
                        next_poll_str = f"next in {until_secs}s"
                    else:
                        next_poll_str = f"next in {until_secs // 60}m"
                except (ValueError, TypeError):
                    next_poll_str = ""

            pstatus = pinfo.get("status", "unknown")
            click.echo(
                f"  {pname:<12} {last_poll_str:<22} {next_poll_str:<16} {pstatus}"
            )

    # Database info
    store = HistoryStore()
    store.open()
    try:
        info = store.stats()
        click.echo(
            f"Database: {info['db_path']} ({_format_size(info['db_size'])})"
        )
    finally:
        store.close()


@daemon_group.command(name="install")
@click.option("--config", "-c", "config_path", default=None, help="Config file path override.")
def daemon_install(config_path: str | None) -> None:
    """Install systemd user service."""
    import shutil
    import subprocess

    if is_container_mode():
        click.echo(
            "Error: 'daemon install' is not available in container mode.\n"
            "Fix: Use 'daemon run' directly with Docker.",
            err=True,
        )
        sys.exit(1)

    _load_config_or_exit(config_path)

    # Resolve the llm-monitor binary path
    binary = shutil.which("llm-monitor")
    if not binary:
        binary = f"{sys.executable} -m llm_monitor"

    service_dir = os.path.expanduser("~/.config/systemd/user")
    service_path = os.path.join(service_dir, "llm-monitor.service")

    unit_content = f"""\
[Unit]
Description=LLM Usage Monitor Daemon
Documentation=https://github.com/danielithomas/llm-monitor
After=network-online.target
Wants=network-online.target

[Service]
Type=exec
ExecStart={binary} daemon run
Restart=on-failure
RestartSec=30
Environment=LLM_MONITOR_LOG_LEVEL=info

[Install]
WantedBy=default.target
"""

    os.makedirs(service_dir, exist_ok=True)
    with open(service_path, "w") as f:
        f.write(unit_content)

    click.echo(f"Service file written: {service_path}", err=True)

    # Enable and start
    try:
        subprocess.run(
            ["systemctl", "--user", "daemon-reload"],
            check=True, capture_output=True,
        )
        subprocess.run(
            ["systemctl", "--user", "enable", "--now", "llm-monitor"],
            check=True, capture_output=True,
        )
        click.echo("Service enabled and started.", err=True)
    except FileNotFoundError:
        click.echo(
            "Warning: systemctl not found. Service file written but not enabled.\n"
            "Fix: Run manually: systemctl --user daemon-reload && "
            "systemctl --user enable --now llm-monitor",
            err=True,
        )
    except subprocess.CalledProcessError as exc:
        click.echo(
            f"Warning: systemctl command failed: {exc.stderr.decode().strip()}\n"
            "Fix: Check systemd status with: systemctl --user status llm-monitor",
            err=True,
        )


@daemon_group.command(name="uninstall")
def daemon_uninstall() -> None:
    """Remove systemd user service."""
    import subprocess

    if is_container_mode():
        click.echo(
            "Error: 'daemon uninstall' is not available in container mode.",
            err=True,
        )
        sys.exit(1)

    service_path = os.path.expanduser(
        "~/.config/systemd/user/llm-monitor.service"
    )

    # Disable and stop
    try:
        subprocess.run(
            ["systemctl", "--user", "disable", "--now", "llm-monitor"],
            check=True, capture_output=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    # Remove unit file
    try:
        os.unlink(service_path)
        click.echo("Service file removed.", err=True)
    except FileNotFoundError:
        click.echo("Service file not found — already uninstalled.", err=True)
        return

    # Reload
    try:
        subprocess.run(
            ["systemctl", "--user", "daemon-reload"],
            check=True, capture_output=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    click.echo("Service uninstalled.", err=True)
