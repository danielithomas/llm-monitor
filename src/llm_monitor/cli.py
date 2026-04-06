"""Click CLI entry point for llm-monitor.

All data goes to stdout, all messages go to stderr.
See SPEC.md Section 4 for the full CLI specification.
"""

from __future__ import annotations

import asyncio
import os
import sys

import click

import llm_monitor
from llm_monitor.cache import ProviderCache
from llm_monitor.config import get_cache_dir, load_config
from llm_monitor.core import determine_exit_code, fetch_all
from llm_monitor.formatters.json_fmt import format_json
from llm_monitor.formatters.table_fmt import format_table
from llm_monitor.providers import PROVIDERS, get_enabled_providers


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


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--now", is_flag=True, default=False, help="Display table output.")
@click.option(
    "--provider", "-p", default=None,
    help="Comma-separated list of providers to query.",
)
@click.option("--fresh", "-f", is_flag=True, default=False, help="Bypass cache.")
@click.option("--verbose", "-v", is_flag=True, default=False, help="Verbose logging to stderr.")
@click.option("--quiet", "-q", is_flag=True, default=False, help="Suppress non-error stderr output.")
@click.option(
    "--version", "-V", is_flag=True, default=False,
    help="Print version and exit.",
)
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
    help="Disable history recording (placeholder for v0.2.0).",
)
def cli(
    now: bool,
    provider: str | None,
    fresh: bool,
    verbose: bool,
    quiet: bool,
    version: bool,
    config_path: str | None,
    clear_cache: bool,
    list_providers: bool,
    no_colour: bool,
    colour: str | None,
    no_history: bool,
) -> None:
    """Monitor LLM service usage across providers."""
    # --version
    if version:
        click.echo(f"llm-monitor {llm_monitor.__version__}")
        return

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
            # Check if configured by instantiating
            try:
                instance = cls(config)
                configured = instance.is_configured()
            except Exception:
                configured = False
            configured_str = "configured" if configured else "not configured"
            click.echo(f"{name}: {enabled_str}, {configured_str}")
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
