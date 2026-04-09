"""Configuration loading and path resolution for llm-monitor.

Follows XDG Base Directory specification with environment variable overrides.
See SPEC.md Section 4.6 for the full configuration schema.
"""

from __future__ import annotations

import copy
import os
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]

from llm_monitor.security import check_file_permissions, is_container_mode, secure_mkdir

DEFAULT_CONFIG: dict = {
    "general": {
        "default_providers": ["claude"],
        "poll_interval": 600,
        "notification_enabled": False,
    },
    "thresholds": {
        "warning": 70,
        "critical": 90,
    },
    "providers": {
        "claude": {
            "enabled": True,
            "credentials_path": "",
            "show_opus": True,
        },
        "openai": {
            "enabled": False,
            "admin_key_env": "OPENAI_ADMIN_KEY",
        },
    },
    "history": {
        "enabled": True,
        "retention_days": 90,
    },
    "daemon": {
        "log_file": "",
        "pid_file": "",
    },
}


def get_config_path() -> Path:
    """Resolve the config file path using the standard resolution order.

    Resolution order:
    1. ``$LLM_MONITOR_CONFIG`` environment variable
    2. ``$XDG_CONFIG_HOME/llm-monitor/config.toml``
    3. ``~/.config/llm-monitor/config.toml``
    """
    env_path = os.environ.get("LLM_MONITOR_CONFIG")
    if env_path:
        return Path(env_path)

    xdg_config = os.environ.get("XDG_CONFIG_HOME")
    if xdg_config:
        return Path(xdg_config) / "llm-monitor" / "config.toml"

    return Path.home() / ".config" / "llm-monitor" / "config.toml"


def get_data_dir() -> Path:
    """Resolve the data directory (used for history DB).

    Resolution order:
    1. ``$LLM_MONITOR_DATA_DIR``
    2. ``$XDG_DATA_HOME/llm-monitor/``
    3. ``~/.local/share/llm-monitor/``
    """
    env_path = os.environ.get("LLM_MONITOR_DATA_DIR")
    if env_path:
        return Path(env_path)

    xdg_data = os.environ.get("XDG_DATA_HOME")
    if xdg_data:
        return Path(xdg_data) / "llm-monitor"

    return Path.home() / ".local" / "share" / "llm-monitor"


def get_pid_dir() -> Path:
    """Resolve the PID file directory.

    Resolution order:
    1. ``$XDG_RUNTIME_DIR/llm-monitor/``
    2. ``/tmp/llm-monitor-<uid>/``
    """
    xdg_runtime = os.environ.get("XDG_RUNTIME_DIR")
    if xdg_runtime:
        return Path(xdg_runtime) / "llm-monitor"

    return Path(f"/tmp/llm-monitor-{os.getuid()}")


def get_log_dir() -> Path:
    """Resolve the daemon log directory.

    Resolution order:
    1. ``$XDG_STATE_HOME/llm-monitor/``
    2. ``~/.local/state/llm-monitor/``
    """
    xdg_state = os.environ.get("XDG_STATE_HOME")
    if xdg_state:
        return Path(xdg_state) / "llm-monitor"

    return Path.home() / ".local" / "state" / "llm-monitor"


def get_pid_file(config: dict) -> Path:
    """Resolve the daemon PID file path from config or default."""
    custom = config.get("daemon", {}).get("pid_file", "")
    if custom:
        return Path(custom)
    return get_pid_dir() / "daemon.pid"


def get_log_file(config: dict) -> Path:
    """Resolve the daemon log file path from config or default."""
    custom = config.get("daemon", {}).get("log_file", "")
    if custom:
        return Path(custom)
    return get_log_dir() / "daemon.log"


def get_state_file(config: dict) -> Path:
    """Resolve the daemon state file path (always derived, not configurable)."""
    return get_pid_dir() / "daemon.state"


def get_cache_dir() -> Path:
    """Resolve the cache directory.

    Resolution order:
    1. ``$LLM_MONITOR_CACHE_DIR``
    2. ``$XDG_CACHE_HOME/llm-monitor/``
    3. ``~/.cache/llm-monitor/``
    """
    env_path = os.environ.get("LLM_MONITOR_CACHE_DIR")
    if env_path:
        return Path(env_path)

    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache:
        return Path(xdg_cache) / "llm-monitor"

    return Path.home() / ".cache" / "llm-monitor"


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into a copy of *base*.

    Values in *override* take precedence. Nested dicts are merged rather than
    replaced wholesale so that partial provider sections don't clobber defaults.
    """
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(path: str | None = None) -> dict:
    """Load TOML configuration, falling back to defaults when the file is absent.

    Parameters
    ----------
    path:
        Explicit path to a TOML config file.  When *None*, the standard
        resolution order is used (see :func:`get_config_path`).

    Returns
    -------
    dict
        The merged configuration dictionary (defaults + file overrides).

    Raises
    ------
    ValueError
        If the config file exists but contains malformed TOML.
    """
    config_path = Path(path) if path else get_config_path()

    if not config_path.exists():
        return copy.deepcopy(DEFAULT_CONFIG)

    # Parse the TOML file
    try:
        raw_bytes = config_path.read_bytes()
        user_config = tomllib.loads(raw_bytes.decode("utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(
            f"Malformed TOML in config file {config_path}: {exc}"
        ) from exc

    # Check file permissions (unless running in a container)
    if not is_container_mode():
        warnings = check_file_permissions(str(config_path))
        for warning in warnings:
            print(warning, file=sys.stderr)

    return _deep_merge(DEFAULT_CONFIG, user_config)
