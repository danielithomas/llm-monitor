"""Migration utilities for the llm-monitor → clawmeter rename.

Handles one-time migration of XDG directories, keyring credentials,
and deprecation warnings for old environment variables.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Old env var names that users may still have set.
_OLD_ENV_VARS = {
    "LLM_MONITOR_CONFIG": "CLAWMETER_CONFIG",
    "LLM_MONITOR_DATA_DIR": "CLAWMETER_DATA_DIR",
    "LLM_MONITOR_CACHE_DIR": "CLAWMETER_CACHE_DIR",
    "LLM_MONITOR_LOG_LEVEL": "CLAWMETER_LOG_LEVEL",
    "LLM_MONITOR_CONTAINER": "CLAWMETER_CONTAINER",
    "LLM_MONITOR_NO_COLOR": "CLAWMETER_NO_COLOR",
}

_OLD_APP_DIR = "llm-monitor"
_NEW_APP_DIR = "clawmeter"

# Module-level flag to emit deprecation warnings at most once per process.
_env_warning_emitted = False


def warn_old_env_vars() -> None:
    """Emit a stderr warning if any old LLM_MONITOR_* env vars are detected."""
    global _env_warning_emitted
    if _env_warning_emitted:
        return

    found = [old for old in _OLD_ENV_VARS if os.environ.get(old)]
    if found:
        _env_warning_emitted = True
        names = ", ".join(found)
        print(
            f"Warning: Deprecated environment variable(s) detected: {names}. "
            f"Please update to the new CLAWMETER_* equivalents. "
            f"The old LLM_MONITOR_* variables are no longer read.",
            file=sys.stderr,
        )


def migrate_xdg_directories() -> None:
    """Move old llm-monitor XDG directories to clawmeter if applicable.

    For each XDG base (config, data, cache, state), if the old directory
    exists and the new one does not, rename (move) it. Logs actions to stderr.
    """
    bases = [
        (
            os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config")),
            "config",
        ),
        (
            os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local" / "share")),
            "data",
        ),
        (
            os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")),
            "cache",
        ),
        (
            os.environ.get("XDG_STATE_HOME", str(Path.home() / ".local" / "state")),
            "state",
        ),
    ]

    for base_dir, label in bases:
        old_path = Path(base_dir) / _OLD_APP_DIR
        new_path = Path(base_dir) / _NEW_APP_DIR

        if old_path.is_dir() and not new_path.exists():
            try:
                old_path.rename(new_path)
                print(
                    f"Migrated {label} directory: {old_path} → {new_path}",
                    file=sys.stderr,
                )
            except OSError as exc:
                print(
                    f"Warning: Could not migrate {label} directory "
                    f"{old_path} → {new_path}: {exc}",
                    file=sys.stderr,
                )


def migrate_keyring_credential(provider_name: str) -> None:
    """Migrate a stored keyring credential from the old service name.

    Reads from ``llm-monitor/<provider>``, stores under
    ``clawmeter/<provider>``, and deletes the old entry.
    Best-effort: silently skipped if keyring is unavailable.
    """
    try:
        import keyring as kr
    except ImportError:
        return

    old_service = f"llm-monitor/{provider_name}"
    new_service = f"clawmeter/{provider_name}"

    # Common username patterns used by providers
    usernames = ["api_key", "management_key", "admin_key"]

    for username in usernames:
        try:
            existing_new = kr.get_password(new_service, username)
            if existing_new:
                # Already migrated or user set it directly.
                continue

            old_value = kr.get_password(old_service, username)
            if old_value:
                kr.set_password(new_service, username, old_value)
                kr.delete_password(old_service, username)
        except Exception:
            # keyring backends can raise various errors; skip gracefully.
            continue
