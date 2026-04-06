"""JSON output formatter for llm-monitor."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from llm_monitor.models import ProviderStatus


def format_resets_in_human(resets_at: datetime | None) -> str | None:
    """Compute a human-readable duration from now until *resets_at*.

    Format uses the two largest non-zero units:
      "2d 13h", "2h 15m", "45m", "< 1m"

    Returns None when *resets_at* is None.
    """
    if resets_at is None:
        return None

    now = datetime.now(timezone.utc)
    delta = resets_at - now
    total_seconds = int(delta.total_seconds())

    if total_seconds < 60:
        return "< 1m"

    days, remainder = divmod(total_seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes = remainder // 60

    parts: list[str] = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if minutes and len(parts) < 2:
        parts.append(f"{minutes}m")

    return " ".join(parts) if parts else "< 1m"


def _serialize_window(window: object) -> dict:
    """Serialize a UsageWindow to a JSON-safe dict."""
    from llm_monitor.models import UsageWindow

    assert isinstance(window, UsageWindow)
    return {
        "name": window.name,
        "utilisation": window.utilisation,
        "resets_at": window.resets_at.isoformat() if window.resets_at else None,
        "resets_in_human": format_resets_in_human(window.resets_at),
        "status": window.status,
        "unit": window.unit,
        "raw_value": window.raw_value,
        "raw_limit": window.raw_limit,
    }


def _serialize_status(status: ProviderStatus) -> dict:
    """Serialize a ProviderStatus to a JSON-safe dict, stripping secrets."""
    # Build extras, converting any SecretStr values to redacted strings
    from llm_monitor.models import SecretStr

    safe_extras: dict = {}
    for key, value in status.extras.items():
        if isinstance(value, SecretStr):
            continue  # omit secrets entirely
        safe_extras[key] = value

    return {
        "provider_name": status.provider_name,
        "provider_display": status.provider_display,
        "timestamp": status.timestamp.isoformat(),
        "cached": status.cached,
        "cache_age_seconds": status.cache_age_seconds,
        "windows": [_serialize_window(w) for w in status.windows],
        "extras": safe_extras,
        "errors": list(status.errors),
    }


def format_json(statuses: list[ProviderStatus], version: str) -> str:
    """Produce JSON output matching the spec schema (Section 4.2.3).

    Parameters
    ----------
    statuses:
        List of provider status objects to serialise.
    version:
        The package version string (e.g. "0.1.0").

    Returns
    -------
    str
        Pretty-printed JSON string (``indent=2``).
    """
    now = datetime.now(timezone.utc).astimezone()

    payload = {
        "timestamp": now.isoformat(),
        "version": version,
        "providers": [_serialize_status(s) for s in statuses],
    }

    return json.dumps(payload, indent=2)
