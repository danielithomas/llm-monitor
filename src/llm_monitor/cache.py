"""Per-provider cache layer for llm-monitor.

Stores the latest ProviderStatus per provider on disk with TTL-based
invalidation and advisory file locking.
"""

from __future__ import annotations

import fcntl
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from llm_monitor.models import ProviderStatus, UsageWindow
from llm_monitor.security import secure_mkdir, secure_write


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------


def _datetime_to_iso(dt: Optional[datetime]) -> Optional[str]:
    """Convert a datetime to ISO 8601 string, or None."""
    if dt is None:
        return None
    return dt.isoformat()


def _iso_to_datetime(s: Optional[str]) -> Optional[datetime]:
    """Parse an ISO 8601 string to a datetime, or None."""
    if s is None:
        return None
    return datetime.fromisoformat(s)


def _window_to_dict(w: UsageWindow) -> dict:
    return {
        "name": w.name,
        "utilisation": w.utilisation,
        "resets_at": _datetime_to_iso(w.resets_at),
        "status": w.status,
        "unit": w.unit,
        "raw_value": w.raw_value,
        "raw_limit": w.raw_limit,
    }


def _dict_to_window(d: dict) -> UsageWindow:
    return UsageWindow(
        name=d["name"],
        utilisation=d["utilisation"],
        resets_at=_iso_to_datetime(d.get("resets_at")),
        status=d["status"],
        unit=d["unit"],
        raw_value=d.get("raw_value"),
        raw_limit=d.get("raw_limit"),
    )


def _status_to_dict(status: ProviderStatus) -> dict:
    return {
        "provider_name": status.provider_name,
        "provider_display": status.provider_display,
        "timestamp": _datetime_to_iso(status.timestamp),
        "cached": status.cached,
        "cache_age_seconds": status.cache_age_seconds,
        "windows": [_window_to_dict(w) for w in status.windows],
        "extras": status.extras,
        "errors": status.errors,
    }


def _dict_to_status(d: dict) -> ProviderStatus:
    return ProviderStatus(
        provider_name=d["provider_name"],
        provider_display=d["provider_display"],
        timestamp=_iso_to_datetime(d["timestamp"]),  # type: ignore[arg-type]
        cached=d.get("cached", True),
        cache_age_seconds=d.get("cache_age_seconds", 0),
        windows=[_dict_to_window(w) for w in d.get("windows", [])],
        extras=d.get("extras", {}),
        errors=d.get("errors", []),
    )


# ---------------------------------------------------------------------------
# ProviderCache
# ---------------------------------------------------------------------------

_LOCK_TIMEOUT = 2  # seconds


class ProviderCache:
    """On-disk per-provider cache with file locking and TTL support.

    Cache structure::

        cache_dir/
          <provider_name>/
            last.json
    """

    def __init__(self, cache_dir: Path) -> None:
        self._cache_dir = cache_dir

    def _cache_path(self, provider_name: str) -> Path:
        return self._cache_dir / provider_name / "last.json"

    # ------------------------------------------------------------------
    # Read / write status
    # ------------------------------------------------------------------

    def read(self, provider_name: str, poll_interval: int) -> ProviderStatus | None:
        """Read cached status for *provider_name*.

        Returns None when:
        - The cache file does not exist
        - The cache file is older than *poll_interval* seconds (stale)
        - The cache file is corrupt
        """
        path = self._cache_path(provider_name)
        if not path.exists():
            return None

        try:
            data = self._read_locked(path)
        except Exception:
            return None

        cached_at_str = data.get("cached_at")
        if not cached_at_str:
            return None

        cached_at = _iso_to_datetime(cached_at_str)
        if cached_at is None:
            return None

        now = datetime.now(timezone.utc)
        age_seconds = (now - cached_at).total_seconds()
        if age_seconds > poll_interval:
            return None

        status_data = data.get("status")
        if not status_data:
            return None

        status = _dict_to_status(status_data)
        status.cached = True
        status.cache_age_seconds = int(age_seconds)
        return status

    def write(self, provider_name: str, status: ProviderStatus) -> None:
        """Write *status* to the cache, atomically."""
        path = self._cache_path(provider_name)

        # Build the cache envelope
        envelope: dict = {
            "status": _status_to_dict(status),
            "cached_at": _datetime_to_iso(datetime.now(timezone.utc)),
        }

        # Preserve existing backoff data if present
        if path.exists():
            try:
                existing = self._read_locked(path)
                if "backoff" in existing:
                    envelope["backoff"] = existing["backoff"]
            except Exception:
                pass

        payload = json.dumps(envelope, indent=2)
        secure_write(str(path), payload)

    # ------------------------------------------------------------------
    # Backoff state
    # ------------------------------------------------------------------

    def read_backoff(self, provider_name: str) -> tuple[int, datetime | None]:
        """Read backoff count and backoff_until from cache metadata.

        Returns (0, None) if no backoff state exists.
        """
        path = self._cache_path(provider_name)
        if not path.exists():
            return 0, None

        try:
            data = self._read_locked(path)
        except Exception:
            return 0, None

        backoff = data.get("backoff", {})
        count = backoff.get("count", 0)
        until = _iso_to_datetime(backoff.get("until"))
        return count, until

    def write_backoff(self, provider_name: str, count: int, until: datetime) -> None:
        """Persist backoff state into the cache file."""
        path = self._cache_path(provider_name)

        # Read existing envelope or create a new one
        envelope: dict = {}
        if path.exists():
            try:
                envelope = self._read_locked(path)
            except Exception:
                pass

        envelope["backoff"] = {
            "count": count,
            "until": _datetime_to_iso(until),
        }

        payload = json.dumps(envelope, indent=2)
        secure_write(str(path), payload)

    def clear_backoff(self, provider_name: str) -> None:
        """Reset backoff state on success."""
        path = self._cache_path(provider_name)
        if not path.exists():
            return

        try:
            envelope = self._read_locked(path)
        except Exception:
            return

        if "backoff" in envelope:
            del envelope["backoff"]
            payload = json.dumps(envelope, indent=2)
            secure_write(str(path), payload)

    # ------------------------------------------------------------------
    # clear_all
    # ------------------------------------------------------------------

    def clear_all(self) -> None:
        """Delete all cache files (for --clear-cache)."""
        if self._cache_dir.exists():
            shutil.rmtree(str(self._cache_dir))

    # ------------------------------------------------------------------
    # File locking helpers
    # ------------------------------------------------------------------

    def _read_locked(self, path: Path) -> dict:
        """Read and parse a JSON cache file with a shared (read) lock."""
        fd = os.open(str(path), os.O_RDONLY)
        try:
            _flock_with_timeout(fd, fcntl.LOCK_SH, _LOCK_TIMEOUT)
            with os.fdopen(os.dup(fd), "r") as f:
                return json.load(f)
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                pass
            os.close(fd)


def _flock_with_timeout(fd: int, operation: int, timeout: float) -> None:
    """Attempt fcntl.flock with LOCK_NB, retrying up to *timeout* seconds."""
    import time

    deadline = time.monotonic() + timeout
    while True:
        try:
            fcntl.flock(fd, operation | fcntl.LOCK_NB)
            return
        except OSError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.05)
