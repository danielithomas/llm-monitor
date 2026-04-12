"""Tests for the cache layer."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from clawmeter.cache import (
    ProviderCache,
    _datetime_to_iso,
    _dict_to_status,
    _iso_to_datetime,
    _status_to_dict,
)
from clawmeter.models import ProviderStatus, UsageWindow


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sample_status(
    provider_name: str = "claude",
    errors: list[str] | None = None,
) -> ProviderStatus:
    return ProviderStatus(
        provider_name=provider_name,
        provider_display="Anthropic Claude",
        timestamp=datetime(2026, 4, 5, 10, 30, 0, tzinfo=timezone.utc),
        cached=False,
        cache_age_seconds=0,
        windows=[
            UsageWindow(
                name="Session (5h)",
                utilisation=42.0,
                resets_at=datetime(2026, 4, 5, 15, 0, 0, tzinfo=timezone.utc),
                status="normal",
                unit="percent",
            ),
        ],
        errors=errors or [],
    )


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


class TestSerialisation:
    def test_datetime_round_trip(self):
        dt = datetime(2026, 4, 5, 10, 30, 0, tzinfo=timezone.utc)
        iso = _datetime_to_iso(dt)
        assert isinstance(iso, str)
        restored = _iso_to_datetime(iso)
        assert restored == dt

    def test_datetime_none(self):
        assert _datetime_to_iso(None) is None
        assert _iso_to_datetime(None) is None

    def test_status_round_trip(self):
        status = _sample_status()
        d = _status_to_dict(status)
        restored = _dict_to_status(d)
        assert restored.provider_name == status.provider_name
        assert restored.provider_display == status.provider_display
        assert restored.timestamp == status.timestamp
        assert len(restored.windows) == len(status.windows)
        assert restored.windows[0].name == status.windows[0].name
        assert restored.windows[0].utilisation == status.windows[0].utilisation
        assert restored.windows[0].resets_at == status.windows[0].resets_at

    def test_status_with_errors_round_trip(self):
        status = _sample_status(errors=["Something went wrong"])
        d = _status_to_dict(status)
        restored = _dict_to_status(d)
        assert restored.errors == ["Something went wrong"]


# ---------------------------------------------------------------------------
# Write / Read round-trip
# ---------------------------------------------------------------------------


class TestWriteRead:
    def test_round_trip(self, tmp_path):
        cache = ProviderCache(tmp_path)
        status = _sample_status()
        cache.write("claude", status)

        result = cache.read("claude", poll_interval=600)
        assert result is not None
        assert result.provider_name == "claude"
        assert result.cached is True
        assert result.cache_age_seconds >= 0
        assert len(result.windows) == 1
        assert result.windows[0].name == "Session (5h)"

    def test_ttl_expiry(self, tmp_path):
        """Cache older than poll_interval returns None."""
        cache = ProviderCache(tmp_path)
        status = _sample_status()
        cache.write("claude", status)

        # Manually backdate the cached_at
        cache_path = tmp_path / "claude" / "last.json"
        data = json.loads(cache_path.read_text())
        old_time = datetime.now(timezone.utc) - timedelta(seconds=700)
        data["cached_at"] = old_time.isoformat()
        cache_path.write_text(json.dumps(data))

        result = cache.read("claude", poll_interval=600)
        assert result is None

    def test_nonexistent_provider_returns_none(self, tmp_path):
        cache = ProviderCache(tmp_path)
        result = cache.read("nonexistent", poll_interval=600)
        assert result is None

    def test_corrupt_cache_returns_none(self, tmp_path):
        cache = ProviderCache(tmp_path)
        # Write a corrupt file
        cache_dir = tmp_path / "claude"
        cache_dir.mkdir(parents=True)
        (cache_dir / "last.json").write_text("not valid json {{{")
        result = cache.read("claude", poll_interval=600)
        assert result is None


# ---------------------------------------------------------------------------
# File permissions
# ---------------------------------------------------------------------------


class TestFilePermissions:
    def test_cache_file_has_0600(self, tmp_path):
        cache = ProviderCache(tmp_path)
        status = _sample_status()
        cache.write("claude", status)

        cache_path = tmp_path / "claude" / "last.json"
        mode = os.stat(str(cache_path)).st_mode & 0o777
        assert mode == 0o600


# ---------------------------------------------------------------------------
# clear_all
# ---------------------------------------------------------------------------


class TestClearAll:
    def test_deletes_all_files(self, tmp_path):
        cache = ProviderCache(tmp_path)
        cache.write("claude", _sample_status("claude"))
        cache.write("openai", _sample_status("openai"))

        assert (tmp_path / "claude" / "last.json").exists()
        assert (tmp_path / "openai" / "last.json").exists()

        cache.clear_all()
        assert not tmp_path.exists()

    def test_clear_all_no_dir(self, tmp_path):
        """clear_all is safe when cache dir doesn't exist."""
        cache_dir = tmp_path / "nonexistent"
        cache = ProviderCache(cache_dir)
        cache.clear_all()  # should not raise


# ---------------------------------------------------------------------------
# Backoff state
# ---------------------------------------------------------------------------


class TestBackoff:
    def test_write_read_backoff(self, tmp_path):
        cache = ProviderCache(tmp_path)
        # Write initial status so the cache file exists
        cache.write("claude", _sample_status())

        until = datetime(2026, 4, 5, 11, 0, 0, tzinfo=timezone.utc)
        cache.write_backoff("claude", count=3, until=until)

        count, backoff_until = cache.read_backoff("claude")
        assert count == 3
        assert backoff_until == until

    def test_backoff_escalation(self, tmp_path):
        """Successive backoff writes escalate the count."""
        cache = ProviderCache(tmp_path)
        cache.write("claude", _sample_status())

        t1 = datetime(2026, 4, 5, 11, 0, 0, tzinfo=timezone.utc)
        cache.write_backoff("claude", count=1, until=t1)

        t2 = datetime(2026, 4, 5, 11, 5, 0, tzinfo=timezone.utc)
        cache.write_backoff("claude", count=2, until=t2)

        count, until = cache.read_backoff("claude")
        assert count == 2
        assert until == t2

    def test_clear_backoff(self, tmp_path):
        cache = ProviderCache(tmp_path)
        cache.write("claude", _sample_status())

        until = datetime(2026, 4, 5, 11, 0, 0, tzinfo=timezone.utc)
        cache.write_backoff("claude", count=2, until=until)

        cache.clear_backoff("claude")
        count, backoff_until = cache.read_backoff("claude")
        assert count == 0
        assert backoff_until is None

    def test_read_backoff_no_file(self, tmp_path):
        cache = ProviderCache(tmp_path)
        count, until = cache.read_backoff("claude")
        assert count == 0
        assert until is None

    def test_clear_backoff_no_file(self, tmp_path):
        """clear_backoff is safe when no cache file exists."""
        cache = ProviderCache(tmp_path)
        cache.clear_backoff("claude")  # should not raise

    def test_backoff_preserved_on_status_write(self, tmp_path):
        """Writing a new status preserves existing backoff state."""
        cache = ProviderCache(tmp_path)
        cache.write("claude", _sample_status())

        until = datetime(2026, 4, 5, 11, 0, 0, tzinfo=timezone.utc)
        cache.write_backoff("claude", count=2, until=until)

        # Write a new status
        cache.write("claude", _sample_status())

        count, backoff_until = cache.read_backoff("claude")
        assert count == 2
        assert backoff_until == until
