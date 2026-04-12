"""Tests for the daemon module (v0.3.0).

Covers: poll loop, PID management, signal handling, CLI subcommands.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from clawmeter.cli import cli
from clawmeter.daemon import (
    DaemonRunner,
    is_daemon_running,
    read_pid_file,
    remove_pid_file,
    write_pid_file,
    write_state,
    read_state,
)
from clawmeter.history import HistoryStore
from clawmeter.models import ProviderStatus, UsageWindow


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(tmp_path: Path, **overrides) -> dict:
    """Create a minimal config dict pointing at temp paths."""
    config = {
        "general": {"default_providers": ["mock"], "poll_interval": 1},
        "thresholds": {"warning": 70, "critical": 90},
        "providers": {"mock": {"enabled": True}},
        "history": {"enabled": True, "retention_days": 90},
        "daemon": {"log_file": "", "pid_file": str(tmp_path / "daemon.pid")},
    }
    config.update(overrides)
    return config


def _make_status(
    provider: str = "mock",
    utilisation: float = 42.0,
    ts: datetime | None = None,
) -> ProviderStatus:
    """Create a test ProviderStatus."""
    if ts is None:
        ts = datetime.now(timezone.utc)
    return ProviderStatus(
        provider_name=provider,
        provider_display=provider.title(),
        timestamp=ts,
        cached=False,
        cache_age_seconds=0,
        windows=[
            UsageWindow(
                name="Session (5h)",
                utilisation=utilisation,
                resets_at=ts + timedelta(hours=5),
                status="normal",
                unit="percent",
            )
        ],
    )


class _MockProvider:
    """Minimal provider for daemon tests."""

    def __init__(self, name: str = "mock", fail: bool = False):
        self._name = name
        self._fail = fail
        self.call_count = 0

    def name(self) -> str:
        return self._name

    def display_name(self) -> str:
        return self._name.title()

    def is_configured(self) -> bool:
        return True

    async def fetch_usage(self, client):
        self.call_count += 1
        if self._fail:
            return ProviderStatus(
                provider_name=self._name,
                provider_display=self._name.title(),
                timestamp=datetime.now(timezone.utc),
                cached=False,
                cache_age_seconds=0,
                errors=["Test error"],
            )
        return _make_status(provider=self._name)

    def auth_instructions(self) -> str:
        return "Test"


def _setup_daemon_env(tmp_path: Path, monkeypatch) -> dict:
    """Set up env for daemon tests. Returns config dict."""
    monkeypatch.setenv("CLAWMETER_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("CLAWMETER_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "runtime"))

    config = _make_config(tmp_path)
    return config


# ===========================================================================
# PID file management
# ===========================================================================


class TestPidFile:
    def test_write_and_read(self, tmp_path):
        pid_path = tmp_path / "test.pid"
        write_pid_file(pid_path)
        assert read_pid_file(pid_path) == os.getpid()

    def test_read_missing(self, tmp_path):
        assert read_pid_file(tmp_path / "missing.pid") is None

    def test_read_corrupt(self, tmp_path):
        pid_path = tmp_path / "bad.pid"
        pid_path.write_text("not-a-number")
        assert read_pid_file(pid_path) is None

    def test_remove(self, tmp_path):
        pid_path = tmp_path / "test.pid"
        write_pid_file(pid_path)
        assert pid_path.exists()
        remove_pid_file(pid_path)
        assert not pid_path.exists()

    def test_remove_missing(self, tmp_path):
        # Should not raise
        remove_pid_file(tmp_path / "missing.pid")


class TestDaemonRunningDetection:
    def test_not_running_no_pid_file(self, tmp_path, monkeypatch):
        config = _setup_daemon_env(tmp_path, monkeypatch)
        running, pid = is_daemon_running(config)
        assert not running
        assert pid is None

    def test_running_with_live_process(self, tmp_path, monkeypatch):
        config = _setup_daemon_env(tmp_path, monkeypatch)
        pid_path = Path(config["daemon"]["pid_file"])
        pid_path.parent.mkdir(parents=True, exist_ok=True)
        pid_path.write_text(str(os.getpid()))

        running, pid = is_daemon_running(config)
        assert running
        assert pid == os.getpid()

    def test_stale_pid_file_cleaned(self, tmp_path, monkeypatch):
        config = _setup_daemon_env(tmp_path, monkeypatch)
        pid_path = Path(config["daemon"]["pid_file"])
        pid_path.parent.mkdir(parents=True, exist_ok=True)
        # Use a PID that definitely doesn't exist
        pid_path.write_text("999999999")

        running, pid = is_daemon_running(config)
        assert not running
        assert pid is None
        assert not pid_path.exists()  # Stale file cleaned up


# ===========================================================================
# State file management
# ===========================================================================


class TestStateFile:
    def test_write_and_read(self, tmp_path):
        state_path = tmp_path / "daemon.state"
        state = {"started_at": "2026-04-05T10:00:00+00:00", "pid": 12345}
        write_state(state_path, state)
        result = read_state(state_path)
        assert result == state

    def test_read_missing(self, tmp_path):
        assert read_state(tmp_path / "missing.state") is None


# ===========================================================================
# DaemonRunner — poll loop tests
# ===========================================================================


class TestPollLoop:
    def test_writes_to_history_on_first_tick(self, tmp_path, monkeypatch):
        """daemon run starts poll loop and writes to history DB after first tick."""
        config = _setup_daemon_env(tmp_path, monkeypatch)
        runner = DaemonRunner(config, foreground=True)

        mock_provider = _MockProvider()

        # Patch provider loading to use our mock
        with patch.object(runner, "_init_providers") as mock_init:
            def setup_mocks():
                runner._providers = [mock_provider]
                runner._provider_intervals = {"mock": 600}
                runner._next_poll = {"mock": time.monotonic()}
            mock_init.side_effect = setup_mocks

            # Shutdown after first poll
            original_poll = runner._poll_providers

            async def poll_then_shutdown(providers):
                await original_poll(providers)
                runner._shutdown = True

            runner._poll_providers = poll_then_shutdown

            runner.run()

        assert mock_provider.call_count == 1

        # Verify history DB has records
        db_path = tmp_path / "data" / "history.db"
        assert db_path.exists()
        store = HistoryStore(db_path)
        store.open()
        try:
            info = store.stats()
            assert info["sample_count"] > 0
        finally:
            store.close()

    def test_per_provider_poll_intervals(self, tmp_path, monkeypatch):
        """Poll loop respects per-provider poll_interval overrides."""
        config = _setup_daemon_env(tmp_path, monkeypatch)
        runner = DaemonRunner(config, foreground=True)

        fast_provider = _MockProvider("fast")
        slow_provider = _MockProvider("slow")
        poll_count = 0

        with patch.object(runner, "_init_providers") as mock_init:
            def setup_mocks():
                runner._providers = [fast_provider, slow_provider]
                runner._provider_intervals = {"fast": 1, "slow": 100}
                now = time.monotonic()
                runner._next_poll = {"fast": now, "slow": now}
            mock_init.side_effect = setup_mocks

            original_poll = runner._poll_providers

            async def poll_and_count(providers):
                nonlocal poll_count
                await original_poll(providers)
                poll_count += 1
                # After 3 poll cycles, shutdown
                if poll_count >= 3:
                    runner._shutdown = True

            runner._poll_providers = poll_and_count

            runner.run()

        # Fast provider should have been polled more than slow
        assert fast_provider.call_count >= 2
        # Slow provider polled only on the first tick (interval=100s)
        assert slow_provider.call_count == 1

    def test_survives_provider_errors(self, tmp_path, monkeypatch):
        """Poll loop survives provider errors — one fails, others still polled."""
        config = _setup_daemon_env(tmp_path, monkeypatch)
        runner = DaemonRunner(config, foreground=True)

        good_provider = _MockProvider("good")
        bad_provider = _MockProvider("bad", fail=True)

        with patch.object(runner, "_init_providers") as mock_init:
            def setup_mocks():
                runner._providers = [good_provider, bad_provider]
                runner._provider_intervals = {"good": 600, "bad": 600}
                now = time.monotonic()
                runner._next_poll = {"good": now, "bad": now}
            mock_init.side_effect = setup_mocks

            original_poll = runner._poll_providers

            async def poll_then_shutdown(providers):
                await original_poll(providers)
                runner._shutdown = True

            runner._poll_providers = poll_then_shutdown

            runner.run()

        # Both were polled
        assert good_provider.call_count == 1
        assert bad_provider.call_count == 1

        # Good provider's data was recorded
        db_path = tmp_path / "data" / "history.db"
        store = HistoryStore(db_path)
        store.open()
        try:
            info = store.stats()
            assert info["sample_count"] > 0
            assert "good" in info["providers"]
        finally:
            store.close()

    def test_pid_file_lifecycle(self, tmp_path, monkeypatch):
        """PID file created on start, removed on clean shutdown."""
        config = _setup_daemon_env(tmp_path, monkeypatch)
        pid_path = Path(config["daemon"]["pid_file"])
        runner = DaemonRunner(config, foreground=True)

        pid_existed_during_run = False

        with patch.object(runner, "_init_providers") as mock_init:
            def setup_mocks():
                runner._providers = [_MockProvider()]
                runner._provider_intervals = {"mock": 600}
                runner._next_poll = {"mock": time.monotonic()}
            mock_init.side_effect = setup_mocks

            original_poll = runner._poll_providers

            async def poll_then_check(providers):
                nonlocal pid_existed_during_run
                pid_existed_during_run = pid_path.exists()
                await original_poll(providers)
                runner._shutdown = True

            runner._poll_providers = poll_then_check

            runner.run()

        assert pid_existed_during_run
        assert not pid_path.exists()  # Cleaned up on shutdown


# ===========================================================================
# Signal handling
# ===========================================================================


class TestSignalHandling:
    def test_sigterm_clean_shutdown(self, tmp_path, monkeypatch):
        """SIGTERM triggers clean shutdown — flush writes, close DB, remove PID."""
        config = _setup_daemon_env(tmp_path, monkeypatch)
        pid_path = Path(config["daemon"]["pid_file"])
        runner = DaemonRunner(config, foreground=True)

        with patch.object(runner, "_init_providers") as mock_init:
            def setup_mocks():
                runner._providers = [_MockProvider()]
                runner._provider_intervals = {"mock": 600}
                runner._next_poll = {"mock": time.monotonic()}
            mock_init.side_effect = setup_mocks

            original_poll = runner._poll_providers

            async def poll_then_signal(providers):
                await original_poll(providers)
                # Simulate SIGTERM by calling the handler directly
                runner._handle_sigterm()

            runner._poll_providers = poll_then_signal

            runner.run()

        # PID file should be cleaned up
        assert not pid_path.exists()

        # History DB should have data (written before shutdown)
        db_path = tmp_path / "data" / "history.db"
        store = HistoryStore(db_path)
        store.open()
        try:
            info = store.stats()
            assert info["sample_count"] > 0
        finally:
            store.close()

    def test_sighup_reload(self, tmp_path, monkeypatch):
        """SIGHUP triggers config reload without restart."""
        config = _setup_daemon_env(tmp_path, monkeypatch)
        runner = DaemonRunner(config, foreground=True)
        reload_called = False

        with patch.object(runner, "_init_providers") as mock_init:
            def setup_mocks():
                runner._providers = [_MockProvider()]
                runner._provider_intervals = {"mock": 600}
                runner._next_poll = {"mock": time.monotonic()}
            mock_init.side_effect = setup_mocks

            original_reload = runner._reload_config

            def track_reload():
                nonlocal reload_called
                reload_called = True
                original_reload()

            runner._reload_config = track_reload

            poll_count = 0
            original_poll = runner._poll_providers

            async def poll_with_sighup(providers):
                nonlocal poll_count
                await original_poll(providers)
                poll_count += 1
                if poll_count == 1:
                    # Simulate SIGHUP
                    runner._handle_sighup()
                else:
                    runner._shutdown = True

            runner._poll_providers = poll_with_sighup

            runner.run()

        assert reload_called


# ===========================================================================
# CLI subcommands
# ===========================================================================


class TestDaemonCLI:
    def test_already_running_error(self, tmp_path, monkeypatch):
        """daemon start when already running -> error with existing PID."""
        monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "runtime"))
        monkeypatch.setenv("CLAWMETER_DATA_DIR", str(tmp_path / "data"))
        monkeypatch.setenv("CLAWMETER_CACHE_DIR", str(tmp_path / "cache"))

        # Write a config file
        config_path = tmp_path / "config.toml"
        config_path.write_text(
            f'[daemon]\npid_file = "{tmp_path}/daemon.pid"\n'
        )
        os.chmod(str(config_path), 0o600)
        monkeypatch.setenv("CLAWMETER_CONFIG", str(config_path))

        # Write PID file with our own PID (so it appears "running")
        pid_path = tmp_path / "daemon.pid"
        pid_path.write_text(str(os.getpid()))

        runner = CliRunner()
        result = runner.invoke(cli, ["daemon", "start"])
        assert result.exit_code != 0
        assert "already running" in result.output.lower() or "already running" in (result.stderr or "").lower()

    def test_daemon_stop_not_running(self, tmp_path, monkeypatch):
        """daemon stop when not running prints message."""
        monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "runtime"))
        monkeypatch.setenv("CLAWMETER_DATA_DIR", str(tmp_path / "data"))
        monkeypatch.setenv("CLAWMETER_CACHE_DIR", str(tmp_path / "cache"))

        config_path = tmp_path / "config.toml"
        config_path.write_text(
            f'[daemon]\npid_file = "{tmp_path}/daemon.pid"\n'
        )
        os.chmod(str(config_path), 0o600)
        monkeypatch.setenv("CLAWMETER_CONFIG", str(config_path))

        runner = CliRunner()
        result = runner.invoke(cli, ["daemon", "stop"])
        assert "not running" in result.output.lower()

    def test_daemon_status_stopped(self, tmp_path, monkeypatch):
        """daemon status reports stopped when no daemon running."""
        monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "runtime"))
        monkeypatch.setenv("CLAWMETER_DATA_DIR", str(tmp_path / "data"))
        monkeypatch.setenv("CLAWMETER_CACHE_DIR", str(tmp_path / "cache"))

        config_path = tmp_path / "config.toml"
        config_path.write_text(
            f'[daemon]\npid_file = "{tmp_path}/daemon.pid"\n'
        )
        os.chmod(str(config_path), 0o600)
        monkeypatch.setenv("CLAWMETER_CONFIG", str(config_path))

        runner = CliRunner()
        result = runner.invoke(cli, ["daemon", "status"])
        assert result.exit_code == 1
        assert "stopped" in result.output.lower()

    def test_daemon_status_quiet(self, tmp_path, monkeypatch):
        """daemon status --quiet exits 1 when stopped (for health checks)."""
        monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "runtime"))
        monkeypatch.setenv("CLAWMETER_DATA_DIR", str(tmp_path / "data"))
        monkeypatch.setenv("CLAWMETER_CACHE_DIR", str(tmp_path / "cache"))

        config_path = tmp_path / "config.toml"
        config_path.write_text(
            f'[daemon]\npid_file = "{tmp_path}/daemon.pid"\n'
        )
        os.chmod(str(config_path), 0o600)
        monkeypatch.setenv("CLAWMETER_CONFIG", str(config_path))

        runner = CliRunner()
        result = runner.invoke(cli, ["daemon", "status", "--quiet"])
        assert result.exit_code == 1
        assert result.output.strip() == ""  # Silent

    def test_daemon_status_running(self, tmp_path, monkeypatch):
        """daemon status reports running when daemon PID file exists with live PID."""
        monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "runtime"))
        monkeypatch.setenv("CLAWMETER_DATA_DIR", str(tmp_path / "data"))
        monkeypatch.setenv("CLAWMETER_CACHE_DIR", str(tmp_path / "cache"))

        config_path = tmp_path / "config.toml"
        config_path.write_text(
            f'[daemon]\npid_file = "{tmp_path}/daemon.pid"\n'
        )
        os.chmod(str(config_path), 0o600)
        monkeypatch.setenv("CLAWMETER_CONFIG", str(config_path))

        # Write PID file with current PID and a state file
        pid_path = tmp_path / "daemon.pid"
        pid_path.write_text(str(os.getpid()))

        state_path = tmp_path / "runtime" / "clawmeter" / "daemon.state"
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "started_at": datetime.now(timezone.utc).isoformat(),
            "pid": os.getpid(),
            "providers": {
                "claude": {
                    "last_poll": datetime.now(timezone.utc).isoformat(),
                    "next_poll": (datetime.now(timezone.utc) + timedelta(minutes=8)).isoformat(),
                    "status": "ok",
                },
            },
        }
        state_path.write_text(json.dumps(state))

        # Create a dummy history DB
        data_dir = tmp_path / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        store = HistoryStore(data_dir / "history.db")
        store.open()
        store.close()

        runner = CliRunner()
        result = runner.invoke(cli, ["daemon", "status"])
        assert result.exit_code == 0
        assert "running" in result.output.lower()
        assert str(os.getpid()) in result.output


# ===========================================================================
# CLI daemon detection integration
# ===========================================================================


class TestDaemonDetection:
    def _setup(self, tmp_path, monkeypatch):
        """Set up env with a fake daemon PID and history data."""
        monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "runtime"))
        monkeypatch.setenv("CLAWMETER_DATA_DIR", str(tmp_path / "data"))
        monkeypatch.setenv("CLAWMETER_CACHE_DIR", str(tmp_path / "cache"))

        config_path = tmp_path / "config.toml"
        config_path.write_text(
            f'[daemon]\npid_file = "{tmp_path}/daemon.pid"\n\n'
            f'[providers.claude]\nenabled = true\n'
            f'credentials_path = ""\n'
        )
        os.chmod(str(config_path), 0o600)
        monkeypatch.setenv("CLAWMETER_CONFIG", str(config_path))

        # Write PID file with current PID
        pid_path = tmp_path / "daemon.pid"
        pid_path.write_text(str(os.getpid()))

        # Seed history DB
        data_dir = tmp_path / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        store = HistoryStore(data_dir / "history.db")
        store.open()
        store.record(_make_status(
            provider="claude",
            utilisation=55.0,
            ts=datetime.now(timezone.utc) - timedelta(minutes=2),
        ))
        store.close()

    def test_reads_from_db_when_daemon_running(self, tmp_path, monkeypatch):
        """CLI detects running daemon and reads from DB instead of fetching."""
        self._setup(tmp_path, monkeypatch)

        runner = CliRunner()
        result = runner.invoke(cli, [])

        # Should succeed with data from DB
        assert result.exit_code == 0
        # Output should mention daemon (mixed into output)
        assert "reading from daemon" in result.output.lower()
        # Output should contain JSON data with the provider
        assert '"claude"' in result.output

    def test_fresh_bypasses_daemon(self, tmp_path, monkeypatch):
        """--fresh fetches directly even when daemon is running."""
        self._setup(tmp_path, monkeypatch)

        # With --fresh, the CLI should try to fetch directly.
        # Without proper credentials it will fail, but it should NOT
        # read from daemon (no "Reading from daemon" message).
        runner = CliRunner()
        result = runner.invoke(cli, ["--fresh"])

        # Should NOT contain daemon message
        assert "reading from daemon" not in result.output.lower()
