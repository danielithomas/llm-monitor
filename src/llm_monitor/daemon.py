"""Daemon mode for llm-monitor.

Background service that polls providers on a schedule and writes to the
history database.  See SPEC.md Sections 4.2.7 and 4.2.7.1 for the full
daemon specification.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llm_monitor.cache import ProviderCache
from llm_monitor.config import (
    get_cache_dir,
    get_log_dir,
    get_log_file,
    get_pid_dir,
    get_pid_file,
    get_state_file,
    load_config,
)
from llm_monitor.core import fetch_all
from llm_monitor.history import HistoryStore
from llm_monitor.providers import PROVIDERS, get_enabled_providers
from llm_monitor.security import secure_mkdir, secure_write

logger = logging.getLogger("llm_monitor.daemon")


# ======================================================================
# PID file management
# ======================================================================


def write_pid_file(path: Path) -> None:
    """Write the current process PID to *path*."""
    secure_mkdir(str(path.parent))
    secure_write(str(path), str(os.getpid()))


def read_pid_file(path: Path) -> int | None:
    """Read a PID from *path*. Returns None if missing or corrupt."""
    try:
        text = path.read_text().strip()
        return int(text)
    except (FileNotFoundError, ValueError, OSError):
        return None


def remove_pid_file(path: Path) -> None:
    """Remove the PID file if it exists."""
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def is_daemon_running(config: dict) -> tuple[bool, int | None]:
    """Check whether a daemon is currently running.

    Returns ``(True, pid)`` if a live process owns the PID file, or
    ``(False, None)`` otherwise.  Stale PID files are cleaned up.
    """
    pid_path = get_pid_file(config)
    pid = read_pid_file(pid_path)
    if pid is None:
        return False, None

    try:
        os.kill(pid, 0)  # signal 0 = existence check
        return True, pid
    except ProcessLookupError:
        # Process is dead — stale PID file
        remove_pid_file(pid_path)
        return False, None
    except PermissionError:
        # Process exists but we can't signal it (different user)
        return True, pid


# ======================================================================
# State file management
# ======================================================================


def write_state(path: Path, state: dict[str, Any]) -> None:
    """Write daemon state as JSON."""
    secure_mkdir(str(path.parent))
    secure_write(str(path), json.dumps(state, indent=2))


def read_state(path: Path) -> dict[str, Any] | None:
    """Read daemon state from JSON. Returns None if missing or corrupt."""
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


# ======================================================================
# DaemonRunner
# ======================================================================


class DaemonRunner:
    """Manages the daemon poll loop lifecycle."""

    def __init__(self, config: dict, foreground: bool = False) -> None:
        self._config = config
        self._foreground = foreground
        self._shutdown = False
        self._reload = False
        self._wake_event: asyncio.Event | None = None
        self._history: HistoryStore | None = None
        self._providers: list[Any] = []  # Provider instances
        self._provider_intervals: dict[str, int] = {}  # name -> seconds
        self._next_poll: dict[str, float] = {}  # name -> monotonic time
        self._started_at: str = ""
        self._provider_status: dict[str, dict[str, str]] = {}

    def run(self) -> None:
        """Main entry point — set up and enter the poll loop."""
        self._setup_logging()
        pid_path = get_pid_file(self._config)
        state_path = get_state_file(self._config)

        write_pid_file(pid_path)
        self._started_at = datetime.now(timezone.utc).isoformat()
        logger.info("Daemon starting (PID %d)", os.getpid())

        try:
            self._init_providers()
            self._history = HistoryStore()
            self._history.open()

            # Retention pruning on startup
            retention = self._config.get("history", {}).get("retention_days", 90)
            pruned = self._history.prune(retention)
            if pruned:
                logger.info("Pruned %d old history records", pruned)

            asyncio.run(self._poll_loop())
        except KeyboardInterrupt:
            logger.info("Interrupted")
        except Exception:
            logger.exception("Daemon crashed")
            raise
        finally:
            self._cleanup(pid_path, state_path)

    def _setup_logging(self) -> None:
        """Configure logging for foreground or background mode."""
        log_level_str = os.environ.get("LLM_MONITOR_LOG_LEVEL", "info").upper()
        log_level = getattr(logging, log_level_str, logging.INFO)
        fmt = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s")

        root = logging.getLogger("llm_monitor")
        root.setLevel(log_level)
        # Remove any existing handlers
        root.handlers.clear()

        if self._foreground:
            handler: logging.Handler = logging.StreamHandler(sys.stderr)
        else:
            log_path = get_log_file(self._config)
            secure_mkdir(str(log_path.parent))
            handler = logging.FileHandler(str(log_path))

        handler.setFormatter(fmt)
        root.addHandler(handler)

    def _init_providers(self) -> None:
        """Instantiate enabled providers and record their poll intervals."""
        global_interval = self._config.get("general", {}).get("poll_interval", 600)
        provider_classes = get_enabled_providers(self._config)

        self._providers = []
        self._provider_intervals = {}
        self._next_poll = {}

        for cls in provider_classes:
            try:
                instance = cls(self._config)
                self._providers.append(instance)
                name = instance.name()

                # Per-provider poll_interval override
                provider_cfg = self._config.get("providers", {}).get(name, {})
                interval = provider_cfg.get("poll_interval", global_interval)
                self._provider_intervals[name] = interval
                # Schedule immediate first poll
                self._next_poll[name] = time.monotonic()

                logger.info(
                    "Provider %s: poll_interval=%ds", name, interval
                )
            except Exception:
                logger.exception("Failed to initialise provider %s", cls.__name__)

        if not self._providers:
            logger.error("No providers available — nothing to poll")

    def _get_poll_interval(self, provider_name: str) -> int:
        """Return the poll interval for a provider."""
        return self._provider_intervals.get(provider_name, 600)

    async def _poll_loop(self) -> None:
        """Core asyncio poll loop."""
        loop = asyncio.get_running_loop()
        self._wake_event = asyncio.Event()

        # Register signal handlers
        loop.add_signal_handler(signal.SIGTERM, self._handle_sigterm)
        loop.add_signal_handler(signal.SIGINT, self._handle_sigterm)
        loop.add_signal_handler(signal.SIGHUP, self._handle_sighup)

        while not self._shutdown:
            if self._reload:
                self._reload = False
                self._reload_config()

            # Find providers due for polling
            now = time.monotonic()
            due_providers = [
                p for p in self._providers
                if self._next_poll.get(p.name(), 0) <= now
            ]

            if due_providers:
                await self._poll_providers(due_providers)

            # Write state file
            self._write_state()

            if self._shutdown:
                break

            # Sleep until the next provider is due
            if self._next_poll:
                next_time = min(self._next_poll.values())
                sleep_secs = max(0, next_time - time.monotonic())
            else:
                sleep_secs = 60  # No providers — check again in a minute

            self._wake_event.clear()
            try:
                await asyncio.wait_for(
                    self._wake_event.wait(), timeout=sleep_secs
                )
            except asyncio.TimeoutError:
                pass  # Normal — sleep expired

        logger.info("Poll loop exiting")

    async def _poll_providers(self, providers: list[Any]) -> None:
        """Fetch from the given providers and record to history."""
        names = [p.name() for p in providers]
        logger.info("Polling: %s", ", ".join(names))

        cache = ProviderCache(get_cache_dir())
        statuses = await fetch_all(providers, cache, self._config, fresh=True)

        for status in statuses:
            name = status.provider_name
            if status.errors:
                logger.warning(
                    "Provider %s errors: %s", name, "; ".join(status.errors)
                )
                self._provider_status[name] = {
                    "last_poll": datetime.now(timezone.utc).isoformat(),
                    "status": "error",
                }
            else:
                logger.info("Provider %s: ok", name)
                self._provider_status[name] = {
                    "last_poll": datetime.now(timezone.utc).isoformat(),
                    "status": "ok",
                }

            # Record to history
            if self._history is not None:
                try:
                    self._history.record(status)
                except Exception:
                    logger.exception("Failed to record history for %s", name)

            # Update next poll time
            interval = self._get_poll_interval(name)
            self._next_poll[name] = time.monotonic() + interval

    def _write_state(self) -> None:
        """Write the daemon state file for `daemon status` to read."""
        state_path = get_state_file(self._config)
        providers_state: dict[str, Any] = {}
        now_mono = time.monotonic()

        for p in self._providers:
            name = p.name()
            pstate = self._provider_status.get(name, {})
            next_poll_mono = self._next_poll.get(name, now_mono)
            secs_until = max(0, int(next_poll_mono - now_mono))
            next_poll_utc = datetime.now(timezone.utc).timestamp() + secs_until
            providers_state[name] = {
                "last_poll": pstate.get("last_poll", ""),
                "next_poll": datetime.fromtimestamp(
                    next_poll_utc, tz=timezone.utc
                ).isoformat(),
                "status": pstate.get("status", "pending"),
            }

        state = {
            "started_at": self._started_at,
            "pid": os.getpid(),
            "providers": providers_state,
        }
        try:
            write_state(state_path, state)
        except Exception:
            logger.debug("Failed to write state file", exc_info=True)

    def _handle_sigterm(self) -> None:
        """SIGTERM/SIGINT handler — set shutdown flag and wake the loop."""
        logger.info("Received shutdown signal")
        self._shutdown = True
        if self._wake_event is not None:
            self._wake_event.set()

    def _handle_sighup(self) -> None:
        """SIGHUP handler — set reload flag and wake the loop."""
        logger.info("Received SIGHUP — will reload config")
        self._reload = True
        if self._wake_event is not None:
            self._wake_event.set()

    def _reload_config(self) -> None:
        """Reload configuration and re-initialise providers."""
        logger.info("Reloading configuration")
        try:
            self._config = load_config()
            self._init_providers()
            logger.info("Configuration reloaded successfully")
        except Exception:
            logger.exception("Failed to reload config — keeping old config")

    def _cleanup(self, pid_path: Path, state_path: Path) -> None:
        """Close resources and remove ephemeral files."""
        if self._history is not None:
            try:
                self._history.close()
            except Exception:
                logger.debug("Error closing history DB", exc_info=True)

        remove_pid_file(pid_path)
        try:
            state_path.unlink()
        except FileNotFoundError:
            pass

        logger.info("Daemon stopped")


# ======================================================================
# Daemonise (double-fork)
# ======================================================================


def daemonise(config: dict) -> int:
    """Fork to background using the POSIX double-fork pattern.

    Returns the child PID to the parent process.  The child process
    calls ``DaemonRunner(config).run()`` and never returns.
    """
    # First fork
    pid = os.fork()
    if pid > 0:
        # Parent waits briefly for the grandchild PID file
        return pid

    # First child — create new session
    os.setsid()

    # Second fork — detach from session leader
    pid2 = os.fork()
    if pid2 > 0:
        os._exit(0)

    # Grandchild — this is the actual daemon process
    # Redirect stdio to /dev/null
    devnull_fd = os.open(os.devnull, os.O_RDWR)
    os.dup2(devnull_fd, sys.stdin.fileno())
    os.dup2(devnull_fd, sys.stdout.fileno())
    os.dup2(devnull_fd, sys.stderr.fileno())
    if devnull_fd > 2:
        os.close(devnull_fd)

    # Run the daemon
    runner = DaemonRunner(config, foreground=False)
    try:
        runner.run()
    finally:
        os._exit(0)
