# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

clawmeter is a Linux-native CLI tool for monitoring LLM service usage, costs, and performance across multiple providers. It uses a pluggable provider architecture, launching with Anthropic Claude support and expanding to Grok (xAI), OpenAI, Ollama, and local system metrics. A GTK/GNOME desktop widget is planned for v2.

The tool operates in two modes: **standalone** (CLI fetches directly) and **daemon** (background service collects, CLI reads from DB). The daemon is recommended for continuous history collection.

The full specification lives in `docs/SPEC.md` - consult it for detailed provider specs, security requirements, CLI interface design, and open questions.

## Build & Development

```bash
# Setup (from source)
uv sync

# Run
uv run clawmeter

# Run tests
uv run pytest
uv run pytest tests/test_security.py           # single test file
uv run pytest tests/test_security.py::test_name # single test

# Build
uv build

# Docker
docker build -t clawmeter .
docker compose up -d
```

Build backend is `hatchling` + `hatch-vcs`. Version is derived from git tags (no VERSION file). Tag format: `v0.1.0`.

## Architecture

**Daemon + thin clients:** The daemon polls providers on a schedule, writes to a SQLite history database. CLI/TUI/GTK are thin readers of that database. Without the daemon, the CLI falls back to direct fetching (standalone mode).

**Provider abstraction** is the core pattern: every LLM service implements `Provider` (ABC) with `name()`, `display_name()`, `is_configured()`, `fetch_usage()`, and `auth_instructions()`. The base class handles credential resolution. Providers return a unified `ProviderStatus` dataclass containing `UsageWindow` and `ModelUsage` entries.

Key components and their locations under `src/clawmeter/`:

- `daemon.py` - Background service: poll loop, PID file management, systemd install
- `models.py` - `SecretStr`, `UsageWindow`, `ModelUsage`, `ProviderStatus` dataclasses
- `providers/base.py` - Abstract `Provider` class with `resolve_credential()` (4-tier: key_command > env var > keyring > provider-specific file). key_command failure raises `CredentialError` (no silent fallthrough). Provider registry via `@register_provider` decorator and module-level dict.
- `providers/claude.py` - Read-only consumer of `~/.claude/.credentials.json` (`claudeAiOauth.accessToken` + `expiresAt`). Does NOT refresh tokens or write to the file.
- `core.py` - Orchestrator: loads providers, aggregates results
- `cli.py` - Click/Typer CLI with modes: JSON (default, no flag), `--now` (table), `--monitor` (Rich Live TUI)
- `security.py` - Credential sanitisation, secure file I/O (`secure_write`, `secure_mkdir`), permission warnings
- `cache.py` - Per-provider JSON cache (standalone mode only), `poll_interval`-based TTL, `fcntl.flock()`
- `config.py` - TOML loader from `~/.config/clawmeter/config.toml`, permission warnings (not hard failures)
- `history.py` - SQLite at `~/.local/share/clawmeter/history.db` (WAL mode), retention pruning, reporting
- `formatters/` - JSON, Rich table (TTY-adaptive), Rich Live TUI

## Critical Design Decisions

- **stdout is data only, stderr is messaging only** - JSON/table output to stdout; errors, warnings, spinners to stderr. No exceptions.
- **JSON is the default output mode** (no flag needed). `--now` for table, `--monitor` for TUI.
- **Daemon decouples collection from presentation** - CLI/TUI/GTK read from the shared SQLite DB. Standalone mode still works without daemon.
- **No plaintext secrets in config files** - credentials resolved via keyring, env vars, or `key_command` (executed with `shell=False`).
- **All secrets wrapped in `SecretStr`** - `__repr__` always returns `SecretStr('***')` (never leaks real characters). Never log/serialize raw secrets. Failures raise `CredentialError` (defined in models.py).
- **Read-only consumer of Claude credentials** - the tool NEVER writes to `~/.claude/.credentials.json`. On token expiry, direct user to `claude /login`.
- **key_command failure is hard, not silent** - if explicitly configured and it fails/times out, raise `CredentialError`. Don't silently fall through to env vars.
- **Global poll_interval (10m default)** - one setting replaces the old `cache_ttl`/`refresh_interval` split. Per-provider override available. Local providers default to 60s.
- **Async concurrency** - `Provider.fetch_usage()` is async. CLI uses `asyncio.run()` + `asyncio.gather()` for concurrent provider fetches. Daemon uses a persistent asyncio event loop. `return_exceptions=True` for error isolation.
- **Rate-limit backoff** - exponential backoff on 429s (10m -> 20m -> 40m, cap 60m), persisted in cache file, resets on success.
- **Files created with `0o600`/dirs `0o700` from the start** - use `os.open()` + `os.fdopen()`, not `open()` then `chmod()`.
- **Atomic file writes** - write to `.tmp` then `os.rename()`.
- **Permission checks are warnings, not hard failures** - config contains no secrets (by design). Container mode skips checks entirely.
- **Provider errors are isolated** - one provider failing doesn't block others. Exit codes: 0=all ok, 1=config error, 2=all auth fail, 3=partial, 4=all network fail.
- **Australian English in docs/UI** ("utilisation", "colour") but US English for Python API identifiers where convention requires it.
- **Lightweight base install, additive extras** - base has no C extensions. `[local]` adds psutil/pynvml. `[gtk]` adds PyGObject. `[all]` is everything.
- **Env var overrides for all paths** - `CLAWMETER_CONFIG`, `CLAWMETER_DATA_DIR`, `CLAWMETER_CACHE_DIR`. Essential for Docker.

## Security Checklist

When touching credential or network code, verify:
- `SecretStr` wraps all secrets immediately on read — `__repr__` is always `SecretStr('***')`
- Redaction patterns applied to any user-visible output (see SPEC Section 7.3)
- `httpx` requests: `verify=True` (TLS), `follow_redirects=False` on auth-bearing requests
- `key_command`: `shell=False`, timeout=10s with `TimeoutExpired` handling, only log stderr on failure (never stdout)
- Credential-bearing requests only sent to provider's `allowed_hosts`
- No secrets in cache files, JSON output, `extras` dict, or error messages
- Never write to Claude's credential file — read-only consumer

## Tech Stack

Python 3.10+, httpx, rich, click/typer, keyring, tomllib/tomli, sqlite3, hatchling+hatch-vcs, uv, pytest+respx, Docker, systemd user units. Optional: psutil, pynvml (via `[local]` extra), PyGObject (via `[gtk]` extra).

## Release Milestones

v0.1.0: CLI MVP — Claude provider, standalone, JSON + table output, CI from day one
v0.2.0: History + reporting — SQLite store, meaningful-change detection, report/export commands
v0.3.0: Daemon + Docker — background collection, systemd, container deployment
v0.4.0: Monitor TUI — Rich Live dashboard, sparklines, compact mode
v0.5.0–v0.8.0: Providers — Grok, OpenAI, Ollama, Local system metrics
v0.9.0: Notifications and polish
v1.0.0: Stable release — security audit, schema freeze, fuzz testing
v2.0.0: GTK4/libadwaita desktop widget
