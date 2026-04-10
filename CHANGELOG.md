# Changelog

All notable changes to llm-monitor are documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.7.0] - 2026-04-10

### Added

- **Ollama provider** (`providers/ollama.py`) — local instance monitoring with model inventory, loaded model state, VRAM/RAM usage reporting
- Multi-host support: single `host` config or `[[providers.ollama.hosts]]` array for monitoring multiple Ollama instances across a network
- Per-host polling via `GET /api/tags` (model inventory + health) and `GET /api/ps` (loaded models + memory allocation)
- Cloud model detection via `:cloud` tag suffix in model names (labelling only)
- Error isolation per host — one host being unreachable does not affect others
- **Alpha features framework** (D-053) — `enable_alpha_features` config flag gates unstable data sources with stderr warning, `alpha: true` labelling in extras, and graceful failure
- Ollama Cloud session/weekly usage windows (alpha) — probes `ollama.com/api/account/usage` when `cloud_enabled = true` and alpha flag set
- Cloud API key authentication via credential chain (`api_key_command` > `api_key_env`/`$OLLAMA_API_KEY` > keyring)
- `is_alpha_enabled()` config helper for providers to check alpha flag
- `[providers.ollama]` config section with `host`/`hosts` mutual exclusivity validation
- Research report: `docs/research/ollama-v0.7.0-research.md`

### Fixed

- Table and monitor formatters now render count and MB values (e.g. "3", "5,086 MB") instead of showing "0%" for non-percentage usage windows

## [0.6.0] - 2026-04-10

### Added

- **OpenAI provider** (`providers/openai.py`) — organisation-level spend and per-model usage monitoring via the Administration API
- Usage API: `GET /v1/organization/usage/completions` with `group_by=model` for per-model token counts (input, output, cached, requests)
- Costs API: `GET /v1/organization/costs` with `group_by=line_item` for per-model cost in USD and total MTD spend
- Admin key credential resolution via `$OPENAI_ADMIN_KEY` (`sk-admin-*`) with 4-tier chain
- Token and cost merge into unified `ModelUsage` entries with `top_model_spend` extras
- Config section for OpenAI (`providers.openai` with `admin_key_env`)

### Changed

- SPEC.md Section 3.3 rewritten with correct admin key auth, response schemas, and query parameters
- OQ-012 closed: undocumented billing endpoints confirmed dead, admin key required
- A-006 falsified: standard API keys cannot access org usage endpoints

## [0.5.0] - 2026-04-09

### Added

- **xAI Grok provider** (`providers/grok.py`) — spend monitoring via the xAI Management API
- Invoice preview endpoint for month-to-date spend with per-model cost breakdown
- Spending limits endpoint with utilisation percentage and threshold detection
- Prepaid balance endpoint
- Usage analytics endpoint for per-model time-series token and spend data
- Dual credential support: management key (primary, 4-tier resolution) and optional API key
- Team ID resolution from config (`providers.grok.team_id`) or `$XAI_TEAM_ID` env var
- Redaction pattern for `xai-*` keys in security module
- Config section for Grok provider
- Test fixtures for all four Management API responses

### Fixed

- Table and monitor formatters now render USD-based usage windows as dollar values instead of percentages

## [0.4.0] - 2026-04-08

### Added

- **Monitor TUI** (`--monitor` flag) — Rich Live dashboard with auto-refreshing provider status
- Per-provider status panels with usage bars, model breakdowns, and sparklines
- Compact mode for smaller terminals
- Configurable refresh interval
- Colour-coded threshold indicators (normal/warning/critical)

## [0.3.0] - 2026-04-07

### Added

- **Background daemon** (`llm-monitor daemon start/stop/status/install`) — polls providers on a schedule and writes to the shared SQLite history database
- PID file management with stale PID detection
- `daemon install` generates a systemd user unit for automatic startup
- CLI reads from daemon's database when available, falls back to direct fetching
- **Docker deployment** — `Dockerfile` and `docker-compose.yml` for containerised daemon operation
- Container mode detection (skips permission checks, uses env var paths)
- Per-provider poll interval override in config
- Daemon-aware provider base class with backoff state persistence

## [0.2.0] - 2026-04-07

### Added

- **SQLite history store** (`history.py`) — records provider snapshots with WAL mode and retention pruning
- `llm-monitor history` — query historical usage data with time range filters
- `llm-monitor report` — generate usage summaries (daily/weekly/monthly) with trend analysis
- `llm-monitor export` — export history to CSV or JSON
- Meaningful-change detection to avoid storing duplicate snapshots
- Per-model usage tracking in dedicated `model_usage` table

## [0.1.0] - 2026-04-06

### Added

- **Anthropic Claude provider** — subscription utilisation tracking via Claude Code OAuth credentials
- Read-only consumer of `~/.claude/.credentials.json` (never writes to the file)
- Five usage windows: Session (5h), Daily, Weekly, Monthly, and Opus-specific
- Per-model usage breakdown (Opus, Sonnet, Haiku token counts)
- **CLI with three output modes**: JSON (default, no flag), `--now` (Rich table), `--monitor` (TUI, added in v0.4.0)
- **Provider architecture** — abstract `Provider` base class with `@register_provider` decorator
- 4-tier credential resolution: `key_command` > env var > keyring > provider-specific file
- `SecretStr` wrapper with safe `__repr__` — secrets never leak in logs or output
- **JSON cache** with `poll_interval`-based TTL and `fcntl.flock()` file locking
- **TOML configuration** from `~/.config/llm-monitor/config.toml` with env var path overrides
- **Security module** — credential sanitisation, secure file I/O (`0o600`/`0o700`), atomic writes
- Allowed-hosts enforcement for credential-bearing HTTP requests
- `key_command` execution with `shell=False` and 10s timeout
- Rich table formatter with TTY-adaptive column widths and colour-coded thresholds
- JSON formatter with stable schema (stdout-only, stderr for messaging)
- Exit codes: 0 (ok), 1 (config error), 2 (all auth fail), 3 (partial), 4 (all network fail)
- CI pipeline via GitHub Actions (lint, test, type check)
- Full test suite: providers, formatters, cache, config, security, CLI, models
