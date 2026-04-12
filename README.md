# clawmeter

Monitor your LLM consumption from local and online services.

Currently supports **Anthropic Claude** (subscription utilisation tracking), **xAI Grok** (spend monitoring, spending limits, prepaid balance), **OpenAI** (API spend and per-model usage via Admin API), and **Ollama** (local instance monitoring — loaded models, VRAM/RAM usage, multi-host support; cloud usage tracking as an alpha feature). Future versions will add local system metrics.

## Quick Start

```bash
# Install
uv tool install clawmeter
# or: pip install clawmeter

# Prerequisites: Claude Code must be installed and authenticated
claude /login

# Check your Claude usage (JSON output, default)
clawmeter

# Human-readable table
clawmeter --now
```

### Example JSON Output

```json
{
  "timestamp": "2026-04-07T08:50:06+10:00",
  "version": "0.1.0",
  "providers": [
    {
      "provider_name": "claude",
      "provider_display": "Anthropic Claude",
      "timestamp": "2026-04-06T22:50:05+00:00",
      "cached": false,
      "cache_age_seconds": 0,
      "windows": [
        {
          "name": "Session (5h)",
          "utilisation": 42.0,
          "resets_at": "2026-04-07T03:00:00+00:00",
          "resets_in_human": "2h 15m",
          "status": "normal",
          "unit": "percent",
          "raw_value": null,
          "raw_limit": null
        },
        {
          "name": "Weekly (7d)",
          "utilisation": 68.0,
          "resets_at": "2026-04-10T00:00:00+00:00",
          "resets_in_human": "2d 13h",
          "status": "warning",
          "unit": "percent",
          "raw_value": null,
          "raw_limit": null
        }
      ],
      "extras": {},
      "errors": []
    }
  ]
}
```

### Example Table Output

```
LLM Monitor                              07 Apr 2026, 10:30 AEST
 Anthropic Claude                                        fresh
   Session (5h)    ████████░░░░░░░░░░░░  42%    resets in 2h 15m
   Weekly (7d)     █████████████░░░░░░░  68%    resets in 2d 13h
```

## Installation

```bash
# Via uv (recommended)
uv tool install clawmeter

# Via pip
pip install clawmeter --user
```

### Running from Source

```bash
git clone https://github.com/danielithomas/clawmeter.git
cd clawmeter
uv sync --group dev

# Run directly
uv run clawmeter
uv run clawmeter --now
uv run clawmeter history stats

# Run tests
uv run pytest
uv run pytest tests/test_history.py -v    # single test file

# Build a wheel
uv build
```

## Prerequisites

### Claude

The Claude provider reads OAuth credentials managed by Claude Code:

1. Install [Claude Code](https://claude.ai/code)
2. Run `claude /login` to authenticate
3. The tool reads `~/.claude/.credentials.json` (read-only, never writes to it)

### Grok (xAI)

The Grok provider uses the xAI Management API for spend and usage monitoring:

1. Go to [console.x.ai](https://console.x.ai) and create a **Management Key**
2. Set `$XAI_MANAGEMENT_KEY` (or configure `management_key_command` in config)
3. Set `team_id` in `[providers.grok]` config (or `$XAI_TEAM_ID` env var)
4. Enable the provider: set `enabled = true` in `[providers.grok]`

The Management Key is separate from a regular xAI API key. It provides access to billing data, spending limits, and usage analytics. An optional `$XAI_API_KEY` can be set for rate limit header data but is not required.

**Monitored data:**
- **Spend (MTD)** — month-to-date spend in the current billing cycle
- **Spend vs Limit** — percentage of hard spending limit consumed
- **Prepaid Balance** — remaining prepaid credits
- **Per-model breakdown** — token counts and costs per model (grok-3, grok-3-mini, etc.)

### OpenAI

The OpenAI provider uses the Administration API, which requires an **Admin API Key** (`sk-admin-*`):

1. Go to [platform.openai.com](https://platform.openai.com) → Settings → Organisation → Admin Keys
2. Create an admin key (only Organisation Owners can do this)
3. Set the environment variable: `export OPENAI_ADMIN_KEY="sk-admin-..."`
4. Enable in config: set `[providers.openai] enabled = true`

Standard project keys (`sk-proj-*`) do **not** have access to the Usage or Costs APIs.

### Ollama

The Ollama provider monitors local (and network) Ollama instances. **No credentials are required** for local monitoring.

1. Install [Ollama](https://ollama.com/download) and start it: `ollama serve`
2. Enable in config: set `[providers.ollama] enabled = true`

Ollama is polled every 60 seconds by default (configurable via `poll_interval`).

**Monitored data:**
- **Models Available** — total downloaded models per host
- **Models Loaded** — models currently in memory
- **VRAM Usage** — GPU memory allocated to loaded models
- **RAM Usage** — system RAM allocated to loaded models
- **Cloud model detection** — models with `:cloud` suffix are identified

**Multi-host monitoring:** Monitor multiple Ollama instances across your network:

```toml
[providers.ollama]
enabled = true

[[providers.ollama.hosts]]
name = "workstation"
url = "http://localhost:11434"

[[providers.ollama.hosts]]
name = "gpu-server"
url = "http://gpu-server.local:11434"
```

## Alpha Features

Some monitoring data is only available via undocumented or unstable interfaces (web scraping, unversioned API endpoints). These features are gated behind a global opt-in flag and may break between releases without notice.

To enable:

```toml
[general]
enable_alpha_features = true
```

When alpha features are active, the tool emits a one-time warning to stderr per session. Alpha-sourced data is flagged in JSON output (`"alpha": true` in the extras dict).

### Ollama Cloud Usage (v0.7.0)

Tracks session and weekly quota consumption for Ollama Cloud subscribers. Requires an Ollama account with a cloud plan (Free, Pro, or Max).

**Setup:**

1. Sign in to your Ollama account: `ollama signin`
2. Create an API key at [ollama.com/settings/keys](https://ollama.com/settings/keys)
3. Set the environment variable: `export OLLAMA_API_KEY="your_key"`
4. Enable in config:

```toml
[general]
enable_alpha_features = true

[providers.ollama]
enabled = true
host = "http://localhost:11434"    # local instance (always works)
cloud_enabled = true               # enable cloud usage tracking (alpha)
api_key_env = "OLLAMA_API_KEY"     # default, can be omitted
# api_key_command = "pass show clawmeter/ollama-cloud"
```

**Why alpha?** Ollama does not yet offer a programmatic API for cloud usage data ([ollama/ollama#12532](https://github.com/ollama/ollama/issues/12532)). This feature probes for an expected endpoint and falls back to scraping. It will graduate to stable when Ollama ships an official usage API.

**What it monitors:**
- Session usage (% consumed, resets every 5 hours)
- Weekly usage (% consumed, resets every 7 days)
- Plan type (Free / Pro / Max)

Local instance monitoring (models loaded, VRAM, health) is **not** an alpha feature — it uses stable, documented APIs and works without `enable_alpha_features`.

### Claude Extra Usage Spend (v0.7.1)

Tracks dollar spend for Claude usage beyond the subscription cap. No additional credentials needed — uses the existing Claude Code OAuth token. Enable alpha features in your config:

```toml
[general]
enable_alpha_features = true
```

**What it monitors:**
- Extra usage spend vs monthly limit (percentage + dollar amounts)
- Whether extra usage is enabled on your account

**Why alpha?** The extra usage data comes from an undocumented field in the Claude usage API. It could change without notice. It will graduate to stable when Anthropic formally documents the endpoint.

## Configuration

Config file location: `~/.config/clawmeter/config.toml`

The tool works with zero configuration — Claude is enabled by default. Create a config file to customise behaviour:

```toml
[general]
default_providers = ["claude"]
poll_interval = 600              # 10 minutes
# enable_alpha_features = false  # opt-in to unstable data sources (see Alpha Features)

[thresholds]
warning = 70
critical = 90

[providers.claude]
enabled = true
credentials_path = ""            # empty = default (~/.claude/.credentials.json)
show_opus = true

[providers.grok]
enabled = false                  # enable when credentials are configured
team_id = ""                     # required: xAI team ID (or set $XAI_TEAM_ID)
management_key_env = "XAI_MANAGEMENT_KEY"
# management_key_command = "secret-tool lookup application clawmeter provider grok-management"
# key_env = "XAI_API_KEY"       # optional: for rate limit data

[providers.openai]
enabled = false
admin_key_env = "OPENAI_ADMIN_KEY"       # Admin key (sk-admin-*), NOT project key
# admin_key_command = "pass show clawmeter/openai-admin"

[providers.ollama]
enabled = false
poll_interval = 60               # local service, can poll more frequently
host = "http://localhost:11434"
# cloud_enabled = false          # requires enable_alpha_features = true
# api_key_env = "OLLAMA_API_KEY"

[history]
enabled = true
retention_days = 90
```

Override paths via environment variables:

| Variable | Overrides |
|----------|-----------|
| `CLAWMETER_CONFIG` | Config file path |
| `CLAWMETER_DATA_DIR` | Data directory |
| `CLAWMETER_CACHE_DIR` | Cache directory |

## Credential Setup

Credentials are **never stored in the config file**. Resolution order:

1. **`key_command`** — execute a shell command (e.g., `pass show clawmeter/openai-admin`)
2. **Environment variable** — e.g., `$OPENAI_ADMIN_KEY`, `$XAI_API_KEY`
3. **System keyring** — GNOME Keyring, KDE Wallet, KeePassXC via Python `keyring`
4. **Provider credential file** — Claude only (`~/.claude/.credentials.json`)

## CLI Flags

| Flag | Short | Description |
|------|-------|-------------|
| `--now` | | Human-readable table output |
| `--monitor` | | Launch persistent Rich Live TUI |
| `--compact` | | Single-line per provider (`--monitor` only) |
| `--interval` | `-i` | UI refresh interval in seconds (`--monitor` only, default 30) |
| `--provider` | `-p` | Filter providers (comma-separated) |
| `--fresh` | `-f` | Bypass cache, force API call |
| `--verbose` | `-v` | Verbose logging to stderr |
| `--quiet` | `-q` | Suppress non-error stderr output |
| `--version` | `-V` | Print version |
| `--config` | `-c` | Override config file path |
| `--clear-cache` | | Delete cached data and exit |
| `--list-providers` | | Show available providers |
| `--no-colour` | | Disable colour output |
| `--colour=always` | | Force colour even when piped |
| `--no-history` | | Disable history recording for this invocation |
| `--report` | | Display usage report (alias for `history report`) |
| `--help` | `-h` | Show help |

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Configuration error |
| 2 | Authentication error |
| 3 | Partial success (some providers failed) |
| 4 | All providers unreachable |

## History and Reporting

Usage data is recorded to a local SQLite database on every fetch where data changes meaningfully. This enables trend analysis and historical reporting.

**Data location:** `~/.local/share/clawmeter/history.db`

### Configuration

```toml
[history]
enabled = true           # Set to false to disable history collection
retention_days = 90      # How long to keep history (default: 90 days)
```

Use `--no-history` to skip recording for a single invocation.

### Reports

```bash
# Table summary of last 7 days
clawmeter --report

# JSON report for the last 30 days, Claude only
clawmeter --report --days 30 --provider claude --format json

# CSV report with daily granularity
clawmeter --report --days 14 --format csv --granularity daily

# Include per-model token breakdown
clawmeter history report --days 7 --models

# Hourly granularity
clawmeter history report --granularity hourly --days 3
```

**Report flags:** `--days`, `--from`, `--to`, `--format` (table/json/csv), `--granularity` (raw/hourly/daily), `--provider`, `--window`, `--models`

### History Commands

```bash
# Database summary
clawmeter history stats

# Export for backup
clawmeter history export --format jsonl > backup.jsonl
clawmeter history export --format csv > backup.csv
clawmeter history export --format sql > backup.sql

# Purge all history (requires confirmation)
clawmeter history purge
clawmeter history purge --confirm   # non-interactive
```

## Daemon Mode

The daemon runs in the background, continuously polling providers and writing to the history database. When a daemon is running, CLI commands automatically read from the database instead of fetching directly.

### Usage

```bash
# Start background daemon
clawmeter daemon start

# Run in foreground (for systemd/Docker)
clawmeter daemon run

# Check status
clawmeter daemon status

# Stop the daemon
clawmeter daemon stop

# Install as systemd user service (auto-start on login)
clawmeter daemon install

# Remove systemd service
clawmeter daemon uninstall
```

When the daemon is running, normal CLI commands read from the database:

```bash
clawmeter --now          # reads from daemon's DB
clawmeter --report       # reports from daemon's history
clawmeter --fresh        # bypass daemon, fetch directly
```

### Configuration

```toml
[general]
poll_interval = 600          # global poll interval (10 minutes)

[daemon]
log_file = ""                # empty = default (~/.local/state/clawmeter/daemon.log)
pid_file = ""                # empty = default ($XDG_RUNTIME_DIR/clawmeter/daemon.pid)

[providers.ollama]
poll_interval = 60           # per-provider override (local services can poll faster)
```

**PID file:** `$XDG_RUNTIME_DIR/clawmeter/daemon.pid` (or `/tmp/clawmeter-<uid>/daemon.pid`)
**Log file:** `$XDG_STATE_HOME/clawmeter/daemon.log` (or `~/.local/state/clawmeter/daemon.log`)

## Monitor Mode

Launch a persistent Rich Live dashboard that auto-refreshes and displays all configured providers.

```bash
# Full dashboard
clawmeter --monitor

# Monitor a specific provider
clawmeter --monitor --provider claude

# Compact mode (single line per provider, ideal for tmux)
clawmeter --monitor --compact

# Custom refresh interval (default 30s, minimum 5s)
clawmeter --monitor --interval 10
```

When the daemon is running, `--monitor` reads from the history database (no API calls). Without a daemon, it fetches directly from providers on each refresh cycle.

### Key Bindings

| Key | Action |
|-----|--------|
| `r` | Force refresh all providers |
| `1-9` | Force refresh provider by index |
| `q` | Quit |
| `j` | Dump current state as JSON to file |
| `?` | Show/dismiss help overlay |

### Display Features

- Progress bars with status colour transitions (green/yellow/red/magenta)
- Live countdown timers for usage window resets
- Sparkline history (last 24 hours, hourly resolution)
- Provider health indicators: green `●` = healthy, yellow `●` = stale, red `●` = error
- Daemon status indicator (running/standalone, last poll time)

### Signals

| Signal | Action |
|--------|--------|
| `SIGUSR1` | Force refresh all providers |
| `SIGHUP` | Reload configuration |
| `SIGINT`/`SIGTERM` | Clean shutdown |

### Configuration

```toml
[monitor]
compact = false           # default to compact mode
show_sparkline = true     # show usage sparklines from history
```

## Docker

The daemon mode maps naturally to Docker. The container runs `clawmeter daemon run` in the foreground.

### Quick Start

```bash
docker build -t clawmeter .
docker compose up -d
```

### docker-compose.yml

```yaml
services:
  clawmeter:
    build: .
    restart: unless-stopped
    volumes:
      - clawmeter-data:/data
      - ${HOME}/.config/clawmeter/config.toml:/home/monitor/.config/clawmeter/config.toml:ro
      - ${HOME}/.claude/.credentials.json:/home/monitor/.claude/.credentials.json:ro
    environment:
      - OPENAI_ADMIN_KEY=${OPENAI_ADMIN_KEY}
      - XAI_MANAGEMENT_KEY=${XAI_MANAGEMENT_KEY}
      - XAI_TEAM_ID=${XAI_TEAM_ID}
      - XAI_API_KEY=${XAI_API_KEY}

volumes:
  clawmeter-data:
```

### Container-Aware Mode

When `$CLAWMETER_CONTAINER=1` is set (or `/.dockerenv` exists):

- Permission checks are skipped
- Keyring is not attempted (no D-Bus in containers)
- `daemon install`/`uninstall` are disabled (use `daemon run` directly)

### Accessing Data from Host

```bash
# Point host CLI at the container's database
export CLAWMETER_DATA_DIR=/path/to/docker/volume
clawmeter --now
clawmeter --report --days 7

# Or use docker exec
docker exec clawmeter clawmeter --now
```

## Scripting Examples

```bash
# Get Claude session utilisation as a number
clawmeter | jq -r '.providers[0].windows[0].utilisation'

# Alert when usage is high
USAGE=$(clawmeter | jq -r '.providers[0].windows[0].utilisation')
if (( $(echo "$USAGE > 80" | bc -l) )); then
    notify-send "Claude usage high: ${USAGE}%"
fi

# Pipe to other tools
clawmeter | jq '.providers[].windows[] | {name, utilisation, status}'
```

## Security

- All credentials wrapped in `SecretStr` — never appear in logs, output, or cache files
- Config file permission warnings when more permissive than `0o600`
- TLS verification enforced on all HTTPS connections
- No redirect following on credential-bearing requests
- `key_command` executed with `shell=False` (no shell injection)

## Development

See [Running from Source](#running-from-source) for setup. Additional commands:

```bash
uv run pytest -v                          # verbose test output
uv run pytest tests/test_security.py      # single file
uv run pytest -k "test_purge"             # run tests matching pattern
```

## Licence

MIT
