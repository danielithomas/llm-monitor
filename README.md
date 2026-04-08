# llm-monitor

Monitor your LLM consumption from local and online services.

Currently supports **Anthropic Claude** (subscription utilisation tracking) and **xAI Grok** (spend monitoring, spending limits, prepaid balance). Future versions will add OpenAI, Ollama, and local system metrics.

## Quick Start

```bash
# Install
uv tool install llm-monitor
# or: pip install llm-monitor

# Prerequisites: Claude Code must be installed and authenticated
claude /login

# Check your Claude usage (JSON output, default)
llm-monitor

# Human-readable table
llm-monitor --now
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
uv tool install llm-monitor

# Via pip
pip install llm-monitor --user
```

### Running from Source

```bash
git clone https://github.com/danielithomas/llm-monitor.git
cd llm-monitor
uv sync --group dev

# Run directly
uv run llm-monitor
uv run llm-monitor --now
uv run llm-monitor history stats

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

## Configuration

Config file location: `~/.config/llm-monitor/config.toml`

The tool works with zero configuration — Claude is enabled by default. Create a config file to customise behaviour:

```toml
[general]
default_providers = ["claude"]
poll_interval = 600              # 10 minutes

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
# management_key_command = "secret-tool lookup application llm-monitor provider grok-management"
# key_env = "XAI_API_KEY"       # optional: for rate limit data

[history]
enabled = true
retention_days = 90
```

Override paths via environment variables:

| Variable | Overrides |
|----------|-----------|
| `LLM_MONITOR_CONFIG` | Config file path |
| `LLM_MONITOR_DATA_DIR` | Data directory |
| `LLM_MONITOR_CACHE_DIR` | Cache directory |

## Credential Setup

Credentials are **never stored in the config file**. Resolution order:

1. **`key_command`** — execute a shell command (e.g., `pass show llm-monitor/openai`)
2. **Environment variable** — e.g., `$OPENAI_API_KEY`, `$XAI_API_KEY`
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

**Data location:** `~/.local/share/llm-monitor/history.db`

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
llm-monitor --report

# JSON report for the last 30 days, Claude only
llm-monitor --report --days 30 --provider claude --format json

# CSV report with daily granularity
llm-monitor --report --days 14 --format csv --granularity daily

# Include per-model token breakdown
llm-monitor history report --days 7 --models

# Hourly granularity
llm-monitor history report --granularity hourly --days 3
```

**Report flags:** `--days`, `--from`, `--to`, `--format` (table/json/csv), `--granularity` (raw/hourly/daily), `--provider`, `--window`, `--models`

### History Commands

```bash
# Database summary
llm-monitor history stats

# Export for backup
llm-monitor history export --format jsonl > backup.jsonl
llm-monitor history export --format csv > backup.csv
llm-monitor history export --format sql > backup.sql

# Purge all history (requires confirmation)
llm-monitor history purge
llm-monitor history purge --confirm   # non-interactive
```

## Daemon Mode

The daemon runs in the background, continuously polling providers and writing to the history database. When a daemon is running, CLI commands automatically read from the database instead of fetching directly.

### Usage

```bash
# Start background daemon
llm-monitor daemon start

# Run in foreground (for systemd/Docker)
llm-monitor daemon run

# Check status
llm-monitor daemon status

# Stop the daemon
llm-monitor daemon stop

# Install as systemd user service (auto-start on login)
llm-monitor daemon install

# Remove systemd service
llm-monitor daemon uninstall
```

When the daemon is running, normal CLI commands read from the database:

```bash
llm-monitor --now          # reads from daemon's DB
llm-monitor --report       # reports from daemon's history
llm-monitor --fresh        # bypass daemon, fetch directly
```

### Configuration

```toml
[general]
poll_interval = 600          # global poll interval (10 minutes)

[daemon]
log_file = ""                # empty = default (~/.local/state/llm-monitor/daemon.log)
pid_file = ""                # empty = default ($XDG_RUNTIME_DIR/llm-monitor/daemon.pid)

[providers.ollama]
poll_interval = 60           # per-provider override (local services can poll faster)
```

**PID file:** `$XDG_RUNTIME_DIR/llm-monitor/daemon.pid` (or `/tmp/llm-monitor-<uid>/daemon.pid`)
**Log file:** `$XDG_STATE_HOME/llm-monitor/daemon.log` (or `~/.local/state/llm-monitor/daemon.log`)

## Monitor Mode

Launch a persistent Rich Live dashboard that auto-refreshes and displays all configured providers.

```bash
# Full dashboard
llm-monitor --monitor

# Monitor a specific provider
llm-monitor --monitor --provider claude

# Compact mode (single line per provider, ideal for tmux)
llm-monitor --monitor --compact

# Custom refresh interval (default 30s, minimum 5s)
llm-monitor --monitor --interval 10
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

The daemon mode maps naturally to Docker. The container runs `llm-monitor daemon run` in the foreground.

### Quick Start

```bash
docker build -t llm-monitor .
docker compose up -d
```

### docker-compose.yml

```yaml
services:
  llm-monitor:
    build: .
    restart: unless-stopped
    volumes:
      - llm-monitor-data:/data
      - ${HOME}/.config/llm-monitor/config.toml:/home/monitor/.config/llm-monitor/config.toml:ro
      - ${HOME}/.claude/.credentials.json:/home/monitor/.claude/.credentials.json:ro
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - XAI_MANAGEMENT_KEY=${XAI_MANAGEMENT_KEY}
      - XAI_TEAM_ID=${XAI_TEAM_ID}
      - XAI_API_KEY=${XAI_API_KEY}

volumes:
  llm-monitor-data:
```

### Container-Aware Mode

When `$LLM_MONITOR_CONTAINER=1` is set (or `/.dockerenv` exists):

- Permission checks are skipped
- Keyring is not attempted (no D-Bus in containers)
- `daemon install`/`uninstall` are disabled (use `daemon run` directly)

### Accessing Data from Host

```bash
# Point host CLI at the container's database
export LLM_MONITOR_DATA_DIR=/path/to/docker/volume
llm-monitor --now
llm-monitor --report --days 7

# Or use docker exec
docker exec llm-monitor llm-monitor --now
```

## Scripting Examples

```bash
# Get Claude session utilisation as a number
llm-monitor | jq -r '.providers[0].windows[0].utilisation'

# Alert when usage is high
USAGE=$(llm-monitor | jq -r '.providers[0].windows[0].utilisation')
if (( $(echo "$USAGE > 80" | bc -l) )); then
    notify-send "Claude usage high: ${USAGE}%"
fi

# Pipe to other tools
llm-monitor | jq '.providers[].windows[] | {name, utilisation, status}'
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
