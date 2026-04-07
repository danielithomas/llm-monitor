# llm-monitor

Monitor your LLM consumption from local and online services.

Currently supports **Anthropic Claude** (subscription utilisation tracking). Future versions will add Grok (xAI), OpenAI, Ollama, and local system metrics.

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

# From source
git clone https://github.com/<user>/llm-monitor.git
cd llm-monitor
uv sync --group dev
```

## Prerequisites

The Claude provider reads OAuth credentials managed by Claude Code:

1. Install [Claude Code](https://claude.ai/code)
2. Run `claude /login` to authenticate
3. The tool reads `~/.claude/.credentials.json` (read-only, never writes to it)

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

```bash
git clone https://github.com/<user>/llm-monitor.git
cd llm-monitor
uv sync --group dev
uv run pytest -v
```

## Licence

MIT
