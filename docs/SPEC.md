# LLM Monitor - Specification

**Project codename:** `llm-monitor`
**Version:** 0.1.0 (CLI MVP - Claude provider)
**Author:** Daniel Thomas
**Date:** 2026-04-05
**Status:** Final Draft

---

## 1. Overview

A Linux-native application for monitoring LLM service usage, costs, and performance from the command line and (in v2) a GTK/GNOME desktop interface. The tool uses a pluggable provider architecture, launching with Anthropic Claude support and expanding to cover Grok (xAI), OpenAI, Ollama, and local system metrics.

### 1.1 Goals

- Provide at-a-glance visibility into LLM usage across multiple providers without opening browser dashboards.
- Track subscription utilisation windows (Claude), API spend and credit balances (OpenAI, xAI), and local inference metrics (Ollama).
- Support scripting and pipeline integration via structured JSON output.
- Offer a persistent terminal monitor mode for real-time tracking during work sessions.
- Plan for a GTK/GNOME (KDE-compatible) desktop widget in a future release.
- Use a pluggable provider architecture so new LLM services can be added without modifying core logic.
- Handle all credentials securely, never storing secrets in plaintext configuration files.

### 1.2 Non-Goals (v1)

- Replacing any provider's native dashboard entirely.
- Managing or modifying usage limits, spending caps, or billing.
- Supporting macOS Keychain credential retrieval (Linux only).
- Providing a unified cost normalisation across providers (each reports in its native units).
- Acting as a proxy or gateway for LLM traffic.

---

## 2. Architecture

### 2.1 System Context

The tool operates in two modes: **standalone** (CLI fetches directly) and **daemon** (background service collects, CLI reads from DB). The daemon is the recommended mode for continuous monitoring and history collection.

```
┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ Claude Code   │  │ xAI Console  │  │ OpenAI API   │  │ Ollama       │
│ credentials   │  │ API key      │  │ API key      │  │ localhost    │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                 │                 │                 │
       ▼                 ▼                 ▼                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   llm-monitor daemon                                │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                    Provider Registry                          │  │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────────┐│  │
│  │  │ Claude   │ │ Grok     │ │ OpenAI   │ │ Ollama / Local  ││  │
│  │  │ Provider │ │ Provider │ │ Provider │ │ Provider         ││  │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────────────┘│  │
│  └───────────────────────────────────────────────────────────────┘  │
│  ┌─────────┐ ┌──────────┐ ┌──────────────┐ ┌───────────────────┐  │
│  │ Poll    │ │ Config   │ │ Notification │ │ Rate-Limit        │  │
│  │ Sched.  │ │ Loader   │ │ Engine       │ │ Backoff           │  │
│  └─────────┘ └──────────┘ └──────────────┘ └───────────────────┘  │
└────────────────────────────┬────────────────────────────────────────┘
                             │ writes
                             ▼
                   ┌──────────────────┐
                   │  SQLite History   │
                   │  (history.db)     │
                   └────────┬─────────┘
                            │ reads
         ┌──────────────────┼──────────────────┐
         ▼                  ▼                  ▼
┌─────────────────┐ ┌──────────────┐ ┌──────────────┐
│ CLI (JSON/Table)│ │ TUI Monitor  │ │ GTK UI (v2)  │
│ llm-monitor     │ │ --monitor    │ │ --ux         │
└─────────────────┘ └──────────────┘ └──────────────┘
```

In standalone mode (no daemon running), the CLI fetches directly from providers and writes to the DB itself. This supports ad-hoc usage but does not provide continuous history collection.

### 2.2 Provider Abstraction

Every LLM service is represented by a **Provider** - a Python class implementing a common interface. The core application knows nothing about specific APIs; it only works with the provider contract. Credential resolution is a framework-level concern handled by the base class, not reimplemented per provider.

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
import keyring as kr
import os, shlex, subprocess


class SecretStr:
    """String wrapper that prevents accidental logging of secrets."""

    def __init__(self, value: str):
        self._value = value

    def get_secret_value(self) -> str:
        return self._value

    def __repr__(self) -> str:
        return "SecretStr('***')"

    def __str__(self) -> str:
        return "***REDACTED***"

    def __len__(self) -> int:
        return len(self._value)

    def __bool__(self) -> bool:
        return bool(self._value)


class CredentialError(Exception):
    """Raised when credential resolution fails.

    This is a hard failure — the provider cannot authenticate.
    Used when key_command returns non-zero, times out, or produces
    no output. NOT used for "no credential found" (which returns None).
    """

    def __init__(self, message: str, provider: str = ""):
        self.provider = provider
        super().__init__(message)


@dataclass
class UsageWindow:
    """A time-bounded usage allocation (e.g., 5-hour session, monthly budget)."""
    name: str                          # e.g., "Session (5h)", "Monthly Budget"
    utilisation: float                 # 0.0 - 100.0+ (percentage)
    resets_at: Optional[datetime]      # when this window resets, if applicable
    status: str                        # normal | warning | critical | exceeded
    unit: str                          # "percent" | "usd" | "tokens"
    raw_value: Optional[float]         # underlying value (e.g., $12.50, 45000 tokens)
    raw_limit: Optional[float]         # the cap (e.g., $50.00, 100000 tokens)


@dataclass
class ModelUsage:
    """Per-model usage breakdown within a provider."""
    model: str                         # e.g., "claude-opus-4-6", "gpt-4o", "llama3.2:3b"
    input_tokens: Optional[int]        # total input tokens consumed
    output_tokens: Optional[int]       # total output tokens generated
    total_tokens: Optional[int]        # combined token count
    cost: Optional[float]              # cost in provider's currency (USD), if known
    request_count: Optional[int]       # number of requests/messages
    period: Optional[str]              # time period this covers (e.g., "5h", "7d", "mtd")


@dataclass
class ProviderStatus:
    """Unified status response from any provider."""
    provider_name: str                 # e.g., "claude", "openai", "grok", "ollama"
    provider_display: str              # e.g., "Anthropic Claude", "xAI Grok"
    timestamp: datetime
    cached: bool
    cache_age_seconds: int
    windows: list[UsageWindow]         # one or more usage windows
    model_usage: list[ModelUsage]      # per-model breakdown (empty if not available)
    extras: dict                       # provider-specific data (plan name, etc.)
    errors: list[str]


class Provider(ABC):
    """Base class for all LLM service providers."""

    @abstractmethod
    def name(self) -> str:
        """Short identifier (e.g., 'claude', 'openai')."""
        ...

    @abstractmethod
    def display_name(self) -> str:
        """Human-readable name (e.g., 'Anthropic Claude')."""
        ...

    @abstractmethod
    def is_configured(self) -> bool:
        """Return True if credentials/config exist for this provider."""
        ...

    @abstractmethod
    async def fetch_usage(self) -> ProviderStatus:
        """Query the provider and return a unified status."""
        ...

    @abstractmethod
    def auth_instructions(self) -> str:
        """Return human-readable setup instructions."""
        ...

    def resolve_credential(self, config: dict, logger=None) -> Optional[SecretStr]:
        """Framework-level credential resolution. Providers may override.

        Resolution order (first match wins):
          1. key_command - execute shell command, read stdout (HARD FAIL on error)
          2. key_env / well-known env var
          3. System keyring (GNOME Keyring / KDE Wallet via keyring lib)
          4. Provider-specific credential file (Claude only)

        If key_command is configured and fails, resolution stops with an error
        rather than silently falling through. The user explicitly configured a
        credential source; ignoring its failure masks misconfiguration.
        """
        # Tier 1: key_command (hard fail if configured and broken)
        if cmd := config.get("key_command"):
            try:
                args = shlex.split(cmd)
                result = subprocess.run(
                    args, capture_output=True, text=True,
                    timeout=10, shell=False,  # NEVER shell=True (D-024)
                )
                if result.returncode == 0 and result.stdout.strip():
                    return SecretStr(result.stdout.strip())
                # Command ran but returned no output or non-zero exit
                raise CredentialError(
                    f"key_command failed (exit {result.returncode}): "
                    f"{result.stderr.strip()}"
                )
            except subprocess.TimeoutExpired:
                raise CredentialError(
                    f"key_command timed out after 10s: {cmd}"
                )

        # Tier 2: environment variable
        env_name = config.get("key_env", self._default_env_var())
        if env_name and (val := os.environ.get(env_name)):
            return SecretStr(val)

        # Tier 3: system keyring
        if config.get("key_keyring", True):
            try:
                val = kr.get_password("llm-monitor", f"{self.name()}_api_key")
                if val:
                    return SecretStr(val)
            except Exception:
                if logger:
                    logger.debug("Keyring unavailable for %s", self.name())

        return None

    def _default_env_var(self) -> Optional[str]:
        """Override in subclass to specify the conventional env var."""
        return None

    @property
    def allowed_hosts(self) -> list[str]:
        """Hosts this provider is permitted to send credentials to."""
        return []


# ─── Provider Registry ────────────────────────────────────────
# For v1, providers are registered explicitly in a dict.
# Third-party providers via entry_points are deferred (OQ-015).

PROVIDERS: dict[str, type[Provider]] = {}

def register_provider(cls: type[Provider]) -> type[Provider]:
    """Decorator to register a provider class."""
    instance = cls.__new__(cls)
    PROVIDERS[instance.name()] = cls
    return cls

def get_enabled_providers(config: dict) -> list[Provider]:
    """Instantiate and return providers that are enabled in config."""
    enabled = []
    for name, cls in PROVIDERS.items():
        provider_config = config.get("providers", {}).get(name, {})
        if provider_config.get("enabled", False):
            enabled.append(cls(provider_config))
    return enabled
```

Providers register themselves using the `@register_provider` decorator in their module. The CLI loads all provider modules on startup via explicit imports in `providers/__init__.py`. The registry is a module-level dict — no entry_points complexity for v1.

### 2.3 Concurrency Model

`Provider.fetch_usage()` is an `async` method. The CLI and daemon both use `asyncio` to execute provider fetches:

- **CLI (standalone):** `asyncio.run()` wraps the top-level orchestrator. All enabled providers are fetched concurrently via `asyncio.gather()`. A single shared `httpx.AsyncClient` is used across all providers within a fetch cycle (connection pooling, shared timeouts).
- **Daemon:** The daemon runs a persistent `asyncio` event loop. Each poll cycle calls `asyncio.gather()` across all enabled providers. The event loop also manages timers for the poll schedule.
- **Error isolation:** `asyncio.gather(return_exceptions=True)` ensures one provider's failure does not cancel others. Failed providers return their exception, which the orchestrator converts to an error entry in `ProviderStatus.errors`.

### 2.4 Component Breakdown

| Component | Responsibility | Technology |
|-----------|---------------|------------|
| Daemon | Background service: poll providers on schedule, write to history DB, fire notifications | `asyncio` event loop |
| Provider Registry | Discover and manage provider plugins | Python entry_points / importlib |
| Provider: Claude | Query `/api/oauth/usage`, detect token staleness | `httpx` |
| Provider: Grok | Query xAI usage/billing endpoints | `httpx` |
| Provider: OpenAI | Query `/v1/usage` and billing endpoints | `httpx` |
| Provider: Ollama | Query local Ollama API for running model stats | `httpx` |
| Provider: Local | System GPU/VRAM/CPU metrics for local inference | `psutil`, `pynvml` |
| Security Layer | SecretStr, credential resolution, sanitisation, secure I/O | `keyring`, stdlib |
| Cache Layer | Per-provider cached responses (standalone mode only) | JSON files in `~/.cache/llm-monitor/` |
| Config Loader | TOML configuration with per-provider sections | `tomllib` / `tomli` |
| Notification Engine | Desktop notifications on status transitions | `notify-send` / `gi.repository.Notify` |
| CLI Framework | Argument parsing, mode dispatch, signal handling | `click` or `typer` |
| Output: JSON | Machine-readable structured output | `json` stdlib |
| Output: Table | Human-readable Rich table | `rich` |
| Output: Monitor | Auto-refreshing Rich Live TUI (reads from history DB) | `rich.live` |
| Output: GTK (v2) | GNOME system tray indicator with popover (reads from history DB) | PyGObject + libadwaita |

---

## 3. Provider Specifications

### 3.1 Claude (Anthropic) - v0.1.0

**Type:** Subscription utilisation monitoring

**Data source:** `GET https://api.anthropic.com/api/oauth/usage`

**Authentication:**
- OAuth token from `~/.claude/.credentials.json` (or `$CLAUDE_CONFIG_DIR/.credentials.json`).
- Requires Claude Code installed and authenticated via `claude /login`.
- Token structure: `claudeAiOauth.accessToken` (bearer token), wrapped in `SecretStr` on read.
- Required header: `anthropic-beta: oauth-2025-04-20`.
- Required header: `Authorization: Bearer <accessToken>`.
- This provider does not use the standard `resolve_credential()` flow; it reads Claude Code's managed credential file directly.

**Credentials file schema (`~/.claude/.credentials.json`):**
```json
{
  "claudeAiOauth": {
    "accessToken": "sk-ant-oat01-...",
    "refreshToken": "sk-ant-ort01-...",
    "expiresAt": "2026-04-05T16:30:00.000Z"
  }
}
```

The tool reads only `claudeAiOauth.accessToken` (wrapped in `SecretStr` immediately) and `claudeAiOauth.expiresAt` (ISO 8601 UTC). The `refreshToken` is not used — the tool never refreshes tokens (see D-036). The `$CLAUDE_CONFIG_DIR` environment variable, if set by Claude Code, overrides the default `~/.claude/` directory. Resolution order: `$CLAUDE_CONFIG_DIR/.credentials.json` → `~/.claude/.credentials.json`.

**Token expiry check:** Before each API call, parse `expiresAt` and compare to current UTC time. If `expiresAt - now() <= 300` seconds (5 minutes), skip the API call and report the expiry error (see Section 7.7).

**Response schema (observed):**
```json
{
  "five_hour": {
    "utilization": 42.0,
    "resets_at": "2026-04-05T15:00:00+00:00"
  },
  "seven_day": {
    "utilization": 68.0,
    "resets_at": "2026-04-08T00:00:00+00:00"
  },
  "seven_day_opus": {
    "utilization": 12.0,
    "resets_at": "2026-04-08T00:00:00+00:00"
  },
  "seven_day_oauth_apps": null,
  "iguana_necktie": null
}
```

**Mapped usage windows:**

| Window | Source Field | Unit | Notes |
|--------|-------------|------|-------|
| Session (5h) | `five_hour.utilization` | percent | Resets every 5 hours |
| Weekly (7d) | `seven_day.utilization` | percent | Rolling 7-day window |
| Weekly Opus | `seven_day_opus.utilization` | percent | Opus-specific cap (if on plan with Opus) |

**Rate limiting:**
- Aggressively rate-limited. 429 errors observed even at 60-second intervals.
- No `Retry-After` header returned.
- Default poll interval: 10 minutes (see Section 4.6).
- Once rate-limited, 429s can persist for 30+ minutes.

**Rate-limit backoff strategy:**
When a 429 is received, the provider enters a backoff state. Subsequent poll attempts skip the API call and return cached data until the backoff period expires. The backoff escalates exponentially:
- 1st 429: wait 10 minutes (1x poll interval)
- 2nd consecutive 429: wait 20 minutes
- 3rd consecutive 429: wait 40 minutes
- Cap: 60 minutes maximum backoff

The backoff state is persisted in the provider's cache file so it survives process restarts. A successful fetch resets the backoff counter. The `--fresh` flag overrides backoff and forces an API call regardless (accepting the risk of another 429). In `--verbose` mode, the tool logs the current backoff state and next retry time to stderr.

**Token handling:**
- Access tokens expire roughly every 6 hours.
- Claude Code manages token refresh automatically; llm-monitor is a read-only consumer (see Section 7.7).
- On token expiry, the tool emits a clear error directing the user to run `claude /login`.

**Extra usage spend:** No REST endpoint exists. Deferred (see OQ-001).

**Extras dict:** `{ "plan": null, "extra_usage_enabled": null }` (plan detection deferred to OQ-006).

**Per-model breakdown:** The Claude usage endpoint does not provide a per-model token breakdown. The `seven_day_opus` window provides an Opus-specific utilisation percentage, which is mapped as a separate `UsageWindow`. The `model_usage` list will contain an entry for Opus (derived from the Opus window) but Sonnet usage can only be inferred as the difference between the all-models weekly window and the Opus window. If Anthropic expands the endpoint to include per-model detail, the provider will populate `model_usage` accordingly.

**Allowed hosts:** `api.anthropic.com` (HTTPS only).

---

### 3.2 Grok (xAI) - v0.3.0

**Type:** API credit/spend monitoring + rate limit tracking

**Data source:** xAI Console API (API key based)

**Authentication:**
- Standard API key from xAI Console, resolved via `resolve_credential()` (keyring, env var, or key_command).
- Default env var: `$XAI_API_KEY`.

**Available metrics:**

| Metric | Source | Notes |
|--------|--------|-------|
| Rate limit remaining | Response headers (`x-ratelimit-remaining-requests`, `x-ratelimit-reset-requests`) | Per-request, needs aggregation |
| Token consumption | Response `usage` object (`prompt_tokens`, `completion_tokens`, `total_tokens`) | Per-request |
| Spend/billing | xAI Console dashboard | No known programmatic API for balance/spend |

**Mapped usage windows (planned):**

| Window | Source | Unit | Notes |
|--------|--------|------|-------|
| Rate Limit (daily) | Response headers | percent | Derived from remaining/limit |
| Spend (MTD) | Console API or scrape | usd | Availability TBD |

**Open question:** xAI does not appear to have a dedicated usage/billing REST API comparable to OpenAI's. The xAI Console shows spend, but programmatic access may require scraping or an undocumented endpoint. See OQ-011.

**Per-model breakdown:** Rate limit headers are per-model (each Grok model has its own limits). The `usage` object in API responses includes per-request token counts tagged by model. The provider aggregates these into `model_usage` entries when available.

**Allowed hosts:** `api.x.ai` (HTTPS only).

---

### 3.3 OpenAI - v0.4.0

**Type:** API spend and credit monitoring

**Data source:** OpenAI platform API endpoints

**Authentication:**
- API key from OpenAI Console, resolved via `resolve_credential()` (keyring, env var, or key_command).
- Default env var: `$OPENAI_API_KEY`.

**Available endpoints:**

| Endpoint | Data | Notes |
|----------|------|-------|
| `GET /v1/organization/usage/completions` | Token usage by model, project, time bucket | Official Usage API |
| `GET /v1/organization/costs` | Cost breakdown by line item | Official Costs API |
| `GET /v1/dashboard/billing/subscription` | Plan and billing cycle details | Undocumented but widely used |
| `GET /v1/dashboard/billing/credit_grants` | Remaining credit balance | Undocumented but widely used |

**Mapped usage windows (planned):**

| Window | Source | Unit | Notes |
|--------|--------|------|-------|
| Credit Balance | `/v1/dashboard/billing/credit_grants` | usd | Remaining prepaid credits |
| Spend (MTD) | `/v1/organization/costs` | usd | Month-to-date API spend |
| Rate Limit | Response headers | percent | Per-model RPM/TPM |

**Extras dict:** `{ "plan": "...", "models_used": [...], "top_model_spend": {...} }`

**Per-model breakdown:** The OpenAI Usage API natively supports grouping by model via the `group_by[]=model` parameter. The provider populates `model_usage` with per-model token counts and costs. This is the richest per-model data of any provider.

**Allowed hosts:** `api.openai.com` (HTTPS only).

---

### 3.4 Ollama (Network / Local) - v0.5.0

**Type:** Local and network inference performance monitoring

**Data source:** Ollama REST API (one or more endpoints)

**Authentication:** None (local/network service). No credentials required.

**Multi-host support:** The tool supports monitoring multiple Ollama instances across the local network. Each host is a separate logical endpoint but reports under the same provider. This is common in homelab setups where inference is distributed across machines (e.g., a workstation with an RTX 5080 running one set of models and a secondary server with an RTX 3090 running others).

**Configuration:**
```toml
# Single host (simple form)
[providers.ollama]
enabled = true
host = "http://localhost:11434"

# Multiple hosts (array form)
[providers.ollama]
enabled = true

[[providers.ollama.hosts]]
name = "workstation"               # human-readable label
url = "http://localhost:11434"

[[providers.ollama.hosts]]
name = "gpu-server"
url = "http://gpu-server.local:11434"

[[providers.ollama.hosts]]
name = "nas-inference"
url = "http://192.168.1.50:11434"
```

When multiple hosts are configured, each host's models and metrics are reported with a host label prefix in the output (e.g., `workstation: llama3.2 (3B)`, `gpu-server: mistral (7B)`). The `ProviderStatus.extras` dict includes a per-host breakdown.

**Available endpoints (per host):**

| Endpoint | Data | Notes |
|----------|------|-------|
| `GET /api/tags` | List of available models | Health check |
| `GET /api/ps` | Currently loaded models, VRAM usage | Real-time state |
| `GET /metrics` | Prometheus-format metrics (if enabled) | Not available in all versions |
| Response `usage` fields | Per-request token counts and timing | `eval_count`, `eval_duration`, etc. |

**Mapped usage windows (planned):**

| Window | Source | Unit | Notes |
|--------|--------|------|-------|
| Models Loaded | `/api/ps` (all hosts) | count | Total models in memory across all hosts |
| VRAM Usage | `/api/ps` (per host) | MB/GB | Memory allocated per host |
| Inference Speed | Aggregated from responses | tokens/sec | Rolling average tokens/second |

**Extras dict:**
```json
{
  "hosts": [
    {
      "name": "workstation",
      "url": "http://localhost:11434",
      "status": "connected",
      "models_loaded": ["llama3.2:3b"],
      "vram_used_mb": 4096,
      "vram_total_mb": 16384,
      "avg_tokens_per_sec": 45.2
    },
    {
      "name": "gpu-server",
      "url": "http://gpu-server.local:11434",
      "status": "connected",
      "models_loaded": ["mistral:7b", "codellama:13b"],
      "vram_used_mb": 18432,
      "vram_total_mb": 24576,
      "avg_tokens_per_sec": 38.1
    }
  ]
}
```

**Note:** Ollama's monitoring story is fundamentally different from cloud providers. There are no quotas or spend limits - it is a performance and resource utilisation monitor. The provider maps to the same `ProviderStatus` structure but uses resource-oriented windows rather than quota-oriented ones.

**Allowed hosts:** `localhost`, `127.0.0.1`, `[::1]`, or any user-configured host in the `hosts` array (HTTP or HTTPS). Network hosts are trusted by configuration - the user explicitly adds them.

---

### 3.5 Local System Metrics - v0.6.0

**Type:** Hardware resource monitoring for local inference

**Data source:** System APIs

**Authentication:** None.

**Available metrics:**

| Metric | Source | Notes |
|--------|--------|-------|
| GPU utilisation | `pynvml` (NVIDIA) / `rocm_smi` (AMD) | Percentage |
| GPU VRAM | `pynvml` / `rocm_smi` | Used/total in MB |
| GPU temperature | `pynvml` / `rocm_smi` | Celsius |
| CPU usage | `psutil` | Percentage |
| RAM usage | `psutil` | Used/total in GB |
| Disk I/O | `psutil` | Read/write rates |

**Mapped usage windows (planned):**

| Window | Source | Unit | Notes |
|--------|--------|------|-------|
| GPU Load | pynvml/rocm_smi | percent | Current GPU utilisation |
| GPU VRAM | pynvml/rocm_smi | percent | VRAM usage as percentage |
| GPU Temp | pynvml/rocm_smi | celsius | Current temperature |
| System RAM | psutil | percent | System memory usage |

**Multi-GPU support:** Must handle multiple GPUs (e.g., RTX 5080 primary + RTX 3090 secondary). Each GPU reported as a separate set of windows with a device index.

**Extras dict:** `{ "gpus": [{ "index": 0, "name": "RTX 5080", "vram_total_mb": 16384, ... }], "cpu_count": 16, ... }`

---

## 4. CLI Interface

### 4.1 Command Structure

```
llm-monitor [MODE] [OPTIONS]
llm-monitor daemon <start|stop|status|run|install>
llm-monitor history <report|purge|stats|export>
llm-monitor config <set-key|check>
```

### 4.2 Modes

#### 4.2.1 Output Stream Rules

All modes follow these rules without exception:

| Content | Destination | Rationale |
|---------|-------------|-----------|
| JSON output (default mode) | stdout | Machine-parseable data for piping |
| Table output (`--now`) | stdout | Primary output the user requested |
| `--help` output | stdout | Conventional; allows `llm-monitor --help \| less` |
| `--version` output | stdout | Conventional |
| `--list-providers` output | stdout | Data the user requested |
| Error messages | stderr | Must not corrupt piped data |
| Warnings (rate-limited, stale cache) | stderr | Informational, not data |
| Progress spinners / status updates | stderr | Ephemeral UI, not data |
| `--verbose` debug logging | stderr | Diagnostic, not data |

#### 4.2.2 TTY Detection and Adaptive Output

The tool MUST detect whether stdout is connected to a TTY and adapt behaviour accordingly.

**When stdout is NOT a TTY (piped or redirected):**
- Disable all ANSI colour codes, progress bars, and spinners on stdout.
- In `--now` mode, output plain ASCII table (no Unicode box-drawing characters).
- In `--monitor` mode, refuse to start and exit with an error: `Error: --monitor requires an interactive terminal`.

**Colour override precedence (highest to lowest):**
1. `--no-colour` flag (always disables).
2. `$NO_COLOR` environment variable (if set to any value, disables) - per https://no-color.org/.
3. `$LLM_MONITOR_NO_COLOR` env var (app-specific override).
4. `$TERM=dumb` (disables).
5. TTY detection (auto).
6. `--colour=always` flag (force enable even when piped, for `llm-monitor --now --colour=always | less -R`).

#### 4.2.3 JSON Mode (default, no flag required)

Returns structured JSON to stdout and exits. Designed for consumption by scripts, `jq`, waybar modules, polybar scripts, monitoring pipelines.

```bash
# All configured providers
llm-monitor | jq '.providers[].provider_name'

# Single provider
llm-monitor --provider claude | jq '.providers[0].windows'

# Use in a script
USAGE=$(llm-monitor --provider claude | jq -r '.providers[0].windows[0].utilisation')
if (( $(echo "$USAGE > 80" | bc -l) )); then
    notify-send "Claude session usage high: ${USAGE}%"
fi
```

**Output schema:**
```json
{
  "timestamp": "2026-04-05T10:30:00+10:00",
  "version": "0.1.0",
  "providers": [
    {
      "provider_name": "claude",
      "provider_display": "Anthropic Claude",
      "timestamp": "2026-04-05T10:30:00+10:00",
      "cached": false,
      "cache_age_seconds": 0,
      "windows": [
        {
          "name": "Session (5h)",
          "utilisation": 42.0,
          "resets_at": "2026-04-05T15:00:00+00:00",
          "resets_in_human": "2h 15m",
          "status": "normal",
          "unit": "percent",
          "raw_value": null,
          "raw_limit": null
        },
        {
          "name": "Weekly (7d)",
          "utilisation": 68.0,
          "resets_at": "2026-04-08T00:00:00+00:00",
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

**Computed fields in JSON output:** The `resets_in_human` field is NOT part of the `UsageWindow` dataclass — it is computed at JSON serialisation time from `resets_at` relative to the current timestamp. Format: largest two units, e.g., `"2h 15m"`, `"2d 13h"`, `"45m"`, `"< 1m"`. If `resets_at` is `null`, `resets_in_human` is `null`. The top-level `timestamp` is the time the CLI was invoked. The top-level `version` is the package version (from `importlib.metadata.version("llm-monitor")`).

**Status values:** `normal` (0-69%), `warning` (70-89%), `critical` (90-99%), `exceeded` (100%+). These thresholds are configurable via the `[thresholds]` config section (see Section 4.6). If `[thresholds]` is present, its `warning` and `critical` values override the defaults. The `exceeded` threshold (100%) is not configurable.

**Exit codes:**

| Code | Meaning |
|------|---------|
| 0 | Success (all requested providers returned data) |
| 1 | General error (config missing, parse error) |
| 2 | Authentication error (one or more providers failed auth) |
| 3 | Partial success (some providers returned data, others failed) |
| 4 | Network error (no providers reachable) |
| 130 | SIGINT received (128 + 2, conventional) |
| 143 | SIGTERM received (128 + 15, conventional) |

#### 4.2.4 Table Mode (`--now`)

Renders a human-readable table to stdout and exits. Groups output by provider. Falls back to plain ASCII when stdout is not a TTY.

```bash
llm-monitor --now
llm-monitor --now --provider claude
llm-monitor --now --provider claude,openai
```

**Example output (multi-provider):**
```
LLM Monitor                               05 Apr 2026, 10:30 AEST
═══════════════════════════════════════════════════════════════════

 Anthropic Claude                          cached 3m ago
───────────────────────────────────────────────────────────────────
 Session (5h)   ████████░░░░░░░░░░░░  42%    resets in 2h 15m
 Weekly (7d)    █████████████░░░░░░░  68%    resets in 2d 13h
 Weekly Opus    ██░░░░░░░░░░░░░░░░░░  12%    resets in 2d 13h

 xAI Grok                                  fresh
───────────────────────────────────────────────────────────────────
 Rate Limit     ████░░░░░░░░░░░░░░░░  22%    resets in 18h
 Spend (MTD)    ██████░░░░░░░░░░░░░░  $12.40 / $50.00

 Ollama (local)                             live
───────────────────────────────────────────────────────────────────
 Models Loaded  llama3.2 (3B), mistral (7B)
 GPU VRAM       ██████████████░░░░░░  71%    11.6 / 16.0 GB
 Inference      45.2 tok/s avg

═══════════════════════════════════════════════════════════════════
```

**Colour coding:**
- Green (normal): 0-69%
- Yellow (warning): 70-89%
- Red (critical): 90-99%
- Magenta pulsing (exceeded): 100%+

#### 4.2.5 Persistent Monitor Mode (`--monitor`)

Launches a Rich Live TUI that auto-refreshes and remains running until the user presses `q` or `Ctrl+C`. Displays all configured providers in a stacked layout. Requires an interactive terminal (exits with error if stdout is not a TTY).

```bash
llm-monitor --monitor
llm-monitor --monitor --provider claude
llm-monitor --monitor --compact    # single-line per provider for tmux
```

**Data source:** When the daemon is running, `--monitor` reads from the history database (no direct API calls). This makes the TUI a lightweight display-only process. When no daemon is running, `--monitor` fetches directly from providers on each refresh cycle (standalone behaviour).

**Features:**
- Auto-refresh display at configurable interval (default 30s UI refresh, data freshness depends on daemon poll interval).
- Live countdown timers for reset windows.
- Status colour transitions as utilisation changes.
- Compact single-line mode via `--compact` for tmux/polybar/waybar embedding.
- Rate-limit backoff indicator per provider (when in standalone mode).
- Desktop notification on status transitions (configurable via `--notify`).
- Provider health indicators (green dot = connected, yellow = stale, red = error).
- Daemon status indicator (shows whether daemon is running and last poll time).

**Key bindings:**
- `r` - Force refresh all providers (bypass cache).
- `1-9` - Force refresh specific provider by index.
- `q` - Quit.
- `j` - Dump current state as JSON to file.
- `?` - Show help overlay.

#### 4.2.6 GTK/GNOME Mode (`--ux`) [v2]

Launches a GTK4/libadwaita application with a system tray indicator.

**Planned features (v2):**
- System tray icon with worst-status colour across all providers.
- Click to expand popover with per-provider breakdown.
- Each provider shown as a collapsible section.
- Desktop notifications on status transitions.
- Auto-start via `.desktop` file and XDG autostart.
- Respects system dark/light theme via libadwaita.
- KDE compatibility via StatusNotifierItem (SNI) protocol.

#### 4.2.7 Daemon Mode (`daemon`)

The daemon is a background service that polls providers on a schedule and writes results to the history database. It decouples data collection from presentation: the CLI, TUI, and GTK frontends become thin readers of the shared SQLite database.

**Why a daemon?** Without it, history is only recorded when a user happens to run the tool. Usage spikes, rate-limit events, and cost changes that occur between invocations are invisible. The daemon collects 24/7, enabling meaningful trend analysis, reliable notifications, and a consistent data source for all frontends.

**Subcommands:**

```bash
llm-monitor daemon start           # start as background process
llm-monitor daemon stop            # stop the running daemon
llm-monitor daemon status          # show daemon state, PID, last poll time
llm-monitor daemon run             # run in foreground (for systemd/Docker)
llm-monitor daemon install         # install systemd user service
llm-monitor daemon uninstall       # remove systemd user service
```

**`daemon start`:** Forks to background, writes PID to `$XDG_RUNTIME_DIR/llm-monitor/daemon.pid` (or `/tmp/llm-monitor-$UID/daemon.pid` if `$XDG_RUNTIME_DIR` is unset). Logs to `$XDG_STATE_HOME/llm-monitor/daemon.log` (or `~/.local/state/llm-monitor/daemon.log`). Exits immediately after fork; the parent prints the PID and returns.

**`daemon run`:** Runs in the foreground, logging to stderr. Designed for systemd `ExecStart`, Docker `ENTRYPOINT`, or manual debugging. This is the primary entry point for containerised deployments.

**`daemon stop`:** Reads the PID file, sends `SIGTERM`, waits up to 5 seconds for clean shutdown, then `SIGKILL` if needed. If no PID file exists, prints "Daemon is not running."

**`daemon status`:** Reports whether the daemon is running, its PID, uptime, last successful poll per provider, next scheduled poll, and database size.

```
$ llm-monitor daemon status
Daemon: running (PID 48231, uptime 3h 12m)
  claude    last poll 2m ago    next in 8m     ok
  openai    last poll 2m ago    next in 8m     ok
  ollama    last poll 32s ago   next in 28s    ok
Database: ~/.local/share/llm-monitor/history.db (4.2 MB)
```

**`daemon install`:** Writes a systemd user service unit file and enables it:

```ini
# ~/.config/systemd/user/llm-monitor.service
[Unit]
Description=LLM Usage Monitor Daemon
Documentation=man:llm-monitor(1)
After=network-online.target
Wants=network-online.target

[Service]
Type=exec
ExecStart=/path/to/llm-monitor daemon run
Restart=on-failure
RestartSec=30
Environment=LLM_MONITOR_LOG_LEVEL=info

[Install]
WantedBy=default.target
```

After writing the unit file, runs `systemctl --user daemon-reload && systemctl --user enable --now llm-monitor`. The `ExecStart` path is resolved from the current `llm-monitor` binary location (via `shutil.which()` or `sys.argv[0]`).

**Poll loop:**
- On startup: read config, initialise providers, run retention pruning, perform an immediate first poll.
- Each provider is polled independently at the global `poll_interval` (default 600s / 10 minutes). Per-provider `poll_interval` overrides are respected if configured.
- After each successful fetch, write to the history database if the data has meaningfully changed (same delta logic as Section 6.4).
- On 429 / rate limit: enter backoff state (see Section 3.1), skip provider until backoff expires.
- On network error: log warning, retry on next cycle.
- On `SIGHUP`: reload config file, re-initialise providers.
- On `SIGTERM` / `SIGINT`: flush pending writes, close database, remove PID file, exit cleanly.

**Standalone fallback:** All CLI modes (`llm-monitor`, `--now`, `--monitor`) continue to work without the daemon. When no daemon is running, the CLI fetches directly from providers and writes to the history database itself (the v0.1.0 behaviour). The daemon is additive, not required.

**Daemon detection:** The CLI checks for a running daemon by testing the PID file (`$XDG_RUNTIME_DIR/llm-monitor/daemon.pid`). If the daemon is running:
- Default mode / `--now` / `--monitor`: read latest data from the history database instead of fetching from providers.
- `--fresh`: fetch directly from providers (bypass daemon), write to DB.
- The CLI emits a note to stderr if the daemon is running: `Reading from daemon (last poll 2m ago)`.

### 4.3 Global Options

| Flag | Short | Description | Default |
|------|-------|-------------|---------|
| `--provider` | `-p` | Comma-separated list of providers to query | All configured |
| `--config` | `-c` | Path to config file | `~/.config/llm-monitor/config.toml` |
| `--fresh` | `-f` | Bypass cache/daemon, force direct API calls | `false` |
| `--no-colour` | | Disable colour output | Auto-detect (see 4.2.2) |
| `--colour=always` | | Force colour output even when piped | `false` |
| `--quiet` | `-q` | Suppress all non-error stderr output | `false` |
| `--verbose` | `-v` | Verbose logging to stderr (mutually exclusive with `-q`) | `false` |
| `--version` | `-V` | Print version and exit | |
| `--help` | `-h` | Print help and exit | |
| `--notify` | `-n` | Enable desktop notifications on status changes | `false` |
| `--interval` | `-i` | UI refresh interval in seconds (monitor mode display refresh) | 30 |
| `--list-providers` | | List available providers and their status | |
| `--report` | | Show usage report from history (alias for `history report`) | |
| `--days` | | Number of days for `--report` | 7 |
| `--no-history` | | Disable history recording for this invocation | `false` |
| `--clear-cache` | | Delete all cached data and exit | |

`--quiet` and `--verbose` are mutually exclusive. If both are provided, exit with an error.

### 4.4 Signal Handling

The tool handles Unix signals gracefully, particularly in `--monitor` and `--ux` modes.

| Signal | Behaviour |
|--------|-----------|
| `SIGINT` (Ctrl+C) | Clean shutdown. Flush pending cache writes, restore terminal state (Rich cleanup), exit code 130. |
| `SIGTERM` | Clean shutdown, same as SIGINT. Exit code 143. |
| `SIGHUP` | Reload configuration file without restarting. Re-read `config.toml` and re-initialise providers. Log reload to stderr. |
| `SIGPIPE` | Silently exit with code 0. Occurs when piping to `head`, `grep -q`, etc. Python raises `BrokenPipeError` by default; this must be caught and handled cleanly without a traceback. |
| `SIGUSR1` | Force refresh all providers (equivalent to pressing `r` in monitor mode). |

**Terminal state restoration:** In `--monitor` mode, if the process exits abnormally, the terminal may be left with a hidden cursor or in an alternate screen buffer. Use `atexit` and signal handlers to ensure Rich's cleanup always runs.

### 4.5 Error Message Format

All user-facing error messages follow a structured format:

```
Error: <what went wrong>
<why it matters / context>
Fix: <actionable remediation command>
Docs: <URL if applicable>
```

**Example:**
```
Error: Claude credentials not found at ~/.claude/.credentials.json
The Claude provider requires Claude Code to be installed and authenticated.
Fix: Install Claude Code and run 'claude /login' to authenticate.
Docs: https://code.claude.com/docs/en/authentication
```

Errors MUST NOT include stack traces unless `--verbose` is set. Unhandled exceptions are caught at the top level, logged via `--verbose`, and presented as a clean error message otherwise.

### 4.6 Configuration File

Location: `~/.config/llm-monitor/config.toml`

Overridable via `$LLM_MONITOR_CONFIG` environment variable.

The config file MUST NEVER contain API keys or secrets. Credentials are resolved indirectly via `key_command`, `key_env`, or system keyring. See Section 7 (Security Model) for the full credential resolution hierarchy.

**Environment variable overrides for paths:**

| Variable | Overrides | Default |
|----------|-----------|---------|
| `LLM_MONITOR_CONFIG` | Config file path | `$XDG_CONFIG_HOME/llm-monitor/config.toml` |
| `LLM_MONITOR_DATA_DIR` | History DB directory | `$XDG_DATA_HOME/llm-monitor/` |
| `LLM_MONITOR_CACHE_DIR` | Cache directory | `$XDG_CACHE_HOME/llm-monitor/` |
| `LLM_MONITOR_LOG_LEVEL` | Daemon log level | `info` |

These variables take precedence over XDG defaults and config file values. They are particularly useful for Docker deployments (Section 15).

```toml
[general]
default_providers = ["claude"]
poll_interval = 600              # 10 minutes; applies to all providers unless overridden
notification_enabled = false

[thresholds]
warning = 70
critical = 90

[notifications]
on_warning = true
on_critical = true
on_reset = true
sound = false

# ─── Daemon ───────────────────────────────────────────────────
[daemon]
log_file = ""                    # empty = default ($XDG_STATE_HOME/llm-monitor/daemon.log)
pid_file = ""                    # empty = default ($XDG_RUNTIME_DIR/llm-monitor/daemon.pid)

# ─── Provider: Claude ──────────────────────────────────────────
[providers.claude]
enabled = true
# poll_interval = 600            # override global default (optional)
credentials_path = ""            # empty = default (~/.claude/.credentials.json)
show_opus = true
# Claude uses its own credential file exclusively; no key_* fields

# ─── Provider: Grok (xAI) ─────────────────────────────────────
[providers.grok]
enabled = false
key_env = "XAI_API_KEY"
# key_command = "secret-tool lookup application llm-monitor provider grok"
# key_keyring = true             # use system keyring (default: true)

# ─── Provider: OpenAI ─────────────────────────────────────────
[providers.openai]
enabled = false
key_env = "OPENAI_API_KEY"
# key_command = "pass show llm-monitor/openai"
# key_keyring = true

# ─── Provider: Ollama ─────────────────────────────────────────
[providers.ollama]
enabled = false
poll_interval = 60               # local service, can poll more frequently
# Simple: single host
host = "http://localhost:11434"
# Advanced: multiple hosts (uncomment and use instead of 'host')
# [[providers.ollama.hosts]]
# name = "workstation"
# url = "http://localhost:11434"
# [[providers.ollama.hosts]]
# name = "gpu-server"
# url = "http://gpu-server.local:11434"

# ─── Provider: Local System ───────────────────────────────────
[providers.local]
enabled = false
poll_interval = 60               # local metrics, can poll more frequently
show_gpu = true
show_cpu = true
show_ram = true
gpu_backend = "auto"             # "nvidia" | "amd" | "auto"
# No credentials required

# ─── History ───────────────────────────────────────────────────
[history]
enabled = true
retention_days = 90

# ─── Monitor Mode ─────────────────────────────────────────────
[monitor]
compact = false
show_sparkline = true

# ─── GTK/UX Mode (v2) ─────────────────────────────────────────
[ux]
autostart = false
start_minimised = true
```

**Poll interval design:** A single `poll_interval` replaces the former `cache_ttl` and `refresh_interval` fields. In daemon mode, this controls how often the daemon fetches from each provider. In standalone mode (no daemon), it serves as the cache TTL — the CLI won't re-fetch if the cached data is younger than `poll_interval`. The default of 600 seconds (10 minutes) is appropriate for all cloud providers: usage data changes slowly, and Claude's aggressive rate limiting (Section 3.1) makes frequent polling counterproductive. Local providers (Ollama, Local) default to 60 seconds since they have no rate limits and report real-time operational state.

---

## 5. Caching Strategy

### 5.1 Overview

Caching serves two modes differently:

- **Daemon mode:** The daemon writes fetched data to the history database. The cache layer is not used — the database is the canonical store. CLI reads from the database.
- **Standalone mode (no daemon):** Each provider maintains a cache file. The cache TTL equals the provider's `poll_interval`. This prevents redundant API calls when the CLI is invoked multiple times in quick succession.

### 5.2 Cache Location

`~/.cache/llm-monitor/<provider>/last.json`

Follows XDG Base Directory specification. Respects `$XDG_CACHE_HOME` and `$LLM_MONITOR_CACHE_DIR` (see Section 4.6).

### 5.3 Cache Behaviour (Standalone Mode)

- On fresh fetch: write to cache atomically (see 7.4) with current timestamp.
- On cache hit (within `poll_interval`): return cached data, set `cached: true`.
- On 429 / rate limit: return cached data regardless of age, enter backoff state (see Section 3.1), append warning to `errors`.
- On network error: return cached data if available, append error.
- On `--fresh` flag: bypass cache, force API call. Still writes to cache.
- On `--clear-cache` flag: delete all cache files and exit.

### 5.4 Cache Security

- Cache files MUST NOT contain raw API responses verbatim. Store only parsed, sanitised `ProviderStatus` data.
- Cache files MUST NOT contain credentials, tokens, or API keys under any circumstances.
- Cache files are written with `0o600` permissions using the atomic write pattern from Section 7.4.
- Use `fcntl.flock()` (advisory locking) when reading/writing cache files to prevent corruption from concurrent access. Lock with `LOCK_SH` for reads, `LOCK_EX` for writes, with a 2-second timeout before falling back to stale data.

---

## 6. History and Reporting

### 6.1 Overview

The tool maintains a local SQLite database of usage samples over time, enabling trend analysis, historical reporting, and visualisations such as sparklines and charts. Every successful provider fetch where data has meaningfully changed writes a row to the history database.

This is **usage pattern data**, not credentials. It reveals work habits (when and how heavily LLM services are used) and should be treated accordingly - stored securely but not with the same rigour as API keys.

### 6.2 Storage Location

`~/.local/share/llm-monitor/history.db`

Follows XDG Base Directory specification. Respects `$XDG_DATA_HOME` if set. Created with `0o600` permissions using the secure file I/O pattern from Section 7.4. The directory is created with `0o700`.

### 6.3 Database Schema

```sql
-- Usage samples: one row per provider per window per fetch
CREATE TABLE usage_samples (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    provider      TEXT NOT NULL,
    timestamp     TEXT NOT NULL,              -- ISO 8601 UTC
    window_name   TEXT NOT NULL,
    utilisation   REAL,
    status        TEXT,
    unit          TEXT NOT NULL,
    raw_value     REAL,
    raw_limit     REAL,
    resets_at     TEXT,
    cached        INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX idx_samples_provider_time ON usage_samples(provider, timestamp);
CREATE INDEX idx_samples_time ON usage_samples(timestamp);

-- Per-model usage breakdown (populated when providers supply it)
CREATE TABLE model_usage (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    provider      TEXT NOT NULL,
    timestamp     TEXT NOT NULL,
    model         TEXT NOT NULL,
    input_tokens  INTEGER,
    output_tokens INTEGER,
    total_tokens  INTEGER,
    cost          REAL,
    request_count INTEGER,
    period        TEXT
);

CREATE INDEX idx_model_usage_provider_time ON model_usage(provider, timestamp);
CREATE INDEX idx_model_usage_model ON model_usage(model);

-- Provider extras: per-fetch metadata (plan name, models loaded, etc.)
CREATE TABLE provider_extras (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    provider      TEXT NOT NULL,
    timestamp     TEXT NOT NULL,
    extras_json   TEXT NOT NULL
);

CREATE INDEX idx_extras_provider_time ON provider_extras(provider, timestamp);

-- Schema version tracking for migrations
CREATE TABLE schema_version (
    version       INTEGER NOT NULL
);
```

**Design notes:**
- Each `UsageWindow` from a provider fetch becomes a separate row in `usage_samples`. A Claude fetch that returns three windows (Session, Weekly, Opus) inserts three rows sharing the same `timestamp`.
- Each `ModelUsage` entry becomes a row in `model_usage`, enabling per-model trend analysis over time.
- The `provider_extras` table stores the per-fetch `extras` dict as serialised JSON. This avoids schema changes when providers add new fields.
- `schema_version` enables forward-compatible migrations as the schema evolves.
- All timestamps are stored in UTC ISO 8601 format for unambiguous ordering and portability.

### 6.4 Write Behaviour

**Meaningful-change detection:** A sample is written when the fetched data differs meaningfully from the most recent row for that provider+window in the database. The comparison is:
- Utilisation delta > 0.1% (absolute), OR
- Status value changed (e.g., `normal` → `warning`), OR
- Window reset detected (`resets_at` changed to a later time)

The "last known" values are loaded into memory on startup (one `SELECT` per provider+window for the most recent row) and kept in a dict. This avoids a database query on every write — only the initial load and subsequent inserts update the in-memory state. If the history database is empty or disabled, all fetches are treated as changes.

- Cached responses (where `cached: true` and no change detected) are NOT written to history.
- Writes use WAL (Write-Ahead Logging) mode for concurrent read safety: `PRAGMA journal_mode=WAL`.
- Writes are performed in a transaction to ensure atomicity across the multi-row insert per provider.
- The history database is NOT locked with `fcntl.flock()` (SQLite handles its own locking internally).

### 6.5 Retention Policy

- Default retention: 90 days.
- Configurable via `config.toml`:
  ```toml
  [history]
  enabled = true
  retention_days = 90
  ```
- On startup, the tool runs a pruning pass: `DELETE FROM usage_samples WHERE timestamp < datetime('now', '-90 days')`. Same for `model_usage` and `provider_extras`.
- `PRAGMA auto_vacuum = INCREMENTAL` is set on database creation to reclaim disk space after pruning.
- At 10-minute intervals across 3 cloud providers with 3 windows each, plus 2 local providers at 1-minute intervals, 90 days produces roughly 400,000 rows - well within SQLite's comfort zone.

### 6.6 Disabling History

History collection can be disabled entirely:

```toml
[history]
enabled = false
```

Or via the CLI flag `--no-history` for a single invocation. When disabled, the database file is not created and no writes occur. Existing history is preserved (not deleted) when disabling.

### 6.7 History CLI Commands

#### `llm-monitor history report` (aliased as `llm-monitor --report`)

Display a summary report of usage over time.

```bash
llm-monitor --report
llm-monitor --report --days 30 --provider claude
llm-monitor --report --days 30 --format csv > usage-march.csv
llm-monitor --report --days 7 --provider claude --format json
```

**Report flags:**

| Flag | Description | Default |
|------|-------------|---------|
| `--days` | Number of days to report on | 7 |
| `--from` | Start date (ISO 8601 or YYYY-MM-DD) | (derived from `--days`) |
| `--to` | End date (ISO 8601 or YYYY-MM-DD) | now |
| `--format` | Output format: `table`, `json`, `csv` | `table` |
| `--provider` | Filter to specific provider(s) | All with history |
| `--window` | Filter to specific window name | All windows |
| `--granularity` | Aggregation: `raw`, `hourly`, `daily` | `daily` |
| `--models` | Include per-model breakdown in output | `false` |

**Example table output:**
```
LLM Usage Report                     29 Mar - 05 Apr 2026
═══════════════════════════════════════════════════════════

 Anthropic Claude
───────────────────────────────────────────────────────────
 Session (5h)    avg 38%   peak 94%   exceeded 2x
                 ▂▃▅▇▅▃▂▃▅▆▄▃▂▁▂▃▅▇█▇▅▃▂▁▂▃▅▆▅▃
 Weekly (7d)     avg 52%   peak 81%   exceeded 0x
                 ▃▃▄▄▅▅▅▆▆▆▇▇▇▇▇▆▆▅▅▅▄▄▃▃▃▃▄▅▅▆

 Ollama (workstation + gpu-server)
───────────────────────────────────────────────────────────
 GPU VRAM        avg 64%   peak 92%
                 ▅▅▆▆▇▇▇▇▆▅▅▃▁▁▁▅▅▆▇▇▇▇▆▅▅▃▁▁▅▆
 Inference       avg 42 tok/s   peak 51 tok/s

═══════════════════════════════════════════════════════════
 Period: 7 days │ Samples: 4,218 │ DB size: 1.2 MB
```

#### `llm-monitor history purge`

Permanently delete all history data. Requires explicit confirmation to prevent accidental data loss.

**Interactive mode (default):**
```
$ llm-monitor history purge

WARNING: This will permanently delete all usage history.
  Database: ~/.local/share/llm-monitor/history.db
  Records:  14,832 samples across 3 providers
  Oldest:   2026-01-05
  Size:     4.2 MB

This action cannot be undone.

Type 'purge' to confirm: purge
History purged successfully.
```

If the user types anything other than exactly `purge` (case-sensitive), the operation is aborted:
```
Type 'purge' to confirm: yes
Aborted. History was not modified.
```

**Non-interactive / scripted mode:**
```bash
llm-monitor history purge --confirm
```

The `--confirm` flag bypasses the interactive prompt. Without it, the tool requires the interactive typed confirmation above.

When `stdin` is not a TTY (piped context) and `--confirm` is not provided, the interactive prompt is skipped and the tool exits with an error:
```
Error: history purge requires interactive confirmation.
Fix: Use --confirm to bypass: llm-monitor history purge --confirm
```

#### `llm-monitor history stats`

Quick summary of the history database.

```
$ llm-monitor history stats

History Database: ~/.local/share/llm-monitor/history.db
  Size:       4.2 MB
  Samples:    14,832
  Providers:  claude, openai, ollama
  Oldest:     2026-01-05T08:12:00Z
  Newest:     2026-04-05T10:25:00Z
  Retention:  90 days (next prune removes 0 records)
```

#### `llm-monitor history export`

Full raw export of the database for backup or migration. Export always includes all data — no `--models` or `--provider` filtering (that's what `--report` is for). Export is a full dump for backup and migration.

```bash
llm-monitor history export --format sql > backup.sql
llm-monitor history export --format jsonl > backup.jsonl
llm-monitor history export --format csv > backup.csv
```

**Export formats and column structure:**

**SQL** — Valid `INSERT INTO` statements, one per row. Produces a standalone script that can recreate the data in any SQLite database with the same schema. Includes `CREATE TABLE IF NOT EXISTS` preamble.

**CSV** — Two logical sections separated by a blank line, each with its own header row. The first section is `usage_samples`, the second is `model_usage`. `provider_extras` rows are omitted from CSV (the JSON blob doesn't map well to flat columns). NULL values are rendered as empty strings.

`usage_samples` columns:
```
id,provider,timestamp,window_name,utilisation,status,unit,raw_value,raw_limit,resets_at,cached
```

`model_usage` columns:
```
id,provider,timestamp,model,input_tokens,output_tokens,total_tokens,cost,request_count,period
```

**JSONL** — One JSON object per line. Each line includes a `"type"` discriminator field to distinguish record types. All three tables are included. NULL values are rendered as JSON `null`. Timestamps in ISO 8601 UTC.

```jsonl
{"type":"usage_sample","id":1,"provider":"claude","timestamp":"2026-04-01T10:00:00Z","window_name":"Session (5h)","utilisation":42.3,"status":"normal","unit":"percent","raw_value":null,"raw_limit":null,"resets_at":"2026-04-01T15:00:00Z","cached":false}
{"type":"model_usage","id":1,"provider":"claude","timestamp":"2026-04-01T10:00:00Z","model":"claude-opus-4-6","input_tokens":15000,"output_tokens":8000,"total_tokens":23000,"cost":null,"request_count":12,"period":"5h"}
{"type":"provider_extras","id":1,"provider":"claude","timestamp":"2026-04-01T10:00:00Z","extras":{"plan":"Pro","token_expires_at":"2026-04-01T12:00:00Z"}}
```

### 6.8 Report Aggregation Algorithms

When `--granularity` is `hourly` or `daily`, multiple raw samples within a time bucket are aggregated. Different fields have different semantics and require different algorithms.

**Key insight:** The `raw_value` and token count fields in this system are **running totals** within a provider window (e.g., "tokens used this week so far"), not per-interval deltas. Aggregation must account for this — summing running totals would double-count.

| Field | Algorithm | Rationale |
|-------|-----------|-----------|
| `utilisation` | **mean** | Average usage over the bucket gives the truest picture of sustained load. |
| `status` | **max severity** (`exceeded` > `critical` > `warning` > `normal`) | A bucket that hit `exceeded` even once should surface that — worst-case is what matters for alerting and review. |
| `raw_value` | **last** (most recent sample in bucket) | Running total within a window; the last sample is the most current reading. |
| `raw_limit` | **last** | Limits rarely change mid-bucket; the latest value is authoritative. |
| `resets_at` | **last** | The most recent reset timestamp is the relevant one. |
| `input_tokens` | **max** | Running total within a window — max captures the high-water mark, not a per-interval delta. |
| `output_tokens` | **max** | Same as `input_tokens`. |
| `total_tokens` | **max** | Same as `input_tokens`. |
| `cost` | **max** | Cumulative within a window, not incremental. |
| `request_count` | **max** | Same reasoning as tokens. |

**Metadata included in aggregated output:**
- `sample_count`: number of raw samples in the bucket (data density indicator)
- `bucket_start` / `bucket_end`: ISO 8601 timestamps for the aggregation window

**Bucket boundaries:**
- **Hourly:** aligned to clock hours in UTC (e.g., `10:00:00Z` to `10:59:59Z`)
- **Daily:** aligned to calendar days in UTC (e.g., `2026-04-01T00:00:00Z` to `2026-04-01T23:59:59Z`)

**Delta-based analysis** (e.g., "tokens consumed per hour") is a derived calculation on top of aggregated data. This is deferred to v1.x and not part of the v0.2.0 aggregation engine.

---

## 7. Security Model

### 7.1 Threat Model

The tool handles sensitive credentials (OAuth tokens, API keys) for cloud services that incur real financial cost. The threat model considers:

**In scope:**
- Credential theft from disk (plaintext files, world-readable permissions).
- Credential leakage via logs, error messages, JSON output, or stack traces.
- Credential leakage via process environment (`/proc/$PID/environ`).
- Man-in-the-middle attacks on API connections.
- Race conditions on shared credential/cache files.
- Inadvertent inclusion of secrets in cache files.
- Shell injection via `key_command` configuration.

**Out of scope:**
- Root-level compromise (attacker with root can read kernel memory, keyring, etc.).
- Supply-chain attacks on Python dependencies (mitigated by standard packaging practices).
- Physical access attacks.

### 7.2 Credential Storage Hierarchy

Credentials MUST NOT be stored in plaintext configuration files. The tool uses a tiered credential resolution strategy, from most secure to least:

**Tier 1: System keyring (preferred)**

Use the Python `keyring` library, which interfaces with the D-Bus Secret Service API on Linux (GNOME Keyring, KDE Wallet, KeePassXC, etc.).

```bash
# User stores a key (one-time setup)
llm-monitor config set-key --provider openai
# Prompts securely for the key, stores via keyring

# Or via secret-tool directly
secret-tool store --label="llm-monitor: OpenAI API Key" \
    application llm-monitor provider openai
```

**Tier 2: Command-based credential helper (`key_command`)**

The config file may contain a `key_command` directive that executes a shell command to retrieve the key. This supports vault integrations, password managers, and custom credential helpers.

```toml
[providers.openai]
key_command = "pass show llm-monitor/openai"
# or: key_command = "secret-tool lookup application llm-monitor provider openai"
# or: key_command = "vault kv get -field=api_key secret/llm-monitor/openai"
```

**Tier 3: Environment variables**

Standard practice for CI/CD and containerised environments. Acknowledged risk: readable via `/proc/$PID/environ` by the same user. Acceptable for ephemeral contexts.

```bash
export OPENAI_API_KEY="sk-..."
export XAI_API_KEY="xai-..."
```

**Tier 4: Claude Code credential file (Claude provider only)**

Read-only access to `~/.claude/.credentials.json`. This file is owned and managed by Claude Code, not by llm-monitor. The tool reads tokens but never writes to this file (see D-036).

**Resolution order per provider:**
1. `key_command` (if configured, execute and read stdout)
2. `key_env` / well-known env var (e.g., `$OPENAI_API_KEY`)
3. System keyring lookup (`keyring.get_password(...)`)
4. Provider-specific credential file (Claude only)

If no credential is found, the provider reports `is_configured() = False` and the tool emits setup instructions via `auth_instructions()`.

### 7.3 Credential Sanitisation

**Rule: No secret shall ever appear in output, logs, cache files, error messages, or stack traces.**

Implementation requirements:
- API keys and tokens MUST be wrapped in `SecretStr` (see Section 2.2) which overrides `__repr__` and `__str__` to return fully-masked values (`SecretStr('***')` and `***REDACTED***` respectively).
- JSON output MUST NOT contain credentials. The `extras` dict must be filtered before serialisation.
- Stack traces displayed to the user (via `--verbose` or unhandled exceptions) MUST pass through a sanitisation filter that replaces any string matching known credential patterns.
- Log output (stderr in `--verbose` mode) MUST redact credentials using the same filter.
- The `key_command` directive's stdout is read into a `SecretStr` immediately; the raw subprocess output is never stored as a plain string.

**Credential patterns to redact (regex):**
```python
REDACTION_PATTERNS = [
    r"sk-ant-oat\S+",         # Claude OAuth access token
    r"sk-ant-ort\S+",         # Claude OAuth refresh token
    r"sk-ant-api\S+",         # Anthropic API key
    r"sk-ant-admin\S+",       # Anthropic Admin API key
    r"sk-[a-zA-Z0-9-]{20,}",  # OpenAI API key
    r"xai-[a-zA-Z0-9-]{20,}", # xAI API key
    r"Bearer\s+\S+",          # Any bearer token in logs
]
```

### 7.4 File Security

**Created files:**
All files created by the tool MUST be created with restrictive permissions from the outset, not retroactively changed. Use `os.open()` with explicit mode `0o600` followed by `os.fdopen()`. Do not use `open()` then `chmod()` (race condition between creation and permission change). Directories use mode `0o700`.

```python
import os

def secure_write(path: str, data: str) -> None:
    """Write data to file with 0600 permissions, atomically."""
    tmp_path = path + ".tmp"
    fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(data)
        os.rename(tmp_path, path)  # atomic on same filesystem
    except Exception:
        os.unlink(tmp_path)
        raise

def secure_mkdir(path: str) -> None:
    """Create directory with 0700 permissions."""
    os.makedirs(path, mode=0o700, exist_ok=True)
```

**Permission enforcement on existing files:**
- On startup, check permissions on config file, cache directory, and Claude credentials file.
- If config file permissions are more permissive than `0o600`, emit a **warning to stderr** on every invocation:
  ```
  Warning: Config file has loose permissions (0644): ~/.config/llm-monitor/config.toml
  Other users on this system could read your configuration, which may contain credential command paths.
  Fix: chmod 600 ~/.config/llm-monitor/config.toml
  ```
  The tool continues to run. The config file contains no secrets by design (D-016), so this is a defence-in-depth warning rather than a hard failure. The `--quiet` flag suppresses this warning.
- If Claude credential file permissions are more permissive than `0o600`, emit a **warning to stderr** (do not refuse, as Claude Code manages this file).
- In containerised environments (detected via `/.dockerenv` or `$LLM_MONITOR_CONTAINER=1`), permission checks are skipped entirely — container volume mounts have their own permission model.

**File locking:**
Use `fcntl.flock()` (advisory locking) when reading/writing cache files to prevent corruption from concurrent access (e.g., two standalone CLI invocations running simultaneously). Lock with `LOCK_SH` for reads, `LOCK_EX` for writes, with a 2-second timeout before falling back to stale data. Note: the tool never writes to credential files (D-036), so credential file locking is not needed.

### 7.5 Network Security

- All HTTPS connections MUST verify TLS certificates. Do not set `verify=False` in httpx under any circumstances.
- The tool MUST NOT follow redirects on credential-bearing requests. Set `follow_redirects=False` on httpx requests that include `Authorization` headers to prevent credential forwarding to unexpected hosts.
- Connection timeouts MUST be set explicitly (connect: 10s, read: 30s, pool: 5s) to prevent hanging.
- The tool MUST NOT send credentials to any host other than the provider's documented API domain. Validate the URL scheme and host before attaching credentials.

**Allowed hosts per provider:**

| Provider | Allowed Hosts | Scheme |
|----------|--------------|--------|
| Claude | `api.anthropic.com` | HTTPS only |
| OpenAI | `api.openai.com` | HTTPS only |
| Grok | `api.x.ai` | HTTPS only |
| Ollama | `localhost`, `127.0.0.1`, `[::1]`, or user-configured host | HTTP or HTTPS |

### 7.6 Process Security

- Secrets loaded into memory SHOULD be held in as few variables as possible and for as short a duration as possible.
- After a provider fetch completes, the raw HTTP response object should be dereferenced promptly. Do not cache full response objects; cache only the parsed, sanitised data model.
- The tool MUST NOT write secrets to temporary files. Use in-memory processing exclusively.
- When the `key_command` directive is used, the subprocess MUST be run with `shell=False` (pass as a list via `shlex.split()`) to prevent shell injection. On failure, log only stderr from the subprocess (never stdout, which contains the secret).

```python
import shlex, subprocess

def run_key_command(command: str) -> SecretStr:
    """Execute a key command securely and return the key as SecretStr."""
    args = shlex.split(command)
    try:
        result = subprocess.run(
            args, capture_output=True, text=True,
            timeout=10, shell=False,  # NEVER use shell=True
        )
    except subprocess.TimeoutExpired:
        raise CredentialError(f"key_command timed out after 10s: {command}")
    if result.returncode != 0:
        raise CredentialError(
            f"key_command failed (exit {result.returncode}): {result.stderr.strip()}"
            # Note: stderr only, never stdout (which contains the secret)
        )
    return SecretStr(result.stdout.strip())
```

### 7.7 Claude Token Handling

The tool does **not** self-manage Claude token refresh. The `~/.claude/.credentials.json` file is owned and managed by Claude Code. Writing to it from a second process introduces race conditions (Claude Code uses Node.js and does not use POSIX `flock`; advisory locking only works when all processes opt in).

Instead, the tool is a **read-only consumer** of Claude Code's credentials:

- Read `expiresAt` from the credentials file before each API call. The token is wrapped in `SecretStr` immediately on read.
- If the token has expired or will expire within 5 minutes, skip the API call and report a clear error:
  ```
  Error: Claude OAuth token has expired.
  Claude Code manages token refresh automatically when running.
  Fix: Run 'claude /login' to refresh your credentials.
  ```
- On HTTP 401 from the API, re-read the credentials file (Claude Code may have refreshed the token concurrently between our expiry check and the API call). If the re-read token differs, retry once with the new token. If it still fails, report the auth error.
- The tool NEVER writes to `~/.claude/.credentials.json`.

---

## 8. Error Handling

### 8.1 Per-Provider Errors

Each provider handles errors independently. A failure in one provider does not prevent others from returning data. The top-level exit code reflects the aggregate state.

| Scenario | Behaviour |
|----------|-----------|
| Provider not configured | Skip silently unless explicitly requested via `--provider` |
| Credentials missing | Provider returns error in its `errors` list with setup instructions; other providers unaffected |
| Claude token expired | Report auth error with `claude /login` fix command (see Section 7.7) |
| Claude token expired + 401 retry | Re-read credential file once (Claude Code may have refreshed); if still expired, report error |
| `key_command` fails or times out | Hard error for that provider; do not silently fall through to other tiers (see Section 2.2) |
| API returns 429 | Return cached/DB data, enter backoff state (see Section 3.1) |
| API returns 5xx | Return cached/DB data with error detail |
| Network unreachable | Return cached/DB data with error detail |
| Unexpected response schema | Log warning, attempt partial parse, include raw (sanitised) in `extras` |
| Local service (Ollama) not running | Provider reports "service unavailable" |
| Config file insecure permissions | Warning to stderr (see Section 7.4) |
| Credential leakage detected in output | Abort and redact before writing to any stream |
| Daemon already running | `daemon start` reports PID and exits with error |
| Daemon not running | CLI falls back to standalone mode (direct fetch) |

### 8.2 Exit Code Matrix

| Condition | Exit Code |
|-----------|-----------|
| All requested providers succeed | 0 |
| Config/parse error prevents startup | 1 |
| All requested providers fail auth | 2 |
| Some providers succeed, some fail | 3 |
| All providers fail (network/unreachable) | 4 |
| SIGINT received | 130 |
| SIGTERM received | 143 |

---

## 9. Technology Stack

| Layer | Choice | Rationale |
|-------|--------|-----------|
| Language | Python 3.10+ | Broad Linux availability, Rich ecosystem, PyGObject bindings |
| HTTP client | `httpx` | Async-capable, modern, timeout/retry support |
| CLI framework | `click` or `typer` | Clean argument parsing, help generation, shell completion |
| Terminal UI | `rich` | Progress bars, tables, Live display, colour support |
| Configuration | `tomllib` (3.11+) / `tomli` | TOML standard for Python tooling |
| History store | `sqlite3` (stdlib) | Zero-dependency, single-file, SQL query support for time-series reporting |
| Credential storage | `keyring` | Cross-DE Linux keyring integration (GNOME Keyring, KDE Wallet, KeePassXC) |
| Secret types | Custom `SecretStr` class | Prevents accidental credential leakage in repr/str/json |
| File locking | `fcntl` (stdlib) | POSIX advisory locking for concurrent access safety |
| Subprocess | `shlex` + `subprocess` (stdlib) | Secure command execution for `key_command` |
| GPU metrics | `pynvml` (NVIDIA) / `rocm_smi` subprocess (AMD) | Native GPU access |
| System metrics | `psutil` | Cross-platform CPU/RAM/disk |
| GTK (v2) | PyGObject (GTK4 + libadwaita) | Native GNOME integration |
| Build backend | `hatchling` + `hatch-vcs` | Modern Python build system with git-tag version derivation |
| Packaging | `uv` | Fast dependency resolution, `uv tool install` for CLI apps, replaces pipx |
| Testing | `pytest` + `respx` | Async HTTP mocking |
| Containerisation | Docker | Optional deployment for headless/server environments |
| Service management | systemd (user units) | Daemon lifecycle on Linux desktop/server |

---

## 10. Project Structure

```
llm-monitor/
├── pyproject.toml
├── README.md
├── LICENSE
├── CHANGELOG.md
├── SPEC.md                          # This document
├── src/
│   └── llm_monitor/
│       ├── __init__.py
│       ├── __main__.py              # Entry point, signal handlers
│       ├── cli.py                   # CLI argument parsing and mode dispatch
│       ├── core.py                  # Orchestrator: load providers, aggregate results
│       ├── daemon.py                # Background service: poll loop, PID file, systemd install
│       ├── models.py                # UsageWindow, ProviderStatus, SecretStr
│       ├── security.py              # Credential resolution, sanitisation, secure I/O
│       ├── history.py               # SQLite history store, reporting, purge
│       ├── cache.py                 # Per-provider cache with TTL and file locking (standalone mode)
│       ├── config.py                # TOML config loader with permission checks
│       ├── notifications.py         # Desktop notification integration
│       │
│       ├── providers/
│       │   ├── __init__.py          # Provider base class + registry
│       │   ├── base.py              # Abstract Provider class with resolve_credential()
│       │   ├── claude.py            # Anthropic Claude provider
│       │   ├── grok.py              # xAI Grok provider
│       │   ├── openai.py            # OpenAI provider
│       │   ├── ollama.py            # Ollama provider
│       │   └── local.py             # Local system metrics provider
│       │
│       ├── formatters/
│       │   ├── __init__.py
│       │   ├── json_fmt.py          # JSON output formatter
│       │   ├── table_fmt.py         # Rich table formatter (TTY-adaptive)
│       │   └── monitor_fmt.py       # Rich Live TUI formatter
│       │
│       └── gtk/                     # v2
│           ├── __init__.py
│           ├── app.py               # GTK Application
│           ├── indicator.py         # System tray indicator
│           └── popover.py           # Detail popover widget
│
├── tests/
│   ├── conftest.py
│   ├── test_core.py
│   ├── test_cache.py
│   ├── test_daemon.py               # Daemon lifecycle, poll loop, PID management
│   ├── test_security.py             # Credential sanitisation, SecretStr, file perms
│   ├── test_history.py              # History store, reporting, purge, retention
│   ├── test_formatters.py
│   ├── providers/
│   │   ├── test_claude.py
│   │   ├── test_grok.py
│   │   ├── test_openai.py
│   │   ├── test_ollama.py
│   │   └── test_local.py
│   └── fixtures/
│       ├── claude_credentials.json
│       ├── claude_usage_response.json
│       ├── openai_usage_response.json
│       ├── ollama_ps_response.json
│       └── config_full.toml
│
├── Dockerfile
├── docker-compose.yml
│
└── assets/
    ├── llm-monitor.desktop          # XDG autostart (v2)
    └── icons/                       # Tray icons (v2)
```

### 10.1 Installation

```bash
# Default install - cloud providers (Claude, Grok, OpenAI, Ollama API)
uv tool install llm-monitor

# With local GPU/system metrics (adds psutil, pynvml)
uv tool install "llm-monitor[local]"

# Everything (local metrics + GTK desktop frontend)
uv tool install "llm-monitor[all]"

# Via pip (if uv is not available)
pip install llm-monitor --user

# From source (development)
git clone https://github.com/<user>/llm-monitor.git
cd llm-monitor
uv sync
```

**Dependency groups in `pyproject.toml`:**

The default install is lightweight: cloud providers and Ollama API monitoring with no native library dependencies. Extras are additive — they add capabilities on top of the base.

| Install | Extra Packages | What's Added |
|---------|----------------|--------------|
| (default) | `httpx`, `rich`, `click`, `keyring`, `tomli` | Cloud providers (Claude, Grok, OpenAI), Ollama via API, daemon, full CLI |
| `[local]` | + `psutil`, `pynvml` | Local system metrics provider (GPU, CPU, RAM) |
| `[gtk]` | + `PyGObject` | GTK4/libadwaita desktop frontend (v2) |
| `[all]` | `[local]` + `[gtk]` | Everything |

**Rationale:** The base install has no compiled native dependencies (`psutil` and `pynvml` require C extensions and GPU drivers). This makes the default install work cleanly in containers, CI, and minimal environments. Users on a workstation who want GPU monitoring add `[local]`. This is the correct direction for extras: additive, not subtractive.

### 10.2 PyPI Distribution

The package is published to PyPI as `llm-monitor`. The build backend is `hatchling` with `hatch-vcs` for version derivation from git tags.

```toml
# pyproject.toml (key sections)
[build-system]
requires = ["hatchling", "hatch-vcs"]
build-backend = "hatchling.build"

[project]
name = "llm-monitor"
dynamic = ["version"]
description = "Monitor LLM service usage across providers from the CLI"
readme = "README.md"
requires-python = ">=3.10"
license = { text = "MIT" }
authors = [
    { name = "Daniel Thomas" },
]
keywords = ["llm", "monitoring", "claude", "openai", "ollama", "usage"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Environment :: Console",
    "Intended Audience :: Developers",
    "Operating System :: POSIX :: Linux",
    "Programming Language :: Python :: 3",
    "Topic :: System :: Monitoring",
]
dependencies = [
    "httpx>=0.27",
    "rich>=13.0",
    "click>=8.0",
    "keyring>=25.0",
    "tomli>=2.0; python_version < '3.11'",
]

[project.optional-dependencies]
local = [
    "psutil>=5.9",
    "pynvml>=11.5",
]
gtk = [
    "PyGObject>=3.42",
]
all = [
    "llm-monitor[local,gtk]",
]

[project.scripts]
llm-monitor = "llm_monitor.__main__:main"

[tool.hatch.version]
source = "vcs"

[tool.hatch.build.hooks.vcs]
version-file = "src/llm_monitor/_version.py"
```

### 10.3 Versioning and Release

**Single source of truth: git tags.**

The version is derived from git tags using `hatch-vcs` (a `setuptools-scm` equivalent for the `hatchling` build backend). There is no standalone `VERSION` file. The flow is:

1. Development builds get automatic versions like `0.1.0.dev12+g1a2b3c4` based on distance from the last tag.
2. To release, tag a commit: `git tag v0.1.0 && git push --tags`.
3. CI/CD (GitHub Actions) detects the tag, builds the sdist and wheel, and publishes to PyPI.
4. The built package contains a generated `src/llm_monitor/_version.py` with the exact version string.
5. The application reads its own version at runtime via `importlib.metadata.version("llm-monitor")`.

**Why not a `VERSION` file?**
A standalone file creates a synchronisation problem: someone bumps the file but forgets to tag, or tags without updating the file. Git tags are the canonical release mechanism and should be the single source. The `hatch-vcs` plugin derives the version from the tag automatically, eliminating the possibility of mismatch.

**GitHub Actions release workflow:**

```yaml
# .github/workflows/release.yml
name: Release to PyPI
on:
  push:
    tags: ["v*"]

jobs:
  publish:
    runs-on: ubuntu-latest
    permissions:
      id-token: write  # PyPI trusted publishing
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0  # full history needed for hatch-vcs

      - uses: astral-sh/setup-uv@v4

      - run: uv build

      - uses: pypa/gh-action-pypi-publish@release/v1
        # Uses trusted publishing (OIDC), no API token needed
```

**Semantic versioning rules:**

| Version Bump | When |
|-------------|------|
| Patch (0.1.x) | Bug fixes, documentation, dependency updates |
| Minor (0.x.0) | New provider, new CLI command, new feature |
| Major (x.0.0) | Breaking changes to JSON output schema, config format, or CLI flags |

The JSON output schema (Section 4.2.3) and config file format (Section 4.6) are considered stable interfaces from v1.0.0 onwards. Changes to these require a major version bump.

---

## 11. Open Questions

| ID | Question | Impact | Relates To | Status |
|----|----------|--------|------------|--------|
| OQ-001 | **How to surface Claude extra usage spend data?** No REST API exists. Options: (a) Playwright scrape of `claude.ai/settings/usage`; (b) leave as "not available" with web link; (c) wait for Anthropic to expose an endpoint. | High | Claude | Open |
| OQ-002 | **Is the `/api/oauth/usage` endpoint stable enough to depend on?** Undocumented, aggressively rate-limited. Could change or be removed. | High | Claude | Open |
| OQ-003 | **Should Claude token refresh be self-managed or rely on Claude Code?** Self-refreshing is more robust but adds complexity and credential file conflict risk. | Medium | Claude | **Closed (D-036):** Read-only consumer. Self-refresh introduces race conditions with Claude Code's Node.js process. |
| OQ-004 | **What is the actual rate limit on the Claude usage endpoint?** No documentation exists. Empirical testing needed. | Medium | Claude | Open |
| OQ-005 | **Multi-account support?** Should the tool support multiple credentials per provider (e.g., work vs personal Claude accounts)? | Low (v1) | All | Open |
| OQ-006 | **Claude plan detection?** The usage endpoint doesn't return plan type (Pro, Max5, Max20). Infer from thresholds or require user config? | Low | Claude | Open |
| OQ-007 | **Tmux/status-bar integration format?** A `--compact` single-line output could feed tmux `status-right`, polybar, waybar. What format is most useful? | Low (v1) | Output | Open |
| OQ-008 | **What does `utilization > 100` look like in Claude's API?** Does it exceed 100 when using extra usage, or cap at 100 with a separate indicator? | Medium | Claude | Open |
| OQ-009 | **GTK4 vs GTK3 for v2?** GTK4 + libadwaita is modern GNOME, but AppIndicator3 is GTK3. May need bridging or alternative tray approach. | Medium (v2) | GTK | Open |
| ~~OQ-010~~ | ~~**Licensing?** MIT vs GPL. MIT is simpler; GPL aligns with GNOME ecosystem.~~ | ~~Low~~ | ~~All~~ | **Closed:** MIT license committed to repo. `pyproject.toml` updated accordingly. |
| OQ-011 | **Does xAI have a programmatic billing/usage API?** Console shows spend, but no documented REST endpoint for querying balance or MTD cost found. Rate limit headers provide per-request data only. | High | Grok | Open |
| OQ-012 | **OpenAI billing endpoint stability?** `/v1/dashboard/billing/subscription` and `/v1/dashboard/billing/credit_grants` are undocumented but widely used. The official Usage API (`/v1/organization/usage/...`) requires admin-level access. | Medium | OpenAI | Open |
| OQ-013 | **Ollama metrics endpoint availability?** Ollama does not have a built-in `/metrics` Prometheus endpoint in all versions. The proxy approach (ollama-metrics) is an alternative but adds a dependency. Should we support both paths? | Medium | Ollama | Open |
| OQ-014 | **AMD GPU support depth?** `pyamdgpuinfo` is limited. `rocm-smi` subprocess parsing is more complete but slower. What level of AMD support is needed? | Low | Local | Open |
| ~~OQ-015~~ | ~~**Provider plugin system: entry_points vs explicit registration?** Entry points allow third-party providers but add packaging complexity. Explicit registration in a registry dict is simpler for v1.~~ | ~~Low~~ | ~~Architecture~~ | **Closed:** Explicit registration via `@register_provider` decorator and module-level dict for v1. See Section 2.2. |
| ~~OQ-016~~ | ~~**Unified cost normalisation?** Should the tool attempt to normalise costs across providers (e.g., all in USD) or keep each in its native unit? Normalisation is complex; native units are honest.~~ | ~~Low~~ | ~~Output~~ | **Closed (D-013):** Each provider keeps its native units. |
| OQ-017 | **Keyring availability in headless/server environments?** The Python `keyring` library requires a D-Bus Secret Service daemon. On headless servers or containers, keyring is unavailable. Should the tool fall back to encrypted file storage (e.g., `keyrings.alt`), or simply require env vars in those contexts? | Medium | Security | **Closed (D-038):** In containers, use env vars. Keyring is tier 3 in the resolution hierarchy; env vars (tier 2) and key_command (tier 1) cover headless cases. No need for `keyrings.alt`. |
| ~~OQ-018~~ | ~~**Should `key_command` support pipes and shell features?** Using `shell=False` prevents `secret-tool lookup ... \| head -1`. Users expecting shell syntax would need to wrap in `bash -c "..."`. Is this acceptable, or should we allow `shell=True` with a security warning?~~ | ~~Medium~~ | ~~Security~~ | **Closed (D-024, D-039):** `shell=False` is final. Users wrap in `bash -c "..."` if pipes are needed. |
| OQ-019 | **Should the `--insecure` flag be visible in `--help`?** Making it prominent encourages misuse. Alternatively, document it only in the man page and hide from `--help`. | Low | CLI UX | **Closed (D-018):** Permission checks are now warnings, not hard failures. The `--insecure` flag is no longer needed; `--quiet` suppresses the warning. |
| OQ-020 | **Should history support downsampling for long-term storage?** Keep 5-minute samples for 7 days, hourly averages for 90 days, daily averages for 1 year. Reduces disk usage for long retention but adds schema and query complexity. Could be a v1.x feature. | Low | History | Open |
| OQ-021 | **Should `--report` support chart output to terminal?** Rich can render basic bar charts. Alternatively, export to an HTML file with embedded charts. The sparkline in `--monitor` covers the basic case; full charts may be GTK v2 territory. | Low | History/Output | Open |
| OQ-022 | **Should per-model breakdown be available as a `--report --models` flag or a separate `llm-monitor models` subcommand?** The flag approach keeps reporting unified; a subcommand could offer richer model-specific analysis (cost per model per day, model switching patterns). | Low | History/Models | **Closed (D-043):** `--report --models` flag. Keeps reporting unified; a dedicated subcommand can be added later if richer model analysis is needed. |
| OQ-023 | **Should the daemon expose a health endpoint?** A simple HTTP endpoint (e.g., `localhost:9847/health`) would enable container health checks (`HEALTHCHECK` in Dockerfile) and monitoring by external tools. Adds a dependency (or use stdlib `http.server`). Alternative: health via `daemon status` exit code only. | Medium | Daemon/Docker | Open |
| OQ-024 | **Claude credentials in Docker: mount file or extract token to env var?** Mounting `~/.claude/.credentials.json:ro` is simpler but ties the container to a host path and Claude Code's file format. Extracting the OAuth token to an env var is more portable but requires manual refresh. | Medium | Docker/Claude | Open |
| OQ-025 | **Should the Docker image be published to Docker Hub, GHCR, or both?** GHCR is free for public repos and integrates with GitHub Actions. Docker Hub has broader discoverability. | Low | Docker | Open |
| OQ-026 | **Daemon socket vs DB-only IPC?** Current design has daemon write to DB, CLI read from DB (no socket). A Unix socket could enable `--fresh` to signal the daemon to fetch now, and `daemon status` to query live state. Adds complexity but enables richer interaction. DB-only is simpler and sufficient for v1. | Low | Daemon | Open |

---

## 12. Assumptions

| ID | Assumption | Risk if Wrong | Relates To |
|----|-----------|---------------|------------|
| A-001 | The user has Claude Code installed and authenticated via `claude /login` on Linux. | Claude provider cannot function. Fails gracefully with setup instructions. | Claude |
| A-002 | The `/api/oauth/usage` endpoint returns subscription-level utilisation covering all Claude surfaces (web, Code, Cowork). | If Code-only, underreports. Community evidence supports subscription scope. | Claude |
| A-003 | The Claude response schema (`five_hour`, `seven_day`, `seven_day_opus`) is reasonably stable across plan types. | Schema changes require parser updates. Defensive parsing mitigates. | Claude |
| A-004 | The `anthropic-beta: oauth-2025-04-20` header value will remain valid or a successor discoverable. | API calls fail. Monitor Claude Code releases for changes. | Claude |
| A-005 | Python 3.10+ is available on target Linux distributions (Ubuntu 22.04+, Fedora 36+, Arch current). | Older distros need `pyenv` or container. | All |
| A-006 | OpenAI's `/v1/organization/usage/completions` endpoint is accessible with a standard API key (not just admin keys). | If admin-only, fall back to undocumented billing endpoints. | OpenAI |
| A-007 | Ollama runs on `localhost:11434` by default and the `/api/ps` and `/api/tags` endpoints are stable. | Ollama API changes would require updates. These endpoints have been stable for 2+ years. | Ollama |
| A-008 | NVIDIA GPU monitoring via `pynvml` works on Linux with standard NVIDIA drivers installed. | If driver mismatch, GPU metrics fail gracefully. | Local |
| ~~A-009~~ | ~~Rate limiting on Claude's usage endpoint is per-token, not per-IP. Running alongside Claude Code's own polling won't cause mutual 429s.~~ | ~~If per-IP, concurrent polling could cause interference.~~ | ~~Claude~~ | *Superseded by D-004 (10m poll interval) and D-041 (exponential backoff). Design is now robust regardless of rate-limit scope.* |
| A-010 | xAI's rate limit response headers (`x-ratelimit-remaining-requests`, etc.) are present and parseable on all Grok API responses. | If absent on some models/endpoints, rate limit window would show "unavailable". | Grok |
| ~~A-011~~ | ~~A D-Bus Secret Service daemon (GNOME Keyring, KDE Wallet, or equivalent) is running on the user's desktop session.~~ | ~~Keyring storage unavailable; fall back to env vars or `key_command`. Emit clear instructions.~~ | ~~Security~~ | *No longer load-bearing. Keyring is tier 3; env vars (tier 2) and key_command (tier 1) cover all cases. Docker mode (D-038) skips keyring entirely. The tool works fine without a keyring daemon.* |
| A-012 | The user's home filesystem supports POSIX file permissions (not FAT32/NTFS mounted without proper mapping). | Permission checks would be meaningless. Detect and warn. | Security |
| A-013 | In Docker deployments, credentials are provided via environment variables or mounted files. System keyring is not available. | Credential resolution falls through to env vars (tier 2). Document clearly in Docker setup instructions. | Docker |
| A-014 | The user has systemd with user session support (`systemctl --user`) on their Linux desktop/server. | `daemon install` won't work on non-systemd systems (Alpine, older init systems). `daemon start` (direct fork) and `daemon run` (foreground) still work. | Daemon |
| A-015 | SQLite WAL mode supports one writer and many concurrent readers without contention. | If the daemon is writing while the CLI reads, WAL handles this. Verified: SQLite WAL supports exactly this pattern. | Daemon |

---

## 13. Decisions

| ID | Decision | Rationale | Date | Status |
|----|----------|-----------|------|--------|
| D-001 | **Python as implementation language.** | Broadest Linux availability, Rich for TUI, PyGObject for GTK, pynvml for GPU. Single language across all layers. | 2026-04-05 | Accepted |
| D-002 | **Pluggable provider architecture with abstract base class.** | Enables incremental delivery (Claude first, others later) without refactoring core. Third-party providers possible in future. | 2026-04-05 | Accepted |
| D-003 | **JSON as default output mode (no flag required).** | Unix philosophy - default output is machine-parseable. Human-readable output is opt-in via `--now` or `--monitor`. | 2026-04-05 | Accepted |
| D-004 | **Global poll interval (10m default) with per-provider override.** | Cloud usage data changes slowly; 10 minutes is sufficient for Claude, OpenAI, Grok, and avoids Claude's aggressive rate limiting. Local providers (Ollama, Local) override to 60s. A single `poll_interval` replaces the former `cache_ttl` + `refresh_interval` split. | 2026-04-06 | Accepted |
| D-005 | **Defer Claude extra usage spend to future release.** | No clean API exists. Scraping is fragile. The utilisation percentages cover the primary use case. Revisit when OQ-001 is resolved. | 2026-04-05 | Proposed |
| D-006 | **Use `rich` for terminal output.** | Progress bars, tables, colour, Live display with minimal code. Widely used, well-maintained. | 2026-04-05 | Accepted |
| D-007 | **Follow XDG Base Directory specification.** | Config in `~/.config/llm-monitor/`, cache in `~/.cache/llm-monitor/`. Standard Linux practice. | 2026-04-05 | Accepted |
| D-008 | **Lightweight base install with additive extras.** | Base install includes cloud providers only (no compiled C extensions). `[local]` adds psutil/pynvml for GPU metrics. `[gtk]` adds PyGObject. `[all]` adds everything. Extras are additive, not subtractive — this is how pip extras actually work. | 2026-04-05 | Accepted |
| D-009 | **Provider errors are isolated.** | A failure in Grok should not prevent Claude data from displaying. Each provider reports independently. Aggregate exit code reflects overall state. | 2026-04-05 | Accepted |
| D-010 | **TOML for configuration with per-provider sections.** | Python stdlib support (3.11+), human-readable, natural nesting for provider-specific config. | 2026-04-05 | Accepted |
| D-011 | **GTK4 + libadwaita targeted for v2.** | Modern GNOME path. KDE compatibility via SNI. Detailed approach under OQ-009. | 2026-04-05 | Proposed |
| D-012 | **Project name: `llm-monitor`.** | Clear, generic, doesn't tie to a single provider. Short enough for CLI use. Available on PyPI (to be verified). | 2026-04-05 | Proposed |
| D-013 | **Each provider keeps its native units.** | Claude reports percent, OpenAI reports USD, Ollama reports tokens/sec. Forcing normalisation would lose information. The `UsageWindow.unit` field makes the unit explicit. | 2026-04-05 | Accepted |
| D-014 | **Rename from `claude-usage` to `llm-monitor`.** | Supports multi-provider roadmap from day one. Provider architecture baked into initial design rather than retrofitted. | 2026-04-05 | Accepted |
| D-015 | **Australian English spelling throughout codebase and documentation.** | Author preference. `utilisation` not `utilization`, `colour` not `color`, etc. Code identifiers use US English where Python convention requires it (e.g., `color` in Rich API calls). | 2026-04-05 | Accepted |
| D-016 | **No plaintext API keys in config files.** | Credentials are resolved via system keyring, environment variables, or command helpers. The config schema does not include any field for storing key values directly. Config files on disk must never contain secrets. | 2026-04-05 | Accepted |
| D-017 | **System keyring via Python `keyring` library as the preferred credential store.** | Integrates with GNOME Keyring (Secret Service D-Bus API), KDE Wallet, and KeePassXC. Cross-DE support on Linux. Falls back gracefully if no keyring daemon is running. | 2026-04-05 | Accepted |
| D-018 | **Warn (not refuse) on config files with loose permissions.** | The config file contains no secrets by design (D-016), so a hard failure was disproportionate to the risk. Default umask creates files as 0644; every user would hit the error on first run. A visible warning on every invocation trains correct behaviour without blocking usage. Permission checks are skipped entirely in container environments. | 2026-04-06 | Accepted |
| D-019 | **SecretStr wrapper type for all credentials in memory.** | Prevents accidental logging, serialisation, or display of secrets. `__repr__` and `__str__` return masked values. | 2026-04-05 | Accepted |
| D-020 | **stdout for data, stderr for messaging, with no exceptions.** | Follows Unix convention and clig.dev guidelines. Enables clean piping: `llm-monitor \| jq` never sees warnings or spinners. | 2026-04-05 | Accepted |
| D-021 | **SIGPIPE handled silently.** | Prevents tracebacks when piping to `head`, `grep -q`, etc. Standard Unix CLI behaviour. | 2026-04-05 | Accepted |
| D-022 | **SIGHUP reloads configuration.** | Standard daemon/long-running process convention. Allows config changes without restarting monitor mode. | 2026-04-05 | Accepted |
| D-023 | **No redirects on credential-bearing HTTP requests.** | Prevents accidental credential forwarding to unexpected hosts via HTTP 301/302. Defence in depth against open redirect vulnerabilities in upstream APIs. | 2026-04-05 | Accepted |
| D-024 | **`key_command` executed with `shell=False`.** | Prevents shell injection. Commands are split with `shlex.split()` and executed directly. | 2026-04-05 | Accepted |
| D-025 | **Atomic file writes for cache updates.** | Write to `.tmp` then `os.rename()`. Prevents corruption from interrupted writes or concurrent access. Combined with `fcntl.flock()` advisory locking. The tool never writes to credential files (see D-036). | 2026-04-05 | Accepted |
| D-026 | **TTY-adaptive output.** | Colour, Unicode, and interactive features auto-disable when stdout is not a TTY. Respects `$NO_COLOR`, `$TERM=dumb`. Follows https://no-color.org/ convention. | 2026-04-05 | Accepted |
| D-027 | **SQLite for local usage history.** | Python stdlib (`sqlite3`), zero-dependency, single-file, supports time-range queries and aggregation natively. Preferable to JSONL (poor query performance) or RRD (resolution loss, extra dependency). Well-proven for local app storage (Firefox, Calibre, zsh). | 2026-04-06 | Accepted |
| D-028 | **History writes only on meaningful change.** | Avoids bloating the database with identical repeated samples from cached responses. Delta threshold of 0.1% utilisation change or status change triggers a write. | 2026-04-06 | Accepted |
| D-029 | **90-day default retention with automatic pruning.** | Balances useful historical depth against disk usage. At typical polling rates, 90 days produces roughly 230K rows per 3-provider setup - well within SQLite's performance envelope. Configurable in `config.toml`. | 2026-04-06 | Accepted |
| D-030 | **`history purge` requires explicit typed confirmation.** | Irreversible destructive operation. Interactive mode requires the user to type the word `purge` (case-sensitive). Non-interactive mode uses `--confirm`. | 2026-04-06 | Accepted |
| D-031 | **Git-tag-driven versioning via `hatch-vcs`.** | Single source of truth. No standalone `VERSION` file. Tags drive CI/CD release to PyPI. Eliminates version mismatch between file and tag. | 2026-04-06 | Accepted |
| D-032 | **Publish to PyPI as `llm-monitor`. Lightweight base, additive extras.** | Base install has no compiled C extensions (works in containers, CI, minimal environments). `[local]` adds psutil/pynvml for GPU metrics. `[gtk]` adds desktop frontend. `[all]` is the kitchen sink. | 2026-04-06 | Accepted |
| D-033 | **`uv` as the primary packaging and installation tool.** | Replaces `pipx`. Faster dependency resolution, native `uv tool install` for CLI apps, `uv build` for releases. Aligns with modern Python ecosystem direction. `pip install` remains supported as fallback. | 2026-04-06 | Accepted |
| D-034 | **Ollama supports multiple network hosts.** | Local inference is not limited to localhost. Homelab and team setups commonly distribute models across machines. Simple form (`host = "..."`) for single host; array form (`[[providers.ollama.hosts]]`) for multi-host. Each host labelled and reported independently. | 2026-04-06 | Accepted |
| D-035 | **Per-model usage breakdown as a first-class data model.** | `ModelUsage` dataclass captures per-model token counts, costs, and request counts. Stored in dedicated `model_usage` history table. Populated by providers that support it (OpenAI natively, Claude partially via Opus window, Grok via response headers). Enables "which model is costing me the most?" analysis. | 2026-04-06 | Accepted |

| D-036 | **Read-only consumer of Claude credentials.** | The tool never writes to `~/.claude/.credentials.json`. Claude Code owns that file and uses Node.js (no POSIX flock). Self-managing token refresh from a second process introduces race conditions. On token expiry, emit a clear error directing the user to `claude /login`. | 2026-04-06 | Accepted |
| D-037 | **Daemon/service architecture for continuous collection.** | A background daemon decouples data collection from presentation. The CLI, TUI, and GTK frontends become thin readers of the shared SQLite database. Without a daemon, history is only recorded when the user happens to run the tool. The daemon is additive — all CLI modes still work standalone. | 2026-04-06 | Accepted |
| D-038 | **Docker as a first-class deployment target.** | The daemon runs cleanly in a container: lightweight base install (no C extensions), env vars for credentials (no keyring needed), mounted volume for SQLite. Ideal for homelab/server monitoring of Ollama instances and API spend. Permission checks skipped in container mode. | 2026-04-06 | Accepted |
| D-039 | **`key_command` failure is a hard error, not silent fallthrough.** | If the user explicitly configured a credential command, it failing should not silently fall through to an env var. This masks misconfiguration. The tool raises `CredentialError` and reports the failure clearly. | 2026-04-06 | Accepted |
| D-040 | **`SecretStr.__repr__` never reveals any real characters.** | The previous design leaked the first 6 characters. For tokens with standard prefixes this is low-value, but it sets a bad precedent. `__repr__` now always returns `SecretStr('***')`. | 2026-04-06 | Accepted |
| D-041 | **Exponential backoff on rate-limit (429) responses.** | Claude's 429s can persist for 30+ minutes. Without backoff state, the tool retries on every poll cycle and gets re-rate-limited. Backoff escalates exponentially (10m → 20m → 40m, cap 60m), is persisted in the cache file, and resets on success. | 2026-04-06 | Accepted |
| D-042 | **Environment variable overrides for all XDG paths.** | `LLM_MONITOR_CONFIG`, `LLM_MONITOR_DATA_DIR`, `LLM_MONITOR_CACHE_DIR` override defaults. Essential for Docker (where XDG dirs may not exist) and for CI/test environments. | 2026-04-06 | Accepted |
| D-043 | **Report aggregation: mean(utilisation), max-severity(status), last(counters), max(tokens/cost).** | Fields have different semantics: utilisation is a gauge (mean is correct), status is a severity level (worst-case matters), raw_value/raw_limit/resets_at are point-in-time state (last is authoritative), tokens/cost are running totals within a provider window (max captures the high-water mark without double-counting). Delta-based analysis deferred to v1.x. | 2026-04-07 | Accepted |
| D-044 | **Export uses two logical record types with `type` discriminator.** | `usage_samples` and `model_usage` have different cardinality and schemas. Merging into one row with NULLs everywhere is messy. JSONL uses a `type` field per line (`usage_sample`, `model_usage`, `provider_extras`). CSV uses two sections with separate header rows (extras omitted — JSON blobs don't map to flat columns). SQL includes all tables. Export is always a complete dump with no filtering flags. | 2026-04-07 | Accepted |

---

## 14. Milestones

### v0.1.0 - CLI MVP (Claude Provider, Standalone)

**Goal:** `llm-monitor` and `llm-monitor --now` display current Claude usage.

**Core:**
- [x] `pyproject.toml` with `hatchling` + `hatch-vcs` build backend
- [x] `SecretStr` wrapper type with fully-masked `__repr__`/`__str__` (no character leakage)
- [x] Credential sanitisation filter for logs and error output (REDACTION_PATTERNS)
- [x] Secure file I/O helpers (atomic write, secure mkdir, permission warning)
- [x] Permission warning on config file (stderr, not hard failure)
- [x] Config file loader (TOML) with permission warnings and env var overrides
- [x] Environment variable overrides for paths (`LLM_MONITOR_CONFIG`, `LLM_MONITOR_DATA_DIR`, `LLM_MONITOR_CACHE_DIR`)
- [x] Provider base class and registry with `resolve_credential()` (hard fail on `key_command` error)
- [x] `key_command` support with `shell=False` execution and timeout handling
- [x] System keyring integration via `keyring` library
- [x] Claude provider: read-only credential consumer, usage client, token expiry detection
- [x] Rate-limit backoff state (exponential, persisted in cache file)
- [x] Per-provider cache layer for standalone mode (XDG paths, poll_interval-based TTL, file locking)
- [x] JSON output mode (default, stdout only)
- [x] Table output mode (`--now`, stdout only, TTY-adaptive)
- [x] Exit codes (0, 1, 2, 3, 4, 130, 143)
- [x] Error handling with structured format (what/why/fix)
- [x] stdout/stderr separation enforced
- [x] TTY detection for adaptive colour/formatting
- [x] `$NO_COLOR` and `$TERM=dumb` support
- [x] SIGPIPE handling (silent exit)
- [x] SIGINT/SIGTERM handling (clean shutdown)
- [x] `--fresh`, `--verbose`, `--quiet`, `--provider`, `--clear-cache` flags
- [x] `--list-providers` command
- [x] TLS verification enforced (no `verify=False`)
- [x] No-redirect policy on credential-bearing requests

**Tests:**
- [x] GitHub Actions CI pipeline (test on push) — from day one, not deferred to v1.0.0
- [x] `test_models.py` — SecretStr (repr never leaks, str always masked, bool, len), UsageWindow/ModelUsage/ProviderStatus construction and JSON serialisation
- [x] `test_security.py` — sanitisation filter against all REDACTION_PATTERNS, secure_write creates 0o600 files atomically, secure_mkdir creates 0o700 dirs, permission check warns but continues
- [x] `test_config.py` — TOML parsing, env var overrides (`LLM_MONITOR_CONFIG` etc.), default values for missing keys, malformed TOML error message, missing config file creates defaults
- [x] `test_cache.py` — write/read round-trip, poll_interval TTL expiry, `--fresh` bypasses cache, `--clear-cache` deletes files, flock contention (concurrent reads), backoff state persistence and escalation
- [x] `test_providers/test_base.py` — resolve_credential: key_command success returns SecretStr, key_command non-zero raises CredentialError, key_command timeout raises CredentialError, env var fallback, keyring fallback, no credential returns None, allowed_hosts validation
- [x] `test_providers/test_claude.py` — credential file reading (valid, missing, expired token), usage response parsing (all three windows), null window handling (`seven_day_opus: null`), 429 returns cached data and enters backoff, 401 triggers credential re-read and retry, mocked HTTP via `respx`
- [x] `test_formatters/test_json_fmt.py` — output matches documented JSON schema (Section 4.2.3), no secrets in output, timestamp format, cached flag and cache_age_seconds
- [x] `test_formatters/test_table_fmt.py` — TTY output has colour/Unicode, non-TTY output is plain ASCII, `$NO_COLOR` disables colour, `$TERM=dumb` disables colour
- [x] `test_cli.py` — exit codes for each scenario (0/1/2/3/4), `--provider` filtering, `--verbose` and `--quiet` mutual exclusion error, `--version` output, `--help` output, stdout contains only data (no warnings), stderr contains only messages (no data)
- [x] End-to-end integration: `llm-monitor --provider claude` with mocked HTTP returns valid JSON to stdout; `llm-monitor --now --provider claude` returns table to stdout

**Documentation:**
- [x] README.md — project description, installation (`uv tool install`, `pip install`, from source), prerequisites (Claude Code authenticated via `claude /login`), quick start (`llm-monitor`, `llm-monitor --now`), JSON output example, table output example, configuration file location and example, credential setup (keyring, env var, key_command), available CLI flags, exit codes, security model summary, license

### v0.2.0 - History + Reporting

**Goal:** Usage data is recorded over time and can be queried.

**Core:**
- [ ] SQLite history store creation with schema and migrations
- [ ] `ModelUsage` dataclass and `model_usage` history table
- [ ] History write-on-fetch with meaningful-change detection (in-memory last-known state)
- [ ] `[history]` config section (enabled, retention_days)
- [ ] `--no-history` flag
- [ ] Automatic retention pruning on startup
- [ ] `llm-monitor history purge` with typed confirmation and `--confirm`
- [ ] `llm-monitor history stats` summary command
- [ ] `llm-monitor --report` / `llm-monitor history report` (table, JSON, CSV formats)
- [ ] `llm-monitor history export` (sql, jsonl, csv)
- [ ] Report flags: `--days`, `--from`, `--to`, `--format`, `--granularity`, `--models`

**Tests:**
- [ ] `test_history.py` — schema creation and version tracking, write-on-fetch inserts rows, meaningful-change detection: delta < 0.1% → no write, delta > 0.1% → write, status change → write, reset detection → write, cached response → no write
- [ ] `test_history.py` — retention pruning deletes rows older than configured days and keeps recent rows, `PRAGMA auto_vacuum` set
- [ ] `test_history.py` — purge interactive confirmation (mock stdin), purge `--confirm` flag, purge aborted on wrong input, purge when stdin is not TTY and no `--confirm` → error
- [ ] `test_history.py` — stats command output (sample count, providers, date range, DB size)
- [ ] `test_history.py` — report generation: table/JSON/CSV formats, date range filtering (`--from`, `--to`, `--days`), granularity aggregation (raw, hourly, daily), per-model breakdown (`--models`)
- [ ] `test_history.py` — export formats: SQL is valid SQL, JSONL has one JSON object per line, CSV has header row
- [ ] `test_history.py` — WAL mode enabled on new databases, concurrent read during write doesn't block

**Documentation:**
- [ ] README.md update — add history section: `history stats`, `history purge`, `--report` usage with examples, `history export` formats, `--no-history` flag, `[history]` config section, data storage location (`~/.local/share/llm-monitor/history.db`)

### v0.3.0 - Daemon + Docker

**Goal:** Continuous background collection without a terminal open.

**Core:**
- [ ] Daemon mode: `daemon start`, `daemon stop`, `daemon status`, `daemon run`
- [ ] Daemon: poll loop with global `poll_interval` (10m default), per-provider override
- [ ] Daemon: PID file management, log file
- [ ] Daemon: `daemon install` / `daemon uninstall` for systemd user service
- [ ] CLI daemon detection: read from DB when daemon is running, fallback to standalone
- [ ] SIGHUP config reload in daemon
- [ ] Dockerfile and docker-compose.yml
- [ ] Container-aware mode (`$LLM_MONITOR_CONTAINER`): skip permission checks, skip keyring, disable notifications

**Tests:**
- [ ] `test_daemon.py` — `daemon run` starts poll loop and writes to history DB after first tick (with mocked providers)
- [ ] `test_daemon.py` — poll loop respects per-provider `poll_interval` overrides
- [ ] `test_daemon.py` — poll loop survives provider errors (one provider fails, others still polled)
- [ ] `test_daemon.py` — PID file created on start, removed on clean shutdown
- [ ] `test_daemon.py` — `daemon start` when already running → error with existing PID
- [ ] `test_daemon.py` — `daemon stop` sends SIGTERM, waits, removes PID file
- [ ] `test_daemon.py` — `daemon status` reports running/stopped, last poll time, next poll
- [ ] `test_daemon.py` — SIGHUP triggers config reload without restart
- [ ] `test_daemon.py` — SIGTERM triggers clean shutdown (flush pending writes, close DB, remove PID)
- [ ] `test_cli.py` additions — CLI detects running daemon via PID file and reads from DB instead of fetching
- [ ] `test_cli.py` additions — `--fresh` fetches directly even when daemon is running
- [ ] `test_config.py` additions — container-aware mode: permission checks skipped when `$LLM_MONITOR_CONTAINER=1`
- [ ] Docker integration: build image, `docker run` starts daemon, `docker exec llm-monitor --now` returns data (optional, CI-permitting)

**Documentation:**
- [ ] README.md update — add daemon section: `daemon start/stop/status/run` usage, systemd integration (`daemon install`), `[daemon]` config section, poll interval configuration. Add Docker section: Dockerfile usage, docker-compose.yml example, environment variable credentials, volume mount for history DB, container-aware mode

### v0.4.0 - Monitor TUI

**Goal:** Live dashboard with auto-refresh and sparklines.

**Core:**
- [ ] Rich Live TUI (`--monitor`)
- [ ] TTY requirement check (refuse if not interactive)
- [ ] TUI reads from history DB when daemon is running (display-only, no API calls)
- [ ] TUI fetches directly in standalone mode (no daemon)
- [ ] Live countdown timers for reset windows
- [ ] Status colour transitions as utilisation changes
- [ ] Key bindings (r, 1-9, q, j, ?)
- [ ] Rate-limit backoff indicator (standalone mode)
- [ ] `--compact` single-line mode for tmux/polybar/waybar
- [ ] Provider health indicators (connected/stale/error)
- [ ] Daemon status indicator (running, last poll time)
- [ ] SIGUSR1 force refresh
- [ ] Terminal state restoration (cursor, alternate screen) via atexit
- [ ] Sparkline visualisation from history database

**Tests:**
- [ ] `test_formatters/test_monitor_fmt.py` — TUI renders without crash with sample ProviderStatus data
- [ ] `test_formatters/test_monitor_fmt.py` — compact mode produces single-line output per provider
- [ ] `test_formatters/test_monitor_fmt.py` — colour transitions: normal (green), warning (yellow), critical (red), exceeded (magenta)
- [ ] `test_formatters/test_monitor_fmt.py` — countdown timer formatting (hours+minutes, days+hours)
- [ ] `test_formatters/test_monitor_fmt.py` — sparkline rendering from history data (empty history → no sparkline, sufficient data → correct bar characters)
- [ ] `test_cli.py` additions — `--monitor` without TTY → error exit, `--monitor` with `--compact` accepted
- [ ] `test_cli.py` additions — SIGUSR1 triggers refresh in monitor mode (signal handler test)

**Documentation:**
- [ ] README.md update — add monitor mode section: `--monitor` usage, key bindings table, `--compact` mode for tmux/waybar, screenshot or example output, daemon integration (reads from DB when daemon running)

### v0.5.0 - Grok Provider

- [ ] xAI Grok provider implementation
- [ ] Rate limit header parsing
- [ ] Spend tracking (if API available, else deferred)
- [ ] Config section for Grok

**Tests:**
- [ ] `test_providers/test_grok.py` — successful response parsing, rate limit headers mapped to UsageWindow, credential resolution via `$XAI_API_KEY`, 429 handling with backoff, network error → cached data, mocked HTTP via `respx`

**Documentation:**
- [ ] README.md update — add Grok to supported providers list, Grok credential setup (`$XAI_API_KEY`, keyring, key_command), Grok-specific config example, rate limit monitoring explanation

### v0.6.0 - OpenAI Provider

- [ ] OpenAI provider implementation
- [ ] Usage API integration (`/v1/organization/usage/completions`)
- [ ] Per-model usage breakdown via `group_by[]=model` populating `model_usage` table
- [ ] Billing endpoint integration (credit balance, subscription)
- [ ] Config section for OpenAI

**Tests:**
- [ ] `test_providers/test_openai.py` — usage response parsing with per-model breakdown, cost endpoint parsing, credit balance mapping, credential resolution via `$OPENAI_API_KEY`, 429/5xx handling, undocumented endpoint fallback, mocked HTTP via `respx`

**Documentation:**
- [ ] README.md update — add OpenAI to supported providers list, OpenAI credential setup (`$OPENAI_API_KEY`), OpenAI-specific config example, per-model cost breakdown explanation, credit balance monitoring

### v0.7.0 - Ollama Provider

- [ ] Ollama provider implementation
- [ ] Multi-host support (single `host` and `[[providers.ollama.hosts]]` array forms)
- [ ] Per-host status, model listing, and VRAM reporting
- [ ] `/api/ps` for loaded models and VRAM
- [ ] `/api/tags` for health check
- [ ] Response metrics aggregation (tokens/sec rolling average)
- [ ] Per-model token tracking via response `usage` fields written to `model_usage` table
- [ ] Optional Prometheus `/metrics` integration
- [ ] Config section for Ollama

**Tests:**
- [ ] `test_providers/test_ollama.py` — single-host response parsing (`/api/ps`, `/api/tags`), multi-host config with per-host labels, host unreachable → error for that host only (other hosts unaffected), VRAM mapping to UsageWindow, tokens/sec rolling average calculation, no credentials required (is_configured always true when host set), mocked HTTP via `respx`

**Documentation:**
- [ ] README.md update — add Ollama to supported providers list, single-host and multi-host configuration examples, VRAM and inference speed monitoring explanation, no credentials required note

### v0.8.0 - Local System Metrics Provider

- [ ] NVIDIA GPU metrics via `pynvml`
- [ ] AMD GPU metrics via `rocm-smi` subprocess (basic)
- [ ] CPU/RAM via `psutil`
- [ ] Multi-GPU support
- [ ] Config section for local metrics

**Tests:**
- [ ] `test_providers/test_local.py` — GPU metrics with mocked `pynvml` (utilisation, VRAM, temperature), multi-GPU indexing, CPU/RAM via mocked `psutil`, graceful degradation when no GPU detected, AMD fallback to `rocm-smi` subprocess (mocked), `gpu_backend = "auto"` detection logic

**Documentation:**
- [ ] README.md update — add Local System Metrics to supported providers list, `[local]` extra installation (`uv tool install "llm-monitor[local]"`), GPU/CPU/RAM monitoring explanation, multi-GPU support, NVIDIA vs AMD backend configuration

### v0.9.0 - Notifications and Polish

- [ ] Desktop notifications via `notify-send` / `gi.repository.Notify` (daemon fires these)
- [ ] Configurable thresholds per provider
- [ ] `--notify` flag
- [ ] Shell completion scripts (bash, zsh, fish)
- [ ] Man page

**Tests:**
- [ ] `test_notifications.py` — notification fires on status transition (normal → warning, warning → critical), no notification when status unchanged, notification suppressed when disabled in config, `--notify` flag enables for single invocation, reset notification fires when configured

**Documentation:**
- [ ] README.md update — add notifications section: `--notify` flag, `[notifications]` config section, desktop notification requirements (`notify-send`), threshold configuration

### v1.0.0 - Stable CLI Release

- [ ] Comprehensive error handling across all providers
- [ ] `llm-monitor config set-key --provider <name>` interactive key setup
- [ ] `llm-monitor config check` validates permissions, keyring, and provider connectivity
- [ ] Security audit of credential flow (all providers)
- [ ] Fuzz testing on credential sanitisation patterns (randomised strings against REDACTION_PATTERNS)
- [ ] PyPI publication via `uv build` + GitHub Actions trusted publishing
- [ ] GitHub Actions release pipeline (build + publish on tag push)
- [ ] Docker Hub / GHCR image publication in CI
- [ ] JSON output schema and config format declared as stable interfaces
- [ ] CHANGELOG

**Documentation:**
- [ ] README.md update — comprehensive rewrite for v1.0.0: full provider matrix with status, complete configuration reference, all CLI commands and flags, JSON output schema documentation, scripting/pipeline examples (`jq`, waybar, polybar), screenshots of table and monitor modes, troubleshooting section, contributing guidelines

**Tests:**
- [ ] Schema stability tests: JSON output validated against documented schema from Section 4.2.3 (regression guard)
- [ ] `test_config.py` additions — `config set-key` writes to keyring (mocked), `config check` validates connectivity (mocked HTTP), `config check` reports permission issues
- [ ] Fuzz harness for credential sanitisation: generate random strings matching `sk-*`, `Bearer *`, `xai-*` patterns and verify redaction
- [ ] End-to-end integration across all providers with mocked HTTP: multi-provider JSON output, multi-provider table output, `--report` with mixed provider history

### v2.0.0 - GTK/GNOME Desktop Widget

- [ ] GTK4 + libadwaita application
- [ ] System tray indicator with aggregate status icon
- [ ] Per-provider collapsible sections in popover
- [ ] Theme-aware (dark/light)
- [ ] XDG autostart
- [ ] KDE Plasma compatibility (SNI)
- [ ] Extra usage spend display (if APIs become available)

**Documentation:**
- [ ] README.md update — add GTK desktop widget section: `--ux` mode, `[gtk]` extra installation, XDG autostart setup, screenshot, KDE compatibility notes

---

## 15. Docker Deployment

### 15.1 Overview

The daemon architecture (Section 4.2.7) maps naturally to a Docker container. The container runs `llm-monitor daemon run` in the foreground, polling providers on schedule and writing to a SQLite database in a mounted volume. The CLI can then be run on the host (reading the same database) or via `docker exec`.

### 15.2 Dockerfile

```dockerfile
FROM python:3.12-slim

RUN pip install --no-cache-dir llm-monitor

# Create non-root user
RUN useradd --create-home --shell /bin/bash monitor
USER monitor

# Default data directory
ENV LLM_MONITOR_DATA_DIR=/data
ENV LLM_MONITOR_CACHE_DIR=/data/cache
ENV LLM_MONITOR_CONTAINER=1

VOLUME /data

ENTRYPOINT ["llm-monitor", "daemon", "run"]
```

The base install (no `[local]` extra) is used — cloud providers only, no GPU/system metrics. This keeps the image small and avoids C extension compilation.

### 15.3 Docker Compose

```yaml
services:
  llm-monitor:
    build: .
    # or: image: ghcr.io/<user>/llm-monitor:latest
    restart: unless-stopped
    volumes:
      - llm-monitor-data:/data
      - ${HOME}/.config/llm-monitor/config.toml:/home/monitor/.config/llm-monitor/config.toml:ro
      # Mount Claude credentials read-only (if using Claude provider)
      - ${HOME}/.claude/.credentials.json:/home/monitor/.claude/.credentials.json:ro
    environment:
      # Cloud provider API keys (alternative to keyring)
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - XAI_API_KEY=${XAI_API_KEY}
      # Override poll interval (optional)
      # - LLM_MONITOR_POLL_INTERVAL=600

volumes:
  llm-monitor-data:
```

### 15.4 Container-Aware Behaviour

When `$LLM_MONITOR_CONTAINER=1` is set (or `/.dockerenv` is detected):

- **Permission checks are skipped.** Volume mounts have their own UID/permission model; POSIX permission checks on mounted files are unreliable.
- **Keyring is not attempted.** No D-Bus Secret Service daemon is available. Credential resolution skips tier 3 and does not log a warning about keyring unavailability.
- **Desktop notifications are disabled.** No notification daemon exists in a container.
- **`daemon install` / `daemon uninstall` are not available.** No systemd in the container. `daemon run` (foreground) is the only supported mode.
- **`--monitor` and `--ux` are not available.** No TTY by default. Use `docker exec -it` if needed.

### 15.5 Accessing Data from Host

The SQLite database is in the mounted volume. The host CLI can read it directly:

```bash
# Point the host CLI at the container's database
export LLM_MONITOR_DATA_DIR=/path/to/docker/volume

# Now standard CLI commands read from the daemon's database
llm-monitor --now
llm-monitor --report --days 7
```

Alternatively, use `docker exec`:

```bash
docker exec llm-monitor llm-monitor --now
docker exec llm-monitor llm-monitor --report
```

### 15.6 Health Check

If a health endpoint is implemented (see OQ-023), the Dockerfile adds:

```dockerfile
HEALTHCHECK --interval=60s --timeout=5s --retries=3 \
    CMD ["llm-monitor", "daemon", "status", "--quiet"]
```

Without a health endpoint, `daemon status` exit code (0 = running, non-zero = not) serves as the health check.

---

## 16. References

### Claude (Anthropic)

| Source | URL |
|--------|-----|
| OAuth usage endpoint discovery | https://codelynx.dev/posts/claude-code-usage-limits-statusline |
| Claude Code authentication docs | https://code.claude.com/docs/en/authentication |
| Rate limiting bug reports | https://github.com/anthropics/claude-code/issues/31637 |
| Persistent 429 bug | https://github.com/anthropics/claude-code/issues/30930 |
| Feature request: expose usage API | https://github.com/anthropics/claude-code/issues/32796 |
| Extra usage management | https://support.claude.com/en/articles/12429409 |
| Claude-Code-Usage-Monitor | https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor |
| claude-usage-tool (Electron) | https://github.com/IgniteStudiosLtd/claude-usage-tool |
| Anthropic Usage & Cost Admin API | https://platform.claude.com/docs/en/build-with-claude/usage-cost-api |

### Grok (xAI)

| Source | URL |
|--------|-----|
| Consumption and rate limits | https://docs.x.ai/docs/key-information/consumption-and-rate-limits |
| Models and pricing | https://docs.x.ai/developers/models |

### OpenAI

| Source | URL |
|--------|-----|
| Usage Dashboard | https://help.openai.com/en/articles/10478918-api-usage-dashboard |
| Legacy Usage Dashboard | https://help.openai.com/en/articles/8554956-usage-dashboard-legacy |
| Usage API reference | https://platform.openai.com/docs/api-reference/usage |

### Ollama

| Source | URL |
|--------|-----|
| Ollama usage metrics docs | https://docs.ollama.com/api/usage |
| ollama-metrics (Prometheus proxy) | https://github.com/NorskHelsenett/ollama-metrics |
| Metrics endpoint feature request | https://github.com/ollama/ollama/issues/3144 |

### Security and CLI Best Practices

| Source | URL |
|--------|-----|
| Command Line Interface Guidelines | https://clig.dev/ |
| NO_COLOR convention | https://no-color.org/ |
| Python keyring library | https://pypi.org/project/keyring/ |
| D-Bus Secret Service (GNOME Keyring) | https://wiki.gnome.org/Projects/GnomeKeyring |
| 12 Factor CLI Apps | https://medium.com/@jdxcode/12-factor-cli-apps-dd3c227a0e46 |

### Existing Multi-Provider Tools

| Source | URL |
|--------|-----|
| Olla (multi-provider metrics) | https://thushan.github.io/olla/concepts/provider-metrics/ |
| LLM-Observability (Ollama + Prometheus) | https://github.com/anglosherif/LLM-Observability |
