# LLM Monitor - Specification

**Project codename:** `clawmeter`
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
│                   clawmeter daemon                                │
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
│ clawmeter     │ │ --monitor    │ │ --ux         │
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
                val = kr.get_password("clawmeter", f"{self.name()}_api_key")
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
| Cache Layer | Per-provider cached responses (standalone mode only) | JSON files in `~/.cache/clawmeter/` |
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
- Claude Code manages token refresh automatically; clawmeter is a read-only consumer (see Section 7.7).
- On token expiry, the tool emits a clear error directing the user to run `claude /login`.

**Extra usage spend (alpha — D-053, v0.7.1):** The existing `/api/oauth/usage` endpoint returns an `extra_usage` object when extra usage is enabled on the account:

```json
{
  "extra_usage": {
    "is_enabled": true,
    "monthly_limit": 10000,
    "used_credits": 10010.0,
    "utilization": 100.0
  }
}
```

- `is_enabled`: whether the account has extra usage turned on
- `monthly_limit`: spending cap in cents (user's billing currency, not necessarily USD)
- `used_credits`: amount consumed in cents (`used_credits` can exceed `monthly_limit`)
- `utilization`: percentage of limit consumed (0–100%)

Since the endpoint is undocumented and the `extra_usage` field could be removed or changed without notice, this is gated behind `enable_alpha_features` (D-053). The currency is the user's billing currency (no currency identifier in the API response), so values are displayed with a generic `$` symbol using the `"credits"` unit type.

**Mapped extra usage window (alpha):**

| Window | Source | Unit | Notes |
|--------|--------|------|-------|
| Extra Usage | `extra_usage.utilization` | percent | Percentage of monthly limit consumed. `raw_value` = `used_credits / 100`, `raw_limit` = `monthly_limit / 100`. Only when `is_enabled` and alpha flag set. |

**Extras dict:** `{ "extra_usage_enabled": true/false/null, "extra_usage_spent": 100.10, "extra_usage_limit": 100.00 }` (dollar values derived from cents). `extra_usage_enabled` is `null` when `extra_usage` is absent from the response. Spend/limit fields only present when extra usage is enabled and alpha flag is set.

**Per-model breakdown:** The Claude usage endpoint does not provide a per-model token breakdown. The `seven_day_opus` and `seven_day_sonnet` windows provide per-model utilisation percentages, which are mapped as separate `UsageWindow` entries. If Anthropic expands the endpoint to include per-model detail, the provider will populate `model_usage` accordingly.

**Allowed hosts:** `api.anthropic.com` (HTTPS only).

---

### 3.2 Grok (xAI) - v0.5.0

**Type:** API spend monitoring + prepaid balance tracking + usage analytics

**Data sources:**
- **xAI Management API** (`management-api.x.ai`) — billing, spend, usage analytics, prepaid balance. Requires a Management Key (separate from API key) and a team ID.
- **xAI Inference API** (`api.x.ai`) — rate limit headers on chat completion responses (optional, supplementary).

**Authentication:**
- **Management Key** (primary): from xAI Console → Management Keys. Resolved via `resolve_credential()` (keyring, env var, or key_command). Default env var: `$XAI_MANAGEMENT_KEY`. Required for billing/usage data.
- **API Key** (optional): standard API key for rate limit header data. Default env var: `$XAI_API_KEY`. Not required if management key is present.
- **Team ID** (required): `team_id` config field in `[providers.grok]`. Available from xAI Console. Can also be set via `$XAI_TEAM_ID` env var.

**Available endpoints (Management API — `management-api.x.ai`):**

| Endpoint | Method | Data | Notes |
|----------|--------|------|-------|
| `/v1/billing/teams/{team_id}/postpaid/invoice/preview` | GET | Per-model line items (token counts, unit prices, amounts), MTD totals, prepaid credits used | Primary spend data source |
| `/v1/billing/teams/{team_id}/postpaid/spending-limits` | GET | Hard/soft spending limits in USD cents | Used to derive spend-vs-limit percentage |
| `/v1/billing/teams/{team_id}/prepaid/balance` | GET | Prepaid credit purchase/spend history, current total in USD cents | Prepaid balance window |
| `/v1/billing/teams/{team_id}/usage` | POST | Time-series usage analytics with model grouping, cost in USD | Per-model breakdown, sparkline data |
| `/v1/api-key` | GET | Key status (active/blocked/disabled), team ID, ACLs | Health check (uses `api.x.ai`, not management API) |

**Available data (Inference API — `api.x.ai`, chat completion responses only):**

| Data | Source | Notes |
|------|--------|-------|
| Rate limit (requests) | `x-ratelimit-limit-requests`, `x-ratelimit-remaining-requests` headers | Only present on `/v1/chat/completions` responses |
| Rate limit (tokens) | `x-ratelimit-limit-tokens`, `x-ratelimit-remaining-tokens` headers | Only present on `/v1/chat/completions` responses |
| Per-request cost | `usage.cost_in_usd_ticks` in response body (1 tick = 1/10,000,000,000 USD) | Per-request only |
| Per-request tokens | `usage.prompt_tokens`, `usage.completion_tokens`, `usage.total_tokens` with detailed breakdowns | Includes reasoning, cached, audio, image token splits |

**Mapped usage windows:**

| Window | Source | Unit | Notes |
|--------|--------|------|-------|
| Spend (MTD) | Invoice preview `totalWithCorr` | usd | Month-to-date spend in current billing cycle |
| Spend vs Limit | Invoice preview spend / spending limits `effectiveHardSl` | percent | Percentage of hard spending limit consumed |
| Prepaid Balance | Prepaid balance `total` | usd | Remaining prepaid credits |

**Per-model breakdown:** The invoice preview endpoint returns per-model line items with token counts, unit prices, and amounts. The usage analytics endpoint supports `groupBy: ["description"]` for per-model time-series data in USD. The provider populates `model_usage` entries from the invoice preview (current cycle) and can use the usage analytics endpoint for historical sparkline data.

**Extras dict:** `{ "billing_cycle": { "year": ..., "month": ... }, "tier": "...", "models_used": [...] }`

**Cost unit:** The Management API returns costs in USD cents (integer `val` field). The Inference API returns `cost_in_usd_ticks` where 1 tick = 1/10,000,000,000 USD. The provider normalises both to USD floats for `UsageWindow.raw_value`.

**Rate limit tiers:** xAI uses 5 spending-based tiers (Tier 0–4, based on cumulative spend since 2026-01-01) plus Enterprise. Each tier sets per-model RPM and TPM limits. Actual limits are visible at `console.x.ai/team/default/rate-limits` but not programmatically queryable.

**Allowed hosts:** `management-api.x.ai`, `api.x.ai` (HTTPS only).

---

### 3.3 OpenAI - v0.6.0

**Type:** API spend and usage monitoring

**Data source:** OpenAI Administration API endpoints (Usage API and Costs API)

**Authentication:**
- **Admin API key** (`sk-admin-*` prefix) from OpenAI Console → Settings → Organization → Admin Keys.
- Only Organisation Owners can create admin keys. Required scope: `api.usage.read`.
- Resolved via `resolve_credential()` with credential name `admin_key` (keyring, env var, or key_command).
- Default env var: `$OPENAI_ADMIN_KEY`.
- Standard project keys (`sk-proj-*`) do **not** have access to Usage or Costs endpoints.

**Available endpoints:**

| Endpoint | Data | Auth | Notes |
|----------|------|------|-------|
| `GET /v1/organization/usage/completions` | Token usage by model, project, time bucket | Admin key | Official Usage API. `group_by` supports: `project_id`, `user_id`, `api_key_id`, `model`, `batch`, `service_tier`. |
| `GET /v1/organization/costs` | Cost breakdown by line item, project | Admin key | Official Costs API. `group_by` supports: `project_id`, `line_item`. |

**Removed endpoints (no longer viable):**
- ~~`/v1/dashboard/billing/subscription`~~ — undocumented, now requires browser session key (not API key). Dead as of late 2025.
- ~~`/v1/dashboard/billing/credit_grants`~~ — same. No programmatic credit balance API exists.

**Usage API query parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `start_time` | integer | Yes | Unix timestamp (seconds), inclusive |
| `end_time` | integer | No | Unix timestamp (seconds), exclusive. Defaults to now. |
| `bucket_width` | string | No | `1m`, `1h`, or `1d` (default `1d`) |
| `group_by` | array | No | See per-endpoint list above |
| `project_ids` | array | No | Filter by project |
| `models` | array | No | Filter by model |
| `api_key_ids` | array | No | Filter by API key |
| `user_ids` | array | No | Filter by user |

**Usage API response schema:**
```json
{
  "object": "page",
  "data": [
    {
      "object": "bucket",
      "start_time": 1736616660,
      "end_time": 1736640000,
      "results": [
        {
          "object": "organization.usage.completions.result",
          "input_tokens": 141201,
          "output_tokens": 9756,
          "input_cached_tokens": 0,
          "input_audio_tokens": 0,
          "output_audio_tokens": 0,
          "num_model_requests": 470,
          "model": "gpt-4o-2024-08-06"
        }
      ]
    }
  ],
  "has_more": false,
  "next_page": null
}
```

**Costs API response schema:**
```json
{
  "object": "page",
  "data": [
    {
      "object": "bucket",
      "start_time": 1736553600,
      "end_time": 1736640000,
      "results": [
        {
          "object": "organization.costs.result",
          "amount": {
            "value": 0.13,
            "currency": "usd"
          },
          "line_item": "gpt-4o-2024-08-06",
          "project_id": null
        }
      ]
    }
  ],
  "has_more": false,
  "next_page": null
}
```

**Mapped usage windows:**

| Window | Source | Unit | Notes |
|--------|--------|------|-------|
| Spend (MTD) | `/v1/organization/costs` | usd | Month-to-date API spend, summed across all buckets |

**Extras dict:** `{ "models_used": [...], "top_model_spend": {...}, "bucket_width": "1d" }`

**Per-model breakdown:** The Usage API natively supports `group_by=model`. The provider calls both endpoints with model grouping:
- `/v1/organization/usage/completions?group_by=model` → per-model token counts (`input_tokens`, `output_tokens`, `input_cached_tokens`, `num_model_requests`)
- `/v1/organization/costs?group_by=line_item` → per-model cost in USD

These are merged into `ModelUsage` entries with both token counts and costs. This is the richest per-model data of any provider.

**Allowed hosts:** `api.openai.com` (HTTPS only).

---

### 3.4 Ollama (Network / Local / Cloud) - v0.7.0

**Type:** Local and network inference performance monitoring + cloud usage tracking (alpha)

**Data source:** Ollama REST API (one or more endpoints)

**Authentication:** None for local/network instances. Cloud models require an Ollama account (`ollama signin`) or API key (`$OLLAMA_API_KEY`) — see Section 3.4.1.

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
| `GET /` | Liveness probe ("Ollama is running") | Simplest health check |
| `GET /api/version` | `{"version": "0.12.6"}` | Version/compatibility check |
| `GET /api/tags` | List of downloaded + cloud models | Model inventory, health check |
| `GET /api/ps` | Currently loaded models, VRAM/RAM usage, expiry | Real-time state (local models only — cloud models do not appear) |
| `POST /api/show` | Model metadata, capabilities, architecture | Not called in poll loop (POST per model, slow). On-demand enrichment only. |

**Mapped usage windows:**

| Window | Source | Unit | Notes |
|--------|--------|------|-------|
| Models Available | `/api/tags` (per host) | count | Total downloaded models per host |
| Models Loaded | `/api/ps` (per host) | count | Models currently in memory per host |
| VRAM Usage | `/api/ps` `size_vram` (per host) | bytes → MB | GPU memory allocated per host. `size_vram` omitted when CPU-only ([#4840](https://github.com/ollama/ollama/issues/4840)) — treat as 0. |
| RAM Usage | `/api/ps` `size - size_vram` (per host) | bytes → MB | System RAM per host (derived) |

**Deferred windows (no polling endpoint — per-request only):**

| Window | Source | Unit | Notes |
|--------|--------|------|-------|
| ~~Inference Speed~~ | ~~Response `eval_count/eval_duration`~~ | ~~tokens/sec~~ | Deferred — only available in inference response bodies, not from a polling endpoint |

**Extras dict:**
```json
{
  "hosts": [
    {
      "name": "workstation",
      "url": "http://localhost:11434",
      "status": "connected",
      "version": "0.12.6",
      "models_available": 5,
      "models_loaded": [
        {
          "name": "gemma3:latest",
          "parameter_size": "4.3B",
          "quantization": "Q4_K_M",
          "size_bytes": 6591830464,
          "size_vram_bytes": 5333539264,
          "context_length": 4096,
          "expires_at": "2025-10-17T16:47:07Z"
        }
      ],
      "total_vram_used_mb": 5085,
      "total_ram_used_mb": 1200
    }
  ],
  "cloud": {
    "status": "authenticated",
    "plan": "pro",
    "alpha": true,
    "session_used_pct": 4.0,
    "session_resets_at": "2026-04-10T15:00:00Z",
    "weekly_used_pct": 14.3,
    "weekly_resets_at": "2026-04-13T02:00:00Z"
  }
}
```

The `cloud` section is only present when `enable_alpha_features = true` and `cloud_enabled = true`. It is `null` or absent otherwise.

**Note:** Ollama's local monitoring story is fundamentally different from cloud providers. There are no quotas or spend limits — it is a performance and resource utilisation monitor. The provider maps to the same `ProviderStatus` structure but uses resource-oriented windows rather than quota-oriented ones.

**Allowed hosts:** `localhost`, `127.0.0.1`, `[::1]`, or any user-configured host in the `hosts` array (HTTP or HTTPS). Network hosts are trusted by configuration — the user explicitly adds them. Cloud API uses `ollama.com` (HTTPS only).

#### 3.4.1 Ollama Cloud Models (Alpha — D-053)

Since September 2025 (Ollama v0.12), Ollama offers **cloud models** — large models (DeepSeek V3 671B, Qwen3-Coder 480B, GPT-OSS 120B, etc.) that run on Ollama's datacenter GPU infrastructure. Cloud model names include `cloud` in their tag (e.g. `gpt-oss:120b-cloud`, `deepseek-v3.2:cloud`).

**Pricing tiers:**

| Plan | Price | Concurrent Cloud Models | Cloud Usage |
|------|-------|------------------------|-------------|
| Free | $0 | 1 | Light allowance |
| Pro | $20/mo ($200/yr) | 3 | 50x Free |
| Max | $100/mo | 10 | 5x Pro |

Usage is measured by **GPU time** (not tokens). Session limits reset every 5 hours; weekly limits reset every 7 days. Local model usage is always unlimited. Additional per-token usage is "coming soon."

**Authentication:**
- **CLI:** `ollama signin` (SSH key challenge-response with ollama.com account)
- **API keys:** Created at `ollama.com/settings/keys`. Set via `$OLLAMA_API_KEY`. Used as `Authorization: Bearer <key>`. Keys do not expire but can be revoked.
- **Disable cloud:** Set `OLLAMA_NO_CLOUD=1` to reject cloud model requests.

**Cloud usage windows (alpha):**

| Window | Source | Unit | Notes |
|--------|--------|------|-------|
| Session Usage | `ollama.com/api/account/usage` (proposed) | percent | Resets every 5 hours |
| Weekly Usage | `ollama.com/api/account/usage` (proposed) | percent | Resets every 7 days |

**Critical limitation:** There is **no official API endpoint** for cloud usage data. Usage stats are only visible at `ollama.com/settings` in the browser. The community has been requesting a `/api/account/usage` or `/api/me` endpoint since October 2025 ([ollama/ollama#12532](https://github.com/ollama/ollama/issues/12532)). Until this ships, cloud usage monitoring is gated behind `enable_alpha_features` (D-053) and may use undocumented interfaces or web scraping.

**Rate limiting:** When session or weekly limits are exceeded, the API returns HTTP 429 with a `Retry-After` header. The existing exponential backoff logic (D-041) applies.

**Cloud models do NOT appear in `/api/ps`** — they have no local process or VRAM allocation. They are proxied on-demand to Ollama's infrastructure. Cloud models are detected by the `cloud` substring in the model tag from `/api/tags`.

See `docs/research/ollama-v0.7.0-research.md` for the full research report, API response structures, and community monitoring landscape.

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
clawmeter [MODE] [OPTIONS]
clawmeter daemon <start|stop|status|run|install>
clawmeter history <report|purge|stats|export>
clawmeter config <set-key|check>
```

### 4.2 Modes

#### 4.2.1 Output Stream Rules

All modes follow these rules without exception:

| Content | Destination | Rationale |
|---------|-------------|-----------|
| JSON output (default mode) | stdout | Machine-parseable data for piping |
| Table output (`--now`) | stdout | Primary output the user requested |
| `--help` output | stdout | Conventional; allows `clawmeter --help \| less` |
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
3. `$CLAWMETER_NO_COLOR` env var (app-specific override).
4. `$TERM=dumb` (disables).
5. TTY detection (auto).
6. `--colour=always` flag (force enable even when piped, for `clawmeter --now --colour=always | less -R`).

#### 4.2.3 JSON Mode (default, no flag required)

Returns structured JSON to stdout and exits. Designed for consumption by scripts, `jq`, waybar modules, polybar scripts, monitoring pipelines.

```bash
# All configured providers
clawmeter | jq '.providers[].provider_name'

# Single provider
clawmeter --provider claude | jq '.providers[0].windows'

# Use in a script
USAGE=$(clawmeter --provider claude | jq -r '.providers[0].windows[0].utilisation')
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

**Computed fields in JSON output:** The `resets_in_human` field is NOT part of the `UsageWindow` dataclass — it is computed at JSON serialisation time from `resets_at` relative to the current timestamp. Format: largest two units, e.g., `"2h 15m"`, `"2d 13h"`, `"45m"`, `"< 1m"`. If `resets_at` is `null`, `resets_in_human` is `null`. The top-level `timestamp` is the time the CLI was invoked. The top-level `version` is the package version (from `importlib.metadata.version("clawmeter")`).

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
clawmeter --now
clawmeter --now --provider claude
clawmeter --now --provider claude,openai
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
 Spend (MTD)    ██████░░░░░░░░░░░░░░  $24.88 / $500.00  (5%)
 Prepaid Bal.   $95.93 remaining

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
clawmeter --monitor
clawmeter --monitor --provider claude
clawmeter --monitor --compact    # single-line per provider for tmux
```

**Data source:** When the daemon is running, `--monitor` reads from the history database (no direct API calls). This makes the TUI a lightweight display-only process. When no daemon is running, `--monitor` fetches directly from providers on each refresh cycle (standalone behaviour).

**Features:**
- Auto-refresh display at configurable interval (`--interval`, default 30s, minimum 5s — see D-049). Data freshness depends on daemon poll interval.
- Live countdown timers for reset windows (reuses `format_resets_in_human()` from `json_fmt.py`).
- Status colour transitions as utilisation changes (reuses `_STATUS_COLOURS` from `table_fmt.py`).
- Sparkline visualisation per usage window from history data — 24 hourly data points using `▁▂▃▄▅▆▇█`, suppressed if < 3 data points (D-046). Controlled by `[monitor] show_sparkline` config.
- Compact single-line mode via `--compact` (D-045): one line per provider, format `● <name>  <bar> <pct>%  resets <time>`, bar width 10 chars. `--compact` is `--monitor`-only.
- Rate-limit backoff indicator per provider (when in standalone mode).
- Desktop notification on status transitions deferred to v0.9.0 (D-051).
- Provider health indicators based on data age vs poll_interval (D-050): green `●` ≤ 1×, yellow `●` ≤ 3×, red `●` > 3× or errors.
- Daemon status indicator (shows whether daemon is running and last poll time).

**Key bindings:**
- `r` - Force refresh all providers (bypass cache).
- `1-9` - Force refresh specific provider by index.
- `q` - Quit.
- `j` - Dump current state as JSON to `./clawmeter-<YYYYMMDD-HHMMSS>.json` (D-048). Status message in footer for 3s.
- `?` - Show help overlay (D-047). Rich Panel with keybinding list, dismissed on any keypress.

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
clawmeter daemon start           # start as background process
clawmeter daemon stop            # stop the running daemon
clawmeter daemon status          # show daemon state, PID, last poll time
clawmeter daemon run             # run in foreground (for systemd/Docker)
clawmeter daemon install         # install systemd user service
clawmeter daemon uninstall       # remove systemd user service
```

**`daemon start`:** Forks to background, writes PID to `$XDG_RUNTIME_DIR/clawmeter/daemon.pid` (or `/tmp/clawmeter-$UID/daemon.pid` if `$XDG_RUNTIME_DIR` is unset). Logs to `$XDG_STATE_HOME/clawmeter/daemon.log` (or `~/.local/state/clawmeter/daemon.log`). Exits immediately after fork; the parent prints the PID and returns.

**`daemon run`:** Runs in the foreground, logging to stderr. Designed for systemd `ExecStart`, Docker `ENTRYPOINT`, or manual debugging. This is the primary entry point for containerised deployments.

**`daemon stop`:** Reads the PID file, sends `SIGTERM`, waits up to 5 seconds for clean shutdown, then `SIGKILL` if needed. If no PID file exists, prints "Daemon is not running."

**`daemon status`:** Reports whether the daemon is running, its PID, uptime, last successful poll per provider, next scheduled poll, and database size.

```
$ clawmeter daemon status
Daemon: running (PID 48231, uptime 3h 12m)
  claude    last poll 2m ago    next in 8m     ok
  openai    last poll 2m ago    next in 8m     ok
  ollama    last poll 32s ago   next in 28s    ok
Database: ~/.local/share/clawmeter/history.db (4.2 MB)
```

**`daemon install`:** Writes a systemd user service unit file and enables it:

```ini
# ~/.config/systemd/user/clawmeter.service
[Unit]
Description=LLM Usage Monitor Daemon
Documentation=man:clawmeter(1)
After=network-online.target
Wants=network-online.target

[Service]
Type=exec
ExecStart=/path/to/clawmeter daemon run
Restart=on-failure
RestartSec=30
Environment=CLAWMETER_LOG_LEVEL=info

[Install]
WantedBy=default.target
```

After writing the unit file, runs `systemctl --user daemon-reload && systemctl --user enable --now clawmeter`. The `ExecStart` path is resolved from the current `clawmeter` binary location (via `shutil.which()` or `sys.argv[0]`).

**Poll loop:**
- On startup: read config, initialise providers, run retention pruning, perform an immediate first poll.
- Each provider is polled independently at the global `poll_interval` (default 600s / 10 minutes). Per-provider `poll_interval` overrides are respected if configured.
- After each successful fetch, write to the history database if the data has meaningfully changed (same delta logic as Section 6.4).
- On 429 / rate limit: enter backoff state (see Section 3.1), skip provider until backoff expires.
- On network error: log warning, retry on next cycle.
- On `SIGHUP`: reload config file, re-initialise providers.
- On `SIGTERM` / `SIGINT`: flush pending writes, close database, remove PID file, exit cleanly.

**Standalone fallback:** All CLI modes (`clawmeter`, `--now`, `--monitor`) continue to work without the daemon. When no daemon is running, the CLI fetches directly from providers and writes to the history database itself (the v0.1.0 behaviour). The daemon is additive, not required.

**Daemon detection:** The CLI checks for a running daemon by testing the PID file (`$XDG_RUNTIME_DIR/clawmeter/daemon.pid`). If the daemon is running:
- Default mode / `--now` / `--monitor`: read latest data from the history database instead of fetching from providers.
- `--fresh`: fetch directly from providers (bypass daemon), write to DB.
- The CLI emits a note to stderr if the daemon is running: `Reading from daemon (last poll 2m ago)`.

#### 4.2.7.1 Implementation Design

**Daemonisation:** Uses `os.fork()` double-fork (fork → `os.setsid()` → fork again) for proper POSIX daemonisation. No external dependencies. All forking happens before any asyncio or threading setup. The child redirects stdin/stdout/stderr to `/dev/null`, writes the PID file, then enters the poll loop. The parent prints the child PID and exits.

**Poll loop architecture:** The daemon uses a `DaemonRunner` class in `daemon.py` that wraps `core.fetch_all()` in an asyncio event loop. Per-provider polling intervals are tracked via a `dict[str, float]` mapping provider name to the next poll time (monotonic clock). Each iteration, the loop finds the soonest due provider, sleeps via `asyncio.sleep()` until then, polls all providers whose time has arrived, records results to the history DB via `HistoryStore.record()`, and updates each provider's next poll time. This naturally handles mixed intervals (e.g., 600s for cloud, 60s for local).

**Signal handling:** Uses `loop.add_signal_handler()` (the correct asyncio-on-Unix pattern). Handlers only set flags (`_shutdown`, `_reload`) and wake a shared `asyncio.Event` so the sleep is interrupted immediately rather than waiting for the full interval. The poll loop checks flags between iterations.

**State file:** `daemon status` reads per-provider last/next poll times from an ephemeral JSON state file at `$XDG_RUNTIME_DIR/clawmeter/daemon.state`, written via `secure_write()` after each poll cycle. This avoids adding columns to the history DB schema. Contents: `{"started_at": "...", "providers": {"claude": {"last_poll": "...", "next_poll": "...", "status": "ok"}}}`.

**CLI daemon-aware reads:** `HistoryStore` provides a `get_latest_statuses()` method that reconstructs `ProviderStatus` objects from the most recent rows per provider+window (similar query to the existing `_load_last_known()` but returning full data). The CLI calls this instead of `fetch_all()` when a running daemon is detected. A companion `get_last_poll_time()` method provides the "last poll Xm ago" value.

**Logging:** Stdlib `logging` with a named logger `clawmeter.daemon`. Foreground mode (`daemon run`) logs to stderr via `StreamHandler`. Background mode (`daemon start`) logs to the configured log file via `FileHandler`. Log level from `$CLAWMETER_LOG_LEVEL` (default `info`). Format: `%(asctime)s %(levelname)-8s %(message)s`.

**Reused components:**
- `core.fetch_all()` — the polling function the daemon wraps
- `history.HistoryStore.record()` — write-on-fetch with meaningful-change detection
- `cache.ProviderCache` — backoff state persistence, still used in daemon mode
- `security.secure_write()` — for PID file and state file
- `security.is_container_mode()` — for container-aware behaviour
- `config.load_config()` — initial load and SIGHUP reload
- `providers.get_enabled_providers()` — provider instantiation

### 4.3 Global Options

| Flag | Short | Description | Default |
|------|-------|-------------|---------|
| `--provider` | `-p` | Comma-separated list of providers to query | All configured |
| `--config` | `-c` | Path to config file | `~/.config/clawmeter/config.toml` |
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

Location: `~/.config/clawmeter/config.toml`

Overridable via `$CLAWMETER_CONFIG` environment variable.

The config file MUST NEVER contain API keys or secrets. Credentials are resolved indirectly via `key_command`, `key_env`, or system keyring. See Section 7 (Security Model) for the full credential resolution hierarchy.

**Environment variable overrides for paths:**

| Variable | Overrides | Default |
|----------|-----------|---------|
| `CLAWMETER_CONFIG` | Config file path | `$XDG_CONFIG_HOME/clawmeter/config.toml` |
| `CLAWMETER_DATA_DIR` | History DB directory | `$XDG_DATA_HOME/clawmeter/` |
| `CLAWMETER_CACHE_DIR` | Cache directory | `$XDG_CACHE_HOME/clawmeter/` |
| `CLAWMETER_LOG_LEVEL` | Daemon log level | `info` |

These variables take precedence over XDG defaults and config file values. They are particularly useful for Docker deployments (Section 15).

```toml
[general]
default_providers = ["claude"]
poll_interval = 600              # 10 minutes; applies to all providers unless overridden
notification_enabled = false
enable_alpha_features = false    # opt-in to unstable data sources (see D-053)

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
log_file = ""                    # empty = default ($XDG_STATE_HOME/clawmeter/daemon.log)
pid_file = ""                    # empty = default ($XDG_RUNTIME_DIR/clawmeter/daemon.pid)

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
team_id = ""                     # required: xAI team ID (or set $XAI_TEAM_ID)
# Management key (primary — billing, usage, spend data)
management_key_env = "XAI_MANAGEMENT_KEY"
# management_key_command = "secret-tool lookup application clawmeter provider grok-management"
# API key (optional — rate limit header data from inference API)
# key_env = "XAI_API_KEY"
# key_command = "secret-tool lookup application clawmeter provider grok"
# key_keyring = true             # use system keyring (default: true)

# ─── Provider: OpenAI ─────────────────────────────────────────
[providers.openai]
enabled = false
admin_key_env = "OPENAI_ADMIN_KEY"       # Admin key (sk-admin-*), NOT project key
# admin_key_command = "pass show clawmeter/openai-admin"
# admin_key_keyring = true

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
# Cloud usage monitoring (requires enable_alpha_features = true)
# cloud_enabled = false
# api_key_env = "OLLAMA_API_KEY"   # default env var for cloud API key
# api_key_command = "pass show clawmeter/ollama-cloud"
# cloud_poll_interval = 300        # cloud quota checks, 5 min default

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

`~/.cache/clawmeter/<provider>/last.json`

Follows XDG Base Directory specification. Respects `$XDG_CACHE_HOME` and `$CLAWMETER_CACHE_DIR` (see Section 4.6).

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

`~/.local/share/clawmeter/history.db`

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

#### `clawmeter history report` (aliased as `clawmeter --report`)

Display a summary report of usage over time.

```bash
clawmeter --report
clawmeter --report --days 30 --provider claude
clawmeter --report --days 30 --format csv > usage-march.csv
clawmeter --report --days 7 --provider claude --format json
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

#### `clawmeter history purge`

Permanently delete all history data. Requires explicit confirmation to prevent accidental data loss.

**Interactive mode (default):**
```
$ clawmeter history purge

WARNING: This will permanently delete all usage history.
  Database: ~/.local/share/clawmeter/history.db
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
clawmeter history purge --confirm
```

The `--confirm` flag bypasses the interactive prompt. Without it, the tool requires the interactive typed confirmation above.

When `stdin` is not a TTY (piped context) and `--confirm` is not provided, the interactive prompt is skipped and the tool exits with an error:
```
Error: history purge requires interactive confirmation.
Fix: Use --confirm to bypass: clawmeter history purge --confirm
```

#### `clawmeter history stats`

Quick summary of the history database.

```
$ clawmeter history stats

History Database: ~/.local/share/clawmeter/history.db
  Size:       4.2 MB
  Samples:    14,832
  Providers:  claude, openai, ollama
  Oldest:     2026-01-05T08:12:00Z
  Newest:     2026-04-05T10:25:00Z
  Retention:  90 days (next prune removes 0 records)
```

#### `clawmeter history export`

Full raw export of the database for backup or migration. Export always includes all data — no `--models` or `--provider` filtering (that's what `--report` is for). Export is a full dump for backup and migration.

```bash
clawmeter history export --format sql > backup.sql
clawmeter history export --format jsonl > backup.jsonl
clawmeter history export --format csv > backup.csv
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
clawmeter config set-key --provider openai
# Prompts securely for the admin key, stores via keyring

# Or via secret-tool directly
secret-tool store --label="clawmeter: OpenAI Admin Key" \
    application clawmeter provider openai
```

**Tier 2: Command-based credential helper (`key_command`)**

The config file may contain a `key_command` directive that executes a shell command to retrieve the key. This supports vault integrations, password managers, and custom credential helpers.

```toml
[providers.openai]
admin_key_command = "pass show clawmeter/openai-admin"
# or: admin_key_command = "secret-tool lookup application clawmeter provider openai"
# or: admin_key_command = "vault kv get -field=admin_key secret/clawmeter/openai"
```

**Tier 3: Environment variables**

Standard practice for CI/CD and containerised environments. Acknowledged risk: readable via `/proc/$PID/environ` by the same user. Acceptable for ephemeral contexts.

```bash
export OPENAI_ADMIN_KEY="sk-admin-..."
export XAI_API_KEY="xai-..."
export XAI_MANAGEMENT_KEY="xai-mgmt-..."
export XAI_TEAM_ID="..."
```

**Tier 4: Claude Code credential file (Claude provider only)**

Read-only access to `~/.claude/.credentials.json`. This file is owned and managed by Claude Code, not by clawmeter. The tool reads tokens but never writes to this file (see D-036).

**Resolution order per provider:**
1. `key_command` (if configured, execute and read stdout)
2. `key_env` / well-known env var (e.g., `$OPENAI_ADMIN_KEY`)
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
  Warning: Config file has loose permissions (0644): ~/.config/clawmeter/config.toml
  Other users on this system could read your configuration, which may contain credential command paths.
  Fix: chmod 600 ~/.config/clawmeter/config.toml
  ```
  The tool continues to run. The config file contains no secrets by design (D-016), so this is a defence-in-depth warning rather than a hard failure. The `--quiet` flag suppresses this warning.
- If Claude credential file permissions are more permissive than `0o600`, emit a **warning to stderr** (do not refuse, as Claude Code manages this file).
- In containerised environments (detected via `/.dockerenv` or `$CLAWMETER_CONTAINER=1`), permission checks are skipped entirely — container volume mounts have their own permission model.

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
| Grok | `management-api.x.ai`, `api.x.ai` | HTTPS only |
| Ollama (local) | `localhost`, `127.0.0.1`, `[::1]`, or user-configured host | HTTP or HTTPS |
| Ollama (cloud) | `ollama.com` | HTTPS only |

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
clawmeter/
├── pyproject.toml
├── README.md
├── LICENSE
├── CHANGELOG.md
├── SPEC.md                          # This document
├── src/
│   └── clawmeter/
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
    ├── clawmeter.desktop          # XDG autostart (v2)
    └── icons/                       # Tray icons (v2)
```

### 10.1 Installation

```bash
# Default install - cloud providers (Claude, Grok, OpenAI, Ollama API)
uv tool install clawmeter

# With local GPU/system metrics (adds psutil, pynvml)
uv tool install "clawmeter[local]"

# Everything (local metrics + GTK desktop frontend)
uv tool install "clawmeter[all]"

# Via pip (if uv is not available)
pip install clawmeter --user

# From source (development)
git clone https://github.com/<user>/clawmeter.git
cd clawmeter
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

The package is published to PyPI as `clawmeter`. The build backend is `hatchling` with `hatch-vcs` for version derivation from git tags.

```toml
# pyproject.toml (key sections)
[build-system]
requires = ["hatchling", "hatch-vcs"]
build-backend = "hatchling.build"

[project]
name = "clawmeter"
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
    "clawmeter[local,gtk]",
]

[project.scripts]
clawmeter = "clawmeter.__main__:main"

[tool.hatch.version]
source = "vcs"

[tool.hatch.build.hooks.vcs]
version-file = "src/clawmeter/_version.py"
```

### 10.3 Versioning and Release

**Single source of truth: git tags.**

The version is derived from git tags using `hatch-vcs` (a `setuptools-scm` equivalent for the `hatchling` build backend). There is no standalone `VERSION` file. The flow is:

1. Development builds get automatic versions like `0.1.0.dev12+g1a2b3c4` based on distance from the last tag.
2. To release, tag a commit: `git tag v0.1.0 && git push --tags`.
3. CI/CD (GitHub Actions) detects the tag, builds the sdist and wheel, and publishes to PyPI.
4. The built package contains a generated `src/clawmeter/_version.py` with the exact version string.
5. The application reads its own version at runtime via `importlib.metadata.version("clawmeter")`.

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
| ~~OQ-001~~ | ~~**How to surface Claude extra usage spend data?**~~ The existing `/api/oauth/usage` endpoint already returns an `extra_usage` object with `is_enabled`, `monthly_limit` (cents in user's billing currency), `used_credits`, and `utilization` (0–100%). No scraping needed. Ship behind `enable_alpha_features` (D-053) since the endpoint is undocumented. | High | Claude | **Closed (v0.7.1 research):** Data is in the existing endpoint response. Parse `extra_usage` field. |
| ~~OQ-002~~ | ~~**Is the `/api/oauth/usage` endpoint stable enough to depend on?**~~ Undocumented and aggressively rate-limited (~5 requests per token, then persistent 429s with no `Retry-After`). Anthropic closed bug reports as NOT_PLANNED. The existing provider already depends on it with exponential backoff — extra usage parsing adds no additional API calls. | High | Claude | **Closed (v0.7.1 research):** Already depended upon. Backoff handles rate limits. |
| OQ-003 | **Should Claude token refresh be self-managed or rely on Claude Code?** Self-refreshing is more robust but adds complexity and credential file conflict risk. | Medium | Claude | **Closed (D-036):** Read-only consumer. Self-refresh introduces race conditions with Claude Code's Node.js process. |
| ~~OQ-004~~ | ~~**What is the actual rate limit on the Claude usage endpoint?**~~ Approximately 5 requests per access token, then persistent 429s. No `Retry-After` header. Refreshing the token resets the counter but disrupts Claude Code's session. Community issues: [#31021](https://github.com/anthropics/claude-code/issues/31021), [#31637](https://github.com/anthropics/claude-code/issues/31637). | Medium | Claude | **Closed (v0.7.1 research):** ~5 req/token, handled by existing backoff. |
| OQ-005 | **Multi-account support?** Should the tool support multiple credentials per provider (e.g., work vs personal Claude accounts)? | Low (v1) | All | Open |
| OQ-006 | **Claude plan detection?** The usage endpoint doesn't return plan type (Pro, Max5, Max20). Infer from thresholds or require user config? | Low | Claude | Open |
| ~~OQ-007~~ | ~~**Tmux/status-bar integration format?** A `--compact` single-line output could feed tmux `status-right`, polybar, waybar. What format is most useful?~~ | ~~Low (v1)~~ | ~~Output~~ | **Closed (D-045):** `--compact` is a `--monitor`-only modifier rendering one plain-text line per provider. Tmux polling uses `--now` with external formatting. No JSON waybar variant in v0.4.0. |
| ~~OQ-008~~ | ~~**What does `utilization > 100` look like in Claude's API?**~~ Extra usage has its own `extra_usage.utilization` field (0–100%, capped at cap). `used_credits` can exceed `monthly_limit` (e.g. 10010 > 10000). The subscription windows (`five_hour`, `seven_day`) remain separate percentage fields unaffected by extra usage. | Medium | Claude | **Closed (v0.7.1 research):** Separate `extra_usage` object, `used_credits` can exceed `monthly_limit`. |
| OQ-009 | **GTK4 vs GTK3 for v2?** GTK4 + libadwaita is modern GNOME, but AppIndicator3 is GTK3. May need bridging or alternative tray approach. | Medium (v2) | GTK | Open |
| ~~OQ-010~~ | ~~**Licensing?** MIT vs GPL. MIT is simpler; GPL aligns with GNOME ecosystem.~~ | ~~Low~~ | ~~All~~ | **Closed:** MIT license committed to repo. `pyproject.toml` updated accordingly. |
| ~~OQ-011~~ | ~~**Does xAI have a programmatic billing/usage API?** Console shows spend, but no documented REST endpoint for querying balance or MTD cost found. Rate limit headers provide per-request data only.~~ | ~~High~~ | ~~Grok~~ | **Closed:** Yes. The xAI Management API (`management-api.x.ai`) provides full billing, spend, usage analytics, prepaid balance, and spending limit endpoints. Requires a separate Management Key (not the inference API key) and team ID. See updated Section 3.2. |
| ~~OQ-012~~ | ~~**OpenAI billing endpoint stability?** `/v1/dashboard/billing/subscription` and `/v1/dashboard/billing/credit_grants` are undocumented but widely used. The official Usage API (`/v1/organization/usage/...`) requires admin-level access.~~ | ~~Medium~~ | ~~OpenAI~~ | **Closed:** Confirmed. The undocumented `/v1/dashboard/billing/*` endpoints are dead — they require browser session keys as of late 2025. The official Usage API and Costs API (`/v1/organization/usage/*`, `/v1/organization/costs`) require an Admin API Key (`sk-admin-*`), not a standard project key. No programmatic credit balance API exists. See updated Section 3.3. |
| ~~OQ-013~~ | ~~**Ollama metrics endpoint availability?** Ollama does not have a built-in `/metrics` Prometheus endpoint in all versions. The proxy approach (ollama-metrics) is an alternative but adds a dependency. Should we support both paths?~~ | ~~Medium~~ | ~~Ollama~~ | **Closed (D-053 research):** No native `/metrics` endpoint exists ([#3144](https://github.com/ollama/ollama/issues/3144) still open). Do not depend on it. Use `/api/ps` + `/api/tags` for local monitoring. Document ollama-metrics as an optional external integration. |
| OQ-014 | **AMD GPU support depth?** `pyamdgpuinfo` is limited. `rocm-smi` subprocess parsing is more complete but slower. What level of AMD support is needed? | Low | Local | Open |
| ~~OQ-015~~ | ~~**Provider plugin system: entry_points vs explicit registration?** Entry points allow third-party providers but add packaging complexity. Explicit registration in a registry dict is simpler for v1.~~ | ~~Low~~ | ~~Architecture~~ | **Closed:** Explicit registration via `@register_provider` decorator and module-level dict for v1. See Section 2.2. |
| ~~OQ-016~~ | ~~**Unified cost normalisation?** Should the tool attempt to normalise costs across providers (e.g., all in USD) or keep each in its native unit? Normalisation is complex; native units are honest.~~ | ~~Low~~ | ~~Output~~ | **Closed (D-013):** Each provider keeps its native units. |
| OQ-017 | **Keyring availability in headless/server environments?** The Python `keyring` library requires a D-Bus Secret Service daemon. On headless servers or containers, keyring is unavailable. Should the tool fall back to encrypted file storage (e.g., `keyrings.alt`), or simply require env vars in those contexts? | Medium | Security | **Closed (D-038):** In containers, use env vars. Keyring is tier 3 in the resolution hierarchy; env vars (tier 2) and key_command (tier 1) cover headless cases. No need for `keyrings.alt`. |
| ~~OQ-018~~ | ~~**Should `key_command` support pipes and shell features?** Using `shell=False` prevents `secret-tool lookup ... \| head -1`. Users expecting shell syntax would need to wrap in `bash -c "..."`. Is this acceptable, or should we allow `shell=True` with a security warning?~~ | ~~Medium~~ | ~~Security~~ | **Closed (D-024, D-039):** `shell=False` is final. Users wrap in `bash -c "..."` if pipes are needed. |
| OQ-019 | **Should the `--insecure` flag be visible in `--help`?** Making it prominent encourages misuse. Alternatively, document it only in the man page and hide from `--help`. | Low | CLI UX | **Closed (D-018):** Permission checks are now warnings, not hard failures. The `--insecure` flag is no longer needed; `--quiet` suppresses the warning. |
| OQ-020 | **Should history support downsampling for long-term storage?** Keep 5-minute samples for 7 days, hourly averages for 90 days, daily averages for 1 year. Reduces disk usage for long retention but adds schema and query complexity. Could be a v1.x feature. | Low | History | Open |
| OQ-021 | **Should `--report` support chart output to terminal?** Rich can render basic bar charts. Alternatively, export to an HTML file with embedded charts. The sparkline in `--monitor` covers the basic case; full charts may be GTK v2 territory. | Low | History/Output | Open |
| OQ-022 | **Should per-model breakdown be available as a `--report --models` flag or a separate `clawmeter models` subcommand?** The flag approach keeps reporting unified; a subcommand could offer richer model-specific analysis (cost per model per day, model switching patterns). | Low | History/Models | **Closed (D-043):** `--report --models` flag. Keeps reporting unified; a dedicated subcommand can be added later if richer model analysis is needed. |
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
| ~~A-006~~ | ~~OpenAI's `/v1/organization/usage/completions` endpoint is accessible with a standard API key (not just admin keys).~~ | ~~If admin-only, fall back to undocumented billing endpoints.~~ | ~~OpenAI~~ | *Falsified. Both Usage and Costs APIs require an Admin API Key (`sk-admin-*`) with `api.usage.read` scope. The undocumented billing endpoints are also dead. Provider now requires admin key. See OQ-012 closure and updated Section 3.3.* |
| A-007 | Ollama runs on `localhost:11434` by default and the `/api/ps` and `/api/tags` endpoints are stable. Ollama Cloud uses `ollama.com` with API key auth. Cloud models are identified by a `cloud` tag suffix. | Ollama API changes would require updates. Local endpoints have been stable for 2+ years. Cloud usage API does not yet exist ([#12532](https://github.com/ollama/ollama/issues/12532)). | Ollama |
| A-008 | NVIDIA GPU monitoring via `pynvml` works on Linux with standard NVIDIA drivers installed. | If driver mismatch, GPU metrics fail gracefully. | Local |
| ~~A-009~~ | ~~Rate limiting on Claude's usage endpoint is per-token, not per-IP. Running alongside Claude Code's own polling won't cause mutual 429s.~~ | ~~If per-IP, concurrent polling could cause interference.~~ | ~~Claude~~ | *Superseded by D-004 (10m poll interval) and D-041 (exponential backoff). Design is now robust regardless of rate-limit scope.* |
| A-010 | xAI's rate limit response headers (`x-ratelimit-remaining-requests`, `x-ratelimit-remaining-tokens`, etc.) are present on `/v1/chat/completions` responses but NOT on `/v1/models` or Management API responses. Verified 2026-04-08. | Rate limit headers are supplementary only; primary data comes from the Management API. The provider works without them. | Grok |
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
| D-005 | **Claude extra usage spend available as alpha feature.** | The existing `/api/oauth/usage` endpoint returns an `extra_usage` object with `is_enabled`, `monthly_limit` (cents), `used_credits`, and `utilization` (0–100%). No scraping needed — just parse an additional field from the existing response. Gated behind `enable_alpha_features` (D-053) since the endpoint is undocumented. Values are in the user's billing currency (not necessarily USD), displayed using `"credits"` unit type. Graduates to stable when Anthropic documents the endpoint. See OQ-001. Tracked as [#19](https://github.com/danielithomas/clawmeter/issues/19) for v0.7.1. | 2026-04-10 | Accepted |
| D-006 | **Use `rich` for terminal output.** | Progress bars, tables, colour, Live display with minimal code. Widely used, well-maintained. | 2026-04-05 | Accepted |
| D-007 | **Follow XDG Base Directory specification.** | Config in `~/.config/clawmeter/`, cache in `~/.cache/clawmeter/`. Standard Linux practice. | 2026-04-05 | Accepted |
| D-008 | **Lightweight base install with additive extras.** | Base install includes cloud providers only (no compiled C extensions). `[local]` adds psutil/pynvml for GPU metrics. `[gtk]` adds PyGObject. `[all]` adds everything. Extras are additive, not subtractive — this is how pip extras actually work. | 2026-04-05 | Accepted |
| D-009 | **Provider errors are isolated.** | A failure in Grok should not prevent Claude data from displaying. Each provider reports independently. Aggregate exit code reflects overall state. | 2026-04-05 | Accepted |
| D-010 | **TOML for configuration with per-provider sections.** | Python stdlib support (3.11+), human-readable, natural nesting for provider-specific config. | 2026-04-05 | Accepted |
| D-011 | **GTK4 + libadwaita targeted for v2.** | Modern GNOME path. KDE compatibility via SNI. Detailed approach under OQ-009. | 2026-04-05 | Proposed |
| D-012 | **Project name: `clawmeter`.** | Clear, generic, doesn't tie to a single provider. Short enough for CLI use. Available on PyPI (to be verified). | 2026-04-05 | Proposed |
| D-013 | **Each provider keeps its native units.** | Claude reports percent, OpenAI reports USD, Ollama reports tokens/sec. Forcing normalisation would lose information. The `UsageWindow.unit` field makes the unit explicit. | 2026-04-05 | Accepted |
| D-014 | **Rename from `claude-usage` to `clawmeter`.** | Supports multi-provider roadmap from day one. Provider architecture baked into initial design rather than retrofitted. | 2026-04-05 | Accepted |
| D-015 | **Australian English spelling throughout codebase and documentation.** | Author preference. `utilisation` not `utilization`, `colour` not `color`, etc. Code identifiers use US English where Python convention requires it (e.g., `color` in Rich API calls). | 2026-04-05 | Accepted |
| D-016 | **No plaintext API keys in config files.** | Credentials are resolved via system keyring, environment variables, or command helpers. The config schema does not include any field for storing key values directly. Config files on disk must never contain secrets. | 2026-04-05 | Accepted |
| D-017 | **System keyring via Python `keyring` library as the preferred credential store.** | Integrates with GNOME Keyring (Secret Service D-Bus API), KDE Wallet, and KeePassXC. Cross-DE support on Linux. Falls back gracefully if no keyring daemon is running. | 2026-04-05 | Accepted |
| D-018 | **Warn (not refuse) on config files with loose permissions.** | The config file contains no secrets by design (D-016), so a hard failure was disproportionate to the risk. Default umask creates files as 0644; every user would hit the error on first run. A visible warning on every invocation trains correct behaviour without blocking usage. Permission checks are skipped entirely in container environments. | 2026-04-06 | Accepted |
| D-019 | **SecretStr wrapper type for all credentials in memory.** | Prevents accidental logging, serialisation, or display of secrets. `__repr__` and `__str__` return masked values. | 2026-04-05 | Accepted |
| D-020 | **stdout for data, stderr for messaging, with no exceptions.** | Follows Unix convention and clig.dev guidelines. Enables clean piping: `clawmeter \| jq` never sees warnings or spinners. | 2026-04-05 | Accepted |
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
| D-032 | **Publish to PyPI as `clawmeter`. Lightweight base, additive extras.** | Base install has no compiled C extensions (works in containers, CI, minimal environments). `[local]` adds psutil/pynvml for GPU metrics. `[gtk]` adds desktop frontend. `[all]` is the kitchen sink. | 2026-04-06 | Accepted |
| D-033 | **`uv` as the primary packaging and installation tool.** | Replaces `pipx`. Faster dependency resolution, native `uv tool install` for CLI apps, `uv build` for releases. Aligns with modern Python ecosystem direction. `pip install` remains supported as fallback. | 2026-04-06 | Accepted |
| D-034 | **Ollama supports multiple network hosts.** | Local inference is not limited to localhost. Homelab and team setups commonly distribute models across machines. Simple form (`host = "..."`) for single host; array form (`[[providers.ollama.hosts]]`) for multi-host. Each host labelled and reported independently. | 2026-04-06 | Accepted |
| D-035 | **Per-model usage breakdown as a first-class data model.** | `ModelUsage` dataclass captures per-model token counts, costs, and request counts. Stored in dedicated `model_usage` history table. Populated by providers that support it (OpenAI natively, Claude partially via Opus window, Grok via response headers). Enables "which model is costing me the most?" analysis. | 2026-04-06 | Accepted |

| D-036 | **Read-only consumer of Claude credentials.** | The tool never writes to `~/.claude/.credentials.json`. Claude Code owns that file and uses Node.js (no POSIX flock). Self-managing token refresh from a second process introduces race conditions. On token expiry, emit a clear error directing the user to `claude /login`. | 2026-04-06 | Accepted |
| D-037 | **Daemon/service architecture for continuous collection.** | A background daemon decouples data collection from presentation. The CLI, TUI, and GTK frontends become thin readers of the shared SQLite database. Without a daemon, history is only recorded when the user happens to run the tool. The daemon is additive — all CLI modes still work standalone. | 2026-04-06 | Accepted |
| D-038 | **Docker as a first-class deployment target.** | The daemon runs cleanly in a container: lightweight base install (no C extensions), env vars for credentials (no keyring needed), mounted volume for SQLite. Ideal for homelab/server monitoring of Ollama instances and API spend. Permission checks skipped in container mode. | 2026-04-06 | Accepted |
| D-039 | **`key_command` failure is a hard error, not silent fallthrough.** | If the user explicitly configured a credential command, it failing should not silently fall through to an env var. This masks misconfiguration. The tool raises `CredentialError` and reports the failure clearly. | 2026-04-06 | Accepted |
| D-040 | **`SecretStr.__repr__` never reveals any real characters.** | The previous design leaked the first 6 characters. For tokens with standard prefixes this is low-value, but it sets a bad precedent. `__repr__` now always returns `SecretStr('***')`. | 2026-04-06 | Accepted |
| D-041 | **Exponential backoff on rate-limit (429) responses.** | Claude's 429s can persist for 30+ minutes. Without backoff state, the tool retries on every poll cycle and gets re-rate-limited. Backoff escalates exponentially (10m → 20m → 40m, cap 60m), is persisted in the cache file, and resets on success. | 2026-04-06 | Accepted |
| D-042 | **Environment variable overrides for all XDG paths.** | `CLAWMETER_CONFIG`, `CLAWMETER_DATA_DIR`, `CLAWMETER_CACHE_DIR` override defaults. Essential for Docker (where XDG dirs may not exist) and for CI/test environments. | 2026-04-06 | Accepted |
| D-043 | **Report aggregation: mean(utilisation), max-severity(status), last(counters), max(tokens/cost).** | Fields have different semantics: utilisation is a gauge (mean is correct), status is a severity level (worst-case matters), raw_value/raw_limit/resets_at are point-in-time state (last is authoritative), tokens/cost are running totals within a provider window (max captures the high-water mark without double-counting). Delta-based analysis deferred to v1.x. | 2026-04-07 | Accepted |
| D-044 | **Export uses two logical record types with `type` discriminator.** | `usage_samples` and `model_usage` have different cardinality and schemas. Merging into one row with NULLs everywhere is messy. JSONL uses a `type` field per line (`usage_sample`, `model_usage`, `provider_extras`). CSV uses two sections with separate header rows (extras omitted — JSON blobs don't map to flat columns). SQL includes all tables. Export is always a complete dump with no filtering flags. | 2026-04-07 | Accepted |
| D-045 | **`--compact` is a `--monitor`-only modifier, plain text, one line per provider.** | Format: `● <name>  <bar> <pct>%  resets <time>` where `●` is the health indicator dot (coloured). Bar width = 10 chars. No JSON variant in v0.4.0 — tmux polling uses `--now` piped through external formatting. Closes OQ-007 for v0.4.0 scope. | 2026-04-08 | Accepted |
| D-046 | **Sparklines: 24 hourly data points, Unicode block characters, min-max linear mapping.** | Source: `aggregate_samples(granularity="hourly")` filtered to last 24h. Characters: `▁▂▃▄▅▆▇█` mapped linearly across the min-max range. Suppressed if fewer than 3 data points exist (not enough to be meaningful). One sparkline per usage window, displayed after the progress bar. | 2026-04-08 | Accepted |
| D-047 | **`?` help overlay: Rich Panel with keybindings, dismissed on any keypress.** | A `rich.panel.Panel` centred in the display listing all keybindings (one per line), with a "Press any key to dismiss" footer. Replaces the main content in the Live display until a key is pressed. Minimal, no over-engineering. | 2026-04-08 | Accepted |
| D-048 | **`j` dumps JSON snapshot to current working directory.** | Writes same schema as `clawmeter` default output to `./clawmeter-<YYYYMMDD-HHMMSS>.json`. A brief status message appears in the TUI footer for 3 seconds. On write failure (permissions), show error in footer instead. | 2026-04-08 | Accepted |
| D-049 | **`--interval`/`-i` flag: integer seconds, default 30, minimum 5.** | Only meaningful with `--monitor` — controls UI refresh rate. Minimum 5 seconds prevents accidental API hammering in standalone mode. Values < 5 clamped to 5 with a stderr warning. | 2026-04-08 | Accepted |
| D-050 | **Provider health indicator thresholds based on poll_interval multiples.** | Green `●` = data age ≤ 1× poll_interval (healthy). Yellow `●` = data age > 1× but ≤ 3× poll_interval (stale — daemon may have missed a cycle). Red `●` = data age > 3× poll_interval OR provider has errors. poll_interval read from config (default 600s, per-provider override respected). | 2026-04-08 | Accepted |
| D-051 | **`--notify` deferred to v0.9.0. No flag added in v0.4.0.** | The spec's roadmap places notifications at v0.9.0. Adding a dead flag creates confusion. Desktop notification support arrives with the notification engine. | 2026-04-08 | Accepted |
| D-052 | **OpenAI provider requires Admin API Key (`sk-admin-*`), not a standard project key.** | The Usage API and Costs API both require the `api.usage.read` scope, which is only available on admin keys. Standard project keys (`sk-proj-*`) return 403. The undocumented `/v1/dashboard/billing/*` endpoints are dead (require browser session keys since late 2025). No credit balance API exists. Env var: `$OPENAI_ADMIN_KEY`. Only Organisation Owners can create admin keys. See OQ-012. | 2026-04-10 | Accepted |
| D-053 | **`enable_alpha_features` flag for unstable data sources.** | Some monitoring data (Ollama Cloud usage quotas, Claude extra usage spend) is only available via undocumented endpoints, web scraping, or APIs that may change without notice. Rather than deferring these features indefinitely or shipping them as stable, a global `enable_alpha_features = true` config flag gates access. Alpha features: (a) emit a stderr warning on first use per session, (b) label alpha-sourced windows/metrics with an `alpha: true` flag in extras, (c) fail gracefully — errors are swallowed, never fatal, (d) are documented as potentially breaking between releases. This allows power users to opt in while managing expectations. Applies to: Ollama Cloud session/weekly usage (no official API — [ollama/ollama#12532](https://github.com/ollama/ollama/issues/12532)), Claude extra usage spend (no REST API — OQ-001). When a data source graduates to a stable API, the feature moves out from behind the flag. | 2026-04-10 | Accepted |

---

## 14. Milestones

### v0.1.0 - CLI MVP (Claude Provider, Standalone)

**Goal:** `clawmeter` and `clawmeter --now` display current Claude usage.

**Core:**
- [x] `pyproject.toml` with `hatchling` + `hatch-vcs` build backend
- [x] `SecretStr` wrapper type with fully-masked `__repr__`/`__str__` (no character leakage)
- [x] Credential sanitisation filter for logs and error output (REDACTION_PATTERNS)
- [x] Secure file I/O helpers (atomic write, secure mkdir, permission warning)
- [x] Permission warning on config file (stderr, not hard failure)
- [x] Config file loader (TOML) with permission warnings and env var overrides
- [x] Environment variable overrides for paths (`CLAWMETER_CONFIG`, `CLAWMETER_DATA_DIR`, `CLAWMETER_CACHE_DIR`)
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
- [x] `test_config.py` — TOML parsing, env var overrides (`CLAWMETER_CONFIG` etc.), default values for missing keys, malformed TOML error message, missing config file creates defaults
- [x] `test_cache.py` — write/read round-trip, poll_interval TTL expiry, `--fresh` bypasses cache, `--clear-cache` deletes files, flock contention (concurrent reads), backoff state persistence and escalation
- [x] `test_providers/test_base.py` — resolve_credential: key_command success returns SecretStr, key_command non-zero raises CredentialError, key_command timeout raises CredentialError, env var fallback, keyring fallback, no credential returns None, allowed_hosts validation
- [x] `test_providers/test_claude.py` — credential file reading (valid, missing, expired token), usage response parsing (all three windows), null window handling (`seven_day_opus: null`), 429 returns cached data and enters backoff, 401 triggers credential re-read and retry, mocked HTTP via `respx`
- [x] `test_formatters/test_json_fmt.py` — output matches documented JSON schema (Section 4.2.3), no secrets in output, timestamp format, cached flag and cache_age_seconds
- [x] `test_formatters/test_table_fmt.py` — TTY output has colour/Unicode, non-TTY output is plain ASCII, `$NO_COLOR` disables colour, `$TERM=dumb` disables colour
- [x] `test_cli.py` — exit codes for each scenario (0/1/2/3/4), `--provider` filtering, `--verbose` and `--quiet` mutual exclusion error, `--version` output, `--help` output, stdout contains only data (no warnings), stderr contains only messages (no data)
- [x] End-to-end integration: `clawmeter --provider claude` with mocked HTTP returns valid JSON to stdout; `clawmeter --now --provider claude` returns table to stdout

**Documentation:**
- [x] README.md — project description, installation (`uv tool install`, `pip install`, from source), prerequisites (Claude Code authenticated via `claude /login`), quick start (`clawmeter`, `clawmeter --now`), JSON output example, table output example, configuration file location and example, credential setup (keyring, env var, key_command), available CLI flags, exit codes, security model summary, license

### v0.2.0 - History + Reporting

**Goal:** Usage data is recorded over time and can be queried.

**Core:**
- [x] SQLite history store creation with schema and migrations
- [x] `ModelUsage` dataclass and `model_usage` history table
- [x] History write-on-fetch with meaningful-change detection (in-memory last-known state)
- [x] `[history]` config section (enabled, retention_days)
- [x] `--no-history` flag
- [x] Automatic retention pruning on startup
- [x] `clawmeter history purge` with typed confirmation and `--confirm`
- [x] `clawmeter history stats` summary command
- [x] `clawmeter --report` / `clawmeter history report` (table, JSON, CSV formats)
- [x] `clawmeter history export` (sql, jsonl, csv)
- [x] Report flags: `--days`, `--from`, `--to`, `--format`, `--granularity`, `--models`

**Tests:**
- [x] `test_history.py` — schema creation and version tracking, write-on-fetch inserts rows, meaningful-change detection: delta < 0.1% → no write, delta > 0.1% → write, status change → write, reset detection → write, cached response → no write
- [x] `test_history.py` — retention pruning deletes rows older than configured days and keeps recent rows, `PRAGMA auto_vacuum` set
- [x] `test_history.py` — purge interactive confirmation (mock stdin), purge `--confirm` flag, purge aborted on wrong input, purge when stdin is not TTY and no `--confirm` → error
- [x] `test_history.py` — stats command output (sample count, providers, date range, DB size)
- [x] `test_history.py` — report generation: table/JSON/CSV formats, date range filtering (`--from`, `--to`, `--days`), granularity aggregation (raw, hourly, daily), per-model breakdown (`--models`)
- [x] `test_history.py` — export formats: SQL is valid SQL, JSONL has one JSON object per line, CSV has header row
- [x] `test_history.py` — WAL mode enabled on new databases, concurrent read during write doesn't block

**Documentation:**
- [x] README.md update — add history section: `history stats`, `history purge`, `--report` usage with examples, `history export` formats, `--no-history` flag, `[history]` config section, data storage location (`~/.local/share/clawmeter/history.db`)

### v0.3.0 - Daemon + Docker

**Goal:** Continuous background collection without a terminal open.

**Implementation phasing:**

1. `config.py` — add `get_pid_dir()`, `get_log_dir()`, `get_pid_file(config)`, `get_log_file(config)`, `get_state_file(config)` path helpers; add `"daemon": {"log_file": "", "pid_file": ""}` to `DEFAULT_CONFIG`
2. `providers/base.py` — wrap Tier 3 keyring block with `if not is_container_mode():` guard (skip keyring in containers)
3. `history.py` — add `get_latest_statuses() -> list[ProviderStatus]` (reconstruct from most recent rows per provider+window) and `get_last_poll_time() -> datetime | None`
4. `daemon.py` (NEW) — `DaemonRunner` class with asyncio poll loop, PID/state file management, signal handlers; `daemonise()` double-fork function; `is_daemon_running(config)` check (see Section 4.2.7.1 for design)
5. `cli.py` — `daemon` subgroup with `start`/`stop`/`status`/`run`/`install`/`uninstall`; modify `status` command to detect running daemon and read from DB
6. `Dockerfile`, `docker-compose.yml`, `.dockerignore` (see Section 15)
7. `tests/test_daemon.py` (NEW) + additions to `test_cli.py`, `test_config.py`
8. `README.md` — daemon and Docker sections

**Core:**
- [x] Daemon mode: `daemon start`, `daemon stop`, `daemon status`, `daemon run`
- [x] Daemon: poll loop with global `poll_interval` (10m default), per-provider override
- [x] Daemon: PID file management, log file
- [x] Daemon: `daemon install` / `daemon uninstall` for systemd user service
- [x] CLI daemon detection: read from DB when daemon is running, fallback to standalone
- [x] SIGHUP config reload in daemon
- [x] Dockerfile and docker-compose.yml
- [x] Container-aware mode (`$CLAWMETER_CONTAINER`): skip permission checks, skip keyring, disable notifications

**Tests:**
- [x] `test_daemon.py` — `daemon run` starts poll loop and writes to history DB after first tick (with mocked providers)
- [x] `test_daemon.py` — poll loop respects per-provider `poll_interval` overrides
- [x] `test_daemon.py` — poll loop survives provider errors (one provider fails, others still polled)
- [x] `test_daemon.py` — PID file created on start, removed on clean shutdown
- [x] `test_daemon.py` — `daemon start` when already running → error with existing PID
- [x] `test_daemon.py` — `daemon stop` sends SIGTERM, waits, removes PID file
- [x] `test_daemon.py` — `daemon status` reports running/stopped, last poll time, next poll
- [x] `test_daemon.py` — SIGHUP triggers config reload without restart
- [x] `test_daemon.py` — SIGTERM triggers clean shutdown (flush pending writes, close DB, remove PID)
- [x] `test_cli.py` additions — CLI detects running daemon via PID file and reads from DB instead of fetching
- [x] `test_cli.py` additions — `--fresh` fetches directly even when daemon is running
- [x] `test_config.py` additions — container-aware mode: permission checks skipped when `$CLAWMETER_CONTAINER=1`
- [ ] Docker integration: build image, `docker run` starts daemon, `docker exec clawmeter --now` returns data (optional, CI-permitting)

**Documentation:**
- [x] README.md update — add daemon section: `daemon start/stop/status/run` usage, systemd integration (`daemon install`), `[daemon]` config section, poll interval configuration. Add Docker section: Dockerfile usage, docker-compose.yml example, environment variable credentials, volume mount for history DB, container-aware mode

### v0.4.0 - Monitor TUI

**Goal:** Live dashboard with auto-refresh and sparklines.

**Core:**
- [x] Rich Live TUI (`--monitor`)
- [x] TTY requirement check (refuse if not interactive)
- [x] TUI reads from history DB when daemon is running (display-only, no API calls)
- [x] TUI fetches directly in standalone mode (no daemon)
- [x] Live countdown timers for reset windows
- [x] Status colour transitions as utilisation changes
- [x] Key bindings (r, 1-9, q, j, ?)
- [x] Rate-limit backoff indicator (standalone mode)
- [x] `--compact` single-line mode for tmux/polybar/waybar
- [x] Provider health indicators (connected/stale/error)
- [x] Daemon status indicator (running, last poll time)
- [x] SIGUSR1 force refresh
- [x] Terminal state restoration (cursor, alternate screen) via atexit
- [x] Sparkline visualisation from history database

**Tests:**
- [x] `test_formatters/test_monitor_fmt.py` — TUI renders without crash with sample ProviderStatus data
- [x] `test_formatters/test_monitor_fmt.py` — compact mode produces single-line output per provider
- [x] `test_formatters/test_monitor_fmt.py` — colour transitions: normal (green), warning (yellow), critical (red), exceeded (magenta)
- [x] `test_formatters/test_monitor_fmt.py` — countdown timer formatting (hours+minutes, days+hours)
- [x] `test_formatters/test_monitor_fmt.py` — sparkline rendering from history data (empty history → no sparkline, sufficient data → correct bar characters)
- [x] `test_cli.py` additions — `--monitor` without TTY → error exit, `--monitor` with `--compact` accepted
- [x] `test_cli.py` additions — SIGUSR1 triggers refresh in monitor mode (signal handler test)

**Documentation:**
- [x] README.md update — add monitor mode section: `--monitor` usage, key bindings table, `--compact` mode for tmux/waybar, screenshot or example output, daemon integration (reads from DB when daemon running)

**Implementation notes:**

*New files:*
- `src/clawmeter/formatters/monitor_fmt.py` — Rich Live TUI renderer. Uses `rich.live.Live` with `Screen()` layout. Contains: `MonitorDisplay` class (manages layout, refresh, key handling), sparkline renderer, compact-line formatter, help overlay panel.
- `tests/formatters/test_monitor_fmt.py` — TUI formatter unit tests.

*Modified files:*
- `cli.py` — Add `--monitor`, `--compact`, `--interval` flags to the `status` command. Wire up TTY check, monitor launch, and SIGUSR1 handler. `--compact` and `--interval` are silently ignored without `--monitor`.
- `formatters/__init__.py` — Export new formatter.

*Existing code reused:*
- `_STATUS_COLOURS` dict from `table_fmt.py` — extract to a shared location or import directly.
- `format_resets_in_human()` from `json_fmt.py` — countdown timer formatting.
- `HistoryStore.get_latest_statuses()` — data source when daemon is running.
- `HistoryStore.aggregate_samples(granularity="hourly")` — sparkline data source.
- `is_daemon_running()` from `daemon.py` — daemon detection for data source selection.
- `fetch_all()` from `core.py` — standalone mode direct fetching.
- `_resolve_colour()` from `cli.py` — colour detection.

*Implementation phasing:*
1. Core TUI formatter + CLI wiring: `monitor_fmt.py` with `MonitorDisplay`, status command gets `--monitor` flag, TTY check, refresh loop, provider panels with progress bars + countdown timers + colour transitions, `q`/Ctrl+C exit, terminal state restoration, daemon-aware data source.
2. Interactivity + indicators: key bindings (`r`, `1-9`, `j`, `?`), provider health indicators (D-050), daemon status indicator (header), SIGUSR1 handler, rate-limit backoff indicator.
3. Sparklines + compact mode: sparkline rendering (D-046), `--compact` single-line mode (D-045), `[monitor]` config section reading.
4. Tests + documentation: all test cases from checklist, README.md update.

*Key design decisions:* D-045 (compact format), D-046 (sparklines), D-047 (help overlay), D-048 (JSON dump), D-049 (interval flag), D-050 (health thresholds), D-051 (notify deferred).

### v0.5.0 - Grok Provider

- [ ] xAI Grok provider implementation (`providers/grok.py`)
- [ ] Management API integration: invoice preview (MTD spend), spending limits, prepaid balance
- [ ] Usage analytics integration: per-model time-series spend via `POST /v1/billing/teams/{team_id}/usage`
- [ ] Dual credential support: management key (primary) + API key (optional)
- [ ] Team ID resolution from config or `$XAI_TEAM_ID` env var
- [ ] Config section for Grok in `config.py` DEFAULT_CONFIG
- [ ] Provider registration in `providers/__init__.py`
- [ ] Redaction pattern for management keys in `security.py`

**Tests:**
- [ ] `test_providers/test_grok.py` — invoice preview response parsing into UsageWindow (spend MTD, spend vs limit, prepaid balance)
- [ ] `test_providers/test_grok.py` — usage analytics response parsing into ModelUsage entries
- [ ] `test_providers/test_grok.py` — spending limits response parsing (USD cents → USD float)
- [ ] `test_providers/test_grok.py` — prepaid balance response parsing
- [ ] `test_providers/test_grok.py` — management key credential resolution via `$XAI_MANAGEMENT_KEY`
- [ ] `test_providers/test_grok.py` — team_id from config and from `$XAI_TEAM_ID` env var fallback
- [ ] `test_providers/test_grok.py` — `is_configured()` requires management key + team_id
- [ ] `test_providers/test_grok.py` — 401/403 handling (invalid management key)
- [ ] `test_providers/test_grok.py` — 429 handling with backoff
- [ ] `test_providers/test_grok.py` — network error → error in ProviderStatus
- [ ] `test_providers/test_grok.py` — mocked HTTP via `respx` for all management API endpoints

**Documentation:**
- [ ] README.md update — add Grok to supported providers list, dual credential setup (`$XAI_MANAGEMENT_KEY` + optional `$XAI_API_KEY`), team ID configuration, Grok-specific config example, spending limit monitoring explanation

**Implementation notes:**

*New files:*
- `src/clawmeter/providers/grok.py` — Grok provider. Uses Management API (`management-api.x.ai`) as primary data source. Implements `Provider` ABC. Dual credential resolution: `management_key_env`/`management_key_command` for Management API, standard `key_env`/`key_command` for Inference API (optional). Team ID from config `team_id` field or `$XAI_TEAM_ID` env var.
- `tests/providers/test_grok.py` — Grok provider unit tests with respx-mocked HTTP.
- `tests/fixtures/grok_invoice_preview.json` — sample invoice preview response.
- `tests/fixtures/grok_spending_limits.json` — sample spending limits response.
- `tests/fixtures/grok_prepaid_balance.json` — sample prepaid balance response.
- `tests/fixtures/grok_usage_analytics.json` — sample usage analytics response.

*Modified files:*
- `providers/__init__.py` — add `from clawmeter.providers.grok import GrokProvider` import to trigger registration.
- `config.py` — add `grok` section to `DEFAULT_CONFIG` with `enabled: False`, `team_id: ""`, `management_key_env: "XAI_MANAGEMENT_KEY"`.
- `security.py` — add management key redaction pattern (if distinct from `xai-*` pattern).

*Existing code reused:*
- `Provider.resolve_credential()` — for both management key and API key resolution (called twice with different config keys).
- `ProviderStatus`, `UsageWindow`, `ModelUsage` — all fields map naturally to Management API data.
- `compute_status()` — for spend-vs-limit percentage thresholds.
- `ProviderCache` / backoff — no changes needed, works generically.
- All formatters (JSON, table, TUI) — work against `ProviderStatus` with no modifications.

*Key design decisions:*
- Management API is the primary data source. The provider is fully functional with only a management key + team ID. The inference API key is optional and supplementary (rate limit headers only).
- USD cents from the Management API (integer `val` field) are normalised to USD floats (`val / 100`) for `UsageWindow.raw_value`.
- The `cost_in_usd_ticks` field from inference responses (1 tick = 1/10,000,000,000 USD) is not used by the provider; the Management API provides aggregated cost data.
- `team_id` is a required config field (not derived from the API key), because the Management API is team-scoped.
- The usage analytics endpoint (`POST /v1/billing/teams/{team_id}/usage`) uses `timeUnit: TIME_UNIT_DAY` and `groupBy: ["description"]` for per-model daily spend. Time range defaults to current billing cycle (1st of month to now).

*Implementation phasing:*
1. Core provider + credential resolution: `grok.py` with `GrokProvider` class, dual credential resolution, `is_configured()`, `auth_instructions()`, provider registration, config section.
2. Invoice preview + spending limits: `fetch_usage()` calls invoice preview and spending limits endpoints, maps to `UsageWindow` entries (Spend MTD, Spend vs Limit, Prepaid Balance).
3. Usage analytics + per-model breakdown: usage analytics endpoint parsed into `ModelUsage` entries and `extras` dict.
4. Tests + documentation: all test cases from checklist, fixture files, README update.

### v0.6.0 - OpenAI Provider

- [x] OpenAI provider implementation (`providers/openai.py`)
- [x] Usage API integration: `GET /v1/organization/usage/completions` with `group_by=model` for per-model token counts
- [x] Costs API integration: `GET /v1/organization/costs` for MTD spend, `group_by=line_item` for per-model costs
- [x] Merged per-model breakdown: usage tokens + costs joined into `ModelUsage` entries
- [x] Admin key credential resolution via `$OPENAI_ADMIN_KEY` (not standard project key)
- [x] Config section for OpenAI (`providers.openai` with `admin_key_env`)
- [x] Provider registration in `providers/__init__.py`
- [x] Redaction pattern verification (`sk-[a-zA-Z0-9-]{20,}` already covers `sk-admin-*`)

**Tests:**
- [x] Usage API response parsing into per-model `ModelUsage` entries (input/output/cached tokens, request counts)
- [x] Costs API response parsing into "Spend (MTD)" `UsageWindow` (USD)
- [x] Per-model cost parsing via `group_by=line_item`
- [x] Token + cost merge: models from both endpoints combined into unified `ModelUsage`
- [x] Admin key credential resolution via `$OPENAI_ADMIN_KEY`
- [x] `is_configured()` validation (requires admin key)
- [x] 401/403 error handling (wrong key type or missing `api.usage.read` scope)
- [x] 429 rate limit backoff handling
- [x] Network error handling
- [x] Multi-bucket response aggregation (summing across time buckets)
- [x] Mocked HTTP via `respx` for all endpoints

**Documentation:**
- [x] README.md update — add OpenAI to supported providers list, admin key setup (`$OPENAI_ADMIN_KEY`, organisation owner requirement), OpenAI-specific config example, per-model cost and usage breakdown explanation

### v0.7.0 - Ollama Provider

**Config:**
- [x] `config.py` — parse `enable_alpha_features` from `[general]` (default: `false`)
- [x] `config.py` — parse `[providers.ollama]` section: `host`, `hosts` (array form), `poll_interval`, `cloud_enabled`, `api_key_env`, `api_key_command`, `cloud_poll_interval`
- [x] `config.py` — validate mutual exclusivity of `host` (simple) vs `hosts` (array) forms
- [x] `config.py` — expose `is_alpha_enabled()` helper for providers to check

**Core (local instance monitoring — stable):**
- [x] Ollama provider implementation with `@register_provider`
- [x] Multi-host support (single `host` and `[[providers.ollama.hosts]]` array forms)
- [x] Per-host polling: `GET /api/tags` (model inventory + health), `GET /api/ps` (loaded models + VRAM)
- [x] Per-host status, model listing, VRAM/RAM reporting
- [x] Error isolation per host (one host down doesn't affect others)
- [x] `is_configured()` always true when at least one host is set (no credentials needed for local)
- [x] Cloud model detection via `cloud` tag in model names (labelling only)
- [x] Config section for Ollama (local + cloud-ready structure)

**Alpha (cloud usage monitoring — behind `enable_alpha_features`, D-053):**
- [x] `enable_alpha_features` flag in `[general]` config, read by `config.py`
- [x] Alpha feature stderr warning on first use per session
- [x] Ollama Cloud session/weekly usage windows (when `cloud_enabled = true` and alpha flag set)
- [x] Cloud API key authentication via credential chain (`api_key_command` > `api_key_env`/`$OLLAMA_API_KEY` > keyring)
- [x] Probe for `/api/account/usage` endpoint (use if available, graceful failure if not)
- [ ] Fallback: scrape `ollama.com/settings` via authenticated request (cookie or API key)
- [x] Alpha-sourced windows flagged with `alpha: true` in extras dict

**Deferred:**
- [ ] Inference speed / tokens-per-second rolling average (no polling endpoint — per-request only)
- [ ] Per-model token tracking from response `usage` fields (requires proxy/middleware)
- [ ] Prometheus `/metrics` integration (no native endpoint — see OQ-013)

**Tests:**
- [x] `test_providers/test_ollama.py` — single-host response parsing (`/api/ps`, `/api/tags`), multi-host config with per-host labels, host unreachable → error for that host only (other hosts unaffected), VRAM/RAM mapping to UsageWindow, cloud model detection from tag names, no credentials required for local (is_configured always true when host set), mocked HTTP via `respx`
- [x] `test_providers/test_ollama_cloud.py` — cloud usage window parsing, alpha feature gating (disabled when flag off), API key authentication, graceful failure when cloud endpoint unavailable, mocked HTTP via `respx`
- [x] `test_config.py` — `enable_alpha_features` flag loading and default (false)

**Documentation:**
- [x] README.md update — add Ollama to supported providers list, single-host and multi-host configuration examples, VRAM and RAM monitoring explanation, no credentials required note for local, cloud usage monitoring as alpha feature with setup instructions

### v0.7.1 - Claude Extra Usage Spend (Alpha)

**Core (behind `enable_alpha_features`, D-053):**
- [x] Parse `extra_usage` object from `/api/oauth/usage` response (`is_enabled`, `monthly_limit`, `used_credits`, `utilization`)
- [x] Create "Extra Usage" `UsageWindow` with `unit="percent"`, `raw_value` (spent in dollars), `raw_limit` (limit in dollars) — only when `is_enabled` and alpha flag set
- [x] Populate extras dict with `extra_usage_enabled`, `extra_usage_spent`, `extra_usage_limit`
- [x] Skip gracefully when `extra_usage` is `null` or absent from response
- [x] Alpha feature stderr warning reuses existing `_emit_alpha_warning` pattern from Ollama provider (extract to shared utility)

**New response fields:**
- [x] Parse `seven_day_sonnet` window (new per-model window, same structure as `seven_day_opus`)
- [x] Handle `seven_day_cowork` window if non-null (shared/team usage)

**Formatters:**
- [x] Add `"credits"` unit support to `table_fmt.py` `_format_value_and_reset()` — display as `$X.XX` (generic dollar, no currency qualifier)
- [x] Add `"credits"` unit support to `table_fmt.py` bar rendering — no percentage bar for credits
- [x] Add `"credits"` unit support to `monitor_fmt.py` `_build_provider_panel()` — display as `$X.XX`
- [x] Add `"credits"` unit support to `monitor_fmt.py` `format_compact_line()` — display as `$X.XX`

**Tests:**
- [x] `test_providers/test_claude.py` — extra usage parsing (enabled, disabled, absent), alpha gating, `used_credits` exceeding `monthly_limit`, cents-to-dollars conversion, extras dict population
- [x] `test_providers/test_claude.py` — `seven_day_sonnet` window parsing
- [x] `test_formatters/` or existing formatter tests — `"credits"` unit renders as `$X.XX` in table and monitor

**Documentation:**
- [x] README.md alpha features section — update Claude extra usage description from "planned" to active, add setup instructions
- [x] CHANGELOG.md entry

### v0.7.2 - Rename to clawmeter

- [x] Rename Python package from `llm_monitor` to `clawmeter` (`src/llm_monitor/` → `src/clawmeter/`)
- [x] Rename environment variables from `LLM_MONITOR_*` to `CLAWMETER_*`
- [x] Update all internal imports, config paths, CLI entry points, Docker config
- [x] Add migration logic for existing `llm-monitor` config/cache/data directories
- [x] Update all documentation (README, SPEC, CHANGELOG) for new name

### v0.7.3 - Docker Compose Fixes

Tracked as [#24](https://github.com/danielithomas/clawmeter/issues/24).

- [x] Fix incorrect environment variable names in `docker-compose.yml` (`OPENAI_API_KEY` → `OPENAI_ADMIN_KEY`, `XAI_API_KEY` → `XAI_MANAGEMENT_KEY`)
- [x] Fix container build issues

### v0.7.4 - PyPI Deployment

Tracked as [#25](https://github.com/danielithomas/clawmeter/issues/25).

- [x] Add logo and PyPI/Python/licence badges to README.md
- [x] Create `.github/workflows/publish.yml` — tag-triggered build and publish via trusted publishing
- [x] Configure PyPI trusted publisher (GitHub environment `pypi` + PyPI pending publisher)
- [x] Tag `v0.7.4` to trigger first PyPI release
- [x] Verify `uv tool install clawmeter` and `pip install clawmeter` work from PyPI

### v0.8.0 - Local System Metrics Provider

- [ ] NVIDIA GPU metrics via `pynvml`
- [ ] AMD GPU metrics via `rocm-smi` subprocess (basic)
- [ ] CPU/RAM via `psutil`
- [ ] Multi-GPU support
- [ ] Config section for local metrics

**Tests:**
- [ ] `test_providers/test_local.py` — GPU metrics with mocked `pynvml` (utilisation, VRAM, temperature), multi-GPU indexing, CPU/RAM via mocked `psutil`, graceful degradation when no GPU detected, AMD fallback to `rocm-smi` subprocess (mocked), `gpu_backend = "auto"` detection logic

**Documentation:**
- [ ] README.md update — add Local System Metrics to supported providers list, `[local]` extra installation (`uv tool install "clawmeter[local]"`), GPU/CPU/RAM monitoring explanation, multi-GPU support, NVIDIA vs AMD backend configuration

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
- [ ] `clawmeter config set-key --provider <name>` interactive key setup
- [ ] `clawmeter config check` validates permissions, keyring, and provider connectivity
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

The daemon architecture (Section 4.2.7) maps naturally to a Docker container. The container runs `clawmeter daemon run` in the foreground, polling providers on schedule and writing to a SQLite database in a mounted volume. The CLI can then be run on the host (reading the same database) or via `docker exec`.

### 15.2 Dockerfile

```dockerfile
FROM python:3.12-slim

RUN pip install --no-cache-dir clawmeter

# Create non-root user
RUN useradd --create-home --shell /bin/bash monitor
USER monitor

# Default data directory
ENV CLAWMETER_DATA_DIR=/data
ENV CLAWMETER_CACHE_DIR=/data/cache
ENV CLAWMETER_CONTAINER=1

VOLUME /data

ENTRYPOINT ["clawmeter", "daemon", "run"]
```

The base install (no `[local]` extra) is used — cloud providers only, no GPU/system metrics. This keeps the image small and avoids C extension compilation.

### 15.3 Docker Compose

```yaml
services:
  clawmeter:
    build: .
    # or: image: ghcr.io/<user>/clawmeter:latest
    restart: unless-stopped
    volumes:
      - clawmeter-data:/data
      - ${HOME}/.config/clawmeter/config.toml:/home/monitor/.config/clawmeter/config.toml:ro
      # Mount Claude credentials read-only (if using Claude provider)
      - ${HOME}/.claude/.credentials.json:/home/monitor/.claude/.credentials.json:ro
    environment:
      # Cloud provider API keys (alternative to keyring)
      - OPENAI_ADMIN_KEY=${OPENAI_ADMIN_KEY}
      - XAI_API_KEY=${XAI_API_KEY}
      - XAI_MANAGEMENT_KEY=${XAI_MANAGEMENT_KEY}
      - XAI_TEAM_ID=${XAI_TEAM_ID}
      # Override poll interval (optional)
      # - CLAWMETER_POLL_INTERVAL=600

volumes:
  clawmeter-data:
```

### 15.4 Container-Aware Behaviour

When `$CLAWMETER_CONTAINER=1` is set (or `/.dockerenv` is detected):

- **Permission checks are skipped.** Volume mounts have their own UID/permission model; POSIX permission checks on mounted files are unreliable.
- **Keyring is not attempted.** No D-Bus Secret Service daemon is available. Credential resolution skips tier 3 and does not log a warning about keyring unavailability.
- **Desktop notifications are disabled.** No notification daemon exists in a container.
- **`daemon install` / `daemon uninstall` are not available.** No systemd in the container. `daemon run` (foreground) is the only supported mode.
- **`--monitor` and `--ux` are not available.** No TTY by default. Use `docker exec -it` if needed.

### 15.5 Accessing Data from Host

The SQLite database is in the mounted volume. The host CLI can read it directly:

```bash
# Point the host CLI at the container's database
export CLAWMETER_DATA_DIR=/path/to/docker/volume

# Now standard CLI commands read from the daemon's database
clawmeter --now
clawmeter --report --days 7
```

Alternatively, use `docker exec`:

```bash
docker exec clawmeter clawmeter --now
docker exec clawmeter clawmeter --report
```

### 15.6 Health Check

If a health endpoint is implemented (see OQ-023), the Dockerfile adds:

```dockerfile
HEALTHCHECK --interval=60s --timeout=5s --retries=3 \
    CMD ["clawmeter", "daemon", "status", "--quiet"]
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
| Rate limits (developer docs) | https://docs.x.ai/developers/rate-limits |
| Models and pricing | https://docs.x.ai/developers/models |
| Management API reference | https://docs.x.ai/developers/rest-api-reference/management |
| Management API: Billing | https://docs.x.ai/developers/rest-api-reference/management/billing |

### OpenAI

| Source | URL |
|--------|-----|
| Usage Dashboard | https://help.openai.com/en/articles/10478918-api-usage-dashboard |
| Legacy Usage Dashboard | https://help.openai.com/en/articles/8554956-usage-dashboard-legacy |
| Usage API reference | https://platform.openai.com/docs/api-reference/usage |

### Ollama

| Source | URL |
|--------|-----|
| Ollama Cloud docs | https://docs.ollama.com/cloud |
| Ollama pricing | https://ollama.com/pricing |
| Ollama per-response usage fields (not an aggregate API) | https://docs.ollama.com/api/usage |
| Ollama API reference (GitHub) | https://github.com/ollama/ollama/blob/main/docs/api.md |
| Ollama authentication docs | https://docs.ollama.com/api/authentication |
| Cloud usage stats feature request | https://github.com/ollama/ollama/issues/12532 |
| Account Usage API Endpoint request | https://github.com/ollama/ollama/issues/15132 |
| ollama-metrics (Prometheus proxy) | https://github.com/NorskHelsenett/ollama-metrics |
| Metrics endpoint feature request | https://github.com/ollama/ollama/issues/3144 |
| v0.7.0 research report | docs/research/ollama-v0.7.0-research.md |

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
