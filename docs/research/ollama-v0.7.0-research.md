# Ollama Provider Research Report — v0.7.0

**Date:** 2026-04-10
**Author:** Daniel Thomas (with research assistance)
**Status:** Reviewed — ready for implementation planning

---

## 1. Ollama Has Two Distinct Modes: Local and Cloud

Ollama is not purely local. Since September 2025 (v0.12), Ollama offers **cloud models** — large models (e.g. DeepSeek V3 671B, GPT-OSS 120B, Qwen3 480B) that run on Ollama's datacenter GPU infrastructure. This creates two fundamentally different monitoring targets within a single provider:

| Aspect | Local Models | Cloud Models |
|--------|-------------|--------------|
| Where they run | User's hardware | Ollama's cloud (NVIDIA GPUs, primarily US) |
| Authentication | None | `ollama signin` + Ollama account, or API key |
| Pricing | Free (unlimited) | Subscription tiers: Free / Pro ($20/mo) / Max ($100/mo) |
| Usage limits | None | Session limits (reset every 5h) + weekly limits (reset every 7d) |
| Usage measurement | N/A | GPU time (not tokens) — varies by model size and request duration |
| Model naming | `llama3:8b`, `gemma3` | Tag includes `cloud`: `gpt-oss:120b-cloud`, `deepseek-v3.2:cloud` |
| API endpoints | `http://localhost:11434/api/*` | Same local API (transparently proxied) OR direct via `https://ollama.com/api/*` |
| Appears in `/api/ps` | Yes (with VRAM, expiry, context) | **No** — no local process, no VRAM allocation |
| Disable cloud | N/A | Set `OLLAMA_NO_CLOUD=1` to reject cloud requests and hide from lists |
| Data retention | Local | "Never logged or trained on" — zero data retention |

### 1.1 Cloud Model Access Paths

There are **two ways** to use cloud models:

**Path A — Via local Ollama (transparent routing):**
```bash
ollama signin                          # authenticate once
ollama run gpt-oss:120b-cloud         # routes to cloud transparently
# API calls to localhost:11434 work as normal — Ollama handles routing
```

**Path B — Direct API access (hosted endpoint):**
```bash
export OLLAMA_API_KEY=your_key         # from ollama.com/settings/keys
curl https://ollama.com/api/chat \
  -H "Authorization: Bearer $OLLAMA_API_KEY" \
  -d '{"model": "gpt-oss:120b-cloud", "messages": [...]}'
```

### 1.2a Authentication Mechanism Details

Two auth mechanisms exist:

- **`ollama signin` (interactive/CLI):** Initiates an OAuth-like flow. Registers the local machine's SSH public key with the user's ollama.com account. Subsequent cloud requests use a challenge-response mechanism — the proxy attaches the user's public key and a signed challenge to request headers to obtain a bearer token.
- **API keys (non-interactive/programmatic):** Created at [ollama.com/settings/keys](https://ollama.com/settings/keys). Set via `OLLAMA_API_KEY` env var. Used as `Authorization: Bearer <key>` header. Keys **do not expire** but can be revoked.

### 1.2b Rate Limiting and Concurrency

- Each plan has **concurrency limits** (1 / 3 / 10 concurrent cloud models for Free / Pro / Max)
- Requests beyond the concurrency limit are **queued**; if the queue is full, the request is **rejected**
- When session or weekly limits are exceeded: **HTTP 429** (Too Many Requests) with a `Retry-After` header
- Premium model requests (e.g. Gemini 3 Pro Preview) may have separate monthly quotas that don't count against session/weekly limits

### 1.2 Cloud Pricing Tiers

| Plan | Price | Concurrent Cloud Models | Cloud Usage | Local Usage |
|------|-------|------------------------|-------------|-------------|
| Free | $0 | 1 | Light allowance | Unlimited |
| Pro | $20/mo ($200/yr) | 3 | 50x Free | Unlimited |
| Max | $100/mo | 10 | 5x Pro (250x Free) | Unlimited |

Usage is measured by **actual GPU infrastructure consumption** (primarily GPU time), not fixed token counts. Shorter requests and cached context use less. Additional per-token usage purchasing is "coming soon."

### 1.3 Current Cloud Models (as of April 2026)

Over 20 cloud models available, including: GLM-5.1, Gemma4 (26b/31b), MiniMax M2.7, Qwen3.5 (up to 122b), Qwen3-VL (up to 235b), DeepSeek V3.2, Nemotron-3 Super (120B MoE), Kimi K2.5, Devstral-2 (123b), Cogito 2.1 (671b), and others. Full list at [ollama.com/search?c=cloud](https://ollama.com/search?c=cloud).

### 1.4 Implication for llm-monitor

The Ollama provider needs to be **two monitors in one**:
1. **Local instance monitor** — poll `/api/ps`, `/api/tags` on local/network hosts for VRAM, loaded models, health
2. **Cloud usage monitor** — track session and weekly quota consumption against plan limits

This is architecturally similar to how our Claude provider tracks utilisation windows with reset times, but the data source is different (and currently problematic — see Section 3).

---

## 2. Available API Endpoints

### 2.1 GET /api/tags — Model Inventory

Lists all downloaded models (local) and available cloud models. Acts as a health check.

```json
{
  "models": [
    {
      "name": "gemma3",
      "model": "gemma3",
      "modified_at": "2025-10-03T23:34:03.409490317-07:00",
      "size": 3338801804,
      "digest": "a2af6cc3eb7fa8be8504abaf9b04e88f17a119ec3f04a3addf55f92841195f5a",
      "details": {
        "format": "gguf",
        "family": "gemma",
        "families": ["gemma"],
        "parameter_size": "4.3B",
        "quantization_level": "Q4_K_M"
      }
    }
  ]
}
```

- `size` = on-disk model file size in bytes
- `details.parameter_size` = human-readable param count (e.g. "4.3B")
- `details.quantization_level` = quant format (e.g. "Q4_K_M")
- Cloud models have `cloud` in their tag (e.g. `gpt-oss:120b-cloud`, `deepseek-v3.2:cloud`)

### 2.2 GET /api/ps — Running Models + VRAM (Primary Local Monitoring Endpoint)

Returns models currently loaded in memory. **Most important endpoint for local monitoring.**

```json
{
  "models": [
    {
      "name": "gemma3",
      "model": "gemma3",
      "size": 6591830464,
      "digest": "a2af6cc3eb7fa8be8504abaf9b04e88f17a119ec3f04a3addf55f92841195f5a",
      "details": {
        "parent_model": "",
        "format": "gguf",
        "family": "gemma3",
        "families": ["gemma3"],
        "parameter_size": "4.3B",
        "quantization_level": "Q4_K_M"
      },
      "expires_at": "2025-10-17T16:47:07.93355-07:00",
      "size_vram": 5333539264,
      "context_length": 4096
    }
  ]
}
```

| Field | Type | Meaning |
|-------|------|---------|
| `size` | int (bytes) | Total memory footprint (RAM + VRAM) |
| `size_vram` | int (bytes) | Portion loaded into GPU VRAM |
| `expires_at` | ISO datetime | When the model will be evicted from memory |
| `context_length` | int | Active context window size |

**Known quirk:** `size_vram` is **omitted** (not zero) when the model runs entirely on CPU ([ollama/ollama#4840](https://github.com/ollama/ollama/issues/4840)). Treat missing `size_vram` as 0 (CPU-only inference).

**Derived metrics:**
- RAM usage per model = `size - size_vram`
- VRAM usage per model = `size_vram`
- Total models loaded = `len(models)`

**Cloud models do NOT appear in `/api/ps`.** They have no local process, no VRAM allocation, and no `keep_alive` timer. Cloud requests are proxied on-demand to Ollama's infrastructure — there is nothing "loaded" locally to report.

### 2.3 POST /api/show — Model Metadata

Request: `{"model": "gemma3"}`

```json
{
  "parameters": "temperature 0.7\nnum_ctx 2048",
  "template": "...",
  "license": "...",
  "modified_at": "2025-10-03T23:34:03.409490317-07:00",
  "capabilities": ["completion", "vision"],
  "details": {
    "parent_model": "",
    "format": "gguf",
    "family": "gemma3",
    "families": ["gemma3"],
    "parameter_size": "4.3B",
    "quantization_level": "Q4_K_M"
  },
  "model_info": {
    "general.architecture": "gemma3",
    "gemma3.attention.head_count": 8,
    "gemma3.block_count": 26,
    "gemma3.context_length": 8192,
    "gemma3.embedding_length": 2048
  }
}
```

Useful for enriching model info (capabilities, architecture, max context). The `capabilities` field is relatively new.

### 2.4 GET /api/version

```json
{"version": "0.12.6"}
```

Simple health/compatibility check.

### 2.5 GET / — Liveness Probe

Returns `200 OK` with body `"Ollama is running"`. Simplest possible health check.

### 2.6 Per-Request Performance Metrics (Response Fields)

Both `/api/generate` and `/api/chat` include timing/token metrics in the final response chunk (when `done: true`). All durations are in **nanoseconds**. These fields are identical for local and cloud models.

```json
{
  "model": "gemma3",
  "done": true,
  "done_reason": "stop",
  "total_duration": 4883583458,
  "load_duration": 1334875,
  "prompt_eval_count": 26,
  "prompt_eval_duration": 342546000,
  "eval_count": 282,
  "eval_duration": 4535599000
}
```

| Field | Meaning | Derived Metric |
|-------|---------|----------------|
| `total_duration` | Wall-clock time (ns) | End-to-end latency |
| `load_duration` | Model load time (ns) | Cold-start detection |
| `prompt_eval_count` | Input tokens processed | Token accounting |
| `prompt_eval_duration` | Prompt processing time (ns) | Prefill speed: `count / duration * 1e9` tok/s |
| `eval_count` | Output tokens generated | Token accounting |
| `eval_duration` | Generation time (ns) | Decode speed: `count / duration * 1e9` tok/s |

**Known issues:**
- `prompt_eval_count` can be **zero** on subsequent requests with the same prompt due to KV cache hits ([#2068](https://github.com/ollama/ollama/issues/2068))
- `eval_duration` may be **missing** when the response is empty ([#8553](https://github.com/ollama/ollama/issues/8553))
- In streaming mode, metrics appear only in the final chunk

**Critical limitation:** These metrics are **per-request only**. There is no polling endpoint that provides aggregate token counts or historical throughput.

### 2.7 Ollama Cloud Hosted API (ollama.com)

The hosted endpoint at `https://ollama.com/api/*` mirrors the local API but requires authentication:

```bash
# List models available to your account
curl https://ollama.com/api/tags \
  -H "Authorization: Bearer $OLLAMA_API_KEY"

# Chat with a cloud model
curl https://ollama.com/api/chat \
  -H "Authorization: Bearer $OLLAMA_API_KEY" \
  -d '{"model": "gpt-oss:120b-cloud", "messages": [{"role": "user", "content": "Hello"}]}'
```

- API keys created at [ollama.com/settings/keys](https://ollama.com/settings/keys)
- Authentication via `Authorization: Bearer <key>` header
- Same endpoint structure as local API (`/api/tags`, `/api/chat`, `/api/generate`, etc.)

---

## 3. The Cloud Usage Tracking Problem

### 3.1 What Exists Today

Cloud usage statistics (session %, weekly %, plan type, reset times) are visible at **ollama.com/settings** — but only in the browser, behind authentication.

**There is no official API endpoint for cloud usage data.**

This is an actively requested feature:
- [ollama/ollama#12532](https://github.com/ollama/ollama/issues/12532) — "Cloud usage stats" (Oct 2025, open, redirects to #15132)
- [ollama/ollama#15132](https://github.com/ollama/ollama/issues/15132) — "Account Usage API Endpoint" (Mar 2026, closed as dup of #12532)

The community proposed endpoint structure (from #15132):

```json
GET https://ollama.com/api/account/usage
Authorization: Bearer <api_key>

{
  "session": {
    "used_percentage": 4.0,
    "resets_at": "2026-03-29T03:00:00Z"
  },
  "weekly": {
    "used_percentage": 14.3,
    "resets_at": "2026-03-30T02:00:00Z"
  },
  "plan": "pro"
}
```

### 3.2 Current Workarounds

The only known workaround (from GitHub issue #12532) involves **scraping the settings page** using authenticated cookies — fragile and not suitable for a production tool.

Other tools facing this problem:
- [steipete/CodexBar#534](https://github.com/steipete/CodexBar/issues/534) — proposes WKWebView/cookie-based scraping to read ollama.com/settings
- Both CodexBar and the issue authors note this is unsatisfactory and await an official API

### 3.3 Decision: Alpha Features Flag (D-053)

Cloud usage tracking ships in v0.7.0 behind a global `enable_alpha_features` flag in config. This approach:

- **Ships the feature** — power users can opt in immediately
- **Manages expectations** — alpha label signals instability, stderr warning on first use
- **Fails gracefully** — alpha errors are swallowed, never fatal
- **Graduates cleanly** — when Ollama ships an official API, the feature moves out from behind the flag

The same flag also applies to **Claude extra usage spend** (OQ-001 / D-005), which has the same problem — data exists in a browser dashboard but has no stable API.

**Alpha cloud usage implementation strategy:**

1. **Probe first:** Check if `ollama.com/api/account/usage` exists (the community-proposed endpoint). If it responds, use it — the feature may graduate quickly once Ollama ships this.
2. **Fallback:** If no API exists, attempt authenticated scraping of `ollama.com/settings` using the API key as a Bearer token (or cookie-based if needed).
3. **Label clearly:** All alpha-sourced data includes `alpha: true` in the extras dict and windows are labelled accordingly in output.
4. **Monitoring:** Track [ollama/ollama#12532](https://github.com/ollama/ollama/issues/12532) — when an official endpoint ships, update the provider and remove the alpha gate.

**Config:**
```toml
[general]
enable_alpha_features = true     # opt-in to unstable data sources

[providers.ollama]
enabled = true
host = "http://localhost:11434"
cloud_enabled = true             # requires enable_alpha_features
api_key_env = "OLLAMA_API_KEY"
cloud_poll_interval = 300        # 5 min default for cloud quota checks
```

---

## 4. What Does NOT Exist (Local API)

| Expected Feature | Reality |
|-----------------|---------|
| Aggregate usage API (`/api/usage` as cumulative) | **Does not exist.** `/api/usage` docs describe per-response fields only |
| Native Prometheus `/metrics` | **Not shipped.** Requested since early 2024 ([#3144](https://github.com/ollama/ollama/issues/3144)), still open |
| OpenTelemetry support | **Not native.** Requested in [#9254](https://github.com/ollama/ollama/issues/9254), not merged |
| Cloud account usage API | **Not shipped.** Requested in [#12532](https://github.com/ollama/ollama/issues/12532), actively wanted by community |
| Local service authentication | **None.** No API keys, tokens, or auth headers on localhost |
| Service discovery/clustering | **None.** No mDNS, gossip, or federation |
| Persistent request log | **None.** If you don't capture metrics at request time, they're lost |

---

## 5. Community Monitoring Approaches

### 5.1 ollama-metrics (NorskHelsenett) — Proxy Approach

A Go-based transparent HTTP proxy that sits in front of Ollama. Intercepts responses to count tokens and measure latency. Exposed Prometheus metrics:

| Metric | Description |
|--------|-------------|
| `ollama_prompt_tokens_total` | Input tokens (cumulative) |
| `ollama_generated_tokens_total` | Output tokens (cumulative) |
| `ollama_request_duration_seconds` | Request latency |
| `ollama_time_per_token_seconds` | Per-token generation latency |
| `ollama_loaded_models` | Count of loaded models |
| `ollama_model_loaded` | Per-model loaded indicator (1/0) |
| `ollama_model_ram_mb` | RAM usage per loaded model |

All labelled by model. Most production-ready community solution, but requires deploying a proxy.

### 5.2 ollama-exporter (frcooper) — Polling Approach

A standalone Prometheus exporter that queries Ollama's API endpoints and exposes metrics. Does not require proxying — just polls `/api/tags` and `/api/ps`. Simpler but cannot capture per-request token data.

### 5.3 OpenTelemetry Integrations — Client-Side Only

- **OpenLIT** — Python SDK that auto-instruments the `ollama` Python library
- **opentelemetry-instrumentation-ollama** — PyPI package wrapping `ollama.chat()` / `ollama.generate()`

These work at the **client SDK level**, not inside Ollama. Only useful if you control the calling code.

### 5.4 LLM-Observability (anglosherif)

Docker Compose stack: Ollama + OpenWebUI + Prometheus + Grafana with pre-built dashboards. Infrastructure-focused, not a library.

---

## 6. Networking and Multi-Host (Local Instances)

### 6.1 Exposing Ollama on a Network

Default bind: `127.0.0.1:11434`. For LAN access:

```bash
OLLAMA_HOST=0.0.0.0:11434  # in systemd unit or docker-compose
```

No built-in TLS — plain HTTP unless a reverse proxy terminates TLS.

### 6.2 Authentication: None (Local API)

**Ollama has zero authentication on its local API.** Anyone who can reach port 11434 can list models, run inference, pull/delete models. Security relies on:

- Network-level controls (firewall, VLANs, Tailscale/WireGuard)
- Reverse proxy with auth (nginx/Caddy/Traefik adding basic auth or API key checking)

Note: The **cloud API** (`ollama.com`) uses `Authorization: Bearer <key>` — this is entirely separate from local instance auth. See Section 1.2a for the two cloud auth mechanisms (`ollama signin` vs API keys).

### 6.3 Reverse Proxy Considerations

When Ollama sits behind nginx/Caddy/Traefik:
- May return 401/403 if auth is enforced
- TLS may be in play (provider should support `https://`)
- Custom base paths are possible (e.g. `/ollama/api/tags`)

### 6.4 Common Multi-Host Patterns

- Multiple independent instances, each with `OLLAMA_HOST=0.0.0.0:11434`
- Docker Compose per host with GPU passthrough
- Tailscale mesh for stable hostnames (e.g. `gpu-box.tailnet:11434`)
- No built-in clustering, federation, or load balancing

### 6.5 Expected Error Scenarios

| Scenario | Error |
|----------|-------|
| Instance down | `ConnectionRefusedError` / `ConnectError` |
| Network unreachable | `ConnectTimeout` |
| Busy (large model loading) | Slow responses, potential read timeout |
| Behind auth proxy | HTTP 401 or 403 |
| Model pulling in progress | `/api/ps` returns empty models list (not an error) |

**Recommended timeouts:** 5s connect, 10s read (LAN assumption).

---

## 7. Gap Analysis: SPEC vs Reality

### 7.1 Major Gap: Cloud Models Not in SPEC

The current SPEC (Section 3.4) describes Ollama as purely a local/network provider. It has no concept of:
- Ollama Cloud subscription tiers
- Cloud usage windows (session/weekly with reset times)
- Cloud API keys or `ollama signin` authentication
- The `ollama.com` hosted API endpoint
- Distinguishing local vs cloud models in output

This is the biggest revision needed.

### 7.2 The Inference Speed Problem

The SPEC checklist includes:
- "Response metrics aggregation (tokens/sec rolling average)"
- "Per-model token tracking via response `usage` fields written to `model_usage` table"

**Problem:** There is no polling endpoint for tokens/sec or aggregate token counts. Per-request metrics (`eval_count`, `eval_duration`) only exist in inference response bodies. A polling-based provider **cannot** capture this data.

**Options:**

| Option | Pros | Cons |
|--------|------|------|
| **A) Drop inference speed from v0.7.0** | Clean, honest. Focus on what's actually pollable. | Loses a headline metric. |
| **B) Optional proxy mode** | Captures per-request data. Full metrics. | Significant complexity. Changes Ollama's network topology. |
| **C) Parse Ollama server logs** | Non-intrusive. | Fragile, format-dependent, not available for remote hosts. |
| **D) Scrape ollama-metrics proxy** | Prometheus metrics already computed. | Adds external dependency. Not all users will deploy it. |

**Recommendation:** Option A for v0.7.0. Ship pollable metrics and document that per-request token tracking requires a future enhancement.

### 7.3 What We CAN Monitor (and Map to ProviderStatus)

**Local instance metrics (pollable now):**

| Window | Source | Unit | Available |
|--------|--------|------|-----------|
| Models Available | `/api/tags` | count | Yes |
| Models Loaded | `/api/ps` | count | Yes |
| VRAM Usage (per host) | `/api/ps` `size_vram` | bytes → MB/GB | Yes |
| RAM Usage (per host) | `/api/ps` `size - size_vram` | bytes → MB/GB | Yes |
| Model Expiry | `/api/ps` `expires_at` | datetime | Yes |
| Inference Speed | Response `eval_count/eval_duration` | tokens/sec | **No** (per-request only) |

**Cloud usage metrics (blocked on Ollama shipping an API):**

| Window | Proposed Source | Unit | Available |
|--------|----------------|------|-----------|
| Session Usage | `ollama.com/api/account/usage` | percent | **No** (API doesn't exist yet) |
| Weekly Usage | `ollama.com/api/account/usage` | percent | **No** (API doesn't exist yet) |
| Plan Type | `ollama.com/api/account/usage` | string | **No** (API doesn't exist yet) |
| Session Reset | `ollama.com/api/account/usage` | datetime | **No** (API doesn't exist yet) |
| Weekly Reset | `ollama.com/api/account/usage` | datetime | **No** (API doesn't exist yet) |

### 7.4 SPEC Corrections Needed

1. **Section 3.4 version label:** Says "v0.5.0" but should be "v0.7.0" per the milestone checklist
2. **Add Ollama Cloud section:** Cloud models, pricing tiers, authentication, usage windows (session/weekly), the `ollama.com` hosted API, API key management
3. **Cloud usage windows:** Design the `UsageWindow` mapping for session/weekly usage (similar to Claude's utilisation windows) — ready for when the API ships
4. **Inference Speed window:** Mark as deferred — not achievable via polling
5. **`/api/usage` reference in Sources table:** Misleading title. This documents per-response fields, not an aggregate endpoint
6. **Config section:** Add cloud config (`api_key`, `cloud_enabled`, etc.) alongside local host config
7. **Allowed hosts:** Add `ollama.com` as an allowed host for cloud API calls (HTTPS only)
8. **Authentication:** Document the two-tier auth model — none for local, Bearer token for cloud

---

## 8. Proposed Config (Revised)

```toml
[providers.ollama]
enabled = true

# ─── Local instance monitoring ───────────────────────────────
# Single host (simple form)
host = "http://localhost:11434"
poll_interval = 60                   # local service, can poll frequently

# Multiple hosts (array form — uncomment instead of 'host')
# [[providers.ollama.hosts]]
# name = "workstation"
# url = "http://localhost:11434"

# [[providers.ollama.hosts]]
# name = "gpu-server"
# url = "http://gpu-server.local:11434"

# [[providers.ollama.hosts]]
# name = "nas-inference"
# url = "http://192.168.1.50:11434"

# ─── Cloud usage monitoring ──────────────────────────────────
# Requires an Ollama account and API key (ollama.com/settings/keys)
# cloud_enabled = true
# api_key_env = "OLLAMA_API_KEY"     # default env var
# api_key_command = "pass show llm-monitor/ollama-cloud"
# cloud_poll_interval = 300          # cloud quota, 5 min is fine
```

### Key design decisions:
- **Separate poll intervals** — local instances benefit from 60s; cloud quota checks don't need to be frequent (5m default)
- **Cloud auth uses the standard credential chain** — `api_key_command` > `api_key_env` (OLLAMA_API_KEY) > keyring
- **Cloud is opt-in** — `cloud_enabled = true` to activate; many users only run local
- **Local hosts have no auth by default** — but could support optional per-host bearer tokens for reverse proxy setups (future enhancement)

---

## 9. Proposed Extras Dict (Revised)

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
    "session_used_pct": 4.0,
    "session_resets_at": "2026-04-10T15:00:00Z",
    "weekly_used_pct": 14.3,
    "weekly_resets_at": "2026-04-13T02:00:00Z"
  }
}
```

The `cloud` section will be `null` or absent until the usage API ships.

---

## 10. Open Questions

| ID | Question | Impact | Recommendation |
|----|----------|--------|----------------|
| **OQ-A** | ~~Should v0.7.0 include cloud usage tracking, or ship local-only and add cloud later?~~ | ~~High~~ | **Resolved (D-053).** Ship cloud in v0.7.0 behind `enable_alpha_features` flag. |
| **OQ-B** | ~~Should we track ollama/ollama#12532 and add cloud support as a patch release?~~ | ~~Medium~~ | **Resolved (D-053).** Cloud ships in v0.7.0 as alpha. Graduates to stable when API lands. |
| **OQ-C** | Should the provider distinguish local vs cloud models in `/api/tags` output? | Medium — UX | **Yes.** Detect `cloud` in the model tag. Show cloud models under a separate section in output. |
| **OQ-D** | Should we support the hosted API at `ollama.com` for users who don't run Ollama locally? | Medium — scope | **Defer to post-v0.7.0.** Focus on local instance monitoring first. The hosted API is an alternative access path, not primary. |
| **OQ-E** | Should v0.7.0 attempt inference speed tracking, or defer it? | High — scope | **Defer.** No polling endpoint exists. Ship pollable metrics only. |
| **OQ-F** | Should `/api/show` be called for each model in the poll loop? | Low — performance | **Skip.** It's a POST per model and slow with many models. Use only for enrichment on demand. |
| **OQ-G** | How should `size_vram: 0` (CPU-only) be displayed? | Low — UX | Show "CPU only" label, don't report 0 MB VRAM. |
| **OQ-H** | Close OQ-013 (Prometheus `/metrics`)? | Medium | **Yes, close it.** Decision: Do not depend on `/metrics`. Use `/api/ps` + `/api/tags`. |
| **OQ-I** | Should we support optional per-host bearer tokens for reverse proxy auth? | Medium — homelab QoL | **Yes, in v0.7.0.** Simple to implement, valuable for real setups. Wrapped in `SecretStr`. |

---

## 11. Recommended v0.7.0 Scope

### In scope — stable (local instance monitoring):
- [ ] Ollama provider implementation with `@register_provider`
- [ ] Multi-host support (single `host` and `[[providers.ollama.hosts]]` array forms)
- [ ] Per-host polling: `GET /api/tags` (inventory + health), `GET /api/ps` (loaded models + VRAM)
- [ ] Per-host status, model listing, VRAM/RAM reporting
- [ ] `is_configured()` always true when at least one host is set (no credentials needed for local)
- [ ] Error isolation per host (one host down doesn't affect others)
- [ ] Cloud model detection via `cloud` tag in model names (labelling only)
- [ ] Config section with local + cloud structure

### In scope — alpha (cloud usage monitoring, behind `enable_alpha_features`):
- [ ] `enable_alpha_features` flag in `[general]` config
- [ ] Alpha feature stderr warning on first use per session
- [ ] Ollama Cloud session/weekly usage windows (when `cloud_enabled = true` + alpha flag)
- [ ] Cloud API key authentication via credential chain
- [ ] Probe for `/api/account/usage`; fallback to scraping `ollama.com/settings`
- [ ] Alpha-sourced windows flagged with `alpha: true` in extras

### Deferred:
- [ ] Inference speed / tokens-per-second rolling average (no polling endpoint)
- [ ] Per-model token tracking from response `usage` fields (requires proxy/middleware)
- [ ] Prometheus `/metrics` integration (no native endpoint)

---

## 12. Sources

| Source | URL |
|--------|-----|
| **Ollama Cloud docs** | https://docs.ollama.com/cloud |
| **Ollama pricing** | https://ollama.com/pricing |
| **Cloud models blog post** | https://ollama.com/blog/cloud-models |
| **Cloud model library** | https://ollama.com/search?c=cloud |
| Cloud usage stats feature request | https://github.com/ollama/ollama/issues/12532 |
| Account Usage API Endpoint request | https://github.com/ollama/ollama/issues/15132 |
| CodexBar Ollama provider issue | https://github.com/steipete/CodexBar/issues/534 |
| Ollama API docs (GitHub) | https://github.com/ollama/ollama/blob/main/docs/api.md |
| Ollama /api/tags docs | https://docs.ollama.com/api/tags |
| Ollama /api/ps docs | https://docs.ollama.com/api/ps |
| Ollama /api/generate docs | https://docs.ollama.com/api/generate |
| Ollama per-response usage fields | https://docs.ollama.com/api/usage |
| Feature request: /metrics endpoint | https://github.com/ollama/ollama/issues/3144 |
| Feature request: OpenTelemetry | https://github.com/ollama/ollama/issues/9254 |
| Bug: size_vram omitted on CPU | https://github.com/ollama/ollama/issues/4840 |
| Bug: prompt_eval_count zero (KV cache) | https://github.com/ollama/ollama/issues/2068 |
| ollama-metrics (Prometheus proxy) | https://github.com/NorskHelsenett/ollama-metrics |
| ollama-exporter (polling exporter) | https://github.com/frcooper/ollama-exporter |
| OpenLIT (OTel instrumentation) | https://docs.openlit.io/latest/sdk/integrations/ollama |
| LLM-Observability (Docker stack) | https://github.com/anglosherif/LLM-Observability |
| Ollama Cloud guide (Knightli) | https://www.knightli.com/en/2026/04/09/ollama-cloud-models-guide/ |
| Ollama authentication docs | https://docs.ollama.com/api/authentication |
| Ollama auth (GitHub source) | https://github.com/ollama/ollama/blob/main/docs/api/authentication.mdx |
| Cloud models (DeepWiki) | https://deepwiki.com/ollama/ollama/4.7-cloud-models |
| Auth and API keys (DeepWiki) | https://deepwiki.com/ollama/ollama/3.6-authentication-and-api-keys |
| Cloud token limit issue | https://github.com/ollama/ollama/issues/13089 |
| Ollama Cloud review (AwesomeAgents) | https://awesomeagents.ai/reviews/review-ollama-cloud/ |
