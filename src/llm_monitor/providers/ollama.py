"""Ollama usage provider — local instance monitoring and cloud usage (alpha).

Polls one or more Ollama instances via the REST API for model inventory,
loaded model state, and VRAM/RAM consumption.  Optionally tracks Ollama
Cloud session/weekly usage quotas (requires ``enable_alpha_features``).

See SPEC.md Section 3.4 for endpoint mapping and design rationale.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import urlparse

import httpx

from llm_monitor.config import is_alpha_enabled
from llm_monitor.models import (
    CredentialError,
    ProviderStatus,
    SecretStr,
    UsageWindow,
    compute_status,
)
from llm_monitor.providers import register_provider
from llm_monitor.providers.base import Provider
from llm_monitor.security import is_container_mode, run_key_command

CLOUD_API_BASE = "https://ollama.com"

# Module-level flag: emit the alpha warning at most once per process
_alpha_warning_emitted = False


def _emit_alpha_warning() -> None:
    """Print one-time alpha feature warning to stderr."""
    global _alpha_warning_emitted
    if not _alpha_warning_emitted:
        print(
            "Warning: Alpha features are enabled. Some data sources are "
            "unstable and may break between releases.",
            file=sys.stderr,
        )
        _alpha_warning_emitted = True


@register_provider
class OllamaProvider(Provider):
    """Monitors Ollama local instances and (optionally) cloud usage."""

    def __init__(self, config: dict) -> None:
        self._config = config
        provider_cfg = config.get("providers", {}).get("ollama", {})

        # Build host list from config
        self._hosts = self._resolve_hosts(provider_cfg)
        self._cloud_enabled = bool(provider_cfg.get("cloud_enabled", False))

    # ------------------------------------------------------------------
    # Host resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_hosts(provider_cfg: dict) -> list[dict]:
        """Build a list of host dicts from config.

        Returns a list of ``{"name": str, "url": str}`` dicts.
        Supports both the simple ``host`` key and the array ``hosts`` form.
        """
        hosts_array = provider_cfg.get("hosts")
        if hosts_array and isinstance(hosts_array, list):
            result = []
            for entry in hosts_array:
                if isinstance(entry, dict):
                    url = entry.get("url", "")
                    name = entry.get("name", urlparse(url).hostname or url)
                    result.append({"name": name, "url": url.rstrip("/")})
            return result

        # Simple single-host form
        host = provider_cfg.get("host", "http://localhost:11434")
        if host:
            parsed = urlparse(host)
            name = parsed.hostname or "localhost"
            return [{"name": name, "url": host.rstrip("/")}]

        return [{"name": "localhost", "url": "http://localhost:11434"}]

    # ------------------------------------------------------------------
    # Provider ABC
    # ------------------------------------------------------------------

    def name(self) -> str:
        return "ollama"

    def display_name(self) -> str:
        return "Ollama"

    def is_configured(self) -> bool:
        """Always true when at least one host is set — no credentials needed for local."""
        return len(self._hosts) > 0

    def auth_instructions(self) -> str:
        return (
            "Ollama local monitoring requires no credentials.\n"
            "1. Install Ollama: https://ollama.com/download\n"
            "2. Start the service: ollama serve\n"
            "3. Enable in config: [providers.ollama] enabled = true\n\n"
            "For cloud usage monitoring (alpha):\n"
            "1. Sign in: ollama signin\n"
            "2. Create an API key at ollama.com/settings/keys\n"
            "3. Set $OLLAMA_API_KEY\n"
            "4. Set cloud_enabled = true and enable_alpha_features = true"
        )

    @property
    def allowed_hosts(self) -> list[str]:
        hosts = []
        for h in self._hosts:
            parsed = urlparse(h["url"])
            if parsed.hostname:
                hosts.append(parsed.hostname)
        if self._cloud_enabled:
            hosts.append("ollama.com")
        return hosts

    # ------------------------------------------------------------------
    # Credential resolution (cloud only)
    # ------------------------------------------------------------------

    def _resolve_cloud_key(self) -> SecretStr | None:
        """Resolve the Ollama Cloud API key through the credential chain.

        Resolution order mirrors ``Provider.resolve_credential()``:
        1. api_key_command (hard fail on error)
        2. Environment variable (api_key_env, default $OLLAMA_API_KEY)
        3. Keyring
        4. None
        """
        provider_cfg = self._config.get("providers", {}).get("ollama", {})

        # Tier 1: api_key_command (hard fail on error)
        key_cmd = provider_cfg.get("api_key_command")
        if key_cmd:
            return run_key_command(key_cmd)

        # Tier 2: environment variable
        env_var = provider_cfg.get("api_key_env") or "OLLAMA_API_KEY"
        value = os.environ.get(env_var)
        if value:
            return SecretStr(value)

        # Tier 3: keyring (skip in container mode)
        if not is_container_mode():
            try:
                import keyring as kr

                secret = kr.get_password("llm-monitor/ollama", "api_key")
                if secret:
                    return SecretStr(secret)
            except Exception:
                pass

        # Tier 4: nothing found
        return None

    # ------------------------------------------------------------------
    # fetch_usage
    # ------------------------------------------------------------------

    async def fetch_usage(self, client: httpx.AsyncClient) -> ProviderStatus:
        """Fetch model state from all configured hosts and optional cloud usage."""
        now = datetime.now(timezone.utc)
        windows: list[UsageWindow] = []
        errors: list[str] = []
        extras: dict = {"hosts": []}

        # Poll each local/network host
        for host in self._hosts:
            host_data = await self._poll_host(client, host, windows, errors)
            extras["hosts"].append(host_data)

        # Cloud usage (alpha-gated)
        if self._cloud_enabled and is_alpha_enabled(self._config):
            _emit_alpha_warning()
            cloud_data = await self._fetch_cloud_usage(client, windows, errors)
            if cloud_data:
                extras["cloud"] = cloud_data

        thresholds = self._config.get("thresholds")
        for w in windows:
            if w.unit == "percent":
                w.status = compute_status(w.utilisation, thresholds)

        return ProviderStatus(
            provider_name=self.name(),
            provider_display=self.display_name(),
            timestamp=now,
            cached=False,
            cache_age_seconds=0,
            windows=windows,
            extras=extras,
            errors=errors,
        )

    # ------------------------------------------------------------------
    # Per-host polling
    # ------------------------------------------------------------------

    async def _poll_host(
        self,
        client: httpx.AsyncClient,
        host: dict,
        windows: list[UsageWindow],
        errors: list[str],
    ) -> dict:
        """Poll a single Ollama host for /api/tags and /api/ps."""
        host_name = host["name"]
        base_url = host["url"]
        prefix = f"{host_name}: " if len(self._hosts) > 1 else ""

        host_data: dict = {
            "name": host_name,
            "url": base_url,
            "status": "unknown",
            "version": None,
            "models_available": 0,
            "models_loaded": [],
            "total_vram_used_mb": 0,
            "total_ram_used_mb": 0,
        }

        # Fetch /api/tags (model inventory + health check)
        tags_data = await self._fetch_host_endpoint(
            client, f"{base_url}/api/tags", host_name, errors
        )
        if tags_data is None:
            host_data["status"] = "unreachable"
            return host_data

        host_data["status"] = "connected"

        # Parse model inventory
        models = tags_data.get("models", [])
        host_data["models_available"] = len(models)

        # Detect cloud models
        cloud_models = []
        local_models = []
        for m in models:
            model_name = m.get("name", "")
            if ":cloud" in model_name or model_name.endswith("-cloud"):
                cloud_models.append(model_name)
            else:
                local_models.append(model_name)

        if cloud_models:
            host_data["cloud_models"] = cloud_models

        windows.append(
            UsageWindow(
                name=f"{prefix}Models Available",
                utilisation=0.0,
                resets_at=None,
                status="normal",
                unit="count",
                raw_value=float(len(models)),
            )
        )

        # Fetch /api/ps (loaded models + VRAM)
        ps_data = await self._fetch_host_endpoint(
            client, f"{base_url}/api/ps", host_name, errors
        )
        if ps_data is None:
            # /api/ps failure is not fatal — we still have /api/tags data
            return host_data

        loaded = ps_data.get("models", [])
        total_vram = 0
        total_ram = 0

        for m in loaded:
            size_bytes = m.get("size", 0)
            # size_vram omitted when CPU-only (ollama/ollama#4840) — treat as 0
            size_vram = m.get("size_vram", 0)
            size_ram = size_bytes - size_vram

            total_vram += size_vram
            total_ram += max(0, size_ram)

            host_data["models_loaded"].append({
                "name": m.get("name", "unknown"),
                "parameter_size": m.get("details", {}).get("parameter_size", ""),
                "quantization": m.get("details", {}).get("quantization_level", ""),
                "size_bytes": size_bytes,
                "size_vram_bytes": size_vram,
                "expires_at": m.get("expires_at"),
            })

        host_data["total_vram_used_mb"] = round(total_vram / (1024 * 1024))
        host_data["total_ram_used_mb"] = round(total_ram / (1024 * 1024))

        windows.append(
            UsageWindow(
                name=f"{prefix}Models Loaded",
                utilisation=0.0,
                resets_at=None,
                status="normal",
                unit="count",
                raw_value=float(len(loaded)),
            )
        )

        if total_vram > 0:
            windows.append(
                UsageWindow(
                    name=f"{prefix}VRAM Usage",
                    utilisation=0.0,
                    resets_at=None,
                    status="normal",
                    unit="mb",
                    raw_value=round(total_vram / (1024 * 1024), 1),
                )
            )

        if total_ram > 0:
            windows.append(
                UsageWindow(
                    name=f"{prefix}RAM Usage",
                    utilisation=0.0,
                    resets_at=None,
                    status="normal",
                    unit="mb",
                    raw_value=round(total_ram / (1024 * 1024), 1),
                )
            )

        return host_data

    # ------------------------------------------------------------------
    # Cloud usage (alpha)
    # ------------------------------------------------------------------

    async def _fetch_cloud_usage(
        self,
        client: httpx.AsyncClient,
        windows: list[UsageWindow],
        errors: list[str],
    ) -> dict | None:
        """Fetch cloud usage data (alpha — no stable API exists)."""
        try:
            api_key = self._resolve_cloud_key()
        except CredentialError as exc:
            errors.append(
                f"Ollama Cloud API key command failed: {exc}\n"
                "Fix: Check your api_key_command configuration."
            )
            return None

        if not api_key:
            errors.append(
                "Ollama Cloud API key not found.\n"
                "Fix: Set $OLLAMA_API_KEY or configure api_key_command\n"
                "in [providers.ollama]. Create a key at ollama.com/settings/keys."
            )
            return None

        headers = {
            "Authorization": f"Bearer {api_key.get_secret_value()}",
        }

        # Probe for the proposed /api/account/usage endpoint
        usage_data = await self._fetch_cloud_endpoint(
            client, f"{CLOUD_API_BASE}/api/account/usage", headers, errors
        )

        cloud_info: dict = {
            "status": "authenticated",
            "alpha": True,
        }

        if usage_data is not None:
            self._parse_cloud_usage(usage_data, windows, cloud_info)
        else:
            cloud_info["status"] = "no_usage_endpoint"
            # The endpoint doesn't exist yet — this is expected (alpha)
            # Don't add to errors since it's a known limitation

        return cloud_info

    def _parse_cloud_usage(
        self,
        data: dict,
        windows: list[UsageWindow],
        cloud_info: dict,
    ) -> None:
        """Parse cloud usage response into windows."""
        plan = data.get("plan", "")
        if plan:
            cloud_info["plan"] = plan

        session = data.get("session", {})
        if session:
            used_pct = session.get("used_percent", 0.0)
            resets_at_str = session.get("resets_at")
            resets_at = None
            if resets_at_str:
                try:
                    resets_at = datetime.fromisoformat(resets_at_str)
                except (ValueError, TypeError):
                    pass

            cloud_info["session_used_pct"] = used_pct
            if resets_at_str:
                cloud_info["session_resets_at"] = resets_at_str

            windows.append(
                UsageWindow(
                    name="Cloud Session",
                    utilisation=float(used_pct),
                    resets_at=resets_at,
                    status="normal",
                    unit="percent",
                )
            )

        weekly = data.get("weekly", {})
        if weekly:
            used_pct = weekly.get("used_percent", 0.0)
            resets_at_str = weekly.get("resets_at")
            resets_at = None
            if resets_at_str:
                try:
                    resets_at = datetime.fromisoformat(resets_at_str)
                except (ValueError, TypeError):
                    pass

            cloud_info["weekly_used_pct"] = used_pct
            if resets_at_str:
                cloud_info["weekly_resets_at"] = resets_at_str

            windows.append(
                UsageWindow(
                    name="Cloud Weekly",
                    utilisation=float(used_pct),
                    resets_at=resets_at,
                    status="normal",
                    unit="percent",
                )
            )

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    async def _fetch_host_endpoint(
        self,
        client: httpx.AsyncClient,
        url: str,
        host_name: str,
        errors: list[str],
    ) -> dict | None:
        """GET an Ollama host endpoint, appending errors on failure."""
        try:
            resp = await client.get(url, follow_redirects=False)
        except httpx.ConnectError as exc:
            errors.append(
                f"Cannot reach Ollama host '{host_name}': {exc}\n"
                "Fix: Check that Ollama is running (ollama serve) and the host URL is correct."
            )
            return None
        except httpx.TimeoutException as exc:
            errors.append(f"Request to Ollama host '{host_name}' timed out: {exc}")
            return None
        except httpx.HTTPError as exc:
            errors.append(f"HTTP error contacting Ollama host '{host_name}': {exc}")
            return None

        if resp.status_code != 200:
            errors.append(
                f"Ollama host '{host_name}' returned HTTP {resp.status_code} for {url}."
            )
            return None

        try:
            return resp.json()
        except (ValueError, Exception) as exc:
            errors.append(f"Ollama host '{host_name}' returned invalid JSON: {exc}")
            return None

    async def _fetch_cloud_endpoint(
        self,
        client: httpx.AsyncClient,
        url: str,
        headers: dict,
        errors: list[str],
    ) -> dict | None:
        """GET an Ollama Cloud endpoint (alpha), appending errors on failure."""
        try:
            resp = await client.get(
                url, headers=headers, follow_redirects=False
            )
        except httpx.ConnectError as exc:
            errors.append(f"Cannot reach ollama.com: {exc}")
            return None
        except httpx.TimeoutException as exc:
            errors.append(f"Request to ollama.com timed out: {exc}")
            return None
        except httpx.HTTPError as exc:
            errors.append(f"HTTP error contacting ollama.com: {exc}")
            return None

        if resp.status_code == 401:
            errors.append(
                "Ollama Cloud authentication failed (HTTP 401).\n"
                "Your API key may be invalid or revoked.\n"
                "Fix: Create a new key at ollama.com/settings/keys."
            )
            return None

        if resp.status_code == 404:
            # Expected — the endpoint doesn't exist yet
            return None

        if resp.status_code == 429:
            errors.append(
                "Rate limited by Ollama Cloud (HTTP 429). _backoff"
            )
            return None

        if resp.status_code != 200:
            return None

        try:
            return resp.json()
        except (ValueError, Exception):
            return None
