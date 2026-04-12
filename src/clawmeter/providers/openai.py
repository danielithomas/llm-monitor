"""OpenAI usage provider.

Uses the OpenAI Administration API (Usage API and Costs API) for
organisation-level usage and spend monitoring.  Requires an Admin API
Key (sk-admin-*) with the ``api.usage.read`` scope.

See SPEC.md Section 3.3 for endpoint mapping and design rationale.
See D-052 for the admin key requirement decision.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Optional

import httpx

from clawmeter.models import (
    CredentialError,
    ModelUsage,
    ProviderStatus,
    SecretStr,
    UsageWindow,
    compute_status,
)
from clawmeter.providers import register_provider
from clawmeter.providers.base import Provider
from clawmeter.security import is_container_mode, run_key_command

API_BASE = "https://api.openai.com/v1"


@register_provider
class OpenAIProvider(Provider):
    """Fetches usage and cost data from the OpenAI Administration API."""

    def __init__(self, config: dict) -> None:
        self._config = config

    def name(self) -> str:
        return "openai"

    def display_name(self) -> str:
        return "OpenAI"

    def is_configured(self) -> bool:
        """Requires an admin API key."""
        try:
            return self._resolve_admin_key() is not None
        except CredentialError:
            return False

    def _resolve_admin_key(self) -> SecretStr | None:
        """Resolve the admin key through the credential chain.

        Resolution order mirrors ``Provider.resolve_credential()`` but
        reads ``admin_key_command`` / ``admin_key_env`` config keys.
        """
        provider_cfg = self._config.get("providers", {}).get("openai", {})

        # Tier 1: admin_key_command (hard fail on error)
        key_cmd = provider_cfg.get("admin_key_command")
        if key_cmd:
            return run_key_command(key_cmd)

        # Tier 2: environment variable
        env_var = provider_cfg.get("admin_key_env") or "OPENAI_ADMIN_KEY"
        value = os.environ.get(env_var)
        if value:
            return SecretStr(value)

        # Tier 3: keyring (skip in container mode)
        if not is_container_mode():
            try:
                import keyring as kr

                secret = kr.get_password("clawmeter/openai", "admin_key")
                if secret:
                    return SecretStr(secret)
            except Exception:
                pass

        # Tier 4: nothing found
        return None

    async def fetch_usage(self, client: httpx.AsyncClient) -> ProviderStatus:
        """Fetch usage and cost data from the OpenAI Administration API."""
        now = datetime.now(timezone.utc)

        def _error_status(msg: str, **extra_fields: object) -> ProviderStatus:
            extras = dict(extra_fields) if extra_fields else {}
            return ProviderStatus(
                provider_name=self.name(),
                provider_display=self.display_name(),
                timestamp=now,
                cached=False,
                cache_age_seconds=0,
                errors=[msg],
                extras=extras,
            )

        # Resolve admin key
        try:
            admin_key = self._resolve_admin_key()
        except CredentialError as exc:
            return _error_status(
                f"Admin key command failed: {exc}\n"
                "Fix: Check your admin_key_command configuration."
            )

        if not admin_key:
            return _error_status(
                "OpenAI admin key not found.\n"
                "Fix: Set $OPENAI_ADMIN_KEY or configure admin_key_command\n"
                "in [providers.openai]. Create an admin key at\n"
                "platform.openai.com → Settings → Organisation → Admin Keys.\n"
                "Note: Only Organisation Owners can create admin keys."
            )

        headers = {
            "Authorization": f"Bearer {admin_key.get_secret_value()}",
            "Content-Type": "application/json",
        }

        # Time range: start of current month to now
        start_of_month = now.replace(
            day=1, hour=0, minute=0, second=0, microsecond=0
        )
        start_ts = int(start_of_month.timestamp())

        windows: list[UsageWindow] = []
        model_usage: list[ModelUsage] = []
        errors: list[str] = []
        extras: dict = {}

        # Fetch usage (per-model tokens) and costs concurrently
        usage_data = await self._fetch_endpoint(
            client,
            f"{API_BASE}/organization/usage/completions",
            headers,
            errors,
            params={
                "start_time": str(start_ts),
                "bucket_width": "1d",
                "group_by[]": "model",
            },
        )

        costs_data = await self._fetch_endpoint(
            client,
            f"{API_BASE}/organization/costs",
            headers,
            errors,
            params={
                "start_time": str(start_ts),
                "bucket_width": "1d",
                "group_by[]": "line_item",
            },
        )

        # Check for backoff across all endpoints
        if any("_backoff" in str(e) for e in errors):
            return ProviderStatus(
                provider_name=self.name(),
                provider_display=self.display_name(),
                timestamp=now,
                cached=False,
                cache_age_seconds=0,
                errors=errors,
                extras={"_backoff": True},
            )

        # Parse costs into Spend (MTD) window and per-model costs
        model_costs: dict[str, float] = {}
        if costs_data is not None:
            self._parse_costs(costs_data, windows, model_costs, extras)

        # Parse usage into per-model token counts
        model_tokens: dict[str, dict] = {}
        if usage_data is not None:
            self._parse_usage(usage_data, model_tokens, extras)

        # Merge tokens and costs into ModelUsage entries
        self._merge_model_data(model_tokens, model_costs, model_usage, extras)

        return ProviderStatus(
            provider_name=self.name(),
            provider_display=self.display_name(),
            timestamp=now,
            cached=False,
            cache_age_seconds=0,
            windows=windows,
            model_usage=model_usage,
            extras=extras,
            errors=errors,
        )

    # ------------------------------------------------------------------
    # HTTP helper
    # ------------------------------------------------------------------

    async def _fetch_endpoint(
        self,
        client: httpx.AsyncClient,
        url: str,
        headers: dict,
        errors: list[str],
        params: dict | None = None,
    ) -> dict | None:
        """GET an OpenAI API endpoint, appending errors on failure."""
        try:
            resp = await client.get(
                url,
                headers=headers,
                params=params,
                follow_redirects=False,
            )
        except httpx.ConnectError as exc:
            errors.append(
                f"Cannot reach api.openai.com: {exc}\n"
                "Fix: Check your network connection and DNS resolution."
            )
            return None
        except httpx.TimeoutException as exc:
            errors.append(f"Request to api.openai.com timed out: {exc}")
            return None
        except httpx.HTTPError as exc:
            errors.append(f"HTTP error contacting OpenAI API: {exc}")
            return None

        if resp.status_code == 401:
            errors.append(
                "Authentication failed (HTTP 401).\n"
                "Your OpenAI admin key may be invalid or revoked.\n"
                "Fix: Create a new admin key at platform.openai.com →\n"
                "Settings → Organisation → Admin Keys."
            )
            return None

        if resp.status_code == 403:
            errors.append(
                "Access denied (HTTP 403).\n"
                "Your API key may lack the api.usage.read scope.\n"
                "Fix: Ensure you are using an Admin API Key (sk-admin-*),\n"
                "not a standard project key (sk-proj-*)."
            )
            return None

        if resp.status_code == 429:
            errors.append(
                "Rate limited by OpenAI API (HTTP 429).\n"
                "Cached data will be used until backoff expires. _backoff"
            )
            return None

        if resp.status_code != 200:
            errors.append(
                f"OpenAI API returned HTTP {resp.status_code}.\n"
                f"Response: {resp.text[:200]}"
            )
            return None

        try:
            return resp.json()
        except (ValueError, Exception) as exc:
            errors.append(f"OpenAI API returned invalid JSON: {exc}")
            return None

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_costs(
        self,
        data: dict,
        windows: list[UsageWindow],
        model_costs: dict[str, float],
        extras: dict,
    ) -> None:
        """Parse costs response into Spend (MTD) window and per-model costs."""
        total_spend = 0.0

        for bucket in data.get("data", []):
            for result in bucket.get("results", []):
                amount = result.get("amount", {})
                value = amount.get("value", 0.0)
                try:
                    value = float(value)
                except (ValueError, TypeError):
                    value = 0.0

                total_spend += value

                line_item = result.get("line_item")
                if line_item:
                    model_costs[line_item] = (
                        model_costs.get(line_item, 0.0) + value
                    )

        windows.append(
            UsageWindow(
                name="Spend (MTD)",
                utilisation=0.0,
                resets_at=None,
                status="normal",
                unit="usd",
                raw_value=round(total_spend, 6),
            )
        )

    def _parse_usage(
        self,
        data: dict,
        model_tokens: dict[str, dict],
        extras: dict,
    ) -> None:
        """Parse usage/completions response into per-model token dicts."""
        for bucket in data.get("data", []):
            for result in bucket.get("results", []):
                model = result.get("model") or "unknown"

                if model not in model_tokens:
                    model_tokens[model] = {
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "input_cached_tokens": 0,
                        "num_model_requests": 0,
                    }

                entry = model_tokens[model]
                entry["input_tokens"] += result.get("input_tokens", 0) or 0
                entry["output_tokens"] += result.get("output_tokens", 0) or 0
                entry["input_cached_tokens"] += (
                    result.get("input_cached_tokens", 0) or 0
                )
                entry["num_model_requests"] += (
                    result.get("num_model_requests", 0) or 0
                )

    def _merge_model_data(
        self,
        model_tokens: dict[str, dict],
        model_costs: dict[str, float],
        model_usage: list[ModelUsage],
        extras: dict,
    ) -> None:
        """Merge token counts and costs into unified ModelUsage entries."""
        all_models = sorted(set(model_tokens.keys()) | set(model_costs.keys()))

        for model_name in all_models:
            tokens = model_tokens.get(model_name, {})
            cost = model_costs.get(model_name)

            input_t = tokens.get("input_tokens") or None
            output_t = tokens.get("output_tokens") or None
            total_t = None
            if input_t is not None or output_t is not None:
                total_t = (input_t or 0) + (output_t or 0)

            request_count = tokens.get("num_model_requests") or None

            model_usage.append(
                ModelUsage(
                    model=model_name,
                    input_tokens=input_t,
                    output_tokens=output_t,
                    total_tokens=total_t,
                    cost=cost,
                    request_count=request_count,
                    period="mtd",
                )
            )

        if all_models:
            extras["models_used"] = all_models

        # Top model by spend
        if model_costs:
            top_model = max(model_costs, key=model_costs.get)  # type: ignore[arg-type]
            extras["top_model_spend"] = {
                "model": top_model,
                "cost": round(model_costs[top_model], 6),
            }

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def auth_instructions(self) -> str:
        return (
            "OpenAI monitoring requires an Admin API Key (sk-admin-*).\n"
            "1. Go to platform.openai.com → Settings → Organisation → Admin Keys\n"
            "   (only Organisation Owners can create admin keys)\n"
            "2. Create an admin key with api.usage.read scope\n"
            "3. Set $OPENAI_ADMIN_KEY (or configure admin_key_command\n"
            "   in [providers.openai])"
        )

    @property
    def allowed_hosts(self) -> list[str]:
        return ["api.openai.com"]
