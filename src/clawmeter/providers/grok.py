"""xAI Grok usage provider.

Uses the xAI Management API (management-api.x.ai) as the primary data
source for billing, spend, and usage analytics.  Optionally reads rate
limit headers from the Inference API (api.x.ai).

See SPEC.md Section 3.2 for endpoint mapping and design rationale.
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

MANAGEMENT_API_BASE = "https://management-api.x.ai/v1"


@register_provider
class GrokProvider(Provider):
    """Fetches usage data from the xAI Management API."""

    def __init__(self, config: dict) -> None:
        self._config = config
        provider_cfg = config.get("providers", {}).get("grok", {})
        self._team_id = (
            provider_cfg.get("team_id") or os.environ.get("XAI_TEAM_ID", "")
        )

    def name(self) -> str:
        return "grok"

    def display_name(self) -> str:
        return "xAI Grok"

    def is_configured(self) -> bool:
        """Requires a management key and team ID."""
        if not self._team_id:
            return False
        try:
            return self._resolve_management_key() is not None
        except CredentialError:
            return False

    def _resolve_management_key(self) -> SecretStr | None:
        """Resolve the management key through the credential chain.

        Resolution order mirrors ``Provider.resolve_credential()`` but
        reads ``management_key_command`` / ``management_key_env`` config
        keys instead of the standard ``key_command`` / ``env_var``.
        """
        provider_cfg = self._config.get("providers", {}).get("grok", {})

        # Tier 1: management_key_command (hard fail on error)
        key_cmd = provider_cfg.get("management_key_command")
        if key_cmd:
            return run_key_command(key_cmd)

        # Tier 2: environment variable
        env_var = provider_cfg.get("management_key_env") or "XAI_MANAGEMENT_KEY"
        value = os.environ.get(env_var)
        if value:
            return SecretStr(value)

        # Tier 3: keyring (skip in container mode)
        if not is_container_mode():
            try:
                import keyring as kr

                secret = kr.get_password("clawmeter/grok", "management_key")
                if secret:
                    return SecretStr(secret)
            except Exception:
                pass

        # Tier 4: nothing found
        return None

    async def fetch_usage(self, client: httpx.AsyncClient) -> ProviderStatus:
        """Fetch billing and usage data from the xAI Management API."""
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

        # Resolve management key
        try:
            mgmt_key = self._resolve_management_key()
        except CredentialError as exc:
            return _error_status(
                f"Management key command failed: {exc}\n"
                "Fix: Check your management_key_command configuration."
            )

        if not mgmt_key:
            return _error_status(
                "xAI management key not found.\n"
                "Fix: Set $XAI_MANAGEMENT_KEY or configure management_key_command\n"
                "in [providers.grok]. Create a management key at console.x.ai."
            )

        if not self._team_id:
            return _error_status(
                "xAI team ID not configured.\n"
                "Fix: Set team_id in [providers.grok] or $XAI_TEAM_ID.\n"
                "Find your team ID at console.x.ai."
            )

        headers = {
            "Authorization": f"Bearer {mgmt_key.get_secret_value()}",
            "Content-Type": "application/json",
        }
        base = f"{MANAGEMENT_API_BASE}/billing/teams/{self._team_id}"

        # Fetch invoice preview and spending limits concurrently
        windows: list[UsageWindow] = []
        model_usage: list[ModelUsage] = []
        errors: list[str] = []
        extras: dict = {}

        # --- Invoice preview ---
        invoice_data = await self._fetch_endpoint(
            client, f"{base}/postpaid/invoice/preview", headers, errors
        )

        # --- Spending limits ---
        limits_data = await self._fetch_endpoint(
            client, f"{base}/postpaid/spending-limits", headers, errors
        )

        # --- Prepaid balance ---
        balance_data = await self._fetch_endpoint(
            client, f"{base}/prepaid/balance", headers, errors
        )

        # Check for auth failure across all endpoints
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

        # Parse invoice preview
        if invoice_data is not None:
            self._parse_invoice(invoice_data, windows, model_usage, extras)

        # Parse spending limits and compute spend-vs-limit
        spend_mtd_window = next(
            (w for w in windows if w.name == "Spend (MTD)"), None
        )
        if limits_data is not None:
            self._parse_spending_limits(
                limits_data, spend_mtd_window, windows
            )

        # Parse prepaid balance
        if balance_data is not None:
            self._parse_prepaid_balance(balance_data, windows)

        # --- Usage analytics (per-model breakdown) ---
        analytics_data = await self._fetch_usage_analytics(
            client, base, headers, errors
        )
        if analytics_data is not None:
            self._parse_usage_analytics(analytics_data, model_usage)

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
            model_usage=model_usage,
            extras=extras,
            errors=errors,
        )

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    async def _fetch_endpoint(
        self,
        client: httpx.AsyncClient,
        url: str,
        headers: dict,
        errors: list[str],
    ) -> dict | None:
        """GET a management API endpoint, appending errors on failure."""
        try:
            resp = await client.get(
                url, headers=headers, follow_redirects=False
            )
        except httpx.ConnectError as exc:
            errors.append(
                f"Cannot reach management-api.x.ai: {exc}\n"
                "Fix: Check your network connection and DNS resolution."
            )
            return None
        except httpx.TimeoutException as exc:
            errors.append(f"Request to management-api.x.ai timed out: {exc}")
            return None
        except httpx.HTTPError as exc:
            errors.append(f"HTTP error contacting xAI Management API: {exc}")
            return None

        if resp.status_code == 401:
            errors.append(
                "Authentication failed (HTTP 401).\n"
                "Your xAI management key may be invalid or revoked.\n"
                "Fix: Create a new management key at console.x.ai."
            )
            return None

        if resp.status_code == 403:
            errors.append(
                "Access denied (HTTP 403).\n"
                "Your management key may lack permissions for this team.\n"
                "Fix: Verify the team_id and management key permissions."
            )
            return None

        if resp.status_code == 429:
            errors.append(
                "Rate limited by xAI Management API (HTTP 429).\n"
                "Cached data will be used until backoff expires. _backoff"
            )
            return None

        if resp.status_code != 200:
            errors.append(
                f"xAI Management API returned HTTP {resp.status_code}.\n"
                f"Response: {resp.text[:200]}"
            )
            return None

        try:
            return resp.json()
        except (ValueError, Exception) as exc:
            errors.append(f"xAI Management API returned invalid JSON: {exc}")
            return None

    async def _fetch_usage_analytics(
        self,
        client: httpx.AsyncClient,
        base_url: str,
        headers: dict,
        errors: list[str],
    ) -> dict | None:
        """POST the usage analytics endpoint for per-model daily spend."""
        now = datetime.now(timezone.utc)
        start_of_month = now.replace(
            day=1, hour=0, minute=0, second=0, microsecond=0
        )

        payload = {
            "analyticsRequest": {
                "timeRange": {
                    "startTime": start_of_month.strftime("%Y-%m-%d %H:%M:%S"),
                    "endTime": now.strftime("%Y-%m-%d %H:%M:%S"),
                    "timezone": "UTC",
                },
                "timeUnit": "TIME_UNIT_DAY",
                "values": [
                    {
                        "name": "usd",
                        "aggregation": "AGGREGATION_SUM",
                    }
                ],
                "groupBy": ["description"],
                "filters": [],
            }
        }

        try:
            resp = await client.post(
                f"{base_url}/usage",
                headers=headers,
                json=payload,
                follow_redirects=False,
            )
        except httpx.ConnectError as exc:
            errors.append(f"Cannot reach management-api.x.ai for analytics: {exc}")
            return None
        except httpx.TimeoutException as exc:
            errors.append(f"Usage analytics request timed out: {exc}")
            return None
        except httpx.HTTPError as exc:
            errors.append(f"HTTP error fetching usage analytics: {exc}")
            return None

        if resp.status_code == 401:
            # Already reported from other endpoint calls
            return None

        if resp.status_code == 429:
            errors.append(
                "Rate limited on usage analytics (HTTP 429). _backoff"
            )
            return None

        if resp.status_code != 200:
            errors.append(
                f"Usage analytics returned HTTP {resp.status_code}.\n"
                f"Response: {resp.text[:200]}"
            )
            return None

        try:
            return resp.json()
        except (ValueError, Exception) as exc:
            errors.append(f"Usage analytics returned invalid JSON: {exc}")
            return None

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _cents_to_usd(val: str | int | float | None) -> float | None:
        """Convert a USD-cents value (string or numeric) to USD float."""
        if val is None:
            return None
        try:
            return float(val) / 100.0
        except (ValueError, TypeError):
            return None

    def _parse_invoice(
        self,
        data: dict,
        windows: list[UsageWindow],
        model_usage: list[ModelUsage],
        extras: dict,
    ) -> None:
        """Parse invoice preview into Spend (MTD) window and ModelUsage."""
        invoice = data.get("coreInvoice", {})

        # Total MTD spend
        total_val = invoice.get("totalWithCorr", {})
        total_cents = total_val.get("val") if isinstance(total_val, dict) else None
        total_usd = self._cents_to_usd(total_cents)

        if total_usd is not None:
            windows.append(
                UsageWindow(
                    name="Spend (MTD)",
                    utilisation=0.0,  # Updated when spending limits are known
                    resets_at=None,
                    status="normal",
                    unit="usd",
                    raw_value=total_usd,
                )
            )

        # Billing cycle
        cycle = data.get("billingCycle", {})
        if cycle:
            extras["billing_cycle"] = {
                "year": cycle.get("year"),
                "month": cycle.get("month"),
            }

        # Per-model line items
        models_seen: set[str] = set()
        model_costs: dict[str, float] = {}
        model_tokens: dict[str, dict[str, int]] = {}

        for line in invoice.get("lines", []):
            desc = line.get("description", "")
            if not desc:
                continue

            amount_str = line.get("amount", "0")
            try:
                amount_cents = float(amount_str)
            except (ValueError, TypeError):
                amount_cents = 0.0

            units_str = line.get("numUnits", "0")
            try:
                num_units = int(float(units_str))
            except (ValueError, TypeError):
                num_units = 0

            unit_type = line.get("unitType", "").lower()

            models_seen.add(desc)
            model_costs[desc] = model_costs.get(desc, 0.0) + amount_cents

            if desc not in model_tokens:
                model_tokens[desc] = {"input": 0, "output": 0}

            if "input" in unit_type:
                model_tokens[desc]["input"] += num_units
            elif "output" in unit_type:
                model_tokens[desc]["output"] += num_units

        for model_name in sorted(models_seen):
            cost_usd = model_costs.get(model_name, 0.0) / 100.0
            tokens = model_tokens.get(model_name, {})
            input_t = tokens.get("input") or None
            output_t = tokens.get("output") or None
            total_t = None
            if input_t is not None or output_t is not None:
                total_t = (input_t or 0) + (output_t or 0)

            model_usage.append(
                ModelUsage(
                    model=model_name,
                    input_tokens=input_t,
                    output_tokens=output_t,
                    total_tokens=total_t,
                    cost=cost_usd,
                    period="mtd",
                )
            )

        if models_seen:
            extras["models_used"] = sorted(models_seen)

    def _parse_spending_limits(
        self,
        data: dict,
        spend_window: UsageWindow | None,
        windows: list[UsageWindow],
    ) -> None:
        """Parse spending limits into a Spend vs Limit window."""
        limits = data.get("spendingLimits", {})
        hard_sl = limits.get("effectiveHardSl", {})
        hard_cents = hard_sl.get("val") if isinstance(hard_sl, dict) else None
        hard_usd = self._cents_to_usd(hard_cents)

        if hard_usd is not None and hard_usd > 0 and spend_window is not None:
            utilisation = (
                (spend_window.raw_value or 0.0) / hard_usd
            ) * 100.0
            windows.append(
                UsageWindow(
                    name="Spend vs Limit",
                    utilisation=utilisation,
                    resets_at=None,
                    status="normal",  # Set by compute_status later
                    unit="percent",
                    raw_value=spend_window.raw_value,
                    raw_limit=hard_usd,
                )
            )

    def _parse_prepaid_balance(
        self,
        data: dict,
        windows: list[UsageWindow],
    ) -> None:
        """Parse prepaid balance into a Prepaid Balance window."""
        total_val = data.get("total", {})
        total_cents = total_val.get("val") if isinstance(total_val, dict) else None
        total_usd = self._cents_to_usd(total_cents)

        if total_usd is not None:
            windows.append(
                UsageWindow(
                    name="Prepaid Balance",
                    utilisation=0.0,
                    resets_at=None,
                    status="normal",
                    unit="usd",
                    raw_value=total_usd,
                )
            )

    def _parse_usage_analytics(
        self,
        data: dict,
        model_usage: list[ModelUsage],
    ) -> None:
        """Parse usage analytics time-series into ModelUsage entries.

        If a model already has a ModelUsage entry from invoice parsing,
        update its cost with analytics data.  Otherwise create new entries.
        """
        existing = {m.model: m for m in model_usage}

        for series in data.get("timeSeries", []):
            groups = series.get("group", [])
            model_name = groups[0] if groups else "unknown"

            # Sum all data point values for total spend
            total_cost = 0.0
            for dp in series.get("dataPoints", []):
                for v in dp.get("values", []):
                    try:
                        total_cost += float(v)
                    except (ValueError, TypeError):
                        pass

            if model_name in existing:
                # Analytics cost is in USD already — update if invoice
                # didn't provide cost or if analytics is more precise
                if existing[model_name].cost is None:
                    existing[model_name].cost = total_cost
            else:
                model_usage.append(
                    ModelUsage(
                        model=model_name,
                        cost=total_cost,
                        period="mtd",
                    )
                )

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def auth_instructions(self) -> str:
        return (
            "xAI Grok monitoring requires a Management Key and Team ID.\n"
            "1. Go to console.x.ai and create a Management Key\n"
            "2. Set $XAI_MANAGEMENT_KEY (or configure management_key_command)\n"
            "3. Set team_id in [providers.grok] config (or $XAI_TEAM_ID)\n"
            "4. Optionally set $XAI_API_KEY for rate limit data"
        )

    @property
    def allowed_hosts(self) -> list[str]:
        return ["management-api.x.ai", "api.x.ai"]
