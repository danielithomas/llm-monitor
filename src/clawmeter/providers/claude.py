"""Claude (Anthropic) usage provider."""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import httpx

from clawmeter.config import emit_alpha_warning, is_alpha_enabled
from clawmeter.models import ProviderStatus, SecretStr, UsageWindow, compute_status
from clawmeter.providers.base import Provider
from clawmeter.providers import register_provider


@register_provider
class ClaudeProvider(Provider):
    """Fetches usage data from the Anthropic Claude OAuth API."""

    def __init__(self, config: dict) -> None:
        self._config = config
        provider_cfg = config.get("providers", {}).get("claude", {})

        # Resolve credentials path
        cred_path = provider_cfg.get("credentials_path") or ""
        if cred_path:
            self._credentials_path = Path(cred_path)
        else:
            config_dir = os.environ.get("CLAUDE_CONFIG_DIR")
            if config_dir:
                self._credentials_path = Path(config_dir) / ".credentials.json"
            else:
                self._credentials_path = (
                    Path.home() / ".claude" / ".credentials.json"
                )

    def name(self) -> str:
        return "claude"

    def display_name(self) -> str:
        return "Anthropic Claude"

    def is_configured(self) -> bool:
        """Check if the credentials file exists and has valid structure."""
        if not self._credentials_path.exists():
            return False
        try:
            data = json.loads(self._credentials_path.read_text())
            oauth = data.get("claudeAiOauth", {})
            return bool(oauth.get("accessToken") and oauth.get("expiresAt"))
        except (json.JSONDecodeError, OSError):
            return False

    def _read_credentials(self) -> tuple[SecretStr, datetime]:
        """Read the access token and expiry from the credentials file.

        Returns:
            Tuple of (access_token, expires_at).

        Raises:
            FileNotFoundError: If credentials file doesn't exist.
            KeyError: If required fields are missing.
            ValueError: If expiresAt can't be parsed.
        """
        data = json.loads(self._credentials_path.read_text())
        oauth = data["claudeAiOauth"]
        token = SecretStr(oauth["accessToken"])
        expires_raw = oauth["expiresAt"]
        # Handle multiple formats: Unix millis (int), Unix seconds (int), ISO 8601 (str)
        if isinstance(expires_raw, (int, float)):
            # Unix timestamp — detect millis vs seconds by magnitude
            if expires_raw > 1e12:
                expires_at = datetime.fromtimestamp(expires_raw / 1000, tz=timezone.utc)
            else:
                expires_at = datetime.fromtimestamp(expires_raw, tz=timezone.utc)
        else:
            expires_str = str(expires_raw)
            if expires_str.endswith("Z"):
                expires_str = expires_str[:-1] + "+00:00"
            expires_at = datetime.fromisoformat(expires_str)
        return token, expires_at

    async def fetch_usage(self, client: httpx.AsyncClient) -> ProviderStatus:
        """Fetch usage from the Anthropic OAuth API."""
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

        # Read credentials
        try:
            token, expires_at = self._read_credentials()
        except FileNotFoundError:
            return _error_status(
                f"Claude credentials not found at {self._credentials_path}\n"
                "The Claude provider requires Claude Code to be installed and authenticated.\n"
                "Fix: Install Claude Code and run 'claude /login' to authenticate.\n"
                "Docs: https://docs.anthropic.com/en/docs/claude-code"
            )
        except (KeyError, json.JSONDecodeError) as exc:
            return _error_status(
                f"Claude credentials file is malformed: {exc}\n"
                f"File: {self._credentials_path}\n"
                "Fix: Re-authenticate with 'claude /login' to regenerate credentials."
            )
        except Exception as exc:
            return _error_status(
                f"Failed to read Claude credentials: {exc}\n"
                f"File: {self._credentials_path}\n"
                "Fix: Check file permissions and re-authenticate with 'claude /login'."
            )

        # Check expiry (5-minute buffer)
        if expires_at <= now + timedelta(minutes=5):
            return _error_status(
                "Claude OAuth token has expired.\n"
                "Claude Code manages token refresh automatically when running.\n"
                "Fix: Run 'claude /login' to refresh your credentials."
            )

        # Make API request
        headers = {
            "Authorization": f"Bearer {token.get_secret_value()}",
            "anthropic-beta": "oauth-2025-04-20",
        }

        try:
            resp = await client.get(
                "https://api.anthropic.com/api/oauth/usage",
                headers=headers,
                follow_redirects=False,
            )
        except httpx.ConnectError as exc:
            return _error_status(
                f"Cannot reach api.anthropic.com: {exc}\n"
                "Fix: Check your network connection and DNS resolution."
            )
        except httpx.TimeoutException as exc:
            return _error_status(
                f"Request to api.anthropic.com timed out: {exc}\n"
                "Fix: Check your network connection or retry with --fresh."
            )
        except httpx.HTTPError as exc:
            return _error_status(f"HTTP error contacting Claude API: {exc}")

        # Handle response codes
        if resp.status_code == 401:
            # Re-read credentials and retry once (Claude Code may have refreshed)
            try:
                new_token, new_expires = self._read_credentials()
            except Exception:
                return _error_status(
                    "Authentication failed (HTTP 401).\n"
                    "Fix: Run 'claude /login' to refresh your credentials."
                )

            if new_token != token:
                headers["Authorization"] = (
                    f"Bearer {new_token.get_secret_value()}"
                )
                try:
                    resp = await client.get(
                        "https://api.anthropic.com/api/oauth/usage",
                        headers=headers,
                        follow_redirects=False,
                    )
                except httpx.HTTPError as exc:
                    return _error_status(f"HTTP error on retry: {exc}")

            if resp.status_code != 200:
                return _error_status(
                    "Authentication failed (HTTP 401).\n"
                    "Your Claude OAuth token may be invalid or revoked.\n"
                    "Fix: Run 'claude /login' to re-authenticate."
                )

        if resp.status_code == 429:
            return _error_status(
                "Rate limited by the Claude API (HTTP 429).\n"
                "Cached data will be used until backoff expires.",
                _backoff=True,
            )

        if resp.status_code != 200:
            return _error_status(
                f"Claude API returned HTTP {resp.status_code}.\n"
                f"Response: {resp.text[:200]}"
            )

        # Parse response
        try:
            data = resp.json()
        except (json.JSONDecodeError, ValueError) as exc:
            return _error_status(
                f"Claude API returned invalid JSON: {exc}\n"
                "This may indicate an API change. Check for clawmeter updates."
            )

        thresholds = self._config.get("thresholds")
        windows = self._parse_windows(data, thresholds)
        extras = self._parse_extras(data)

        return ProviderStatus(
            provider_name=self.name(),
            provider_display=self.display_name(),
            timestamp=now,
            cached=False,
            cache_age_seconds=0,
            windows=windows,
            extras=extras,
        )

    def _parse_windows(
        self, data: dict, thresholds: dict | None
    ) -> list[UsageWindow]:
        """Parse API response into UsageWindow objects."""
        windows: list[UsageWindow] = []

        window_map = {
            "five_hour": "Session (5h)",
            "seven_day": "Weekly (7d)",
            "seven_day_opus": "Weekly Opus (7d)",
            "seven_day_sonnet": "Weekly Sonnet (7d)",
            "seven_day_cowork": "Weekly Cowork (7d)",
        }

        for key, display in window_map.items():
            entry = data.get(key)
            if entry is None:
                continue

            utilisation = entry.get("utilization", 0.0)
            resets_str = entry.get("resets_at")
            resets_at: Optional[datetime] = None
            if resets_str:
                resets_at = datetime.fromisoformat(resets_str)

            status = compute_status(utilisation, thresholds)
            windows.append(
                UsageWindow(
                    name=display,
                    utilisation=utilisation,
                    resets_at=resets_at,
                    status=status,
                    unit="percent",
                )
            )

        # Extra usage (alpha — D-053)
        if is_alpha_enabled(self._config):
            extra = data.get("extra_usage")
            if extra and extra.get("is_enabled"):
                emit_alpha_warning()
                utilisation = extra.get("utilization", 0.0)
                limit_cents = extra.get("monthly_limit", 0)
                spent_cents = extra.get("used_credits", 0.0)

                status = compute_status(utilisation, thresholds)
                windows.append(
                    UsageWindow(
                        name="Extra Usage",
                        utilisation=utilisation,
                        resets_at=None,
                        status=status,
                        unit="percent",
                        raw_value=spent_cents / 100.0,
                        raw_limit=limit_cents / 100.0,
                    )
                )

        return windows

    def _parse_extras(self, data: dict) -> dict:
        """Build the extras dict from the API response."""
        extras: dict = {}

        extra = data.get("extra_usage")
        if extra is None:
            extras["extra_usage_enabled"] = None
        else:
            extras["extra_usage_enabled"] = extra.get("is_enabled", False)
            if extra.get("is_enabled") and is_alpha_enabled(self._config):
                extras["extra_usage_spent"] = extra.get("used_credits", 0.0) / 100.0
                extras["extra_usage_limit"] = extra.get("monthly_limit", 0) / 100.0

        return extras

    def auth_instructions(self) -> str:
        return (
            "Claude usage monitoring requires Claude Code with an active session.\n"
            "1. Install Claude Code (https://docs.anthropic.com/en/docs/claude-code)\n"
            "2. Run: claude /login\n"
            "3. Complete the OAuth flow in your browser.\n"
            "The credentials file will be created automatically."
        )

    @property
    def allowed_hosts(self) -> list[str]:
        return ["api.anthropic.com"]
