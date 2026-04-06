"""Tests for the Claude provider."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest
import respx

from llm_monitor.models import ProviderStatus
from llm_monitor.providers.claude import ClaudeProvider

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_credentials(
    token: str = "sk-ant-oat01-test-token-value",
    expires_at: str | None = None,
) -> dict:
    """Build a credentials dict with a future expiry by default."""
    if expires_at is None:
        future = datetime.now(timezone.utc) + timedelta(hours=2)
        expires_at = future.isoformat()
    return {
        "claudeAiOauth": {
            "accessToken": token,
            "refreshToken": "sk-ant-ort01-test-refresh",
            "expiresAt": expires_at,
        },
    }


def _write_credentials(path: Path, creds: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(creds or _make_credentials()))


def _usage_response() -> dict:
    return json.loads(
        (FIXTURES_DIR / "claude_usage_response.json").read_text()
    )


def _make_provider(tmp_path: Path, creds: dict | None = None) -> ClaudeProvider:
    """Create a ClaudeProvider pointing at tmp_path credentials."""
    creds_path = tmp_path / ".credentials.json"
    _write_credentials(creds_path, creds)
    config = {
        "providers": {
            "claude": {
                "credentials_path": str(creds_path),
            },
        },
    }
    return ClaudeProvider(config)


# ---------------------------------------------------------------------------
# Credential reading
# ---------------------------------------------------------------------------


class TestCredentialReading:
    def test_valid_credentials(self, tmp_path):
        provider = _make_provider(tmp_path)
        token, expires = provider._read_credentials()
        assert token.get_secret_value() == "sk-ant-oat01-test-token-value"
        assert isinstance(expires, datetime)

    def test_missing_credentials_file(self, tmp_path):
        config = {
            "providers": {
                "claude": {
                    "credentials_path": str(tmp_path / "nonexistent.json"),
                },
            },
        }
        provider = ClaudeProvider(config)
        assert provider.is_configured() is False

    def test_is_configured_true(self, tmp_path):
        provider = _make_provider(tmp_path)
        assert provider.is_configured() is True

    def test_is_configured_bad_json(self, tmp_path):
        creds_path = tmp_path / ".credentials.json"
        creds_path.parent.mkdir(parents=True, exist_ok=True)
        creds_path.write_text("not json at all")
        config = {
            "providers": {
                "claude": {"credentials_path": str(creds_path)},
            },
        }
        provider = ClaudeProvider(config)
        assert provider.is_configured() is False


# ---------------------------------------------------------------------------
# fetch_usage
# ---------------------------------------------------------------------------


class TestFetchUsage:
    @respx.mock
    @pytest.mark.asyncio
    async def test_successful_response_three_windows(self, tmp_path):
        """200 with full response produces 3 UsageWindow objects."""
        provider = _make_provider(tmp_path)

        route = respx.get("https://api.anthropic.com/api/oauth/usage").respond(
            200, json=_usage_response()
        )

        async with httpx.AsyncClient() as client:
            status = await provider.fetch_usage(client)

        assert route.called
        assert isinstance(status, ProviderStatus)
        assert len(status.errors) == 0
        assert len(status.windows) == 3
        names = {w.name for w in status.windows}
        assert "Session (5h)" in names
        assert "Weekly (7d)" in names
        assert "Weekly Opus (7d)" in names

    @respx.mock
    @pytest.mark.asyncio
    async def test_null_opus_two_windows(self, tmp_path):
        """When seven_day_opus is null, only 2 windows are returned."""
        provider = _make_provider(tmp_path)

        response_data = _usage_response()
        response_data["seven_day_opus"] = None

        respx.get("https://api.anthropic.com/api/oauth/usage").respond(
            200, json=response_data
        )

        async with httpx.AsyncClient() as client:
            status = await provider.fetch_usage(client)

        assert len(status.windows) == 2
        names = {w.name for w in status.windows}
        assert "Weekly Opus (7d)" not in names

    @pytest.mark.asyncio
    async def test_expired_token_no_http_call(self, tmp_path):
        """Expired token returns error without making HTTP call."""
        past = datetime.now(timezone.utc) - timedelta(hours=1)
        creds = _make_credentials(expires_at=past.isoformat())
        provider = _make_provider(tmp_path, creds)

        async with httpx.AsyncClient() as client:
            status = await provider.fetch_usage(client)

        assert len(status.errors) == 1
        assert "expired" in status.errors[0].lower() or "Token expired" in status.errors[0]

    @respx.mock
    @pytest.mark.asyncio
    async def test_429_backoff(self, tmp_path):
        """429 response returns error with backoff signal."""
        provider = _make_provider(tmp_path)

        respx.get("https://api.anthropic.com/api/oauth/usage").respond(429)

        async with httpx.AsyncClient() as client:
            status = await provider.fetch_usage(client)

        assert len(status.errors) == 1
        assert "Rate limited" in status.errors[0]
        assert status.extras.get("_backoff") is True

    @respx.mock
    @pytest.mark.asyncio
    async def test_401_reread_retry(self, tmp_path):
        """401 triggers credential re-read and retry with new token."""
        creds_path = tmp_path / ".credentials.json"

        # Write initial credentials
        _write_credentials(creds_path, _make_credentials(token="old-token"))

        config = {
            "providers": {
                "claude": {"credentials_path": str(creds_path)},
            },
        }
        provider = ClaudeProvider(config)

        call_count = 0

        def side_effect(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call with old token -> 401
                # Simultaneously update the credentials file
                _write_credentials(
                    creds_path, _make_credentials(token="new-token")
                )
                return httpx.Response(401)
            else:
                # Second call with new token -> 200
                return httpx.Response(200, json=_usage_response())

        respx.get("https://api.anthropic.com/api/oauth/usage").mock(
            side_effect=side_effect
        )

        async with httpx.AsyncClient() as client:
            status = await provider.fetch_usage(client)

        assert call_count == 2
        assert len(status.errors) == 0
        assert len(status.windows) == 3

    @respx.mock
    @pytest.mark.asyncio
    async def test_500_error(self, tmp_path):
        """Non-200/401/429 returns an error status."""
        provider = _make_provider(tmp_path)

        respx.get("https://api.anthropic.com/api/oauth/usage").respond(
            500, text="Internal Server Error"
        )

        async with httpx.AsyncClient() as client:
            status = await provider.fetch_usage(client)

        assert len(status.errors) == 1
        assert "500" in status.errors[0]


# ---------------------------------------------------------------------------
# CLAUDE_CONFIG_DIR override
# ---------------------------------------------------------------------------


class TestConfigDirOverride:
    def test_claude_config_dir_env(self, tmp_path, monkeypatch):
        """$CLAUDE_CONFIG_DIR overrides the default credentials path."""
        custom_dir = tmp_path / "custom-claude"
        custom_dir.mkdir()
        creds_path = custom_dir / ".credentials.json"
        _write_credentials(creds_path)

        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(custom_dir))
        config = {"providers": {"claude": {}}}
        provider = ClaudeProvider(config)

        assert provider._credentials_path == creds_path
        assert provider.is_configured() is True


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


class TestMetadata:
    def test_name(self):
        config = {"providers": {"claude": {}}}
        p = ClaudeProvider(config)
        assert p.name() == "claude"

    def test_display_name(self):
        config = {"providers": {"claude": {}}}
        p = ClaudeProvider(config)
        assert p.display_name() == "Anthropic Claude"

    def test_auth_instructions(self):
        config = {"providers": {"claude": {}}}
        p = ClaudeProvider(config)
        instructions = p.auth_instructions()
        assert "claude /login" in instructions

    def test_allowed_hosts(self):
        config = {"providers": {"claude": {}}}
        p = ClaudeProvider(config)
        assert "api.anthropic.com" in p.allowed_hosts
