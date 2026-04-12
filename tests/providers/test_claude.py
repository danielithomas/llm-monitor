"""Tests for the Claude provider."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest
import respx

from clawmeter.models import ProviderStatus
from clawmeter.providers.claude import ClaudeProvider

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


def _make_provider(
    tmp_path: Path,
    creds: dict | None = None,
    alpha: bool = False,
) -> ClaudeProvider:
    """Create a ClaudeProvider pointing at tmp_path credentials."""
    creds_path = tmp_path / ".credentials.json"
    _write_credentials(creds_path, creds)
    config = {
        "general": {
            "enable_alpha_features": alpha,
        },
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
    async def test_successful_response_windows(self, tmp_path):
        """200 with full response produces expected UsageWindow objects."""
        provider = _make_provider(tmp_path)

        route = respx.get("https://api.anthropic.com/api/oauth/usage").respond(
            200, json=_usage_response()
        )

        async with httpx.AsyncClient() as client:
            status = await provider.fetch_usage(client)

        assert route.called
        assert isinstance(status, ProviderStatus)
        assert len(status.errors) == 0
        names = {w.name for w in status.windows}
        assert "Session (5h)" in names
        assert "Weekly (7d)" in names
        assert "Weekly Opus (7d)" in names
        assert "Weekly Sonnet (7d)" in names
        # Extra Usage not present without alpha flag
        assert "Extra Usage" not in names

    @respx.mock
    @pytest.mark.asyncio
    async def test_null_opus_skipped(self, tmp_path):
        """When seven_day_opus is null, that window is not returned."""
        provider = _make_provider(tmp_path)

        response_data = _usage_response()
        response_data["seven_day_opus"] = None

        respx.get("https://api.anthropic.com/api/oauth/usage").respond(
            200, json=response_data
        )

        async with httpx.AsyncClient() as client:
            status = await provider.fetch_usage(client)

        names = {w.name for w in status.windows}
        assert "Weekly Opus (7d)" not in names
        # Other windows still present
        assert "Session (5h)" in names
        assert "Weekly (7d)" in names

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
        assert len(status.windows) >= 3

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


# ---------------------------------------------------------------------------
# seven_day_sonnet window
# ---------------------------------------------------------------------------


class TestSonnetWindow:
    @respx.mock
    @pytest.mark.asyncio
    async def test_sonnet_window_parsed(self, tmp_path):
        """seven_day_sonnet is mapped to Weekly Sonnet (7d) window."""
        provider = _make_provider(tmp_path)

        respx.get("https://api.anthropic.com/api/oauth/usage").respond(
            200, json=_usage_response()
        )

        async with httpx.AsyncClient() as client:
            status = await provider.fetch_usage(client)

        sonnet = next(
            (w for w in status.windows if w.name == "Weekly Sonnet (7d)"), None
        )
        assert sonnet is not None
        assert sonnet.utilisation == 5.0
        assert sonnet.unit == "percent"

    @respx.mock
    @pytest.mark.asyncio
    async def test_null_sonnet_skipped(self, tmp_path):
        """When seven_day_sonnet is null, no Sonnet window is returned."""
        provider = _make_provider(tmp_path)

        data = _usage_response()
        data["seven_day_sonnet"] = None
        respx.get("https://api.anthropic.com/api/oauth/usage").respond(
            200, json=data
        )

        async with httpx.AsyncClient() as client:
            status = await provider.fetch_usage(client)

        names = {w.name for w in status.windows}
        assert "Weekly Sonnet (7d)" not in names


# ---------------------------------------------------------------------------
# Extra usage (alpha)
# ---------------------------------------------------------------------------


class TestExtraUsage:
    @respx.mock
    @pytest.mark.asyncio
    async def test_extra_usage_with_alpha(self, tmp_path, monkeypatch):
        """Extra Usage window appears when alpha enabled and extra_usage present."""
        import clawmeter.config as config_mod
        monkeypatch.setattr(config_mod, "_alpha_warning_emitted", False)

        provider = _make_provider(tmp_path, alpha=True)

        respx.get("https://api.anthropic.com/api/oauth/usage").respond(
            200, json=_usage_response()
        )

        async with httpx.AsyncClient() as client:
            status = await provider.fetch_usage(client)

        extra = next(
            (w for w in status.windows if w.name == "Extra Usage"), None
        )
        assert extra is not None
        assert extra.utilisation == 42.5
        assert extra.unit == "percent"
        # 4250 cents / 100 = $42.50
        assert extra.raw_value == pytest.approx(42.50)
        # 10000 cents / 100 = $100.00
        assert extra.raw_limit == pytest.approx(100.00)

    @respx.mock
    @pytest.mark.asyncio
    async def test_extra_usage_hidden_without_alpha(self, tmp_path):
        """Extra Usage window NOT present when alpha disabled."""
        provider = _make_provider(tmp_path, alpha=False)

        respx.get("https://api.anthropic.com/api/oauth/usage").respond(
            200, json=_usage_response()
        )

        async with httpx.AsyncClient() as client:
            status = await provider.fetch_usage(client)

        names = {w.name for w in status.windows}
        assert "Extra Usage" not in names

    @respx.mock
    @pytest.mark.asyncio
    async def test_extra_usage_disabled_on_account(self, tmp_path):
        """No Extra Usage window when is_enabled is false."""
        provider = _make_provider(tmp_path, alpha=True)

        data = _usage_response()
        data["extra_usage"] = {
            "is_enabled": False,
            "monthly_limit": 0,
            "used_credits": 0.0,
            "utilization": 0.0,
        }
        respx.get("https://api.anthropic.com/api/oauth/usage").respond(
            200, json=data
        )

        async with httpx.AsyncClient() as client:
            status = await provider.fetch_usage(client)

        names = {w.name for w in status.windows}
        assert "Extra Usage" not in names

    @respx.mock
    @pytest.mark.asyncio
    async def test_extra_usage_absent_from_response(self, tmp_path):
        """No Extra Usage window when extra_usage field is missing entirely."""
        provider = _make_provider(tmp_path, alpha=True)

        data = _usage_response()
        del data["extra_usage"]
        respx.get("https://api.anthropic.com/api/oauth/usage").respond(
            200, json=data
        )

        async with httpx.AsyncClient() as client:
            status = await provider.fetch_usage(client)

        names = {w.name for w in status.windows}
        assert "Extra Usage" not in names

    @respx.mock
    @pytest.mark.asyncio
    async def test_extra_usage_exceeds_limit(self, tmp_path, monkeypatch):
        """used_credits can exceed monthly_limit."""
        import clawmeter.config as config_mod
        monkeypatch.setattr(config_mod, "_alpha_warning_emitted", False)

        provider = _make_provider(tmp_path, alpha=True)

        data = _usage_response()
        data["extra_usage"] = {
            "is_enabled": True,
            "monthly_limit": 10000,
            "used_credits": 10010.0,
            "utilization": 100.0,
        }
        respx.get("https://api.anthropic.com/api/oauth/usage").respond(
            200, json=data
        )

        async with httpx.AsyncClient() as client:
            status = await provider.fetch_usage(client)

        extra = next(w for w in status.windows if w.name == "Extra Usage")
        assert extra.raw_value == pytest.approx(100.10)
        assert extra.raw_limit == pytest.approx(100.00)
        assert extra.utilisation == 100.0

    @respx.mock
    @pytest.mark.asyncio
    async def test_alpha_warning_emitted(self, tmp_path, monkeypatch, capfd):
        """Alpha warning is emitted to stderr when extra usage parsed."""
        import clawmeter.config as config_mod
        monkeypatch.setattr(config_mod, "_alpha_warning_emitted", False)

        provider = _make_provider(tmp_path, alpha=True)

        respx.get("https://api.anthropic.com/api/oauth/usage").respond(
            200, json=_usage_response()
        )

        async with httpx.AsyncClient() as client:
            await provider.fetch_usage(client)

        captured = capfd.readouterr()
        assert "alpha features are enabled" in captured.err.lower()


# ---------------------------------------------------------------------------
# Extras dict
# ---------------------------------------------------------------------------


class TestExtras:
    @respx.mock
    @pytest.mark.asyncio
    async def test_extras_with_extra_usage_alpha(self, tmp_path, monkeypatch):
        """Extras dict includes spend/limit when alpha enabled."""
        import clawmeter.config as config_mod
        monkeypatch.setattr(config_mod, "_alpha_warning_emitted", False)

        provider = _make_provider(tmp_path, alpha=True)

        respx.get("https://api.anthropic.com/api/oauth/usage").respond(
            200, json=_usage_response()
        )

        async with httpx.AsyncClient() as client:
            status = await provider.fetch_usage(client)

        assert status.extras["extra_usage_enabled"] is True
        assert status.extras["extra_usage_spent"] == pytest.approx(42.50)
        assert status.extras["extra_usage_limit"] == pytest.approx(100.00)

    @respx.mock
    @pytest.mark.asyncio
    async def test_extras_without_alpha(self, tmp_path):
        """Extras shows enabled status but no spend/limit without alpha."""
        provider = _make_provider(tmp_path, alpha=False)

        respx.get("https://api.anthropic.com/api/oauth/usage").respond(
            200, json=_usage_response()
        )

        async with httpx.AsyncClient() as client:
            status = await provider.fetch_usage(client)

        assert status.extras["extra_usage_enabled"] is True
        assert "extra_usage_spent" not in status.extras

    @respx.mock
    @pytest.mark.asyncio
    async def test_extras_extra_usage_null(self, tmp_path):
        """Extras shows null when extra_usage absent from response."""
        provider = _make_provider(tmp_path)

        data = _usage_response()
        del data["extra_usage"]
        respx.get("https://api.anthropic.com/api/oauth/usage").respond(
            200, json=data
        )

        async with httpx.AsyncClient() as client:
            status = await provider.fetch_usage(client)

        assert status.extras["extra_usage_enabled"] is None
