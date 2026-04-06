"""Tests for the CLI entry point and orchestrator."""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest
import respx
from click.testing import CliRunner

from llm_monitor.cli import cli, _resolve_colour
from llm_monitor.core import determine_exit_code, fetch_all
from llm_monitor.cache import ProviderCache
from llm_monitor.models import ProviderStatus, UsageWindow
from llm_monitor.providers.claude import ClaudeProvider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _make_credentials(
    token: str = "sk-ant-oat01-test-token-value",
    expires_at: str | None = None,
) -> dict:
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


def _setup_env(tmp_path: Path, monkeypatch) -> Path:
    """Set up a temporary environment with config, cache, and credentials."""
    # Cache dir
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    monkeypatch.setenv("LLM_MONITOR_CACHE_DIR", str(cache_dir))

    # Credentials
    creds_path = tmp_path / "claude" / ".credentials.json"
    _write_credentials(creds_path)

    # Config (point to tmp credentials)
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        f'[providers.claude]\n'
        f'enabled = true\n'
        f'credentials_path = "{creds_path}"\n'
    )
    os.chmod(str(config_path), 0o600)
    monkeypatch.setenv("LLM_MONITOR_CONFIG", str(config_path))

    return creds_path


# ---------------------------------------------------------------------------
# CLI: --version
# ---------------------------------------------------------------------------


class TestVersion:
    def test_version_flag(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "llm-monitor" in result.output

    def test_short_version_flag(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["-V"])
        assert result.exit_code == 0
        assert "llm-monitor" in result.output


# ---------------------------------------------------------------------------
# CLI: --help
# ---------------------------------------------------------------------------


class TestHelp:
    def test_help_flag(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Monitor LLM service usage" in result.output

    def test_short_help_flag(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["-h"])
        assert result.exit_code == 0
        assert "--now" in result.output


# ---------------------------------------------------------------------------
# CLI: --verbose and --quiet mutual exclusion
# ---------------------------------------------------------------------------


class TestMutualExclusion:
    def test_verbose_and_quiet_together(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["--verbose", "--quiet"])
        assert result.exit_code == 1
        assert "mutually exclusive" in result.stderr.lower()


# ---------------------------------------------------------------------------
# CLI: --clear-cache
# ---------------------------------------------------------------------------


class TestClearCache:
    def test_clear_cache(self, tmp_path, monkeypatch):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        monkeypatch.setenv("LLM_MONITOR_CACHE_DIR", str(cache_dir))
        # Create a dummy cache file
        provider_dir = cache_dir / "claude"
        provider_dir.mkdir()
        (provider_dir / "last.json").write_text("{}")

        runner = CliRunner()
        result = runner.invoke(cli, ["--clear-cache"])
        assert result.exit_code == 0
        assert "cache cleared" in result.stderr.lower()
        assert not cache_dir.exists()


# ---------------------------------------------------------------------------
# CLI: --list-providers
# ---------------------------------------------------------------------------


class TestListProviders:
    def test_list_providers(self, tmp_path, monkeypatch):
        _setup_env(tmp_path, monkeypatch)
        runner = CliRunner()
        result = runner.invoke(cli, ["--list-providers"])
        assert result.exit_code == 0
        assert "claude" in result.stdout.lower()

    def test_list_providers_shows_enabled(self, tmp_path, monkeypatch):
        _setup_env(tmp_path, monkeypatch)
        runner = CliRunner()
        result = runner.invoke(cli, ["--list-providers"])
        assert "enabled" in result.stdout.lower()

    def test_list_providers_shows_configured(self, tmp_path, monkeypatch):
        _setup_env(tmp_path, monkeypatch)
        runner = CliRunner()
        result = runner.invoke(cli, ["--list-providers"])
        assert "configured" in result.stdout.lower()


# ---------------------------------------------------------------------------
# CLI: --provider nonexistent
# ---------------------------------------------------------------------------


class TestProviderFilter:
    def test_unknown_provider(self, tmp_path, monkeypatch):
        _setup_env(tmp_path, monkeypatch)
        runner = CliRunner()
        result = runner.invoke(cli, ["--provider", "nonexistent"])
        assert result.exit_code == 1
        assert "unknown provider" in result.stderr.lower()


# ---------------------------------------------------------------------------
# CLI: default mode (JSON output) with mocked HTTP
# ---------------------------------------------------------------------------


class TestDefaultMode:
    @respx.mock
    def test_default_json_output(self, tmp_path, monkeypatch):
        _setup_env(tmp_path, monkeypatch)

        respx.get("https://api.anthropic.com/api/oauth/usage").respond(
            200, json=_usage_response()
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["--provider", "claude", "--fresh"])
        assert result.exit_code == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
        data = json.loads(result.stdout)
        assert "providers" in data
        assert "version" in data
        assert len(data["providers"]) == 1
        assert data["providers"][0]["provider_name"] == "claude"

    @respx.mock
    def test_json_output_has_windows(self, tmp_path, monkeypatch):
        _setup_env(tmp_path, monkeypatch)

        respx.get("https://api.anthropic.com/api/oauth/usage").respond(
            200, json=_usage_response()
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["--provider", "claude", "--fresh"])
        data = json.loads(result.stdout)
        windows = data["providers"][0]["windows"]
        assert len(windows) == 3
        names = {w["name"] for w in windows}
        assert "Session (5h)" in names

    @respx.mock
    def test_json_output_no_errors(self, tmp_path, monkeypatch):
        _setup_env(tmp_path, monkeypatch)

        respx.get("https://api.anthropic.com/api/oauth/usage").respond(
            200, json=_usage_response()
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["--provider", "claude", "--fresh"])
        data = json.loads(result.stdout)
        assert data["providers"][0]["errors"] == []


# ---------------------------------------------------------------------------
# CLI: --now (table mode) with mocked HTTP
# ---------------------------------------------------------------------------


class TestTableMode:
    @respx.mock
    def test_now_produces_table(self, tmp_path, monkeypatch):
        _setup_env(tmp_path, monkeypatch)

        respx.get("https://api.anthropic.com/api/oauth/usage").respond(
            200, json=_usage_response()
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["--now", "--provider", "claude", "--fresh"])
        assert result.exit_code == 0
        # Table output should contain provider display name
        assert "Anthropic Claude" in result.stdout or "claude" in result.stdout.lower()

    @respx.mock
    def test_now_provider_claude(self, tmp_path, monkeypatch):
        _setup_env(tmp_path, monkeypatch)

        respx.get("https://api.anthropic.com/api/oauth/usage").respond(
            200, json=_usage_response()
        )

        runner = CliRunner()
        result = runner.invoke(
            cli, ["--now", "--provider", "claude", "--fresh"]
        )
        assert result.exit_code == 0
        # Table should contain window names
        assert "Session" in result.stdout or "5h" in result.stdout


# ---------------------------------------------------------------------------
# CLI: --no-colour and --now together
# ---------------------------------------------------------------------------


class TestNoColour:
    @respx.mock
    def test_no_colour_no_ansi(self, tmp_path, monkeypatch):
        _setup_env(tmp_path, monkeypatch)

        respx.get("https://api.anthropic.com/api/oauth/usage").respond(
            200, json=_usage_response()
        )

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["--now", "--provider", "claude", "--fresh", "--no-colour"],
        )
        assert result.exit_code == 0
        # ANSI escape codes start with \x1b[
        assert "\x1b[" not in result.stdout


# ---------------------------------------------------------------------------
# CLI: auth failure → exit code 2
# ---------------------------------------------------------------------------


class TestAuthFailure:
    def test_expired_token_exit_code_2(self, tmp_path, monkeypatch):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        monkeypatch.setenv("LLM_MONITOR_CACHE_DIR", str(cache_dir))

        # Create expired credentials
        past = datetime.now(timezone.utc) - timedelta(hours=1)
        creds_path = tmp_path / "claude" / ".credentials.json"
        _write_credentials(creds_path, _make_credentials(expires_at=past.isoformat()))

        config_path = tmp_path / "config.toml"
        config_path.write_text(
            f'[providers.claude]\n'
            f'enabled = true\n'
            f'credentials_path = "{creds_path}"\n'
        )
        monkeypatch.setenv("LLM_MONITOR_CONFIG", str(config_path))

        runner = CliRunner()
        result = runner.invoke(cli, ["--provider", "claude", "--fresh"])
        assert result.exit_code == 2


# ---------------------------------------------------------------------------
# CLI: stderr has no data, stdout has no messages
# ---------------------------------------------------------------------------


class TestStreamSeparation:
    @respx.mock
    def test_stderr_no_data_stdout_no_messages(self, tmp_path, monkeypatch):
        _setup_env(tmp_path, monkeypatch)

        respx.get("https://api.anthropic.com/api/oauth/usage").respond(
            200, json=_usage_response()
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["--provider", "claude", "--fresh"])
        assert result.exit_code == 0

        # stdout should be valid JSON
        data = json.loads(result.stdout)
        assert "providers" in data

        # stderr should be empty (no warnings/errors on success)
        assert result.stderr.strip() == ""


# ---------------------------------------------------------------------------
# _resolve_colour
# ---------------------------------------------------------------------------


class TestResolveColour:
    def test_no_colour_flag_disables(self):
        assert _resolve_colour(no_colour=True, colour=None) is False

    def test_no_color_env_disables(self, monkeypatch):
        monkeypatch.setenv("NO_COLOR", "1")
        assert _resolve_colour(no_colour=False, colour=None) is False

    def test_term_dumb_disables(self, monkeypatch):
        monkeypatch.delenv("NO_COLOR", raising=False)
        monkeypatch.delenv("LLM_MONITOR_NO_COLOR", raising=False)
        monkeypatch.setenv("TERM", "dumb")
        assert _resolve_colour(no_colour=False, colour=None) is False

    def test_colour_always_enables(self, monkeypatch):
        monkeypatch.delenv("NO_COLOR", raising=False)
        monkeypatch.delenv("LLM_MONITOR_NO_COLOR", raising=False)
        monkeypatch.delenv("TERM", raising=False)
        assert _resolve_colour(no_colour=False, colour="always") is True

    def test_no_colour_overrides_colour_always(self):
        """--no-colour has higher precedence than --colour=always."""
        assert _resolve_colour(no_colour=True, colour="always") is False


# ---------------------------------------------------------------------------
# determine_exit_code
# ---------------------------------------------------------------------------


class TestDetermineExitCode:
    def test_all_success(self):
        now = datetime.now(timezone.utc)
        statuses = [
            ProviderStatus(
                provider_name="claude",
                provider_display="Anthropic Claude",
                timestamp=now, cached=False, cache_age_seconds=0,
                windows=[], errors=[],
            ),
        ]
        assert determine_exit_code(statuses) == 0

    def test_all_auth_errors(self):
        now = datetime.now(timezone.utc)
        statuses = [
            ProviderStatus(
                provider_name="claude",
                provider_display="Anthropic Claude",
                timestamp=now, cached=False, cache_age_seconds=0,
                windows=[], errors=["Authentication failed (401)"],
            ),
        ]
        assert determine_exit_code(statuses) == 2

    def test_partial_failure(self):
        now = datetime.now(timezone.utc)
        statuses = [
            ProviderStatus(
                provider_name="claude",
                provider_display="Anthropic Claude",
                timestamp=now, cached=False, cache_age_seconds=0,
                windows=[], errors=[],
            ),
            ProviderStatus(
                provider_name="openai",
                provider_display="OpenAI",
                timestamp=now, cached=False, cache_age_seconds=0,
                windows=[], errors=["Network error"],
            ),
        ]
        assert determine_exit_code(statuses) == 3

    def test_all_network_errors(self):
        now = datetime.now(timezone.utc)
        statuses = [
            ProviderStatus(
                provider_name="claude",
                provider_display="Anthropic Claude",
                timestamp=now, cached=False, cache_age_seconds=0,
                windows=[], errors=["Network error"],
            ),
        ]
        assert determine_exit_code(statuses) == 4

    def test_empty_statuses(self):
        assert determine_exit_code([]) == 0

    def test_token_expired_is_auth(self):
        now = datetime.now(timezone.utc)
        statuses = [
            ProviderStatus(
                provider_name="claude",
                provider_display="Anthropic Claude",
                timestamp=now, cached=False, cache_age_seconds=0,
                windows=[],
                errors=["Token expired -- run `claude /login` to refresh."],
            ),
        ]
        assert determine_exit_code(statuses) == 2


# ---------------------------------------------------------------------------
# Orchestrator: fetch_all
# ---------------------------------------------------------------------------


class TestFetchAll:
    @respx.mock
    @pytest.mark.asyncio
    async def test_fetch_all_success(self, tmp_path):
        creds_path = tmp_path / ".credentials.json"
        _write_credentials(creds_path)

        config = {
            "general": {"poll_interval": 600},
            "providers": {
                "claude": {"credentials_path": str(creds_path)},
            },
        }
        provider = ClaudeProvider(config)
        cache = ProviderCache(tmp_path / "cache")

        respx.get("https://api.anthropic.com/api/oauth/usage").respond(
            200, json=_usage_response()
        )

        statuses = await fetch_all([provider], cache, config, fresh=True)
        assert len(statuses) == 1
        assert statuses[0].provider_name == "claude"
        assert len(statuses[0].errors) == 0
        assert len(statuses[0].windows) == 3

    @respx.mock
    @pytest.mark.asyncio
    async def test_fetch_all_caches_result(self, tmp_path):
        creds_path = tmp_path / ".credentials.json"
        _write_credentials(creds_path)

        config = {
            "general": {"poll_interval": 600},
            "providers": {
                "claude": {"credentials_path": str(creds_path)},
            },
        }
        provider = ClaudeProvider(config)
        cache = ProviderCache(tmp_path / "cache")

        route = respx.get("https://api.anthropic.com/api/oauth/usage").respond(
            200, json=_usage_response()
        )

        # First fetch (fresh)
        await fetch_all([provider], cache, config, fresh=True)
        assert route.call_count == 1

        # Second fetch (not fresh) — should use cache
        statuses = await fetch_all([provider], cache, config, fresh=False)
        assert route.call_count == 1  # No second HTTP call
        assert statuses[0].cached is True

    @respx.mock
    @pytest.mark.asyncio
    async def test_fetch_all_429_backoff(self, tmp_path):
        creds_path = tmp_path / ".credentials.json"
        _write_credentials(creds_path)

        config = {
            "general": {"poll_interval": 1},  # short TTL so cache expires
            "providers": {
                "claude": {"credentials_path": str(creds_path)},
            },
        }
        provider = ClaudeProvider(config)
        cache = ProviderCache(tmp_path / "cache")

        respx.get("https://api.anthropic.com/api/oauth/usage").respond(429)

        statuses = await fetch_all([provider], cache, config, fresh=True)
        assert len(statuses) == 1
        # Should have errors mentioning rate limit
        errors = " ".join(statuses[0].errors)
        assert "rate" in errors.lower() or "limit" in errors.lower()

    @pytest.mark.asyncio
    async def test_fetch_all_exception_handling(self, tmp_path):
        """If a provider raises an unexpected exception, it's caught."""
        creds_path = tmp_path / "nonexistent.json"
        config = {
            "general": {"poll_interval": 600},
            "providers": {
                "claude": {"credentials_path": str(creds_path)},
            },
        }
        provider = ClaudeProvider(config)
        cache = ProviderCache(tmp_path / "cache")

        # fetch_usage won't raise — it returns errors in status
        # But the orchestrator still handles it
        statuses = await fetch_all([provider], cache, config, fresh=True)
        assert len(statuses) == 1
        # Should have an error about credentials
        assert len(statuses[0].errors) > 0


# ---------------------------------------------------------------------------
# CLI: --no-history is accepted (no-op placeholder)
# ---------------------------------------------------------------------------


class TestNoHistory:
    def test_no_history_accepted(self, tmp_path, monkeypatch):
        """--no-history is accepted without error."""
        _setup_env(tmp_path, monkeypatch)
        runner = CliRunner()
        # Just check it doesn't fail — it's a no-op
        result = runner.invoke(cli, ["--no-history", "--version"])
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# CLI: --fresh flag
# ---------------------------------------------------------------------------


class TestFreshFlag:
    @respx.mock
    def test_fresh_bypasses_cache(self, tmp_path, monkeypatch):
        _setup_env(tmp_path, monkeypatch)

        route = respx.get("https://api.anthropic.com/api/oauth/usage").respond(
            200, json=_usage_response()
        )

        runner = CliRunner()
        # First call (populates cache)
        result1 = runner.invoke(cli, ["--provider", "claude", "--fresh"])
        assert result1.exit_code == 0
        assert route.call_count == 1

        # Second call with --fresh (should hit API again)
        result2 = runner.invoke(cli, ["--provider", "claude", "--fresh"])
        assert result2.exit_code == 0
        assert route.call_count == 2
