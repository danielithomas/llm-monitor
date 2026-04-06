"""Tests for configuration loading and path resolution."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from llm_monitor.config import (
    DEFAULT_CONFIG,
    get_cache_dir,
    get_config_path,
    get_data_dir,
    load_config,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean_config_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove all env vars that influence config path resolution."""
    for var in (
        "LLM_MONITOR_CONFIG",
        "LLM_MONITOR_DATA_DIR",
        "LLM_MONITOR_CACHE_DIR",
        "LLM_MONITOR_CONTAINER",
        "XDG_CONFIG_HOME",
        "XDG_DATA_HOME",
        "XDG_CACHE_HOME",
    ):
        monkeypatch.delenv(var, raising=False)


# ---------------------------------------------------------------------------
# DEFAULT_CONFIG
# ---------------------------------------------------------------------------

class TestDefaultConfig:
    def test_general_defaults(self):
        assert DEFAULT_CONFIG["general"]["default_providers"] == ["claude"]
        assert DEFAULT_CONFIG["general"]["poll_interval"] == 600
        assert DEFAULT_CONFIG["general"]["notification_enabled"] is False

    def test_threshold_defaults(self):
        assert DEFAULT_CONFIG["thresholds"]["warning"] == 70
        assert DEFAULT_CONFIG["thresholds"]["critical"] == 90

    def test_provider_claude_defaults(self):
        claude = DEFAULT_CONFIG["providers"]["claude"]
        assert claude["enabled"] is True
        assert claude["credentials_path"] == ""
        assert claude["show_opus"] is True

    def test_history_defaults(self):
        assert DEFAULT_CONFIG["history"]["enabled"] is True
        assert DEFAULT_CONFIG["history"]["retention_days"] == 90


# ---------------------------------------------------------------------------
# get_config_path
# ---------------------------------------------------------------------------

class TestGetConfigPath:
    def test_env_var_overrides(self, monkeypatch):
        _clean_config_env(monkeypatch)
        monkeypatch.setenv("LLM_MONITOR_CONFIG", "/custom/path/config.toml")
        assert get_config_path() == Path("/custom/path/config.toml")

    def test_xdg_config_home(self, monkeypatch):
        _clean_config_env(monkeypatch)
        monkeypatch.setenv("XDG_CONFIG_HOME", "/xdg/config")
        assert get_config_path() == Path("/xdg/config/llm-monitor/config.toml")

    def test_default_fallback(self, monkeypatch):
        _clean_config_env(monkeypatch)
        expected = Path.home() / ".config" / "llm-monitor" / "config.toml"
        assert get_config_path() == expected

    def test_env_var_takes_precedence_over_xdg(self, monkeypatch):
        _clean_config_env(monkeypatch)
        monkeypatch.setenv("LLM_MONITOR_CONFIG", "/explicit/config.toml")
        monkeypatch.setenv("XDG_CONFIG_HOME", "/xdg/config")
        assert get_config_path() == Path("/explicit/config.toml")


# ---------------------------------------------------------------------------
# get_data_dir
# ---------------------------------------------------------------------------

class TestGetDataDir:
    def test_env_var_overrides(self, monkeypatch):
        _clean_config_env(monkeypatch)
        monkeypatch.setenv("LLM_MONITOR_DATA_DIR", "/custom/data")
        assert get_data_dir() == Path("/custom/data")

    def test_xdg_data_home(self, monkeypatch):
        _clean_config_env(monkeypatch)
        monkeypatch.setenv("XDG_DATA_HOME", "/xdg/data")
        assert get_data_dir() == Path("/xdg/data/llm-monitor")

    def test_default_fallback(self, monkeypatch):
        _clean_config_env(monkeypatch)
        expected = Path.home() / ".local" / "share" / "llm-monitor"
        assert get_data_dir() == expected


# ---------------------------------------------------------------------------
# get_cache_dir
# ---------------------------------------------------------------------------

class TestGetCacheDir:
    def test_env_var_overrides(self, monkeypatch):
        _clean_config_env(monkeypatch)
        monkeypatch.setenv("LLM_MONITOR_CACHE_DIR", "/custom/cache")
        assert get_cache_dir() == Path("/custom/cache")

    def test_xdg_cache_home(self, monkeypatch):
        _clean_config_env(monkeypatch)
        monkeypatch.setenv("XDG_CACHE_HOME", "/xdg/cache")
        assert get_cache_dir() == Path("/xdg/cache/llm-monitor")

    def test_default_fallback(self, monkeypatch):
        _clean_config_env(monkeypatch)
        expected = Path.home() / ".cache" / "llm-monitor"
        assert get_cache_dir() == expected


# ---------------------------------------------------------------------------
# load_config — valid TOML parsing
# ---------------------------------------------------------------------------

class TestLoadConfigValid:
    def test_full_fixture(self, monkeypatch):
        _clean_config_env(monkeypatch)
        config = load_config(str(FIXTURES_DIR / "config_full.toml"))
        assert config["general"]["poll_interval"] == 600
        assert config["providers"]["claude"]["enabled"] is True
        assert config["providers"]["grok"]["enabled"] is False
        assert config["history"]["retention_days"] == 90

    def test_partial_config_merges_with_defaults(self, tmp_path, monkeypatch):
        _clean_config_env(monkeypatch)
        partial = tmp_path / "partial.toml"
        partial.write_text('[general]\npoll_interval = 300\n')
        config = load_config(str(partial))
        # Overridden value
        assert config["general"]["poll_interval"] == 300
        # Defaults still present
        assert config["general"]["default_providers"] == ["claude"]
        assert config["thresholds"]["warning"] == 70
        assert config["providers"]["claude"]["enabled"] is True
        assert config["history"]["enabled"] is True

    def test_partial_provider_does_not_clobber_defaults(self, tmp_path, monkeypatch):
        _clean_config_env(monkeypatch)
        partial = tmp_path / "partial.toml"
        partial.write_text('[providers.claude]\nshow_opus = false\n')
        config = load_config(str(partial))
        # Overridden
        assert config["providers"]["claude"]["show_opus"] is False
        # Defaults still present within same provider section
        assert config["providers"]["claude"]["enabled"] is True
        assert config["providers"]["claude"]["credentials_path"] == ""


# ---------------------------------------------------------------------------
# load_config — missing config file returns defaults
# ---------------------------------------------------------------------------

class TestLoadConfigMissing:
    def test_nonexistent_path_returns_defaults(self, monkeypatch):
        _clean_config_env(monkeypatch)
        config = load_config("/nonexistent/path/config.toml")
        assert config == DEFAULT_CONFIG

    def test_returned_dict_is_independent_copy(self, monkeypatch):
        _clean_config_env(monkeypatch)
        config = load_config("/nonexistent/path/config.toml")
        config["general"]["poll_interval"] = 9999
        assert DEFAULT_CONFIG["general"]["poll_interval"] == 600

    def test_default_path_missing_no_error(self, tmp_path, monkeypatch):
        """When the default config path doesn't exist, return defaults silently."""
        _clean_config_env(monkeypatch)
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "nonexistent_xdg"))
        config = load_config()
        assert config == DEFAULT_CONFIG


# ---------------------------------------------------------------------------
# load_config — env var override for path
# ---------------------------------------------------------------------------

class TestLoadConfigEnvVar:
    def test_llm_monitor_config_env_var(self, tmp_path, monkeypatch):
        _clean_config_env(monkeypatch)
        config_file = tmp_path / "env_config.toml"
        config_file.write_text('[general]\npoll_interval = 120\n')
        monkeypatch.setenv("LLM_MONITOR_CONFIG", str(config_file))
        config = load_config()
        assert config["general"]["poll_interval"] == 120

    def test_explicit_path_overrides_env_var(self, tmp_path, monkeypatch):
        _clean_config_env(monkeypatch)
        env_config = tmp_path / "env.toml"
        env_config.write_text('[general]\npoll_interval = 111\n')
        explicit_config = tmp_path / "explicit.toml"
        explicit_config.write_text('[general]\npoll_interval = 222\n')
        monkeypatch.setenv("LLM_MONITOR_CONFIG", str(env_config))
        config = load_config(str(explicit_config))
        assert config["general"]["poll_interval"] == 222


# ---------------------------------------------------------------------------
# load_config — malformed TOML
# ---------------------------------------------------------------------------

class TestLoadConfigMalformed:
    def test_malformed_toml_raises_value_error(self, tmp_path, monkeypatch):
        _clean_config_env(monkeypatch)
        bad_file = tmp_path / "bad.toml"
        bad_file.write_text("this is not [valid toml = {\n")
        with pytest.raises(ValueError, match="Malformed TOML"):
            load_config(str(bad_file))

    def test_error_message_includes_path(self, tmp_path, monkeypatch):
        _clean_config_env(monkeypatch)
        bad_file = tmp_path / "bad2.toml"
        bad_file.write_text("[broken\n")
        with pytest.raises(ValueError, match=str(bad_file)):
            load_config(str(bad_file))


# ---------------------------------------------------------------------------
# load_config — permission warnings
# ---------------------------------------------------------------------------

class TestLoadConfigPermissions:
    def test_loose_permissions_emit_warning(self, tmp_path, monkeypatch, capfd):
        """A config file with 0o644 should produce a warning on stderr."""
        _clean_config_env(monkeypatch)
        monkeypatch.delenv("LLM_MONITOR_CONTAINER", raising=False)
        config_file = tmp_path / "config.toml"
        config_file.write_text('[general]\npoll_interval = 600\n')
        config_file.chmod(0o644)
        load_config(str(config_file))
        captured = capfd.readouterr()
        assert "loose permissions" in captured.err
        assert "chmod 600" in captured.err

    def test_secure_permissions_no_warning(self, tmp_path, monkeypatch, capfd):
        """A config file with 0o600 should not produce any warning."""
        _clean_config_env(monkeypatch)
        monkeypatch.delenv("LLM_MONITOR_CONTAINER", raising=False)
        config_file = tmp_path / "config.toml"
        config_file.write_text('[general]\npoll_interval = 600\n')
        config_file.chmod(0o600)
        load_config(str(config_file))
        captured = capfd.readouterr()
        assert "loose permissions" not in captured.err


# ---------------------------------------------------------------------------
# load_config — container mode skips permission check
# ---------------------------------------------------------------------------

class TestLoadConfigContainerMode:
    def test_container_mode_skips_permission_warning(
        self, tmp_path, monkeypatch, capfd
    ):
        """When $LLM_MONITOR_CONTAINER=1, no permission warning is emitted."""
        _clean_config_env(monkeypatch)
        monkeypatch.setenv("LLM_MONITOR_CONTAINER", "1")
        config_file = tmp_path / "config.toml"
        config_file.write_text('[general]\npoll_interval = 600\n')
        config_file.chmod(0o644)
        load_config(str(config_file))
        captured = capfd.readouterr()
        assert "loose permissions" not in captured.err

    def test_container_mode_still_loads_config(self, tmp_path, monkeypatch):
        _clean_config_env(monkeypatch)
        monkeypatch.setenv("LLM_MONITOR_CONTAINER", "1")
        config_file = tmp_path / "config.toml"
        config_file.write_text('[general]\npoll_interval = 42\n')
        config = load_config(str(config_file))
        assert config["general"]["poll_interval"] == 42
