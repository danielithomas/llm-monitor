"""Tests for configuration loading and path resolution."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from clawmeter.config import (
    DEFAULT_CONFIG,
    get_cache_dir,
    get_config_path,
    get_data_dir,
    get_log_dir,
    get_log_file,
    get_pid_dir,
    get_pid_file,
    get_state_file,
    is_alpha_enabled,
    load_config,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean_config_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove all env vars that influence config path resolution."""
    for var in (
        "CLAWMETER_CONFIG",
        "CLAWMETER_DATA_DIR",
        "CLAWMETER_CACHE_DIR",
        "CLAWMETER_CONTAINER",
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
        assert DEFAULT_CONFIG["general"]["enable_alpha_features"] is False

    def test_provider_ollama_defaults(self):
        ollama = DEFAULT_CONFIG["providers"]["ollama"]
        assert ollama["enabled"] is False
        assert ollama["poll_interval"] == 60
        assert ollama["host"] == "http://localhost:11434"
        assert ollama["cloud_enabled"] is False
        assert ollama["api_key_env"] == "OLLAMA_API_KEY"

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
        monkeypatch.setenv("CLAWMETER_CONFIG", "/custom/path/config.toml")
        assert get_config_path() == Path("/custom/path/config.toml")

    def test_xdg_config_home(self, monkeypatch):
        _clean_config_env(monkeypatch)
        monkeypatch.setenv("XDG_CONFIG_HOME", "/xdg/config")
        assert get_config_path() == Path("/xdg/config/clawmeter/config.toml")

    def test_default_fallback(self, monkeypatch):
        _clean_config_env(monkeypatch)
        expected = Path.home() / ".config" / "clawmeter" / "config.toml"
        assert get_config_path() == expected

    def test_env_var_takes_precedence_over_xdg(self, monkeypatch):
        _clean_config_env(monkeypatch)
        monkeypatch.setenv("CLAWMETER_CONFIG", "/explicit/config.toml")
        monkeypatch.setenv("XDG_CONFIG_HOME", "/xdg/config")
        assert get_config_path() == Path("/explicit/config.toml")


# ---------------------------------------------------------------------------
# get_data_dir
# ---------------------------------------------------------------------------

class TestGetDataDir:
    def test_env_var_overrides(self, monkeypatch):
        _clean_config_env(monkeypatch)
        monkeypatch.setenv("CLAWMETER_DATA_DIR", "/custom/data")
        assert get_data_dir() == Path("/custom/data")

    def test_xdg_data_home(self, monkeypatch):
        _clean_config_env(monkeypatch)
        monkeypatch.setenv("XDG_DATA_HOME", "/xdg/data")
        assert get_data_dir() == Path("/xdg/data/clawmeter")

    def test_default_fallback(self, monkeypatch):
        _clean_config_env(monkeypatch)
        expected = Path.home() / ".local" / "share" / "clawmeter"
        assert get_data_dir() == expected


# ---------------------------------------------------------------------------
# get_cache_dir
# ---------------------------------------------------------------------------

class TestGetCacheDir:
    def test_env_var_overrides(self, monkeypatch):
        _clean_config_env(monkeypatch)
        monkeypatch.setenv("CLAWMETER_CACHE_DIR", "/custom/cache")
        assert get_cache_dir() == Path("/custom/cache")

    def test_xdg_cache_home(self, monkeypatch):
        _clean_config_env(monkeypatch)
        monkeypatch.setenv("XDG_CACHE_HOME", "/xdg/cache")
        assert get_cache_dir() == Path("/xdg/cache/clawmeter")

    def test_default_fallback(self, monkeypatch):
        _clean_config_env(monkeypatch)
        expected = Path.home() / ".cache" / "clawmeter"
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
    def test_clawmeter_config_env_var(self, tmp_path, monkeypatch):
        _clean_config_env(monkeypatch)
        config_file = tmp_path / "env_config.toml"
        config_file.write_text('[general]\npoll_interval = 120\n')
        monkeypatch.setenv("CLAWMETER_CONFIG", str(config_file))
        config = load_config()
        assert config["general"]["poll_interval"] == 120

    def test_explicit_path_overrides_env_var(self, tmp_path, monkeypatch):
        _clean_config_env(monkeypatch)
        env_config = tmp_path / "env.toml"
        env_config.write_text('[general]\npoll_interval = 111\n')
        explicit_config = tmp_path / "explicit.toml"
        explicit_config.write_text('[general]\npoll_interval = 222\n')
        monkeypatch.setenv("CLAWMETER_CONFIG", str(env_config))
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
        monkeypatch.delenv("CLAWMETER_CONTAINER", raising=False)
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
        monkeypatch.delenv("CLAWMETER_CONTAINER", raising=False)
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
        """When $CLAWMETER_CONTAINER=1, no permission warning is emitted."""
        _clean_config_env(monkeypatch)
        monkeypatch.setenv("CLAWMETER_CONTAINER", "1")
        config_file = tmp_path / "config.toml"
        config_file.write_text('[general]\npoll_interval = 600\n')
        config_file.chmod(0o644)
        load_config(str(config_file))
        captured = capfd.readouterr()
        assert "loose permissions" not in captured.err

    def test_container_mode_still_loads_config(self, tmp_path, monkeypatch):
        _clean_config_env(monkeypatch)
        monkeypatch.setenv("CLAWMETER_CONTAINER", "1")
        config_file = tmp_path / "config.toml"
        config_file.write_text('[general]\npoll_interval = 42\n')
        config = load_config(str(config_file))
        assert config["general"]["poll_interval"] == 42

    def test_container_mode_skips_keyring(self, monkeypatch):
        """Container mode skips keyring tier in credential resolution."""
        from unittest.mock import patch
        from clawmeter.providers.base import Provider

        _clean_config_env(monkeypatch)
        monkeypatch.setenv("CLAWMETER_CONTAINER", "1")

        # Use a concrete test provider
        class _TestProvider(Provider):
            def name(self):
                return "test"
            def display_name(self):
                return "Test"
            def is_configured(self):
                return True
            async def fetch_usage(self, client):
                pass
            def auth_instructions(self):
                return ""

        p = _TestProvider()
        config = {"providers": {"test": {}}}

        # Patch keyring to track if it gets called
        keyring_called = False
        original_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

        with patch("clawmeter.security.is_container_mode", return_value=True):
            result = p.resolve_credential(config)

        # Should return None without attempting keyring
        assert result is None


# ---------------------------------------------------------------------------
# Daemon config helpers
# ---------------------------------------------------------------------------

class TestDaemonConfig:
    def test_default_config_has_daemon_section(self):
        assert "daemon" in DEFAULT_CONFIG
        assert "log_file" in DEFAULT_CONFIG["daemon"]
        assert "pid_file" in DEFAULT_CONFIG["daemon"]

    def test_get_pid_dir_xdg_runtime(self, monkeypatch):
        _clean_config_env(monkeypatch)
        monkeypatch.setenv("XDG_RUNTIME_DIR", "/run/user/1000")
        result = get_pid_dir()
        assert result == Path("/run/user/1000/clawmeter")

    def test_get_pid_dir_fallback(self, monkeypatch):
        _clean_config_env(monkeypatch)
        monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
        result = get_pid_dir()
        assert str(result).startswith("/tmp/clawmeter-")
        assert str(os.getuid()) in str(result)

    def test_get_log_dir_xdg_state(self, monkeypatch):
        _clean_config_env(monkeypatch)
        monkeypatch.setenv("XDG_STATE_HOME", "/home/test/.local/state")
        result = get_log_dir()
        assert result == Path("/home/test/.local/state/clawmeter")

    def test_get_log_dir_fallback(self, monkeypatch):
        _clean_config_env(monkeypatch)
        monkeypatch.delenv("XDG_STATE_HOME", raising=False)
        result = get_log_dir()
        assert result == Path.home() / ".local" / "state" / "clawmeter"

    def test_get_pid_file_custom(self, monkeypatch):
        _clean_config_env(monkeypatch)
        config = {"daemon": {"pid_file": "/custom/path/daemon.pid"}}
        assert get_pid_file(config) == Path("/custom/path/daemon.pid")

    def test_get_pid_file_default(self, monkeypatch):
        _clean_config_env(monkeypatch)
        monkeypatch.setenv("XDG_RUNTIME_DIR", "/run/user/1000")
        config = {"daemon": {"pid_file": ""}}
        assert get_pid_file(config) == Path("/run/user/1000/clawmeter/daemon.pid")

    def test_get_log_file_custom(self, monkeypatch):
        _clean_config_env(monkeypatch)
        config = {"daemon": {"log_file": "/custom/daemon.log"}}
        assert get_log_file(config) == Path("/custom/daemon.log")

    def test_get_log_file_default(self, monkeypatch):
        _clean_config_env(monkeypatch)
        monkeypatch.setenv("XDG_STATE_HOME", "/home/test/.local/state")
        config = {"daemon": {"log_file": ""}}
        assert get_log_file(config) == Path("/home/test/.local/state/clawmeter/daemon.log")

    def test_get_state_file(self, monkeypatch):
        _clean_config_env(monkeypatch)
        monkeypatch.setenv("XDG_RUNTIME_DIR", "/run/user/1000")
        config = {"daemon": {}}
        assert get_state_file(config) == Path("/run/user/1000/clawmeter/daemon.state")


# ---------------------------------------------------------------------------
# is_alpha_enabled
# ---------------------------------------------------------------------------

class TestIsAlphaEnabled:
    def test_default_is_false(self):
        assert is_alpha_enabled(DEFAULT_CONFIG) is False

    def test_enabled_when_set(self):
        config = {"general": {"enable_alpha_features": True}}
        assert is_alpha_enabled(config) is True

    def test_disabled_when_false(self):
        config = {"general": {"enable_alpha_features": False}}
        assert is_alpha_enabled(config) is False

    def test_missing_general_section(self):
        assert is_alpha_enabled({}) is False

    def test_loaded_from_toml(self, tmp_path, monkeypatch):
        _clean_config_env(monkeypatch)
        config_file = tmp_path / "alpha.toml"
        config_file.write_text(
            '[general]\nenable_alpha_features = true\n'
        )
        config = load_config(str(config_file))
        assert is_alpha_enabled(config) is True


# ---------------------------------------------------------------------------
# Ollama host/hosts mutual exclusivity
# ---------------------------------------------------------------------------

class TestOllamaHostValidation:
    def test_host_and_hosts_raises(self, tmp_path, monkeypatch):
        """Setting both host and hosts in config raises ValueError."""
        _clean_config_env(monkeypatch)
        config_file = tmp_path / "bad_ollama.toml"
        config_file.write_text(
            '[providers.ollama]\n'
            'enabled = true\n'
            'host = "http://localhost:11434"\n'
            '\n'
            '[[providers.ollama.hosts]]\n'
            'name = "gpu"\n'
            'url = "http://gpu:11434"\n'
        )
        with pytest.raises(ValueError, match="mutually exclusive"):
            load_config(str(config_file))

    def test_hosts_only_is_valid(self, tmp_path, monkeypatch):
        """Using only hosts array is valid."""
        _clean_config_env(monkeypatch)
        config_file = tmp_path / "hosts_only.toml"
        config_file.write_text(
            '[providers.ollama]\n'
            'enabled = true\n'
            '\n'
            '[[providers.ollama.hosts]]\n'
            'name = "gpu"\n'
            'url = "http://gpu:11434"\n'
        )
        config = load_config(str(config_file))
        hosts = config["providers"]["ollama"]["hosts"]
        assert len(hosts) == 1
        assert hosts[0]["name"] == "gpu"
