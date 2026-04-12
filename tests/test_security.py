"""Tests for security utilities."""

from __future__ import annotations

import os
import stat
import sys

import pytest

from clawmeter.models import CredentialError, SecretStr
from clawmeter.security import (
    REDACTION_PATTERNS,
    check_file_permissions,
    is_container_mode,
    run_key_command,
    sanitise_text,
    secure_mkdir,
    secure_write,
)


# ---------------------------------------------------------------------------
# REDACTION_PATTERNS
# ---------------------------------------------------------------------------

class TestRedactionPatterns:
    """Each compiled pattern must match its intended target."""

    @pytest.mark.parametrize(
        "token",
        [
            "sk-ant-oat01-abc123-real-token-value",
            "sk-ant-ort01-refresh-token-value-here",
            "sk-ant-api03-key-abcdefghijklmnopqrst",
            "sk-ant-admin01-key-abcdefghijklmn",
            "sk-proj-abcdefghij1234567890",
            "xai-abcdefghij1234567890xx",
            "Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9",
        ],
    )
    def test_at_least_one_pattern_matches(self, token: str):
        matched = any(p.search(token) for p in REDACTION_PATTERNS)
        assert matched, f"No pattern matched token: {token}"

    def test_claude_oauth_access_token_pattern(self):
        assert REDACTION_PATTERNS[0].search("sk-ant-oat01-abc123def456")

    def test_claude_oauth_refresh_token_pattern(self):
        assert REDACTION_PATTERNS[1].search("sk-ant-ort01-abc123def456")

    def test_anthropic_api_key_pattern(self):
        assert REDACTION_PATTERNS[2].search("sk-ant-api03-abc123def456")

    def test_anthropic_admin_key_pattern(self):
        assert REDACTION_PATTERNS[3].search("sk-ant-admin01-abc123def456")

    def test_openai_api_key_pattern(self):
        assert REDACTION_PATTERNS[4].search("sk-proj-abcdefghij1234567890")

    def test_xai_api_key_pattern(self):
        assert REDACTION_PATTERNS[5].search("xai-abcdefghij1234567890xx")

    def test_bearer_token_pattern(self):
        assert REDACTION_PATTERNS[6].search("Bearer some-jwt-token-value")


# ---------------------------------------------------------------------------
# sanitise_text
# ---------------------------------------------------------------------------

class TestSanitiseText:
    def test_replaces_claude_oauth_token(self):
        text = "token is sk-ant-oat01-super-secret-stuff here"
        result = sanitise_text(text)
        assert "sk-ant-oat01" not in result
        assert "***REDACTED***" in result

    def test_replaces_openai_key(self):
        text = "key: sk-proj-abcdefghij1234567890"
        result = sanitise_text(text)
        assert "sk-proj" not in result
        assert "***REDACTED***" in result

    def test_replaces_xai_key(self):
        text = "xai key: xai-abcdefghij1234567890xx"
        result = sanitise_text(text)
        assert "xai-abcdefghij" not in result
        assert "***REDACTED***" in result

    def test_replaces_bearer_token(self):
        text = "Authorization: Bearer eyJhbGciOiJSUzI1NiJ9.payload.sig"
        result = sanitise_text(text)
        assert "eyJhbGci" not in result
        assert "***REDACTED***" in result

    def test_replaces_multiple_tokens(self):
        text = (
            "access=sk-ant-oat01-aaa refresh=sk-ant-ort01-bbb "
            "api=sk-ant-api03-ccc"
        )
        result = sanitise_text(text)
        assert result.count("***REDACTED***") == 3

    def test_normal_text_unchanged(self):
        text = "This is a perfectly normal log line with no secrets."
        assert sanitise_text(text) == text

    def test_short_sk_prefix_not_matched(self):
        """A bare 'sk-' followed by fewer than 20 chars should not match."""
        text = "sk-short is fine"
        assert sanitise_text(text) == text

    def test_preserves_surrounding_text(self):
        text = "before sk-ant-oat01-secret-value-here after"
        result = sanitise_text(text)
        assert result.startswith("before ")
        assert result.endswith(" after")


# ---------------------------------------------------------------------------
# secure_write
# ---------------------------------------------------------------------------

class TestSecureWrite:
    def test_creates_file_with_0600(self, tmp_path):
        target = str(tmp_path / "secret.txt")
        secure_write(target, "hello")
        mode = os.stat(target).st_mode & 0o777
        assert mode == 0o600

    def test_file_content_written(self, tmp_path):
        target = str(tmp_path / "data.txt")
        secure_write(target, "payload")
        assert open(target).read() == "payload"

    def test_atomic_no_tmp_left_on_success(self, tmp_path):
        target = str(tmp_path / "clean.txt")
        secure_write(target, "ok")
        assert not os.path.exists(target + ".tmp")

    def test_creates_parent_directories(self, tmp_path):
        target = str(tmp_path / "a" / "b" / "c" / "file.txt")
        secure_write(target, "deep")
        assert os.path.isfile(target)
        assert open(target).read() == "deep"

    def test_overwrites_existing_file(self, tmp_path):
        target = str(tmp_path / "overwrite.txt")
        secure_write(target, "first")
        secure_write(target, "second")
        assert open(target).read() == "second"


# ---------------------------------------------------------------------------
# secure_mkdir
# ---------------------------------------------------------------------------

class TestSecureMkdir:
    def test_creates_dir_with_0700(self, tmp_path):
        target = str(tmp_path / "secure_dir")
        secure_mkdir(target)
        mode = os.stat(target).st_mode & 0o777
        assert mode == 0o700

    def test_exist_ok(self, tmp_path):
        target = str(tmp_path / "existing")
        secure_mkdir(target)
        secure_mkdir(target)  # should not raise
        assert os.path.isdir(target)

    def test_creates_parents(self, tmp_path):
        target = str(tmp_path / "x" / "y" / "z")
        secure_mkdir(target)
        assert os.path.isdir(target)


# ---------------------------------------------------------------------------
# check_file_permissions
# ---------------------------------------------------------------------------

class TestCheckFilePermissions:
    def test_detects_644_as_too_permissive(self, tmp_path):
        target = tmp_path / "loose.txt"
        target.write_text("data")
        os.chmod(str(target), 0o644)
        warnings = check_file_permissions(str(target))
        assert len(warnings) == 1
        assert "loose permissions" in warnings[0]

    def test_accepts_0600(self, tmp_path):
        target = tmp_path / "tight.txt"
        target.write_text("data")
        os.chmod(str(target), 0o600)
        warnings = check_file_permissions(str(target))
        assert warnings == []

    def test_nonexistent_file_returns_empty(self, tmp_path):
        warnings = check_file_permissions(str(tmp_path / "nope.txt"))
        assert warnings == []

    def test_detects_world_readable(self, tmp_path):
        target = tmp_path / "world.txt"
        target.write_text("data")
        os.chmod(str(target), 0o604)
        warnings = check_file_permissions(str(target))
        assert len(warnings) == 1

    def test_accepts_0400(self, tmp_path):
        """Read-only for owner is acceptable (no extra bits)."""
        target = tmp_path / "readonly.txt"
        target.write_text("data")
        os.chmod(str(target), 0o400)
        warnings = check_file_permissions(str(target))
        assert warnings == []


# ---------------------------------------------------------------------------
# is_container_mode
# ---------------------------------------------------------------------------

class TestIsContainerMode:
    def test_detects_env_var(self, monkeypatch):
        monkeypatch.setenv("CLAWMETER_CONTAINER", "1")
        assert is_container_mode() is True

    def test_env_var_not_set(self, monkeypatch, tmp_path):
        monkeypatch.delenv("CLAWMETER_CONTAINER", raising=False)
        # Also ensure /.dockerenv doesn't affect the test on non-container hosts
        # by patching os.path.exists for the dockerenv check.
        original_exists = os.path.exists

        def fake_exists(path):
            if path == "/.dockerenv":
                return False
            return original_exists(path)

        monkeypatch.setattr(os.path, "exists", fake_exists)
        assert is_container_mode() is False

    def test_detects_dockerenv_file(self, monkeypatch):
        monkeypatch.delenv("CLAWMETER_CONTAINER", raising=False)
        original_exists = os.path.exists

        def fake_exists(path):
            if path == "/.dockerenv":
                return True
            return original_exists(path)

        monkeypatch.setattr(os.path, "exists", fake_exists)
        assert is_container_mode() is True


# ---------------------------------------------------------------------------
# run_key_command
# ---------------------------------------------------------------------------

class TestRunKeyCommand:
    def test_returns_secret_str_on_success(self):
        result = run_key_command("echo my-secret-key")
        assert isinstance(result, SecretStr)
        assert result.get_secret_value() == "my-secret-key"

    def test_strips_whitespace(self):
        result = run_key_command("echo '  padded  '")
        assert result.get_secret_value() == "padded"

    def test_raises_credential_error_on_nonzero_exit(self):
        with pytest.raises(CredentialError, match="key_command failed"):
            run_key_command("false")

    def test_raises_credential_error_on_timeout(self):
        with pytest.raises(CredentialError, match="timed out"):
            run_key_command(f"{sys.executable} -c \"import time; time.sleep(30)\"")

    def test_raises_credential_error_on_empty_output(self):
        with pytest.raises(CredentialError, match="no output"):
            run_key_command("echo ''")

    def test_str_repr_does_not_leak_secret(self):
        result = run_key_command("echo super-secret")
        assert "super-secret" not in str(result)
        assert "super-secret" not in repr(result)

    def test_shell_injection_prevented(self):
        """Shell metacharacters are passed literally, not interpreted."""
        # With shell=False, shlex.split turns this into
        # ["echo", "ok;", "echo", "injected"] -- echo prints them all
        # literally, including the semicolon.  The semicolon is NOT
        # treated as a command separator.
        result = run_key_command("echo 'ok; echo injected'")
        assert result.get_secret_value() == "ok; echo injected"
