"""Security utilities for clawmeter.

Credential sanitisation, secure file I/O, and process security helpers.
See spec Sections 7.3, 7.4, 7.6.
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
from typing import List

from clawmeter.models import CredentialError, SecretStr

# ---------------------------------------------------------------------------
# 7.3 Credential Sanitisation - compiled redaction patterns
# ---------------------------------------------------------------------------

REDACTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"sk-ant-oat\S+"),          # Claude OAuth access token
    re.compile(r"sk-ant-ort\S+"),          # Claude OAuth refresh token
    re.compile(r"sk-ant-api\S+"),          # Anthropic API key
    re.compile(r"sk-ant-admin\S+"),        # Anthropic Admin API key
    re.compile(r"sk-[a-zA-Z0-9-]{20,}"),   # OpenAI API key
    re.compile(r"xai-[a-zA-Z0-9-]{20,}"),  # xAI API key / management key
    re.compile(r"Bearer\s+\S+"),           # Any bearer token in logs
]


def sanitise_text(text: str) -> str:
    """Apply all redaction patterns, replacing matches with ``***REDACTED***``."""
    for pattern in REDACTION_PATTERNS:
        text = pattern.sub("***REDACTED***", text)
    return text


# ---------------------------------------------------------------------------
# 7.4 File Security
# ---------------------------------------------------------------------------

def secure_write(path: str, data: str) -> None:
    """Write *data* to *path* with ``0o600`` permissions, atomically.

    Writes to a temporary file (``path + ".tmp"``) then renames into place.
    The temporary file is cleaned up on failure.  Parent directories are
    created (``0o700``) if they do not already exist.
    """
    parent = os.path.dirname(path)
    if parent:
        secure_mkdir(parent)

    tmp_path = path + ".tmp"
    fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(data)
        os.rename(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def secure_mkdir(path: str) -> None:
    """Create directory (and parents) with ``0o700`` permissions."""
    os.makedirs(path, mode=0o700, exist_ok=True)


def check_file_permissions(path: str) -> List[str]:
    """Return a list of warning strings if *path* is more permissive than ``0o600``.

    Returns an empty list when the file does not exist or permissions are
    acceptable.
    """
    if not os.path.exists(path):
        return []

    mode = os.stat(path).st_mode & 0o777
    warnings: list[str] = []
    if mode & 0o177:  # any bits beyond owner-rw (0o600)
        warnings.append(
            f"File has loose permissions ({oct(mode)}): {path}. "
            f"Other users on this system could read this file. "
            f"Fix: chmod 600 {path}"
        )
    return warnings


# ---------------------------------------------------------------------------
# Container detection
# ---------------------------------------------------------------------------

def is_container_mode() -> bool:
    """Return ``True`` when running inside a container.

    Checks the ``$LLM_MONITOR_CONTAINER`` environment variable (value ``1``)
    and the presence of ``/.dockerenv``.
    """
    if os.environ.get("LLM_MONITOR_CONTAINER") == "1":
        return True
    return os.path.exists("/.dockerenv")


# ---------------------------------------------------------------------------
# 7.6 Process Security - key_command execution
# ---------------------------------------------------------------------------

def run_key_command(command: str) -> SecretStr:
    """Execute a key command securely and return the key as ``SecretStr``.

    Uses ``shlex.split()`` and ``shell=False`` to prevent shell injection.
    On failure only stderr is logged (never stdout, which contains the secret).

    Raises:
        CredentialError: On non-zero exit, timeout, or empty output.
    """
    args = shlex.split(command)
    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=10,
            shell=False,
        )
    except subprocess.TimeoutExpired:
        raise CredentialError(f"key_command timed out after 10s: {command}")

    if result.returncode != 0:
        raise CredentialError(
            f"key_command failed (exit {result.returncode}): {result.stderr.strip()}"
        )

    output = result.stdout.strip()
    if not output:
        raise CredentialError(f"key_command produced no output: {command}")

    return SecretStr(output)
