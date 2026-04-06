"""Core data models for llm-monitor."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


class SecretStr:
    """String wrapper that prevents accidental logging of secrets."""

    def __init__(self, value: str):
        self._value = value

    def get_secret_value(self) -> str:
        return self._value

    def __repr__(self) -> str:
        return "SecretStr('***')"

    def __str__(self) -> str:
        return "***REDACTED***"

    def __len__(self) -> int:
        return len(self._value)

    def __bool__(self) -> bool:
        return bool(self._value)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, SecretStr):
            return self._value == other._value
        return NotImplemented


class CredentialError(Exception):
    """Raised when credential resolution fails.

    This is a hard failure — the provider cannot authenticate.
    Used when key_command returns non-zero, times out, or produces
    no output. NOT used for "no credential found" (which returns None).
    """

    def __init__(self, message: str, provider: str = ""):
        self.provider = provider
        super().__init__(message)


@dataclass
class UsageWindow:
    """A time-bounded usage allocation (e.g., 5-hour session, monthly budget)."""

    name: str
    utilisation: float  # 0.0 - 100.0+ (percentage)
    resets_at: Optional[datetime]
    status: str  # normal | warning | critical | exceeded
    unit: str  # "percent" | "usd" | "tokens"
    raw_value: Optional[float] = None
    raw_limit: Optional[float] = None


@dataclass
class ModelUsage:
    """Per-model usage breakdown within a provider."""

    model: str
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    cost: Optional[float] = None
    request_count: Optional[int] = None
    period: Optional[str] = None


@dataclass
class ProviderStatus:
    """Unified status response from any provider."""

    provider_name: str
    provider_display: str
    timestamp: datetime
    cached: bool
    cache_age_seconds: int
    windows: list[UsageWindow] = field(default_factory=list)
    model_usage: list[ModelUsage] = field(default_factory=list)
    extras: dict = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


# Status threshold defaults (configurable via [thresholds] config section)
DEFAULT_THRESHOLDS = {
    "warning": 70.0,
    "critical": 90.0,
}


def compute_status(utilisation: float, thresholds: dict | None = None) -> str:
    """Determine status string from utilisation percentage."""
    t = thresholds or DEFAULT_THRESHOLDS
    if utilisation >= 100.0:
        return "exceeded"
    if utilisation >= t.get("critical", 90.0):
        return "critical"
    if utilisation >= t.get("warning", 70.0):
        return "warning"
    return "normal"
