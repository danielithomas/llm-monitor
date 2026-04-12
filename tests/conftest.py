"""Shared test fixtures for clawmeter."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from clawmeter.models import ModelUsage, ProviderStatus, UsageWindow

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_window() -> UsageWindow:
    return UsageWindow(
        name="Session (5h)",
        utilisation=42.0,
        resets_at=datetime(2026, 4, 5, 15, 0, 0, tzinfo=timezone.utc),
        status="normal",
        unit="percent",
    )


@pytest.fixture
def sample_status(sample_window: UsageWindow) -> ProviderStatus:
    return ProviderStatus(
        provider_name="claude",
        provider_display="Anthropic Claude",
        timestamp=datetime(2026, 4, 5, 10, 30, 0, tzinfo=timezone.utc),
        cached=False,
        cache_age_seconds=0,
        windows=[sample_window],
    )


@pytest.fixture
def claude_credentials_json() -> dict:
    return {
        "claudeAiOauth": {
            "accessToken": "sk-ant-oat01-test-token-value",
            "refreshToken": "sk-ant-ort01-test-refresh-token",
            "expiresAt": "2026-04-05T16:30:00.000Z",
        }
    }


@pytest.fixture
def claude_usage_response() -> dict:
    return {
        "five_hour": {
            "utilization": 42.0,
            "resets_at": "2026-04-05T15:00:00+00:00",
        },
        "seven_day": {
            "utilization": 68.0,
            "resets_at": "2026-04-08T00:00:00+00:00",
        },
        "seven_day_opus": {
            "utilization": 12.0,
            "resets_at": "2026-04-08T00:00:00+00:00",
        },
        "seven_day_oauth_apps": None,
        "iguana_necktie": None,
    }
