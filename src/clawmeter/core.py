"""Orchestrator for concurrent provider fetching.

Manages cache lookup, backoff escalation, and parallel fetches via
``asyncio.gather``.  See SPEC.md Sections 2.3 and 8 for details.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

import httpx

from clawmeter.cache import ProviderCache
from clawmeter.models import ProviderStatus

if TYPE_CHECKING:
    from clawmeter.providers.base import Provider

# Backoff constants (Section 3.1)
_BACKOFF_BASE_MINUTES = 10
_BACKOFF_MULTIPLIER = 2
_BACKOFF_CAP_MINUTES = 60


async def fetch_all(
    providers: list[Provider],
    cache: ProviderCache,
    config: dict,
    fresh: bool = False,
) -> list[ProviderStatus]:
    """Fetch usage from all *providers* concurrently.

    Parameters
    ----------
    providers:
        Instantiated provider objects to query.
    cache:
        Cache layer for reading/writing per-provider state.
    config:
        Merged application configuration dict.
    fresh:
        When True, bypass cache and backoff — always fetch from the API.

    Returns
    -------
    list[ProviderStatus]
        One status per provider, in the same order as *providers*.
    """
    poll_interval = config.get("general", {}).get("poll_interval", 600)

    timeout = httpx.Timeout(connect=10.0, read=30.0, write=10.0, pool=5.0)
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:
        tasks = [
            _fetch_one(provider, client, cache, poll_interval, fresh)
            for provider in providers
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    statuses: list[ProviderStatus] = []
    for i, result in enumerate(results):
        if isinstance(result, BaseException):
            now = datetime.now(timezone.utc)
            statuses.append(
                ProviderStatus(
                    provider_name=providers[i].name(),
                    provider_display=providers[i].display_name(),
                    timestamp=now,
                    cached=False,
                    cache_age_seconds=0,
                    errors=[
                        f"Unexpected error fetching {providers[i].name()}: {result}\n"
                        "Fix: Run with --verbose for details, or retry with --fresh."
                    ],
                )
            )
        else:
            statuses.append(result)

    return statuses


async def _fetch_one(
    provider: Provider,
    client: httpx.AsyncClient,
    cache: ProviderCache,
    poll_interval: int,
    fresh: bool,
) -> ProviderStatus:
    """Fetch usage for a single provider with cache and backoff handling."""
    name = provider.name()

    # Step 1: check cache (unless fresh)
    if not fresh:
        cached = cache.read(name, poll_interval)
        if cached is not None:
            return cached

    # Step 2: check backoff state (unless fresh)
    if not fresh:
        count, until = cache.read_backoff(name)
        if count > 0 and until is not None:
            now = datetime.now(timezone.utc)
            if now < until:
                # Still in backoff — return cached data with warning
                cached = cache.read(name, poll_interval=999_999_999)
                if cached is not None:
                    cached.errors.append(
                        f"Rate-limited — backing off until {until.isoformat()}"
                    )
                    return cached
                # No cached data at all; return an error status
                return ProviderStatus(
                    provider_name=name,
                    provider_display=provider.display_name(),
                    timestamp=now,
                    cached=False,
                    cache_age_seconds=0,
                    errors=[
                        f"Rate-limited — backing off until {until.isoformat()}"
                    ],
                )

    # Step 3: fetch from provider
    status = await provider.fetch_usage(client)

    # Step 4: on success (no errors), cache and clear backoff
    if not status.errors:
        cache.write(name, status)
        cache.clear_backoff(name)
        return status

    # Step 5: check for 429 / backoff signal
    if status.extras.get("_backoff"):
        count, _until = cache.read_backoff(name)
        new_count = count + 1
        minutes = min(
            _BACKOFF_BASE_MINUTES * (_BACKOFF_MULTIPLIER ** (new_count - 1)),
            _BACKOFF_CAP_MINUTES,
        )
        backoff_until = datetime.now(timezone.utc) + timedelta(minutes=minutes)
        cache.write_backoff(name, count=new_count, until=backoff_until)

        # Return cached data with warning appended
        cached = cache.read(name, poll_interval=999_999_999)
        if cached is not None:
            cached.errors.append(
                f"Rate-limited — backing off for {int(minutes)}m"
            )
            return cached
        # No cached data — return the error status from the provider
        status.errors.append(
            f"Rate-limited — backing off for {int(minutes)}m"
        )
        return status

    # Step 6: other errors — return as-is
    return status


def determine_exit_code(statuses: list[ProviderStatus]) -> int:
    """Determine the CLI exit code from aggregate provider results.

    Returns
    -------
    int
        0 - all succeeded (no errors)
        2 - all failed auth (all have auth-related errors)
        3 - some succeeded, some failed
        4 - all failed (network/unreachable)
        0 - default / empty list
    """
    if not statuses:
        return 0

    has_errors = [bool(s.errors) for s in statuses]

    # All succeeded
    if not any(has_errors):
        return 0

    # All failed
    if all(has_errors):
        # Check if all are auth errors
        auth_keywords = [
            "auth", "credential", "token expired", "401",
            "login", "permission", "forbidden", "403",
        ]
        all_auth = True
        for s in statuses:
            for err in s.errors:
                err_lower = err.lower()
                if not any(kw in err_lower for kw in auth_keywords):
                    all_auth = False
                    break
            if not all_auth:
                break

        if all_auth:
            return 2
        return 4

    # Some succeeded, some failed
    return 3
