"""Tests for the Ollama provider — local instance monitoring."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import respx

from llm_monitor.models import ProviderStatus
from llm_monitor.providers.ollama import OllamaProvider

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
HOST_URL = "http://localhost:11434"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_fixture(name: str) -> dict:
    return json.loads((FIXTURES_DIR / name).read_text())


def _make_config(
    enabled: bool = True,
    host: str = HOST_URL,
    hosts: list | None = None,
    cloud_enabled: bool = False,
) -> dict:
    cfg: dict = {
        "providers": {
            "ollama": {
                "enabled": enabled,
                "host": host,
                "cloud_enabled": cloud_enabled,
            },
        },
    }
    if hosts is not None:
        cfg["providers"]["ollama"]["hosts"] = hosts
        # When using hosts array, remove host key
        del cfg["providers"]["ollama"]["host"]
    return cfg


def _make_provider(
    host: str = HOST_URL,
    hosts: list | None = None,
) -> OllamaProvider:
    config = _make_config(host=host, hosts=hosts)
    return OllamaProvider(config)


def _mock_host(
    base_url: str = HOST_URL,
    tags: dict | None = None,
    ps: dict | None = None,
) -> None:
    """Set up respx mocks for a single Ollama host."""
    respx.get(f"{base_url}/api/tags").respond(
        200, json=tags or _load_fixture("ollama_tags.json")
    )
    respx.get(f"{base_url}/api/ps").respond(
        200, json=ps or _load_fixture("ollama_ps.json")
    )


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


class TestMetadata:
    def test_name(self):
        provider = OllamaProvider(_make_config())
        assert provider.name() == "ollama"

    def test_display_name(self):
        provider = OllamaProvider(_make_config())
        assert provider.display_name() == "Ollama"

    def test_auth_instructions(self):
        provider = OllamaProvider(_make_config())
        instructions = provider.auth_instructions()
        assert "no credentials" in instructions.lower()
        assert "ollama serve" in instructions

    def test_allowed_hosts_local(self):
        provider = OllamaProvider(_make_config())
        assert "localhost" in provider.allowed_hosts

    def test_allowed_hosts_cloud(self):
        provider = OllamaProvider(_make_config(cloud_enabled=True))
        assert "ollama.com" in provider.allowed_hosts


# ---------------------------------------------------------------------------
# is_configured
# ---------------------------------------------------------------------------


class TestIsConfigured:
    def test_configured_with_default_host(self):
        provider = OllamaProvider(_make_config())
        assert provider.is_configured() is True

    def test_configured_with_custom_host(self):
        provider = OllamaProvider(_make_config(host="http://gpu-server:11434"))
        assert provider.is_configured() is True

    def test_configured_with_hosts_array(self):
        hosts = [
            {"name": "local", "url": "http://localhost:11434"},
            {"name": "gpu", "url": "http://gpu-server:11434"},
        ]
        provider = _make_provider(hosts=hosts)
        assert provider.is_configured() is True


# ---------------------------------------------------------------------------
# Host resolution
# ---------------------------------------------------------------------------


class TestHostResolution:
    def test_single_host(self):
        provider = _make_provider(host="http://myhost:11434")
        assert len(provider._hosts) == 1
        assert provider._hosts[0]["url"] == "http://myhost:11434"
        assert provider._hosts[0]["name"] == "myhost"

    def test_single_host_trailing_slash_stripped(self):
        provider = _make_provider(host="http://myhost:11434/")
        assert provider._hosts[0]["url"] == "http://myhost:11434"

    def test_multi_host_array(self):
        hosts = [
            {"name": "workstation", "url": "http://localhost:11434"},
            {"name": "gpu-server", "url": "http://gpu-server.local:11434"},
        ]
        provider = _make_provider(hosts=hosts)
        assert len(provider._hosts) == 2
        assert provider._hosts[0]["name"] == "workstation"
        assert provider._hosts[1]["name"] == "gpu-server"

    def test_multi_host_name_derived_from_url(self):
        hosts = [
            {"url": "http://192.168.1.50:11434"},
        ]
        provider = _make_provider(hosts=hosts)
        assert provider._hosts[0]["name"] == "192.168.1.50"

    def test_default_host(self):
        config = {"providers": {"ollama": {"enabled": True}}}
        provider = OllamaProvider(config)
        assert len(provider._hosts) == 1
        assert provider._hosts[0]["url"] == "http://localhost:11434"


# ---------------------------------------------------------------------------
# fetch_usage — single host, successful responses
# ---------------------------------------------------------------------------


class TestFetchUsageSingleHost:
    @respx.mock
    @pytest.mark.asyncio
    async def test_full_response_produces_windows(self):
        """200 from /api/tags and /api/ps produces expected windows."""
        provider = _make_provider()
        _mock_host()

        async with httpx.AsyncClient() as client:
            status = await provider.fetch_usage(client)

        assert isinstance(status, ProviderStatus)
        assert len(status.errors) == 0
        names = {w.name for w in status.windows}
        assert "Models Available" in names
        assert "Models Loaded" in names
        assert "VRAM Usage" in names

    @respx.mock
    @pytest.mark.asyncio
    async def test_models_available_count(self):
        """Models Available window has correct count from /api/tags."""
        provider = _make_provider()
        _mock_host()

        async with httpx.AsyncClient() as client:
            status = await provider.fetch_usage(client)

        avail = next(w for w in status.windows if w.name == "Models Available")
        # Fixture has 3 models (gemma3, llama3.2, deepseek-v3.2:cloud)
        assert avail.raw_value == 3.0
        assert avail.unit == "count"

    @respx.mock
    @pytest.mark.asyncio
    async def test_models_loaded_count(self):
        """Models Loaded window has correct count from /api/ps."""
        provider = _make_provider()
        _mock_host()

        async with httpx.AsyncClient() as client:
            status = await provider.fetch_usage(client)

        loaded = next(w for w in status.windows if w.name == "Models Loaded")
        assert loaded.raw_value == 1.0

    @respx.mock
    @pytest.mark.asyncio
    async def test_vram_usage_calculated(self):
        """VRAM Usage is calculated from /api/ps size_vram."""
        provider = _make_provider()
        _mock_host()

        async with httpx.AsyncClient() as client:
            status = await provider.fetch_usage(client)

        vram = next(w for w in status.windows if w.name == "VRAM Usage")
        # size_vram = 5333539264 bytes = ~5085 MB
        assert vram.raw_value == pytest.approx(5085.7, abs=1.0)
        assert vram.unit == "mb"

    @respx.mock
    @pytest.mark.asyncio
    async def test_ram_usage_calculated(self):
        """RAM Usage is derived from size - size_vram."""
        provider = _make_provider()
        _mock_host()

        async with httpx.AsyncClient() as client:
            status = await provider.fetch_usage(client)

        ram = next(w for w in status.windows if w.name == "RAM Usage")
        # size=6591830464 - size_vram=5333539264 = 1258291200 bytes = ~1200 MB
        assert ram.raw_value == pytest.approx(1200.0, abs=2.0)
        assert ram.unit == "mb"

    @respx.mock
    @pytest.mark.asyncio
    async def test_extras_host_data(self):
        """Extras dict contains per-host data."""
        provider = _make_provider()
        _mock_host()

        async with httpx.AsyncClient() as client:
            status = await provider.fetch_usage(client)

        assert len(status.extras["hosts"]) == 1
        host = status.extras["hosts"][0]
        assert host["status"] == "connected"
        assert host["models_available"] == 3
        assert len(host["models_loaded"]) == 1
        assert host["models_loaded"][0]["name"] == "gemma3:latest"


# ---------------------------------------------------------------------------
# CPU-only model (size_vram omitted)
# ---------------------------------------------------------------------------


class TestCpuOnlyModel:
    @respx.mock
    @pytest.mark.asyncio
    async def test_cpu_only_no_vram_window(self):
        """When size_vram is omitted (CPU-only), no VRAM window is created."""
        provider = _make_provider()
        _mock_host(ps=_load_fixture("ollama_ps_cpu_only.json"))

        async with httpx.AsyncClient() as client:
            status = await provider.fetch_usage(client)

        names = {w.name for w in status.windows}
        assert "VRAM Usage" not in names
        assert "RAM Usage" in names

    @respx.mock
    @pytest.mark.asyncio
    async def test_cpu_only_ram_equals_full_size(self):
        """When size_vram is absent, all memory is RAM."""
        provider = _make_provider()
        _mock_host(ps=_load_fixture("ollama_ps_cpu_only.json"))

        async with httpx.AsyncClient() as client:
            status = await provider.fetch_usage(client)

        ram = next(w for w in status.windows if w.name == "RAM Usage")
        # size=2019266048 bytes, size_vram=0 -> all RAM = ~1925 MB
        assert ram.raw_value == pytest.approx(1925.5, abs=1.0)


# ---------------------------------------------------------------------------
# Cloud model detection
# ---------------------------------------------------------------------------


class TestCloudModelDetection:
    @respx.mock
    @pytest.mark.asyncio
    async def test_cloud_models_detected_in_tags(self):
        """Models with :cloud suffix are flagged in host extras."""
        provider = _make_provider()
        _mock_host()

        async with httpx.AsyncClient() as client:
            status = await provider.fetch_usage(client)

        host = status.extras["hosts"][0]
        assert "cloud_models" in host
        assert "deepseek-v3.2:cloud" in host["cloud_models"]

    @respx.mock
    @pytest.mark.asyncio
    async def test_no_cloud_models_no_key(self):
        """When no cloud models exist, cloud_models key is absent."""
        tags = {"models": [
            {
                "name": "gemma3:latest",
                "model": "gemma3:latest",
                "modified_at": "2026-04-08T10:30:00Z",
                "size": 3346018048,
                "digest": "abc123",
                "details": {"parameter_size": "4.3B", "quantization_level": "Q4_K_M"},
            }
        ]}
        provider = _make_provider()
        _mock_host(tags=tags)

        async with httpx.AsyncClient() as client:
            status = await provider.fetch_usage(client)

        host = status.extras["hosts"][0]
        assert "cloud_models" not in host


# ---------------------------------------------------------------------------
# Multi-host support
# ---------------------------------------------------------------------------


class TestMultiHost:
    @respx.mock
    @pytest.mark.asyncio
    async def test_two_hosts_both_polled(self):
        """Both hosts are polled and reported independently."""
        hosts = [
            {"name": "workstation", "url": "http://localhost:11434"},
            {"name": "gpu-server", "url": "http://gpu-server:11434"},
        ]
        provider = _make_provider(hosts=hosts)

        _mock_host(base_url="http://localhost:11434")
        _mock_host(base_url="http://gpu-server:11434")

        async with httpx.AsyncClient() as client:
            status = await provider.fetch_usage(client)

        assert len(status.extras["hosts"]) == 2
        assert status.extras["hosts"][0]["name"] == "workstation"
        assert status.extras["hosts"][1]["name"] == "gpu-server"

    @respx.mock
    @pytest.mark.asyncio
    async def test_multi_host_prefixed_window_names(self):
        """Window names include host prefix when multiple hosts configured."""
        hosts = [
            {"name": "ws", "url": "http://localhost:11434"},
            {"name": "gpu", "url": "http://gpu-server:11434"},
        ]
        provider = _make_provider(hosts=hosts)

        _mock_host(base_url="http://localhost:11434")
        _mock_host(base_url="http://gpu-server:11434")

        async with httpx.AsyncClient() as client:
            status = await provider.fetch_usage(client)

        names = {w.name for w in status.windows}
        assert "ws: Models Available" in names
        assert "gpu: Models Available" in names

    @respx.mock
    @pytest.mark.asyncio
    async def test_one_host_down_other_unaffected(self):
        """One host failing doesn't block the other."""
        hosts = [
            {"name": "healthy", "url": "http://localhost:11434"},
            {"name": "down", "url": "http://down-host:11434"},
        ]
        provider = _make_provider(hosts=hosts)

        _mock_host(base_url="http://localhost:11434")
        respx.get("http://down-host:11434/api/tags").mock(
            side_effect=httpx.ConnectError("Connection refused")
        )

        async with httpx.AsyncClient() as client:
            status = await provider.fetch_usage(client)

        # Healthy host has windows
        healthy_windows = [w for w in status.windows if w.name.startswith("healthy:")]
        assert len(healthy_windows) >= 1

        # Down host reported as unreachable
        down_host = next(
            h for h in status.extras["hosts"] if h["name"] == "down"
        )
        assert down_host["status"] == "unreachable"

        # Errors contain the failure
        assert any("down" in e.lower() for e in status.errors)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    @respx.mock
    @pytest.mark.asyncio
    async def test_tags_unreachable(self):
        """Connection error on /api/tags marks host as unreachable."""
        provider = _make_provider()
        respx.get(f"{HOST_URL}/api/tags").mock(
            side_effect=httpx.ConnectError("Connection refused")
        )

        async with httpx.AsyncClient() as client:
            status = await provider.fetch_usage(client)

        assert len(status.errors) > 0
        assert any("Cannot reach" in e for e in status.errors)
        assert status.extras["hosts"][0]["status"] == "unreachable"

    @respx.mock
    @pytest.mark.asyncio
    async def test_ps_failure_still_returns_tags_data(self):
        """/api/ps failure doesn't prevent /api/tags data from being returned."""
        provider = _make_provider()
        respx.get(f"{HOST_URL}/api/tags").respond(
            200, json=_load_fixture("ollama_tags.json")
        )
        respx.get(f"{HOST_URL}/api/ps").respond(500, text="Internal Error")

        async with httpx.AsyncClient() as client:
            status = await provider.fetch_usage(client)

        # Models Available should still be present
        names = {w.name for w in status.windows}
        assert "Models Available" in names
        # But Models Loaded should not
        assert "Models Loaded" not in names

    @respx.mock
    @pytest.mark.asyncio
    async def test_timeout_error(self):
        """Timeout produces error in ProviderStatus."""
        provider = _make_provider()
        respx.get(f"{HOST_URL}/api/tags").mock(
            side_effect=httpx.ReadTimeout("Read timed out")
        )

        async with httpx.AsyncClient() as client:
            status = await provider.fetch_usage(client)

        assert any("timed out" in e for e in status.errors)

    @respx.mock
    @pytest.mark.asyncio
    async def test_invalid_json_response(self):
        """Invalid JSON from host produces error."""
        provider = _make_provider()
        respx.get(f"{HOST_URL}/api/tags").respond(200, text="not json")

        async with httpx.AsyncClient() as client:
            status = await provider.fetch_usage(client)

        assert any("invalid JSON" in e for e in status.errors)

    @respx.mock
    @pytest.mark.asyncio
    async def test_empty_models_list(self):
        """Empty models list is handled gracefully."""
        provider = _make_provider()
        _mock_host(
            tags={"models": []},
            ps={"models": []},
        )

        async with httpx.AsyncClient() as client:
            status = await provider.fetch_usage(client)

        assert len(status.errors) == 0
        avail = next(w for w in status.windows if w.name == "Models Available")
        assert avail.raw_value == 0.0
