"""Tests for core data models."""

from datetime import datetime, timezone

from clawmeter.models import (
    CredentialError,
    ModelUsage,
    ProviderStatus,
    SecretStr,
    UsageWindow,
    compute_status,
)


class TestSecretStr:
    def test_repr_never_leaks(self):
        s = SecretStr("sk-ant-oat01-super-secret-token")
        assert repr(s) == "SecretStr('***')"

    def test_str_always_masked(self):
        s = SecretStr("sk-ant-oat01-super-secret-token")
        assert str(s) == "***REDACTED***"

    def test_get_secret_value_returns_real(self):
        s = SecretStr("my-secret")
        assert s.get_secret_value() == "my-secret"

    def test_bool_true_for_nonempty(self):
        assert bool(SecretStr("value"))

    def test_bool_false_for_empty(self):
        assert not bool(SecretStr(""))

    def test_len_returns_actual_length(self):
        s = SecretStr("12345")
        assert len(s) == 5

    def test_len_empty(self):
        assert len(SecretStr("")) == 0

    def test_equality(self):
        a = SecretStr("same")
        b = SecretStr("same")
        assert a == b

    def test_inequality(self):
        a = SecretStr("one")
        b = SecretStr("two")
        assert a != b

    def test_fstring_does_not_leak(self):
        s = SecretStr("sk-ant-oat01-token")
        result = f"token={s}"
        assert "sk-ant" not in result
        assert "***REDACTED***" in result

    def test_repr_in_container_does_not_leak(self):
        s = SecretStr("secret")
        result = repr([s])
        assert "secret" not in result


class TestCredentialError:
    def test_message(self):
        e = CredentialError("command failed")
        assert str(e) == "command failed"

    def test_provider_attribute(self):
        e = CredentialError("timeout", provider="claude")
        assert e.provider == "claude"

    def test_default_provider_empty(self):
        e = CredentialError("error")
        assert e.provider == ""

    def test_is_exception(self):
        assert issubclass(CredentialError, Exception)


class TestUsageWindow:
    def test_construction(self):
        w = UsageWindow(
            name="Session (5h)",
            utilisation=42.0,
            resets_at=datetime(2026, 4, 5, 15, 0, tzinfo=timezone.utc),
            status="normal",
            unit="percent",
        )
        assert w.name == "Session (5h)"
        assert w.utilisation == 42.0
        assert w.status == "normal"

    def test_optional_fields_default_none(self):
        w = UsageWindow(
            name="test",
            utilisation=0.0,
            resets_at=None,
            status="normal",
            unit="percent",
        )
        assert w.raw_value is None
        assert w.raw_limit is None


class TestModelUsage:
    def test_construction(self):
        m = ModelUsage(model="claude-opus-4-6", input_tokens=1000, output_tokens=500)
        assert m.model == "claude-opus-4-6"
        assert m.total_tokens is None  # not set

    def test_all_optional_default_none(self):
        m = ModelUsage(model="test")
        assert m.input_tokens is None
        assert m.cost is None
        assert m.period is None


class TestProviderStatus:
    def test_construction(self, sample_status):
        assert sample_status.provider_name == "claude"
        assert len(sample_status.windows) == 1
        assert sample_status.errors == []

    def test_defaults(self):
        s = ProviderStatus(
            provider_name="test",
            provider_display="Test",
            timestamp=datetime.now(timezone.utc),
            cached=False,
            cache_age_seconds=0,
        )
        assert s.windows == []
        assert s.model_usage == []
        assert s.extras == {}
        assert s.errors == []


class TestComputeStatus:
    def test_normal(self):
        assert compute_status(0.0) == "normal"
        assert compute_status(42.0) == "normal"
        assert compute_status(69.9) == "normal"

    def test_warning(self):
        assert compute_status(70.0) == "warning"
        assert compute_status(80.0) == "warning"
        assert compute_status(89.9) == "warning"

    def test_critical(self):
        assert compute_status(90.0) == "critical"
        assert compute_status(95.0) == "critical"
        assert compute_status(99.9) == "critical"

    def test_exceeded(self):
        assert compute_status(100.0) == "exceeded"
        assert compute_status(150.0) == "exceeded"

    def test_custom_thresholds(self):
        t = {"warning": 50.0, "critical": 80.0}
        assert compute_status(49.9, t) == "normal"
        assert compute_status(50.0, t) == "warning"
        assert compute_status(80.0, t) == "critical"
        assert compute_status(100.0, t) == "exceeded"
