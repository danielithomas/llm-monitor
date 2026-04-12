"""Tests for the history store (SQLite, change detection, pruning, export, report)."""

from __future__ import annotations

import csv
import io
import json
import sqlite3
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import respx
from click.testing import CliRunner

from clawmeter.cli import cli
from clawmeter.history import HistoryStore, SCHEMA_VERSION
from clawmeter.models import ModelUsage, ProviderStatus, UsageWindow


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_window(
    name: str = "Session (5h)",
    utilisation: float = 42.0,
    status: str = "normal",
    unit: str = "percent",
    resets_at: datetime | None = None,
    raw_value: float | None = None,
    raw_limit: float | None = None,
) -> UsageWindow:
    if resets_at is None:
        resets_at = datetime(2026, 4, 5, 15, 0, 0, tzinfo=timezone.utc)
    return UsageWindow(
        name=name,
        utilisation=utilisation,
        resets_at=resets_at,
        status=status,
        unit=unit,
        raw_value=raw_value,
        raw_limit=raw_limit,
    )


def _make_status(
    windows: list[UsageWindow] | None = None,
    provider_name: str = "claude",
    timestamp: datetime | None = None,
    cached: bool = False,
    model_usage: list[ModelUsage] | None = None,
    extras: dict | None = None,
    errors: list[str] | None = None,
) -> ProviderStatus:
    if timestamp is None:
        timestamp = datetime(2026, 4, 5, 10, 30, 0, tzinfo=timezone.utc)
    return ProviderStatus(
        provider_name=provider_name,
        provider_display="Anthropic Claude",
        timestamp=timestamp,
        cached=cached,
        cache_age_seconds=0,
        windows=windows or [_make_window()],
        model_usage=model_usage or [],
        extras=extras or {},
        errors=errors or [],
    )


def _make_model_usage(
    model: str = "claude-opus-4-6",
    input_tokens: int = 15000,
    output_tokens: int = 8000,
    total_tokens: int = 23000,
    cost: float | None = None,
    request_count: int = 12,
    period: str = "5h",
) -> ModelUsage:
    return ModelUsage(
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cost=cost,
        request_count=request_count,
        period=period,
    )


# ---------------------------------------------------------------------------
# Schema and database creation
# ---------------------------------------------------------------------------


class TestSchemaCreation:
    def test_creates_database(self, tmp_path):
        db_path = tmp_path / "test.db"
        with HistoryStore(db_path) as store:
            assert db_path.exists()

    def test_tables_created(self, tmp_path):
        db_path = tmp_path / "test.db"
        with HistoryStore(db_path) as store:
            tables = store.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ).fetchall()
            table_names = {r[0] for r in tables}
            assert "usage_samples" in table_names
            assert "model_usage" in table_names
            assert "provider_extras" in table_names
            assert "schema_version" in table_names

    def test_schema_version_set(self, tmp_path):
        db_path = tmp_path / "test.db"
        with HistoryStore(db_path) as store:
            row = store.conn.execute("SELECT version FROM schema_version").fetchone()
            assert row[0] == SCHEMA_VERSION

    def test_wal_mode_enabled(self, tmp_path):
        db_path = tmp_path / "test.db"
        with HistoryStore(db_path) as store:
            row = store.conn.execute("PRAGMA journal_mode").fetchone()
            assert row[0] == "wal"

    def test_auto_vacuum_incremental(self, tmp_path):
        db_path = tmp_path / "test.db"
        with HistoryStore(db_path) as store:
            row = store.conn.execute("PRAGMA auto_vacuum").fetchone()
            # 2 = INCREMENTAL
            assert row[0] == 2

    def test_indexes_created(self, tmp_path):
        db_path = tmp_path / "test.db"
        with HistoryStore(db_path) as store:
            indexes = store.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' "
                "AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
            index_names = {r[0] for r in indexes}
            assert "idx_samples_provider_time" in index_names
            assert "idx_samples_time" in index_names
            assert "idx_model_usage_provider_time" in index_names
            assert "idx_model_usage_model" in index_names
            assert "idx_extras_provider_time" in index_names

    def test_reopen_existing_db(self, tmp_path):
        db_path = tmp_path / "test.db"
        # Create and close
        with HistoryStore(db_path) as store:
            store.record(_make_status())

        # Reopen
        with HistoryStore(db_path) as store:
            count = store.conn.execute("SELECT COUNT(*) FROM usage_samples").fetchone()[0]
            assert count == 1

    def test_file_permissions(self, tmp_path):
        db_path = tmp_path / "subdir" / "test.db"
        with HistoryStore(db_path) as store:
            pass
        import stat
        mode = db_path.stat().st_mode & 0o777
        assert mode == 0o600


# ---------------------------------------------------------------------------
# Write-on-fetch
# ---------------------------------------------------------------------------


class TestWriteOnFetch:
    def test_basic_write(self, tmp_path):
        db_path = tmp_path / "test.db"
        with HistoryStore(db_path) as store:
            rows = store.record(_make_status())
            assert rows == 1

            result = store.conn.execute("SELECT * FROM usage_samples").fetchall()
            assert len(result) == 1
            assert result[0]["provider"] == "claude"
            assert result[0]["window_name"] == "Session (5h)"
            assert result[0]["utilisation"] == 42.0
            assert result[0]["status"] == "normal"

    def test_multiple_windows(self, tmp_path):
        db_path = tmp_path / "test.db"
        windows = [
            _make_window(name="Session (5h)", utilisation=42.0),
            _make_window(name="Weekly (7d)", utilisation=68.0),
            _make_window(name="Opus (7d)", utilisation=12.0),
        ]
        status = _make_status(windows=windows)

        with HistoryStore(db_path) as store:
            rows = store.record(status)
            assert rows == 3

            result = store.conn.execute("SELECT * FROM usage_samples").fetchall()
            assert len(result) == 3

    def test_model_usage_written(self, tmp_path):
        db_path = tmp_path / "test.db"
        mu = _make_model_usage()
        status = _make_status(model_usage=[mu])

        with HistoryStore(db_path) as store:
            rows = store.record(status)
            assert rows == 2  # 1 window + 1 model_usage

            result = store.conn.execute("SELECT * FROM model_usage").fetchall()
            assert len(result) == 1
            assert result[0]["model"] == "claude-opus-4-6"
            assert result[0]["input_tokens"] == 15000

    def test_extras_written(self, tmp_path):
        db_path = tmp_path / "test.db"
        status = _make_status(extras={"plan": "Pro", "token_expires_at": "2026-04-05T12:00:00Z"})

        with HistoryStore(db_path) as store:
            store.record(status)

            result = store.conn.execute("SELECT * FROM provider_extras").fetchall()
            assert len(result) == 1
            extras = json.loads(result[0]["extras_json"])
            assert extras["plan"] == "Pro"

    def test_internal_extras_filtered(self, tmp_path):
        """Keys starting with _ should not be stored in extras."""
        db_path = tmp_path / "test.db"
        status = _make_status(extras={"plan": "Pro", "_backoff": True})

        with HistoryStore(db_path) as store:
            store.record(status)

            result = store.conn.execute("SELECT * FROM provider_extras").fetchall()
            assert len(result) == 1
            extras = json.loads(result[0]["extras_json"])
            assert "_backoff" not in extras
            assert "plan" in extras

    def test_errors_not_written(self, tmp_path):
        """Statuses with errors should not be written to history."""
        db_path = tmp_path / "test.db"
        status = _make_status(errors=["Network error"])

        with HistoryStore(db_path) as store:
            rows = store.record(status)
            assert rows == 0

    def test_timestamp_stored_as_iso(self, tmp_path):
        db_path = tmp_path / "test.db"
        ts = datetime(2026, 4, 5, 10, 30, 0, tzinfo=timezone.utc)
        status = _make_status(timestamp=ts)

        with HistoryStore(db_path) as store:
            store.record(status)
            row = store.conn.execute("SELECT timestamp FROM usage_samples").fetchone()
            assert row[0] == ts.isoformat()


# ---------------------------------------------------------------------------
# Meaningful-change detection
# ---------------------------------------------------------------------------


class TestMeaningfulChange:
    def test_first_sample_always_written(self, tmp_path):
        db_path = tmp_path / "test.db"
        with HistoryStore(db_path) as store:
            rows = store.record(_make_status())
            assert rows == 1

    def test_no_change_skipped(self, tmp_path):
        """Identical data should not produce a second write."""
        db_path = tmp_path / "test.db"
        status1 = _make_status(
            timestamp=datetime(2026, 4, 5, 10, 0, 0, tzinfo=timezone.utc)
        )
        status2 = _make_status(
            timestamp=datetime(2026, 4, 5, 10, 10, 0, tzinfo=timezone.utc)
        )

        with HistoryStore(db_path) as store:
            store.record(status1)
            rows = store.record(status2)
            assert rows == 0

            count = store.conn.execute("SELECT COUNT(*) FROM usage_samples").fetchone()[0]
            assert count == 1

    def test_small_delta_skipped(self, tmp_path):
        """Utilisation delta <= 0.1% should not trigger a write."""
        db_path = tmp_path / "test.db"
        status1 = _make_status(
            windows=[_make_window(utilisation=42.0)],
            timestamp=datetime(2026, 4, 5, 10, 0, 0, tzinfo=timezone.utc),
        )
        status2 = _make_status(
            windows=[_make_window(utilisation=42.05)],
            timestamp=datetime(2026, 4, 5, 10, 10, 0, tzinfo=timezone.utc),
        )

        with HistoryStore(db_path) as store:
            store.record(status1)
            rows = store.record(status2)
            assert rows == 0

    def test_large_delta_written(self, tmp_path):
        """Utilisation delta > 0.1% should trigger a write."""
        db_path = tmp_path / "test.db"
        status1 = _make_status(
            windows=[_make_window(utilisation=42.0)],
            timestamp=datetime(2026, 4, 5, 10, 0, 0, tzinfo=timezone.utc),
        )
        status2 = _make_status(
            windows=[_make_window(utilisation=43.0)],
            timestamp=datetime(2026, 4, 5, 10, 10, 0, tzinfo=timezone.utc),
        )

        with HistoryStore(db_path) as store:
            store.record(status1)
            rows = store.record(status2)
            assert rows == 1

    def test_status_change_written(self, tmp_path):
        """Status change should trigger a write even if utilisation is the same."""
        db_path = tmp_path / "test.db"
        status1 = _make_status(
            windows=[_make_window(utilisation=42.0, status="normal")],
            timestamp=datetime(2026, 4, 5, 10, 0, 0, tzinfo=timezone.utc),
        )
        status2 = _make_status(
            windows=[_make_window(utilisation=42.0, status="warning")],
            timestamp=datetime(2026, 4, 5, 10, 10, 0, tzinfo=timezone.utc),
        )

        with HistoryStore(db_path) as store:
            store.record(status1)
            rows = store.record(status2)
            assert rows == 1

    def test_reset_detection_written(self, tmp_path):
        """A new resets_at should trigger a write."""
        db_path = tmp_path / "test.db"
        status1 = _make_status(
            windows=[_make_window(
                utilisation=42.0,
                resets_at=datetime(2026, 4, 5, 15, 0, 0, tzinfo=timezone.utc),
            )],
            timestamp=datetime(2026, 4, 5, 10, 0, 0, tzinfo=timezone.utc),
        )
        status2 = _make_status(
            windows=[_make_window(
                utilisation=42.0,
                resets_at=datetime(2026, 4, 5, 20, 0, 0, tzinfo=timezone.utc),
            )],
            timestamp=datetime(2026, 4, 5, 15, 10, 0, tzinfo=timezone.utc),
        )

        with HistoryStore(db_path) as store:
            store.record(status1)
            rows = store.record(status2)
            assert rows == 1

    def test_cached_no_change_skipped(self, tmp_path):
        """Cached response with no change should not be written."""
        db_path = tmp_path / "test.db"
        status1 = _make_status(
            timestamp=datetime(2026, 4, 5, 10, 0, 0, tzinfo=timezone.utc),
        )
        status2 = _make_status(
            cached=True,
            timestamp=datetime(2026, 4, 5, 10, 10, 0, tzinfo=timezone.utc),
        )

        with HistoryStore(db_path) as store:
            store.record(status1)
            rows = store.record(status2)
            assert rows == 0

    def test_last_known_survives_reopen(self, tmp_path):
        """Last-known state should be reloaded when reopening the DB."""
        db_path = tmp_path / "test.db"
        status1 = _make_status(
            windows=[_make_window(utilisation=42.0)],
            timestamp=datetime(2026, 4, 5, 10, 0, 0, tzinfo=timezone.utc),
        )

        with HistoryStore(db_path) as store:
            store.record(status1)

        # Reopen and try to write the same data
        status2 = _make_status(
            windows=[_make_window(utilisation=42.0)],
            timestamp=datetime(2026, 4, 5, 10, 10, 0, tzinfo=timezone.utc),
        )
        with HistoryStore(db_path) as store:
            rows = store.record(status2)
            assert rows == 0


# ---------------------------------------------------------------------------
# Retention pruning
# ---------------------------------------------------------------------------


class TestRetentionPruning:
    def test_prune_old_records(self, tmp_path):
        db_path = tmp_path / "test.db"
        now = datetime.now(timezone.utc)
        old_ts = now - timedelta(days=100)
        recent_ts = now - timedelta(days=10)

        with HistoryStore(db_path) as store:
            # Insert old record
            store.record(_make_status(
                windows=[_make_window(utilisation=42.0)],
                timestamp=old_ts,
            ))
            # Insert recent record
            store.record(_make_status(
                windows=[_make_window(utilisation=50.0)],
                timestamp=recent_ts,
            ))

            count_before = store.conn.execute(
                "SELECT COUNT(*) FROM usage_samples"
            ).fetchone()[0]
            assert count_before == 2

            deleted = store.prune(90)
            assert deleted >= 1

            count_after = store.conn.execute(
                "SELECT COUNT(*) FROM usage_samples"
            ).fetchone()[0]
            assert count_after == 1

    def test_prune_keeps_recent(self, tmp_path):
        db_path = tmp_path / "test.db"
        now = datetime.now(timezone.utc)
        recent_ts = now - timedelta(days=10)

        with HistoryStore(db_path) as store:
            store.record(_make_status(timestamp=recent_ts))
            deleted = store.prune(90)
            assert deleted == 0

    def test_prune_count(self, tmp_path):
        db_path = tmp_path / "test.db"
        now = datetime.now(timezone.utc)
        old_ts = now - timedelta(days=100)

        with HistoryStore(db_path) as store:
            store.record(_make_status(timestamp=old_ts))
            count = store.prune_count(90)
            assert count >= 1


# ---------------------------------------------------------------------------
# Purge
# ---------------------------------------------------------------------------


class TestPurge:
    def test_purge_deletes_all(self, tmp_path):
        db_path = tmp_path / "test.db"
        with HistoryStore(db_path) as store:
            store.record(_make_status(model_usage=[_make_model_usage()],
                                      extras={"plan": "Pro"}))
            assert store.conn.execute("SELECT COUNT(*) FROM usage_samples").fetchone()[0] > 0

            deleted = store.purge()
            assert deleted > 0
            assert store.conn.execute("SELECT COUNT(*) FROM usage_samples").fetchone()[0] == 0
            assert store.conn.execute("SELECT COUNT(*) FROM model_usage").fetchone()[0] == 0
            assert store.conn.execute("SELECT COUNT(*) FROM provider_extras").fetchone()[0] == 0

    def test_purge_cli_confirm_flag(self, tmp_path, monkeypatch):
        """--confirm should bypass interactive prompt."""
        db_path = tmp_path / "data" / "history.db"
        monkeypatch.setenv("LLM_MONITOR_DATA_DIR", str(tmp_path / "data"))

        # Pre-populate the DB
        with HistoryStore(db_path) as store:
            store.record(_make_status())

        runner = CliRunner()
        result = runner.invoke(cli, ["history", "purge", "--confirm"])
        assert result.exit_code == 0
        assert "purged successfully" in result.output.lower() or "purged successfully" in (result.stderr or "").lower()

    def test_purge_cli_wrong_input(self, tmp_path, monkeypatch):
        """Typing anything other than 'purge' should abort."""
        db_path = tmp_path / "data" / "history.db"
        monkeypatch.setenv("LLM_MONITOR_DATA_DIR", str(tmp_path / "data"))

        with HistoryStore(db_path) as store:
            store.record(_make_status())

        monkeypatch.setattr("clawmeter.cli._stdin_is_tty", lambda: True)
        monkeypatch.setattr("builtins.input", lambda prompt="": "yes")

        runner = CliRunner()
        result = runner.invoke(cli, ["history", "purge"])
        combined = result.output + (result.stderr or "")
        assert "aborted" in combined.lower()

    def test_purge_cli_correct_input(self, tmp_path, monkeypatch):
        """Typing 'purge' should proceed."""
        db_path = tmp_path / "data" / "history.db"
        monkeypatch.setenv("LLM_MONITOR_DATA_DIR", str(tmp_path / "data"))

        with HistoryStore(db_path) as store:
            store.record(_make_status())

        monkeypatch.setattr("clawmeter.cli._stdin_is_tty", lambda: True)
        monkeypatch.setattr("builtins.input", lambda prompt="": "purge")

        runner = CliRunner()
        result = runner.invoke(cli, ["history", "purge"])
        combined = result.output + (result.stderr or "")
        assert "purged successfully" in combined.lower()

    def test_purge_cli_non_tty_no_confirm(self, tmp_path, monkeypatch):
        """When stdin is not a TTY and --confirm is missing, should error."""
        db_path = tmp_path / "data" / "history.db"
        monkeypatch.setenv("LLM_MONITOR_DATA_DIR", str(tmp_path / "data"))

        with HistoryStore(db_path) as store:
            store.record(_make_status())

        monkeypatch.setattr("clawmeter.cli._stdin_is_tty", lambda: False)

        runner = CliRunner()
        result = runner.invoke(cli, ["history", "purge"])
        combined = result.output + (result.stderr or "")
        assert "interactive confirmation" in combined.lower()


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


class TestStats:
    def test_stats_empty_db(self, tmp_path):
        db_path = tmp_path / "test.db"
        with HistoryStore(db_path) as store:
            info = store.stats()
            assert info["sample_count"] == 0
            assert info["providers"] == []
            assert info["oldest"] is None
            assert info["newest"] is None

    def test_stats_with_data(self, tmp_path):
        db_path = tmp_path / "test.db"
        with HistoryStore(db_path) as store:
            store.record(_make_status(model_usage=[_make_model_usage()]))
            info = store.stats()
            assert info["sample_count"] == 1
            assert info["model_count"] == 1
            assert "claude" in info["providers"]
            assert info["oldest"] is not None
            assert info["db_size"] > 0

    def test_stats_cli(self, tmp_path, monkeypatch):
        db_path = tmp_path / "data" / "history.db"
        monkeypatch.setenv("LLM_MONITOR_DATA_DIR", str(tmp_path / "data"))

        with HistoryStore(db_path) as store:
            store.record(_make_status())

        runner = CliRunner()
        result = runner.invoke(cli, ["history", "stats"])
        assert result.exit_code == 0
        assert "Samples:" in result.output
        assert "Providers:" in result.output
        assert "claude" in result.output


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


class TestReport:
    def _populate_db(self, store: HistoryStore) -> None:
        """Insert a series of samples for report testing."""
        base_ts = datetime(2026, 4, 1, 0, 0, 0, tzinfo=timezone.utc)
        for i in range(24):
            ts = base_ts + timedelta(hours=i)
            util = 30.0 + i * 2.5  # 30% -> 87.5%
            status_str = "normal"
            if util >= 90:
                status_str = "critical"
            elif util >= 70:
                status_str = "warning"

            store.record(_make_status(
                windows=[
                    _make_window(utilisation=util, status=status_str),
                    _make_window(name="Weekly (7d)", utilisation=util * 0.8, status="normal"),
                ],
                model_usage=[_make_model_usage(
                    total_tokens=1000 * (i + 1),
                    input_tokens=600 * (i + 1),
                    output_tokens=400 * (i + 1),
                    request_count=i + 1,
                )],
                timestamp=ts,
            ))

    def test_report_table_format(self, tmp_path):
        db_path = tmp_path / "test.db"
        with HistoryStore(db_path) as store:
            self._populate_db(store)

            samples = store.query_samples(
                from_dt=datetime(2026, 3, 30, tzinfo=timezone.utc),
                to_dt=datetime(2026, 4, 5, tzinfo=timezone.utc),
            )
            assert len(samples) > 0

    def test_report_date_filtering(self, tmp_path):
        db_path = tmp_path / "test.db"
        with HistoryStore(db_path) as store:
            self._populate_db(store)

            # Only query first 12 hours
            samples = store.query_samples(
                from_dt=datetime(2026, 4, 1, 0, 0, 0, tzinfo=timezone.utc),
                to_dt=datetime(2026, 4, 1, 12, 0, 0, tzinfo=timezone.utc),
            )
            # Should have fewer than all 24 samples
            assert 0 < len(samples) < 48  # 24 hours × 2 windows

    def test_report_provider_filter(self, tmp_path):
        db_path = tmp_path / "test.db"
        with HistoryStore(db_path) as store:
            self._populate_db(store)

            # Add a different provider
            store.record(_make_status(
                provider_name="openai",
                windows=[_make_window(name="Budget", utilisation=50.0)],
                timestamp=datetime(2026, 4, 1, 5, 0, 0, tzinfo=timezone.utc),
            ))

            claude_only = store.query_samples(provider="claude")
            all_samples = store.query_samples()
            assert len(all_samples) > len(claude_only)

    def test_report_window_filter(self, tmp_path):
        db_path = tmp_path / "test.db"
        with HistoryStore(db_path) as store:
            self._populate_db(store)

            session_only = store.query_samples(window="Session (5h)")
            all_samples = store.query_samples()
            assert 0 < len(session_only) < len(all_samples)

    def test_aggregation_raw(self, tmp_path):
        db_path = tmp_path / "test.db"
        with HistoryStore(db_path) as store:
            self._populate_db(store)
            samples = store.query_samples()
            raw = store.aggregate_samples(samples, "raw")
            assert raw is samples  # Raw returns the same list

    def test_aggregation_hourly(self, tmp_path):
        db_path = tmp_path / "test.db"
        with HistoryStore(db_path) as store:
            self._populate_db(store)
            samples = store.query_samples()
            hourly = store.aggregate_samples(samples, "hourly")
            # Each hour has at most 1 sample per window in our test data
            # so hourly should have same or fewer entries
            assert len(hourly) <= len(samples)
            for bucket in hourly:
                assert "bucket_start" in bucket
                assert "bucket_end" in bucket
                assert "sample_count" in bucket

    def test_aggregation_daily(self, tmp_path):
        db_path = tmp_path / "test.db"
        with HistoryStore(db_path) as store:
            self._populate_db(store)
            samples = store.query_samples()
            daily = store.aggregate_samples(samples, "daily")
            # All data is within one day, so should aggregate
            assert len(daily) <= len(samples)

    def test_aggregation_mean_utilisation(self, tmp_path):
        """Daily aggregation should use mean for utilisation."""
        db_path = tmp_path / "test.db"
        with HistoryStore(db_path) as store:
            self._populate_db(store)
            samples = store.query_samples(window="Session (5h)")
            daily = store.aggregate_samples(samples, "daily")
            assert len(daily) == 1  # All in one day

            # Mean of 30.0, 32.5, 35.0, ..., 87.5 = 58.75
            bucket = daily[0]
            assert 50.0 < bucket["utilisation"] < 70.0

    def test_aggregation_max_severity_status(self, tmp_path):
        """Daily aggregation should use max-severity for status."""
        db_path = tmp_path / "test.db"
        with HistoryStore(db_path) as store:
            self._populate_db(store)
            samples = store.query_samples(window="Session (5h)")
            daily = store.aggregate_samples(samples, "daily")
            # Our data goes up to 87.5% which is "warning"
            bucket = daily[0]
            assert bucket["status"] in ("warning", "critical")

    def test_aggregation_last_raw_value(self, tmp_path):
        """Aggregation should use last() for raw_value."""
        db_path = tmp_path / "test.db"
        with HistoryStore(db_path) as store:
            # Insert samples with different raw_values
            store.record(_make_status(
                windows=[_make_window(utilisation=30.0, raw_value=100.0)],
                timestamp=datetime(2026, 4, 1, 10, 0, 0, tzinfo=timezone.utc),
            ))
            store.record(_make_status(
                windows=[_make_window(utilisation=50.0, raw_value=200.0)],
                timestamp=datetime(2026, 4, 1, 11, 0, 0, tzinfo=timezone.utc),
            ))

            samples = store.query_samples()
            daily = store.aggregate_samples(samples, "daily")
            assert daily[0]["raw_value"] == 200.0  # last

    def test_aggregation_model_usage_max(self, tmp_path):
        """Model usage aggregation should use max() for tokens."""
        db_path = tmp_path / "test.db"
        with HistoryStore(db_path) as store:
            self._populate_db(store)
            rows = store.query_model_usage()
            daily = store.aggregate_model_usage(rows, "daily")
            assert len(daily) >= 1
            # Max total_tokens = 1000 * 24 = 24000
            opus_bucket = [b for b in daily if b["model"] == "claude-opus-4-6"]
            assert opus_bucket[0]["total_tokens"] == 24000

    def test_report_json_format(self, tmp_path, monkeypatch):
        db_path = tmp_path / "data" / "history.db"
        monkeypatch.setenv("LLM_MONITOR_DATA_DIR", str(tmp_path / "data"))

        with HistoryStore(db_path) as store:
            self._populate_db(store)

        runner = CliRunner()
        result = runner.invoke(cli, [
            "history", "report",
            "--format", "json",
            "--days", "30",
        ])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "report" in data
        assert "usage" in data

    def test_report_csv_format(self, tmp_path, monkeypatch):
        db_path = tmp_path / "data" / "history.db"
        monkeypatch.setenv("LLM_MONITOR_DATA_DIR", str(tmp_path / "data"))

        with HistoryStore(db_path) as store:
            self._populate_db(store)

        runner = CliRunner()
        result = runner.invoke(cli, [
            "history", "report",
            "--format", "csv",
            "--days", "30",
        ])
        assert result.exit_code == 0
        # Should have CSV header
        lines = result.output.strip().split("\n")
        assert len(lines) > 1
        assert "provider" in lines[0] or "utilisation" in lines[0]

    def test_report_table_format_cli(self, tmp_path, monkeypatch):
        db_path = tmp_path / "data" / "history.db"
        monkeypatch.setenv("LLM_MONITOR_DATA_DIR", str(tmp_path / "data"))

        with HistoryStore(db_path) as store:
            self._populate_db(store)

        runner = CliRunner()
        result = runner.invoke(cli, [
            "history", "report",
            "--format", "table",
            "--days", "30",
        ])
        assert result.exit_code == 0
        assert "LLM Usage Report" in result.output

    def test_report_with_models(self, tmp_path, monkeypatch):
        db_path = tmp_path / "data" / "history.db"
        monkeypatch.setenv("LLM_MONITOR_DATA_DIR", str(tmp_path / "data"))

        with HistoryStore(db_path) as store:
            self._populate_db(store)

        runner = CliRunner()
        result = runner.invoke(cli, [
            "history", "report",
            "--format", "json",
            "--days", "30",
            "--models",
        ])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "model_usage" in data

    def test_report_empty_db(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LLM_MONITOR_DATA_DIR", str(tmp_path / "data"))

        runner = CliRunner()
        result = runner.invoke(cli, ["history", "report", "--days", "7"])
        assert result.exit_code == 0
        assert "No history data" in result.output

    def test_report_via_status_flag(self, tmp_path, monkeypatch):
        """--report on the default command should work like 'history report'."""
        db_path = tmp_path / "data" / "history.db"
        monkeypatch.setenv("LLM_MONITOR_DATA_DIR", str(tmp_path / "data"))

        with HistoryStore(db_path) as store:
            self._populate_db(store)

        runner = CliRunner()
        result = runner.invoke(cli, [
            "--report", "--format", "json", "--days", "30",
        ])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "report" in data

    def test_report_granularity_choices(self, tmp_path, monkeypatch):
        db_path = tmp_path / "data" / "history.db"
        monkeypatch.setenv("LLM_MONITOR_DATA_DIR", str(tmp_path / "data"))

        with HistoryStore(db_path) as store:
            self._populate_db(store)

        runner = CliRunner()
        for gran in ("raw", "hourly", "daily"):
            result = runner.invoke(cli, [
                "history", "report",
                "--format", "json",
                "--days", "30",
                "--granularity", gran,
            ])
            assert result.exit_code == 0, f"Failed for granularity={gran}"


# ---------------------------------------------------------------------------
# Export formats
# ---------------------------------------------------------------------------


class TestExport:
    def _populate_store(self, store: HistoryStore) -> None:
        store.record(_make_status(
            model_usage=[_make_model_usage()],
            extras={"plan": "Pro"},
        ))

    def test_export_sql(self, tmp_path):
        db_path = tmp_path / "test.db"
        with HistoryStore(db_path) as store:
            self._populate_store(store)
            sql = store.export_sql()

        assert "INSERT INTO usage_samples" in sql
        assert "INSERT INTO model_usage" in sql
        assert "INSERT INTO provider_extras" in sql
        assert "CREATE TABLE" in sql

        # Verify the SQL is valid by executing it
        conn = sqlite3.connect(":memory:")
        conn.executescript(sql)
        count = conn.execute("SELECT COUNT(*) FROM usage_samples").fetchone()[0]
        assert count == 1
        conn.close()

    def test_export_jsonl(self, tmp_path):
        db_path = tmp_path / "test.db"
        with HistoryStore(db_path) as store:
            self._populate_store(store)
            jsonl = store.export_jsonl()

        lines = jsonl.strip().split("\n")
        assert len(lines) >= 2  # At least usage_sample + model_usage

        types_seen = set()
        for line in lines:
            obj = json.loads(line)
            assert "type" in obj
            types_seen.add(obj["type"])

        assert "usage_sample" in types_seen
        assert "model_usage" in types_seen

    def test_export_jsonl_extras(self, tmp_path):
        db_path = tmp_path / "test.db"
        with HistoryStore(db_path) as store:
            self._populate_store(store)
            jsonl = store.export_jsonl()

        lines = jsonl.strip().split("\n")
        extras_lines = [json.loads(l) for l in lines if '"provider_extras"' in l]
        assert len(extras_lines) == 1
        assert "extras" in extras_lines[0]
        assert extras_lines[0]["extras"]["plan"] == "Pro"

    def test_export_csv(self, tmp_path):
        db_path = tmp_path / "test.db"
        with HistoryStore(db_path) as store:
            self._populate_store(store)
            csv_output = store.export_csv()

        # Should have two sections separated by blank line
        sections = csv_output.split("\n\n")
        assert len(sections) >= 2

        # First section is usage_samples
        reader = csv.reader(io.StringIO(sections[0]))
        header = next(reader)
        assert "provider" in header
        assert "utilisation" in header
        rows = list(reader)
        assert len(rows) >= 1

    def test_export_csv_nulls_as_empty(self, tmp_path):
        db_path = tmp_path / "test.db"
        with HistoryStore(db_path) as store:
            store.record(_make_status(
                windows=[_make_window(raw_value=None, raw_limit=None)],
            ))
            csv_output = store.export_csv()

        # raw_value and raw_limit should be empty strings
        assert ",," in csv_output  # Adjacent commas indicate empty fields

    def test_export_cli_sql(self, tmp_path, monkeypatch):
        db_path = tmp_path / "data" / "history.db"
        monkeypatch.setenv("LLM_MONITOR_DATA_DIR", str(tmp_path / "data"))

        with HistoryStore(db_path) as store:
            self._populate_store(store)

        runner = CliRunner()
        result = runner.invoke(cli, ["history", "export", "--format", "sql"])
        assert result.exit_code == 0
        assert "INSERT INTO" in result.output

    def test_export_cli_jsonl(self, tmp_path, monkeypatch):
        db_path = tmp_path / "data" / "history.db"
        monkeypatch.setenv("LLM_MONITOR_DATA_DIR", str(tmp_path / "data"))

        with HistoryStore(db_path) as store:
            self._populate_store(store)

        runner = CliRunner()
        result = runner.invoke(cli, ["history", "export", "--format", "jsonl"])
        assert result.exit_code == 0
        for line in result.output.strip().split("\n"):
            json.loads(line)  # Each line should be valid JSON

    def test_export_cli_csv(self, tmp_path, monkeypatch):
        db_path = tmp_path / "data" / "history.db"
        monkeypatch.setenv("LLM_MONITOR_DATA_DIR", str(tmp_path / "data"))

        with HistoryStore(db_path) as store:
            self._populate_store(store)

        runner = CliRunner()
        result = runner.invoke(cli, ["history", "export", "--format", "csv"])
        assert result.exit_code == 0
        assert "provider" in result.output


# ---------------------------------------------------------------------------
# Concurrent read/write (WAL mode)
# ---------------------------------------------------------------------------


class TestConcurrency:
    def test_concurrent_read_during_write(self, tmp_path):
        """WAL mode should allow reads while a write is in progress."""
        db_path = tmp_path / "test.db"

        # Populate the DB first
        with HistoryStore(db_path) as store:
            store.record(_make_status())

        results = {"read_ok": False, "write_ok": False}

        def writer():
            with HistoryStore(db_path) as store:
                for i in range(10):
                    store.record(_make_status(
                        windows=[_make_window(utilisation=float(i * 10))],
                        timestamp=datetime(2026, 4, 1, i, 0, 0, tzinfo=timezone.utc),
                    ))
                results["write_ok"] = True

        def reader():
            time.sleep(0.01)  # Brief delay to let writer start
            with HistoryStore(db_path) as store:
                count = store.conn.execute(
                    "SELECT COUNT(*) FROM usage_samples"
                ).fetchone()[0]
                results["read_ok"] = count >= 0  # Just needs to not block

        t_write = threading.Thread(target=writer)
        t_read = threading.Thread(target=reader)
        t_write.start()
        t_read.start()
        t_write.join(timeout=10)
        t_read.join(timeout=10)

        assert results["write_ok"]
        assert results["read_ok"]


# ---------------------------------------------------------------------------
# CLI integration: history recording during fetch
# ---------------------------------------------------------------------------


class TestHistoryIntegration:
    @pytest.fixture
    def _setup(self, tmp_path, monkeypatch):
        """Set up environment for history integration tests."""
        import json as json_mod

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        monkeypatch.setenv("LLM_MONITOR_CACHE_DIR", str(cache_dir))
        monkeypatch.setenv("LLM_MONITOR_DATA_DIR", str(tmp_path / "data"))

        creds_path = tmp_path / "claude" / ".credentials.json"
        creds_path.parent.mkdir(parents=True, exist_ok=True)
        future = datetime.now(timezone.utc) + timedelta(hours=2)
        creds_path.write_text(json_mod.dumps({
            "claudeAiOauth": {
                "accessToken": "sk-ant-oat01-test-token",
                "refreshToken": "sk-ant-ort01-test-refresh",
                "expiresAt": future.isoformat(),
            }
        }))

        config_path = tmp_path / "config.toml"
        config_path.write_text(
            f'[providers.claude]\n'
            f'enabled = true\n'
            f'credentials_path = "{creds_path}"\n'
        )
        import os
        os.chmod(str(config_path), 0o600)
        monkeypatch.setenv("LLM_MONITOR_CONFIG", str(config_path))

        return tmp_path

    @respx.mock
    def test_fetch_records_history(self, _setup, tmp_path):
        """A successful fetch should write to the history database."""
        import respx as respx_mod
        usage_resp = json.loads(
            (Path(__file__).parent / "fixtures" / "claude_usage_response.json").read_text()
        )
        respx_mod.get("https://api.anthropic.com/api/oauth/usage").respond(
            200, json=usage_resp
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["--provider", "claude", "--fresh"])
        assert result.exit_code == 0

        db_path = tmp_path / "data" / "history.db"
        assert db_path.exists()

        with HistoryStore(db_path) as store:
            count = store.conn.execute(
                "SELECT COUNT(*) FROM usage_samples"
            ).fetchone()[0]
            assert count > 0

    @respx.mock
    def test_no_history_flag_skips(self, _setup, tmp_path):
        """--no-history should prevent history recording."""
        import respx as respx_mod
        usage_resp = json.loads(
            (Path(__file__).parent / "fixtures" / "claude_usage_response.json").read_text()
        )
        respx_mod.get("https://api.anthropic.com/api/oauth/usage").respond(
            200, json=usage_resp
        )

        runner = CliRunner()
        result = runner.invoke(
            cli, ["--provider", "claude", "--fresh", "--no-history"]
        )
        assert result.exit_code == 0

        db_path = tmp_path / "data" / "history.db"
        assert not db_path.exists()


