"""SQLite history store for llm-monitor.

Stores usage samples over time for trend analysis and reporting.
See SPEC.md Section 6 for the full history specification.
"""

from __future__ import annotations

import csv
import io
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from llm_monitor.config import get_data_dir
from llm_monitor.models import ModelUsage, ProviderStatus, UsageWindow
from llm_monitor.security import secure_mkdir

SCHEMA_VERSION = 1

# Status severity ordering for max-severity aggregation
_STATUS_SEVERITY = {"normal": 0, "warning": 1, "critical": 2, "exceeded": 3}
_SEVERITY_STATUS = {v: k for k, v in _STATUS_SEVERITY.items()}

# Meaningful-change detection thresholds
_UTILISATION_DELTA = 0.1  # absolute percentage change


_SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS usage_samples (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    provider      TEXT NOT NULL,
    timestamp     TEXT NOT NULL,
    window_name   TEXT NOT NULL,
    utilisation   REAL,
    status        TEXT,
    unit          TEXT NOT NULL,
    raw_value     REAL,
    raw_limit     REAL,
    resets_at     TEXT,
    cached        INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_samples_provider_time
    ON usage_samples(provider, timestamp);
CREATE INDEX IF NOT EXISTS idx_samples_time
    ON usage_samples(timestamp);

CREATE TABLE IF NOT EXISTS model_usage (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    provider      TEXT NOT NULL,
    timestamp     TEXT NOT NULL,
    model         TEXT NOT NULL,
    input_tokens  INTEGER,
    output_tokens INTEGER,
    total_tokens  INTEGER,
    cost          REAL,
    request_count INTEGER,
    period        TEXT
);

CREATE INDEX IF NOT EXISTS idx_model_usage_provider_time
    ON model_usage(provider, timestamp);
CREATE INDEX IF NOT EXISTS idx_model_usage_model
    ON model_usage(model);

CREATE TABLE IF NOT EXISTS provider_extras (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    provider      TEXT NOT NULL,
    timestamp     TEXT NOT NULL,
    extras_json   TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_extras_provider_time
    ON provider_extras(provider, timestamp);

CREATE TABLE IF NOT EXISTS schema_version (
    version       INTEGER NOT NULL
);
"""


class HistoryStore:
    """SQLite-backed usage history store.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file.  When *None*, uses the default
        XDG data directory location.
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        if db_path is None:
            data_dir = get_data_dir()
            secure_mkdir(str(data_dir))
            self.db_path = data_dir / "history.db"
        else:
            self.db_path = Path(db_path)

        self._conn: sqlite3.Connection | None = None
        # In-memory last-known state for meaningful-change detection.
        # Key: (provider, window_name) -> dict with utilisation, status, resets_at
        self._last_known: dict[tuple[str, str], dict[str, Any]] = {}

    def open(self) -> None:
        """Open the database, create schema if needed, and load last-known state."""
        is_new = not self.db_path.exists()

        if is_new:
            # Create file with secure permissions
            parent = str(self.db_path.parent)
            secure_mkdir(parent)
            fd = os.open(str(self.db_path), os.O_WRONLY | os.O_CREAT, 0o600)
            os.close(fd)

        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row

        if is_new:
            # auto_vacuum must be set before any tables are created
            self._conn.execute("PRAGMA auto_vacuum=INCREMENTAL")

        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA_SQL)

        # Set schema version if new database
        row = self._conn.execute("SELECT COUNT(*) FROM schema_version").fetchone()
        if row[0] == 0:
            self._conn.execute(
                "INSERT INTO schema_version (version) VALUES (?)",
                (SCHEMA_VERSION,),
            )
            self._conn.commit()

        self._load_last_known()

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> HistoryStore:
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("HistoryStore is not open")
        return self._conn

    # ------------------------------------------------------------------
    # Last-known state for meaningful-change detection
    # ------------------------------------------------------------------

    def _load_last_known(self) -> None:
        """Load the most recent row per provider+window into memory."""
        self._last_known.clear()
        rows = self.conn.execute(
            """
            SELECT provider, window_name, utilisation, status, resets_at
            FROM usage_samples
            WHERE id IN (
                SELECT MAX(id) FROM usage_samples
                GROUP BY provider, window_name
            )
            """
        ).fetchall()
        for row in rows:
            key = (row["provider"], row["window_name"])
            self._last_known[key] = {
                "utilisation": row["utilisation"],
                "status": row["status"],
                "resets_at": row["resets_at"],
            }

    def _has_meaningful_change(
        self, provider: str, window: UsageWindow
    ) -> bool:
        """Check if a window differs meaningfully from the last-known state."""
        key = (provider, window.name)
        last = self._last_known.get(key)

        # First sample for this provider+window — always meaningful
        if last is None:
            return True

        # Utilisation delta > 0.1%
        last_util = last.get("utilisation")
        if last_util is not None and window.utilisation is not None:
            if abs(window.utilisation - last_util) > _UTILISATION_DELTA:
                return True
        elif last_util != window.utilisation:
            # One is None, the other isn't
            return True

        # Status changed
        if last.get("status") != window.status:
            return True

        # Window reset detected (resets_at changed to a later time)
        last_reset = last.get("resets_at")
        current_reset = (
            window.resets_at.isoformat() if window.resets_at else None
        )
        if last_reset != current_reset:
            return True

        return False

    def _update_last_known(
        self, provider: str, window: UsageWindow
    ) -> None:
        """Update in-memory last-known state after a write."""
        key = (provider, window.name)
        self._last_known[key] = {
            "utilisation": window.utilisation,
            "status": window.status,
            "resets_at": (
                window.resets_at.isoformat() if window.resets_at else None
            ),
        }

    # ------------------------------------------------------------------
    # Writing history
    # ------------------------------------------------------------------

    def record(self, status: ProviderStatus) -> int:
        """Record a provider fetch to history if data changed meaningfully.

        Cached responses with no changes are skipped. Returns the number
        of rows written.
        """
        if status.errors:
            return 0

        provider = status.provider_name
        ts = status.timestamp.isoformat()
        rows_written = 0

        # Determine which windows have meaningful changes
        windows_to_write = []
        for window in status.windows:
            if status.cached and not self._has_meaningful_change(provider, window):
                continue
            if not status.cached and not self._has_meaningful_change(provider, window):
                continue
            windows_to_write.append(window)

        if not windows_to_write and not status.model_usage:
            return 0

        with self.conn:
            # Write usage_samples
            for window in windows_to_write:
                self.conn.execute(
                    """
                    INSERT INTO usage_samples
                        (provider, timestamp, window_name, utilisation,
                         status, unit, raw_value, raw_limit, resets_at, cached)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        provider,
                        ts,
                        window.name,
                        window.utilisation,
                        window.status,
                        window.unit,
                        window.raw_value,
                        window.raw_limit,
                        window.resets_at.isoformat() if window.resets_at else None,
                        1 if status.cached else 0,
                    ),
                )
                self._update_last_known(provider, window)
                rows_written += 1

            # Write model_usage (only if we wrote windows too)
            if windows_to_write:
                for mu in status.model_usage:
                    self.conn.execute(
                        """
                        INSERT INTO model_usage
                            (provider, timestamp, model, input_tokens,
                             output_tokens, total_tokens, cost,
                             request_count, period)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            provider,
                            ts,
                            mu.model,
                            mu.input_tokens,
                            mu.output_tokens,
                            mu.total_tokens,
                            mu.cost,
                            mu.request_count,
                            mu.period,
                        ),
                    )
                    rows_written += 1

                # Write provider_extras
                if status.extras:
                    # Filter out internal keys (starting with _)
                    public_extras = {
                        k: v for k, v in status.extras.items()
                        if not k.startswith("_")
                    }
                    if public_extras:
                        self.conn.execute(
                            """
                            INSERT INTO provider_extras
                                (provider, timestamp, extras_json)
                            VALUES (?, ?, ?)
                            """,
                            (provider, ts, json.dumps(public_extras)),
                        )
                        rows_written += 1

        return rows_written

    # ------------------------------------------------------------------
    # Retention pruning
    # ------------------------------------------------------------------

    def prune(self, retention_days: int = 90) -> int:
        """Delete history older than *retention_days*. Returns rows deleted."""
        cutoff = f"-{retention_days} days"
        total = 0
        with self.conn:
            for table in ("usage_samples", "model_usage", "provider_extras"):
                cur = self.conn.execute(
                    f"DELETE FROM {table} WHERE timestamp < datetime('now', ?)",
                    (cutoff,),
                )
                total += cur.rowcount
        return total

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        """Return summary statistics about the history database."""
        db_size = self.db_path.stat().st_size if self.db_path.exists() else 0

        sample_count = self.conn.execute(
            "SELECT COUNT(*) FROM usage_samples"
        ).fetchone()[0]

        model_count = self.conn.execute(
            "SELECT COUNT(*) FROM model_usage"
        ).fetchone()[0]

        providers_row = self.conn.execute(
            "SELECT DISTINCT provider FROM usage_samples ORDER BY provider"
        ).fetchall()
        providers = [r[0] for r in providers_row]

        oldest = self.conn.execute(
            "SELECT MIN(timestamp) FROM usage_samples"
        ).fetchone()[0]

        newest = self.conn.execute(
            "SELECT MAX(timestamp) FROM usage_samples"
        ).fetchone()[0]

        return {
            "db_path": str(self.db_path),
            "db_size": db_size,
            "sample_count": sample_count,
            "model_count": model_count,
            "providers": providers,
            "oldest": oldest,
            "newest": newest,
        }

    # ------------------------------------------------------------------
    # Purge
    # ------------------------------------------------------------------

    def purge(self) -> int:
        """Delete ALL history data. Returns total rows deleted."""
        total = 0
        with self.conn:
            for table in ("usage_samples", "model_usage", "provider_extras"):
                cur = self.conn.execute(f"DELETE FROM {table}")
                total += cur.rowcount
        return total

    # ------------------------------------------------------------------
    # Querying for reports
    # ------------------------------------------------------------------

    def query_samples(
        self,
        *,
        provider: str | None = None,
        window: str | None = None,
        from_dt: datetime | None = None,
        to_dt: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Query usage_samples with optional filters."""
        conditions: list[str] = []
        params: list[Any] = []

        if provider:
            conditions.append("provider = ?")
            params.append(provider)
        if window:
            conditions.append("window_name = ?")
            params.append(window)
        if from_dt:
            conditions.append("timestamp >= ?")
            params.append(from_dt.isoformat())
        if to_dt:
            conditions.append("timestamp <= ?")
            params.append(to_dt.isoformat())

        where = " AND ".join(conditions) if conditions else "1=1"
        rows = self.conn.execute(
            f"SELECT * FROM usage_samples WHERE {where} ORDER BY timestamp",
            params,
        ).fetchall()
        return [dict(r) for r in rows]

    def query_model_usage(
        self,
        *,
        provider: str | None = None,
        from_dt: datetime | None = None,
        to_dt: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Query model_usage with optional filters."""
        conditions: list[str] = []
        params: list[Any] = []

        if provider:
            conditions.append("provider = ?")
            params.append(provider)
        if from_dt:
            conditions.append("timestamp >= ?")
            params.append(from_dt.isoformat())
        if to_dt:
            conditions.append("timestamp <= ?")
            params.append(to_dt.isoformat())

        where = " AND ".join(conditions) if conditions else "1=1"
        rows = self.conn.execute(
            f"SELECT * FROM model_usage WHERE {where} ORDER BY timestamp",
            params,
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Aggregation for reports
    # ------------------------------------------------------------------

    def aggregate_samples(
        self,
        samples: list[dict[str, Any]],
        granularity: str = "daily",
    ) -> list[dict[str, Any]]:
        """Aggregate raw samples into time buckets.

        Parameters
        ----------
        samples:
            Raw sample dicts from :meth:`query_samples`.
        granularity:
            ``"raw"``, ``"hourly"``, or ``"daily"``.

        Returns
        -------
        list[dict]
            Aggregated buckets with computed fields plus ``sample_count``,
            ``bucket_start``, and ``bucket_end``.
        """
        if granularity == "raw":
            return samples

        # Group by (provider, window_name, bucket_key)
        buckets: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
        for sample in samples:
            ts = sample["timestamp"]
            key = _bucket_key(ts, granularity)
            group = (sample["provider"], sample["window_name"], key)
            buckets.setdefault(group, []).append(sample)

        result: list[dict[str, Any]] = []
        for (provider, window_name, bkey), bucket_samples in sorted(buckets.items()):
            agg = _aggregate_bucket(bucket_samples, granularity)
            agg["provider"] = provider
            agg["window_name"] = window_name
            result.append(agg)

        return result

    def aggregate_model_usage(
        self,
        rows: list[dict[str, Any]],
        granularity: str = "daily",
    ) -> list[dict[str, Any]]:
        """Aggregate model_usage rows into time buckets.

        Uses max() for all numeric fields (running totals).
        """
        if granularity == "raw":
            return rows

        buckets: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
        for row in rows:
            ts = row["timestamp"]
            key = _bucket_key(ts, granularity)
            group = (row["provider"], row["model"], key)
            buckets.setdefault(group, []).append(row)

        result: list[dict[str, Any]] = []
        for (provider, model, bkey), bucket_rows in sorted(buckets.items()):
            agg = _aggregate_model_bucket(bucket_rows, granularity)
            agg["provider"] = provider
            agg["model"] = model
            result.append(agg)

        return result

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_sql(self) -> str:
        """Export all history data as SQL INSERT statements."""
        lines: list[str] = []
        lines.append("-- llm-monitor history export (SQL)")
        lines.append(f"-- Exported: {datetime.now(timezone.utc).isoformat()}")
        lines.append("")
        lines.append(_SCHEMA_SQL)
        lines.append("")

        # usage_samples
        rows = self.conn.execute(
            "SELECT * FROM usage_samples ORDER BY id"
        ).fetchall()
        for row in rows:
            vals = _sql_values(row)
            lines.append(
                f"INSERT INTO usage_samples "
                f"(id, provider, timestamp, window_name, utilisation, "
                f"status, unit, raw_value, raw_limit, resets_at, cached) "
                f"VALUES ({vals});"
            )

        lines.append("")

        # model_usage
        rows = self.conn.execute(
            "SELECT * FROM model_usage ORDER BY id"
        ).fetchall()
        for row in rows:
            vals = _sql_values(row)
            lines.append(
                f"INSERT INTO model_usage "
                f"(id, provider, timestamp, model, input_tokens, "
                f"output_tokens, total_tokens, cost, request_count, period) "
                f"VALUES ({vals});"
            )

        lines.append("")

        # provider_extras
        rows = self.conn.execute(
            "SELECT * FROM provider_extras ORDER BY id"
        ).fetchall()
        for row in rows:
            vals = _sql_values(row)
            lines.append(
                f"INSERT INTO provider_extras "
                f"(id, provider, timestamp, extras_json) "
                f"VALUES ({vals});"
            )

        return "\n".join(lines) + "\n"

    def export_jsonl(self) -> str:
        """Export all history data as JSONL (one JSON object per line)."""
        lines: list[str] = []

        rows = self.conn.execute(
            "SELECT * FROM usage_samples ORDER BY id"
        ).fetchall()
        for row in rows:
            d = dict(row)
            d["type"] = "usage_sample"
            d["cached"] = bool(d["cached"])
            # Reorder so type is first
            ordered = {"type": d.pop("type")}
            ordered.update(d)
            lines.append(json.dumps(ordered))

        rows = self.conn.execute(
            "SELECT * FROM model_usage ORDER BY id"
        ).fetchall()
        for row in rows:
            d = dict(row)
            d["type"] = "model_usage"
            ordered = {"type": d.pop("type")}
            ordered.update(d)
            lines.append(json.dumps(ordered))

        rows = self.conn.execute(
            "SELECT * FROM provider_extras ORDER BY id"
        ).fetchall()
        for row in rows:
            d = dict(row)
            d["type"] = "provider_extras"
            extras = json.loads(d.pop("extras_json"))
            ordered = {"type": d.pop("type")}
            ordered.update(d)
            ordered["extras"] = extras
            lines.append(json.dumps(ordered))

        return "\n".join(lines) + "\n" if lines else ""

    def export_csv(self) -> str:
        """Export usage_samples and model_usage as CSV with two sections."""
        output = io.StringIO()

        # Section 1: usage_samples
        sample_cols = [
            "id", "provider", "timestamp", "window_name", "utilisation",
            "status", "unit", "raw_value", "raw_limit", "resets_at", "cached",
        ]
        writer = csv.writer(output)
        writer.writerow(sample_cols)
        rows = self.conn.execute(
            "SELECT * FROM usage_samples ORDER BY id"
        ).fetchall()
        for row in rows:
            writer.writerow([
                row[col] if row[col] is not None else "" for col in sample_cols
            ])

        # Blank line separator
        output.write("\n")

        # Section 2: model_usage
        model_cols = [
            "id", "provider", "timestamp", "model", "input_tokens",
            "output_tokens", "total_tokens", "cost", "request_count", "period",
        ]
        writer.writerow(model_cols)
        rows = self.conn.execute(
            "SELECT * FROM model_usage ORDER BY id"
        ).fetchall()
        for row in rows:
            writer.writerow([
                row[col] if row[col] is not None else "" for col in model_cols
            ])

        return output.getvalue()

    # ------------------------------------------------------------------
    # Prune count (for stats display)
    # ------------------------------------------------------------------

    def prune_count(self, retention_days: int = 90) -> int:
        """Count rows that would be pruned without deleting them."""
        cutoff = f"-{retention_days} days"
        total = 0
        for table in ("usage_samples", "model_usage", "provider_extras"):
            row = self.conn.execute(
                f"SELECT COUNT(*) FROM {table} WHERE timestamp < datetime('now', ?)",
                (cutoff,),
            ).fetchone()
            total += row[0]
        return total


# ======================================================================
# Module-level helpers
# ======================================================================


def _bucket_key(ts: str, granularity: str) -> str:
    """Derive a bucket key from an ISO 8601 timestamp string."""
    # Parse just enough of the timestamp
    if granularity == "hourly":
        return ts[:13]  # "2026-04-01T10"
    elif granularity == "daily":
        return ts[:10]  # "2026-04-01"
    return ts


def _bucket_boundaries(key: str, granularity: str) -> tuple[str, str]:
    """Return (start, end) ISO 8601 strings for a bucket key."""
    if granularity == "hourly":
        start = key + ":00:00"
        end = key + ":59:59"
    elif granularity == "daily":
        start = key + "T00:00:00"
        end = key + "T23:59:59"
    else:
        return key, key
    return start, end


def _aggregate_bucket(
    samples: list[dict[str, Any]], granularity: str
) -> dict[str, Any]:
    """Aggregate a bucket of samples according to the spec algorithms."""
    # Use the first sample as a template
    first = samples[0]
    bkey = _bucket_key(first["timestamp"], granularity)
    bucket_start, bucket_end = _bucket_boundaries(bkey, granularity)

    # mean(utilisation)
    utils = [s["utilisation"] for s in samples if s["utilisation"] is not None]
    avg_util = sum(utils) / len(utils) if utils else None

    # max-severity(status)
    severities = [
        _STATUS_SEVERITY.get(s["status"], 0)
        for s in samples
        if s["status"] is not None
    ]
    max_status = _SEVERITY_STATUS.get(max(severities), "normal") if severities else None

    # last(raw_value, raw_limit, resets_at)
    last = samples[-1]

    return {
        "bucket_start": bucket_start,
        "bucket_end": bucket_end,
        "utilisation": round(avg_util, 2) if avg_util is not None else None,
        "status": max_status,
        "unit": first["unit"],
        "raw_value": last["raw_value"],
        "raw_limit": last["raw_limit"],
        "resets_at": last["resets_at"],
        "sample_count": len(samples),
    }


def _aggregate_model_bucket(
    rows: list[dict[str, Any]], granularity: str
) -> dict[str, Any]:
    """Aggregate a bucket of model_usage rows using max()."""
    first = rows[0]
    bkey = _bucket_key(first["timestamp"], granularity)
    bucket_start, bucket_end = _bucket_boundaries(bkey, granularity)

    def _max_field(field: str) -> Any:
        vals = [r[field] for r in rows if r[field] is not None]
        return max(vals) if vals else None

    return {
        "bucket_start": bucket_start,
        "bucket_end": bucket_end,
        "input_tokens": _max_field("input_tokens"),
        "output_tokens": _max_field("output_tokens"),
        "total_tokens": _max_field("total_tokens"),
        "cost": _max_field("cost"),
        "request_count": _max_field("request_count"),
        "period": rows[-1].get("period"),
        "sample_count": len(rows),
    }


def _sql_values(row: sqlite3.Row) -> str:
    """Format a Row as a comma-separated SQL VALUES string."""
    parts: list[str] = []
    for key in row.keys():
        val = row[key]
        if val is None:
            parts.append("NULL")
        elif isinstance(val, (int, float)):
            parts.append(str(val))
        else:
            escaped = str(val).replace("'", "''")
            parts.append(f"'{escaped}'")
    return ", ".join(parts)
