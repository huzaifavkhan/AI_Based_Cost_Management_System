# SQLite database layer — invoices, audit_log, feedback tables
from __future__ import annotations
import sqlite3
import json
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "invoices.db"


def _get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create all tables if they don't exist."""
    with _get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS invoices (
                id              TEXT PRIMARY KEY,
                filename        TEXT,
                uploaded_at     TEXT,
                overall_status  TEXT,
                invoice_number  TEXT,
                vendor          TEXT,
                date            TEXT,
                po_number       TEXT,
                total_amount    REAL,
                match_status    TEXT,
                match_score     INTEGER,
                rule_flags      TEXT,
                is_outlier      INTEGER,
                anomaly_score   REAL,
                extraction_method TEXT,
                raw_json        TEXT
            );

            CREATE TABLE IF NOT EXISTS audit_log (
                log_id      INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp   TEXT,
                invoice_id  TEXT,
                action      TEXT,
                user        TEXT,
                details     TEXT
            );

            CREATE TABLE IF NOT EXISTS feedback (
                fb_id           INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp       TEXT,
                invoice_id      TEXT,
                field_name      TEXT,
                corrected_value TEXT,
                user            TEXT
            );

            CREATE TABLE IF NOT EXISTS fl_training_runs (
                run_id          TEXT PRIMARY KEY,
                started_at      TEXT NOT NULL,
                completed_at    TEXT,
                status          TEXT NOT NULL,
                n_rounds        INTEGER,
                sigma           REAL,
                clip_norm       REAL,
                local_epochs    INTEGER,
                final_f1        REAL,
                final_accuracy  REAL,
                epsilon         REAL,
                delta           REAL,
                training_time_s REAL,
                client_a_size   INTEGER,
                client_b_size   INTEGER,
                client_c_size   INTEGER,
                error_message   TEXT
            );

            CREATE TABLE IF NOT EXISTS fl_round_metrics (
                metric_id       INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id          TEXT NOT NULL,
                round_number    INTEGER NOT NULL,
                f1_score        REAL,
                accuracy        REAL,
                coef_change     REAL,
                epsilon_so_far  REAL,
                client_a_loss   REAL,
                client_b_loss   REAL,
                client_c_loss   REAL,
                timestamp       TEXT,
                FOREIGN KEY (run_id) REFERENCES fl_training_runs(run_id)
            );
        """)


def insert_invoice(data: dict) -> None:
    fields = data.get("extracted_fields", {})
    match  = data.get("match_result", {})
    anomaly= data.get("anomaly_result", {})

    with _get_conn() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO invoices VALUES (
                :id, :filename, :uploaded_at, :overall_status,
                :invoice_number, :vendor, :date, :po_number, :total_amount,
                :match_status, :match_score,
                :rule_flags, :is_outlier, :anomaly_score,
                :extraction_method, :raw_json
            )
        """, {
            "id":               data["invoice_id"],
            "filename":         data["filename"],
            "uploaded_at":      datetime.utcnow().isoformat(),
            "overall_status":   data["overall_status"],
            "invoice_number":   fields.get("invoice_number"),
            "vendor":           fields.get("vendor"),
            "date":             fields.get("date"),
            "po_number":        fields.get("po_number"),
            "total_amount":     fields.get("total_amount"),
            "match_status":     match.get("status"),
            "match_score":      match.get("match_score"),
            "rule_flags":       json.dumps(anomaly.get("rule_flags", [])),
            "is_outlier":       int(anomaly.get("is_statistical_outlier", False)),
            "anomaly_score":    anomaly.get("anomaly_score", 0.0),
            "extraction_method": data.get("extraction_method", "unknown"),
            "raw_json":         json.dumps(data),
        })
    insert_audit_log(data["invoice_id"], "processed", "system",
                     f"Status: {data['overall_status']}")


def get_all_invoices() -> list[dict]:
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM invoices ORDER BY uploaded_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def insert_audit_log(invoice_id: str, action: str, user: str, details: str) -> None:
    with _get_conn() as conn:
        conn.execute(
            "INSERT INTO audit_log (timestamp, invoice_id, action, user, details) VALUES (?,?,?,?,?)",
            (datetime.utcnow().isoformat(), invoice_id, action, user, details)
        )


def get_audit_log() -> list[dict]:
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM audit_log ORDER BY timestamp DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def insert_feedback(invoice_id: str, field_name: str, corrected_value: str, user: str) -> None:
    with _get_conn() as conn:
        conn.execute(
            "INSERT INTO feedback (timestamp, invoice_id, field_name, corrected_value, user) VALUES (?,?,?,?,?)",
            (datetime.utcnow().isoformat(), invoice_id, field_name, corrected_value, user)
        )
    insert_audit_log(invoice_id, "user_correction", user,
                     f"Field '{field_name}' corrected to '{corrected_value}'")


def insert_fl_run(data: dict) -> None:
    with _get_conn() as conn:
        conn.execute("""
            INSERT INTO fl_training_runs
                (run_id, started_at, status, n_rounds, sigma, clip_norm, local_epochs, delta,
                 client_a_size, client_b_size, client_c_size)
            VALUES (:run_id, :started_at, :status, :n_rounds, :sigma, :clip_norm, :local_epochs,
                    :delta, :client_a_size, :client_b_size, :client_c_size)
        """, data)


def update_fl_run(run_id: str, status: str, completed_at: str, metrics: dict) -> None:
    with _get_conn() as conn:
        conn.execute("""
            UPDATE fl_training_runs SET
                status=:status, completed_at=:completed_at,
                final_f1=:final_f1, final_accuracy=:final_accuracy,
                epsilon=:epsilon, training_time_s=:training_time_s,
                error_message=:error_message
            WHERE run_id=:run_id
        """, {
            "run_id": run_id, "status": status, "completed_at": completed_at,
            "final_f1": metrics.get("final_f1"), "final_accuracy": metrics.get("final_accuracy"),
            "epsilon": metrics.get("epsilon"), "training_time_s": metrics.get("training_time_s"),
            "error_message": metrics.get("error_message"),
        })


def insert_fl_round_metric(run_id: str, round_num: int, metrics: dict) -> None:
    with _get_conn() as conn:
        conn.execute("""
            INSERT INTO fl_round_metrics
                (run_id, round_number, f1_score, accuracy, coef_change,
                 epsilon_so_far, client_a_loss, client_b_loss, client_c_loss, timestamp)
            VALUES (?,?,?,?,?,?,?,?,?,?)
        """, (
            run_id, round_num,
            metrics.get("f1_score"), metrics.get("accuracy"), metrics.get("coef_change"),
            metrics.get("epsilon_so_far"), metrics.get("client_a_loss"),
            metrics.get("client_b_loss"), metrics.get("client_c_loss"),
            datetime.utcnow().isoformat(),
        ))


def get_fl_runs() -> list[dict]:
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM fl_training_runs ORDER BY started_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def get_fl_round_metrics(run_id: str) -> list[dict]:
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM fl_round_metrics WHERE run_id=? ORDER BY round_number ASC",
            (run_id,)
        ).fetchall()
    return [dict(r) for r in rows]


def get_latest_fl_run() -> dict | None:
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM fl_training_runs ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
    return dict(row) if row else None


def get_dashboard_stats() -> dict:
    with _get_conn() as conn:
        total    = conn.execute("SELECT COUNT(*) FROM invoices").fetchone()[0]
        verified = conn.execute("SELECT COUNT(*) FROM invoices WHERE overall_status='VERIFIED'").fetchone()[0]
        flagged  = conn.execute("SELECT COUNT(*) FROM invoices WHERE overall_status='FLAGGED'").fetchone()[0]
        unmatched= conn.execute("SELECT COUNT(*) FROM invoices WHERE overall_status='UNMATCHED'").fetchone()[0]

        # Estimated savings = sum of all amount mismatches (approximate via flagged totals)
        savings_row = conn.execute(
            "SELECT SUM(total_amount) FROM invoices WHERE overall_status='FLAGGED' AND total_amount IS NOT NULL"
        ).fetchone()[0]
        savings = float(savings_row or 0) * 0.05  # Estimate 5% recovery

        # Top anomaly types from rule_flags
        rows = conn.execute("SELECT rule_flags FROM invoices WHERE rule_flags != '[]'").fetchall()

    flag_counts: dict[str, int] = {}
    for row in rows:
        flags = json.loads(row[0] or "[]")
        for f in flags:
            key = f.split(":")[0].strip()
            flag_counts[key] = flag_counts.get(key, 0) + 1

    top_types = sorted(
        [{"type": k, "count": v} for k, v in flag_counts.items()],
        key=lambda x: x["count"], reverse=True
    )[:5]

    return {
        "total_processed":   total,
        "verified_count":    verified,
        "flagged_count":     flagged,
        "unmatched_count":   unmatched,
        "estimated_savings": round(savings, 2),
        "top_anomaly_types": top_types,
    }
