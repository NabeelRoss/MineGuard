"""
SQLite Database Layer for MineGuard AI.

Implements checkpoint inspection records, violation audit logging,
monitoring session tracking, system settings persistence, CSV exports,
and safe history management using parameterized SQL.
Thread-safe and WAL enabled.
"""

import io
import csv
import sqlite3
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from config import AppConfig
from inspection.inspection_result import InspectionResult

DB_PATH = AppConfig.DATABASE_PATH


def get_db_connection() -> sqlite3.Connection:
    """Creates a thread-safe connection to the SQLite database with row dict factory."""
    db_path = AppConfig.DATABASE_PATH
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=15.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")  # Write-Ahead Logging for high concurrency
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    """Initializes SQLite schema for Checkpoint Inspections and System Settings."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Check for and upgrade legacy inspections table if schema differs
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='inspections'")
    if cursor.fetchone():
        cursor.execute("PRAGMA table_info(inspections)")
        existing_cols = [row[1] for row in cursor.fetchall()]
        if "inspection_reference" not in existing_cols:
            cursor.execute("DROP TABLE IF EXISTS violations")
            cursor.execute("DROP TABLE IF EXISTS inspections")
            cursor.execute("DROP TABLE IF EXISTS monitoring_sessions")

    # 1. Monitoring Sessions Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS monitoring_sessions (
            session_id INTEGER PRIMARY KEY AUTOINCREMENT,
            start_time TIMESTAMP NOT NULL,
            end_time TIMESTAMP,
            camera_source TEXT,
            total_inspections INTEGER DEFAULT 0,
            total_pass INTEGER DEFAULT 0,
            total_fail INTEGER DEFAULT 0,
            total_reinspect INTEGER DEFAULT 0,
            status TEXT DEFAULT 'ACTIVE'
        )
    """)

    # 2. Checkpoint Inspections Table (Core quality inspection record per worker visit)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS inspections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            inspection_reference TEXT UNIQUE NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            completion_timestamp TIMESTAMP NOT NULL,
            helmet_status TEXT NOT NULL,
            vest_status TEXT NOT NULL,
            result TEXT NOT NULL,
            person_confidence REAL,
            helmet_confidence REAL,
            vest_confidence REAL,
            average_confidence REAL,
            inspection_duration REAL,
            reinspection_reason TEXT,
            missing_ppe TEXT,
            original_image_path TEXT,
            annotated_image_path TEXT,
            session_id INTEGER,
            total_frames_analyzed INTEGER DEFAULT 0,
            FOREIGN KEY (session_id) REFERENCES monitoring_sessions(session_id) ON DELETE SET NULL
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_insp_ref ON inspections(inspection_reference)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_insp_time ON inspections(timestamp)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_insp_result ON inspections(result)")

    # 3. Confirmed Violations Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS violations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            inspection_reference TEXT UNIQUE NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            missing_ppe TEXT NOT NULL,
            confidence REAL NOT NULL,
            original_image_path TEXT,
            annotated_image_path TEXT,
            acknowledgement_status TEXT DEFAULT 'UNACKNOWLEDGED',
            session_id INTEGER,
            FOREIGN KEY (inspection_reference) REFERENCES inspections(inspection_reference) ON DELETE CASCADE,
            FOREIGN KEY (session_id) REFERENCES monitoring_sessions(session_id) ON DELETE SET NULL
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_viol_ref ON violations(inspection_reference)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_viol_time ON violations(timestamp)")

    # 4. System Settings Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS system_settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TIMESTAMP NOT NULL
        )
    """)

    # Seed Default System Settings if empty
    default_settings = [
        ("camera_source", str(AppConfig.DEFAULT_CAMERA_SOURCE)),
        ("camera_width", str(AppConfig.DEFAULT_FRAME_WIDTH)),
        ("camera_height", str(AppConfig.DEFAULT_FRAME_HEIGHT)),
        ("conf_person", str(AppConfig.CONF_PERSON)),
        ("conf_hardhat", str(AppConfig.CONF_HARDHAT)),
        ("conf_vest", str(AppConfig.CONF_VEST)),
        ("iou_threshold", str(AppConfig.IOU_THRESHOLD)),
        ("safe_confirm_frames", str(AppConfig.REQUIRED_SAFE_OBSERVATIONS)),
        ("fail_confirm_frames", str(AppConfig.REQUIRED_VIOLATION_OBSERVATIONS)),
        ("inspection_timeout", str(AppConfig.INSPECTION_TIMEOUT_SECONDS)),
        ("audio_alert_enabled", "true"),
    ]

    now_iso = datetime.datetime.now().isoformat()
    for k, v in default_settings:
        cursor.execute("""
            INSERT OR IGNORE INTO system_settings (key, value, updated_at)
            VALUES (?, ?, ?)
        """, (k, v, now_iso))

    conn.commit()
    conn.close()


# ==============================================================================
# SESSION LIFECYCLE
# ==============================================================================

def create_session(camera_source: str = "0") -> int:
    """Starts a new monitoring session."""
    conn = get_db_connection()
    cursor = conn.cursor()
    now_iso = datetime.datetime.now().isoformat()
    cursor.execute("""
        INSERT INTO monitoring_sessions (start_time, camera_source, status)
        VALUES (?, ?, 'ACTIVE')
    """, (now_iso, str(camera_source)))
    session_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return session_id


def close_session(session_id: int):
    """Closes an active monitoring session and summarizes stats."""
    if session_id is None:
        return
    conn = get_db_connection()
    cursor = conn.cursor()
    now_iso = datetime.datetime.now().isoformat()

    cursor.execute("""
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN result = 'PASS' THEN 1 ELSE 0 END) as pass_cnt,
            SUM(CASE WHEN result = 'FAIL' THEN 1 ELSE 0 END) as fail_cnt,
            SUM(CASE WHEN result = 'REINSPECT' THEN 1 ELSE 0 END) as reinsp_cnt
        FROM inspections
        WHERE session_id = ?
    """, (session_id,))
    row = cursor.fetchone()

    cursor.execute("""
        UPDATE monitoring_sessions
        SET end_time = ?,
            total_inspections = ?,
            total_pass = ?,
            total_fail = ?,
            total_reinspect = ?,
            status = 'COMPLETED'
        WHERE session_id = ?
    """, (
        now_iso,
        row["total"] or 0,
        row["pass_cnt"] or 0,
        row["fail_cnt"] or 0,
        row["reinsp_cnt"] or 0,
        session_id,
    ))
    conn.commit()
    conn.close()


# ==============================================================================
# INSPECTION & VIOLATION RECORDING
# ==============================================================================

def record_inspection(result: InspectionResult) -> bool:
    """
    Inserts a completed inspection record into the SQLite database.
    If the inspection failed, also creates a confirmed violation entry.
    Returns True on successful insertion.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    entry_dt = datetime.datetime.fromtimestamp(result.entry_timestamp).isoformat()
    comp_dt = datetime.datetime.fromtimestamp(result.completion_timestamp).isoformat()

    try:
        cursor.execute("""
            INSERT INTO inspections (
                inspection_reference, timestamp, completion_timestamp,
                helmet_status, vest_status, result,
                person_confidence, helmet_confidence, vest_confidence,
                average_confidence, inspection_duration, reinspection_reason,
                missing_ppe, original_image_path, annotated_image_path,
                session_id, total_frames_analyzed
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            result.inspection_reference,
            entry_dt,
            comp_dt,
            result.helmet_status,
            result.vest_status,
            result.decision,
            result.person_confidence,
            result.helmet_confidence,
            result.vest_confidence,
            result.average_confidence,
            result.duration,
            result.reinspection_reason,
            result.missing_ppe,
            result.original_image_path,
            result.annotated_image_path,
            result.session_id,
            result.total_frames_analyzed,
        ))

        # If decision is FAIL, record violation
        if result.decision == "FAIL":
            cursor.execute("""
                INSERT OR IGNORE INTO violations (
                    inspection_reference, timestamp, missing_ppe,
                    confidence, original_image_path, annotated_image_path,
                    acknowledgement_status, session_id
                ) VALUES (?, ?, ?, ?, ?, ?, 'UNACKNOWLEDGED', ?)
            """, (
                result.inspection_reference,
                comp_dt,
                result.missing_ppe,
                result.average_confidence,
                result.original_image_path,
                result.annotated_image_path,
                result.session_id,
            ))

        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()


# ==============================================================================
# QUERY & RETRIEVAL APIS
# ==============================================================================

def get_inspections(
    page: int = 1,
    per_page: int = 10,
    search: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    result_filter: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], int, int]:
    """Retrieves paginated inspection records with filtering."""
    conn = get_db_connection()
    cursor = conn.cursor()

    query = "SELECT * FROM inspections WHERE 1=1"
    count_query = "SELECT COUNT(*) FROM inspections WHERE 1=1"
    params = []

    if search:
        query += " AND inspection_reference LIKE ?"
        count_query += " AND inspection_reference LIKE ?"
        params.append(f"%{search.strip()}%")
    if date_from:
        query += " AND timestamp >= ?"
        count_query += " AND timestamp >= ?"
        params.append(date_from)
    if date_to:
        query += " AND timestamp <= ?"
        count_query += " AND timestamp <= ?"
        params.append(date_to + " 23:59:59")
    if result_filter and result_filter.upper() != "ALL":
        query += " AND result = ?"
        count_query += " AND result = ?"
        params.append(result_filter.upper())

    cursor.execute(count_query, params)
    total_count = cursor.fetchone()[0] or 0

    query += " ORDER BY id DESC LIMIT ? OFFSET ?"
    offset = (page - 1) * per_page
    params.extend([per_page, offset])

    cursor.execute(query, params)
    rows = cursor.fetchall()
    records = [dict(r) for r in rows]

    total_pages = max(1, (total_count + per_page - 1) // per_page)
    conn.close()
    return records, total_count, total_pages


def get_inspection_by_id(inspection_id: int) -> Optional[Dict[str, Any]]:
    """Retrieves single inspection by primary ID."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM inspections WHERE id = ?", (inspection_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def get_violations(
    page: int = 1,
    per_page: int = 10,
    search: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    missing_ppe: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], int, int]:
    """Retrieves paginated confirmed violations with filtering."""
    conn = get_db_connection()
    cursor = conn.cursor()

    query = "SELECT * FROM violations WHERE 1=1"
    count_query = "SELECT COUNT(*) FROM violations WHERE 1=1"
    params = []

    if search:
        query += " AND inspection_reference LIKE ?"
        count_query += " AND inspection_reference LIKE ?"
        params.append(f"%{search.strip()}%")
    if date_from:
        query += " AND timestamp >= ?"
        count_query += " AND timestamp >= ?"
        params.append(date_from)
    if date_to:
        query += " AND timestamp <= ?"
        count_query += " AND timestamp <= ?"
        params.append(date_to + " 23:59:59")
    if missing_ppe and missing_ppe.upper() != "ALL":
        query += " AND missing_ppe LIKE ?"
        count_query += " AND missing_ppe LIKE ?"
        params.append(f"%{missing_ppe}%")

    cursor.execute(count_query, params)
    total_count = cursor.fetchone()[0] or 0

    query += " ORDER BY id DESC LIMIT ? OFFSET ?"
    offset = (page - 1) * per_page
    params.extend([per_page, offset])

    cursor.execute(query, params)
    rows = cursor.fetchall()
    records = [dict(r) for r in rows]

    total_pages = max(1, (total_count + per_page - 1) // per_page)
    conn.close()
    return records, total_count, total_pages


def get_violation_by_id(violation_id: int) -> Optional[Dict[str, Any]]:
    """Retrieves single violation record by primary ID."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM violations WHERE id = ?", (violation_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


# ==============================================================================
# DELETION & SAFE HISTORY MANAGEMENT
# ==============================================================================

def delete_inspection(inspection_id: int) -> bool:
    """Deletes an individual inspection record and cleans up associated evidence files."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT original_image_path, annotated_image_path FROM inspections WHERE id = ?", (inspection_id,))
    row = cursor.fetchone()

    if row:
        for p_str in [row["original_image_path"], row["annotated_image_path"]]:
            if p_str:
                full_p = AppConfig.VIOLATIONS_DIR / p_str
                if full_p.exists():
                    try:
                        full_p.unlink()
                    except Exception:
                        pass

    cursor.execute("DELETE FROM inspections WHERE id = ?", (inspection_id,))
    deleted = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return deleted


def delete_violation(violation_id: int) -> bool:
    """Deletes an individual violation record without deleting the inspection."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT original_image_path, annotated_image_path FROM violations WHERE id = ?", (violation_id,))
    row = cursor.fetchone()

    if row:
        for p_str in [row["original_image_path"], row["annotated_image_path"]]:
            if p_str:
                full_p = AppConfig.VIOLATIONS_DIR / p_str
                if full_p.exists():
                    try:
                        full_p.unlink()
                    except Exception:
                        pass

    cursor.execute("DELETE FROM violations WHERE id = ?", (violation_id,))
    deleted = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return deleted


def clear_all_history(target: str = "all") -> Dict[str, int]:
    """
    Clears inspections, violations, and/or sessions safely.
    NEVER touches models/ or dataset/.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    viol_count = 0
    insp_count = 0
    sess_count = 0

    if target in ["violations", "all"]:
        cursor.execute("DELETE FROM violations")
        viol_count = cursor.rowcount

        # Clean up evidence files in outputs/violations/
        if AppConfig.VIOLATIONS_DIR.exists():
            for sub in AppConfig.VIOLATIONS_DIR.iterdir():
                if sub.is_dir():
                    for f in sub.iterdir():
                        if f.is_file():
                            try:
                                f.unlink()
                            except Exception:
                                pass

    if target in ["inspections", "all"]:
        cursor.execute("DELETE FROM inspections")
        insp_count = cursor.rowcount

    if target in ["sessions", "all"]:
        cursor.execute("DELETE FROM monitoring_sessions")
        sess_count = cursor.rowcount

    conn.commit()
    conn.close()

    return {
        "deleted_inspections": insp_count,
        "deleted_violations": viol_count,
        "deleted_sessions": sess_count,
    }


# ==============================================================================
# AGGREGATES & METRICS FOR DASHBOARD & ANALYTICS
# ==============================================================================

def get_overview_statistics(date_from: Optional[str] = None, date_to: Optional[str] = None) -> Dict[str, Any]:
    """Computes summary metrics for Dashboard KPI cards."""
    conn = get_db_connection()
    cursor = conn.cursor()

    where_sql = " WHERE 1=1"
    params = []
    if date_from:
        where_sql += " AND timestamp >= ?"
        params.append(date_from)
    if date_to:
        where_sql += " AND timestamp <= ?"
        params.append(date_to + " 23:59:59")

    cursor.execute(f"""
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN result = 'PASS' THEN 1 ELSE 0 END) as pass_cnt,
            SUM(CASE WHEN result = 'FAIL' THEN 1 ELSE 0 END) as fail_cnt,
            SUM(CASE WHEN result = 'REINSPECT' THEN 1 ELSE 0 END) as reinsp_cnt,
            SUM(CASE WHEN helmet_status = 'MISSING' THEN 1 ELSE 0 END) as helmet_viol_cnt,
            SUM(CASE WHEN vest_status = 'MISSING' THEN 1 ELSE 0 END) as vest_viol_cnt,
            AVG(inspection_duration) as avg_duration
        FROM inspections {where_sql}
    """, params)
    row = cursor.fetchone()

    total = row["total"] or 0
    passed = row["pass_cnt"] or 0
    failed = row["fail_cnt"] or 0
    reinspected = row["reinsp_cnt"] or 0
    helmet_viol = row["helmet_viol_cnt"] or 0
    vest_viol = row["vest_viol_cnt"] or 0
    avg_dur = round(row["avg_duration"] or 0.0, 2)

    # Compliance percentage: PASS / (PASS + FAIL) * 100
    decisive_total = passed + failed
    compliance_pct = round((passed / decisive_total) * 100.0, 1) if decisive_total > 0 else (100.0 if failed == 0 else 0.0)

    # Recent 5 violations
    cursor.execute(f"""
        SELECT id, inspection_reference, timestamp, missing_ppe, confidence
        FROM violations {where_sql}
        ORDER BY id DESC LIMIT 5
    """, params)
    recent_violations = [dict(r) for r in cursor.fetchall()]

    conn.close()

    return {
        "total_inspections": total,
        "passed_inspections": passed,
        "failed_inspections": failed,
        "reinspection_count": reinspected,
        "compliance_percentage": compliance_pct,
        "total_helmet_violations": helmet_viol,
        "total_vest_violations": vest_viol,
        "average_inspection_duration": avg_dur,
        "recent_violations": recent_violations,
        # Standardized Checkpoint Aliases
        "total_passed": passed,
        "total_violations": failed,
        "total_reinspections": reinspected,
        "compliance_rate_pct": compliance_pct,
        "average_duration_seconds": avg_dur,
        "hardhat_adherence_pct": round(100.0 - (helmet_viol / total * 100.0), 1) if total > 0 else 100.0,
        "vest_adherence_pct": round(100.0 - (vest_viol / total * 100.0), 1) if total > 0 else 100.0,
    }


def get_analytics_summary(date_from: Optional[str] = None, date_to: Optional[str] = None) -> Dict[str, Any]:
    """Computes time-series and categorical aggregates for Chart.js analytics."""
    conn = get_db_connection()
    cursor = conn.cursor()

    where_sql = " WHERE 1=1"
    params = []
    if date_from:
        where_sql += " AND timestamp >= ?"
        params.append(date_from)
    if date_to:
        where_sql += " AND timestamp <= ?"
        params.append(date_to + " 23:59:59")

    # 1. Daily Inspection Trend (PASS, FAIL, REINSPECT)
    cursor.execute(f"""
        SELECT strftime('%Y-%m-%d', timestamp) as day,
            SUM(CASE WHEN result = 'PASS' THEN 1 ELSE 0 END) as pass_c,
            SUM(CASE WHEN result = 'FAIL' THEN 1 ELSE 0 END) as fail_c,
            SUM(CASE WHEN result = 'REINSPECT' THEN 1 ELSE 0 END) as reinsp_c
        FROM inspections {where_sql}
        GROUP BY day ORDER BY day ASC LIMIT 14
    """, params)
    trend_rows = cursor.fetchall()
    days = [r["day"] for r in trend_rows]
    passes = [r["pass_c"] for r in trend_rows]
    fails = [r["fail_c"] for r in trend_rows]
    reinsps = [r["reinsp_c"] for r in trend_rows]

    # 2. Decision Distribution
    cursor.execute(f"""
        SELECT result, COUNT(*) as cnt
        FROM inspections {where_sql}
        GROUP BY result
    """, params)
    dist_rows = cursor.fetchall()
    dist = {r["result"]: r["cnt"] for r in dist_rows}

    # 3. Missing PPE Category Breakdown
    cursor.execute(f"""
        SELECT missing_ppe, COUNT(*) as cnt
        FROM violations {where_sql}
        GROUP BY missing_ppe
    """, params)
    ppe_rows = cursor.fetchall()
    ppe_breakdown = {r["missing_ppe"]: r["cnt"] for r in ppe_rows}

    # 4. Hourly Inspection Volume
    cursor.execute(f"""
        SELECT strftime('%H:00', timestamp) as hr, COUNT(*) as cnt
        FROM inspections {where_sql}
        GROUP BY hr ORDER BY hr ASC
    """, params)
    hr_rows = cursor.fetchall()
    hours = [r["hr"] for r in hr_rows]
    hour_counts = [r["cnt"] for r in hr_rows]

    conn.close()

    overview = get_overview_statistics(date_from=date_from, date_to=date_to)

    daily_trends = [
        {"date": d, "pass": p, "fail": f, "reinspect": r}
        for d, p, f, r in zip(days, passes, fails, reinsps)
    ]
    hourly_dist = [
        {"hour": int(h.split(":")[0]) if ":" in h else 0, "count": c, "failures": 0}
        for h, c in zip(hours, hour_counts)
    ]

    return {
        **overview,
        "daily_labels": days,
        "daily_pass": passes,
        "daily_fail": fails,
        "daily_reinspect": reinsps,
        "daily_trends": daily_trends,
        "decision_distribution": dist,
        "ppe_breakdown": ppe_breakdown,
        "missing_ppe_breakdown": ppe_breakdown,
        "hourly_labels": hours,
        "hourly_counts": hour_counts,
        "hourly_distribution": hourly_dist,
    }


# ==============================================================================
# CSV EXPORT
# ==============================================================================

def export_inspections_csv(
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    result_filter: Optional[str] = None,
) -> str:
    """Generates RFC-4180 compliant CSV string of inspections."""
    conn = get_db_connection()
    cursor = conn.cursor()

    query = "SELECT * FROM inspections WHERE 1=1"
    params = []
    if date_from:
        query += " AND timestamp >= ?"
        params.append(date_from)
    if date_to:
        query += " AND timestamp <= ?"
        params.append(date_to + " 23:59:59")
    if result_filter and result_filter.upper() != "ALL":
        query += " AND result = ?"
        params.append(result_filter.upper())

    query += " ORDER BY id DESC"
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "ID", "Inspection Reference", "Entry Timestamp", "Completion Timestamp",
        "Helmet Status", "Vest Status", "Result", "Missing PPE",
        "Person Confidence", "Helmet Confidence", "Vest Confidence", "Average Confidence",
        "Duration (sec)", "Reason", "Original Image", "Annotated Image"
    ])

    for r in rows:
        writer.writerow([
            r["id"], r["inspection_reference"], r["timestamp"], r["completion_timestamp"],
            r["helmet_status"], r["vest_status"], r["result"], r["missing_ppe"],
            r["person_confidence"], r["helmet_confidence"], r["vest_confidence"], r["average_confidence"],
            r["inspection_duration"], r["reinspection_reason"], r["original_image_path"], r["annotated_image_path"]
        ])

    return output.getvalue()


def export_violations_csv(
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> str:
    """Generates RFC-4180 compliant CSV string of violations."""
    conn = get_db_connection()
    cursor = conn.cursor()

    query = "SELECT * FROM violations WHERE 1=1"
    params = []
    if date_from:
        query += " AND timestamp >= ?"
        params.append(date_from)
    if date_to:
        query += " AND timestamp <= ?"
        params.append(date_to + " 23:59:59")

    query += " ORDER BY id DESC"
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "ID", "Inspection Reference", "Timestamp", "Missing PPE",
        "Confidence", "Acknowledgement Status", "Original Evidence", "Annotated Evidence"
    ])

    for r in rows:
        writer.writerow([
            r["id"], r["inspection_reference"], r["timestamp"], r["missing_ppe"],
            r["confidence"], r["acknowledgement_status"], r["original_image_path"], r["annotated_image_path"]
        ])

    return output.getvalue()


# ==============================================================================
# SETTINGS
# ==============================================================================

def get_system_settings() -> Dict[str, str]:
    """Retrieves all system settings as a key-value dictionary."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT key, value FROM system_settings")
    rows = cursor.fetchall()
    conn.close()
    return {r["key"]: r["value"] for r in rows}


def update_system_setting(key: str, value: str):
    """Updates a single setting with timestamp."""
    conn = get_db_connection()
    cursor = conn.cursor()
    now_iso = datetime.datetime.now().isoformat()
    cursor.execute("""
        INSERT INTO system_settings (key, value, updated_at)
        VALUES (?, ?, ?)
        ON CONFLICT(key) DO UPDATE SET
            value = excluded.value,
            updated_at = excluded.updated_at
    """, (key, str(value), now_iso))
    conn.commit()
    conn.close()
