"""
Unit Tests for MineGuard AI SQLite Database Layer.
Verifies table schemas, CRUD operations, pagination, CSV exports,
and safe history maintenance.
"""

import unittest
import sqlite3
import tempfile
import os
import time
from pathlib import Path
from backend.database import (
    init_db,
    record_inspection,
    get_inspections,
    get_inspection_by_id,
    delete_inspection,
    get_violations,
    get_violation_by_id,
    delete_violation,
    clear_all_history,
    get_overview_statistics,
    get_analytics_summary,
    export_inspections_csv,
    export_violations_csv,
    get_system_settings,
    update_system_setting,
)
from inspection.inspection_result import InspectionResult


class TestDatabaseLayer(unittest.TestCase):
    def setUp(self):
        # Create a fresh temporary database file for isolation
        self.temp_db_fd, self.temp_db_path = tempfile.mkstemp(suffix=".db")
        os.close(self.temp_db_fd)

        # Patch AppConfig.DATABASE_PATH to use the temp DB
        from config import AppConfig
        self.original_db_path = AppConfig.DATABASE_PATH
        AppConfig.DATABASE_PATH = Path(self.temp_db_path)

        # Initialize schema
        init_db()

    def tearDown(self):
        from config import AppConfig
        AppConfig.DATABASE_PATH = self.original_db_path
        if os.path.exists(self.temp_db_path):
            try:
                os.remove(self.temp_db_path)
            except Exception:
                pass

    def test_schema_tables_exist(self):
        conn = sqlite3.connect(self.temp_db_path)
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [r[0] for r in cur.fetchall()]
        conn.close()

        self.assertIn("inspections", tables)
        self.assertIn("violations", tables)
        self.assertIn("monitoring_sessions", tables)
        self.assertIn("system_settings", tables)

    def test_record_and_retrieve_inspection(self):
        now = time.time()
        res = InspectionResult(
            inspection_reference="INSP-20261115-120000-TEST",
            entry_timestamp=now - 2.0,
            completion_timestamp=now,
            duration=2.0,
            helmet_status="PRESENT",
            vest_status="PRESENT",
            decision="PASS",
            person_confidence=0.95,
            helmet_confidence=0.92,
            vest_confidence=0.88,
            average_confidence=0.90,
            reinspection_reason="Compliant test inspection",
            missing_ppe="",
            original_image_path="test_orig.jpg",
            annotated_image_path="test_ann.jpg",
            session_id=None,
            total_frames_analyzed=15,
        )

        success = record_inspection(res)
        self.assertTrue(success)

        # Retrieve
        inspections, total, pages = get_inspections(page=1, per_page=10)
        self.assertEqual(total, 1)
        insp = inspections[0]
        self.assertEqual(insp["inspection_reference"], "INSP-20261115-120000-TEST")
        self.assertEqual(insp["result"], "PASS")

    def test_record_and_retrieve_violation(self):
        now = time.time()
        res_fail = InspectionResult(
            inspection_reference="INSP-20261115-120500-FAIL",
            entry_timestamp=now - 2.5,
            completion_timestamp=now,
            duration=2.5,
            helmet_status="MISSING",
            vest_status="PRESENT",
            decision="FAIL",
            person_confidence=0.95,
            helmet_confidence=0.0,
            vest_confidence=0.88,
            average_confidence=0.91,
            reinspection_reason="Missing Hardhat",
            missing_ppe="Hardhat",
            original_image_path="outputs/violations/2026-11-15/fail_orig.jpg",
            annotated_image_path="outputs/violations/2026-11-15/fail_ann.jpg",
            session_id=None,
            total_frames_analyzed=18,
        )

        record_inspection(res_fail)

        # Verify violation record was automatically created
        violations, total, pages = get_violations(page=1, per_page=10)
        self.assertEqual(total, 1)
        vio = violations[0]
        self.assertEqual(vio["inspection_reference"], "INSP-20261115-120500-FAIL")
        self.assertEqual(vio["missing_ppe"], "Hardhat")

    def test_analytics_and_statistics(self):
        now = time.time()
        for i in range(3):
            record_inspection(InspectionResult(
                inspection_reference=f"INSP-PASS-{i}",
                entry_timestamp=now, completion_timestamp=now + 1.5,
                duration=1.5, helmet_status="PRESENT", vest_status="PRESENT",
                decision="PASS", person_confidence=0.9, helmet_confidence=0.9,
                vest_confidence=0.9, average_confidence=0.9,
            ))

        record_inspection(InspectionResult(
            inspection_reference="INSP-FAIL-0",
            entry_timestamp=now, completion_timestamp=now + 2.0,
            duration=2.0, helmet_status="MISSING", vest_status="PRESENT",
            decision="FAIL", person_confidence=0.9, helmet_confidence=0.0,
            vest_confidence=0.9, average_confidence=0.9, missing_ppe="Hardhat",
        ))

        record_inspection(InspectionResult(
            inspection_reference="INSP-REINSP-0",
            entry_timestamp=now, completion_timestamp=now + 2.0,
            duration=2.0, helmet_status="UNCERTAIN", vest_status="UNCERTAIN",
            decision="REINSPECT", person_confidence=0.9, helmet_confidence=0.0,
            vest_confidence=0.0, average_confidence=0.9,
        ))

        stats = get_overview_statistics()
        self.assertEqual(stats["total_inspections"], 5)
        self.assertEqual(stats["total_passed"], 3)
        self.assertEqual(stats["total_violations"], 1)
        self.assertEqual(stats["total_reinspections"], 1)
        # 3 passed out of 4 verified (3 pass + 1 fail) = 75.0%
        self.assertEqual(stats["compliance_rate_pct"], 75.0)

        analytics = get_analytics_summary()
        self.assertEqual(analytics["total_inspections"], 5)
        self.assertEqual(analytics["decision_distribution"]["PASS"], 3)
        self.assertEqual(analytics["decision_distribution"]["FAIL"], 1)

    def test_csv_exports(self):
        now = time.time()
        record_inspection(InspectionResult(
            inspection_reference="INSP-EXPORT-01",
            entry_timestamp=now, completion_timestamp=now + 1.0,
            duration=1.0, helmet_status="PRESENT", vest_status="PRESENT",
            decision="PASS", person_confidence=0.9, helmet_confidence=0.9,
            vest_confidence=0.9, average_confidence=0.9,
        ))

        csv_data = export_inspections_csv()
        self.assertIn("Inspection Reference", csv_data)
        self.assertIn("INSP-EXPORT-01", csv_data)

    def test_safe_history_clearing(self):
        now = time.time()
        record_inspection(InspectionResult(
            inspection_reference="INSP-CLEAR-01",
            entry_timestamp=now, completion_timestamp=now + 1.0,
            duration=1.0, helmet_status="PRESENT", vest_status="PRESENT",
            decision="PASS", person_confidence=0.9, helmet_confidence=0.9,
            vest_confidence=0.9, average_confidence=0.9,
        ))
        self.assertEqual(get_overview_statistics()["total_inspections"], 1)

        # Clear inspections
        clear_all_history(target="inspections")
        self.assertEqual(get_overview_statistics()["total_inspections"], 0)


if __name__ == "__main__":
    unittest.main()
