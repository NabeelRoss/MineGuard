"""
Integration Tests for MineGuard AI Flask Routes and REST APIs.
Verifies all HTML view routes, JSON API endpoints, CSV exports,
and error handling.
"""

import unittest
from app import create_app


class TestFlaskRoutes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = create_app()
        cls.app.config["TESTING"] = True
        cls.client = cls.app.test_client()

    def test_view_routes(self):
        views = [
            ("/", 200),
            ("/live", 200),
            ("/history", 200),
            ("/violations", 200),
            ("/analytics", 200),
            ("/reports", 200),
            ("/settings", 200),
        ]
        for url, expected_code in views:
            with self.subTest(url=url):
                res = self.client.get(url)
                self.assertEqual(res.status_code, expected_code)

    def test_api_status_and_health(self):
        res = self.client.get("/api/status")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("monitoring_status", data)
        self.assertIn("camera_status", data)
        self.assertIn("statistics", data)

        res_health = self.client.get("/api/system/health")
        self.assertEqual(res_health.status_code, 200)
        h = res_health.get_json()
        self.assertIn("device", h)
        self.assertIn("cuda_available", h)
        self.assertIn("database_size_bytes", h)

    def test_api_inspection_current(self):
        res = self.client.get("/api/inspection/current")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("state", data)
        self.assertIn("hardhat", data)
        self.assertIn("vest", data)

    def test_api_inspections_and_violations(self):
        res_insp = self.client.get("/api/inspections?page=1&per_page=5")
        self.assertEqual(res_insp.status_code, 200)
        data_insp = res_insp.get_json()
        self.assertIn("inspections", data_insp)
        self.assertIn("total", data_insp)

        res_vio = self.client.get("/api/violations?page=1&per_page=5")
        self.assertEqual(res_vio.status_code, 200)
        data_vio = res_vio.get_json()
        self.assertIn("violations", data_vio)
        self.assertIn("total", data_vio)

    def test_api_analytics_and_reports(self):
        res_ana = self.client.get("/api/analytics/summary")
        self.assertEqual(res_ana.status_code, 200)
        data_ana = res_ana.get_json()
        self.assertIn("compliance_rate_pct", data_ana)
        self.assertIn("decision_distribution", data_ana)

        res_rep = self.client.get("/api/reports/summary")
        self.assertEqual(res_rep.status_code, 200)
        data_rep = res_rep.get_json()
        self.assertIn("summary", data_rep)
        self.assertIn("daily_breakdown", data_rep)

    def test_api_settings_get_post(self):
        # GET settings
        res = self.client.get("/api/settings")
        self.assertEqual(res.status_code, 200)
        s = res.get_json()
        self.assertIn("conf_person", s)

        # POST settings
        payload = {
            "conf_person": 0.36,
            "conf_hardhat": 0.29,
            "conf_vest": 0.36,
            "iou_threshold": 0.45,
            "required_safe_frames": 3,
            "required_fail_frames": 5,
        }
        res_post = self.client.post("/api/settings", json=payload)
        self.assertEqual(res_post.status_code, 200)

    def test_api_csv_exports(self):
        res_csv_insp = self.client.get("/api/export/inspections")
        self.assertEqual(res_csv_insp.status_code, 200)
        self.assertEqual(res_csv_insp.mimetype, "text/csv")

        res_csv_vio = self.client.get("/api/export/violations")
        self.assertEqual(res_csv_vio.status_code, 200)
        self.assertEqual(res_csv_vio.mimetype, "text/csv")


if __name__ == "__main__":
    unittest.main()
