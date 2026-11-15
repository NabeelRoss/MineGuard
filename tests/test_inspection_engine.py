"""
Unit Tests for MineGuard AI Checkpoint Inspection Engine.
Verifies spatial trigger zones, singularity checks, multi-frame consensus rules,
and state machine transitions.
"""

import unittest
import numpy as np
import time
from inspection.inspection_result import (
    InspectionState,
    InspectionDecision,
    PPEStatus,
    InspectionFrameObservation,
    InspectionResult,
)
from inspection.inspection_trigger import InspectionTrigger
from inspection.frame_collector import FrameCollector
from inspection.ppe_verifier import PPEVerifier
from inspection.inspection_manager import InspectionManager
from detection.detector import Detection, DetectionResult


class TestInspectionTrigger(unittest.TestCase):
    def setUp(self):
        self.trigger = InspectionTrigger()

    def test_no_workers_returns_idle(self):
        in_zone, cand, mult, reason = self.trigger.evaluate_entry([], frame_shape=(720, 1280))
        self.assertFalse(in_zone)
        self.assertIsNone(cand)
        self.assertFalse(mult)

    def test_worker_outside_zone_rejected(self):
        # Worker far to the left (x: 10 to 60 in a 1280 wide frame; zone is 15% to 85% = 192 to 1088)
        outside_det = Detection(box=[10.0, 100.0, 60.0, 500.0], confidence=0.9, class_id=0, class_name="Person")
        in_zone, cand, mult, reason = self.trigger.evaluate_entry([outside_det], frame_shape=(720, 1280))
        self.assertFalse(in_zone)
        self.assertIsNone(cand)

    def test_worker_inside_zone_stability_check(self):
        # Worker centered inside zone (center ~ 640, 350, height > 130px)
        center_det = Detection(box=[500.0, 100.0, 780.0, 600.0], confidence=0.92, class_id=0, class_name="Person")
        
        # Frame 1: Stabilizing
        in_zone1, cand1, mult1, reason1 = self.trigger.evaluate_entry([center_det], frame_shape=(720, 1280))
        self.assertTrue(in_zone1)
        self.assertIn("STABILIZING", reason1)

        # Frame 2: Stabilized -> Ready
        in_zone2, cand2, mult2, reason2 = self.trigger.evaluate_entry([center_det], frame_shape=(720, 1280))
        self.assertTrue(in_zone2)
        self.assertEqual(reason2, "WORKER_STABILIZED")
        self.assertEqual(cand2, center_det)

    def test_multiple_workers_in_zone_triggers_singularity_rejection(self):
        worker1 = Detection(box=[400.0, 100.0, 550.0, 600.0], confidence=0.9, class_id=0, class_name="Person")
        worker2 = Detection(box=[650.0, 100.0, 800.0, 600.0], confidence=0.9, class_id=0, class_name="Person")
        in_zone, cand, mult, reason = self.trigger.evaluate_entry([worker1, worker2], frame_shape=(720, 1280))
        self.assertTrue(in_zone)
        self.assertTrue(mult)
        self.assertEqual(reason, "MULTIPLE_WORKERS_IN_ZONE")


class TestPPEVerifier(unittest.TestCase):
    def setUp(self):
        self.verifier = PPEVerifier(
            safe_confirm_frames=3,
            fail_confirm_frames=5,
            timeout_seconds=2.0,
        )
        self.collector = FrameCollector()
        self.dummy_raw = np.zeros((100, 100, 3), dtype=np.uint8)
        self.dummy_ann = np.zeros((100, 100, 3), dtype=np.uint8)

    def test_three_consecutive_safe_frames_yields_pass(self):
        for i in range(3):
            obs = InspectionFrameObservation(
                timestamp=time.time(),
                frame_idx=i,
                person_box=[400, 100, 700, 600],
                person_confidence=0.92,
                helmet_status=PPEStatus.PRESENT,
                vest_status=PPEStatus.PRESENT,
                helmet_confidence=0.90,
                vest_confidence=0.85,
            )
            self.collector.add_observation(obs, self.dummy_raw, self.dummy_ann)

        decision, h_stat, v_stat, reason = self.verifier.evaluate_cycle(self.collector, elapsed_seconds=0.5)
        self.assertEqual(decision, InspectionDecision.PASS)
        self.assertEqual(h_stat, PPEStatus.PRESENT.value)
        self.assertEqual(v_stat, PPEStatus.PRESENT.value)

    def test_five_consecutive_fail_frames_yields_fail(self):
        for i in range(5):
            obs = InspectionFrameObservation(
                timestamp=time.time(),
                frame_idx=i,
                person_box=[400, 100, 700, 600],
                person_confidence=0.90,
                helmet_status=PPEStatus.MISSING,
                vest_status=PPEStatus.PRESENT,
                helmet_confidence=0.0,
                vest_confidence=0.85,
            )
            self.collector.add_observation(obs, self.dummy_raw, self.dummy_ann)

        decision, h_stat, v_stat, reason = self.verifier.evaluate_cycle(self.collector, elapsed_seconds=0.8)
        self.assertEqual(decision, InspectionDecision.FAIL)
        self.assertEqual(h_stat, PPEStatus.MISSING.value)
        self.assertIn("Hardhat", reason)

    def test_unknown_never_treated_as_safe(self):
        for i in range(5):
            obs = InspectionFrameObservation(
                timestamp=time.time(),
                frame_idx=i,
                person_box=[400, 100, 700, 600],
                person_confidence=0.90,
                helmet_status=PPEStatus.PRESENT,
                vest_status=PPEStatus.UNCERTAIN,
                helmet_confidence=0.90,
                vest_confidence=0.0,
            )
            self.collector.add_observation(obs, self.dummy_raw, self.dummy_ann)

        decision, _, _, _ = self.verifier.evaluate_cycle(self.collector, elapsed_seconds=0.5)
        self.assertNotEqual(decision, InspectionDecision.PASS)

    def test_timeout_yields_reinspect(self):
        obs = InspectionFrameObservation(
            timestamp=time.time(),
            frame_idx=0,
            person_box=[400, 100, 700, 600],
            person_confidence=0.90,
            helmet_status=PPEStatus.PRESENT,
            vest_status=PPEStatus.UNCERTAIN,
            helmet_confidence=0.90,
            vest_confidence=0.0,
        )
        self.collector.add_observation(obs, self.dummy_raw, self.dummy_ann)

        # Elapsed time exceeds 2.0s timeout
        decision, _, _, reason = self.verifier.evaluate_cycle(self.collector, elapsed_seconds=2.5)
        self.assertEqual(decision, InspectionDecision.REINSPECT)
        self.assertIn("timed out", reason)


class TestInspectionManager(unittest.TestCase):
    def setUp(self):
        self.recorded_results = []
        verifier = PPEVerifier(safe_confirm_frames=3, fail_confirm_frames=5, timeout_seconds=5.0)
        self.manager = InspectionManager(
            verifier=verifier,
            db_callback=lambda res: self.recorded_results.append(res),
        )

    def test_manager_lifecycle(self):
        raw_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        annotated_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        # 1. Initially IDLE
        self.assertEqual(self.manager.state, InspectionState.IDLE)
        self.assertIsNone(self.manager.current_reference)

        # 2. Worker appears in center with hardhat and safety vest
        person_det = Detection(box=[500.0, 100.0, 780.0, 600.0], confidence=0.95, class_id=0, class_name="Person")
        hardhat_det = Detection(box=[580.0, 110.0, 700.0, 190.0], confidence=0.92, class_id=1, class_name="Hardhat")
        vest_det = Detection(box=[530.0, 220.0, 750.0, 450.0], confidence=0.90, class_id=2, class_name="Safety Vest")

        det_result = DetectionResult(
            persons=[person_det],
            helmets=[hardhat_det],
            vests=[vest_det],
            all_detections=[person_det, hardhat_det, vest_det],
            inference_time_ms=10.0,
            frame_shape=(720, 1280, 3),
        )

        # Process frames until state moves from DETECTING to VERIFYING to PASS
        for _ in range(12):
            self.manager.process_frame(raw_frame, annotated_frame, det_result)

        # Should complete with a recorded result
        self.assertGreaterEqual(len(self.recorded_results), 1)
        res = self.recorded_results[0]
        self.assertEqual(res.decision, "PASS")
        self.assertTrue(res.inspection_reference.startswith("INSP-"))


if __name__ == "__main__":
    unittest.main()
