"""
Inspection Manager Module for MineGuard AI.

Orchestrates the fixed-point checkpoint quality inspection state machine.
Enforces:
1. Unique inspection reference per visit (INSP-YYYYMMDD-HHMMSS-XXXX).
2. Clean independent lifecycle per worker (IDLE -> DETECTING -> VERIFYING -> DECISION -> IDLE).
3. Zero duplicate inspections during one continuous checkpoint visit.
4. Automatic dual-evidence capture (original & annotated) for confirmed violations.
5. Thread-safe live telemetry for the Flask dashboard.
"""

import time
import uuid
import datetime
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List
import cv2
import numpy as np

import utils.config as config
from detection.detector import DetectionResult, Detection
from compliance.association import PPEAssociator, AssociatedWorkerPPE
from inspection.inspection_result import (
    InspectionState,
    PPEStatus,
    InspectionDecision,
    InspectionFrameObservation,
    InspectionResult,
)
from inspection.inspection_trigger import InspectionTrigger
from inspection.frame_collector import FrameCollector
from inspection.ppe_verifier import PPEVerifier


class InspectionManager:
    """
    State machine manager for automated PPE checkpoint quality inspection.
    """

    def __init__(
        self,
        associator: Optional[PPEAssociator] = None,
        trigger: Optional[InspectionTrigger] = None,
        collector: Optional[FrameCollector] = None,
        verifier: Optional[PPEVerifier] = None,
        violations_dir: Optional[Path] = None,
        db_callback=None,
    ):
        self._lock = threading.Lock()
        self.associator = associator if associator else PPEAssociator()
        self.trigger = trigger if trigger else InspectionTrigger()
        self.collector = collector if collector else FrameCollector()
        self.verifier = verifier if verifier else PPEVerifier()
        self.violations_dir = Path(violations_dir) if violations_dir else config.VIOLATIONS_DIR
        self.db_callback = db_callback  # Callback to database layer

        # State Machine Tracking
        self.state: InspectionState = InspectionState.IDLE
        self.current_reference: Optional[str] = None
        self.entry_timestamp: float = 0.0
        self.completion_timestamp: float = 0.0
        self.terminal_decision: InspectionDecision = InspectionDecision.PENDING
        self.terminal_result: Optional[InspectionResult] = None

        # Exit Tracking to prevent duplicate inspections
        self.consecutive_empty_frames: int = 0
        self.exit_confirmation_frames: int = 5
        self.departure_cooldown_end: float = 0.0

        # Live Display Telemetry
        self.live_telemetry: Dict[str, Any] = {
            "state": InspectionState.IDLE.value,
            "reference": "CHECKPOINT_READY",
            "decision": InspectionDecision.PENDING.value,
            "person_detected": False,
            "helmet_status": "UNCERTAIN",
            "vest_status": "UNCERTAIN",
            "person_confidence": 0.0,
            "helmet_confidence": 0.0,
            "vest_confidence": 0.0,
            "duration": 0.0,
            "reason": "Ready for next worker",
            "frames_analyzed": 0,
        }

    def reset(self):
        """Forces complete reset back to IDLE."""
        with self._lock:
            self.state = InspectionState.IDLE
            self.current_reference = None
            self.entry_timestamp = 0.0
            self.completion_timestamp = 0.0
            self.terminal_decision = InspectionDecision.PENDING
            self.terminal_result = None
            self.consecutive_empty_frames = 0
            self.departure_cooldown_end = 0.0
            self.trigger.reset()
            self.collector.reset()
            self._update_live_telemetry()

    def _generate_reference_id(self) -> str:
        """Generates a human-readable unique Inspection Reference ID."""
        now = datetime.datetime.now()
        short_uuid = uuid.uuid4().hex[:4].upper()
        return f"INSP-{now.strftime('%Y%m%d-%H%M%S')}-{short_uuid}"

    def process_frame(
        self,
        raw_frame: np.ndarray,
        annotated_frame: np.ndarray,
        detection_result: DetectionResult,
        session_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Main pipeline hook called on every camera frame.
        Evaluates checkpoint trigger, advances state machine, runs multi-frame verification,
        and saves evidence on terminal decisions.
        """
        now = time.time()

        with self._lock:
            # Handle post-departure cooldown
            if now < self.departure_cooldown_end:
                self.state = InspectionState.IDLE
                self._update_live_telemetry(reason="Waiting for departure clearance...")
                return self.live_telemetry

            # 1. Evaluate Checkpoint Zone Entry
            in_zone, primary_person, multiple_workers, zone_reason = self.trigger.evaluate_entry(
                detection_result.persons, raw_frame.shape
            )

            # =================================================================
            # STATE: IDLE
            # =================================================================
            if self.state == InspectionState.IDLE:
                if in_zone and not multiple_workers:
                    # New worker entered checkpoint zone
                    self.state = InspectionState.DETECTING
                    self.current_reference = self._generate_reference_id()
                    self.entry_timestamp = now
                    self.collector.reset()
                    self._update_live_telemetry(
                        person_detected=True,
                        person_conf=primary_person.confidence if primary_person else 0.0,
                        reason="Worker detected. Stabilizing in inspection zone...",
                    )
                else:
                    self._update_live_telemetry(
                        person_detected=False,
                        reason="Inspection checkpoint ready. Awaiting worker...",
                    )
                return self.live_telemetry

            # =================================================================
            # STATE: DETECTING (Worker stabilizing)
            # =================================================================
            elif self.state == InspectionState.DETECTING:
                if not in_zone:
                    # Worker left before stabilizing
                    self.state = InspectionState.IDLE
                    self.trigger.reset()
                    self._update_live_telemetry(reason="Worker stepped out of inspection zone.")
                    return self.live_telemetry

                if multiple_workers:
                    # Multiple people crowded into zone
                    self._conclude_inspection(
                        decision=InspectionDecision.REINSPECT,
                        helmet_stat=PPEStatus.UNCERTAIN.value,
                        vest_stat=PPEStatus.UNCERTAIN.value,
                        reason="Multiple workers entered inspection zone simultaneously",
                        raw_frame=raw_frame,
                        annotated_frame=annotated_frame,
                        session_id=session_id,
                    )
                    return self.live_telemetry

                if zone_reason == "WORKER_STABILIZED":
                    # Position is stable, begin multi-frame verification
                    self.state = InspectionState.VERIFYING

            # =================================================================
            # STATE: VERIFYING (Gathering multi-frame evidence)
            # =================================================================
            if self.state == InspectionState.VERIFYING:
                elapsed = now - self.entry_timestamp

                if not in_zone:
                    # Left mid-inspection
                    self._conclude_inspection(
                        decision=InspectionDecision.REINSPECT,
                        helmet_stat=PPEStatus.UNCERTAIN.value,
                        vest_stat=PPEStatus.UNCERTAIN.value,
                        reason="Worker departed before inspection could complete",
                        raw_frame=raw_frame,
                        annotated_frame=annotated_frame,
                        session_id=session_id,
                    )
                    return self.live_telemetry

                # Associate PPE for the candidate person in the zone
                candidate_list = [primary_person] if primary_person else detection_result.persons
                associations = self.associator.associate(
                    persons=candidate_list,
                    helmets=detection_result.helmets,
                    vests=detection_result.vests,
                    frame_shape=raw_frame.shape,
                )

                if associations:
                    assoc = associations[0]
                    h_stat = PPEStatus.PRESENT if assoc.helmet_status == "YES" else (PPEStatus.MISSING if assoc.helmet_status == "NO" else PPEStatus.UNCERTAIN)
                    v_stat = PPEStatus.PRESENT if assoc.vest_status == "YES" else (PPEStatus.MISSING if assoc.vest_status == "NO" else PPEStatus.UNCERTAIN)

                    observation = InspectionFrameObservation(
                        timestamp=now,
                        frame_idx=self.collector.total_frames + 1,
                        person_box=assoc.person_box,
                        person_confidence=assoc.person_confidence,
                        helmet_status=h_stat,
                        vest_status=v_stat,
                        helmet_confidence=assoc.helmet_confidence,
                        vest_confidence=assoc.vest_confidence,
                        is_ambiguous=assoc.is_ambiguous or multiple_workers,
                        ambiguity_reason=assoc.ambiguity_reason,
                    )
                    self.collector.add_observation(observation, raw_frame, annotated_frame)

                # Evaluate consensus via PPEVerifier
                decision, h_final, v_final, verify_reason = self.verifier.evaluate_cycle(
                    self.collector,
                    elapsed_seconds=elapsed,
                    is_multiple_workers=multiple_workers,
                )

                if decision in [InspectionDecision.PASS, InspectionDecision.FAIL, InspectionDecision.REINSPECT]:
                    self._conclude_inspection(
                        decision=decision,
                        helmet_stat=h_final,
                        vest_stat=v_final,
                        reason=verify_reason,
                        raw_frame=raw_frame,
                        annotated_frame=annotated_frame,
                        session_id=session_id,
                    )
                else:
                    self._update_live_telemetry(
                        person_detected=True,
                        person_conf=primary_person.confidence if primary_person else 0.0,
                        helmet_stat=observation.helmet_status.value if associations else "UNCERTAIN",
                        vest_stat=observation.vest_status.value if associations else "UNCERTAIN",
                        helmet_conf=observation.helmet_confidence if associations else 0.0,
                        vest_conf=observation.vest_confidence if associations else 0.0,
                        duration=elapsed,
                        reason=verify_reason,
                    )

                return self.live_telemetry

            # =================================================================
            # STATE: TERMINAL HOLD (PASS / FAIL / REINSPECT)
            # Worker is viewing decision; must leave before next inspection starts
            # =================================================================
            elif self.state in [InspectionState.PASS, InspectionState.FAIL, InspectionState.REINSPECT]:
                if not in_zone:
                    self.consecutive_empty_frames += 1
                    if self.consecutive_empty_frames >= self.exit_confirmation_frames:
                        # Worker completely cleared checkpoint
                        self.state = InspectionState.IDLE
                        self.consecutive_empty_frames = 0
                        self.departure_cooldown_end = now + config.COOLDOWN_BETWEEN_WORKERS_SEC
                        self.trigger.reset()
                        self.collector.reset()
                        self._update_live_telemetry(reason="Checkpoint cleared. Ready for next worker.")
                else:
                    self.consecutive_empty_frames = 0
                    # Maintain displayed terminal result
                    self._update_live_telemetry()

                return self.live_telemetry

        return self.live_telemetry

    def _conclude_inspection(
        self,
        decision: InspectionDecision,
        helmet_stat: str,
        vest_stat: str,
        reason: str,
        raw_frame: np.ndarray,
        annotated_frame: np.ndarray,
        session_id: Optional[int] = None,
    ):
        """Concludes an inspection, generates result record, captures evidence, and dispatches to DB."""
        now = time.time()
        self.completion_timestamp = now
        duration = max(0.1, round(now - self.entry_timestamp, 2))
        self.terminal_decision = decision
        self.state = InspectionState(decision.value)

        avg_p, avg_h, avg_v, avg_overall = self.collector.get_average_confidences()

        # Identify missing PPE text
        missing_parts = []
        if helmet_stat == PPEStatus.MISSING.value:
            missing_parts.append("Hardhat")
        if vest_stat == PPEStatus.MISSING.value:
            missing_parts.append("Safety Vest")
        missing_str = " & ".join(missing_parts) if missing_parts else ("None" if decision == InspectionDecision.PASS else "Uncertain")

        # Save evidence if FAIL (or if configured for REINSPECT)
        orig_path = None
        annot_path = None

        if decision == InspectionDecision.FAIL or (decision == InspectionDecision.REINSPECT and self.collector.best_raw_frame is not None):
            today_str = datetime.date.today().isoformat()
            target_dir = self.violations_dir / today_str
            target_dir.mkdir(parents=True, exist_ok=True)

            ref_clean = self.current_reference.replace("-", "_")
            orig_file = target_dir / f"{ref_clean}_original.jpg"
            annot_file = target_dir / f"{ref_clean}_annotated.jpg"

            save_raw = self.collector.best_raw_frame if self.collector.best_raw_frame is not None else raw_frame
            save_annot = self.collector.best_annotated_frame if self.collector.best_annotated_frame is not None else annotated_frame

            cv2.imwrite(str(orig_file), save_raw)
            cv2.imwrite(str(annot_file), save_annot)

            orig_path = f"{today_str}/{orig_file.name}"
            annot_path = f"{today_str}/{annot_file.name}"

        # Create Result Object
        result = InspectionResult(
            inspection_reference=self.current_reference,
            entry_timestamp=self.entry_timestamp,
            completion_timestamp=self.completion_timestamp,
            duration=duration,
            helmet_status=helmet_stat,
            vest_status=vest_stat,
            decision=decision.value,
            person_confidence=avg_p,
            helmet_confidence=avg_h,
            vest_confidence=avg_v,
            average_confidence=avg_overall,
            reinspection_reason=reason,
            missing_ppe=missing_str,
            original_image_path=orig_path,
            annotated_image_path=annot_path,
            session_id=session_id,
            total_frames_analyzed=self.collector.total_frames,
        )
        self.terminal_result = result

        # Dispatch to SQLite database via callback
        if self.db_callback:
            try:
                self.db_callback(result)
            except Exception as e:
                print(f"[InspectionManager ERROR] DB callback failed: {e}")

        self._update_live_telemetry(
            helmet_stat=helmet_stat,
            vest_stat=vest_stat,
            helmet_conf=avg_h,
            vest_conf=avg_v,
            duration=duration,
            reason=f"{decision.value}: {reason}",
        )

    def _update_live_telemetry(
        self,
        person_detected: Optional[bool] = None,
        person_conf: Optional[float] = None,
        helmet_stat: Optional[str] = None,
        vest_stat: Optional[str] = None,
        helmet_conf: Optional[float] = None,
        vest_conf: Optional[float] = None,
        duration: Optional[float] = None,
        reason: Optional[str] = None,
    ):
        """Updates internal telemetry dictionary."""
        self.live_telemetry["state"] = self.state.value
        self.live_telemetry["reference"] = self.current_reference or "CHECKPOINT_READY"
        self.live_telemetry["decision"] = self.terminal_decision.value
        self.live_telemetry["frames_analyzed"] = self.collector.total_frames

        if person_detected is not None:
            self.live_telemetry["person_detected"] = person_detected
        if person_conf is not None:
            self.live_telemetry["person_confidence"] = round(person_conf, 2)
        if helmet_stat is not None:
            self.live_telemetry["helmet_status"] = helmet_stat
        if vest_stat is not None:
            self.live_telemetry["vest_status"] = vest_stat
        if helmet_conf is not None:
            self.live_telemetry["helmet_confidence"] = round(helmet_conf, 2)
        if vest_conf is not None:
            self.live_telemetry["vest_confidence"] = round(vest_conf, 2)
        if duration is not None:
            self.live_telemetry["duration"] = round(duration, 1)
        if reason is not None:
            self.live_telemetry["reason"] = reason

    def get_live_telemetry(self) -> Dict[str, Any]:
        """Thread-safe getter for current inspection state."""
        with self._lock:
            return dict(self.live_telemetry)
