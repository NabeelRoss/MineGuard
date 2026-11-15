"""
Compliance Engine for MineGuard AI.

Evaluates worker PPE compliance status (SAFE, VIOLATION, UNKNOWN) across frames.
Implements:
1. Temporal violation confirmation (requires consecutive violation frames).
2. Alert cooldown timer to prevent alarm spam.
3. Duplicate alert prevention per worker ID.
4. Worker-wise compliance history and confidence tracking.
5. Automated snapshot generation on confirmed violation.
"""

import sys
import time
from pathlib import Path
from dataclasses import dataclass, field
from collections import deque
from typing import List, Dict, Optional, Tuple
import cv2
import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import utils.config as config
from compliance.association import AssociatedWorkerPPE


@dataclass
class ViolationAlert:
    """Represents a confirmed safety violation event."""
    alert_id: str
    track_id: int
    timestamp: float
    time_str: str
    violation_type: str        # "MISSING_HARDHAT", "MISSING_VEST", "MISSING_HARDHAT_AND_VEST"
    worker_box: List[float]
    confidence: float
    snapshot_path: Optional[str] = None


@dataclass
class WorkerComplianceState:
    """Stateful compliance history for an individual tracked worker."""
    track_id: int
    confirmed_status: str = "UNKNOWN"    # "SAFE", "VIOLATION", "UNKNOWN"
    instant_status: str = "UNKNOWN"
    helmet_status: str = "UNKNOWN"
    vest_status: str = "UNKNOWN"
    consecutive_violation_count: int = 0
    consecutive_safe_count: int = 0
    last_alert_time: float = 0.0
    total_violations_recorded: int = 0
    confidence_history: deque = field(default_factory=lambda: deque(maxlen=20))
    status_history: deque = field(default_factory=lambda: deque(maxlen=30))
    violation_reasons: List[str] = field(default_factory=list)

    @property
    def average_confidence(self) -> float:
        if not self.confidence_history:
            return 0.0
        return float(np.mean(self.confidence_history))


class ComplianceEngine:
    """
    Stateful PPE compliance engine. Evaluates single-frame associations,
    tracks temporal consistency, filters transient noise, and issues rate-limited alerts.
    """

    def __init__(
        self,
        confirmation_frames: int = config.VIOLATION_CONFIRMATION_FRAMES,
        alert_cooldown_seconds: float = config.ALERT_COOLDOWN,
        violations_dir: Optional[Path] = None,
    ):
        self.confirmation_frames = confirmation_frames
        self.alert_cooldown = alert_cooldown_seconds
        self.violations_dir = Path(violations_dir) if violations_dir else config.VIOLATIONS_DIR
        self.violations_dir.mkdir(parents=True, exist_ok=True)

        self.worker_states: Dict[int, WorkerComplianceState] = {}
        self.alert_history: List[ViolationAlert] = []
        self.alert_counter: int = 0

    def reset(self):
        """Clears all worker states and history."""
        self.worker_states.clear()
        self.alert_history.clear()
        self.alert_counter = 0

    def _determine_instant_status(self, assoc: AssociatedWorkerPPE) -> Tuple[str, List[str]]:
        """
        Determines the single-frame compliance status for a worker.
        Rules:
        - SAFE: Helmet YES + Vest YES.
        - VIOLATION: Helmet NO or Vest NO, provided observation is reliable.
        - UNKNOWN: Insufficient visibility, uncertain association or unreliable detection.
        """
        reasons = []

        h_stat = assoc.helmet_status
        v_stat = assoc.vest_status

        # Evaluate confirmed violations first
        has_helmet_violation = (h_stat == "NO" and assoc.head_visible)
        has_vest_violation = (v_stat == "NO" and assoc.torso_visible)

        if has_helmet_violation and has_vest_violation:
            reasons.append("Missing Hardhat & Safety Vest")
            return "VIOLATION", reasons
        elif has_helmet_violation:
            reasons.append("Missing Hardhat")
            return "VIOLATION", reasons
        elif has_vest_violation:
            reasons.append("Missing Safety Vest")
            return "VIOLATION", reasons

        # If both are verified YES
        if h_stat == "YES" and v_stat == "YES":
            return "SAFE", ["Fully Compliant"]

        # If one is YES but other is UNKNOWN (uncertain observation)
        if h_stat == "UNKNOWN" or v_stat == "UNKNOWN":
            unknown_parts = []
            if h_stat == "UNKNOWN":
                unknown_parts.append("Head visibility unconfirmed")
            if v_stat == "UNKNOWN":
                unknown_parts.append("Torso visibility unconfirmed")
            return "UNKNOWN", unknown_parts

        return "UNKNOWN", ["Observation Ambiguity"]

    def evaluate(
        self,
        associations: List[AssociatedWorkerPPE],
        frame: Optional[np.ndarray] = None,
        timestamp: Optional[float] = None,
    ) -> Tuple[Dict[int, WorkerComplianceState], List[ViolationAlert]]:
        """
        Updates compliance state machine for all workers in the current frame.
        Generates violation alerts when consecutive threshold is reached and cooldown has expired.
        """
        current_time = timestamp if timestamp is not None else time.time()
        new_alerts: List[ViolationAlert] = []
        active_track_ids = set()

        for assoc in associations:
            w_id = assoc.track_id
            active_track_ids.add(w_id)

            if w_id not in self.worker_states:
                self.worker_states[w_id] = WorkerComplianceState(track_id=w_id)

            state = self.worker_states[w_id]
            state.helmet_status = assoc.helmet_status
            state.vest_status = assoc.vest_status
            state.confidence_history.append(assoc.worker.confidence)

            # Determine instantaneous frame status
            instant_status, reasons = self._determine_instant_status(assoc)
            state.instant_status = instant_status
            state.violation_reasons = reasons
            state.status_history.append(instant_status)

            # Temporal filtering state transitions
            if instant_status == "VIOLATION":
                state.consecutive_violation_count += 1
                state.consecutive_safe_count = 0

                # Confirm violation after consecutive frames
                if state.consecutive_violation_count >= self.confirmation_frames:
                    state.confirmed_status = "VIOLATION"

                    # Check alert cooldown to prevent duplicate spamming
                    time_since_last_alert = current_time - state.last_alert_time
                    if time_since_last_alert >= self.alert_cooldown:
                        self.alert_counter += 1
                        state.last_alert_time = current_time
                        state.total_violations_recorded += 1

                        alert_id = f"ALERT-{self.alert_counter:04d}-W{w_id}"
                        time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(current_time))
                        v_type = "_".join(reasons).upper().replace(" ", "_").replace("&", "AND")

                        # Save snapshot if frame is provided
                        snapshot_path = None
                        if frame is not None and frame.size > 0:
                            filename = f"violation_w{w_id}_{int(current_time)}_{self.alert_counter}.jpg"
                            save_file = self.violations_dir / filename
                            # Crop or annotate violation snapshot
                            cv2.imwrite(str(save_file), frame)
                            snapshot_path = str(save_file)

                        alert = ViolationAlert(
                            alert_id=alert_id,
                            track_id=w_id,
                            timestamp=current_time,
                            time_str=time_str,
                            violation_type=v_type,
                            worker_box=assoc.box,
                            confidence=state.average_confidence,
                            snapshot_path=snapshot_path,
                        )
                        new_alerts.append(alert)
                        self.alert_history.append(alert)

            elif instant_status == "SAFE":
                state.consecutive_safe_count += 1
                state.consecutive_violation_count = 0

                # Require 3 consecutive safe frames to clear violation state
                if state.consecutive_safe_count >= 3:
                    state.confirmed_status = "SAFE"

            else:  # UNKNOWN
                # Do not immediately flip confirmed status on a transient unknown frame
                state.consecutive_violation_count = 0
                state.consecutive_safe_count = 0
                if state.confirmed_status not in ["SAFE", "VIOLATION"]:
                    state.confirmed_status = "UNKNOWN"

        return self.worker_states, new_alerts
