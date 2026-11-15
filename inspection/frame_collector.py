"""
Frame Collector Module for MineGuard AI.

Collects multi-frame observations, tracks running evidence samples, and preserves
the highest-confidence frames for evidence documentation.
"""

from typing import List, Optional, Tuple
import numpy as np
from inspection.inspection_result import InspectionFrameObservation, PPEStatus


class FrameCollector:
    """
    Buffers and aggregates consecutive frames during an active inspection cycle.
    """

    def __init__(self, max_buffer_size: int = 60):
        self.max_buffer_size = max_buffer_size
        self.observations: List[InspectionFrameObservation] = []
        self.best_raw_frame: Optional[np.ndarray] = None
        self.best_annotated_frame: Optional[np.ndarray] = None
        self.best_confidence: float = 0.0

    def reset(self):
        """Clears frame buffer and resets counters for a new inspection cycle."""
        self.observations.clear()
        self.best_raw_frame = None
        self.best_annotated_frame = None
        self.best_confidence = 0.0

    def add_observation(
        self,
        observation: InspectionFrameObservation,
        raw_frame: np.ndarray,
        annotated_frame: np.ndarray,
    ):
        """Adds a single verified frame observation to the collection buffer."""
        self.observations.append(observation)
        if len(self.observations) > self.max_buffer_size:
            self.observations.pop(0)

        # Retain best quality frames for evidence capture
        combined_conf = observation.person_confidence + (observation.helmet_confidence + observation.vest_confidence) / 2.0
        if combined_conf > self.best_confidence:
            self.best_confidence = combined_conf
            self.best_raw_frame = raw_frame.copy()
            self.best_annotated_frame = annotated_frame.copy()

    @property
    def total_frames(self) -> int:
        return len(self.observations)

    @property
    def safe_observations_count(self) -> int:
        return sum(1 for o in self.observations if o.is_safe)

    @property
    def violation_observations_count(self) -> int:
        return sum(1 for o in self.observations if o.is_violation)

    @property
    def ambiguous_observations_count(self) -> int:
        return sum(1 for o in self.observations if o.is_ambiguous or (o.helmet_status == PPEStatus.UNCERTAIN and o.vest_status == PPEStatus.UNCERTAIN))

    def get_average_confidences(self) -> Tuple[float, float, float, float]:
        """
        Calculates average confidences across all collected observations:
        Returns (person_conf, helmet_conf, vest_conf, overall_avg)
        """
        if not self.observations:
            return 0.0, 0.0, 0.0, 0.0

        p_confs = [o.person_confidence for o in self.observations]
        h_confs = [o.helmet_confidence for o in self.observations if o.helmet_confidence > 0]
        v_confs = [o.vest_confidence for o in self.observations if o.vest_confidence > 0]

        avg_p = float(np.mean(p_confs)) if p_confs else 0.0
        avg_h = float(np.mean(h_confs)) if h_confs else 0.0
        avg_v = float(np.mean(v_confs)) if v_confs else 0.0
        overall = float(np.mean([avg_p, avg_h if avg_h > 0 else avg_p, avg_v if avg_v > 0 else avg_p]))

        return round(avg_p, 3), round(avg_h, 3), round(avg_v, 3), round(overall, 3)
