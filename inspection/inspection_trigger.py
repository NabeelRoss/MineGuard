"""
Inspection Trigger Module for MineGuard AI.

Evaluates spatial position, sizing, stability, and singularity of workers
entering the designated checkpoint inspection zone.
"""

from typing import List, Tuple, Optional
import utils.config as config
from detection.detector import Detection


class InspectionTrigger:
    """
    Determines when a worker has entered the inspection zone,
    stabilized in position, and is ready for multi-frame PPE verification.
    """

    def __init__(
        self,
        zone_x_min: float = config.INSPECTION_ZONE_X_MIN,
        zone_x_max: float = config.INSPECTION_ZONE_X_MAX,
        zone_y_min: float = config.INSPECTION_ZONE_Y_MIN,
        zone_y_max: float = config.INSPECTION_ZONE_Y_MAX,
        min_person_height_px: int = config.MIN_PERSON_HEIGHT_PX,
        stable_frames_required: int = 2,
    ):
        self.zone_x_min = zone_x_min
        self.zone_x_max = zone_x_max
        self.zone_y_min = zone_y_min
        self.zone_y_max = zone_y_max
        self.min_height_px = min_person_height_px
        self.stable_frames_required = stable_frames_required

        self.consecutive_presence_count: int = 0
        self.last_candidate_box: Optional[List[float]] = None

    def reset(self):
        """Resets trigger state."""
        self.consecutive_presence_count = 0
        self.last_candidate_box = None

    def is_in_zone(self, box: List[float], frame_w: int, frame_h: int) -> bool:
        """
        Checks whether the center and body of the person bounding box
        falls inside the configured inspection zone.
        """
        cx = (box[0] + box[2]) / (2.0 * frame_w)
        cy = (box[1] + box[3]) / (2.0 * frame_h)
        height_px = box[3] - box[1]

        if height_px < self.min_height_px:
            return False

        return (
            self.zone_x_min <= cx <= self.zone_x_max
            and self.zone_y_min <= cy <= self.zone_y_max
        )

    def evaluate_entry(
        self,
        persons: List[Detection],
        frame_shape: tuple,
    ) -> Tuple[bool, Optional[Detection], bool, str]:
        """
        Evaluates current frame for checkpoint inspection eligibility.
        Returns:
            (in_zone: bool, primary_candidate: Optional[Detection], multiple_workers: bool, status_reason: str)
        """
        frame_h, frame_w = frame_shape[:2]

        # Find all persons currently in the inspection zone
        in_zone_persons = [p for p in persons if self.is_in_zone(p.box, frame_w, frame_h)]

        if not in_zone_persons:
            self.consecutive_presence_count = 0
            self.last_candidate_box = None
            return False, None, False, "INSPECTION_ZONE_EMPTY"

        # Check singularity: Exactly one worker should be in the inspection zone
        if len(in_zone_persons) > 1:
            self.consecutive_presence_count = 0
            return True, None, True, "MULTIPLE_WORKERS_IN_ZONE"

        # Exactly 1 person in the zone
        candidate = in_zone_persons[0]
        self.consecutive_presence_count += 1
        self.last_candidate_box = candidate.box

        is_stable = self.consecutive_presence_count >= self.stable_frames_required
        reason = "WORKER_STABILIZED" if is_stable else f"STABILIZING_{self.consecutive_presence_count}/{self.stable_frames_required}"

        return True, candidate, False, reason
