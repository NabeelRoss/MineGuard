"""
PPE Association Module for MineGuard AI.

Implements geometric anatomical constraint matching to link detected helmets
and safety vests to human worker bounding boxes.
Features:
1. Normalized anatomical head and torso inspection zones.
2. One-to-one bipartite matching (no duplicate assignments).
3. Inter-person overlap and ambiguity detection (triggers REINSPECT on occlusions).
4. Boundary truncation protection (head/torso cut off by camera frame edges).
5. Implausible association rejection (geometric and spatial sanity checks).
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Union, Any
import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import utils.config as config
from detection.detector import Detection


@dataclass
class AssociatedWorkerPPE:
    """Encapsulates a person bounding box with assigned PPE and diagnostic flags."""
    person_box: List[float]           # [x1, y1, x2, y2]
    person_confidence: float
    worker_id: Optional[int] = None   # Optional tracking ID (if tracking enabled)
    helmet: Optional[Detection] = None
    vest: Optional[Detection] = None
    helmet_status: str = "UNKNOWN"    # "YES", "NO", "UNKNOWN"
    vest_status: str = "UNKNOWN"      # "YES", "NO", "UNKNOWN"
    helmet_confidence: float = 0.0
    vest_confidence: float = 0.0
    head_visible: bool = True
    torso_visible: bool = True
    is_ambiguous: bool = False
    ambiguity_reason: str = ""

    @property
    def box(self) -> List[float]:
        return self.person_box

    @property
    def track_id(self) -> int:
        return self.worker_id if self.worker_id is not None else 1


def bbox_intersection(boxA: List[float], boxB: List[float]) -> float:
    """Computes area of intersection between boxA and boxB."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    return inter_w * inter_h


def bbox_area(box: List[float]) -> float:
    """Computes area of a bounding box [x1, y1, x2, y2]."""
    w = max(0.0, box[2] - box[0])
    h = max(0.0, box[3] - box[1])
    return w * h


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """Computes Intersection over Union (IoU) between two boxes."""
    inter = bbox_intersection(box1, box2)
    a1 = bbox_area(box1)
    a2 = bbox_area(box2)
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


class PPEAssociator:
    """
    Associates detected PPE detections (helmets, vests) with human candidates
    using anatomical geometric constraint matching.
    """

    def __init__(
        self,
        head_top_offset_ratio: float = config.HEAD_TOP_OFFSET_RATIO,
        head_bottom_ratio: float = config.HEAD_BOTTOM_RATIO,
        head_horizontal_expansion: float = config.HEAD_HORIZONTAL_EXPANSION,
        torso_top_ratio: float = config.TORSO_TOP_RATIO,
        torso_bottom_ratio: float = config.TORSO_BOTTOM_RATIO,
        torso_horizontal_expansion: float = config.TORSO_HORIZONTAL_EXPANSION,
        min_overlap_ratio: float = config.MIN_PPE_OVERLAP_RATIO,
        max_h_offset: float = config.MAX_HORIZONTAL_CENTER_OFFSET,
        max_inter_person_iou: float = config.MAX_INTER_PERSON_IOU,
    ):
        self.head_top_offset = head_top_offset_ratio
        self.head_bottom = head_bottom_ratio
        self.head_h_exp = head_horizontal_expansion
        self.torso_top = torso_top_ratio
        self.torso_bottom = torso_bottom_ratio
        self.torso_h_exp = torso_horizontal_expansion
        self.min_overlap = min_overlap_ratio
        self.max_h_offset = max_h_offset
        self.max_person_iou = max_inter_person_iou

    def get_head_zone(self, p_box: List[float], frame_h: int, frame_w: int) -> List[float]:
        """Calculates expected head zone bounding box from person coordinates."""
        px1, py1, px2, py2 = p_box
        pw = px2 - px1
        ph = py2 - py1

        hx1 = max(0.0, px1 - self.head_h_exp * pw)
        hy1 = max(0.0, py1 - self.head_top_offset * ph)
        hx2 = min(float(frame_w), px2 + self.head_h_exp * pw)
        hy2 = min(float(frame_h), py1 + self.head_bottom * ph)
        return [hx1, hy1, hx2, hy2]

    def get_torso_zone(self, p_box: List[float], frame_h: int, frame_w: int) -> List[float]:
        """Calculates expected torso zone bounding box from person coordinates."""
        px1, py1, px2, py2 = p_box
        pw = px2 - px1
        ph = py2 - py1

        tx1 = max(0.0, px1 - self.torso_h_exp * pw)
        ty1 = max(0.0, py1 + self.torso_top * ph)
        tx2 = min(float(frame_w), px2 + self.torso_h_exp * pw)
        ty2 = min(float(frame_h), py1 + self.torso_bottom * ph)
        return [tx1, ty1, tx2, ty2]

    def check_visibility(
        self, p_box: List[float], frame_h: int, frame_w: int, edge_margin: int = 15
    ) -> Tuple[bool, bool, str]:
        """
        Evaluates whether person's head and torso are reliably visible in frame.
        Flags truncation when body parts are cut off at frame borders.
        """
        px1, py1, px2, py2 = p_box
        head_visible = True
        torso_visible = True
        reason = ""

        # Head truncation at top of image
        if py1 <= edge_margin:
            head_visible = False
            reason += "Head truncated at frame top; "

        # Person box too small for reliable PPE observation
        box_height = py2 - py1
        if box_height < 80:
            head_visible = False
            torso_visible = False
            reason += f"Person too distant/small ({int(box_height)}px); "

        # Truncation at bottom before torso finishes
        if py2 >= (frame_h - edge_margin) and box_height < 120:
            torso_visible = False
            reason += "Torso cut off at frame bottom; "

        # Left / Right frame edge cutoffs
        if (px1 <= edge_margin and px2 < 50) or (px2 >= (frame_w - edge_margin) and (frame_w - px1) < 50):
            head_visible = False
            torso_visible = False
            reason += "Worker truncated at frame lateral edge; "

        return head_visible, torso_visible, reason.strip()

    def detect_inter_person_occlusion(self, persons: List[Any]) -> Dict[int, str]:
        """
        Detects significant overlap between multiple persons in view.
        Returns a dictionary of {person_idx: reason} for persons with ambiguous overlap.
        """
        ambiguous_persons = {}
        n = len(persons)
        for i in range(n):
            box_i = persons[i].box if hasattr(persons[i], "box") else persons[i]
            for j in range(i + 1, n):
                box_j = persons[j].box if hasattr(persons[j], "box") else persons[j]
                iou = compute_iou(box_i, box_j)
                if iou >= self.max_person_iou:
                    ambiguous_persons[i] = f"Overlapping with another person (IoU {iou:.2f})"
                    ambiguous_persons[j] = f"Overlapping with another person (IoU {iou:.2f})"
        return ambiguous_persons

    def compute_helmet_affinity(
        self, person_box: List[float], helmet: Detection, frame_h: int, frame_w: int
    ) -> float:
        """
        Computes affinity score [0.0 - 1.0] between a person and a candidate helmet detection.
        Rejects implausible spatial associations.
        """
        head_zone = self.get_head_zone(person_box, frame_h, frame_w)
        h_area = bbox_area(helmet.box)
        p_area = bbox_area(person_box)

        if h_area <= 0 or p_area <= 0:
            return 0.0

        # Implausible size rejection: Helmet cannot be larger than 50% of the entire person box
        if h_area > 0.50 * p_area:
            return 0.0

        inter_area = bbox_intersection(head_zone, helmet.box)
        overlap_ratio = inter_area / h_area  # IoA (Intersection over Helmet Area)

        if overlap_ratio < self.min_overlap:
            return 0.0

        # Horizontal alignment check
        p_cx = (person_box[0] + person_box[2]) / 2.0
        p_w = max(1.0, person_box[2] - person_box[0])
        h_cx = helmet.center_x
        h_offset = abs(h_cx - p_cx) / p_w

        if h_offset > self.max_h_offset:
            return 0.0

        # Vertical center sanity check
        if not (head_zone[1] <= helmet.center_y <= head_zone[3]):
            return 0.0

        alignment_score = max(0.0, 1.0 - (h_offset / self.max_h_offset))
        affinity = 0.5 * overlap_ratio + 0.3 * alignment_score + 0.2 * helmet.confidence
        return float(affinity)

    def compute_vest_affinity(
        self, person_box: List[float], vest: Detection, frame_h: int, frame_w: int
    ) -> float:
        """
        Computes affinity score [0.0 - 1.0] between a person and a candidate safety vest detection.
        Rejects implausible spatial associations.
        """
        torso_zone = self.get_torso_zone(person_box, frame_h, frame_w)
        v_area = bbox_area(vest.box)
        p_area = bbox_area(person_box)

        if v_area <= 0 or p_area <= 0:
            return 0.0

        # Implausible size rejection: Vest cannot exceed 85% of person area or be smaller than 5%
        if v_area > 0.85 * p_area or v_area < 0.05 * p_area:
            return 0.0

        inter_area = bbox_intersection(torso_zone, vest.box)
        overlap_ratio = inter_area / v_area  # IoA (Intersection over Vest Area)

        if overlap_ratio < self.min_overlap:
            return 0.0

        # Horizontal alignment check
        p_cx = (person_box[0] + person_box[2]) / 2.0
        p_w = max(1.0, person_box[2] - person_box[0])
        v_cx = vest.center_x
        v_offset = abs(v_cx - p_cx) / p_w

        if v_offset > self.max_h_offset:
            return 0.0

        # Vertical center sanity check
        if not (torso_zone[1] <= vest.center_y <= torso_zone[3]):
            return 0.0

        alignment_score = max(0.0, 1.0 - (v_offset / self.max_h_offset))
        affinity = 0.5 * overlap_ratio + 0.3 * alignment_score + 0.2 * vest.confidence
        return float(affinity)

    def associate(
        self,
        persons: List[Union[Detection, Any]],
        helmets: List[Detection],
        vests: List[Detection],
        frame_shape: tuple,
    ) -> List[AssociatedWorkerPPE]:
        """
        Performs one-to-one bipartite matching between persons and detected PPE.
        Flags ambiguous inter-person overlaps and truncation errors.
        Outputs AssociatedWorkerPPE records.
        """
        frame_h, frame_w = frame_shape[:2]
        if not persons:
            return []

        # Detect inter-person overlap ambiguity
        occlusion_map = self.detect_inter_person_occlusion(persons)

        # 1. Initialize result structure with visibility diagnostics
        associations: List[AssociatedWorkerPPE] = []
        for idx, person in enumerate(persons):
            p_box = person.box if hasattr(person, "box") else person
            p_conf = person.confidence if hasattr(person, "confidence") else 1.0
            w_id = getattr(person, "track_id", idx + 1)

            head_vis, torso_vis, reason = self.check_visibility(p_box, frame_h, frame_w)
            is_ambiguous = idx in occlusion_map
            if is_ambiguous:
                reason = (reason + " " + occlusion_map[idx]).strip()

            rec = AssociatedWorkerPPE(
                person_box=p_box,
                person_confidence=p_conf,
                worker_id=w_id,
                head_visible=head_vis,
                torso_visible=torso_vis,
                is_ambiguous=is_ambiguous,
                ambiguity_reason=reason,
            )
            associations.append(rec)

        # 2. Match Helmets to Persons (Greedy One-to-One Matching by Affinity)
        if helmets:
            helmet_candidates = []
            for h_idx, helmet in enumerate(helmets):
                for p_idx, rec in enumerate(associations):
                    aff = self.compute_helmet_affinity(rec.person_box, helmet, frame_h, frame_w)
                    if aff > 0.0:
                        helmet_candidates.append((aff, h_idx, p_idx))

            helmet_candidates.sort(key=lambda x: x[0], reverse=True)
            assigned_helmets = set()
            assigned_person_helmets = set()

            for aff, h_idx, p_idx in helmet_candidates:
                if h_idx not in assigned_helmets and p_idx not in assigned_person_helmets:
                    # Check if second-best candidate person has nearly identical affinity
                    competing = [c for c in helmet_candidates if c[1] == h_idx and c[2] != p_idx and c[0] > 0.85 * aff]
                    if competing:
                        # Ambiguous assignment across multiple people!
                        associations[p_idx].is_ambiguous = True
                        associations[p_idx].ambiguity_reason += " Ambiguous helmet assignment with nearby person;"
                        continue

                    assigned_helmets.add(h_idx)
                    assigned_person_helmets.add(p_idx)
                    matched_helmet = helmets[h_idx]
                    rec = associations[p_idx]
                    rec.helmet = matched_helmet
                    rec.helmet_status = "YES"
                    rec.helmet_confidence = matched_helmet.confidence

        # 3. Match Safety Vests to Persons (Greedy One-to-One Matching by Affinity)
        if vests:
            vest_candidates = []
            for v_idx, vest in enumerate(vests):
                for p_idx, rec in enumerate(associations):
                    aff = self.compute_vest_affinity(rec.person_box, vest, frame_h, frame_w)
                    if aff > 0.0:
                        vest_candidates.append((aff, v_idx, p_idx))

            vest_candidates.sort(key=lambda x: x[0], reverse=True)
            assigned_vests = set()
            assigned_person_vests = set()

            for aff, v_idx, p_idx in vest_candidates:
                if v_idx not in assigned_vests and p_idx not in assigned_person_vests:
                    competing = [c for c in vest_candidates if c[1] == v_idx and c[2] != p_idx and c[0] > 0.85 * aff]
                    if competing:
                        associations[p_idx].is_ambiguous = True
                        associations[p_idx].ambiguity_reason += " Ambiguous vest assignment with nearby person;"
                        continue

                    assigned_vests.add(v_idx)
                    assigned_person_vests.add(p_idx)
                    matched_vest = vests[v_idx]
                    rec = associations[p_idx]
                    rec.vest = matched_vest
                    rec.vest_status = "YES"
                    rec.vest_confidence = matched_vest.confidence

        # 4. Resolve Status for Unassigned PPE on Each Person
        for rec in associations:
            if rec.is_ambiguous:
                if rec.helmet_status != "YES":
                    rec.helmet_status = "UNKNOWN"
                if rec.vest_status != "YES":
                    rec.vest_status = "UNKNOWN"
                continue

            # Resolve Helmet Status
            if rec.helmet_status != "YES":
                if not rec.head_visible:
                    rec.helmet_status = "UNKNOWN"
                else:
                    rec.helmet_status = "NO"

            # Resolve Vest Status
            if rec.vest_status != "YES":
                if not rec.torso_visible:
                    rec.vest_status = "UNKNOWN"
                else:
                    rec.vest_status = "NO"

        return associations
