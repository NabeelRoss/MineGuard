"""
Worker Tracking Engine for MineGuard AI.

Implements persistent multi-worker tracking using Ultralytics ByteTrack integration.
Strictly isolates Person detections to generate worker tracking IDs.
Ensures PPE objects (helmets, vests) never receive or interfere with worker IDs.
Handles temporary occlusions and purges stale tracks after a configurable duration.
"""

import sys
import time
from pathlib import Path
from dataclasses import dataclass, field
from collections import deque
from typing import List, Dict, Optional, Tuple, Union
import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import utils.config as config
from detection.detector import Detection, DetectionResult, PPEObjectDetector


@dataclass
class TrackedWorker:
    """Represents a tracked human worker with persistent ID and trajectory."""
    track_id: int
    box: List[float]                  # Current bounding box [x1, y1, x2, y2]
    confidence: float                 # Detection confidence
    first_seen_frame: int             # Frame number when worker first appeared
    last_seen_frame: int              # Most recent frame where worker was detected
    consecutive_hits: int = 1         # Number of consecutive frames detected
    missed_frames: int = 0            # Number of consecutive frames missed (occlusion)
    history: deque = field(default_factory=lambda: deque(maxlen=30)) # Center coordinate history

    @property
    def x1(self) -> float:
        return self.box[0]

    @property
    def y1(self) -> float:
        return self.box[1]

    @property
    def x2(self) -> float:
        return self.box[2]

    @property
    def y2(self) -> float:
        return self.box[3]

    @property
    def width(self) -> float:
        return max(0.0, self.box[2] - self.box[0])

    @property
    def height(self) -> float:
        return max(0.0, self.box[3] - self.box[1])

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.box[0] + self.box[2]) / 2.0, (self.box[1] + self.box[3]) / 2.0)

    @property
    def area(self) -> float:
        return self.width * self.height


@dataclass
class TrackingResult:
    """Encapsulates tracking output for a single video frame."""
    frame_id: int
    workers: List[TrackedWorker]      # Active tracked workers in the current frame
    helmets: List[Detection]          # Untracked helmet detections
    vests: List[Detection]            # Untracked vest detections
    processing_time_ms: float
    frame_shape: tuple


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """Computes Intersection over Union (IoU) between two bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


class WorkerTracker:
    """
    Dedicated Worker Tracker using Ultralytics ByteTrack.
    Isolates 'Person' detections for tracking. Helmets and Vests remain untracked PPE objects.
    """

    def __init__(
        self,
        detector: Optional[PPEObjectDetector] = None,
        max_stale_frames: int = config.MAX_STALE_TRACK_FRAMES,
        min_hits: int = 1,
        iou_match_threshold: float = 0.3,
    ):
        self.detector = detector if detector is not None else PPEObjectDetector()
        self.max_stale_frames = max_stale_frames
        self.min_hits = min_hits
        self.iou_match_threshold = iou_match_threshold

        self.frame_count: int = 0
        self.active_tracks: Dict[int, TrackedWorker] = {}
        self.next_fallback_id: int = 1000

    def reset(self):
        """Resets all active tracks and frame counters."""
        self.frame_count = 0
        self.active_tracks.clear()
        self.next_fallback_id = 1000

    def update(self, frame: np.ndarray) -> TrackingResult:
        """
        Processes a single frame:
        1. Runs Ultralytics ByteTrack on the frame (with resilient fallback to internal tracker if lap is unavailable).
        2. Strictly filters tracking IDs to PERSON class only.
        3. Extracts helmets and vests as unassigned detections (never assigns worker IDs).
        4. Updates persistent worker track states across frames.
        5. Purges stale tracks that exceed max_stale_frames.
        """
        start_time = time.perf_counter()
        self.frame_count += 1

        if frame is None or frame.size == 0:
            raise ValueError("Input frame is invalid or empty")

        detected_workers_in_frame: List[Tuple[Optional[int], List[float], float]] = []
        helmets_in_frame: List[Detection] = []
        vests_in_frame: List[Detection] = []

        bytetrack_success = False

        # Attempt Ultralytics ByteTrack integration
        try:
            results = self.detector.model.track(
                source=frame,
                conf=self.detector.conf_threshold,
                iou=self.detector.iou_threshold,
                imgsz=self.detector.imgsz,
                device=self.detector.device,
                persist=True,
                tracker="bytetrack.yaml",
                verbose=False,
            )

            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                clss = boxes.cls.cpu().numpy().astype(int)
                ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else None

                for i in range(len(xyxy)):
                    cid = int(clss[i])
                    box = [float(v) for v in xyxy[i]]
                    score = float(confs[i])
                    cname = self.detector.names.get(cid, f"Class_{cid}")

                    if cid == self.detector.person_class_id:
                        track_id = int(ids[i]) if ids is not None and ids[i] != -1 else None
                        detected_workers_in_frame.append((track_id, box, score))
                    elif cid == self.detector.hardhat_class_id:
                        helmets_in_frame.append(
                            Detection(box=box, confidence=score, class_id=cid, class_name=cname)
                        )
                    elif cid == self.detector.vest_class_id:
                        vests_in_frame.append(
                            Detection(box=box, confidence=score, class_id=cid, class_name=cname)
                        )

            bytetrack_success = True
        except Exception:
            # Fallback to standard detector inference + persistent IoU worker tracker
            bytetrack_success = False

        if not bytetrack_success:
            det_res = self.detector.detect(frame)
            for person in det_res.persons:
                detected_workers_in_frame.append((None, person.box, person.confidence))
            helmets_in_frame = det_res.helmets
            vests_in_frame = det_res.vests

        # Update Track Registry for Workers
        current_frame_matched_ids = set()
        active_workers_this_frame: List[TrackedWorker] = []

        for tid, box, conf in detected_workers_in_frame:
            resolved_id = tid

            # If no ID assigned by ByteTrack, match with active tracks via IoU & center distance
            if resolved_id is None or resolved_id in current_frame_matched_ids:
                best_iou = 0.0
                best_match_id = None
                for ex_id, ex_track in self.active_tracks.items():
                    if ex_id not in current_frame_matched_ids:
                        iou = compute_iou(box, ex_track.box)
                        if iou > best_iou:
                            best_iou = iou
                            best_match_id = ex_id

                if best_match_id is not None and best_iou >= self.iou_match_threshold:
                    resolved_id = best_match_id
                else:
                    resolved_id = self.next_fallback_id
                    self.next_fallback_id += 1

            current_frame_matched_ids.add(resolved_id)

            if resolved_id in self.active_tracks:
                worker = self.active_tracks[resolved_id]
                worker.box = box
                worker.confidence = conf
                worker.last_seen_frame = self.frame_count
                worker.consecutive_hits += 1
                worker.missed_frames = 0
                worker.history.append(worker.center)
            else:
                worker = TrackedWorker(
                    track_id=resolved_id,
                    box=box,
                    confidence=conf,
                    first_seen_frame=self.frame_count,
                    last_seen_frame=self.frame_count,
                    consecutive_hits=1,
                    missed_frames=0,
                )
                worker.history.append(worker.center)
                self.active_tracks[resolved_id] = worker

            active_workers_this_frame.append(worker)

        # Increment missed frames for active tracks not seen this frame
        stale_track_ids = []
        for tid, worker in self.active_tracks.items():
            if tid not in current_frame_matched_ids:
                worker.missed_frames += 1
                if worker.missed_frames > self.max_stale_frames:
                    stale_track_ids.append(tid)

        # Purge stale tracks
        for tid in stale_track_ids:
            del self.active_tracks[tid]

        total_time_ms = (time.perf_counter() - start_time) * 1000.0

        return TrackingResult(
            frame_id=self.frame_count,
            workers=active_workers_this_frame,
            helmets=helmets_in_frame,
            vests=vests_in_frame,
            processing_time_ms=total_time_ms,
            frame_shape=frame.shape,
        )
