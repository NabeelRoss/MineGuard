"""
Visualization Module for MineGuard AI.

Renders clear, intuitive computer vision overlays on video frames for the
automated quality inspection station:
1. Checkpoint Inspection Zone bounding guidelines.
2. Large high-visibility status indicators:
   - PASS: Green (0, 220, 0)
   - FAIL: Red (0, 0, 240)
   - REINSPECT: Amber (0, 165, 255)
   - VERIFYING: Sky Blue (255, 180, 0)
   - IDLE: Slate (120, 140, 160)
3. Top Heads-Up Display (HUD) banner with Inspection Reference, FPS, latency, and PPE states.
4. Clean, non-cluttering bounding boxes and badges for Person, Hardhat, and Safety Vest.
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import cv2
import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import utils.config as config
from detection.detector import Detection, DetectionResult
from compliance.association import AssociatedWorkerPPE

# BGR Color Definitions
COLOR_PASS = (0, 220, 0)         # Green
COLOR_FAIL = (0, 0, 240)         # Red
COLOR_REINSPECT = (0, 165, 255)   # Amber
COLOR_VERIFYING = (255, 190, 0)   # Sky Blue
COLOR_IDLE = (100, 116, 139)     # Slate Gray
COLOR_HELMET = (0, 230, 255)     # Yellow / Gold
COLOR_VEST = (50, 120, 255)      # Deep Orange
COLOR_TEXT = (255, 255, 255)     # White
COLOR_BANNER = (20, 28, 45)      # Dark Navy Banner


def get_decision_color(state: str) -> Tuple[int, int, int]:
    """Returns corresponding BGR color for inspection state."""
    st = state.upper().strip()
    if st == "PASS":
        return COLOR_PASS
    elif st == "FAIL":
        return COLOR_FAIL
    elif st == "REINSPECT":
        return COLOR_REINSPECT
    elif st in ["VERIFYING", "DETECTING"]:
        return COLOR_VERIFYING
    else:
        return COLOR_IDLE


def draw_corner_rect(
    img: np.ndarray,
    box: List[float],
    color: Tuple[int, int, int],
    thickness: int = 2,
    corner_len: int = 18,
):
    """Draws sleek bounding box with highlighted corners."""
    x1, y1, x2, y2 = [int(v) for v in box]
    # Primary bounding box
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)

    # Top-Left Corner
    cv2.line(img, (x1, y1), (x1 + corner_len, y1), color, thickness)
    cv2.line(img, (x1, y1), (x1, y1 + corner_len), color, thickness)

    # Top-Right Corner
    cv2.line(img, (x2, y1), (x2 - corner_len, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 - corner_len), color, thickness)

    # Bottom-Left Corner
    cv2.line(img, (x1, y2), (x1 + corner_len, y2), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 + corner_len), color, thickness)

    # Bottom-Right Corner
    cv2.line(img, (x2, y2), (x2 - corner_len, y2), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - corner_len), color, thickness)


def draw_badge(
    img: np.ndarray,
    text: str,
    org: Tuple[int, int],
    bg_color: Tuple[int, int, int],
    text_color: Tuple[int, int, int] = COLOR_TEXT,
    font_scale: float = 0.5,
    thickness: int = 1,
    padding: int = 5,
):
    """Draws pill badge background behind text string."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = org
    y1 = max(0, y - th - padding * 2)
    y2 = y + baseline
    x1 = max(0, x - padding)
    x2 = min(img.shape[1], x + tw + padding * 2)

    cv2.rectangle(img, (x1, y1), (x2, y2), bg_color, -1)
    cv2.putText(
        img,
        text,
        (x + padding, y - padding),
        font,
        font_scale,
        text_color,
        thickness,
        cv2.LINE_AA,
    )


class MineGuardVisualizer:
    """
    Renders the unified MineGuard AI visual interface for Checkpoint Inspection.
    """

    def __init__(self):
        pass

    def draw_inspection_zone(
        self,
        canvas: np.ndarray,
        state: str,
    ):
        """Draws designated checkpoint inspection zone boundaries on the frame."""
        h, w = canvas.shape[:2]
        zx1 = int(config.INSPECTION_ZONE_X_MIN * w)
        zx2 = int(config.INSPECTION_ZONE_X_MAX * w)
        zy1 = int(config.INSPECTION_ZONE_Y_MIN * h)
        zy2 = int(config.INSPECTION_ZONE_Y_MAX * h)

        color = get_decision_color(state)

        # Subtle corner guidelines for inspection zone
        corner_len = 24
        thick = 2 if state in ["PASS", "FAIL", "REINSPECT", "VERIFYING"] else 1

        # Top-Left
        cv2.line(canvas, (zx1, zy1), (zx1 + corner_len, zy1), color, thick)
        cv2.line(canvas, (zx1, zy1), (zx1, zy1 + corner_len), color, thick)
        # Top-Right
        cv2.line(canvas, (zx2, zy1), (zx2 - corner_len, zy1), color, thick)
        cv2.line(canvas, (zx2, zy1), (zx2, zy1 + corner_len), color, thick)
        # Bottom-Left
        cv2.line(canvas, (zx1, zy2), (zx1 + corner_len, zy2), color, thick)
        cv2.line(canvas, (zx1, zy2), (zx1, zy2 - corner_len), color, thick)
        # Bottom-Right
        cv2.line(canvas, (zx2, zy2), (zx2 - corner_len, zy2), color, thick)
        cv2.line(canvas, (zx2, zy2), (zx2, zy2 - corner_len), color, thick)

        # Small bottom label
        zone_label = "INSPECTION ZONE"
        font = cv2.FONT_HERSHEY_SIMPLEX
        (lw, lh), _ = cv2.getTextSize(zone_label, font, 0.38, 1)
        cv2.putText(
            canvas,
            zone_label,
            (w // 2 - lw // 2, zy2 - 8),
            font,
            0.38,
            color,
            1,
            cv2.LINE_AA,
        )

    def draw_hud(
        self,
        canvas: np.ndarray,
        fps: float,
        latency_ms: float,
        telemetry: Dict[str, Any],
    ):
        """Draws top HUD banner with checkpoint state and real-time statistics."""
        h, w = canvas.shape[:2]
        hud_h = 44

        # Semi-transparent top bar
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (w, hud_h), COLOR_BANNER, -1)
        cv2.addWeighted(overlay, 0.88, canvas, 0.12, 0, canvas)
        cv2.line(canvas, (0, hud_h), (w, hud_h), (40, 55, 80), 1)

        font = cv2.FONT_HERSHEY_SIMPLEX

        # Title
        cv2.putText(
            canvas,
            "MINEGUARD AI",
            (14, 28),
            font,
            0.62,
            (255, 215, 0),
            2,
            cv2.LINE_AA,
        )

        state = telemetry.get("state", "IDLE")
        state_color = get_decision_color(state)

        # Inspection State Badge
        state_text = f"STATE: {state}"
        (sw, _), _ = cv2.getTextSize(state_text, font, 0.48, 1)
        badge_x = 180
        cv2.rectangle(canvas, (badge_x - 6, 8), (badge_x + sw + 8, 36), state_color, -1)
        text_color = (0, 0, 0) if state not in ["FAIL"] else (255, 255, 255)
        cv2.putText(canvas, state_text, (badge_x, 26), font, 0.48, text_color, 2, cv2.LINE_AA)

        # Inspection Reference
        ref = telemetry.get("reference", "READY")
        if ref and ref != "CHECKPOINT_READY":
            cv2.putText(canvas, ref, (badge_x + sw + 20, 27), font, 0.42, (200, 215, 235), 1, cv2.LINE_AA)

        # FPS & Latency (Right-Aligned)
        perf_text = f"{fps:.1f} FPS | {latency_ms:.1f}ms"
        (pw, _), _ = cv2.getTextSize(perf_text, font, 0.42, 1)
        cv2.putText(canvas, perf_text, (w - pw - 14, 27), font, 0.42, (160, 180, 205), 1, cv2.LINE_AA)

    def draw_terminal_decision_banner(
        self,
        canvas: np.ndarray,
        state: str,
        reason: str,
    ):
        """Draws prominent visual banner when inspection reaches PASS / FAIL / REINSPECT."""
        if state not in ["PASS", "FAIL", "REINSPECT"]:
            return

        h, w = canvas.shape[:2]
        color = get_decision_color(state)
        banner_h = 56
        y1 = h - banner_h
        y2 = h

        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, y1), (w, y2), color, -1)
        cv2.addWeighted(overlay, 0.85, canvas, 0.15, 0, canvas)
        cv2.line(canvas, (0, y1), (w, y1), (255, 255, 255), 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        title = f"[ CHECKPOINT DECISION: {state} ]"
        text_color = (0, 0, 0) if state != "FAIL" else (255, 255, 255)

        cv2.putText(canvas, title, (20, y1 + 24), font, 0.65, text_color, 2, cv2.LINE_AA)
        cv2.putText(canvas, reason, (20, y1 + 46), font, 0.45, text_color, 1, cv2.LINE_AA)

    def draw_ppe_detections(
        self,
        canvas: np.ndarray,
        helmets: List[Detection],
        vests: List[Detection],
    ):
        """Draws non-cluttering bounding boxes around detected PPE."""
        # Draw Helmets
        for h in helmets:
            x1, y1, x2, y2 = [int(v) for v in h.box]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), COLOR_HELMET, 1, cv2.LINE_AA)
            label = f"Hardhat {h.confidence:.2f}"
            draw_badge(canvas, label, (x1, y1 - 2), COLOR_BANNER, COLOR_HELMET, font_scale=0.38)

        # Draw Vests
        for v in vests:
            x1, y1, x2, y2 = [int(v) for v in v.box]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), COLOR_VEST, 1, cv2.LINE_AA)
            label = f"Vest {v.confidence:.2f}"
            draw_badge(canvas, label, (x1, y1 - 2), COLOR_BANNER, COLOR_VEST, font_scale=0.38)

    def draw_persons(
        self,
        canvas: np.ndarray,
        persons: List[Detection],
        state: str,
    ):
        """Renders bounding box and status on inspected person."""
        color = get_decision_color(state)
        for p in persons:
            draw_corner_rect(canvas, p.box, color, thickness=2, corner_len=18)
            label = f"Worker {p.confidence:.2f}"
            x1, y1 = int(p.box[0]), int(p.box[1])
            draw_badge(canvas, label, (x1, max(50, y1 - 4)), COLOR_BANNER, color, font_scale=0.45)

    def render(
        self,
        frame: np.ndarray,
        detection_result: DetectionResult,
        telemetry: Dict[str, Any],
        fps: float,
    ) -> np.ndarray:
        """
        Coordinates full checkpoint rendering pipeline onto a copy of the frame.
        """
        canvas = frame.copy()
        state = telemetry.get("state", "IDLE")
        reason = telemetry.get("reason", "")

        # 1. Draw Checkpoint Inspection Zone
        self.draw_inspection_zone(canvas, state)

        # 2. Draw PPE detections
        self.draw_ppe_detections(canvas, detection_result.helmets, detection_result.vests)

        # 3. Draw Person boxes
        self.draw_persons(canvas, detection_result.persons, state)

        # 4. Draw Top HUD
        latency = detection_result.inference_time_ms
        self.draw_hud(canvas, fps=fps, latency_ms=latency, telemetry=telemetry)

        # 5. Draw Terminal Decision Banner if completed
        self.draw_terminal_decision_banner(canvas, state, reason)

        return canvas
