"""
Real-Time Webcam Detection and Compliance Monitoring Pipeline for MineGuard AI.

Integrates:
- OpenCV VideoCapture ingestion with auto-reconnect handling
- YOLO11n object detection and ByteTrack worker tracking
- Anatomical PPE association (Hardhat and Safety Vest)
- Stateful compliance engine with temporal filtering and alert cooldown
- Full-screen / windowed visualization with HUD, FPS, and status badges
- Manual snapshot capture ('s') and graceful exit (ESC / 'q')
"""

import sys
import time
import argparse
from pathlib import Path
from typing import Optional
import cv2
import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import utils.config as config
from detection.detector import PPEObjectDetector
from detection.tracker import WorkerTracker
from compliance.association import PPEAssociator
from compliance.compliance import ComplianceEngine, ViolationAlert
from utils.visualization import MineGuardVisualizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="MineGuard AI - Real-Time Webcam PPE Compliance Monitoring"
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=config.CAMERA_INDEX,
        help=f"Webcam device index (default: {config.CAMERA_INDEX})",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(config.MODEL_PATH),
        help=f"Path to model weights (default: {config.MODEL_PATH})",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=config.CONFIDENCE_THRESHOLD,
        help=f"Detection confidence threshold (default: {config.CONFIDENCE_THRESHOLD})",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=config.IOU_THRESHOLD,
        help=f"NMS IoU threshold (default: {config.IOU_THRESHOLD})",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=config.IMAGE_SIZE,
        help=f"Inference input resolution (default: {config.IMAGE_SIZE})",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=config.DEVICE,
        help=f"Computation device (default: {config.DEVICE})",
    )
    parser.add_argument(
        "--skip-frames",
        type=int,
        default=0,
        help="Inference frame skipping rate for ultra-low latency (default: 0)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without graphical display window (for benchmarking or server mode)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum frames to run before terminating (useful for tests)",
    )
    parser.add_argument(
        "--record",
        type=str,
        default=None,
        help="Optional path to output video file (e.g. outputs/predictions/live.mp4)",
    )
    return parser.parse_args()


def open_camera(camera_index: int) -> cv2.VideoCapture:
    """Initializes OpenCV VideoCapture with DirectShow fallback on Windows."""
    # Attempt DirectShow backend first on Windows for faster startup and device enumeration
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        # Fallback to default backend
        cap = cv2.VideoCapture(camera_index)

    if cap.isOpened():
        # Optimize camera stream buffer and target standard 640x480 resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    return cap


def run_webcam_pipeline(args):
    """Executes the full real-time PPE monitoring loop."""
    print("=" * 70)
    print("           MINEGUARD AI - REAL-TIME PPE COMPLIANCE MONITOR")
    print("=" * 70)
    print(f"Target Camera Index     : {args.camera}")
    print(f"Model Path              : {args.model}")
    print(f"Hardware Device Target  : {args.device}")
    print(f"Inference Resolution    : {args.imgsz}")
    print(f"Confidence Threshold    : {args.conf}")
    print(f"IoU Threshold           : {args.iou}")
    print(f"Violations Directory    : {config.VIOLATIONS_DIR}")
    print(f"Predictions Directory   : {config.PREDICTIONS_DIR}")
    print("=" * 70)
    print("CONTROLS:")
    print("  [ESC] or [q] : Exit application cleanly")
    print("  [s]         : Save manual annotated snapshot to outputs/predictions/")
    print("  [r]         : Reset worker tracking & compliance state")
    print("=" * 70)

    # Initialize Object Detector
    detector = PPEObjectDetector(
        model_path=args.model,
        device=args.device,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        imgsz=args.imgsz,
    )
    detector.warmup()

    # Initialize Modules
    tracker = WorkerTracker(detector=detector)
    associator = PPEAssociator()
    compliance_engine = ComplianceEngine()
    visualizer = MineGuardVisualizer()

    # Open Camera Stream
    print(f"\nConnecting to webcam (Index: {args.camera})...")
    cap = open_camera(args.camera)

    if not cap.isOpened():
        print(f"\n[ERROR] Unable to open camera with index {args.camera}.")
        print("Please check that your webcam is connected and not in use by another application.")
        return

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera successfully connected. Stream Resolution: {actual_w}x{actual_h}")

    # Optional Video Recording
    video_writer = None
    if args.record:
        rec_path = Path(args.record)
        rec_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(str(rec_path), fourcc, 20.0, (actual_w, actual_h))
        print(f"Recording output stream to: {rec_path}")

    # Performance Monitoring
    fps_history = []
    fps_smooth = 30.0
    frame_idx = 0
    recent_alert: Optional[ViolationAlert] = None
    alert_banner_expire_time = 0.0

    window_name = "MineGuard AI - PPE Compliance Monitor"
    if not args.headless:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    consecutive_failed_reads = 0
    max_failed_reads = 15

    try:
        while True:
            t_start = time.perf_counter()

            # Read next frame
            ret, frame = cap.read()
            if not ret or frame is None:
                consecutive_failed_reads += 1
                print(f"[WARN] Failed to read frame from camera ({consecutive_failed_reads}/{max_failed_reads})...")

                if consecutive_failed_reads >= max_failed_reads:
                    print("\n[ERROR] Camera disconnected or unresponsive. Exiting pipeline.")
                    break

                time.sleep(0.1)
                continue

            consecutive_failed_reads = 0
            frame_idx += 1

            # 1. Update Worker Tracker & Extract PPE Detections
            tracking_res = tracker.update(frame)

            # 2. Geometric PPE Association (One-to-One Matching)
            associations = associator.associate(
                workers=tracking_res.workers,
                helmets=tracking_res.helmets,
                vests=tracking_res.vests,
                frame_shape=frame.shape,
            )

            # 3. Compliance Engine Temporal Evaluation & Alert Generation
            states, new_alerts = compliance_engine.evaluate(
                associations=associations,
                frame=frame,
                timestamp=time.time(),
            )

            # Check if new alerts occurred
            if new_alerts:
                recent_alert = new_alerts[-1]
                alert_banner_expire_time = time.time() + 4.0  # Show banner for 4 seconds
                print(f"\n[ALERT TRIGGERED] Worker #{recent_alert.track_id}: {recent_alert.violation_type} at {recent_alert.time_str}")

            if time.time() > alert_banner_expire_time:
                recent_alert = None

            # Calculate FPS
            t_elapsed = time.perf_counter() - t_start
            instant_fps = 1.0 / max(1e-4, t_elapsed)
            fps_history.append(instant_fps)
            if len(fps_history) > 15:
                fps_history.pop(0)
            fps_smooth = float(np.mean(fps_history))

            # 4. Render Visual Overlays
            rendered_frame = visualizer.render(
                frame=frame,
                associations=associations,
                states=states,
                helmets=tracking_res.helmets,
                vests=tracking_res.vests,
                fps=fps_smooth,
                recent_alert=recent_alert,
            )

            # Write to video recorder if active
            if video_writer is not None:
                video_writer.write(rendered_frame)

            # 5. Display Window
            if not args.headless:
                cv2.imshow(window_name, rendered_frame)
                key = cv2.waitKey(1) & 0xFF

                if key in [27, ord("q"), ord("Q")]:  # ESC or Q
                    print("\nUser requested exit. Closing pipeline...")
                    break
                elif key in [ord("s"), ord("S")]:  # Snapshot
                    snap_time = int(time.time())
                    snap_filename = f"snapshot_{snap_time}_{frame_idx}.jpg"
                    snap_path = config.PREDICTIONS_DIR / snap_filename
                    cv2.imwrite(str(snap_path), rendered_frame)
                    print(f"\n[SNAPSHOT SAVED] {snap_path}")
                elif key in [ord("r"), ord("R")]:  # Reset
                    tracker.reset()
                    compliance_engine.reset()
                    print("\n[STATE RESET] Worker tracker and compliance history cleared.")

            # Frame limit check for automated testing
            if args.max_frames is not None and frame_idx >= args.max_frames:
                print(f"\nReached configured max_frames limit ({args.max_frames}). Terminating test.")
                break

    except KeyboardInterrupt:
        print("\nPipeline interrupted by user via KeyboardInterrupt.")
    finally:
        # Resource cleanup
        if cap is not None and cap.isOpened():
            cap.release()
        if video_writer is not None:
            video_writer.release()
        if not args.headless:
            cv2.destroyAllWindows()
        print("Camera resources released and windows destroyed successfully.")
        print(f"Total frames processed: {frame_idx}")


if __name__ == "__main__":
    cli_args = parse_args()
    run_webcam_pipeline(cli_args)
