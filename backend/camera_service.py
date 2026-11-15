"""
Camera and AI Integration Service for MineGuard AI.

Orchestrates background video capture, YOLO inference, Checkpoint Inspection management,
and real-time visualization rendering.
Exposes thread-safe MJPEG streaming, live checkpoint telemetry, and system health metrics.
"""

import time
import threading
from pathlib import Path
from typing import Optional, Dict, Any, Union, Tuple
import cv2
import numpy as np
import torch

from config import AppConfig
import utils.config as base_config
from detection.detector import PPEObjectDetector, DetectionResult
from inspection.inspection_manager import InspectionManager
from utils.visualization import MineGuardVisualizer
from backend.alert_manager import alert_manager
from backend.database import create_session, close_session, record_inspection


class CameraService:
    """
    Singleton Camera & AI Pipeline Service.
    Executes all computer vision processing asynchronously in a background worker thread.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Camera & Hardware State
        self.camera_source: Union[int, str] = AppConfig.DEFAULT_CAMERA_SOURCE
        self.camera_width: int = AppConfig.DEFAULT_FRAME_WIDTH
        self.camera_height: int = AppConfig.DEFAULT_FRAME_HEIGHT
        self.camera_connected: bool = False
        self.is_monitoring: bool = False
        self.error_message: Optional[str] = None
        self.current_camera_fps: float = 0.0
        self.current_inference_fps: float = 0.0
        self.current_inference_latency_ms: float = 0.0
        self.last_frame_timestamp: float = 0.0
        self.session_id: Optional[int] = None
        self.total_frames_processed: int = 0

        # Model Loading & Verification
        self.trained_model_path: Path = AppConfig.TRAINED_MODEL_PATH
        self.model_loaded: bool = False
        self.model_error: Optional[str] = None

        # AI Pipeline Components
        self.detector: Optional[PPEObjectDetector] = None
        self.inspection_manager: Optional[InspectionManager] = None
        self.visualizer: Optional[MineGuardVisualizer] = None

        # In-memory Buffers
        self._latest_jpeg: Optional[bytes] = None
        self._latest_raw_frame: Optional[np.ndarray] = None

        # Initialize AI components
        self._initialize_ai_pipeline()
        # Pre-render standby placeholder image
        self._generate_standby_frame("CHECKPOINT IDLE - CLICK START TO ACTIVATE INSPECTION")

    def _initialize_ai_pipeline(self):
        """
        Loads the trained YOLO11n model.
        Strictly enforces that models/trained/best.pt must exist.
        Does NOT silently substitute pretrained COCO weights.
        """
        if not self.trained_model_path.exists():
            self.model_loaded = False
            self.model_error = (
                f"Trained PPE model not found at '{self.trained_model_path}'. "
                "Please run 'python training/train.py' to generate trained weights before monitoring."
            )
            print(f"[CameraService ERROR] {self.model_error}")
            return

        try:
            print(f"[CameraService] Loading trained PPE model: {self.trained_model_path} on {AppConfig.DEVICE}")
            self.detector = PPEObjectDetector(
                model_path=self.trained_model_path,
                device=AppConfig.DEVICE,
                conf_person=AppConfig.CONF_PERSON,
                conf_hardhat=AppConfig.CONF_HARDHAT,
                conf_vest=AppConfig.CONF_VEST,
                iou_threshold=AppConfig.IOU_THRESHOLD,
                imgsz=AppConfig.IMAGE_SIZE,
            )
            self.detector.warmup()

            # Callback for database and alert persistence
            def on_inspection_complete(result):
                record_inspection(result)
                alert_manager.dispatch_inspection_alert(result)

            self.inspection_manager = InspectionManager(
                violations_dir=AppConfig.VIOLATIONS_DIR,
                db_callback=on_inspection_complete,
            )
            self.visualizer = MineGuardVisualizer()

            self.model_loaded = True
            self.model_error = None
            print("[CameraService] Checkpoint AI Pipeline successfully initialized.")
        except Exception as e:
            self.model_loaded = False
            self.model_error = f"Failed to initialize AI pipeline: {e}"
            print(f"[CameraService ERROR] {self.model_error}")

    def _generate_standby_frame(self, message: str, bg_color: Tuple[int, int, int] = (15, 23, 42)):
        """Generates a professional standby / offline placeholder frame."""
        h, w = self.camera_height, self.camera_width
        canvas = np.full((h, w, 3), bg_color, dtype=np.uint8)

        # Subtle grid lines
        for y in range(0, h, 40):
            cv2.line(canvas, (0, y), (w, y), (25, 33, 55), 1)
        for x in range(0, w, 40):
            cv2.line(canvas, (x, 0), (x, h), (25, 33, 55), 1)

        # Branding Header
        cv2.putText(
            canvas,
            "MINEGUARD AI",
            (w // 2 - 120, h // 2 - 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 215, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "Automated Checkpoint Quality Inspection",
            (w // 2 - 155, h // 2 - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (160, 175, 200),
            1,
            cv2.LINE_AA,
        )

        # Status Message Box
        (tw, th), _ = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        bx1 = max(10, w // 2 - tw // 2 - 15)
        bx2 = min(w - 10, w // 2 + tw // 2 + 15)
        by1 = h // 2 + 15
        by2 = h // 2 + 45
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (30, 41, 59), -1)
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (71, 85, 105), 1)
        cv2.putText(
            canvas,
            message,
            (w // 2 - tw // 2, h // 2 + 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (226, 232, 240),
            1,
            cv2.LINE_AA,
        )

        ret, jpeg = cv2.imencode(".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if ret:
            with self._lock:
                self._latest_jpeg = jpeg.tobytes()

    def _open_capture_source(self) -> Optional[cv2.VideoCapture]:
        """Opens webcam or network video stream with fallback handling."""
        src = self.camera_source

        # Parse integer index if numeric
        if isinstance(src, str) and src.isdigit():
            src = int(src)

        cap = None
        if isinstance(src, int):
            # DirectShow on Windows for fast device opening
            cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(src)
        else:
            # RTSP / HTTP MJPEG stream (e.g. ESP32-CAM)
            cap = cv2.VideoCapture(str(src))

        if cap is not None and cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return cap

        return None

    def start_monitoring(self, source: Optional[Union[int, str]] = None) -> Tuple[bool, str]:
        """Starts background checkpoint camera capture and AI inspection."""
        if not self.model_loaded:
            self._initialize_ai_pipeline()
            if not self.model_loaded:
                return False, self.model_error or "Model not loaded"

        if self.is_monitoring:
            return True, "Monitoring is already active"

        if source is not None:
            self.camera_source = source

        self._stop_event.clear()
        self.session_id = create_session(camera_source=str(self.camera_source))
        self.total_frames_processed = 0

        # Reset inspection state
        if self.inspection_manager:
            self.inspection_manager.reset()

        self._thread = threading.Thread(target=self._processing_loop, daemon=True)
        self._thread.start()
        self.is_monitoring = True

        return True, "Checkpoint inspection started successfully"

    def stop_monitoring(self) -> Tuple[bool, str]:
        """Stops background monitoring and releases camera resources."""
        if not self.is_monitoring:
            return True, "Monitoring is not running"

        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)

        self.is_monitoring = False
        self.camera_connected = False
        self.current_camera_fps = 0.0
        self.current_inference_fps = 0.0

        if self.session_id is not None:
            close_session(self.session_id)
            self.session_id = None

        if self.inspection_manager:
            self.inspection_manager.reset()

        self._generate_standby_frame("CHECKPOINT IDLE - CLICK START TO ACTIVATE INSPECTION")
        return True, "Checkpoint inspection stopped successfully"

    def restart_camera(self) -> Tuple[bool, str]:
        """Restarts the camera capture service without shutting down the web app."""
        was_running = self.is_monitoring
        if was_running:
            self.stop_monitoring()
        time.sleep(0.5)
        return self.start_monitoring()

    def _processing_loop(self):
        """
        Continuous background thread executing camera ingestion, YOLO detection,
        checkpoint inspection state machine, and overlay rendering.
        """
        print(f"[CameraService] Background inspection loop initiated for source: {self.camera_source}")
        cap = self._open_capture_source()

        consecutive_read_failures = 0
        max_read_failures = 20
        fps_buffer = []

        while not self._stop_event.is_set():
            loop_start = time.perf_counter()

            if cap is None or not cap.isOpened():
                self.camera_connected = False
                self._generate_standby_frame("CONNECTING TO CHECKPOINT CAMERA...", bg_color=(20, 20, 40))
                time.sleep(1.0)
                cap = self._open_capture_source()
                continue

            ret, frame = cap.read()
            if not ret or frame is None or frame.size == 0:
                consecutive_read_failures += 1
                if consecutive_read_failures >= max_read_failures:
                    print("[CameraService] Excessive read failures. Reconnecting camera...")
                    cap.release()
                    cap = None
                    consecutive_read_failures = 0
                    time.sleep(1.0)
                continue

            consecutive_read_failures = 0
            self.camera_connected = True
            self.total_frames_processed += 1
            self.last_frame_timestamp = time.time()

            # 1. Run YOLO Object Detection with Per-Class Calibrated Thresholds
            t_infer_start = time.perf_counter()
            detection_res = self.detector.detect(frame)
            infer_duration_ms = (time.perf_counter() - t_infer_start) * 1000.0
            self.current_inference_latency_ms = round(infer_duration_ms, 2)
            self.current_inference_fps = round(1000.0 / max(1e-4, infer_duration_ms), 1)

            # 2. Advance Checkpoint Inspection State Machine
            annotated_frame_for_evidence = frame.copy()
            telemetry = self.inspection_manager.process_frame(
                raw_frame=frame,
                annotated_frame=annotated_frame_for_evidence,
                detection_result=detection_res,
                session_id=self.session_id,
            )

            # Compute Camera FPS
            elapsed = time.perf_counter() - loop_start
            instant_fps = 1.0 / max(1e-4, elapsed)
            fps_buffer.append(instant_fps)
            if len(fps_buffer) > 15:
                fps_buffer.pop(0)
            self.current_camera_fps = float(np.mean(fps_buffer))

            # 3. Render Visual Overlays
            rendered_frame = self.visualizer.render(
                frame=frame,
                detection_result=detection_res,
                telemetry=telemetry,
                fps=self.current_camera_fps,
            )

            # 4. Encode to JPEG for MJPEG Stream
            ret_enc, jpeg_data = cv2.imencode(".jpg", rendered_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])

            with self._lock:
                if ret_enc:
                    self._latest_jpeg = jpeg_data.tobytes()
                self._latest_raw_frame = frame

        # Cleanup on loop exit
        if cap is not None and cap.isOpened():
            cap.release()
        print("[CameraService] Background inspection loop terminated cleanly.")

    def get_latest_frame_bytes(self) -> Optional[bytes]:
        """Returns the most recent annotated JPEG frame bytes (thread-safe)."""
        with self._lock:
            return self._latest_jpeg

    def get_current_inspection(self) -> Dict[str, Any]:
        """Returns real-time telemetry for the active inspection cycle."""
        if self.inspection_manager:
            return self.inspection_manager.get_live_telemetry()
        return {
            "state": "IDLE",
            "reference": "CHECKPOINT_READY",
            "decision": "PENDING",
            "person_detected": False,
            "helmet_status": "UNCERTAIN",
            "vest_status": "UNCERTAIN",
            "duration": 0.0,
            "reason": "AI engine idle",
            "frames_analyzed": 0,
        }

    def get_status(self) -> Dict[str, Any]:
        """Returns operational status dictionary."""
        return {
            "application_status": "ONLINE",
            "model_status": "LOADED" if self.model_loaded else "NOT_FOUND",
            "model_error": self.model_error,
            "camera_status": "CONNECTED" if self.camera_connected else "DISCONNECTED",
            "camera_source": str(self.camera_source),
            "monitoring_status": "ACTIVE" if self.is_monitoring else "IDLE",
            "camera_fps": round(self.current_camera_fps, 1),
            "inference_fps": self.current_inference_fps,
            "inference_latency_ms": self.current_inference_latency_ms,
            "device": AppConfig.DEVICE,
            "total_frames": self.total_frames_processed,
            "session_id": self.session_id,
        }

    def get_system_health(self) -> Dict[str, Any]:
        """Gathers measured system and hardware health metrics without inventing values."""
        gpu_avail = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if gpu_avail else "N/A"
        gpu_vram_used = "N/A"
        gpu_vram_total = "N/A"

        allocated_mb = 0.0
        reserved_mb = 0.0
        if gpu_avail:
            try:
                allocated_mb = torch.cuda.memory_allocated(0) / (1024 ** 2)
                reserved_mb = torch.cuda.memory_reserved(0) / (1024 ** 2)
                total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
                gpu_vram_used = f"{allocated_mb:.1f} MB (reserved: {reserved_mb:.1f} MB)"
                gpu_vram_total = f"{total / 1024:.2f} GB"
            except Exception:
                pass

        db_size = AppConfig.DATABASE_PATH.stat().st_size if AppConfig.DATABASE_PATH.exists() else 0

        return {
            "camera_connected": self.camera_connected,
            "camera_status": "CONNECTED" if self.camera_connected else "DISCONNECTED",
            "monitoring_active": self.is_monitoring,
            "monitoring_status": "ACTIVE" if self.is_monitoring else "IDLE",
            "camera_fps": round(self.current_camera_fps, 1),
            "inference_fps": self.current_inference_fps,
            "inference_latency_ms": self.current_inference_latency_ms,
            "model_loaded": self.model_loaded,
            "model_status": "LOADED" if self.model_loaded else "ERROR",
            "model_path": str(self.trained_model_path),
            "device": AppConfig.DEVICE,
            "device_name": gpu_name if gpu_avail else "CPU",
            "gpu_available": gpu_avail,
            "cuda_available": gpu_avail,
            "cuda_version": getattr(torch.version, "cuda", None) if gpu_avail else None,
            "gpu_name": gpu_name,
            "gpu_vram_used": gpu_vram_used,
            "gpu_vram_total": gpu_vram_total,
            "gpu_memory_allocated_mb": allocated_mb,
            "gpu_memory_reserved_mb": reserved_mb,
            "database_size_bytes": db_size,
            "total_frames_processed": self.total_frames_processed,
            "last_frame_timestamp": self.last_frame_timestamp,
        }

    def generate_mjpeg(self):
        """Generator yielding MJPEG stream chunks."""
        while True:
            frame_bytes = self.get_latest_frame_bytes()
            if frame_bytes:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                )
            time.sleep(0.033)  # ~30 FPS throttle

    def update_settings(
        self,
        camera_source: Optional[Union[int, str]] = None,
        conf_person: Optional[float] = None,
        conf_hardhat: Optional[float] = None,
        conf_vest: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        safe_confirm_frames: Optional[int] = None,
        fail_confirm_frames: Optional[int] = None,
        inspection_timeout: Optional[float] = None,
    ):
        """Dynamically applies runtime settings without application restart."""
        if self.detector:
            if conf_person is not None:
                self.detector.conf_person = float(conf_person)
            if conf_hardhat is not None:
                self.detector.conf_hardhat = float(conf_hardhat)
            if conf_vest is not None:
                self.detector.conf_vest = float(conf_vest)
            if iou_threshold is not None:
                self.detector.iou_threshold = float(iou_threshold)

        if self.inspection_manager and self.inspection_manager.verifier:
            if safe_confirm_frames is not None:
                self.inspection_manager.verifier.safe_confirm_frames = int(safe_confirm_frames)
            if fail_confirm_frames is not None:
                self.inspection_manager.verifier.fail_confirm_frames = int(fail_confirm_frames)
            if inspection_timeout is not None:
                self.inspection_manager.verifier.timeout_seconds = float(inspection_timeout)

        if camera_source is not None and camera_source != self.camera_source:
            was_monitoring = self.is_monitoring
            if was_monitoring:
                self.stop_monitoring()
            self.camera_source = camera_source
            if was_monitoring:
                self.start_monitoring()


# Global Singleton Camera Service
camera_service = CameraService()
