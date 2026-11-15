"""
Object Detection Engine for MineGuard AI.

Provides a robust, reusable PPEObjectDetector class wrapping Ultralytics YOLO.
Dynamically resolves class mappings (Person, Hardhat, Safety Vest), applies per-class
calibrated confidence thresholds, uses torch.inference_mode() for speed,
and outputs structured detections.
"""

import sys
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Union, Optional, List, Dict
import numpy as np
import torch

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import utils.config as config
from ultralytics import YOLO


@dataclass
class Detection:
    """Represents a single detected bounding box with metadata."""
    box: List[float]       # [x1, y1, x2, y2] absolute coordinates
    confidence: float      # Confidence score [0.0 - 1.0]
    class_id: int          # Model class ID
    class_name: str        # Resolved class name ('Person', 'Hardhat', 'Safety Vest')

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
    def center_x(self) -> float:
        return (self.box[0] + self.box[2]) / 2.0

    @property
    def center_y(self) -> float:
        return (self.box[1] + self.box[3]) / 2.0

    @property
    def area(self) -> float:
        return self.width * self.height


@dataclass
class DetectionResult:
    """Structured container for all detections on a single frame."""
    persons: List[Detection]
    helmets: List[Detection]
    vests: List[Detection]
    all_detections: List[Detection]
    inference_time_ms: float
    frame_shape: tuple  # (height, width, channels)


class PPEObjectDetector:
    """
    Reusable YOLO-based PPE detector for checkpoint compliance inspection.
    Dynamically identifies Person, Hardhat, and Safety Vest class IDs from the model
    and applies per-class calibrated confidence filters.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        conf_person: Optional[float] = None,
        conf_hardhat: Optional[float] = None,
        conf_vest: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        imgsz: Optional[int] = None,
    ):
        # Resolve model path
        if model_path is None:
            self.model_path = Path(config.MODEL_PATH)
        else:
            self.model_path = Path(model_path)

        if not self.model_path.exists():
            if self.model_path == config.TRAINED_MODEL_PATH and config.PRETRAINED_MODEL_PATH.exists():
                print(f"[PPEObjectDetector] Note: {self.model_path} not found. Using pretrained {config.PRETRAINED_MODEL_PATH}")
                self.model_path = config.PRETRAINED_MODEL_PATH
            else:
                raise FileNotFoundError(
                    f"Model weights file not found: {self.model_path}. "
                    "Ensure weights exist or train model first."
                )

        # Resolve device
        self.device = device if device is not None else config.DEVICE
        self.conf_person = conf_person if conf_person is not None else config.CONF_PERSON
        self.conf_hardhat = conf_hardhat if conf_hardhat is not None else config.CONF_HARDHAT
        self.conf_vest = conf_vest if conf_vest is not None else config.CONF_VEST
        self.iou_threshold = iou_threshold if iou_threshold is not None else config.IOU_THRESHOLD
        self.imgsz = imgsz if imgsz is not None else config.IMAGE_SIZE

        # Load YOLO model once
        try:
            print(f"[PPEObjectDetector] Loading model from: {self.model_path} on device: {self.device}")
            self.model = YOLO(str(self.model_path))
            self.model.to(self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model from {self.model_path}: {e}")

        # Extract and verify model class mapping dynamically
        self.names: Dict[int, str] = self.model.names if hasattr(self.model, "names") else {}
        self._resolve_class_mappings()

    def _resolve_class_mappings(self):
        """
        Dynamically locates class IDs for Person, Hardhat, and Safety Vest
        without hardcoding assumption of class order.
        """
        self.person_class_id: Optional[int] = None
        self.hardhat_class_id: Optional[int] = None
        self.vest_class_id: Optional[int] = None

        for cid, cname in self.names.items():
            name_lower = str(cname).lower().strip()
            if "person" in name_lower or "worker" in name_lower:
                self.person_class_id = int(cid)
            elif "hardhat" in name_lower or "helmet" in name_lower or "hard hat" in name_lower:
                self.hardhat_class_id = int(cid)
            elif "vest" in name_lower or "safety vest" in name_lower:
                self.vest_class_id = int(cid)

        print("[PPEObjectDetector] Dynamic Class Mapping Resolved:")
        print(f"  Person Class ID      : {self.person_class_id} ('{self.names.get(self.person_class_id, 'Not Found')}')")
        print(f"  Hardhat Class ID     : {self.hardhat_class_id} ('{self.names.get(self.hardhat_class_id, 'Not Found')}')")
        print(f"  Safety Vest Class ID : {self.vest_class_id} ('{self.names.get(self.vest_class_id, 'Not Found')}')")

    def warmup(self):
        """Runs a zero-tensor through the model to initialize CUDA context."""
        dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        self.detect(dummy)

    def detect(
        self,
        image: np.ndarray,
        conf_person: Optional[float] = None,
        conf_hardhat: Optional[float] = None,
        conf_vest: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        imgsz: Optional[int] = None,
    ) -> DetectionResult:
        """
        Executes inference on an image (BGR numpy array from cv2) using torch.inference_mode().
        Applies separate, calibrated confidence thresholds for Person, Hardhat, and Safety Vest.
        """
        if image is None or image.size == 0:
            raise ValueError("Input image is empty or None")

        cp = conf_person if conf_person is not None else self.conf_person
        ch = conf_hardhat if conf_hardhat is not None else self.conf_hardhat
        cv = conf_vest if conf_vest is not None else self.conf_vest
        iou = iou_threshold if iou_threshold is not None else self.iou_threshold
        img_size = imgsz if imgsz is not None else self.imgsz

        # Minimum confidence to pass into YOLO model to prevent premature filtering
        min_conf = max(0.10, min(cp, ch, cv) - 0.05)

        start_time = time.perf_counter()

        with torch.inference_mode():
            results = self.model.predict(
                source=image,
                conf=min_conf,
                iou=iou,
                imgsz=img_size,
                device=self.device,
                verbose=False,
            )

        inference_time_ms = (time.perf_counter() - start_time) * 1000.0

        persons: List[Detection] = []
        helmets: List[Detection] = []
        vests: List[Detection] = []
        all_detections: List[Detection] = []

        if len(results) > 0 and results[0].boxes is not None:
            boxes_data = results[0].boxes

            xyxy = boxes_data.xyxy.cpu().numpy()
            confs = boxes_data.conf.cpu().numpy()
            clss = boxes_data.cls.cpu().numpy().astype(int)

            for i in range(len(xyxy)):
                cid = int(clss[i])
                score = float(confs[i])
                box = [float(v) for v in xyxy[i]]
                cname = self.names.get(cid, f"Class_{cid}")

                # Apply per-class calibrated threshold
                if cid == self.person_class_id:
                    if score >= cp:
                        det = Detection(box=box, confidence=score, class_id=cid, class_name=cname)
                        persons.append(det)
                        all_detections.append(det)
                elif cid == self.hardhat_class_id:
                    if score >= ch:
                        det = Detection(box=box, confidence=score, class_id=cid, class_name=cname)
                        helmets.append(det)
                        all_detections.append(det)
                elif cid == self.vest_class_id:
                    if score >= cv:
                        det = Detection(box=box, confidence=score, class_id=cid, class_name=cname)
                        vests.append(det)
                        all_detections.append(det)
                else:
                    # Unrecognized class
                    if score >= 0.35:
                        all_detections.append(
                            Detection(box=box, confidence=score, class_id=cid, class_name=cname)
                        )

        return DetectionResult(
            persons=persons,
            helmets=helmets,
            vests=vests,
            all_detections=all_detections,
            inference_time_ms=inference_time_ms,
            frame_shape=image.shape,
        )
