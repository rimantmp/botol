"""YOLO-based bottle detection module."""
import logging
from dataclasses import dataclass
from typing import List, Optional

from config import DetectionConfig

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """One detected object in one frame."""
    bbox: tuple  # (x1, y1, x2, y2)
    confidence: float
    class_id: int
    class_name: str


class BottleDetector:
    """Wraps an Ultralytics YOLO model, filtered to bottle classes."""

    def __init__(self, config: DetectionConfig):
        self.config = config
        self.model = None
        self.class_names: dict = {}

    def load(self) -> None:
        """Load the YOLO model. Downloads the pre-trained weights if missing."""
        from ultralytics import YOLO  # imported here so tests don't require torch

        try:
            self.model = YOLO(self.config.model_path)
            self.class_names = self.model.names
        except Exception as exc:
            logger.error("Failed to load YOLO model %r: %s", self.config.model_path, exc)
            raise

    def detect(self, frame) -> List[Detection]:
        """Detect bottles in one BGR frame. Returns [] on empty results or failure."""
        if self.model is None:
            return []
        try:
            results = self.model.predict(
                frame,
                conf=self.config.conf_threshold,
                iou=self.config.iou_threshold,
                imgsz=self.config.imgsz,
                device=self.config.device,
                verbose=False,
            )
        except Exception as exc:
            logger.error("Inference failed: %s", exc)
            return []

        detections: List[Detection] = []
        if not results or results[0].boxes is None:
            return detections

        for box in results[0].boxes:
            class_id = int(box.cls.item())
            if class_id not in self.config.target_classes:
                continue
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
            raw_name = self.class_names.get(class_id, str(class_id))
            display_name = "bottle" if class_id in (39, 75) else raw_name
            detections.append(Detection(
                bbox=(x1, y1, x2, y2),
                confidence=float(box.conf.item()),
                class_id=class_id,
                class_name=display_name,
            ))
        return detections
