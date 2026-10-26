"""Application configuration dataclasses for Bottle Fall Detection System."""
from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class DetectionConfig:
    """YOLO detection settings."""
    model_path: str = "yolov8n.pt"     # pre-trained lightweight model (auto-downloaded if missing)
    conf_threshold: float = 0.35
    iou_threshold: float = 0.50
    imgsz: int = 640                   # input size; lower (e.g. 480) = faster on CPU
    device: str = "cpu"                # CPU-first per project constraint
    target_classes: Tuple[int, ...] = (39, 75)  # COCO 39="bottle", 75="vase" (glass bottles)


@dataclass
class FallConfig:
    """
    Baseline fall-decision parameters.

    aspect_ratio = bbox height / width. It is a coarse orientation estimate,
    NOT a physical tilt angle: perspective, bottle shape, occlusion and
    detection noise all affect it. Thresholds are configurable for that reason.
    """
    fall_ratio: float = 0.95        # ratio <= fall_ratio      -> fallen candidate
    stand_ratio: float = 1.55       # ratio >= stand_ratio     -> standing candidate
    recover_ratio: float = 1.30     # from JATUH, ratio >= this starts recovery candidate
    confirm_frames: int = 3         # consecutive frames needed to confirm a transition
    min_status_frames: int = 2      # min frames a confirmed status must hold before next transition
    max_gray_frames: int = 15       # gray-zone/low-conf frames kept in confirmed status before -> TIDAK_PASTI
    event_cooldown_sec: float = 1.0 # duplicate-event guard per (track, event_type)

    # Height fall detection parameters (deteksi jatuh dari ketinggian)
    enable_height_fall: bool = True
    min_downward_speed_px: float = 8.0      # min downward velocity (px/frame)
    min_downward_drop_px: float = 60.0      # min cumulative vertical drop (px)
    virtual_line_y: Optional[float] = None  # optional Y coordinate for virtual boundary line


@dataclass
class DatabaseConfig:
    host: str = "127.0.0.1"
    port: int = 3306
    name: str = "bottle_fall_detection"
    user: str = "bottle_app"
    password: str = ""
    # Test-only override (unit tests run on SQLite; production uses MySQL).
    url_override: str = ""


@dataclass
class AppConfig:
    camera_name: str = "default"
    save_snapshots: bool = True
    snapshot_dir: str = "videos/snapshots"
    output_dir: str = "videos/output"
    input_dir: str = "videos/input"
    save_output_video: bool = False
    log_detections_every_sec: float = 0.0  # 0 = disabled (detections table is optional)
    display: bool = True
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    fall: FallConfig = field(default_factory=FallConfig)
    db: DatabaseConfig = field(default_factory=DatabaseConfig)
