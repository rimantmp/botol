"""Configuration loader: builds AppConfig from .env + optional overrides."""
import os
from dataclasses import replace

from dotenv import load_dotenv

from config import AppConfig, DatabaseConfig, DetectionConfig, FallConfig


def load_config(env_file: str = ".env", **overrides) -> AppConfig:
    """Load .env, apply overrides, return AppConfig."""
    load_dotenv(env_file)

    detection = DetectionConfig(
        model_path=os.getenv("YOLO_MODEL_PATH", DetectionConfig.model_path),
        conf_threshold=float(os.getenv("CONF_THRESHOLD", DetectionConfig.conf_threshold)),
        imgsz=int(os.getenv("IMGSZ", DetectionConfig.imgsz)),
        device=os.getenv("DEVICE", DetectionConfig.device),
    )
    fall = FallConfig(
        fall_ratio=float(os.getenv("FALL_RATIO", FallConfig.fall_ratio)),
        stand_ratio=float(os.getenv("STAND_RATIO", FallConfig.stand_ratio)),
        confirm_frames=int(os.getenv("CONFIRM_FRAMES", FallConfig.confirm_frames)),
        enable_height_fall=os.getenv("ENABLE_HEIGHT_FALL", "1").lower() in ("1", "true", "yes"),
        min_downward_speed_px=float(os.getenv("MIN_DOWNWARD_SPEED_PX", FallConfig.min_downward_speed_px)),
        min_downward_drop_px=float(os.getenv("MIN_DOWNWARD_DROP_PX", FallConfig.min_downward_drop_px)),
        virtual_line_y=float(os.getenv("VIRTUAL_LINE_Y")) if os.getenv("VIRTUAL_LINE_Y") else None,
    )
    db = DatabaseConfig(
        host=os.getenv("DB_HOST", "127.0.0.1"),
        port=int(os.getenv("DB_PORT", "3306")),
        name=os.getenv("DB_NAME", "bottle_fall_detection"),
        user=os.getenv("DB_USER", "bottle_app"),
        password=os.getenv("DB_PASSWORD", ""),
        url_override=os.getenv("DB_URL", ""),
    )
    cfg = AppConfig(camera_name=os.getenv("CAMERA_NAME", "default"),
                    detection=detection, fall=fall, db=db)

    # Explicit keyword overrides win over .env (used by tests and CLI flags).
    for key, value in overrides.items():
        if value is None:
            continue
        if key.startswith("detection_"):
            cfg.detection = replace(cfg.detection, **{key[len("detection_"):]: value})
        elif key.startswith("fall_"):
            cfg.fall = replace(cfg.fall, **{key[len("fall_"):]: value})
        elif key.startswith("db_"):
            cfg.db = replace(cfg.db, **{key[len("db_"):]: value})
        else:
            cfg = replace(cfg, **{key: value})
    return cfg
