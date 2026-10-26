"""Event logging: persists confirmed transitions to the database.

Kept independent from detection logic and from database internals so it can
be tested with a fake session/engine.
"""
import logging
import time
from typing import Optional

from .db_models import FallEvent

logger = logging.getLogger(__name__)


class EventLogger:
    """Writes FallEvent rows for confirmed transitions; snapshot-first design.

    Failure semantics: if the DB write fails, the snapshot file may still
    exist on disk but we do NOT claim the event is stored.
    """

    def __init__(self, engine, camera_id: int, snapshot_dir: str = "",
                 save_snapshots: bool = True, max_retries: int = 2,
                 retry_delay: float = 0.5):
        self.engine = engine
        self.camera_id = camera_id
        self.snapshot_dir = snapshot_dir
        self.save_snapshots = save_snapshots
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def log_event(self, transition: dict, frame=None) -> bool:
        """Persist one confirmed transition. Returns True only on commit."""
        snapshot_path = None
        if self.save_snapshots and frame is not None:
            snapshot_path = self._save_snapshot(transition, frame)

        from datetime import datetime, timezone

        occurred_at = datetime.fromtimestamp(transition["occurred_at"], tz=timezone.utc)
        data = {
            "camera_id": self.camera_id,
            "track_id": transition.get("track_id"),
            "event_type": transition["event_type"],
            "confidence": float(transition.get("confidence", 0.0)),
            "orientation_ratio": transition.get("orientation_ratio"),
            "occurred_at": occurred_at,
            "snapshot_path": snapshot_path,
            "notes": None,
        }

        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                from sqlalchemy.orm import Session

                with Session(self.engine, expire_on_commit=False) as session:
                    event = FallEvent(**data)
                    session.add(event)
                    session.commit()
                    logger.info("Event stored: id=%s type=%s track=%s",
                                event.id, event.event_type, event.track_id)
                    return True
            except Exception as exc:  # DB down, constraint error, etc.
                last_error = exc
                logger.warning("DB write attempt %d/%d failed: %s",
                               attempt, self.max_retries, exc.__class__.__name__)
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay)
        logger.error("Event NOT stored after %d attempts: %s",
                     self.max_retries, last_error.__class__.__name__)
        return False

    def _save_snapshot(self, transition: dict, frame) -> Optional[str]:
        """Write the frame as evidence image; return relative path or None."""
        try:
            import os
            import cv2

            os.makedirs(self.snapshot_dir, exist_ok=True)
            ts = transition["occurred_at"]
            track_id = transition.get("track_id", "x")
            filename = f"{ts:.0f}_track{track_id}_{transition['event_type']}.jpg"
            path = os.path.join(self.snapshot_dir, filename)
            if not cv2.imwrite(path, frame):
                logger.error("cv2.imwrite failed for %s", path)
                return None
            return path
        except Exception as exc:
            logger.error("Snapshot write failed: %s", exc.__class__.__name__)
            return None
