"""Fall-condition logic: orientation estimate + temporal confirmation.

The aspect ratio (bbox height / width) is a coarse orientation estimate, not a
physical tilt angle. Perspective, bottle shape, occlusion and detection noise
affect it — hence configurable thresholds and temporal confirmation.

Statuses: BERDIRI (standing), JATUH (fallen), TIDAK_PASTI (uncertain).
Transitions are only confirmed after `confirm_frames` consecutive frames
supporting the new status; each confirmed transition is returned once.
"""
import time
from typing import List

from config import FallConfig

STATUS_STAND = "BERDIRI"
STATUS_FALL = "JATUH"
STATUS_UNCERTAIN = "TIDAK_PASTI"
STATUS_NONE = "TIDAK_TERDETEKSI"


class FallDetector:
    """Turns per-frame bbox geometry + movement into confirmed status transitions."""

    def __init__(self, config: FallConfig):
        self.config = config
        # track_id -> state dict (status machinery lives here, not in tracker)
        self._states: dict = {}

    # ------------------------------------------------------------------ core

    def classify_frame(self, ratio: float, confidence: float, moved_px: float) -> str:
        """Classify ONE frame from geometry + motion. Gray zone -> TIDAK_PASTI."""
        cfg = self.config
        if confidence < 0.20:
            return STATUS_UNCERTAIN
        if ratio <= cfg.fall_ratio:
            return STATUS_FALL
        if ratio >= cfg.stand_ratio:
            return STATUS_STAND
        # Between thresholds: use recovery band if currently fallen, else uncertain.
        if ratio >= cfg.recover_ratio:
            return STATUS_STAND
        return STATUS_UNCERTAIN

    def _movement_px(self, track) -> float:
        """Center displacement between the two most recent history entries."""
        h = track.history
        if len(h) < 2:
            return 0.0
        (_, x0, y0, _), (_, x1, y1, _) = h[-2], h[-1]
        return ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5

    def process_frame(self, tracks, frame_idx: int, timestamp: float) -> List[dict]:
        """Update all track states for one frame; return confirmed transitions.

        Each transition dict: {track_id, from_status, to_status, confidence,
        orientation_ratio, occurred_at}.
        """
        cfg = self.config
        transitions: List[dict] = []

        for tr in tracks:
            ratio = tr.history[-1][3] if tr.history else 0.0
            moved = self._movement_px(tr)
            frame_status = self.classify_frame(ratio, getattr(tr, "confidence", 1.0), moved)

            if tr.track_id not in self._states:
                # First sighting: adopt the observed frame status directly.
                # No event — this is an initial state, not an observed transition.
                self._states[tr.track_id] = {
                    "status": frame_status,
                    "candidate": "",
                    "candidate_frames": 0,
                    "status_frames": 1,
                    "last_event": {},  # event_type -> timestamp of last emit
                }
                continue

            st = self._states[tr.track_id]

            # Check height fall (rapid downward movement or significant vertical drop)
            if self._check_height_fall(tr, st):
                if st["status"] != STATUS_FALL:
                    old = st["status"]
                    st["status"] = STATUS_FALL
                    st["status_frames"] = 1
                    st["candidate"] = ""
                    st["candidate_frames"] = 0
                    event_type = "BOTOL_JATUH"
                    if self._cooldown_ok(tr.track_id, event_type, timestamp, st):
                        st["last_event"][event_type] = timestamp
                        transitions.append({
                            "track_id": tr.track_id,
                            "from_status": old,
                            "to_status": STATUS_FALL,
                            "confidence": getattr(tr, "confidence", 0.0),
                            "orientation_ratio": ratio,
                            "occurred_at": timestamp,
                            "event_type": event_type,
                            "reason": "JATUH_DARI_KETINGGIAN",
                        })
                continue

            # Live status counters
            if frame_status == st["status"]:
                st["status_frames"] += 1
                st["candidate"] = ""
                st["candidate_frames"] = 0
            else:
                if frame_status == st["candidate"]:
                    st["candidate_frames"] += 1
                else:
                    st["candidate"] = frame_status
                    st["candidate_frames"] = 1

                # Confirm transition after enough consecutive supporting frames,
                # and only if current status held long enough (anti-flicker).
                if (st["candidate_frames"] >= cfg.confirm_frames
                        and st["status_frames"] >= cfg.min_status_frames):
                    old = st["status"]
                    st["status"] = st["candidate"]
                    st["status_frames"] = st["candidate_frames"]
                    st["candidate"] = ""
                    st["candidate_frames"] = 0
                    event_type = self._event_for(old, st["status"])
                    if event_type and self._cooldown_ok(tr.track_id, event_type, timestamp, st):
                        st["last_event"][event_type] = timestamp
                        transitions.append({
                            "track_id": tr.track_id,
                            "from_status": old,
                            "to_status": st["status"],
                            "confidence": getattr(tr, "confidence", 0.0),
                            "orientation_ratio": ratio,
                            "occurred_at": timestamp,
                            "event_type": event_type,
                        })
        return transitions

    def _check_height_fall(self, track, st: dict) -> bool:
        """Check if track exhibits rapid downward motion or significant vertical drop."""
        cfg = self.config
        if not getattr(cfg, "enable_height_fall", False):
            return False
        h = track.history
        if len(h) < 2:
            return False

        curr_frame, curr_cx, curr_cy, _ = h[-1]
        # Look back in history (up to last 20 frames)
        for f_idx, _, prev_cy, _ in reversed(h[:-1]):
            frames_diff = curr_frame - f_idx
            if frames_diff <= 0:
                continue
            drop_dy = curr_cy - prev_cy
            if drop_dy <= 0:
                continue
            downward_speed = drop_dy / frames_diff
            if drop_dy >= cfg.min_downward_drop_px and downward_speed >= cfg.min_downward_speed_px:
                return True
            if cfg.virtual_line_y is not None:
                if prev_cy <= cfg.virtual_line_y < curr_cy:
                    return True
        return False

    @staticmethod
    def _event_for(old_status: str, new_status: str) -> str:
        if old_status != STATUS_FALL and new_status == STATUS_FALL:
            return "BOTOL_JATUH"
        if old_status == STATUS_FALL and new_status == STATUS_STAND:
            return "BOTOL_KEMBALI_BERDIRI"
        return ""

    @staticmethod
    def _cooldown_ok(track_id: int, event_type: str, timestamp: float, st: dict) -> bool:
        last = st["last_event"].get(event_type)
        if last is None:
            return True
        return (timestamp - last) >= 1.0  # duplicate guard (seconds)

    def get_status(self, track_id: int) -> str:
        st = self._states.get(track_id)
        return st["status"] if st else STATUS_NONE

    def prune_states(self, active_ids) -> None:
        """Drop state for tracks that no longer exist."""
        keep = set(active_ids)
        self._states = {k: v for k, v in self._states.items() if k in keep}
