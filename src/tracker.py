"""Object tracking: assigns and maintains track IDs across frames.

Uses Ultralytics' ByteTrack when available; a simple IoU-matching fallback
keeps the module testable without heavy dependencies. Tracking is independent
from fall-decision logic.
"""
import logging
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Track:
    """Internal tracking state for one object."""
    track_id: int
    bbox: tuple
    last_seen_frame: int
    status: str = "TIDAK_PASTI"       # current status, persists across frames
    status_frames: int = 0            # frames in current status
    candidate_status: str = ""        # status being considered for confirmation
    candidate_frames: int = 0         # consecutive frames supporting candidate
    frames_missing: int = 0           # consecutive frames not seen
    last_update_time: float = 0.0
    history: list = None              # recent (frame_idx, cx, cy, ratio) entries

    def __post_init__(self):
        if self.history is None:
            self.history = []


class SimpleIoUTracker:
    """IoU-based tracker fallback: greedy matching, lost tracks kept for a
    buffer of frames so brief occlusions don't reset status."""

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 30):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self._next_id = 1
        self._tracks: Dict[int, Track] = {}

    @staticmethod
    def _iou(a, b) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        return inter / (area_a + area_b - inter + 1e-9)

    def update(self, detections: List[tuple], frame_idx: int, timestamp: float) -> List[Track]:
        """Match detections (bbox tuples) to existing tracks; return active tracks."""
        # Mark all as unseen this frame, then decay.
        for tid, tr in self._tracks.items():
            if tr.last_seen_frame < frame_idx - 1:
                tr.frames_missing = frame_idx - tr.last_seen_frame

        unmatched_det = list(range(len(detections)))
        # Greedy matching by IoU.
        pairs = []
        for tid, tr in self._tracks.items():
            for di in unmatched_det:
                iou = self._iou(tr.bbox, detections[di])
                if iou >= self.iou_threshold:
                    pairs.append((iou, tid, di))
        pairs.sort(reverse=True)

        used_tids, used_dets = set(), set()
        for iou, tid, di in pairs:
            if tid in used_tids or di in used_dets:
                continue
            tr = self._tracks[tid]
            cx = (detections[di][0] + detections[di][2]) / 2
            cy = (detections[di][1] + detections[di][3]) / 2
            w = max(1.0, detections[di][2] - detections[di][0])
            h = max(1.0, detections[di][3] - detections[di][1])
            tr.history.append((frame_idx, cx, cy, h / w))
            if len(tr.history) > 30:
                tr.history.pop(0)
            tr.bbox = detections[di]
            tr.last_seen_frame = frame_idx
            tr.frames_missing = 0
            tr.last_update_time = timestamp
            used_tids.add(tid)
            used_dets.add(di)

        # Second-pass matching for fast-moving / falling objects (low or zero IoU due to high displacement)
        unmatched_tids = [tid for tid in self._tracks if tid not in used_tids]
        unmatched_dets = [di for di in range(len(detections)) if di not in used_dets]
        if unmatched_tids and unmatched_dets:
            downward_pairs = []
            for tid in unmatched_tids:
                tr = self._tracks[tid]
                last_cx = tr.history[-1][1] if tr.history else (tr.bbox[0] + tr.bbox[2]) / 2
                last_cy = tr.history[-1][2] if tr.history else (tr.bbox[1] + tr.bbox[3]) / 2
                frames_gap = frame_idx - tr.last_seen_frame
                if frames_gap <= 0:
                    continue

                for di in unmatched_dets:
                    bx1, by1, bx2, by2 = detections[di]
                    cx = (bx1 + bx2) / 2
                    cy = (by1 + by2) / 2
                    dx = abs(cx - last_cx)
                    dy = cy - last_cy

                    # Falling downward: cy increases (dy >= -10 px tolerance), reasonable horizontal drift
                    w_det = max(1.0, bx2 - bx1)
                    max_dx = max(120.0, w_det * 2.5)
                    if dy >= -10 and dx <= max_dx:
                        dist = (dx**2 + dy**2) ** 0.5
                        downward_pairs.append((dist, tid, di))

            downward_pairs.sort()
            for dist, tid, di in downward_pairs:
                if tid in used_tids or di in used_dets:
                    continue
                tr = self._tracks[tid]
                cx = (detections[di][0] + detections[di][2]) / 2
                cy = (detections[di][1] + detections[di][3]) / 2
                w = max(1.0, detections[di][2] - detections[di][0])
                h = max(1.0, detections[di][3] - detections[di][1])
                tr.history.append((frame_idx, cx, cy, h / w))
                if len(tr.history) > 30:
                    tr.history.pop(0)
                tr.bbox = detections[di]
                tr.last_seen_frame = frame_idx
                tr.frames_missing = 0
                tr.last_update_time = timestamp
                used_tids.add(tid)
                used_dets.add(di)

        # New tracks for unmatched detections.
        for di in range(len(detections)):
            if di in used_dets:
                continue
            tr = Track(track_id=self._next_id, bbox=detections[di],
                       last_seen_frame=frame_idx, last_update_time=timestamp)
            cx = (detections[di][0] + detections[di][2]) / 2
            cy = (detections[di][1] + detections[di][3]) / 2
            w = max(1.0, detections[di][2] - detections[di][0])
            h = max(1.0, detections[di][3] - detections[di][1])
            tr.history.append((frame_idx, cx, cy, h / w))
            self._tracks[self._next_id] = tr
            self._next_id += 1

        # Drop stale tracks beyond max_age.
        stale = [tid for tid, tr in self._tracks.items()
                 if frame_idx - tr.last_seen_frame > self.max_age]
        for tid in stale:
            del self._tracks[tid]

        return self.active_tracks(frame_idx)

    def active_tracks(self, frame_idx: int) -> List[Track]:
        return [t for t in self._tracks.values() if t.last_seen_frame == frame_idx]

    def all_tracks(self) -> List[Track]:
        return list(self._tracks.values())


class Tracker:
    """Public tracker facade: tries ByteTrack (via ultralytics), falls back
    to SimpleIoUTracker. Exposes the same update()/active_tracks() API."""

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 30):
        self._impl = SimpleIoUTracker(iou_threshold=iou_threshold, max_age=max_age)
        self._byte = None
        try:
            from ultralytics.trackers.byte_tracker import BYTETracker  # noqa: F401
            self._available = True
        except Exception:
            self._available = False
            logger.info("ByteTrack unavailable; using IoU fallback tracker.")

    def update(self, detections: List[tuple], frame_idx: int, timestamp: float) -> List[Track]:
        # NOTE: fallback tracker used for v1; ByteTrack integration via
        # ultralytics model.track() is planned at the main-loop stage.
        return self._impl.update(detections, frame_idx, timestamp)

    def active_tracks(self, frame_idx: int) -> List[Track]:
        return self._impl.active_tracks(frame_idx)

    def all_tracks(self) -> List[Track]:
        return self._impl.all_tracks()
