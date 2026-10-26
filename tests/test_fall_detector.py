"""Unit tests for FallDetector logic (no DB, no camera)."""
import time

import pytest

from config import FallConfig
from src.fall_detector import (FallDetector, STATUS_FALL, STATUS_NONE,
                               STATUS_STAND, STATUS_UNCERTAIN)
from src.tracker import Track


def make_track(track_id=1, ratio=2.0, conf=0.9, history=None):
    t = Track(track_id=track_id, bbox=(0, 0, 50, 100), last_seen_frame=0)
    t.history = history or [(0, 25, 50, ratio)]
    t.confidence = conf
    return t


def feed(det, tr, n, timestamp=1000.0, start_frame=0):
    """Feed the same track n frames; return all transitions."""
    out = []
    for i in range(n):
        tr.last_seen_frame = start_frame + i
        out.extend(det.process_frame([tr], start_frame + i, timestamp + i * 0.033))
    return out


def test_stand_stays_stand():
    det = FallDetector(FallConfig())
    tr = make_track(ratio=2.0)
    transitions = feed(det, tr, 10)
    assert det.get_status(1) == STATUS_STAND
    assert transitions == []  # standing is initial-ish; no event for standing


def test_fall_confirmed_after_confirm_frames():
    det = FallDetector(FallConfig())  # confirm_frames=3
    tr = make_track(ratio=2.0)
    feed(det, tr, 5)  # establish standing
    assert det.get_status(1) == STATUS_STAND

    # Bottle topples: ratio drops.
    tr.history = [(99, 25, 50, 0.7)]
    transitions = feed(det, tr, 4, start_frame=100)
    assert det.get_status(1) == STATUS_FALL
    types = [t["event_type"] for t in transitions]
    assert types.count("BOTOL_JATUH") == 1


def test_no_repeat_event_while_lying():
    det = FallDetector(FallConfig())
    tr = make_track(ratio=0.7)
    feed(det, tr, 20)
    types = [t["event_type"] for t in feed(det, tr, 20, start_frame=100)]
    assert types == []  # already JATUH, stays JATUH: no new BOTOL_JATUH events


def test_brief_tilt_does_not_confirm():
    det = FallDetector(FallConfig())  # needs 3 consecutive frames
    tr = make_track(ratio=2.0)
    feed(det, tr, 5)

    tr.history = [(99, 25, 50, 0.7)]
    feed(det, tr, 2, start_frame=100)  # only 2 frames fallen (< confirm_frames)
    tr.history = [(199, 25, 50, 2.0)]
    feed(det, tr, 3, start_frame=200)  # back to standing
    assert det.get_status(1) == STATUS_STAND


def test_get_up_triggers_kembali_berdiri():
    det = FallDetector(FallConfig())
    tr = make_track(ratio=0.7)
    feed(det, tr, 5)  # become JATUH
    tr.history = [(99, 25, 50, 2.0)]
    transitions = feed(det, tr, 5, start_frame=100)
    assert det.get_status(1) == STATUS_STAND
    assert any(t["event_type"] == "BOTOL_KEMBALI_BERDIRI" for t in transitions)


def test_uncertain_in_gray_zone():
    det = FallDetector(FallConfig())
    tr = make_track(ratio=1.2)  # between fall_ratio 0.95 and stand_ratio 1.55
    feed(det, tr, 10)
    assert det.get_status(1) == STATUS_UNCERTAIN


def test_unknown_track_status_is_none():
    det = FallDetector(FallConfig())
    assert det.get_status(999) == STATUS_NONE


def test_prune_states():
    det = FallDetector(FallConfig())
    tr = make_track(track_id=1)
    feed(det, tr, 2)
    det.prune_states([2])  # keep only id 2
    assert det.get_status(1) == STATUS_NONE


def test_height_fall_detected_on_downward_drop():
    cfg = FallConfig(enable_height_fall=True, min_downward_drop_px=50.0, min_downward_speed_px=10.0)
    det = FallDetector(cfg)
    tr = make_track(ratio=2.0)
    feed(det, tr, 3)
    assert det.get_status(1) == STATUS_STAND

    # Bottle drops rapidly downward (cy 50 -> cy 200 in 2 frames: dy=150, speed=75)
    tr.history.append((5, 25, 200, 2.0))
    transitions = det.process_frame([tr], 5, 1000.2)
    assert det.get_status(1) == STATUS_FALL
    assert len(transitions) == 1
    assert transitions[0]["event_type"] == "BOTOL_JATUH"
    assert transitions[0]["reason"] == "JATUH_DARI_KETINGGIAN"


def test_height_fall_virtual_line():
    cfg = FallConfig(enable_height_fall=True, virtual_line_y=150.0)
    det = FallDetector(cfg)
    tr = make_track(ratio=2.0)
    tr.history = [(1, 25, 100, 2.0)]
    feed(det, tr, 2)

    # Crosses virtual line at y=150
    tr.history.append((3, 25, 180, 2.0))
    transitions = det.process_frame([tr], 3, 1000.1)
    assert det.get_status(1) == STATUS_FALL
    assert any(t["event_type"] == "BOTOL_JATUH" for t in transitions)


def test_slow_downward_does_not_trigger_height_fall():
    cfg = FallConfig(enable_height_fall=True, min_downward_drop_px=50.0, min_downward_speed_px=15.0)
    det = FallDetector(cfg)
    tr = make_track(ratio=2.0)
    tr.history = [(1, 25, 50, 2.0)]
    feed(det, tr, 2)

    # Slow drift downward (dy=20, frames=10 -> speed=2.0 < 15.0)
    tr.history.append((12, 25, 70, 2.0))
    transitions = det.process_frame([tr], 12, 1000.5)
    # Status should remain STAND (not FALL)
    assert det.get_status(1) == STATUS_STAND
    assert transitions == []

