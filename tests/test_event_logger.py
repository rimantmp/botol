"""Tests for EventLogger using an in-memory SQLite database.

Production is MySQL; SQLite exercises the same SQLAlchemy 2.x logic.
"""
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from src.db_models import Base, Camera, FallEvent
from src.event_logger import EventLogger


@pytest.fixture()
def engine():
    eng = create_engine("sqlite://")
    Base.metadata.create_all(eng)
    with Session(eng) as s:
        s.add(Camera(name="test-cam", source_type="video", source_identifier="test.mp4"))
        s.commit()
        cam_id = s.query(Camera).first().id
    yield eng, cam_id


def make_transition(event_type="BOTOL_JATUH", track_id=7):
    return {"track_id": track_id, "event_type": event_type,
            "confidence": 0.87, "orientation_ratio": 0.72,
            "occurred_at": 1700000000.0}


def test_event_stored(engine):
    eng, cam_id = engine
    el = EventLogger(eng, camera_id=cam_id, save_snapshots=False)
    assert el.log_event(make_transition()) is True
    with Session(eng) as s:
        rows = s.scalars(select(FallEvent)).all()
        assert len(rows) == 1
        assert rows[0].event_type == "BOTOL_JATUH"
        assert rows[0].track_id == 7
        assert rows[0].confidence == pytest.approx(0.87)
        assert rows[0].snapshot_path is None


def test_db_failure_returns_false_not_claim_stored(engine):
    eng, cam_id = engine
    closed = create_engine("sqlite://")  # different engine, no schema → fails
    el = EventLogger(closed, camera_id=cam_id, save_snapshots=False)
    assert el.log_event(make_transition()) is False  # must NOT return True


def test_no_duplicate_rows_from_repeated_log(engine):
    eng, cam_id = engine
    el = EventLogger(eng, camera_id=cam_id, save_snapshots=False)
    t = make_transition()
    assert el.log_event(t) is True
    assert el.log_event(t) is True  # second call is a separate legit event
    with Session(eng) as s:
        assert len(s.scalars(select(FallEvent)).all()) == 2  # no explosion


def test_snapshot_saved_on_event(engine, tmp_path):
    import numpy as np
    eng, cam_id = engine
    el = EventLogger(eng, camera_id=cam_id, snapshot_dir=str(tmp_path),
                     save_snapshots=True)
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    ok = el.log_event(make_transition(), frame=frame)
    assert ok is True
    files = list(tmp_path.iterdir())
    assert len(files) == 1
    with Session(eng) as s:
        row = s.scalars(select(FallEvent)).first()
        assert row.snapshot_path == str(files[0])
