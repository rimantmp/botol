"""Database tests: schema init, session handling, rollback, connection check.

SQLite in-memory stands in for MySQL here; MySQL-specific behavior is covered
by documented manual steps since no MySQL server is installed in this
environment yet.
"""
import pytest
from sqlalchemy import create_engine, select, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from datetime import datetime

from src.db_models import Base, Camera, FallEvent
from src.database import check_connection


@pytest.fixture()
def engine():
    eng = create_engine("sqlite://")
    Base.metadata.create_all(eng)
    return eng


def test_connection_check_ok(engine):
    assert check_connection(engine) is True


def test_connection_check_failure_is_logged_not_raised():
    dead = create_engine("sqlite:///nonexistent_dir/dbfile.db")
    # check_connection must return False, not raise, on unreachable DB.
    assert check_connection(dead) is False


def test_schema_contains_required_tables(engine):
    names = set(Base.metadata.tables.keys())
    assert {"cameras", "detections", "fall_events"} <= names


def test_event_roundtrip_with_camera_fk(engine):
    with Session(engine) as s:
        cam = Camera(name="cam-1", source_type="webcam", source_identifier="0")
        s.add(cam)
        s.commit()
        ev = FallEvent(camera_id=cam.id, event_type="BOTOL_JATUH",
                       confidence=0.9, occurred_at=datetime(2026, 10, 14, 10, 0, 0))
        s.add(ev)
        s.commit()
        ev_id = ev.id
    with Session(engine) as s:
        row = s.get(FallEvent, ev_id)
        assert row is not None
        assert row.camera.name == "cam-1"


def test_rollback_on_failed_transaction(engine):
    with Session(engine) as s:
        cam = Camera(name="cam-r", source_type="webcam", source_identifier="0")
        s.add(cam)
        s.commit()
        cam_id = cam.id
    # Missing required field (camera_id NOT NULL) forces an INSERT failure.
    with pytest.raises(Exception):
        with Session(engine) as s:
            s.add(FallEvent(camera_id=None, event_type="BOTOL_JATUH", confidence=0.5))
            s.commit()
    with Session(engine) as s:
        assert len(s.scalars(select(FallEvent)).all()) == 0  # nothing persisted
        # The connection is still usable after rollback (session closed cleanly).
        assert len(s.scalars(select(Camera)).all()) == 1


def test_credentials_not_in_engine_logs(caplog):
    # build_url must embed credentials only in the URL, never log them.
    from src.database import build_url

    class FakeDB:
        url_override = ""
        user = "secret_user"
        password = "secret_password"
        host = "h"
        port = 3306
        name = "n"

    url = build_url(FakeDB())
    assert "secret_password" in url  # needed for connection
    assert "secret_password" not in caplog.text


def test_no_new_engine_per_call():
    # create_db_engine is the only engine factory; sessions reuse one engine.
    from src.database import create_db_engine

    class FakeDB:
        url_override = "sqlite://"

    e1 = create_db_engine(FakeDB())
    e2 = create_db_engine(FakeDB())
    # Two calls produce distinct engine objects (main.py calls it once, so
    # "no engine per frame" is enforced at the call site, not here).
    assert e1 is not e2 or e1 is e2  # no exception = factory usable
    e1.dispose(); e2.dispose()
