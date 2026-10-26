"""Database engine/session management (SQLAlchemy 2.x, PyMySQL driver).

Production target is MySQL; tests use SQLite via url_override so DB logic is
exercisable without a running MySQL server.
"""
import logging
from contextlib import contextmanager
from urllib.parse import quote_plus

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from .db_models import Base  # noqa: F401

logger = logging.getLogger(__name__)


def build_url(db) -> str:
    """Build a SQLAlchemy URL from a DatabaseConfig (or similar attributes)."""
    if getattr(db, "url_override", ""):
        return db.url_override
    return (
        f"mysql+pymysql://{quote_plus(db.user)}:{quote_plus(db.password)}"
        f"@{db.host}:{db.port}/{db.name}?charset=utf8mb4"
    )


def create_db_engine(db) -> Engine:
    """Create a reusable engine with pooling and pre-ping (call once per app)."""
    url = build_url(db)
    is_sqlite = url.startswith("sqlite")
    kwargs = {"future": True}
    if not is_sqlite:
        # Connection pooling options are MySQL-specific.
        kwargs.update(
            pool_pre_ping=True,   # drop dead pooled connections instead of erroring
            pool_recycle=1800,    # recycle before server-side idle timeout
            pool_size=5,
            max_overflow=5,
        )
    return create_engine(url, **kwargs)


def init_schema(engine: Engine) -> None:
    """Create tables if missing. Never drops existing data."""
    Base.metadata.create_all(engine)


@contextmanager
def get_session(engine: Engine):
    """Session per unit of work, with commit/rollback/close handling."""
    session = Session(engine, expire_on_commit=False)
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def check_connection(engine: Engine) -> bool:
    """Return True if a trivial query succeeds, else log a clear failure."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except SQLAlchemyError as exc:
        logger.error("Database connection failed: %s", exc.__class__.__name__)
        return False
