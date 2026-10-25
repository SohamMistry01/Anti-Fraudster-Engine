"""Database configuration and session management for Alert and Case Management.

Supports PostgreSQL with psycopg3 via DATABASE_URL, with seamless fallback
to local SQLite for zero-configuration development and automated testing.
"""

import logging
import os
from pathlib import Path
from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session

logger = logging.getLogger(__name__)

# Base declarative class for all ORM models
Base = declarative_base()

# Resolve Database URL:
# 1. Environment variable DATABASE_URL
# 2. Fallback to local SQLite database in data/ directory
DEFAULT_DB_PATH = Path("data/fraud_cases.db").resolve()


def get_database_url() -> str:
    """Resolve database connection URL with PostgreSQL and SQLite fallback."""
    env_url = os.getenv("DATABASE_URL")
    if env_url:
        # Normalize postgres:// to postgresql+psycopg:// if needed
        if env_url.startswith("postgres://"):
            env_url = env_url.replace("postgres://", "postgresql+psycopg://", 1)
        elif env_url.startswith("postgresql://") and "+psycopg" not in env_url:
            env_url = env_url.replace("postgresql://", "postgresql+psycopg://", 1)
        return env_url

    # Fallback to local SQLite database
    DEFAULT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{DEFAULT_DB_PATH}"


DATABASE_URL = get_database_url()
is_sqlite = DATABASE_URL.startswith("sqlite")

# Configure engine with connection pooling appropriate for database dialect
engine_kwargs = {"echo": False}
if is_sqlite:
    engine_kwargs["connect_args"] = {"check_same_thread": False}
else:
    engine_kwargs.update({
        "pool_size": 10,
        "max_overflow": 20,
        "pool_pre_ping": True,
    })

logger.info("Initializing database engine with dialect: %s", DATABASE_URL.split("://")[0])
engine = create_engine(DATABASE_URL, **engine_kwargs)

# Thread-local session factory with expire_on_commit=False to keep detached instances accessible
SessionLocal = sessionmaker(autocommit=False, autoflush=False, expire_on_commit=False, bind=engine)


def init_db() -> None:
    """Create tables if they do not exist."""
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables initialized successfully.")


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency yielding a database session with automatic cleanup."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


from contextlib import contextmanager


@contextmanager
def get_db_context() -> Generator[Session, None, None]:
    """Context manager for obtaining a database session outside FastAPI request cycles."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

