from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from assistant import db


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """A throwaway database with schema and reference data loaded."""
    path = tmp_path / "test.db"
    db.bootstrap(path)
    return path


def slot(weekday: int, hour: int, minute: int = 0) -> datetime:
    """A datetime on a known weekday (0 = Monday), independent of today's date."""
    monday = datetime(2026, 1, 1)
    monday -= timedelta(days=monday.weekday())
    return (monday + timedelta(days=weekday)).replace(hour=hour, minute=minute)
