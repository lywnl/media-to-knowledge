from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect

EXPECTED_TABLES = {
    "job",
    "video_asset",
    "video_object",
    "video_segment",
    "video_summary",
    "video_understanding_run",
}


def _config(database_url: str) -> Config:
    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", database_url)
    return config


def test_alembic_upgrade_and_downgrade_round_trip(tmp_path: Path) -> None:
    database_url = f"sqlite+pysqlite:///{tmp_path / 'migration.db'}"
    config = _config(database_url)

    command.upgrade(config, "head")
    engine = create_engine(database_url)
    assert EXPECTED_TABLES.issubset(set(inspect(engine).get_table_names()))

    command.downgrade(config, "base")
    assert not EXPECTED_TABLES.intersection(inspect(engine).get_table_names())

    command.upgrade(config, "head")
    assert EXPECTED_TABLES.issubset(set(inspect(engine).get_table_names()))
