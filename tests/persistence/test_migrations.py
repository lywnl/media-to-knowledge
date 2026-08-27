from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text

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


def test_document_artifact_migration_preserves_legacy_run_on_upgrade_and_downgrade(
    tmp_path: Path,
) -> None:
    database_url = f"sqlite+pysqlite:///{tmp_path / 'document-artifact.db'}"
    config = _config(database_url)
    command.upgrade(config, "0001_video_demo")
    engine = create_engine(database_url)
    with engine.begin() as connection:
        connection.execute(
            text(
                """
                INSERT INTO video_understanding_run (
                    tenant_id, application_id, knowledge_base_id, run_id, asset_id,
                    object_ref, idempotency_key, status, current_stage, warning_codes,
                    error_code, config_snapshot, artifact_manifest_relative_path,
                    artifact_manifest_sha256, created_at, updated_at
                ) VALUES (
                    'tenant-a', 'app-a', 'kb-a', 'run-legacy', 'asset-a',
                    'object-a', 'idempotency-legacy', 'SUCCEEDED', 'RESULT', '[]',
                    NULL, '{}', 'runs/legacy/result.json', :sha, CURRENT_TIMESTAMP,
                    CURRENT_TIMESTAMP
                )
                """,
            ),
            {"sha": "a" * 64},
        )

    command.upgrade(config, "head")
    columns = {
        column["name"]: column for column in inspect(engine).get_columns("video_understanding_run")
    }
    assert columns["document_relative_path"]["nullable"] is True
    assert columns["document_sha256"]["nullable"] is True
    assert columns["document_size_bytes"]["nullable"] is True
    with engine.connect() as connection:
        legacy = connection.execute(
            text(
                "SELECT artifact_manifest_relative_path, document_relative_path, "
                "document_sha256, document_size_bytes FROM video_understanding_run "
                "WHERE run_id = 'run-legacy'",
            ),
        ).one()
    assert legacy == ("runs/legacy/result.json", None, None, None)

    command.downgrade(config, "0001_video_demo")
    remaining_columns = {
        column["name"] for column in inspect(engine).get_columns("video_understanding_run")
    }
    assert "artifact_manifest_relative_path" in remaining_columns
    assert not {
        "document_relative_path",
        "document_sha256",
        "document_size_bytes",
    }.intersection(remaining_columns)
    with engine.connect() as connection:
        assert (
            connection.execute(
                text("SELECT run_id FROM video_understanding_run WHERE run_id = 'run-legacy'"),
            ).scalar_one()
            == "run-legacy"
        )
