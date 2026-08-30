from __future__ import annotations

from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text

from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.persistence.database import Database
from video_demo.persistence.migrations import upgrade_runtime_database

HEAD_REVISION = "0006_audio_document_result"


@pytest.fixture
def workspace_runtime() -> Iterator[tuple[Path, Path]]:
    workspace_root = Path(__file__).resolve().parents[2]
    parent = workspace_root / ".codex" / "task9-tests"
    parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=parent) as directory:
        yield workspace_root, Path(directory)


def _alembic_config(workspace_root: Path, database_url: str) -> Config:
    config = Config()
    config.set_main_option("script_location", str((workspace_root / "migrations").resolve()))
    config.set_main_option("sqlalchemy.url", database_url)
    config.attributes["configure_logging"] = False
    return config


def _insert_legacy_run(database_url: str, run_id: str) -> None:
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
                    'tenant-a', 'app-a', 'kb-a', :run_id, 'asset-a', 'object-a',
                    :idempotency_key, 'SUCCEEDED', 'RESULT', '[]', NULL, '{}',
                    'runs/legacy/result.json', :sha, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
                )
                """,
            ),
            {"run_id": run_id, "idempotency_key": f"key-{run_id}", "sha": "a" * 64},
        )


def _assert_upgraded(database_url: str, run_id: str) -> None:
    engine = create_engine(database_url)
    assert {
        "document_relative_path",
        "document_sha256",
        "document_size_bytes",
    }.issubset(
        {column["name"] for column in inspect(engine).get_columns("video_understanding_run")}
    )
    assert not ({"retrieval_text", "retrieval_hash"} & {
        column["name"] for column in inspect(engine).get_columns("video_segment")
    })
    assert not ({"retrieval_text", "retrieval_hash"} & {
        column["name"] for column in inspect(engine).get_columns("video_summary")
    })
    with engine.connect() as connection:
        assert (
            connection.execute(
                text("SELECT version_num FROM alembic_version"),
            ).scalar_one()
            == HEAD_REVISION
        )
        assert (
            connection.execute(
                text("SELECT run_id FROM video_understanding_run WHERE run_id = :run_id"),
                {"run_id": run_id},
            ).scalar_one()
            == run_id
        )


def _upgrade_in_process(arguments: tuple[str, str, str]) -> str:
    workspace_root, runtime_root, database_url = arguments
    upgrade_runtime_database(Path(workspace_root), Path(runtime_root), database_url)
    return "ok"


def test_runtime_migration_upgrades_unversioned_0001_database_and_preserves_data(
    workspace_runtime: tuple[Path, Path],
) -> None:
    workspace_root, runtime_root = workspace_runtime
    database_url = f"sqlite+pysqlite:///{runtime_root / 'migration.db'}"
    command.upgrade(_alembic_config(workspace_root, database_url), "0001_video_demo")
    _insert_legacy_run(database_url, "run-from-migration")
    with create_engine(database_url).begin() as connection:
        connection.execute(text("DROP TABLE alembic_version"))

    upgrade_runtime_database(workspace_root, runtime_root, database_url)

    _assert_upgraded(database_url, "run-from-migration")


def test_runtime_migration_accepts_legacy_create_schema_equivalent_database(
    workspace_runtime: tuple[Path, Path],
) -> None:
    workspace_root, runtime_root = workspace_runtime
    database_url = f"sqlite+pysqlite:///{runtime_root / 'orm.db'}"
    command.upgrade(_alembic_config(workspace_root, database_url), "0001_video_demo")
    _insert_legacy_run(database_url, "run-from-create-schema")

    upgrade_runtime_database(workspace_root, runtime_root, database_url)

    _assert_upgraded(database_url, "run-from-create-schema")


def test_runtime_migration_rejects_unversioned_head_without_stamping(
    workspace_runtime: tuple[Path, Path],
) -> None:
    workspace_root, runtime_root = workspace_runtime
    database_url = f"sqlite+pysqlite:///{runtime_root / 'unversioned-head.db'}"
    Database(database_url).create_schema()

    with pytest.raises(VideoDemoError) as raised:
        upgrade_runtime_database(workspace_root, runtime_root, database_url)

    assert raised.value.code == ErrorCode.ARTIFACT_SCHEMA_INVALID
    assert "alembic_version" not in inspect(create_engine(database_url)).get_table_names()


@pytest.mark.parametrize("object_type", ["VIEW", "TRIGGER"])
def test_runtime_migration_rejects_unknown_view_or_trigger_without_stamping(
    workspace_runtime: tuple[Path, Path],
    object_type: str,
) -> None:
    workspace_root, runtime_root = workspace_runtime
    database_url = f"sqlite+pysqlite:///{runtime_root / f'unknown-{object_type.lower()}.db'}"
    command.upgrade(_alembic_config(workspace_root, database_url), "0001_video_demo")
    with create_engine(database_url).begin() as connection:
        connection.execute(text("DROP TABLE alembic_version"))
        if object_type == "VIEW":
            connection.execute(text("CREATE VIEW unknown_view AS SELECT id FROM video_object"))
        else:
            connection.execute(
                text(
                    "CREATE TRIGGER unknown_trigger AFTER INSERT ON video_object "
                    "BEGIN UPDATE video_object SET status = NEW.status WHERE id = NEW.id; END"
                )
            )

    with pytest.raises(VideoDemoError) as raised:
        upgrade_runtime_database(workspace_root, runtime_root, database_url)

    assert raised.value.code == ErrorCode.ARTIFACT_SCHEMA_INVALID
    assert "alembic_version" not in inspect(create_engine(database_url)).get_table_names()


def test_runtime_migration_rejects_nonempty_unknown_schema(
    workspace_runtime: tuple[Path, Path],
) -> None:
    workspace_root, runtime_root = workspace_runtime
    database_url = f"sqlite+pysqlite:///{runtime_root / 'invalid.db'}"
    with create_engine(database_url).begin() as connection:
        connection.execute(text("CREATE TABLE unknown_business_table (id INTEGER PRIMARY KEY)"))

    with pytest.raises(VideoDemoError) as raised:
        upgrade_runtime_database(workspace_root, runtime_root, database_url)

    assert raised.value.code == ErrorCode.ARTIFACT_SCHEMA_INVALID
    assert inspect(create_engine(database_url)).get_table_names() == ["unknown_business_table"]


def test_runtime_migration_serializes_two_starting_processes(
    workspace_runtime: tuple[Path, Path],
) -> None:
    workspace_root, runtime_root = workspace_runtime
    database_url = f"sqlite+pysqlite:///{runtime_root / 'concurrent.db'}"
    arguments = (str(workspace_root), str(runtime_root), database_url)

    with ProcessPoolExecutor(max_workers=2) as executor:
        assert list(executor.map(_upgrade_in_process, (arguments, arguments))) == ["ok", "ok"]

    engine = create_engine(database_url)
    with engine.connect() as connection:
        assert connection.execute(text("SELECT version_num FROM alembic_version")).scalar_one() == (
            HEAD_REVISION
        )
    assert (runtime_root / ".database-migration.lock").stat().st_mode & 0o777 == 0o600


def test_runtime_migration_upgrades_versioned_0001_and_is_idempotent(
    workspace_runtime: tuple[Path, Path],
) -> None:
    workspace_root, runtime_root = workspace_runtime
    database_url = f"sqlite+pysqlite:///{runtime_root / 'versioned.db'}"
    command.upgrade(_alembic_config(workspace_root, database_url), "0001_video_demo")

    upgrade_runtime_database(workspace_root, runtime_root, database_url)
    upgrade_runtime_database(workspace_root, runtime_root, database_url)

    with create_engine(database_url).connect() as connection:
        assert connection.execute(text("SELECT version_num FROM alembic_version")).scalar_one() == (
            HEAD_REVISION
        )


def test_runtime_migration_rejects_unknown_revision(
    workspace_runtime: tuple[Path, Path],
) -> None:
    workspace_root, runtime_root = workspace_runtime
    database_url = f"sqlite+pysqlite:///{runtime_root / 'unknown-revision.db'}"
    command.upgrade(_alembic_config(workspace_root, database_url), "0001_video_demo")
    with create_engine(database_url).begin() as connection:
        connection.execute(text("UPDATE alembic_version SET version_num = 'unknown_revision'"))

    with pytest.raises(VideoDemoError) as raised:
        upgrade_runtime_database(workspace_root, runtime_root, database_url)

    assert raised.value.code == ErrorCode.ARTIFACT_SCHEMA_INVALID


def test_runtime_migration_rejects_head_revision_with_incomplete_schema(
    workspace_runtime: tuple[Path, Path],
) -> None:
    workspace_root, runtime_root = workspace_runtime
    database_url = f"sqlite+pysqlite:///{runtime_root / 'forged-head.db'}"
    command.upgrade(_alembic_config(workspace_root, database_url), "head")
    with create_engine(database_url).begin() as connection:
        connection.execute(text("ALTER TABLE video_understanding_run DROP COLUMN document_sha256"))

    with pytest.raises(VideoDemoError) as raised:
        upgrade_runtime_database(workspace_root, runtime_root, database_url)

    assert raised.value.code == ErrorCode.ARTIFACT_SCHEMA_INVALID


def test_runtime_migration_rejects_symlink_lock(
    workspace_runtime: tuple[Path, Path],
) -> None:
    workspace_root, runtime_root = workspace_runtime
    target = runtime_root / "target.lock"
    target.touch()
    (runtime_root / ".database-migration.lock").symlink_to(target)

    with pytest.raises(VideoDemoError) as raised:
        upgrade_runtime_database(
            workspace_root,
            runtime_root,
            f"sqlite+pysqlite:///{runtime_root / 'locked.db'}",
        )
    assert raised.value.code == ErrorCode.INVALID_CONFIGURATION


def test_runtime_migration_validates_workspace_and_runtime_roots(tmp_path: Path) -> None:
    narrow_workspace = tmp_path / "narrow-workspace"
    narrow_workspace.mkdir()
    outside = tmp_path / "outside-narrow-workspace"
    outside.mkdir()

    with pytest.raises(VideoDemoError) as raised:
        upgrade_runtime_database(
            narrow_workspace,
            outside,
            f"sqlite+pysqlite:///{outside / 'outside.db'}",
        )

    assert raised.value.code == ErrorCode.WORKSPACE_PATH_ESCAPE


def test_runtime_migration_rejects_workspace_without_migrations(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    runtime = workspace / "runtime"
    runtime.mkdir(parents=True)

    with pytest.raises(VideoDemoError) as raised:
        upgrade_runtime_database(
            workspace,
            runtime,
            f"sqlite+pysqlite:///{runtime / 'missing-migrations.db'}",
        )

    assert raised.value.code == ErrorCode.INVALID_CONFIGURATION
