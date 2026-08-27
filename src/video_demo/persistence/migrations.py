from __future__ import annotations

import os
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine, make_url

from video_demo.errors import ErrorCode, VideoDemoError

try:
    import fcntl
except ImportError:  # pragma: no cover - 受支持平台均提供，分支用于明确失败语义。
    fcntl = None  # type: ignore[assignment]

_LEGACY_REVISION = "0001_video_demo"
_HEAD_REVISION = "0002_document_artifact"
_SCOPE = {
    "tenant_id": ("VARCHAR(128)", False),
    "application_id": ("VARCHAR(128)", False),
    "knowledge_base_id": ("VARCHAR(128)", False),
}
_TIMESTAMPS = {"created_at": ("DATETIME", False), "updated_at": ("DATETIME", False)}


def _columns(**specific: tuple[str, bool]) -> dict[str, tuple[str, bool]]:
    return {"id": ("INTEGER", False), **specific, **_SCOPE, **_TIMESTAMPS}


_LEGACY_COLUMNS = {
    "video_object": _columns(
        object_ref=("VARCHAR(128)", False),
        original_filename=("VARCHAR(512)", False),
        declared_mime=("VARCHAR(128)", False),
        detected_mime=("VARCHAR(128)", False),
        size_bytes=("BIGINT", False),
        sha256=("VARCHAR(64)", False),
        relative_path=("VARCHAR(1024)", False),
        status=("VARCHAR(32)", False),
        scan_details=("JSON", False),
    ),
    "video_asset": _columns(
        asset_id=("VARCHAR(128)", False),
        object_ref=("VARCHAR(128)", False),
        source_sha256=("VARCHAR(64)", False),
        manifest_relative_path=("VARCHAR(1024)", False),
        manifest_sha256=("VARCHAR(64)", False),
        schema_version=("VARCHAR(32)", False),
    ),
    "video_understanding_run": _columns(
        run_id=("VARCHAR(128)", False),
        asset_id=("VARCHAR(128)", False),
        object_ref=("VARCHAR(128)", False),
        idempotency_key=("VARCHAR(128)", False),
        status=("VARCHAR(32)", False),
        current_stage=("VARCHAR(64)", False),
        warning_codes=("JSON", False),
        error_code=("VARCHAR(128)", True),
        config_snapshot=("JSON", False),
        artifact_manifest_relative_path=("VARCHAR(1024)", True),
        artifact_manifest_sha256=("VARCHAR(64)", True),
    ),
    "job": _columns(
        job_id=("VARCHAR(128)", False),
        job_type=("VARCHAR(64)", False),
        resource_type=("VARCHAR(64)", False),
        resource_id=("VARCHAR(128)", False),
        status=("VARCHAR(32)", False),
        attempt_count=("INTEGER", False),
        max_attempts=("INTEGER", False),
        next_attempt_at=("DATETIME", False),
        worker_id=("VARCHAR(128)", True),
        lease_expires_at=("DATETIME", True),
        heartbeat_at=("DATETIME", True),
        cancel_requested=("BOOLEAN", False),
        error_code=("VARCHAR(128)", True),
    ),
    "video_segment": _columns(
        run_id=("VARCHAR(128)", False),
        segment_id=("VARCHAR(128)", False),
        start_ms=("INTEGER", False),
        end_ms=("INTEGER", False),
        schema_version=("VARCHAR(32)", False),
        payload_json=("JSON", False),
        retrieval_text=("TEXT", False),
        retrieval_hash=("VARCHAR(64)", False),
    ),
    "video_summary": _columns(
        run_id=("VARCHAR(128)", False),
        schema_version=("VARCHAR(32)", False),
        payload_json=("JSON", False),
        retrieval_text=("TEXT", False),
        retrieval_hash=("VARCHAR(64)", False),
    ),
}
_HEAD_COLUMNS = {
    **_LEGACY_COLUMNS,
    "video_understanding_run": {
        **_LEGACY_COLUMNS["video_understanding_run"],
        "document_relative_path": ("VARCHAR(1024)", True),
        "document_sha256": ("VARCHAR(64)", True),
        "document_size_bytes": ("BIGINT", True),
    },
}
_LEGACY_UNIQUES = {
    "video_object": {
        "uq_video_object_scope_ref": (
            "tenant_id",
            "application_id",
            "knowledge_base_id",
            "object_ref",
        )
    },
    "video_asset": {
        "uq_video_asset_scope_object_sha": (
            "tenant_id",
            "application_id",
            "knowledge_base_id",
            "object_ref",
            "source_sha256",
        ),
        "uq_video_asset_scope_id": ("tenant_id", "application_id", "knowledge_base_id", "asset_id"),
    },
    "video_understanding_run": {
        "uq_video_run_scope_id": ("tenant_id", "application_id", "knowledge_base_id", "run_id"),
        "uq_video_run_scope_idempotency": (
            "tenant_id",
            "application_id",
            "knowledge_base_id",
            "idempotency_key",
        ),
    },
    "job": {"uq_job_scope_id": ("tenant_id", "application_id", "knowledge_base_id", "job_id")},
    "video_segment": {
        "uq_video_segment_scope_run_id": (
            "tenant_id",
            "application_id",
            "knowledge_base_id",
            "run_id",
            "segment_id",
        )
    },
    "video_summary": {
        "uq_video_summary_scope_run": ("tenant_id", "application_id", "knowledge_base_id", "run_id")
    },
}
_LEGACY_INDEXES = {
    "video_object": {
        "ix_video_object_scope_sha": ("tenant_id", "application_id", "knowledge_base_id", "sha256")
    },
    "video_asset": {},
    "video_understanding_run": {},
    "job": {"ix_job_claim": ("status", "next_attempt_at", "lease_expires_at")},
    "video_segment": {
        "ix_video_segment_scope_run_time": (
            "tenant_id",
            "application_id",
            "knowledge_base_id",
            "run_id",
            "start_ms",
        )
    },
    "video_summary": {},
}


def upgrade_runtime_database(workspace_root: Path, runtime_root: Path, database_url: str) -> None:
    """在进程启动前串行升级本地 SQLite；未知旧结构一律失败关闭。"""

    workspace, runtime, migrations = _validate_paths(workspace_root, runtime_root)
    _validate_database_url(database_url, runtime)
    config = _alembic_config(migrations, database_url)
    with _migration_lock(runtime):
        engine = create_engine(database_url)
        try:
            _upgrade_locked(engine, config)
        except VideoDemoError:
            raise
        except Exception as error:
            raise VideoDemoError(
                ErrorCode.ARTIFACT_SCHEMA_INVALID,
                "运行时数据库迁移失败",
            ) from error
        finally:
            engine.dispose()
    _ = workspace


def _upgrade_locked(engine: Engine, config: Config) -> None:
    _require_no_unknown_schema_objects(engine)
    tables = _user_tables(engine)
    if not tables:
        command.upgrade(config, "head")
    elif "alembic_version" in tables:
        _require_single_revision(engine)
        command.upgrade(config, "head")
    else:
        _require_legacy_schema(engine)
        command.stamp(config, _LEGACY_REVISION)
        command.upgrade(config, "head")
    _require_no_unknown_schema_objects(engine)
    if _require_single_revision(engine) != _HEAD_REVISION:
        raise VideoDemoError(ErrorCode.ARTIFACT_SCHEMA_INVALID, "数据库未升级到唯一 head")
    if not _schema_matches(engine, _HEAD_COLUMNS):
        raise VideoDemoError(ErrorCode.ARTIFACT_SCHEMA_INVALID, "数据库 head 结构与冻结定义不一致")


def _validate_paths(workspace_root: Path, runtime_root: Path) -> tuple[Path, Path, Path]:
    try:
        workspace = workspace_root.expanduser().resolve(strict=True)
        runtime = runtime_root.expanduser().resolve(strict=True)
    except OSError as error:
        raise VideoDemoError(
            ErrorCode.INVALID_CONFIGURATION,
            "工作区或运行目录不存在",
        ) from error
    if not workspace.is_dir() or not runtime.is_dir() or not runtime.is_relative_to(workspace):
        raise VideoDemoError(ErrorCode.WORKSPACE_PATH_ESCAPE, "运行目录必须位于工作区内")
    workspace_migrations = workspace / "migrations"
    if not workspace_migrations.is_dir():
        raise VideoDemoError(ErrorCode.INVALID_CONFIGURATION, "工作区缺少 Alembic 迁移目录")
    try:
        migrations = workspace_migrations.resolve(strict=True)
    except OSError as error:
        raise VideoDemoError(ErrorCode.INVALID_CONFIGURATION, "Alembic 迁移目录不可用") from error
    if (
        not migrations.is_relative_to(workspace)
        or not (migrations / "env.py").is_file()
        or not (migrations / "versions" / "0001_video_demo.py").is_file()
    ):
        raise VideoDemoError(ErrorCode.INVALID_CONFIGURATION, "Alembic 迁移目录不可用")
    return workspace, runtime, migrations


def _validate_database_url(database_url: str, runtime: Path) -> None:
    url = make_url(database_url)
    if not url.drivername.startswith("sqlite") or not url.database or url.database == ":memory:":
        raise VideoDemoError(ErrorCode.INVALID_CONFIGURATION, "运行时迁移仅支持本地 SQLite 文件")
    database_path = Path(url.database).expanduser().resolve(strict=False)
    if not database_path.is_relative_to(runtime):
        raise VideoDemoError(ErrorCode.WORKSPACE_PATH_ESCAPE, "数据库文件必须位于运行目录内")


def _alembic_config(migrations: Path, database_url: str) -> Config:
    config = Config()
    config.attributes["configure_logging"] = False
    config.set_main_option("script_location", str(migrations))
    config.set_main_option("sqlalchemy.url", database_url.replace("%", "%%"))
    return config


@contextmanager
def _migration_lock(runtime: Path) -> Iterator[None]:
    if fcntl is None or not hasattr(os, "O_NOFOLLOW"):
        raise VideoDemoError(ErrorCode.INVALID_CONFIGURATION, "当前平台不支持安全数据库迁移锁")
    try:
        descriptor = os.open(
            runtime / ".database-migration.lock",
            os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW,
            0o600,
        )
    except OSError as error:
        raise VideoDemoError(
            ErrorCode.INVALID_CONFIGURATION,
            "数据库迁移锁无法安全打开",
        ) from error
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise VideoDemoError(ErrorCode.INVALID_CONFIGURATION, "数据库迁移锁必须是普通文件")
        os.fchmod(descriptor, 0o600)
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        os.close(descriptor)


def _user_tables(engine: Engine) -> set[str]:
    return {name for name in inspect(engine).get_table_names() if not name.startswith("sqlite_")}


def _require_no_unknown_schema_objects(engine: Engine) -> None:
    with engine.connect() as connection:
        objects = connection.execute(
            text(
                "SELECT type, name FROM sqlite_master "
                "WHERE type IN ('view', 'trigger') AND name NOT LIKE 'sqlite_%'"
            )
        ).all()
    if objects:
        raise VideoDemoError(ErrorCode.ARTIFACT_SCHEMA_INVALID, "数据库包含未知视图或触发器")


def _require_single_revision(engine: Engine) -> str:
    with engine.connect() as connection:
        rows = connection.execute(text("SELECT version_num FROM alembic_version")).scalars().all()
    if len(rows) != 1 or not rows[0]:
        raise VideoDemoError(ErrorCode.ARTIFACT_SCHEMA_INVALID, "Alembic revision 非法")
    return str(rows[0])


def _require_legacy_schema(engine: Engine) -> None:
    if _schema_matches(engine, _LEGACY_COLUMNS):
        return
    raise VideoDemoError(ErrorCode.ARTIFACT_SCHEMA_INVALID, "无版本数据库结构不是 0001")


def _schema_matches(
    engine: Engine,
    expected_schema: dict[str, dict[str, tuple[str, bool]]],
) -> bool:
    inspector = inspect(engine)
    business_tables = _user_tables(engine) - {"alembic_version"}
    if business_tables != set(expected_schema):
        return False
    for table, expected_columns in expected_schema.items():
        columns = {column["name"]: column for column in inspector.get_columns(table)}
        actual = {
            name: (str(column["type"]), bool(column["nullable"]))
            for name, column in columns.items()
        }
        primary_keys = {name for name, column in columns.items() if bool(column.get("primary_key"))}
        if (
            actual != expected_columns
            or primary_keys != {"id"}
            or any(column.get("default") is not None for column in columns.values())
            or inspector.get_foreign_keys(table)
            or inspector.get_check_constraints(table)
        ):
            return False
        uniques = {
            item["name"]: tuple(item["column_names"])
            for item in inspector.get_unique_constraints(table)
        }
        if uniques != _LEGACY_UNIQUES[table]:
            return False
        inspected_indexes = inspector.get_indexes(table)
        if any(item.get("unique") for item in inspected_indexes):
            return False
        indexes = {item["name"]: tuple(item["column_names"]) for item in inspected_indexes}
        if indexes != _LEGACY_INDEXES[table]:
            return False
    return True
