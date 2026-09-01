from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from video_demo.application.publication_contracts import scope_key
from video_demo.persistence.database import Database
from video_demo.persistence.models import (
    JobStatus,
    RunStatusValue,
    VideoPipelineStageModel,
    VideoStageName,
    VideoUnderstandingRunModel,
)
from video_demo.persistence.scope import Scope


def _migration_module():
    path = Path(__file__).parents[2] / ".codex/audio-mp3-migration/migrate.py"
    spec = importlib.util.spec_from_file_location("audio_mp3_migration", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_migration_dry_run_then_apply_is_idempotent(tmp_path: Path) -> None:
    module = _migration_module()
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    database_path = runtime / "video-demo.db"
    database = Database(f"sqlite+pysqlite:///{database_path}")
    database.create_schema()
    scope = Scope("tenant", "app", "kb")
    run_id = "run_001"
    run_root = runtime / "runs" / scope_key(scope) / run_id
    checkpoint = run_root / "stages/transcription-checkpoint.json"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text(json.dumps({"schema_version": "2.0.0", "audio_path": "media/audio.wav"}))
    old_audio = run_root / "media/audio.wav"
    old_audio.parent.mkdir(parents=True)
    old_audio.write_bytes(b"wav")
    old_slice = run_root / "speech/slices/window.wav"
    old_slice.parent.mkdir(parents=True)
    old_slice.write_bytes(b"wav")
    old_snapshot = run_root / "speech/snapshots/asr-windows/window-old.json"
    old_snapshot.parent.mkdir(parents=True)
    old_snapshot.write_text(json.dumps({"schema_version": "2.0.0"}))

    with database.session() as session:
        session.add(
            VideoUnderstandingRunModel(
                tenant_id=scope.tenant_id,
                application_id=scope.application_id,
                knowledge_base_id=scope.knowledge_base_id,
                run_id=run_id,
                asset_id="asset_001",
                object_ref="obj_001",
                idempotency_key="idem_001",
                status=RunStatusValue.RUNNING,
                current_stage="LLM",
            )
        )
        session.add_all(
            [
                VideoPipelineStageModel(
                    tenant_id=scope.tenant_id,
                    application_id=scope.application_id,
                    knowledge_base_id=scope.knowledge_base_id,
                    run_id=run_id,
                    stage_name=VideoStageName.TRANSCRIPTION,
                    status=JobStatus.SUCCEEDED,
                    checkpoint_relative_path=(
                        f"runs/{scope_key(scope)}/{run_id}/stages/transcription-checkpoint.json"
                    ),
                    checkpoint_sha256="a" * 64,
                ),
                VideoPipelineStageModel(
                    tenant_id=scope.tenant_id,
                    application_id=scope.application_id,
                    knowledge_base_id=scope.knowledge_base_id,
                    run_id=run_id,
                    stage_name=VideoStageName.LLM,
                    status=JobStatus.RUNNING,
                    worker_id="worker",
                ),
            ]
        )

    url = f"sqlite+pysqlite:///{database_path}"
    dry_report = module.migrate(tmp_path, runtime, url, dry_run=True)
    assert dry_report.reset_runs == 1
    assert old_audio.exists()
    assert checkpoint.exists()

    report = module.migrate(tmp_path, runtime, url, dry_run=False)
    assert report.reset_runs == 1
    assert not old_audio.exists()
    assert not old_slice.exists()
    assert not old_snapshot.exists()
    assert not checkpoint.exists()

    repeat = module.migrate(tmp_path, runtime, url, dry_run=False)
    assert repeat.reset_runs == 0

    with database.session() as session:
        stages = session.query(VideoPipelineStageModel).all()
        assert all(stage.status == JobStatus.PENDING for stage in stages)
        assert all(stage.checkpoint_relative_path is None for stage in stages)
        run = session.query(VideoUnderstandingRunModel).one()
        assert run.current_stage == "TRANSCRIPTION"
        assert run.status == RunStatusValue.PENDING


def test_current_video_mp3_checkpoint_is_not_legacy(tmp_path: Path) -> None:
    module = _migration_module()
    checkpoint = tmp_path / "transcription-checkpoint.json"
    checkpoint.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "upstream_sha256": "a" * 64,
                "payload": {
                    "schema_version": "3.0.0",
                    "prepared": {
                        "audio_path": "runs/tenant_app_kb/run_001/media/audio.mp3",
                        "audio_format_version": "mp3-192k-v1",
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    assert module._is_legacy_checkpoint(checkpoint, "3.0.0") is False


def test_current_audio_mp3_checkpoint_envelope_is_not_legacy(tmp_path: Path) -> None:
    module = _migration_module()
    checkpoint = tmp_path / "transcription-checkpoint.json"
    checkpoint.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "upstream_sha256": "b" * 64,
                "payload": {
                    "schema_version": "2.0.0",
                    "audio_format_version": "mp3-192k-v1",
                    "audio_path": "runs/tenant_app_kb/run_001/media/audio.mp3",
                },
            }
        ),
        encoding="utf-8",
    )

    assert module._is_legacy_checkpoint(checkpoint, "2.0.0") is False


def test_migration_ignores_checkpoint_symlink_without_deleting_target(tmp_path: Path) -> None:
    module = _migration_module()
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    target = runtime / "current-checkpoint.json"
    target.write_text("{}", encoding="utf-8")
    link = runtime / "linked-checkpoint.json"
    link.symlink_to(target)
    stage = SimpleNamespace(checkpoint_relative_path="linked-checkpoint.json")

    paths = module._old_checkpoint_paths(
        runtime,
        (stage,),
        latest_checkpoint_schema="3.0.0",
    )

    assert paths == ()
    assert target.exists()
