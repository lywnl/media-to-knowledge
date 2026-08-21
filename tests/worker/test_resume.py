from __future__ import annotations

from pathlib import Path

import pytest

from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.storage.artifacts import AtomicArtifactStore
from video_demo.worker.stages import StageContext, StageDefinition, StageRunner


def test_new_stage_runner_reuses_verified_artifact_after_restart(tmp_path: Path) -> None:
    calls: list[str] = []
    stage = StageDefinition(
        name="probe",
        schema_version="1.0.0",
        relative_path=Path("probe/ffprobe.json"),
        execute=lambda _context: calls.append("probe") or {"format": "mp4"},
    )
    context = StageContext(
        run_id="run_001",
        run_relative_root=Path("runs/run_001"),
        source_sha256="a" * 64,
    )

    first = StageRunner(AtomicArtifactStore(tmp_path)).execute(context, (stage,))
    second = StageRunner(AtomicArtifactStore(tmp_path)).execute(context, (stage,))

    assert calls == ["probe"]
    assert first["probe"].reused is False
    assert second["probe"].reused is True


def test_new_stage_runner_recomputes_corrupted_artifact_after_restart(tmp_path: Path) -> None:
    calls: list[str] = []
    stages = (
        StageDefinition(
            name="probe",
            schema_version="1.0.0",
            relative_path=Path("probe/ffprobe.json"),
            execute=lambda _context: calls.append("probe") or {"format": "mp4"},
        ),
        StageDefinition(
            name="transcode",
            schema_version="1.0.0",
            relative_path=Path("media/transcode.json"),
            execute=lambda _context: calls.append("transcode") or {"audio": "audio.wav"},
        ),
    )
    context = StageContext(
        run_id="run_001",
        run_relative_root=Path("runs/run_001"),
        source_sha256="a" * 64,
    )
    receipts = StageRunner(AtomicArtifactStore(tmp_path)).execute(context, stages)
    corrupted = tmp_path / receipts["probe"].receipt.relative_path
    corrupted.write_text("{}", encoding="utf-8")

    second = StageRunner(AtomicArtifactStore(tmp_path)).execute(context, stages)

    assert calls == ["probe", "transcode", "probe", "transcode"]
    assert second["probe"].reused is False
    assert second["transcode"].reused is False


def test_stage_runner_checks_cancellation_before_each_stage(tmp_path: Path) -> None:
    calls: list[str] = []

    def is_cancel_requested() -> bool:
        return len(calls) == 1

    context = StageContext(
        run_id="run_001",
        run_relative_root=Path("runs/run_001"),
        source_sha256="a" * 64,
        is_cancel_requested=is_cancel_requested,
    )
    stages = (
        StageDefinition(
            name="probe",
            schema_version="1.0.0",
            relative_path=Path("probe/ffprobe.json"),
            execute=lambda _context: calls.append("probe") or {"format": "mp4"},
        ),
        StageDefinition(
            name="transcode",
            schema_version="1.0.0",
            relative_path=Path("media/transcode.json"),
            execute=lambda _context: calls.append("transcode") or {"audio": "audio.wav"},
        ),
    )

    with pytest.raises(VideoDemoError) as raised:
        StageRunner(AtomicArtifactStore(tmp_path)).execute(context, stages)

    assert raised.value.code == ErrorCode.JOB_CANCELLED
    assert calls == ["probe"]


def test_stage_runner_records_duration_for_executed_stage(tmp_path: Path) -> None:
    timestamps = iter((10.0, 10.125))
    runner = StageRunner(AtomicArtifactStore(tmp_path), clock=lambda: next(timestamps))
    context = StageContext(
        run_id="run_001",
        run_relative_root=Path("runs/run_001"),
        source_sha256="a" * 64,
    )
    stage = StageDefinition(
        name="probe",
        schema_version="1.0.0",
        relative_path=Path("probe/ffprobe.json"),
        execute=lambda _context: {"format": "mp4"},
    )

    execution = runner.execute(context, (stage,))["probe"]

    assert execution.duration_ms == 125
