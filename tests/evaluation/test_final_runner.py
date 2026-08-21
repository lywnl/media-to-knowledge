from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from video_demo.config import Settings
from video_demo.evaluation import final_runner as final_runner_module
from video_demo.evaluation.durability import (
    DurabilityRunReport,
    DurabilitySampleResult,
)
from video_demo.evaluation.evidence import (
    AuthorizedDatasetDetails,
    CommandTrace,
    EvidenceStore,
    MachineEvidenceReport,
    PerformanceDetails,
    PerformanceSampleDetails,
)
from video_demo.evaluation.final_runner import (
    REQUIREMENT_SPECS,
    FinalValidationBundle,
    FinalValidationRunner,
    OfflineGateRunner,
    RequirementEvidenceReport,
    RequirementEvidenceRow,
    bind_durability_to_quality,
    build_requirement_evidence_report,
    cleanup_evaluation_run,
    render_report_schema,
    stage_evaluation_run_id,
    write_report_schema,
)
from video_demo.evaluation.gate import (
    _FINAL_GATE_BUILD_TOKEN,
    FINAL_GATE_CHECKS,
    FinalGateReport,
    build_final_gate_report,
)
from video_demo.evaluation.metrics import RuntimeResourceMetrics
from video_demo.evaluation.report import (
    BoundQualityReport,
    GateStatus,
    MetricResult,
    build_quality_report,
)
from video_demo.evaluation.thresholds import QUALITY_THRESHOLDS

_NOW = datetime(2026, 8, 20, tzinfo=UTC)
_SHA = "a" * 64


def _annotation(sample_id: str, media_sha256: str, language: str) -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "sample_id": sample_id,
        "media_sha256": media_sha256,
        "duration_ms": 1_000,
        "language": language,
        "reference_text": "测试",
        "words": [
            {
                "word_id": f"word_{sample_id}",
                "text": "测试",
                "start_ms": 0,
                "end_ms": 500,
            }
        ],
        "speaker_turns": [
            {
                "turn_id": f"turn_{sample_id}",
                "speaker_id": "speaker_001",
                "start_ms": 0,
                "end_ms": 500,
            }
        ],
        "ocr_frames": [
            {
                "frame_id": f"frame_{sample_id}",
                "timestamp_ms": 100,
                "text_lines": ["测试"],
            }
        ],
        "audio_events": [
            {
                "event_id": f"event_{sample_id}",
                "normalized_event": "speech",
                "start_ms": 0,
                "end_ms": 500,
            }
        ],
        "scene_boundaries_ms": [100],
        "semantic_boundaries_ms": [200],
        "supported_facts": [
            {"fact_id": f"fact_{sample_id}", "canonical_text": "测试事实"}
        ],
        "key_fact_ids": [f"fact_{sample_id}"],
        "known_people": [],
    }


def _write_authorized_dataset(tmp_path: Path) -> tuple[Settings, EvidenceStore]:
    settings = Settings(workspace_root=tmp_path)
    assert settings.runtime_root is not None
    eval_root = settings.runtime_root / "eval"
    eval_root.mkdir(parents=True)
    media = eval_root / "media/shared.mp4"
    media.parent.mkdir(parents=True)
    media.write_bytes(b"authorized-media")
    media_sha256 = hashlib.sha256(media.read_bytes()).hexdigest()
    languages = ("zh", "en", "ja", "ko", "es")
    lines: list[str] = []
    for index in range(30):
        sample_id = f"sample_{index:02d}"
        language = languages[index % len(languages)]
        annotation = eval_root / f"annotations/{sample_id}.json"
        annotation.parent.mkdir(parents=True, exist_ok=True)
        annotation.write_text(
            json.dumps(
                _annotation(sample_id, media_sha256, language),
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
        lines.append(
            json.dumps(
                {
                    "sample_id": sample_id,
                    "language": language,
                    "authorization_id": "auth_001",
                    "media_relative_path": "media/shared.mp4",
                    "media_sha256": media_sha256,
                    "annotations_relative_path": f"annotations/{sample_id}.json",
                    "annotations_sha256": hashlib.sha256(
                        annotation.read_bytes()
                    ).hexdigest(),
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
    (eval_root / "dataset.jsonl").write_text("\n".join(lines), encoding="utf-8")
    (eval_root / "authorization.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "records": [
                    {
                        "schema_version": "1.0.0",
                        "authorization_id": "auth_001",
                        "source_category": "OWNED",
                        "allowed_purposes": ["VIDEO_QUALITY_EVALUATION"],
                        "confirmed_at": "2026-08-20T00:00:00Z",
                        "media_sha256": [media_sha256],
                    }
                ],
            },
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    return settings, EvidenceStore(tmp_path, settings.runtime_root)


def _quality(*, rtf_status: GateStatus = GateStatus.NOT_RUN) -> BoundQualityReport:
    rtf = MetricResult(
        name="rtf",
        value=None,
        threshold=3.0,
        direction="max",
        status=rtf_status,
        not_run_reason="尚未执行耐久测试",
    )
    return BoundQualityReport(
        schema_version="1.0.0",
        evaluation_run_id="eval_20260820_001",
        dataset_sha256=_SHA,
        authorization_sha256="b" * 64,
        prediction_index_sha256="c" * 64,
        judgment_index_sha256="d" * 64,
        sample_details_sha256="e" * 64,
        durability_report_sha256=None,
        status=GateStatus.NOT_RUN,
        metrics=(rtf,),
        resources=None,
        resources_not_run_reason="尚未执行耐久测试",
    )


def _sample(
    suffix: str,
    *,
    rtf: float,
    rss: int,
    disk: int,
    oom: bool = False,
    concurrency: int = 1,
    outside_writes: int = 0,
    terminal_status: str = "SUCCEEDED",
    failure_code: str | None = None,
) -> DurabilitySampleResult:
    return DurabilitySampleResult(
        media_sha256=hashlib.sha256(suffix.encode()).hexdigest(),
        duration_ms=1_800_000,
        width=1920,
        height=1080,
        elapsed_seconds=rtf * 1_800,
        rtf=rtf,
        peak_rss_bytes=rss,
        peak_disk_bytes=disk,
        oom=oom,
        peak_worker_concurrency=concurrency,
        outside_workspace_write_count=outside_writes,
        terminal_status=terminal_status,
        failure_code=failure_code,
    )


def _durability(*, status: GateStatus = GateStatus.PASS) -> DurabilityRunReport:
    return DurabilityRunReport(
        schema_version="1.0.0",
        evaluation_run_id="eval_20260820_001_durability",
        status=status,
        samples=(
            _sample("first", rtf=1.2, rss=120, disk=90),
            _sample("second", rtf=2.4, rss=100, disk=180),
        ),
        started_at=_NOW,
        finished_at=_NOW + timedelta(hours=1),
    )


def test_bind_durability_uses_worst_value_from_each_resource_dimension() -> None:
    report = bind_durability_to_quality(
        _quality(),
        _durability(),
        durability_report_sha256="f" * 64,
    )

    assert report.status == GateStatus.PASS
    assert report.resources == RuntimeResourceMetrics(
        rtf=2.4,
        peak_rss_bytes=120,
        peak_disk_bytes=180,
    )
    assert next(metric for metric in report.metrics if metric.name == "rtf").value == 2.4
    assert report.durability_report_sha256 == "f" * 64
    assert report.resources_not_run_reason is None


@pytest.mark.parametrize("status", (GateStatus.FAIL, GateStatus.NOT_RUN))
def test_bind_durability_rejects_non_pass_report(status: GateStatus) -> None:
    with pytest.raises(ValueError, match="耐久"):
        bind_durability_to_quality(
            _quality(),
            _durability(status=status),
            durability_report_sha256="f" * 64,
        )


@pytest.mark.parametrize(
    "bad_sample",
    (
        _sample("oom", rtf=1, rss=1, disk=1, oom=True),
        _sample("concurrency", rtf=1, rss=1, disk=1, concurrency=2),
        _sample("outside", rtf=1, rss=1, disk=1, outside_writes=1),
        _sample(
            "failed",
            rtf=1,
            rss=1,
            disk=1,
            terminal_status="FAILED",
            failure_code="CONTROLLED_FAILURE",
        ),
        _sample("slow", rtf=3.1, rss=1, disk=1),
    ),
)
def test_bind_durability_rejects_sample_that_does_not_meet_gate(
    bad_sample: DurabilitySampleResult,
) -> None:
    durability = _durability().model_copy(
        update={"samples": (_durability().samples[0], bad_sample)}
    )

    with pytest.raises(ValueError, match="耐久"):
        bind_durability_to_quality(
            _quality(),
            durability,
            durability_report_sha256="f" * 64,
        )


def test_requirement_report_contains_exactly_fixed_01_to_37(tmp_path: Path) -> None:
    final = build_final_gate_report(
        quality=_quality(),
        checks=(),
        workspace_root=tmp_path,
    )
    final_path = tmp_path / ".codex/video-rag-demo/eval/reports/eval_20260820_001/final.json"
    final_path.parent.mkdir(parents=True)
    final_path.write_text(final.model_dump_json(), encoding="utf-8")

    report = build_requirement_evidence_report(
        evaluation_run_id="eval_20260820_001",
        final=final,
        final_path=final_path,
        workspace_root=tmp_path,
    )

    assert tuple(row.requirement_id for row in report.rows) == tuple(range(1, 38))
    assert tuple(row.requirement for row in report.rows) == tuple(
        spec.requirement for spec in REQUIREMENT_SPECS
    )
    assert all(row.status == GateStatus.NOT_RUN for row in report.rows)
    assert set(check for row in report.rows for check in row.check_ids) <= set(
        FINAL_GATE_CHECKS
    )
    assert all(row.evidence_paths for row in report.rows)


def test_requirement_report_rejects_duplicate_or_missing_requirement() -> None:
    rows = tuple(
        RequirementEvidenceRow(
            requirement_id=spec.requirement_id,
            requirement=spec.requirement,
            check_ids=spec.check_ids,
            evidence_paths=(".codex/video-rag-demo/eval/reports/run/final.json",),
            status=GateStatus.NOT_RUN,
        )
        for spec in REQUIREMENT_SPECS
    )

    with pytest.raises(ValidationError, match=r"01|37|要求"):
        RequirementEvidenceReport(
            schema_version="1.0.0",
            evaluation_run_id="eval_20260820_001",
            final_report_sha256=_SHA,
            rows=(*rows[:-1], rows[0]),
        )


def test_report_schema_is_deterministic_and_covers_final_bundle(tmp_path: Path) -> None:
    first = render_report_schema()
    second = render_report_schema()

    assert first == second
    assert b'"FinalValidationBundle"' in first
    schema_path = tmp_path / "report.schema.json"
    schema_path.write_bytes(first)
    assert schema_path.read_bytes() == render_report_schema()
    assert FinalValidationBundle.model_json_schema()


def test_committed_report_schema_matches_current_model_bytes() -> None:
    schema_path = (
        Path(__file__).resolve().parents[2]
        / ".codex/video-rag-demo/eval/report.schema.json"
    )

    assert schema_path.read_bytes() == render_report_schema()


def test_summary_writer_rejects_symlinked_parent(tmp_path: Path) -> None:
    runtime = tmp_path / ".codex/video-rag-demo"
    runtime.mkdir(parents=True)
    target = tmp_path / "linked-eval-target"
    target.mkdir()
    (runtime / "eval").symlink_to(target, target_is_directory=True)

    with pytest.raises((OSError, ValueError)):
        write_report_schema(tmp_path)

    assert not (target / "report.schema.json").exists()


def test_authorized_dataset_runner_writes_and_reverifies_authoritative_report(
    tmp_path: Path,
) -> None:
    settings, store = _write_authorized_dataset(tmp_path)
    runner = FinalValidationRunner(settings, store)

    check = runner.build_authorized_dataset_check("eval_authorized")

    assert check is not None
    assert check.status == GateStatus.PASS
    assert len(check.evidence) == 1
    report_path = tmp_path / check.evidence[0].relative_path
    report = MachineEvidenceReport.model_validate_json(report_path.read_bytes())
    assert isinstance(report.details, AuthorizedDatasetDetails)
    assert report.details.item_count == 30
    assert report.details.language_counts == {
        "zh": 6,
        "en": 6,
        "ja": 6,
        "ko": 6,
        "es": 6,
    }
    assert {artifact.role for artifact in report.artifacts} == {
        "DATASET_MANIFEST",
        "AUTHORIZATION_RECORD",
        "COMMAND_STDOUT",
        "COMMAND_STDERR",
    }


def test_authorized_dataset_runner_keeps_missing_inputs_as_not_run(
    tmp_path: Path,
) -> None:
    settings = Settings(workspace_root=tmp_path)
    assert settings.runtime_root is not None
    settings.runtime_root.mkdir(parents=True)
    runner = FinalValidationRunner(
        settings,
        EvidenceStore(tmp_path, settings.runtime_root),
    )

    assert runner.build_authorized_dataset_check("eval_missing") is None
    assert not (
        settings.runtime_root / "eval/reports/eval_missing/authorized_dataset.json"
    ).exists()


def test_authorized_dataset_runner_fails_closed_for_present_but_invalid_inputs(
    tmp_path: Path,
) -> None:
    settings, store = _write_authorized_dataset(tmp_path)
    assert settings.runtime_root is not None
    (settings.runtime_root / "eval/annotations/sample_00.json").write_text(
        "{}",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="授权数据集非法或损坏"):
        FinalValidationRunner(settings, store).build_authorized_dataset_check(
            "eval_invalid"
        )

    assert not (
        settings.runtime_root / "eval/reports/eval_invalid/authorized_dataset.json"
    ).exists()


def test_final_runner_writes_exact_fifteen_checks_and_is_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(workspace_root=tmp_path)
    assert settings.runtime_root is not None
    settings.runtime_root.mkdir(parents=True)
    store = EvidenceStore(tmp_path, settings.runtime_root)
    runner = FinalValidationRunner(settings, store)
    failing_mypy = OfflineGateRunner(settings, store).run(
        "mypy",
        evaluation_run_id="eval_final",
    )
    assert failing_mypy.status == GateStatus.FAIL
    monkeypatch.setattr(
        runner,
        "_build_offline_checks",
        lambda _run_id: (failing_mypy,),
    )
    monkeypatch.setattr(
        runner,
        "_load_quality",
        lambda _run_id: build_quality_report({}, QUALITY_THRESHOLDS),
    )

    first = runner.final("eval_final")
    report_root = settings.runtime_root / "eval/reports/eval_final"
    paths = (
        report_root / "final.json",
        report_root / "requirement-evidence.json",
        report_root / "bundle.json",
    )
    first_bytes = tuple(path.read_bytes() for path in paths)
    second = runner.final("eval_final")

    final = FinalGateReport.model_validate_json(
        paths[0].read_bytes(),
        context={
            "workspace_root": tmp_path,
            "settings": settings,
            "build_token": _FINAL_GATE_BUILD_TOKEN,
        },
    )
    requirements = RequirementEvidenceReport.model_validate_json(paths[1].read_bytes())
    assert first.status == second.status == GateStatus.FAIL
    assert tuple(check.check_id for check in final.checks) == FINAL_GATE_CHECKS
    assert len(final.checks) == 15
    assert len(requirements.rows) == 37
    assert tuple(path.read_bytes() for path in paths) == first_bytes
    assert (
        settings.runtime_root / "eval/report.schema.json"
    ).read_bytes() == render_report_schema()


@pytest.mark.parametrize("evaluation_run_id", ("../escape", "run/escape", "..", "a"))
def test_cleanup_rejects_invalid_evaluation_run_id(
    tmp_path: Path,
    evaluation_run_id: str,
) -> None:
    with pytest.raises(ValueError, match="运行 ID"):
        cleanup_evaluation_run(tmp_path, evaluation_run_id)


def test_cleanup_rejects_active_or_symlinked_run(tmp_path: Path) -> None:
    runtime = tmp_path / ".codex/video-rag-demo"
    active = runtime / "eval/reports/eval_active"
    active.mkdir(parents=True)
    (active / ".real-media.incomplete").write_bytes(b"")
    with pytest.raises(ValueError, match="活跃"):
        cleanup_evaluation_run(tmp_path, "eval_active")

    target = runtime / "eval/reports/real-target"
    target.mkdir(parents=True)
    symlink = runtime / "eval/reports/eval_link"
    symlink.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="符号链接"):
        cleanup_evaluation_run(tmp_path, "eval_link")


def test_cleanup_only_removes_bound_eval_subtrees_and_writes_manifest(tmp_path: Path) -> None:
    runtime = tmp_path / ".codex/video-rag-demo"
    report = runtime / "eval/reports/eval_cleanup"
    prediction = runtime / "eval/predictions/eval_cleanup"
    stage_ids = tuple(
        stage_evaluation_run_id("eval_cleanup", stage)
        for stage in ("media", "baidu", "qwen", "pyannote", "models", "durability")
    )
    stage_reports = tuple(runtime / "eval/reports" / run_id for run_id in stage_ids)
    live_authorities = tuple(
        runtime / "eval/live-authority" / run_id
        for run_id in stage_ids
        if any(marker in run_id for marker in ("baidu", "qwen", "pyannote", "models"))
    )
    foreign = runtime / "eval/reports/eval_other"
    product = runtime / "runs/scope/run-unbound"
    for path in (
        report,
        prediction,
        *stage_reports,
        *live_authorities,
        foreign,
        product,
    ):
        path.mkdir(parents=True)
        (path / "keep.txt").write_text(path.name, encoding="utf-8")

    result = cleanup_evaluation_run(tmp_path, "eval_cleanup")

    assert not report.exists()
    assert not prediction.exists()
    assert all(not path.exists() for path in stage_reports)
    assert all(not path.exists() for path in live_authorities)
    assert foreign.is_dir()
    assert product.is_dir()
    assert set(result.deleted_paths) == {
        path.relative_to(tmp_path).as_posix()
        for path in (report, prediction, *stage_reports, *live_authorities)
    }
    assert result.manifest_path.startswith(".codex/video-rag-demo/eval/cleanup/")
    assert (tmp_path / result.manifest_path).is_file()


def test_durability_cleanup_requires_database_binding_even_with_result_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = tmp_path / ".codex/video-rag-demo"
    evaluation_run_id = "eval_cleanup_unbound_durability"
    stage_run_id = stage_evaluation_run_id(evaluation_run_id, "durability")
    report_path = runtime / "eval/reports" / stage_run_id / "durability.json"
    report_path.parent.mkdir(parents=True)
    report_path.write_bytes(b"{}")
    trace = CommandTrace(
        command=("python", "durability"),
        exit_code=0,
        stdout_sha256="1" * 64,
        stderr_sha256="2" * 64,
    )
    samples = tuple(
        PerformanceSampleDetails(
            sample_id=f"sample_{index}",
            media_relative_path=f"eval/durability/sample_{index}.mp4",
            sample_sha256=str(index) * 64,
            authorization_id=f"authorization_{index}",
            duration_ms=1_800_000,
            width=1920,
            height=1080,
            elapsed_seconds=1.0,
            rtf=1.0 / 1_800,
            oom_detected=False,
            peak_concurrency=1,
            outside_workspace_write_count=0,
            peak_rss_bytes=1,
            peak_disk_bytes=1,
            succeeded=True,
            terminal_status="SUCCEEDED",
            production_run_id=f"run_unbound_{index}",
            job_id=f"job_unbound_{index}",
            result_manifest_relative_path=(
                ".codex/video-rag-demo/runs/scope/"
                f"run_unbound_{index}/result/bundle-result.json"
            ),
            result_manifest_sha256=str(index + 2) * 64,
            probe_report_sha256=str(index + 4) * 64,
        )
        for index in (1, 2)
    )
    details = PerformanceDetails(
        type="PERFORMANCE",
        trace=trace,
        performance_report_sha256="7" * 64,
        evaluation_run_id=stage_run_id,
        manifest_sha256="8" * 64,
        authorization_sha256="9" * 64,
        implementation_sha256="a" * 64,
        settings_fingerprint="b" * 64,
        sample_report_sha256s=("c" * 64, "d" * 64),
        samples=samples,  # type: ignore[arg-type]
    )
    monkeypatch.setattr(
        final_runner_module,
        "build_verified_gate_check",
        lambda *_args, **_kwargs: type(
            "VerifiedCheck",
            (),
            {"status": GateStatus.PASS},
        )(),
    )
    monkeypatch.setattr(
        final_runner_module.MachineEvidenceReport,
        "model_validate_json",
        lambda _payload: type("VerifiedReport", (), {"details": details})(),
    )
    monkeypatch.setattr(
        final_runner_module,
        "_database_run_is_owned",
        lambda *_args, **_kwargs: False,
    )

    run_ids = final_runner_module._verified_durability_run_ids(
        tmp_path,
        runtime,
        evaluation_run_id,
    )

    assert run_ids == ()


def test_cleanup_database_binding_rejects_inconsistent_terminal_statuses(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / ".codex/video-rag-demo"
    runtime.mkdir(parents=True)
    database = runtime / "video-demo.db"
    connection = sqlite3.connect(database)
    try:
        connection.executescript(
            """
            CREATE TABLE video_asset (
                tenant_id TEXT NOT NULL,
                application_id TEXT NOT NULL,
                knowledge_base_id TEXT NOT NULL,
                asset_id TEXT NOT NULL,
                source_sha256 TEXT NOT NULL
            );
            CREATE TABLE video_understanding_run (
                tenant_id TEXT NOT NULL,
                application_id TEXT NOT NULL,
                knowledge_base_id TEXT NOT NULL,
                asset_id TEXT NOT NULL,
                run_id TEXT NOT NULL,
                idempotency_key TEXT NOT NULL,
                status TEXT NOT NULL
            );
            CREATE TABLE job (
                tenant_id TEXT NOT NULL,
                application_id TEXT NOT NULL,
                knowledge_base_id TEXT NOT NULL,
                resource_id TEXT NOT NULL,
                job_id TEXT NOT NULL,
                status TEXT NOT NULL
            );
            INSERT INTO video_asset VALUES (
                'evaluation', 'video-demo', 'evaluation', 'asset_1',
                'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'
            );
            INSERT INTO video_understanding_run VALUES (
                'evaluation', 'video-demo', 'evaluation', 'asset_1', 'run_1',
                'evaluation-key', 'FAILED'
            );
            INSERT INTO job VALUES (
                'evaluation', 'video-demo', 'evaluation', 'run_1', 'job_1',
                'SUCCEEDED'
            );
            """
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(ValueError, match="终态"):
        final_runner_module._database_run_is_owned(
            runtime,
            idempotency_key="evaluation-key",
            run_id="run_1",
            job_id="job_1",
            media_sha256="a" * 64,
            terminal_status="FAILED",
        )
