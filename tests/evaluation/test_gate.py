from __future__ import annotations

import hashlib
import json
import logging
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from xml.etree import ElementTree

import pytest
from pydantic import ValidationError

import video_demo.evaluation.evidence as evidence_module
import video_demo.evaluation.gate as gate_module
from video_demo.application.composition import build_production_model_identity_report
from video_demo.config import Settings
from video_demo.domain.run import ModelIdentity
from video_demo.errors import ErrorCode
from video_demo.evaluation.evidence import (
    BaiduLiveDetails,
    BaiduLiveRawReport,
    CommandTrace,
    EvidenceStore,
    FiveLanguageModelsDetails,
    FiveLanguageModelsRawReport,
    LiveExecutionSummary,
    LiveInputArtifact,
    LiveSample,
    MachineEvidenceReport,
    ModelExecutionFact,
    OfflineEvidenceDetails,
    OfflineRawReport,
    PreflightDetails,
    PreflightIssue,
    PreflightRawReport,
    PyannoteLiveDetails,
    PyannoteLiveRawReport,
    QwenLiveDetails,
    QwenLiveRawReport,
    build_verified_gate_check,
)
from video_demo.evaluation.final_runner import OfflineGateRunner
from video_demo.evaluation.gate import (
    FAILURE_SCENARIO_TESTS,
    FINAL_GATE_CHECKS,
    REQUIRED_FAILURE_SCENARIOS,
    EvidenceKind,
    EvidenceLevel,
    EvidenceReference,
    FinalGateReport,
    GateCheck,
    build_automated_tests_check,
    build_failure_matrix_check,
    build_final_gate_report,
)
from video_demo.evaluation.report import GateStatus, build_quality_report
from video_demo.evaluation.thresholds import QUALITY_THRESHOLDS


def test_gate_keeps_legacy_evidence_model_imports() -> None:
    assert gate_module.CommandTrace.__module__ == "video_demo.evaluation.evidence"


def test_evaluation_package_keeps_public_exports_lazy() -> None:
    script = """
import json
import sys
import video_demo.evaluation as package
before = sorted(name for name in sys.modules if name.endswith(('live_runner', 'media_runner')))
evidence_module = package.EvidenceStore.__module__
live_module = package.LiveValidationRunner.__module__
print(json.dumps({'before': before, 'evidence': evidence_module, 'live': live_module}))
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) == {
        "before": [],
        "evidence": "video_demo.evaluation.evidence",
        "live": "video_demo.evaluation.live_runner",
    }


def test_strict_no_indexing_runner_can_form_pass_and_binds_current_inputs(
    tmp_path: Path,
) -> None:
    (tmp_path / "src/video_demo").mkdir(parents=True)
    (tmp_path / "src/video_demo/example.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text(
        "[project]\nname='minimal'\ndependencies=[]\n",
        encoding="utf-8",
    )
    (tmp_path / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    settings = Settings(workspace_root=tmp_path)
    assert settings.runtime_root is not None
    settings.runtime_root.mkdir(parents=True)

    check = OfflineGateRunner(
        settings,
        EvidenceStore(tmp_path, settings.runtime_root),
    ).run("no_indexing", evaluation_run_id="eval_offline_no_indexing")

    assert check.status == GateStatus.PASS
    report_path = tmp_path / check.evidence[0].relative_path
    report = MachineEvidenceReport.model_validate_json(report_path.read_bytes())
    assert isinstance(report.details, OfflineEvidenceDetails)
    assert report.details.raw_report_sha256
    assert report.details.input_sha256
    assert build_verified_gate_check(
        "no_indexing",
        report_path,
        workspace_root=tmp_path,
    ) == check


def test_strict_offline_verifier_rejects_changed_current_input(tmp_path: Path) -> None:
    (tmp_path / "src/video_demo").mkdir(parents=True)
    source = tmp_path / "src/video_demo/example.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text(
        "[project]\nname='minimal'\ndependencies=[]\n",
        encoding="utf-8",
    )
    (tmp_path / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    settings = Settings(workspace_root=tmp_path)
    assert settings.runtime_root is not None
    settings.runtime_root.mkdir(parents=True)
    check = OfflineGateRunner(
        settings,
        EvidenceStore(tmp_path, settings.runtime_root),
    ).run("no_indexing", evaluation_run_id="eval_offline_changed")
    source.write_text("VALUE = 2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="可信门禁检查"):
        build_verified_gate_check(
            "no_indexing",
            tmp_path / check.evidence[0].relative_path,
            workspace_root=tmp_path,
        )


def test_alembic_roundtrip_observation_does_not_leak_process_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    project_root = Path(__file__).resolve().parents[2]
    shutil.copy2(project_root / "alembic.ini", tmp_path / "alembic.ini")
    shutil.copytree(project_root / "migrations", tmp_path / "migrations")

    observations = gate_module._alembic_roundtrip_observations(tmp_path)
    captured = capsys.readouterr()

    assert observations == ()
    assert captured.out == ""
    assert captured.err == ""


def test_alembic_roundtrip_observation_preserves_process_logging_handlers(
    tmp_path: Path,
) -> None:
    class CloseTrackingHandler(logging.Handler):
        def __init__(self) -> None:
            super().__init__()
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1
            super().close()

    project_root = Path(__file__).resolve().parents[2]
    shutil.copy2(project_root / "alembic.ini", tmp_path / "alembic.ini")
    shutil.copytree(project_root / "migrations", tmp_path / "migrations")
    logger = logging.getLogger("video_demo.tests.alembic_process_logging")
    original_disabled = logger.disabled
    handler = CloseTrackingHandler()
    logger.addHandler(handler)

    try:
        observations = gate_module._alembic_roundtrip_observations(tmp_path)

        assert observations == ()
        assert handler.close_calls == 0
        assert logger.disabled is original_disabled
    finally:
        logger.removeHandler(handler)
        logger.disabled = original_disabled
        handler.close()


def test_offline_raw_and_detail_require_fixed_command_and_exact_binding() -> None:
    raw = OfflineRawReport(
        schema_version="1.0.0",
        check_id="no_indexing",
        evaluation_run_id="eval_offline_raw",
        status=GateStatus.PASS,
        input_sha256="1" * 64,
        command=("python", "-m", "video_demo.evaluation.final_runner", "no_indexing"),
        exit_code=0,
        stdout_sha256="2" * 64,
        stderr_sha256="3" * 64,
        audited_paths=("src", "pyproject.toml", "uv.lock"),
        violation_count=0,
        observations=(),
        observation_sha256=evidence_module.offline_observation_sha256(()),
    )

    with pytest.raises(ValidationError, match="命令"):
        OfflineRawReport.model_validate(
            raw.model_copy(
                update={"command": ("python", "-c", "print('self reported pass')")}
            ).model_dump()
        )


def _write_junit(
    path: Path,
    *,
    missing: set[str] | None = None,
    failed: set[str] | None = None,
    skipped: set[str] | None = None,
) -> None:
    missing = missing or set()
    failed = failed or set()
    skipped = skipped or set()
    suite = ElementTree.Element("testsuite", name="pytest")
    for node_ids in FAILURE_SCENARIO_TESTS.values():
        for node_id in node_ids:
            if node_id in missing:
                continue
            relative_path, test_name = node_id.split("::", maxsplit=1)
            class_name = relative_path.removesuffix(".py").replace("/", ".")
            case = ElementTree.SubElement(
                suite,
                "testcase",
                classname=class_name,
                name=test_name,
            )
            if node_id in failed:
                ElementTree.SubElement(case, "failure", message="failed")
            if node_id in skipped:
                ElementTree.SubElement(case, "skipped", message="skipped")
    path.write_bytes(ElementTree.tostring(suite, encoding="utf-8", xml_declaration=True))


def test_failure_matrix_requires_executed_junit_evidence(tmp_path: Path) -> None:
    report_path = tmp_path / "failure-matrix.xml"
    _write_junit(report_path)

    check = build_failure_matrix_check(report_path, workspace_root=tmp_path)

    assert check.status == GateStatus.PASS
    assert check.evidence[0].kind == EvidenceKind.PYTEST_JUNIT
    assert check.evidence[0].sha256 == hashlib.sha256(report_path.read_bytes()).hexdigest()
    assert set(check.evidence[0].covered_items) == set(REQUIRED_FAILURE_SCENARIOS)


def test_missing_or_skipped_junit_case_is_not_run_and_failure_dominates(
    tmp_path: Path,
) -> None:
    corrupted_case = FAILURE_SCENARIO_TESTS["corrupted_media"][0]
    missing_path = tmp_path / "missing.xml"
    failed_path = tmp_path / "failed.xml"
    _write_junit(missing_path, missing={corrupted_case})
    _write_junit(failed_path, failed={corrupted_case})

    missing = build_failure_matrix_check(missing_path, workspace_root=tmp_path)
    failed = build_failure_matrix_check(failed_path, workspace_root=tmp_path)

    assert missing.status == GateStatus.NOT_RUN
    assert "corrupted_media" in (missing.not_run_reason or "")
    assert missing.evidence
    assert failed.status == GateStatus.FAIL


def test_failure_matrix_rejects_report_outside_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.xml"
    _write_junit(outside)

    with pytest.raises(ValueError, match="工作区"):
        build_failure_matrix_check(outside, workspace_root=workspace)


def test_failure_matrix_rejects_symlink_and_malformed_junit(tmp_path: Path) -> None:
    report = tmp_path / "report.xml"
    report.write_text("<broken", encoding="utf-8")
    link = tmp_path / "linked.xml"
    link.symlink_to(report)

    with pytest.raises(ValueError, match="符号链接"):
        build_failure_matrix_check(link, workspace_root=tmp_path)
    with pytest.raises(ValueError, match="JUnit XML"):
        build_failure_matrix_check(report, workspace_root=tmp_path)


def test_evidence_file_over_limit_is_rejected(tmp_path: Path) -> None:
    oversized = tmp_path / "oversized.xml"
    with oversized.open("wb") as output:
        output.seek(64 * 1024 * 1024)
        output.write(b"x")

    with pytest.raises(ValueError, match="大小上限"):
        build_failure_matrix_check(oversized, workspace_root=tmp_path)


def _evidence_file(tmp_path: Path, name: str = "evidence.txt") -> EvidenceReference:
    path = tmp_path / name
    path.write_text("verified evidence", encoding="utf-8")
    return EvidenceReference(
        kind=EvidenceKind.COMMAND_REPORT,
        level=EvidenceLevel.CONTRACT,
        relative_path=name,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        covered_items=("failure_matrix",),
        summary="命令成功",
    )


def _forged_live_evidence(tmp_path: Path) -> EvidenceReference:
    path = tmp_path / "live.json"
    path.write_text(json.dumps({"message": "相信我"}), encoding="utf-8")
    return EvidenceReference(
        kind=EvidenceKind.LIVE_SERVICE_REPORT,
        level=EvidenceLevel.REAL_SERVICE,
        relative_path=path.name,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        covered_items=("qwen_live",),
        summary="自报已通过",
    )


@pytest.mark.parametrize(
    ("check_id", "details_type", "details_literal", "verifier_name"),
    (
        ("baidu_ocr_live", BaiduLiveDetails, "BAIDU_LIVE", "_verify_baidu_live"),
        ("qwen_live", QwenLiveDetails, "QWEN_LIVE", "_verify_qwen_live"),
        ("pyannote_live", PyannoteLiveDetails, "PYANNOTE_LIVE", "_verify_pyannote_live"),
        (
            "five_language_models",
            FiveLanguageModelsDetails,
            "FIVE_LANGUAGE_MODELS",
            "_verify_five_language_models",
        ),
    ),
)
def test_live_gate_dispatches_to_check_specific_verifier(
    check_id: str,
    details_type: type[object],
    details_literal: str,
    verifier_name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    details = details_type.model_validate(  # type: ignore[attr-defined]
        {
            "type": details_literal,
            "trace": {
                "command": ["pytest"],
                "exit_code": 0,
                "stdout_sha256": "1" * 64,
                "stderr_sha256": "2" * 64,
            },
            "raw_report_sha256": "3" * 64,
            "implementation_sha256": "4" * 64,
            "settings_fingerprint": "5" * 64,
            "dataset_sha256": "6" * 64,
            "authorization_sha256": "7" * 64,
        }
    )
    report = MachineEvidenceReport(
        schema_version="1.0.0",
        check_id=check_id,
        status=GateStatus.PASS,
        kind=EvidenceKind.LIVE_SERVICE_REPORT,
        level=EvidenceLevel.REAL_SERVICE,
        covered_items=(check_id,),
        summary="live verifier dispatch",
        producer="test",
        started_at="2026-08-18T01:00:00Z",
        finished_at="2026-08-18T01:00:01Z",
        details=details,  # type: ignore[arg-type]
    )
    called: list[str] = []

    def verifier(*_args: object, **_kwargs: object) -> bool:
        called.append(verifier_name)
        return True

    monkeypatch.setattr(gate_module, verifier_name, verifier)

    status, reason = gate_module._derive_machine_gate_status(
        check_id,
        report,
        {},
        tmp_path,
    )

    assert (status, reason) == (GateStatus.PASS, None)
    assert called == [verifier_name]


def test_current_live_implementation_digest_is_deterministic_with_missing_runner(
    tmp_path: Path,
) -> None:
    for relative_path in gate_module._LIVE_IMPLEMENTATION_FILES:
        if relative_path == Path("src/video_demo/evaluation/live_runner.py"):
            continue
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative_path.as_posix(), encoding="utf-8")

    digest = gate_module._current_live_implementation_sha256(tmp_path)

    assert isinstance(digest, str)
    assert len(digest) == 64
    assert digest == gate_module._current_live_implementation_sha256(tmp_path)


def test_current_live_implementation_digest_rejects_missing_required_source(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="实现文件"):
        gate_module._current_live_implementation_sha256(tmp_path)


def test_live_implementation_digest_covers_in_process_speech_runtime(
    tmp_path: Path,
) -> None:
    changed_path = Path("src/video_demo/speech/runtime.py")
    assert changed_path in gate_module._LIVE_IMPLEMENTATION_FILES
    project_root = Path(__file__).parents[2]
    for relative_path in gate_module._LIVE_IMPLEMENTATION_FILES:
        destination = tmp_path / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(project_root / relative_path, destination)

    before = gate_module._current_live_implementation_sha256(tmp_path)
    copied_source = tmp_path / changed_path
    copied_source.write_bytes(copied_source.read_bytes() + b"\n")

    assert gate_module._current_live_implementation_sha256(tmp_path) != before


def test_live_implementation_digest_excludes_production_subprocess_chain() -> None:
    assert Path("src/video_demo/speech/isolated.py") not in (
        gate_module._LIVE_IMPLEMENTATION_FILES
    )
    assert Path("src/video_demo/speech/subprocess_main.py") not in (
        gate_module._LIVE_IMPLEMENTATION_FILES
    )


def _live_annotation(
    media_sha256: str,
    *,
    sample_id: str = "sample-001",
    language: str = "zh",
) -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "sample_id": sample_id,
        "media_sha256": media_sha256,
        "duration_ms": 1_000,
        "language": language,
        "reference_text": "测试",
        "words": [
            {
                "word_id": "word-001",
                "text": "测试",
                "start_ms": 0,
                "end_ms": 500,
            }
        ],
        "speaker_turns": [
            {
                "turn_id": "turn-001",
                "speaker_id": "speaker-001",
                "start_ms": 0,
                "end_ms": 900,
            }
        ],
        "ocr_frames": [
            {
                "frame_id": "frame-001",
                "timestamp_ms": 100,
                "text_lines": ["测试"],
            }
        ],
        "audio_events": [
            {
                "event_id": "event-001",
                "normalized_event": "speech",
                "start_ms": 0,
                "end_ms": 500,
            }
        ],
        "scene_boundaries_ms": [100],
        "semantic_boundaries_ms": [200],
        "supported_facts": [{"fact_id": "fact-001", "canonical_text": "事实"}],
        "key_fact_ids": ["fact-001"],
        "known_people": [],
    }


def _stage_live_implementation(workspace_root: Path) -> None:
    project_root = Path(__file__).parents[2]
    for relative_path in gate_module._LIVE_IMPLEMENTATION_FILES:
        source = project_root / relative_path
        if not source.is_file():
            continue
        target = workspace_root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())


def _write_baidu_live_report(
    tmp_path: Path,
    *,
    media_relative_path: str = "media/sample-001.mp4",
    settings_fingerprint_override: str | None = None,
    production_settings: Settings | None = None,
    summary_run_id: str = "run-live",
) -> Path:
    _stage_live_implementation(tmp_path)
    runtime_root = tmp_path / ".codex/video-rag-demo"
    runtime_root.mkdir(parents=True)
    store = EvidenceStore(tmp_path, runtime_root)
    eval_root = runtime_root / "eval"
    media = eval_root / media_relative_path
    media.parent.mkdir(parents=True)
    media.write_bytes(b"authorized-source")
    media_sha256 = hashlib.sha256(media.read_bytes()).hexdigest()
    annotation = eval_root / "annotations/sample-001.json"
    annotation.parent.mkdir(parents=True)
    annotation.write_text(
        json.dumps(_live_annotation(media_sha256), ensure_ascii=False),
        encoding="utf-8",
    )
    annotation_sha256 = hashlib.sha256(annotation.read_bytes()).hexdigest()
    manifest = eval_root / "dataset.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "sample_id": "sample-001",
                "language": "zh",
                "authorization_id": "auth-001",
                "media_relative_path": media_relative_path,
                "media_sha256": media_sha256,
                "annotations_relative_path": "annotations/sample-001.json",
                "annotations_sha256": annotation_sha256,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    authorization = eval_root / "authorization.json"
    authorization.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "records": [
                    {
                        "schema_version": "1.0.0",
                        "authorization_id": "auth-001",
                        "source_category": "OWNED",
                        "allowed_purposes": ["VIDEO_QUALITY_EVALUATION"],
                        "confirmed_at": "2026-08-18T00:00:00Z",
                        "media_sha256": [media_sha256],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    live_root = eval_root / "live/run-live/sample-001"
    live_root.mkdir(parents=True)
    audio = live_root / "audio.wav"
    keyframe = live_root / "keyframe.jpg"
    clip = live_root / "clip.mp4"
    audio.write_bytes(b"audio")
    keyframe.write_bytes(b"keyframe")
    clip.write_bytes(b"clip")
    sample = LiveSample(
        sample_id="sample-001",
        language="zh",
        duration_ms=1_000,
        source_media_relative_path=media.relative_to(tmp_path).as_posix(),
        source_media_sha256=media_sha256,
        audio_relative_path=audio.relative_to(tmp_path).as_posix(),
        audio_sha256=hashlib.sha256(audio.read_bytes()).hexdigest(),
        keyframe_relative_path=keyframe.relative_to(tmp_path).as_posix(),
        keyframe_sha256=hashlib.sha256(keyframe.read_bytes()).hexdigest(),
        clip_relative_path=clip.relative_to(tmp_path).as_posix(),
        clip_sha256=hashlib.sha256(clip.read_bytes()).hexdigest(),
        annotation_sha256=annotation_sha256,
    )
    inputs = tuple(
        LiveInputArtifact(
            kind=kind,
            sample_id=sample.sample_id,
            relative_path=path.relative_to(tmp_path).as_posix(),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            source_media_sha256=media_sha256,
            size_bytes=path.stat().st_size,
        )
        for kind, path in (
            ("SOURCE_MEDIA", media),
            ("AUDIO", audio),
            ("KEYFRAME", keyframe),
            ("CLIP", clip),
        )
    )
    request_id_sha256 = hashlib.sha256(b"123456789").hexdigest()
    model = ModelIdentity(
        component="baidu_ocr",
        provider="baidu_ocr",
        model_id="accurate_basic",
    )
    summary = LiveExecutionSummary(
        schema_version="1.0.0",
        component="baidu_ocr",
        operation="recognize",
        evaluation_run_id=summary_run_id,
        model=model,
        sample_id=sample.sample_id,
        language=sample.language,
        input_kind="KEYFRAME",
        input_sha256=sample.keyframe_sha256,
        request_id_sha256=request_id_sha256,
        http_status=200,
        output_item_count=1,
    )
    report_root = eval_root / "reports/run-live"
    response = store.write_artifact(
        Path("eval/reports/run-live/execution-000.json"),
        "PROVIDER_RESPONSE",
        summary.model_dump_json(exclude_none=True).encode("utf-8"),
    )
    fact = ModelExecutionFact(
        component="baidu_ocr",
        operation="recognize",
        evaluation_run_id="run-live",
        model=model,
        sample_id=sample.sample_id,
        language=sample.language,
        input_kind="KEYFRAME",
        input_sha256=sample.keyframe_sha256,
        output_sha256=response.sha256,
        request_id_sha256=request_id_sha256,
        http_status=200,
    )
    settings = production_settings or Settings(
        workspace_root=tmp_path,
        qwen_model_id="qwen3-vl-plus",
    )
    if settings.workspace_root != tmp_path.resolve(strict=True):
        raise ValueError("测试 settings 必须绑定 tmp_path")
    settings_fingerprint = build_production_model_identity_report(
        settings
    ).settings_fingerprint
    implementation_sha256 = gate_module._current_live_implementation_sha256(tmp_path)
    raw = BaiduLiveRawReport(
        schema_version="1.0.0",
        check_id="baidu_ocr_live",
        status=GateStatus.PASS,
        execution_started=True,
        evaluation_run_id="run-live",
        sample=sample,
        inputs=inputs,
        dataset_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        authorization_sha256=hashlib.sha256(authorization.read_bytes()).hexdigest(),
        settings_fingerprint=settings_fingerprint_override or settings_fingerprint,
        implementation_sha256=implementation_sha256,
        executions=(fact,),
    )
    raw_artifact = store.write_artifact(
        Path("eval/reports/run-live/raw.json"),
        "AUDIT_REPORT",
        raw.model_dump_json(exclude_none=True).encode("utf-8"),
    )
    stdout = store.write_artifact(
        Path("eval/reports/run-live/stdout.txt"),
        "COMMAND_STDOUT",
        b"live validation completed\n",
    )
    stderr = store.write_artifact(
        Path("eval/reports/run-live/stderr.txt"),
        "COMMAND_STDERR",
        b"",
    )
    artifacts = (
        raw_artifact,
        response,
        store.bind_artifact(manifest.relative_to(runtime_root), "DATASET_MANIFEST"),
        store.bind_artifact(
            authorization.relative_to(runtime_root),
            "AUTHORIZATION_RECORD",
        ),
        *(
            store.bind_artifact(path.relative_to(runtime_root), "INPUT_MEDIA")
            for path in (media, audio, keyframe, clip)
        ),
        stdout,
        stderr,
    )
    report = MachineEvidenceReport(
        schema_version="1.0.0",
        check_id="baidu_ocr_live",
        status=GateStatus.PASS,
        kind=EvidenceKind.LIVE_SERVICE_REPORT,
        level=EvidenceLevel.REAL_SERVICE,
        covered_items=("baidu_ocr_live",),
        summary="百度 OCR 真实调用通过",
        producer="live-runner",
        started_at="2026-08-18T01:00:00Z",
        finished_at="2026-08-18T01:00:01Z",
        artifacts=artifacts,
        details=BaiduLiveDetails(
            type="BAIDU_LIVE",
            trace=CommandTrace(
                command=("python", "-m", "video_demo.evaluation.live_runner"),
                exit_code=0,
                stdout_sha256=stdout.sha256,
                stderr_sha256=stderr.sha256,
            ),
            raw_report_sha256=raw_artifact.sha256,
            implementation_sha256=implementation_sha256,
            settings_fingerprint=raw.settings_fingerprint,
            dataset_sha256=raw.dataset_sha256,
            authorization_sha256=raw.authorization_sha256,
        ),
    )
    report_path = report_root / "baidu_ocr_live.json"
    store.write_json(report_path.relative_to(runtime_root), report)
    return report_path


def test_baidu_live_verifier_accepts_bound_authorized_execution(
    tmp_path: Path,
) -> None:
    report_path = _write_baidu_live_report(tmp_path)
    settings = Settings(workspace_root=tmp_path, qwen_model_id="qwen3-vl-plus")

    check = build_verified_gate_check(
        "baidu_ocr_live",
        report_path,
        workspace_root=tmp_path,
        settings=settings,
    )

    assert check.status == GateStatus.PASS


def test_executed_live_verifier_requires_settings(tmp_path: Path) -> None:
    report_path = _write_baidu_live_report(tmp_path)

    with pytest.raises(ValueError, match="可信门禁检查"):
        build_verified_gate_check(
            "baidu_ocr_live",
            report_path,
            workspace_root=tmp_path,
        )


def test_live_verifier_rejects_settings_with_foreign_workspace(tmp_path: Path) -> None:
    report_path = _write_baidu_live_report(tmp_path)
    foreign = tmp_path / "foreign"
    foreign.mkdir()

    with pytest.raises(ValueError, match="可信门禁检查"):
        build_verified_gate_check(
            "baidu_ocr_live",
            report_path,
            workspace_root=tmp_path,
            settings=Settings(
                workspace_root=foreign,
                qwen_model_id="qwen3-vl-plus",
            ),
        )


def test_live_verifier_rejects_noncanonical_runtime_root(tmp_path: Path) -> None:
    settings = Settings(
        workspace_root=tmp_path,
        runtime_root=Path(".codex/other-runtime"),
        qwen_model_id="qwen3-vl-plus",
    )
    report_path = _write_baidu_live_report(
        tmp_path,
        production_settings=settings,
    )

    with pytest.raises(ValueError, match="可信门禁检查"):
        build_verified_gate_check(
            "baidu_ocr_live",
            report_path,
            workspace_root=tmp_path,
            settings=settings,
        )


def test_live_report_publishes_external_authority_journal(tmp_path: Path) -> None:
    report_path = _write_baidu_live_report(tmp_path)
    journal = (
        tmp_path
        / ".codex/video-rag-demo/eval/live-authority/run-live/baidu_ocr_live.json"
    )

    assert journal.is_file()
    record = json.loads(journal.read_text(encoding="utf-8"))
    assert record["check_id"] == "baidu_ocr_live"
    assert record["evaluation_run_id"] == "run-live"
    assert record["machine_report_path"] == report_path.relative_to(tmp_path).as_posix()


def test_live_verifier_rejects_summary_from_other_run(tmp_path: Path) -> None:
    report_path = _write_baidu_live_report(
        tmp_path,
        summary_run_id="run-foreign",
    )

    with pytest.raises(ValueError, match="可信门禁检查"):
        build_verified_gate_check(
            "baidu_ocr_live",
            report_path,
            workspace_root=tmp_path,
            settings=Settings(
                workspace_root=tmp_path,
                qwen_model_id="qwen3-vl-plus",
            ),
        )


def test_live_authority_journal_is_exclusive(tmp_path: Path) -> None:
    report_path = _write_baidu_live_report(tmp_path)
    runtime_root = tmp_path / ".codex/video-rag-demo"
    journal_path = (
        runtime_root / "eval/live-authority/run-live/baidu_ocr_live.json"
    )
    original_journal = journal_path.read_bytes()
    store = EvidenceStore(tmp_path, runtime_root)
    report = MachineEvidenceReport.model_validate_json(report_path.read_bytes())
    report_path.unlink()

    with pytest.raises(ValueError, match="原子写入"):
        store.write_json(report_path.relative_to(runtime_root), report)

    assert journal_path.read_bytes() == original_journal
    with pytest.raises(ValueError, match="可信门禁检查"):
        build_verified_gate_check(
            "baidu_ocr_live",
            report_path,
            workspace_root=tmp_path,
            settings=Settings(
                workspace_root=tmp_path,
                qwen_model_id="qwen3-vl-plus",
            ),
        )


@pytest.mark.parametrize("failure_operation", ("unlink", "fsync"))
def test_live_authority_link_is_the_publication_linearization_point(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_operation: str,
) -> None:
    original_writer = evidence_module._write_exclusive_payload
    writer_calls = 0

    def fail_once_after_journal_link(
        descriptor: int,
        filename: str,
        payload: bytes,
    ) -> None:
        nonlocal writer_calls
        writer_calls += 1
        if writer_calls != 2:
            original_writer(descriptor, filename, payload)
            return
        original_link = evidence_module.os.link
        original_unlink = evidence_module.os.unlink
        original_fsync = evidence_module.os.fsync
        linked = False
        failed = False

        def record_link(*args: object, **kwargs: object) -> None:
            nonlocal linked
            original_link(*args, **kwargs)
            linked = True

        def fail_unlink_once(*args: object, **kwargs: object) -> None:
            nonlocal failed
            if failure_operation == "unlink" and linked and not failed:
                failed = True
                raise OSError("controlled post-link unlink failure")
            original_unlink(*args, **kwargs)

        def fail_fsync_once(descriptor: int) -> None:
            nonlocal failed
            if failure_operation == "fsync" and linked and not failed:
                failed = True
                raise OSError("controlled post-link fsync failure")
            original_fsync(descriptor)

        monkeypatch.setattr(evidence_module.os, "link", record_link)
        monkeypatch.setattr(evidence_module.os, "unlink", fail_unlink_once)
        monkeypatch.setattr(evidence_module.os, "fsync", fail_fsync_once)
        try:
            original_writer(descriptor, filename, payload)
        finally:
            monkeypatch.setattr(evidence_module.os, "link", original_link)
            monkeypatch.setattr(evidence_module.os, "unlink", original_unlink)
            monkeypatch.setattr(evidence_module.os, "fsync", original_fsync)

    monkeypatch.setattr(
        evidence_module,
        "_write_exclusive_payload",
        fail_once_after_journal_link,
    )

    report_path = _write_baidu_live_report(tmp_path)
    check = build_verified_gate_check(
        "baidu_ocr_live",
        report_path,
        workspace_root=tmp_path,
        settings=Settings(
            workspace_root=tmp_path,
            qwen_model_id="qwen3-vl-plus",
        ),
    )

    assert writer_calls == 2
    assert check.status == GateStatus.PASS


def test_live_verifier_rejects_deleted_authority_journal(tmp_path: Path) -> None:
    report_path = _write_baidu_live_report(tmp_path)
    journal = (
        tmp_path
        / ".codex/video-rag-demo/eval/live-authority/run-live/baidu_ocr_live.json"
    )
    journal.unlink()

    with pytest.raises(ValueError, match="可信门禁检查"):
        build_verified_gate_check(
            "baidu_ocr_live",
            report_path,
            workspace_root=tmp_path,
            settings=Settings(
                workspace_root=tmp_path,
                qwen_model_id="qwen3-vl-plus",
            ),
        )


def test_live_verifier_rejects_tampered_authority_journal(tmp_path: Path) -> None:
    report_path = _write_baidu_live_report(tmp_path)
    journal = (
        tmp_path
        / ".codex/video-rag-demo/eval/live-authority/run-live/baidu_ocr_live.json"
    )
    record = json.loads(journal.read_text(encoding="utf-8"))
    record["artifact_manifest_sha256"] = "f" * 64
    journal.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValueError, match="可信门禁检查"):
        build_verified_gate_check(
            "baidu_ocr_live",
            report_path,
            workspace_root=tmp_path,
            settings=Settings(
                workspace_root=tmp_path,
                qwen_model_id="qwen3-vl-plus",
            ),
        )


@pytest.mark.parametrize("copy_old_journal", (False, True))
def test_live_verifier_rejects_coordinated_run_replay_without_publisher(
    tmp_path: Path,
    copy_old_journal: bool,
) -> None:
    original_report = _write_baidu_live_report(tmp_path)
    replay_report = _copy_baidu_live_run(
        tmp_path,
        original_report,
        new_run_id="run-replay",
    )
    if copy_old_journal:
        old_journal = (
            tmp_path
            / ".codex/video-rag-demo/eval/live-authority/run-live/baidu_ocr_live.json"
        )
        copied_journal = (
            tmp_path
            / ".codex/video-rag-demo/eval/live-authority/run-replay/baidu_ocr_live.json"
        )
        copied_journal.parent.mkdir(parents=True)
        shutil.copyfile(old_journal, copied_journal)

    with pytest.raises(ValueError, match="可信门禁检查"):
        build_verified_gate_check(
            "baidu_ocr_live",
            replay_report,
            workspace_root=tmp_path,
            settings=Settings(
                workspace_root=tmp_path,
                qwen_model_id="qwen3-vl-plus",
            ),
        )


def test_public_artifact_writer_cannot_forge_authority_for_replayed_run(
    tmp_path: Path,
) -> None:
    original_report = _write_baidu_live_report(tmp_path)
    replay_report = _copy_baidu_live_run(
        tmp_path,
        original_report,
        new_run_id="run-replay",
    )
    runtime_root = tmp_path / ".codex/video-rag-demo"
    report = MachineEvidenceReport.model_validate_json(replay_report.read_bytes())
    details = report.details
    assert isinstance(details, BaiduLiveDetails)
    record = evidence_module.LiveAuthorityRecord(
        schema_version="1.0.0",
        check_id="baidu_ocr_live",
        evaluation_run_id="run-replay",
        machine_report_path=replay_report.relative_to(tmp_path).as_posix(),
        machine_report_sha256=hashlib.sha256(replay_report.read_bytes()).hexdigest(),
        raw_report_sha256=details.raw_report_sha256,
        settings_fingerprint=details.settings_fingerprint,
        implementation_sha256=details.implementation_sha256,
        machine_report_identity_sha256=evidence_module._identity_sha256(
            evidence_module._identity(replay_report.stat())
        ),
        report_run_directory_identity_sha256=evidence_module._identity_sha256(
            evidence_module._identity(replay_report.parent.stat())
        ),
        artifact_manifest_sha256=evidence_module._artifact_manifest_sha256(
            report.artifacts
        ),
    )
    store = EvidenceStore(tmp_path, runtime_root)

    with pytest.raises(ValueError, match="原子写入"):
        store.write_artifact(
            Path("eval/live-authority/run-replay/baidu_ocr_live.json"),
            "AUDIT_REPORT",
            record.model_dump_json().encode("utf-8"),
        )
    with pytest.raises(ValueError, match="可信门禁检查"):
        build_verified_gate_check(
            "baidu_ocr_live",
            replay_report,
            workspace_root=tmp_path,
            settings=Settings(
                workspace_root=tmp_path,
                qwen_model_id="qwen3-vl-plus",
            ),
        )


def _copy_baidu_live_run(
    tmp_path: Path,
    original_report: Path,
    *,
    new_run_id: str,
) -> Path:
    runtime_root = tmp_path / ".codex/video-rag-demo"
    old_run_id = "run-live"
    source_report_root = runtime_root / f"eval/reports/{old_run_id}"
    replay_report_root = runtime_root / f"eval/reports/{new_run_id}"
    shutil.copytree(source_report_root, replay_report_root)
    shutil.copytree(
        runtime_root / f"eval/live/{old_run_id}",
        runtime_root / f"eval/live/{new_run_id}",
    )
    response_path = replay_report_root / "execution-000.json"
    response = json.loads(response_path.read_text(encoding="utf-8"))
    response["evaluation_run_id"] = new_run_id
    response_path.write_text(json.dumps(response), encoding="utf-8")
    response_sha256 = hashlib.sha256(response_path.read_bytes()).hexdigest()
    raw_path = replay_report_root / "raw.json"
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    raw["evaluation_run_id"] = new_run_id
    raw["executions"][0]["evaluation_run_id"] = new_run_id
    raw["executions"][0]["output_sha256"] = response_sha256
    for field in (
        "audio_relative_path",
        "keyframe_relative_path",
        "clip_relative_path",
    ):
        raw["sample"][field] = raw["sample"][field].replace(
            f"/live/{old_run_id}/",
            f"/live/{new_run_id}/",
        )
    for item in raw["inputs"]:
        item["relative_path"] = item["relative_path"].replace(
            f"/live/{old_run_id}/",
            f"/live/{new_run_id}/",
        )
    raw_path.write_text(json.dumps(raw), encoding="utf-8")
    raw_sha256 = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    replay_report = replay_report_root / original_report.name
    report = json.loads(replay_report.read_text(encoding="utf-8"))
    report["details"]["raw_report_sha256"] = raw_sha256
    for artifact in report["artifacts"]:
        artifact["relative_path"] = artifact["relative_path"].replace(
            f"/reports/{old_run_id}/",
            f"/reports/{new_run_id}/",
        ).replace(
            f"/live/{old_run_id}/",
            f"/live/{new_run_id}/",
        )
        if artifact["relative_path"].endswith("/raw.json"):
            artifact["sha256"] = raw_sha256
        elif artifact["relative_path"].endswith("/execution-000.json"):
            artifact["sha256"] = response_sha256
    replay_report.write_text(json.dumps(report), encoding="utf-8")
    return replay_report


def test_live_verifier_rejects_self_reported_settings_fingerprint(
    tmp_path: Path,
) -> None:
    report_path = _write_baidu_live_report(
        tmp_path,
        settings_fingerprint_override="f" * 64,
    )

    with pytest.raises(ValueError, match="可信门禁检查"):
        build_verified_gate_check(
            "baidu_ocr_live",
            report_path,
            workspace_root=tmp_path,
            settings=Settings(
                workspace_root=tmp_path,
                qwen_model_id="qwen3-vl-plus",
            ),
        )


@pytest.mark.parametrize(
    "settings",
    (
        {"baidu_ocr_endpoint": "https://example.invalid/ocr"},
        {"qwen_base_url": "https://example.invalid/qwen"},
        {"qwen_timeout_seconds": 31.0},
    ),
)
def test_live_verifier_rejects_canonical_endpoint_or_timeout_drift(
    tmp_path: Path,
    settings: dict[str, object],
) -> None:
    report_path = _write_baidu_live_report(tmp_path)

    with pytest.raises(ValueError, match="可信门禁检查"):
        build_verified_gate_check(
            "baidu_ocr_live",
            report_path,
            workspace_root=tmp_path,
            settings=Settings(
                workspace_root=tmp_path,
                qwen_model_id="qwen3-vl-plus",
                **settings,
            ),
        )


@pytest.mark.parametrize(
    "mutation",
    ("raw_digest", "input_digest", "output_digest", "implementation_digest"),
)
def test_baidu_live_verifier_rejects_bound_fact_tampering(
    tmp_path: Path,
    mutation: str,
) -> None:
    report_path = _write_baidu_live_report(tmp_path)
    runtime_root = tmp_path / ".codex/video-rag-demo"
    raw_path = runtime_root / "eval/reports/run-live/raw.json"
    payload = json.loads(raw_path.read_text(encoding="utf-8"))
    if mutation == "raw_digest":
        report_path = runtime_root / "eval/reports/run-live/baidu_ocr_live.json"
        report_payload = json.loads(report_path.read_text(encoding="utf-8"))
        report_payload["details"]["raw_report_sha256"] = "f" * 64
        report_path.write_text(json.dumps(report_payload), encoding="utf-8")
    elif mutation == "input_digest":
        payload["sample"]["keyframe_sha256"] = "f" * 64
        payload["inputs"][2]["sha256"] = "f" * 64
        payload["executions"][0]["input_sha256"] = "f" * 64
        raw_path.write_text(json.dumps(payload), encoding="utf-8")
    elif mutation == "output_digest":
        payload["executions"][0]["output_sha256"] = "f" * 64
        raw_path.write_text(json.dumps(payload), encoding="utf-8")
    else:
        payload["implementation_sha256"] = "f" * 64
        raw_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="可信门禁检查"):
        build_verified_gate_check(
            "baidu_ocr_live",
            report_path,
            workspace_root=tmp_path,
            settings=Settings(
                workspace_root=tmp_path,
                qwen_model_id="qwen3-vl-plus",
            ),
        )


def test_baidu_live_verifier_rejects_extra_authority_artifact(
    tmp_path: Path,
) -> None:
    report_path = _write_baidu_live_report(tmp_path)
    runtime_root = tmp_path / ".codex/video-rag-demo"
    extra = runtime_root / "eval/extra-dataset.jsonl"
    extra.write_text('{"extra":"valid"}\n', encoding="utf-8")
    report_payload = json.loads(report_path.read_text(encoding="utf-8"))
    report_payload["artifacts"].append(
        {
            "role": "DATASET_MANIFEST",
            "relative_path": extra.relative_to(tmp_path).as_posix(),
            "sha256": hashlib.sha256(extra.read_bytes()).hexdigest(),
        }
    )
    report_path.write_text(json.dumps(report_payload), encoding="utf-8")

    with pytest.raises(ValueError, match="可信门禁检查"):
        build_verified_gate_check(
            "baidu_ocr_live",
            report_path,
            workspace_root=tmp_path,
            settings=Settings(
                workspace_root=tmp_path,
                qwen_model_id="qwen3-vl-plus",
            ),
        )


def test_baidu_live_verifier_rejects_machine_report_in_foreign_run(
    tmp_path: Path,
) -> None:
    report_path = _write_baidu_live_report(tmp_path)
    foreign_root = report_path.parent.parent / "foreign-run"
    foreign_root.mkdir()
    foreign_report = foreign_root / report_path.name
    foreign_report.write_bytes(report_path.read_bytes())

    with pytest.raises(ValueError, match="可信门禁检查"):
        build_verified_gate_check(
            "baidu_ocr_live",
            foreign_report,
            workspace_root=tmp_path,
            settings=Settings(
                workspace_root=tmp_path,
                qwen_model_id="qwen3-vl-plus",
            ),
        )


def test_baidu_live_verifier_rejects_generated_media_as_authorized_source(
    tmp_path: Path,
) -> None:
    report_path = _write_baidu_live_report(
        tmp_path,
        media_relative_path="generated/run-live/sample-001.mp4",
    )

    with pytest.raises(ValueError, match="可信门禁检查"):
        build_verified_gate_check(
            "baidu_ocr_live",
            report_path,
            workspace_root=tmp_path,
            settings=Settings(
                workspace_root=tmp_path,
                qwen_model_id="qwen3-vl-plus",
            ),
        )


def test_baidu_live_verifier_rejects_language_changed_only_in_live_evidence(
    tmp_path: Path,
) -> None:
    report_path = _write_baidu_live_report(tmp_path)
    report_root = tmp_path / ".codex/video-rag-demo/eval/reports/run-live"
    response_path = report_root / "execution-000.json"
    response_payload = json.loads(response_path.read_text(encoding="utf-8"))
    response_payload["language"] = "en"
    response_path.write_text(json.dumps(response_payload), encoding="utf-8")
    response_sha256 = hashlib.sha256(response_path.read_bytes()).hexdigest()
    raw_path = report_root / "raw.json"
    raw_payload = json.loads(raw_path.read_text(encoding="utf-8"))
    raw_payload["sample"]["language"] = "en"
    raw_payload["executions"][0]["language"] = "en"
    raw_payload["executions"][0]["output_sha256"] = response_sha256
    raw_path.write_text(json.dumps(raw_payload), encoding="utf-8")
    raw_sha256 = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    report_payload = json.loads(report_path.read_text(encoding="utf-8"))
    report_payload["details"]["raw_report_sha256"] = raw_sha256
    for artifact in report_payload["artifacts"]:
        if artifact["relative_path"].endswith("raw.json"):
            artifact["sha256"] = raw_sha256
        if artifact["relative_path"].endswith("execution-000.json"):
            artifact["sha256"] = response_sha256
    report_path.write_text(json.dumps(report_payload), encoding="utf-8")

    with pytest.raises(ValueError, match="可信门禁检查"):
        build_verified_gate_check(
            "baidu_ocr_live",
            report_path,
            workspace_root=tmp_path,
            settings=Settings(
                workspace_root=tmp_path,
                qwen_model_id="qwen3-vl-plus",
            ),
        )


def test_live_failure_rejects_completed_execution_without_next_stage(
    tmp_path: Path,
) -> None:
    _write_baidu_live_report(tmp_path)
    raw_path = (
        tmp_path
        / ".codex/video-rag-demo/eval/reports/run-live/raw.json"
    )
    payload = json.loads(raw_path.read_text(encoding="utf-8"))
    payload.update(
        status="FAIL",
        failure_code="DEPENDENCY_TEMPORARY_FAILURE",
        failure_component="baidu_ocr",
    )
    raw = BaiduLiveRawReport.model_validate(payload)

    with pytest.raises(ValueError, match="合法执行前缀"):
        gate_module._verify_live_execution_sequence(raw)


def test_live_close_failure_accepts_complete_execution_with_system_failure(
    tmp_path: Path,
) -> None:
    _write_qwen_live_report(tmp_path)
    raw_path = (
        tmp_path
        / ".codex/video-rag-demo/eval/reports/run-live/qwen-raw.json"
    )
    payload = json.loads(raw_path.read_text(encoding="utf-8"))
    payload.update(
        status="FAIL",
        failure_code="SYSTEM_FAILURE",
        failure_component="components_close",
    )
    raw = QwenLiveRawReport.model_validate(payload)

    gate_module._verify_live_execution_sequence(raw)


def test_live_failure_rejects_error_code_from_foreign_component(
    tmp_path: Path,
) -> None:
    _write_baidu_live_report(tmp_path)
    raw_path = (
        tmp_path
        / ".codex/video-rag-demo/eval/reports/run-live/raw.json"
    )
    payload = json.loads(raw_path.read_text(encoding="utf-8"))
    payload.update(
        status="FAIL",
        executions=[],
        failure_code="QWEN_AUTHENTICATION_FAILED",
        failure_component="baidu_ocr",
    )
    raw = BaiduLiveRawReport.model_validate(payload)

    with pytest.raises(ValueError, match="错误码"):
        gate_module._verify_live_execution_sequence(raw)


def _write_qwen_live_report(
    tmp_path: Path,
    *,
    model_id: str = "qwen3-vl-plus",
    status: GateStatus = GateStatus.PASS,
    completed_probe: bool = True,
    production_settings: Settings | None = None,
) -> Path:
    baidu_report_path = _write_baidu_live_report(
        tmp_path,
        production_settings=production_settings,
    )
    runtime_root = tmp_path / ".codex/video-rag-demo"
    store = EvidenceStore(tmp_path, runtime_root)
    baidu_machine = MachineEvidenceReport.model_validate_json(
        baidu_report_path.read_bytes()
    )
    baidu_raw = BaiduLiveRawReport.model_validate_json(
        (runtime_root / "eval/reports/run-live/raw.json").read_bytes()
    )
    model = ModelIdentity(
        component="qwen",
        provider="qwen",
        model_id=model_id,
    )
    facts: list[ModelExecutionFact] = []
    responses = []
    operations = (
        ("capability_probe", ("video_input", "strict_json_schema")),
        ("understand_segment", ()),
    )
    if status == GateStatus.FAIL and completed_probe:
        operations = operations[:1]
    elif status == GateStatus.FAIL:
        operations = ()
    for index, (operation, capabilities) in enumerate(operations):
        request_id_sha256 = hashlib.sha256(f"qwen-{index}".encode()).hexdigest()
        summary = LiveExecutionSummary(
            schema_version="1.0.0",
            component="qwen",
            operation=operation,
            evaluation_run_id="run-live",
            model=model,
            sample_id=baidu_raw.sample.sample_id,
            language=baidu_raw.sample.language,
            input_kind="CLIP",
            input_sha256=baidu_raw.sample.clip_sha256,
            request_id_sha256=request_id_sha256,
            http_status=200,
            capabilities=capabilities,
            output_item_count=index,
        )
        response = store.write_artifact(
            Path(f"eval/reports/run-live/qwen-{index}.json"),
            "PROVIDER_RESPONSE",
            summary.model_dump_json(exclude_none=True).encode("utf-8"),
        )
        responses.append(response)
        facts.append(
            ModelExecutionFact(
                component="qwen",
                operation=operation,
                evaluation_run_id="run-live",
                model=model,
                sample_id=baidu_raw.sample.sample_id,
                language=baidu_raw.sample.language,
                input_kind="CLIP",
                input_sha256=baidu_raw.sample.clip_sha256,
                output_sha256=response.sha256,
                request_id_sha256=request_id_sha256,
                http_status=200,
                capabilities=capabilities,
            )
        )
    raw = QwenLiveRawReport(
        schema_version="1.0.0",
        check_id="qwen_live",
        status=status,
        execution_started=True,
        evaluation_run_id=baidu_raw.evaluation_run_id,
        sample=baidu_raw.sample,
        inputs=baidu_raw.inputs,
        dataset_sha256=baidu_raw.dataset_sha256,
        authorization_sha256=baidu_raw.authorization_sha256,
        settings_fingerprint=baidu_raw.settings_fingerprint,
        implementation_sha256=baidu_raw.implementation_sha256,
        executions=tuple(facts),
        failure_code=(
            ErrorCode.DEPENDENCY_TEMPORARY_FAILURE
            if status == GateStatus.FAIL
            else None
        ),
        failure_component="qwen" if status == GateStatus.FAIL else None,
    )
    raw_artifact = store.write_artifact(
        Path("eval/reports/run-live/qwen-raw.json"),
        "AUDIT_REPORT",
        raw.model_dump_json(exclude_none=True).encode("utf-8"),
    )
    shared_artifacts = tuple(
        artifact
        for artifact in baidu_machine.artifacts
        if artifact.role not in {"AUDIT_REPORT", "PROVIDER_RESPONSE"}
    )
    details = QwenLiveDetails(
        type="QWEN_LIVE",
        trace=baidu_machine.details.trace.model_copy(
            update={"exit_code": 1 if status == GateStatus.FAIL else 0}
        ),
        raw_report_sha256=raw_artifact.sha256,
        implementation_sha256=raw.implementation_sha256,
        settings_fingerprint=raw.settings_fingerprint,
        dataset_sha256=raw.dataset_sha256,
        authorization_sha256=raw.authorization_sha256,
    )
    report = MachineEvidenceReport(
        schema_version="1.0.0",
        check_id="qwen_live",
        status=status,
        kind=EvidenceKind.LIVE_SERVICE_REPORT,
        level=EvidenceLevel.REAL_SERVICE,
        covered_items=("qwen_live",),
        summary=(
            "Qwen 能力探测成功但 segment 执行失败"
            if status == GateStatus.FAIL
            else "Qwen 真实两阶段调用通过"
        ),
        producer="live-runner",
        started_at="2026-08-18T01:00:00Z",
        finished_at="2026-08-18T01:00:01Z",
        artifacts=(*shared_artifacts, raw_artifact, *responses),
        details=details,
    )
    report_path = runtime_root / "eval/reports/run-live/qwen_live.json"
    store.write_json(report_path.relative_to(runtime_root), report)
    return report_path


def test_qwen_live_verifier_accepts_probe_then_segment_on_same_clip(
    tmp_path: Path,
) -> None:
    report_path = _write_qwen_live_report(tmp_path)
    settings = Settings(
        workspace_root=tmp_path,
        qwen_model_id="qwen3-vl-plus",
    )

    check = build_verified_gate_check(
        "qwen_live",
        report_path,
        workspace_root=tmp_path,
        settings=settings,
    )

    assert check.status == GateStatus.PASS


def test_qwen_live_verifier_rejects_other_legal_model_after_coordinated_rehash(
    tmp_path: Path,
) -> None:
    report_path = _write_qwen_live_report(
        tmp_path,
        model_id="qwen2.5-vl-max",
    )

    with pytest.raises(ValueError, match="可信门禁检查"):
        build_verified_gate_check(
            "qwen_live",
            report_path,
            workspace_root=tmp_path,
            settings=Settings(
                workspace_root=tmp_path,
                qwen_model_id="qwen3-vl-plus",
            ),
        )


def test_qwen_live_fail_accepts_successful_probe_prefix_at_gate(
    tmp_path: Path,
) -> None:
    report_path = _write_qwen_live_report(tmp_path, status=GateStatus.FAIL)

    check = build_verified_gate_check(
        "qwen_live",
        report_path,
        workspace_root=tmp_path,
        settings=Settings(
            workspace_root=tmp_path,
            qwen_model_id="qwen3-vl-plus",
        ),
    )

    assert check.status == GateStatus.FAIL


def test_qwen_executed_fail_without_facts_requires_configured_model(
    tmp_path: Path,
) -> None:
    settings = Settings(workspace_root=tmp_path)
    report_path = _write_qwen_live_report(
        tmp_path,
        status=GateStatus.FAIL,
        completed_probe=False,
        production_settings=settings,
    )

    with pytest.raises(ValueError, match="可信门禁检查"):
        build_verified_gate_check(
            "qwen_live",
            report_path,
            workspace_root=tmp_path,
            settings=settings,
        )


def test_qwen_live_verifier_rejects_summary_model_mismatch_after_rehash(
    tmp_path: Path,
) -> None:
    report_path = _write_qwen_live_report(tmp_path)
    runtime_root = tmp_path / ".codex/video-rag-demo"
    response_path = runtime_root / "eval/reports/run-live/qwen-0.json"
    raw_path = runtime_root / "eval/reports/run-live/qwen-raw.json"
    response_payload = json.loads(response_path.read_text(encoding="utf-8"))
    response_payload["model"]["model_id"] = "qwen2.5-vl-max"
    response_path.write_text(json.dumps(response_payload), encoding="utf-8")
    response_sha256 = hashlib.sha256(response_path.read_bytes()).hexdigest()
    raw_payload = json.loads(raw_path.read_text(encoding="utf-8"))
    raw_payload["executions"][0]["output_sha256"] = response_sha256
    raw_path.write_text(json.dumps(raw_payload), encoding="utf-8")
    raw_sha256 = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    report_payload = json.loads(report_path.read_text(encoding="utf-8"))
    report_payload["details"]["raw_report_sha256"] = raw_sha256
    for artifact in report_payload["artifacts"]:
        if artifact["relative_path"].endswith("qwen-raw.json"):
            artifact["sha256"] = raw_sha256
        if artifact["relative_path"].endswith("qwen-0.json"):
            artifact["sha256"] = response_sha256
    report_path.write_text(json.dumps(report_payload), encoding="utf-8")

    with pytest.raises(ValueError, match="可信门禁检查"):
        build_verified_gate_check(
            "qwen_live",
            report_path,
            workspace_root=tmp_path,
            settings=Settings(
                workspace_root=tmp_path,
                qwen_model_id="qwen3-vl-plus",
            ),
        )


def _write_pyannote_live_report(
    tmp_path: Path,
    *,
    device: str = "cpu",
) -> Path:
    baidu_report_path = _write_baidu_live_report(tmp_path)
    runtime_root = tmp_path / ".codex/video-rag-demo"
    store = EvidenceStore(tmp_path, runtime_root)
    baidu_machine = MachineEvidenceReport.model_validate_json(
        baidu_report_path.read_bytes()
    )
    baidu_raw = BaiduLiveRawReport.model_validate_json(
        (runtime_root / "eval/reports/run-live/raw.json").read_bytes()
    )
    model = next(
        item
        for item in build_production_model_identity_report(
            Settings(workspace_root=tmp_path)
        ).models
        if item.component == "pyannote"
    ).model_copy(update={"device": device})
    summary = LiveExecutionSummary(
        schema_version="1.0.0",
        component="pyannote",
        operation="diarize",
        evaluation_run_id="run-live",
        model=model,
        sample_id=baidu_raw.sample.sample_id,
        language=baidu_raw.sample.language,
        input_kind="AUDIO",
        input_sha256=baidu_raw.sample.audio_sha256,
        output_item_count=1,
    )
    response = store.write_artifact(
        Path("eval/reports/run-live/pyannote-0.json"),
        "PROVIDER_RESPONSE",
        summary.model_dump_json(exclude_none=True).encode("utf-8"),
    )
    fact = ModelExecutionFact(
        component="pyannote",
        operation="diarize",
        evaluation_run_id="run-live",
        model=model,
        sample_id=baidu_raw.sample.sample_id,
        language=baidu_raw.sample.language,
        input_kind="AUDIO",
        input_sha256=baidu_raw.sample.audio_sha256,
        output_sha256=response.sha256,
    )
    raw = PyannoteLiveRawReport(
        schema_version="1.0.0",
        check_id="pyannote_live",
        status=GateStatus.PASS,
        execution_started=True,
        evaluation_run_id=baidu_raw.evaluation_run_id,
        sample=baidu_raw.sample,
        inputs=baidu_raw.inputs,
        dataset_sha256=baidu_raw.dataset_sha256,
        authorization_sha256=baidu_raw.authorization_sha256,
        settings_fingerprint=baidu_raw.settings_fingerprint,
        implementation_sha256=baidu_raw.implementation_sha256,
        executions=(fact,),
    )
    raw_artifact = store.write_artifact(
        Path("eval/reports/run-live/pyannote-raw.json"),
        "AUDIT_REPORT",
        raw.model_dump_json(exclude_none=True).encode("utf-8"),
    )
    shared_artifacts = tuple(
        artifact
        for artifact in baidu_machine.artifacts
        if artifact.role not in {"AUDIT_REPORT", "PROVIDER_RESPONSE"}
    )
    report = MachineEvidenceReport(
        schema_version="1.0.0",
        check_id="pyannote_live",
        status=GateStatus.PASS,
        kind=EvidenceKind.LIVE_SERVICE_REPORT,
        level=EvidenceLevel.REAL_SERVICE,
        covered_items=("pyannote_live",),
        summary="pyannote 固定模型真实调用通过",
        producer="live-runner",
        started_at="2026-08-18T01:00:00Z",
        finished_at="2026-08-18T01:00:01Z",
        artifacts=(*shared_artifacts, raw_artifact, response),
        details=PyannoteLiveDetails(
            type="PYANNOTE_LIVE",
            trace=baidu_machine.details.trace,
            raw_report_sha256=raw_artifact.sha256,
            implementation_sha256=raw.implementation_sha256,
            settings_fingerprint=raw.settings_fingerprint,
            dataset_sha256=raw.dataset_sha256,
            authorization_sha256=raw.authorization_sha256,
        ),
    )
    report_path = runtime_root / "eval/reports/run-live/pyannote_live.json"
    store.write_json(report_path.relative_to(runtime_root), report)
    return report_path


def test_pyannote_live_verifier_accepts_fixed_model_and_authorized_audio(
    tmp_path: Path,
) -> None:
    report_path = _write_pyannote_live_report(tmp_path)

    check = build_verified_gate_check(
        "pyannote_live",
        report_path,
        workspace_root=tmp_path,
        settings=Settings(workspace_root=tmp_path, qwen_model_id="qwen3-vl-plus"),
    )

    assert check.status == GateStatus.PASS


def test_pyannote_live_verifier_rejects_noncanonical_mps_device(
    tmp_path: Path,
) -> None:
    report_path = _write_pyannote_live_report(tmp_path, device="mps")

    with pytest.raises(ValueError, match="可信门禁检查"):
        build_verified_gate_check(
            "pyannote_live",
            report_path,
            workspace_root=tmp_path,
            settings=Settings(
                workspace_root=tmp_path,
                qwen_model_id="qwen3-vl-plus",
            ),
        )


def _local_model_identity(
    component: str,
    language: str,
    *,
    device: str = "cpu",
) -> ModelIdentity:
    from importlib.metadata import version

    model_id = {
        "silero_vad": "silero-vad",
        "faster_whisper": "large-v3",
        "whisperx": f"whisperx-align-{language}",
        "yamnet": "yamnet",
    }[component]
    return ModelIdentity(
        component=component,
        provider="local",
        model_id=model_id,
        device=device,
        revision=version(
            {
                "silero_vad": "silero-vad",
                "faster_whisper": "faster-whisper",
                "whisperx": "whisperx",
                "yamnet": "tensorflow-hub",
            }[component]
        ),
    )


def _write_five_language_live_report(
    tmp_path: Path,
    *,
    device_overrides: dict[str, str] | None = None,
    settings: Settings | None = None,
) -> Path:
    _stage_live_implementation(tmp_path)
    runtime_root = tmp_path / ".codex/video-rag-demo"
    runtime_root.mkdir(parents=True)
    store = EvidenceStore(tmp_path, runtime_root)
    settings = settings or Settings(
        workspace_root=tmp_path,
        qwen_model_id="qwen3-vl-plus",
    )
    device_overrides = device_overrides or {}
    eval_root = runtime_root / "eval"
    languages = ("zh", "en", "ja", "ko", "es")
    samples: list[LiveSample] = []
    inputs: list[LiveInputArtifact] = []
    input_artifacts = []
    manifest_lines: list[str] = []
    media_digests: list[str] = []
    for language in languages:
        sample_id = f"sample-{language}"
        media = eval_root / f"media/{sample_id}.mp4"
        media.parent.mkdir(parents=True, exist_ok=True)
        media.write_bytes(f"source-{language}".encode())
        media_sha256 = hashlib.sha256(media.read_bytes()).hexdigest()
        media_digests.append(media_sha256)
        annotation = eval_root / f"annotations/{sample_id}.json"
        annotation.parent.mkdir(parents=True, exist_ok=True)
        annotation.write_text(
            json.dumps(
                _live_annotation(
                    media_sha256,
                    sample_id=sample_id,
                    language=language,
                ),
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        annotation_sha256 = hashlib.sha256(annotation.read_bytes()).hexdigest()
        manifest_lines.append(
            json.dumps(
                {
                    "sample_id": sample_id,
                    "language": language,
                    "authorization_id": "auth-001",
                    "media_relative_path": f"media/{sample_id}.mp4",
                    "media_sha256": media_sha256,
                    "annotations_relative_path": f"annotations/{sample_id}.json",
                    "annotations_sha256": annotation_sha256,
                }
            )
        )
        live_root = eval_root / f"live/run-live/{sample_id}"
        live_root.mkdir(parents=True)
        audio = live_root / "audio.wav"
        keyframe = live_root / "keyframe.jpg"
        clip = live_root / "clip.mp4"
        audio.write_bytes(f"audio-{language}".encode())
        keyframe.write_bytes(f"keyframe-{language}".encode())
        clip.write_bytes(f"clip-{language}".encode())
        sample = LiveSample(
            sample_id=sample_id,
            language=language,
            duration_ms=1_000,
            source_media_relative_path=media.relative_to(tmp_path).as_posix(),
            source_media_sha256=media_sha256,
            audio_relative_path=audio.relative_to(tmp_path).as_posix(),
            audio_sha256=hashlib.sha256(audio.read_bytes()).hexdigest(),
            keyframe_relative_path=keyframe.relative_to(tmp_path).as_posix(),
            keyframe_sha256=hashlib.sha256(keyframe.read_bytes()).hexdigest(),
            clip_relative_path=clip.relative_to(tmp_path).as_posix(),
            clip_sha256=hashlib.sha256(clip.read_bytes()).hexdigest(),
            annotation_sha256=annotation_sha256,
        )
        samples.append(sample)
        for kind, path in (
            ("SOURCE_MEDIA", media),
            ("AUDIO", audio),
            ("KEYFRAME", keyframe),
            ("CLIP", clip),
        ):
            inputs.append(
                LiveInputArtifact(
                    kind=kind,
                    sample_id=sample_id,
                    relative_path=path.relative_to(tmp_path).as_posix(),
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    source_media_sha256=media_sha256,
                    size_bytes=path.stat().st_size,
                )
            )
            input_artifacts.append(
                store.bind_artifact(path.relative_to(runtime_root), "INPUT_MEDIA")
            )
    manifest = eval_root / "dataset.jsonl"
    manifest.write_text("\n".join(manifest_lines), encoding="utf-8")
    authorization = eval_root / "authorization.json"
    authorization.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "records": [
                    {
                        "schema_version": "1.0.0",
                        "authorization_id": "auth-001",
                        "source_category": "OWNED",
                        "allowed_purposes": ["VIDEO_QUALITY_EVALUATION"],
                        "confirmed_at": "2026-08-18T00:00:00Z",
                        "media_sha256": media_digests,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    sample_by_language = {sample.language: sample for sample in samples}
    stages = (
        ("silero_vad", "vad", "zh"),
        *(("faster_whisper", "transcribe", language) for language in languages),
        *(("whisperx", "align", language) for language in languages),
        ("yamnet", "detect", "zh"),
    )
    facts: list[ModelExecutionFact] = []
    response_artifacts = []
    for index, (component, operation, language) in enumerate(stages):
        sample = sample_by_language[language]
        model = _local_model_identity(
            component,
            language,
            device=device_overrides.get(
                component,
                settings.inference_device if component == "faster_whisper" else "cpu",
            ),
        )
        summary = LiveExecutionSummary(
            schema_version="1.0.0",
            component=component,
            operation=operation,
            evaluation_run_id="run-live",
            model=model,
            sample_id=sample.sample_id,
            language=language,
            input_kind="AUDIO",
            input_sha256=sample.audio_sha256,
            output_item_count=1,
        )
        response = store.write_artifact(
            Path(f"eval/reports/run-live/local-{index:02d}.json"),
            "PROVIDER_RESPONSE",
            summary.model_dump_json(exclude_none=True).encode("utf-8"),
        )
        response_artifacts.append(response)
        facts.append(
            ModelExecutionFact(
                component=component,
                operation=operation,
                evaluation_run_id="run-live",
                model=model,
                sample_id=sample.sample_id,
                language=language,
                input_kind="AUDIO",
                input_sha256=sample.audio_sha256,
                output_sha256=response.sha256,
            )
        )
    implementation_sha256 = gate_module._current_live_implementation_sha256(tmp_path)
    settings_fingerprint = build_production_model_identity_report(
        settings
    ).settings_fingerprint
    raw = FiveLanguageModelsRawReport(
        schema_version="1.0.0",
        check_id="five_language_models",
        status=GateStatus.PASS,
        execution_started=True,
        evaluation_run_id="run-live",
        samples=tuple(samples),
        inputs=tuple(inputs),
        dataset_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        authorization_sha256=hashlib.sha256(authorization.read_bytes()).hexdigest(),
        settings_fingerprint=settings_fingerprint,
        implementation_sha256=implementation_sha256,
        executions=tuple(facts),
    )
    raw_artifact = store.write_artifact(
        Path("eval/reports/run-live/five-language-raw.json"),
        "AUDIT_REPORT",
        raw.model_dump_json(exclude_none=True).encode("utf-8"),
    )
    stdout = store.write_artifact(
        Path("eval/reports/run-live/local-stdout.txt"),
        "COMMAND_STDOUT",
        b"local stack completed\n",
    )
    stderr = store.write_artifact(
        Path("eval/reports/run-live/local-stderr.txt"),
        "COMMAND_STDERR",
        b"",
    )
    report = MachineEvidenceReport(
        schema_version="1.0.0",
        check_id="five_language_models",
        status=GateStatus.PASS,
        kind=EvidenceKind.LIVE_SERVICE_REPORT,
        level=EvidenceLevel.REAL_SERVICE,
        covered_items=("five_language_models",),
        summary="五语本地模型栈真实调用通过",
        producer="live-runner",
        started_at="2026-08-18T01:00:00Z",
        finished_at="2026-08-18T01:00:01Z",
        artifacts=(
            raw_artifact,
            store.bind_artifact(
                manifest.relative_to(runtime_root),
                "DATASET_MANIFEST",
            ),
            store.bind_artifact(
                authorization.relative_to(runtime_root),
                "AUTHORIZATION_RECORD",
            ),
            *input_artifacts,
            *response_artifacts,
            stdout,
            stderr,
        ),
        details=FiveLanguageModelsDetails(
            type="FIVE_LANGUAGE_MODELS",
            trace=CommandTrace(
                command=("python", "-m", "video_demo.evaluation.live_runner"),
                exit_code=0,
                stdout_sha256=stdout.sha256,
                stderr_sha256=stderr.sha256,
            ),
            raw_report_sha256=raw_artifact.sha256,
            implementation_sha256=implementation_sha256,
            settings_fingerprint=raw.settings_fingerprint,
            dataset_sha256=raw.dataset_sha256,
            authorization_sha256=raw.authorization_sha256,
        ),
    )
    report_path = runtime_root / "eval/reports/run-live/five_language_models.json"
    store.write_json(report_path.relative_to(runtime_root), report)
    return report_path


def test_five_language_live_verifier_accepts_exact_local_stack_coverage(
    tmp_path: Path,
) -> None:
    report_path = _write_five_language_live_report(tmp_path)

    check = build_verified_gate_check(
        "five_language_models",
        report_path,
        workspace_root=tmp_path,
        settings=Settings(workspace_root=tmp_path, qwen_model_id="qwen3-vl-plus"),
    )

    assert check.status == GateStatus.PASS


@pytest.mark.parametrize("component", ("silero_vad", "whisperx", "yamnet"))
def test_local_stack_verifier_rejects_noncanonical_mps_device(
    tmp_path: Path,
    component: str,
) -> None:
    settings = Settings(
        workspace_root=tmp_path,
        qwen_model_id="qwen3-vl-plus",
    )
    report_path = _write_five_language_live_report(
        tmp_path,
        device_overrides={component: "mps"},
        settings=settings,
    )

    with pytest.raises(ValueError, match="可信门禁检查"):
        build_verified_gate_check(
            "five_language_models",
            report_path,
            workspace_root=tmp_path,
            settings=settings,
        )


def test_faster_whisper_device_must_follow_settings_inference_device(
    tmp_path: Path,
) -> None:
    settings = Settings(
        workspace_root=tmp_path,
        qwen_model_id="qwen3-vl-plus",
        inference_device="mps",
    )
    report_path = _write_five_language_live_report(
        tmp_path,
        device_overrides={"faster_whisper": "cpu"},
        settings=settings,
    )

    with pytest.raises(ValueError, match="可信门禁检查"):
        build_verified_gate_check(
            "five_language_models",
            report_path,
            workspace_root=tmp_path,
            settings=settings,
        )


def test_five_language_live_verifier_rejects_reordered_language_stage(
    tmp_path: Path,
) -> None:
    report_path = _write_five_language_live_report(tmp_path)
    raw_path = (
        tmp_path
        / ".codex/video-rag-demo/eval/reports/run-live/five-language-raw.json"
    )
    raw_payload = json.loads(raw_path.read_text(encoding="utf-8"))
    raw_payload["executions"][1], raw_payload["executions"][2] = (
        raw_payload["executions"][2],
        raw_payload["executions"][1],
    )
    raw_path.write_text(json.dumps(raw_payload), encoding="utf-8")
    raw_sha256 = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    report_payload = json.loads(report_path.read_text(encoding="utf-8"))
    report_payload["details"]["raw_report_sha256"] = raw_sha256
    for artifact in report_payload["artifacts"]:
        if artifact["relative_path"].endswith("five-language-raw.json"):
            artifact["sha256"] = raw_sha256
    report_path.write_text(json.dumps(report_payload), encoding="utf-8")

    with pytest.raises(ValueError, match="可信门禁检查"):
        build_verified_gate_check(
            "five_language_models",
            report_path,
            workspace_root=tmp_path,
            settings=Settings(
                workspace_root=tmp_path,
                qwen_model_id="qwen3-vl-plus",
            ),
        )


def _write_live_preflight_report(tmp_path: Path) -> Path:
    _stage_live_implementation(tmp_path)
    runtime_root = tmp_path / ".codex/video-rag-demo"
    runtime_root.mkdir(parents=True)
    store = EvidenceStore(tmp_path, runtime_root)
    implementation_sha256 = gate_module._current_live_implementation_sha256(tmp_path)
    raw = PreflightRawReport(
        schema_version="1.0.0",
        check_id="baidu_ocr_live",
        reason_code="BAIDU_OCR_CREDENTIALS_UNAVAILABLE",
        execution_started=False,
        issues=(PreflightIssue(code=ErrorCode.BAIDU_API_KEY_UNAVAILABLE),),
        implementation_sha256=implementation_sha256,
        evaluation_run_id="run-preflight",
    )
    raw_artifact = store.write_artifact(
        Path("eval/reports/run-preflight/preflight.json"),
        "AUDIT_REPORT",
        raw.model_dump_json(exclude_none=True).encode("utf-8"),
    )
    stdout = store.write_artifact(
        Path("eval/reports/run-preflight/stdout.txt"),
        "COMMAND_STDOUT",
        b"preflight completed\n",
    )
    stderr = store.write_artifact(
        Path("eval/reports/run-preflight/stderr.txt"),
        "COMMAND_STDERR",
        b"",
    )
    report = MachineEvidenceReport(
        schema_version="1.0.0",
        check_id="baidu_ocr_live",
        status=GateStatus.NOT_RUN,
        kind=EvidenceKind.LIVE_SERVICE_REPORT,
        level=EvidenceLevel.REAL_SERVICE,
        covered_items=("baidu_ocr_live",),
        summary="百度 OCR 前置条件缺失",
        producer="live-runner",
        started_at="2026-08-18T01:00:00Z",
        finished_at="2026-08-18T01:00:01Z",
        not_run_reason="缺少百度 OCR 凭据或真实联调结果",
        artifacts=(raw_artifact, stdout, stderr),
        details=PreflightDetails(
            type="PREFLIGHT",
            trace=CommandTrace(
                command=("python", "-m", "video_demo.evaluation.live_runner"),
                exit_code=0,
                stdout_sha256=stdout.sha256,
                stderr_sha256=stderr.sha256,
            ),
            preflight_report_sha256=raw_artifact.sha256,
        ),
    )
    report_path = runtime_root / "eval/reports/run-preflight/baidu_ocr_live.json"
    store.write_json(report_path.relative_to(runtime_root), report)
    return report_path


def test_public_machine_report_writer_cannot_target_live_authority_namespace(
    tmp_path: Path,
) -> None:
    report_path = _write_live_preflight_report(tmp_path)
    runtime_root = tmp_path / ".codex/video-rag-demo"
    report = MachineEvidenceReport.model_validate_json(report_path.read_bytes())
    store = EvidenceStore(tmp_path, runtime_root)
    reserved = Path("eval/live-authority/run-preflight/baidu_ocr_live.json")

    with pytest.raises(ValueError, match="原子写入"):
        store.write_json(reserved, report)

    assert not (runtime_root / reserved).exists()


@pytest.mark.parametrize(
    "reserved",
    (
        Path("eval/LIVE-AUTHORITY/run-preflight/baidu_ocr_live.json"),
        Path("Eval/Live-Authority/run-preflight/baidu_ocr_live.json"),
        Path("EVAL/live-authority/run-preflight/baidu_ocr_live.json"),
    ),
)
def test_public_writers_reject_case_variant_live_authority_namespace(
    tmp_path: Path,
    reserved: Path,
) -> None:
    report_path = _write_live_preflight_report(tmp_path)
    runtime_root = tmp_path / ".codex/video-rag-demo"
    report = MachineEvidenceReport.model_validate_json(report_path.read_bytes())
    raw_payload = (
        runtime_root / "eval/reports/run-preflight/preflight.json"
    ).read_bytes()
    store = EvidenceStore(tmp_path, runtime_root)

    with pytest.raises(ValueError, match="原子写入"):
        store.write_json(reserved, report)
    with pytest.raises(ValueError, match="原子写入"):
        store.write_artifact(reserved, "AUDIT_REPORT", raw_payload)

    evidence_module._reject_public_live_authority_write(
        Path("eval/live-authorityish/run-preflight/baidu_ocr_live.json")
    )


def test_live_preflight_rejects_stale_implementation_after_rehash(
    tmp_path: Path,
) -> None:
    report_path = _write_live_preflight_report(tmp_path)
    raw_path = (
        tmp_path
        / ".codex/video-rag-demo/eval/reports/run-preflight/preflight.json"
    )
    raw_payload = json.loads(raw_path.read_text(encoding="utf-8"))
    raw_payload["implementation_sha256"] = "f" * 64
    raw_path.write_text(json.dumps(raw_payload), encoding="utf-8")
    raw_sha256 = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    report_payload = json.loads(report_path.read_text(encoding="utf-8"))
    report_payload["details"]["preflight_report_sha256"] = raw_sha256
    report_payload["artifacts"][0]["sha256"] = raw_sha256
    report_path.write_text(json.dumps(report_payload), encoding="utf-8")

    with pytest.raises(ValueError, match="可信门禁检查"):
        build_verified_gate_check(
            "baidu_ocr_live",
            report_path,
            workspace_root=tmp_path,
        )


def test_live_preflight_accepts_current_precise_not_run(tmp_path: Path) -> None:
    report_path = _write_live_preflight_report(tmp_path)

    check = build_verified_gate_check(
        "baidu_ocr_live",
        report_path,
        workspace_root=tmp_path,
    )

    assert check.status == GateStatus.NOT_RUN
    assert check.not_run_reason == "缺少百度 OCR 凭据或真实联调结果"


def test_final_gate_fills_missing_authoritative_checks_as_not_run(tmp_path: Path) -> None:
    quality = build_quality_report({}, QUALITY_THRESHOLDS)

    report = build_final_gate_report(quality=quality, checks=(), workspace_root=tmp_path)

    assert report.status == GateStatus.NOT_RUN
    assert {check.check_id for check in report.checks} == set(FINAL_GATE_CHECKS)
    assert all(check.status == GateStatus.NOT_RUN for check in report.checks)


def test_final_gate_revalidates_live_check_with_same_settings(tmp_path: Path) -> None:
    settings = Settings(
        workspace_root=tmp_path,
        qwen_model_id="qwen3-vl-plus",
    )
    report_path = _write_baidu_live_report(tmp_path)
    live_check = build_verified_gate_check(
        "baidu_ocr_live",
        report_path,
        workspace_root=tmp_path,
        settings=settings,
    )

    report = build_final_gate_report(
        quality=build_quality_report({}, QUALITY_THRESHOLDS),
        checks=(live_check,),
        workspace_root=tmp_path,
        settings=settings,
    )

    assert report.status == GateStatus.NOT_RUN
    with pytest.raises(ValueError, match=r"机器证据|live|settings"):
        build_final_gate_report(
            quality=build_quality_report({}, QUALITY_THRESHOLDS),
            checks=(live_check,),
            workspace_root=tmp_path,
        )
    with pytest.raises(ValueError, match=r"机器证据|live|settings"):
        build_final_gate_report(
            quality=build_quality_report({}, QUALITY_THRESHOLDS),
            checks=(live_check,),
            workspace_root=tmp_path,
            settings=Settings(
                workspace_root=tmp_path,
                qwen_model_id="qwen3-vl-plus",
                qwen_timeout_seconds=31.0,
            ),
        )


def test_final_gate_rejects_explicit_no_evidence_not_run_injection(
    tmp_path: Path,
) -> None:
    injected = GateCheck(
        check_id="qwen_live",
        status=GateStatus.NOT_RUN,
        not_run_reason="缺少 Qwen 凭据或真实联调结果",
    )

    with pytest.raises(ValueError, match="权威输入"):
        build_final_gate_report(
            quality=build_quality_report({}, QUALITY_THRESHOLDS),
            checks=(injected,),
            workspace_root=tmp_path,
        )


def test_final_gate_failure_dominates_missing_external_evidence(tmp_path: Path) -> None:
    quality = build_quality_report({}, QUALITY_THRESHOLDS)
    corrupted_case = FAILURE_SCENARIO_TESTS["corrupted_media"][0]
    report_path = tmp_path / "failed.xml"
    _write_junit(report_path, failed={corrupted_case})
    failed = build_failure_matrix_check(report_path, workspace_root=tmp_path)

    report = build_final_gate_report(
        quality=quality,
        checks=(failed,),
        workspace_root=tmp_path,
    )

    assert report.status == GateStatus.FAIL


def test_final_gate_machine_json_rejects_forged_pass() -> None:
    with pytest.raises(ValidationError):
        FinalGateReport.model_validate(
            {
                "status": "PASS",
                "quality": build_quality_report({}, QUALITY_THRESHOLDS).model_dump(),
                "checks": [],
            },
        )


def test_serialized_final_report_requires_workspace_verification_context(
    tmp_path: Path,
) -> None:
    report = build_final_gate_report(
        quality=build_quality_report({}, QUALITY_THRESHOLDS),
        checks=(),
        workspace_root=tmp_path,
    )

    with pytest.raises(ValidationError, match="工作区证据验证上下文"):
        FinalGateReport.model_validate(report.model_dump(mode="json"))

    with pytest.raises(ValidationError, match="权威构建流程"):
        FinalGateReport.model_validate(
            report.model_dump(mode="json"),
            context={"workspace_root": tmp_path},
        )


def test_gate_check_requires_reason_or_evidence_according_to_status() -> None:
    with pytest.raises(ValidationError):
        GateCheck(check_id="real_media_chain", status=GateStatus.NOT_RUN)
    with pytest.raises(ValidationError):
        GateCheck(check_id="real_media_chain", status=GateStatus.PASS)
    with pytest.raises(ValidationError):
        GateCheck(
            check_id="real_media_chain",
            status=GateStatus.PASS,
            evidence=("相信我",),
        )


def test_evidence_reference_rejects_absolute_and_parent_paths(tmp_path: Path) -> None:
    digest = "a" * 64
    for invalid_path in (str(tmp_path / "evidence.txt"), "../evidence.txt"):
        with pytest.raises(ValidationError):
            EvidenceReference(
                kind=EvidenceKind.COMMAND_REPORT,
                level=EvidenceLevel.CONTRACT,
                relative_path=invalid_path,
                sha256=digest,
                covered_items=("automated_tests",),
                summary="命令成功",
            )


def test_final_gate_rejects_contract_evidence_as_real_media_pass(tmp_path: Path) -> None:
    evidence = _evidence_file(tmp_path)
    check = GateCheck(
        check_id="real_media_chain",
        status=GateStatus.PASS,
        evidence=(
            evidence.model_copy(
                update={"covered_items": ("real_media_chain",)},
            ),
        ),
    )

    with pytest.raises(ValueError, match="证据级别"):
        build_final_gate_report(
            quality=build_quality_report({}, QUALITY_THRESHOLDS),
            checks=(check,),
            workspace_root=tmp_path,
        )


def test_final_gate_rejects_arbitrary_text_disguised_as_live_report(
    tmp_path: Path,
) -> None:
    check = GateCheck(
        check_id="qwen_live",
        status=GateStatus.PASS,
        evidence=(_forged_live_evidence(tmp_path),),
    )

    with pytest.raises(ValueError, match="机器证据"):
        build_final_gate_report(
            quality=build_quality_report({}, QUALITY_THRESHOLDS),
            checks=(check,),
            workspace_root=tmp_path,
        )


def test_live_report_requires_service_request_and_artifact_trace(tmp_path: Path) -> None:
    path = tmp_path / "live.json"
    payload = {
        "schema_version": "1.0.0",
        "check_id": "qwen_live",
        "status": "PASS",
        "kind": "LIVE_SERVICE_REPORT",
        "level": "REAL_SERVICE",
        "covered_items": ["qwen_live"],
        "summary": "真实服务调用通过",
        "producer": "integration-runner",
        "started_at": "2026-08-17T01:00:00Z",
        "finished_at": "2026-08-17T01:00:01Z",
        "details": {
            "type": "LIVE_SERVICE",
            "command": ["pytest", "-m", "integration"],
            "exit_code": 0,
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    check = GateCheck(
        check_id="qwen_live",
        status=GateStatus.PASS,
        evidence=(
            EvidenceReference(
                kind=EvidenceKind.LIVE_SERVICE_REPORT,
                level=EvidenceLevel.REAL_SERVICE,
                relative_path=path.name,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                covered_items=("qwen_live",),
                summary="真实服务调用通过",
            ),
        ),
    )

    with pytest.raises(ValueError, match="机器证据"):
        build_final_gate_report(
            quality=build_quality_report({}, QUALITY_THRESHOLDS),
            checks=(check,),
            workspace_root=tmp_path,
        )


def test_fake_live_service_summary_cannot_form_official_pass(
    tmp_path: Path,
) -> None:
    runtime_root = tmp_path / ".codex/video-rag-demo/eval/live"
    runtime_root.mkdir(parents=True)
    path = runtime_root / "qwen-live.json"
    input_path = runtime_root / "input.mp4"
    response_path = runtime_root / "provider-response.json"
    stdout_path = runtime_root / "stdout.bin"
    stderr_path = runtime_root / "stderr.bin"
    input_path.write_bytes(b"real input bytes")
    response_path.write_text(
        '{"id":"chatcmpl_real_001","model":"qwen3-vl-plus","choices":[{}]}',
        encoding="utf-8",
    )
    stdout_path.write_bytes(b"fake successful output")
    stderr_path.write_bytes(b"")
    summary = "Qwen 真实服务请求通过"
    payload = {
        "schema_version": "1.0.0",
        "check_id": "qwen_live",
        "status": "PASS",
        "kind": "LIVE_SERVICE_REPORT",
        "level": "REAL_SERVICE",
        "covered_items": ["qwen_live"],
        "summary": summary,
        "producer": "integration-runner",
        "started_at": "2026-08-17T01:00:00Z",
        "finished_at": "2026-08-17T01:00:01Z",
        "artifacts": [
            {
                "role": "INPUT_MEDIA",
                "relative_path": input_path.relative_to(tmp_path).as_posix(),
                "sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
            },
            {
                "role": "PROVIDER_RESPONSE",
                "relative_path": response_path.relative_to(tmp_path).as_posix(),
                "sha256": hashlib.sha256(response_path.read_bytes()).hexdigest(),
            },
            {
                "role": "COMMAND_STDOUT",
                "relative_path": stdout_path.relative_to(tmp_path).as_posix(),
                "sha256": hashlib.sha256(stdout_path.read_bytes()).hexdigest(),
            },
            {
                "role": "COMMAND_STDERR",
                "relative_path": stderr_path.relative_to(tmp_path).as_posix(),
                "sha256": hashlib.sha256(stderr_path.read_bytes()).hexdigest(),
            },
        ],
        "details": {
            "type": "LIVE_SERVICE",
            "trace": {
                "command": ["pytest", "-m", "integration", "tests/integration/test_qwen_live.py"],
                "exit_code": 0,
                "stdout_sha256": hashlib.sha256(stdout_path.read_bytes()).hexdigest(),
                "stderr_sha256": hashlib.sha256(stderr_path.read_bytes()).hexdigest(),
            },
            "service": "QWEN",
            "model_id": "qwen3-vl-plus",
            "request_id_sha256": hashlib.sha256(
                b"chatcmpl_real_001"
            ).hexdigest(),
            "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
            "output_sha256": hashlib.sha256(response_path.read_bytes()).hexdigest(),
            "http_status": 200,
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="无法形成可信门禁检查"):
        build_verified_gate_check(
            "qwen_live",
            path,
            workspace_root=tmp_path,
        )


def test_public_gate_builder_traceback_hides_missing_path_and_exception_chain(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "private" / "secret.xml"

    with pytest.raises(ValueError) as captured:
        build_failure_matrix_check(missing, workspace_root=tmp_path)

    rendered = "".join(
        traceback.format_exception(
            captured.type,
            captured.value,
            captured.tb,
            chain=True,
        )
    )
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    assert str(tmp_path) not in rendered
    assert "secret.xml" not in rendered
    assert "FileNotFoundError" not in rendered


def test_machine_report_must_live_under_runtime_root(tmp_path: Path) -> None:
    path = tmp_path / "audit.json"
    raw_path = tmp_path / "audit-output.txt"
    raw_path.write_text("0 violations", encoding="utf-8")
    summary = "无索引静态审计通过"
    payload = {
        "schema_version": "1.0.0",
        "check_id": "no_indexing",
        "status": "PASS",
        "kind": "STATIC_AUDIT",
        "level": "STATIC",
        "covered_items": ["no_indexing"],
        "summary": summary,
        "producer": "audit-runner",
        "started_at": "2026-08-17T01:00:00Z",
        "finished_at": "2026-08-17T01:00:01Z",
        "artifacts": [
            {
                "role": "AUDIT_REPORT",
                "relative_path": raw_path.name,
                "sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
            },
        ],
        "details": {
            "type": "STATIC_AUDIT",
            "trace": {
                "command": ["python", "-m", "video_demo.evaluation.no_indexing"],
                "exit_code": 0,
                "stdout_sha256": "a" * 64,
                "stderr_sha256": "b" * 64,
            },
            "audited_paths": ["src/video_demo"],
            "violation_count": 0,
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    check = GateCheck(
        check_id="no_indexing",
        status=GateStatus.PASS,
        evidence=(
            EvidenceReference(
                kind=EvidenceKind.STATIC_AUDIT,
                level=EvidenceLevel.STATIC,
                relative_path=path.name,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                covered_items=("no_indexing",),
                summary=summary,
            ),
        ),
    )

    with pytest.raises(ValueError, match="机器证据") as captured:
        build_final_gate_report(
            quality=build_quality_report({}, QUALITY_THRESHOLDS),
            checks=(check,),
            workspace_root=tmp_path,
        )
    assert str(tmp_path) not in str(captured.value)
    assert captured.value.__cause__ is None


def test_machine_report_binds_command_output_digests_to_artifacts(tmp_path: Path) -> None:
    runtime_root = tmp_path / ".codex/video-rag-demo/eval/audit"
    runtime_root.mkdir(parents=True)
    path = runtime_root / "audit.json"
    raw_path = runtime_root / "audit-output.txt"
    raw_path.write_text('{"violations":[]}', encoding="utf-8")
    summary = "无索引静态审计通过"
    payload = {
        "schema_version": "1.0.0",
        "check_id": "no_indexing",
        "status": "PASS",
        "kind": "STATIC_AUDIT",
        "level": "STATIC",
        "covered_items": ["no_indexing"],
        "summary": summary,
        "producer": "audit-runner",
        "started_at": "2026-08-17T01:00:00Z",
        "finished_at": "2026-08-17T01:00:01Z",
        "artifacts": [
            {
                "role": "AUDIT_REPORT",
                "relative_path": raw_path.relative_to(tmp_path).as_posix(),
                "sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
            },
        ],
        "details": {
            "type": "STATIC_AUDIT",
            "trace": {
                "command": ["python", "-m", "video_demo.evaluation.no_indexing"],
                "exit_code": 0,
                "stdout_sha256": "a" * 64,
                "stderr_sha256": "b" * 64,
            },
            "audited_paths": ["src/video_demo"],
            "violation_count": 0,
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    check = GateCheck(
        check_id="no_indexing",
        status=GateStatus.PASS,
        evidence=(
            EvidenceReference(
                kind=EvidenceKind.STATIC_AUDIT,
                level=EvidenceLevel.STATIC,
                relative_path=path.relative_to(tmp_path).as_posix(),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                covered_items=("no_indexing",),
                summary=summary,
            ),
        ),
    )

    with pytest.raises(ValueError, match="机器证据") as captured:
        build_final_gate_report(
            quality=build_quality_report({}, QUALITY_THRESHOLDS),
            checks=(check,),
            workspace_root=tmp_path,
        )
    assert captured.value.__cause__ is None


def test_live_report_rejects_missing_or_changed_trace_artifact(tmp_path: Path) -> None:
    runtime_root = tmp_path / ".codex/video-rag-demo/eval/live"
    runtime_root.mkdir(parents=True)
    input_path = runtime_root / "input.mp4"
    response_path = runtime_root / "provider-response.json"
    input_path.write_bytes(b"real input bytes")
    response_path.write_text('{"id":"chatcmpl_real_001"}', encoding="utf-8")
    evidence_path = runtime_root / "qwen-live.json"
    summary = "Qwen 真实服务请求通过"
    payload = {
        "schema_version": "1.0.0",
        "check_id": "qwen_live",
        "status": "PASS",
        "kind": "LIVE_SERVICE_REPORT",
        "level": "REAL_SERVICE",
        "covered_items": ["qwen_live"],
        "summary": summary,
        "producer": "integration-runner",
        "started_at": "2026-08-17T01:00:00Z",
        "finished_at": "2026-08-17T01:00:01Z",
        "artifacts": [
            {
                "role": "INPUT_MEDIA",
                "relative_path": input_path.relative_to(tmp_path).as_posix(),
                "sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
            },
            {
                "role": "PROVIDER_RESPONSE",
                "relative_path": response_path.relative_to(tmp_path).as_posix(),
                "sha256": hashlib.sha256(response_path.read_bytes()).hexdigest(),
            },
        ],
        "details": {
            "type": "LIVE_SERVICE",
            "trace": {
                "command": ["pytest", "-m", "integration"],
                "exit_code": 0,
                "stdout_sha256": "a" * 64,
                "stderr_sha256": "b" * 64,
            },
            "service": "QWEN",
            "model_id": "qwen3-vl-plus",
            "request_id_sha256": hashlib.sha256(
                b"chatcmpl_real_001"
            ).hexdigest(),
            "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
            "output_sha256": hashlib.sha256(response_path.read_bytes()).hexdigest(),
            "http_status": 200,
        },
    }
    evidence_path.write_text(json.dumps(payload), encoding="utf-8")
    reference = EvidenceReference(
        kind=EvidenceKind.LIVE_SERVICE_REPORT,
        level=EvidenceLevel.REAL_SERVICE,
        relative_path=evidence_path.relative_to(tmp_path).as_posix(),
        sha256=hashlib.sha256(evidence_path.read_bytes()).hexdigest(),
        covered_items=("qwen_live",),
        summary=summary,
    )
    response_path.write_text("changed", encoding="utf-8")

    with pytest.raises(ValueError, match="机器证据") as captured:
        build_final_gate_report(
            quality=build_quality_report({}, QUALITY_THRESHOLDS),
            checks=(
                GateCheck(
                    check_id="qwen_live",
                    status=GateStatus.PASS,
                    evidence=(reference,),
                ),
            ),
            workspace_root=tmp_path,
        )
    assert str(response_path) not in str(captured.value)
    assert captured.value.__cause__ is None


def test_final_gate_rejects_wrong_evidence_kind_for_live_check(tmp_path: Path) -> None:
    path = tmp_path / "wrong-kind.json"
    payload = {
        "schema_version": "1.0.0",
        "check_id": "qwen_live",
        "status": "PASS",
        "kind": "COMMAND_REPORT",
        "level": "REAL_SERVICE",
        "covered_items": ["qwen_live"],
        "summary": "真实服务调用通过",
        "producer": "integration-runner",
        "started_at": "2026-08-17T01:00:00Z",
        "finished_at": "2026-08-17T01:00:01Z",
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    check = GateCheck(
        check_id="qwen_live",
        status=GateStatus.PASS,
        evidence=(
            EvidenceReference(
                kind=EvidenceKind.COMMAND_REPORT,
                level=EvidenceLevel.REAL_SERVICE,
                relative_path=path.name,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                covered_items=("qwen_live",),
                summary="真实服务调用通过",
            ),
        ),
    )

    with pytest.raises(ValueError, match="证据类型"):
        build_final_gate_report(
            quality=build_quality_report({}, QUALITY_THRESHOLDS),
            checks=(check,),
            workspace_root=tmp_path,
        )


def test_final_gate_rejects_invalid_or_skipped_automated_test_junit(
    tmp_path: Path,
) -> None:
    invalid = tmp_path / "invalid.xml"
    invalid.write_text("相信我", encoding="utf-8")
    skipped = tmp_path / "skipped.xml"
    suite = ElementTree.Element("testsuite", name="pytest")
    case = ElementTree.SubElement(
        suite,
        "testcase",
        classname="tests.test_example",
        name="test_example",
    )
    ElementTree.SubElement(case, "skipped", message="skipped")
    skipped.write_bytes(ElementTree.tostring(suite, encoding="utf-8", xml_declaration=True))

    for path in (invalid, skipped):
        check = GateCheck(
            check_id="automated_tests",
            status=GateStatus.PASS,
            evidence=(
                EvidenceReference(
                    kind=EvidenceKind.PYTEST_JUNIT,
                    level=EvidenceLevel.CONTRACT,
                    relative_path=path.name,
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    covered_items=("automated_tests",),
                    summary="全量测试通过",
                ),
            ),
        )
        with pytest.raises(ValueError, match="JUnit"):
            build_final_gate_report(
                quality=build_quality_report({}, QUALITY_THRESHOLDS),
                checks=(check,),
                workspace_root=tmp_path,
            )


def test_duplicate_junit_node_skip_cannot_be_hidden_by_pass(tmp_path: Path) -> None:
    report_path = tmp_path / "duplicate.xml"
    _write_junit(report_path)
    root = ElementTree.fromstring(report_path.read_bytes())
    node_id = FAILURE_SCENARIO_TESTS["corrupted_media"][0]
    relative_path, test_name = node_id.split("::", maxsplit=1)
    case = ElementTree.SubElement(
        root,
        "testcase",
        classname=relative_path.removesuffix(".py").replace("/", "."),
        name=test_name,
    )
    ElementTree.SubElement(case, "skipped", message="skipped duplicate")
    report_path.write_bytes(
        ElementTree.tostring(root, encoding="utf-8", xml_declaration=True),
    )

    check = build_failure_matrix_check(report_path, workspace_root=tmp_path)

    assert check.status == GateStatus.NOT_RUN
    assert "corrupted_media" in (check.not_run_reason or "")


def test_automated_tests_require_exact_pytest_collection_manifest(tmp_path: Path) -> None:
    report_path = tmp_path / "full.xml"
    collection_path = tmp_path / "collection.txt"
    suite = ElementTree.Element("testsuite", name="pytest")
    ElementTree.SubElement(
        suite,
        "testcase",
        classname="tests.test_example",
        name="test_one",
    )
    report_path.write_bytes(
        ElementTree.tostring(suite, encoding="utf-8", xml_declaration=True),
    )
    collection_path.write_text(
        "tests/test_example.py::test_one\n"
        "tests/test_example.py::test_two\n"
        "\n2 tests collected in 0.01s\n",
        encoding="utf-8",
    )

    check = build_automated_tests_check(
        report_path,
        collection_path=collection_path,
        workspace_root=tmp_path,
    )

    assert check.status == GateStatus.NOT_RUN
    assert "test_two" in (check.not_run_reason or "")


def test_automated_tests_accept_only_fixed_acknowledged_xfail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    node_id = (
        "tests/evaluation/test_live_runner.py::"
        "test_not_run_verification_rejects_directory_aba_during_entire_verifier"
    )
    reason = (
        "已知延期：路径型 live verifier 存在目录 ABA 窗口；单进程 Demo 主流程不受影响，"
        "后续改为基于 writer fd 的同源验证"
    )
    report_path = tmp_path / "acknowledged-xfail.xml"
    collection_path = tmp_path / "collection.txt"
    suite = ElementTree.Element("testsuite", name="pytest")
    case = ElementTree.SubElement(
        suite,
        "testcase",
        classname="tests.evaluation.test_live_runner",
        name="test_not_run_verification_rejects_directory_aba_during_entire_verifier",
    )
    ElementTree.SubElement(
        case,
        "skipped",
        type="pytest.xfail",
        message=reason,
    )
    report_path.write_bytes(
        ElementTree.tostring(suite, encoding="utf-8", xml_declaration=True),
    )
    collection_path.write_text(f"{node_id}\n\n1 test collected in 0.01s\n", encoding="utf-8")
    monkeypatch.setattr(
        gate_module,
        "_collect_current_pytest_nodes",
        lambda _workspace_root: ({node_id}, None),
    )

    check = build_automated_tests_check(
        report_path,
        collection_path=collection_path,
        workspace_root=tmp_path,
    )

    assert check.status == GateStatus.PASS


@pytest.mark.parametrize(
    ("xfail_type", "xfail_reason"),
    (
        ("pytest.skip", "已知延期：路径型 live verifier 存在目录 ABA 窗口"),
        ("pytest.xfail", "不同的延期原因"),
    ),
)
def test_automated_tests_rejects_unacknowledged_or_changed_xfail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    xfail_type: str,
    xfail_reason: str,
) -> None:
    node_id = (
        "tests/evaluation/test_live_runner.py::"
        "test_not_run_verification_rejects_directory_aba_during_entire_verifier"
    )
    report_path = tmp_path / "changed-xfail.xml"
    collection_path = tmp_path / "collection.txt"
    suite = ElementTree.Element("testsuite", name="pytest")
    case = ElementTree.SubElement(
        suite,
        "testcase",
        classname="tests.evaluation.test_live_runner",
        name="test_not_run_verification_rejects_directory_aba_during_entire_verifier",
    )
    ElementTree.SubElement(
        case,
        "skipped",
        type=xfail_type,
        message=xfail_reason,
    )
    report_path.write_bytes(
        ElementTree.tostring(suite, encoding="utf-8", xml_declaration=True),
    )
    collection_path.write_text(f"{node_id}\n\n1 test collected in 0.01s\n", encoding="utf-8")
    monkeypatch.setattr(
        gate_module,
        "_collect_current_pytest_nodes",
        lambda _workspace_root: ({node_id}, None),
    )

    check = build_automated_tests_check(
        report_path,
        collection_path=collection_path,
        workspace_root=tmp_path,
    )

    assert check.status == GateStatus.NOT_RUN


def test_automated_tests_reject_matching_single_case_when_source_collects_more(
    tmp_path: Path,
) -> None:
    tests_root = tmp_path / "tests"
    tests_root.mkdir()
    (tests_root / "test_example.py").write_text(
        "def test_one():\n    assert True\n\n"
        "def test_two():\n    assert True\n",
        encoding="utf-8",
    )
    report_path = tmp_path / "single.xml"
    collection_path = tmp_path / "single-collection.txt"
    suite = ElementTree.Element("testsuite", name="pytest")
    ElementTree.SubElement(
        suite,
        "testcase",
        classname="tests.test_example",
        name="test_one",
    )
    report_path.write_bytes(
        ElementTree.tostring(suite, encoding="utf-8", xml_declaration=True),
    )
    collection_path.write_text(
        "tests/test_example.py::test_one\n\n1 test collected in 0.01s\n",
        encoding="utf-8",
    )

    check = build_automated_tests_check(
        report_path,
        collection_path=collection_path,
        workspace_root=tmp_path,
    )

    assert check.status == GateStatus.NOT_RUN
    assert "当前源码" in (check.not_run_reason or "")
    assert "test_two" in (check.not_run_reason or "")


def test_pytest_collection_failure_does_not_leak_environment_secret(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "qwen-live-secret-must-not-leak"
    monkeypatch.setenv("QWEN_API_KEY", secret)
    monkeypatch.setattr(
        gate_module.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=args,
            returncode=2,
            stdout=b"collection failed",
            stderr=secret.encode("utf-8"),
        ),
    )
    tests_root = tmp_path / "tests"
    tests_root.mkdir()
    (tests_root / "test_example.py").write_text(
        "def test_one():\n    assert True\n",
        encoding="utf-8",
    )
    report_path = tmp_path / "single.xml"
    collection_path = tmp_path / "single-collection.txt"
    suite = ElementTree.Element("testsuite", name="pytest")
    ElementTree.SubElement(
        suite,
        "testcase",
        classname="tests.test_example",
        name="test_one",
    )
    report_path.write_bytes(
        ElementTree.tostring(suite, encoding="utf-8", xml_declaration=True),
    )
    collection_path.write_text(
        "tests/test_example.py::test_one\n\n1 test collected in 0.01s\n",
        encoding="utf-8",
    )

    check = build_automated_tests_check(
        report_path,
        collection_path=collection_path,
        workspace_root=tmp_path,
    )

    reason = check.not_run_reason or ""
    assert check.status == GateStatus.NOT_RUN
    assert secret not in reason
    assert "PYTEST_COLLECTION_FAILED" in reason
    assert "collection failed" not in reason


def test_final_gate_reparses_failure_matrix_instead_of_trusting_status(
    tmp_path: Path,
) -> None:
    corrupted_case = FAILURE_SCENARIO_TESTS["corrupted_media"][0]
    report_path = tmp_path / "failed.xml"
    _write_junit(report_path, failed={corrupted_case})
    failed = build_failure_matrix_check(report_path, workspace_root=tmp_path)
    forged = GateCheck(
        check_id="failure_matrix",
        status=GateStatus.PASS,
        evidence=failed.evidence,
    )

    with pytest.raises(ValueError, match="JUnit"):
        build_final_gate_report(
            quality=build_quality_report({}, QUALITY_THRESHOLDS),
            checks=(forged,),
            workspace_root=tmp_path,
        )


def test_final_gate_rejects_changed_junit_reference_metadata(tmp_path: Path) -> None:
    report_path = tmp_path / "matrix.xml"
    _write_junit(report_path)
    check = build_failure_matrix_check(report_path, workspace_root=tmp_path)
    forged = check.model_copy(
        update={
            "evidence": (
                check.evidence[0].model_copy(update={"summary": "伪造摘要"}),
            ),
        },
    )

    with pytest.raises(ValueError, match="引用元数据"):
        build_final_gate_report(
            quality=build_quality_report({}, QUALITY_THRESHOLDS),
            checks=(forged,),
            workspace_root=tmp_path,
        )


def test_final_gate_rejects_changed_evidence_file(tmp_path: Path) -> None:
    evidence = _evidence_file(tmp_path)
    (tmp_path / evidence.relative_path).write_text("changed", encoding="utf-8")
    check = GateCheck(
        check_id="failure_matrix",
        status=GateStatus.FAIL,
        evidence=(evidence,),
    )

    with pytest.raises(ValueError, match="摘要"):
        build_final_gate_report(
            quality=build_quality_report({}, QUALITY_THRESHOLDS),
            checks=(check,),
            workspace_root=tmp_path,
        )
