from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from fastapi import FastAPI
from sqlalchemy import select

from video_demo.api.app import create_app
from video_demo.application.queries import ResultWriteFence
from video_demo.config import Settings
from video_demo.domain.evidence import KeyframeEvidence, SpeechSegment
from video_demo.domain.result import (
    SummaryChapter,
    VideoSegment,
    VideoSummary,
    VideoUnderstandingResult,
)
from video_demo.domain.run import RunStatus
from video_demo.evaluation.annotations import (
    AuthorizationFile,
    AuthorizationRecord,
    EvaluationAnnotation,
    ReferenceAudioEvent,
    ReferenceOcrFrame,
    ReferenceSpeakerTurn,
    ReferenceWord,
    SupportedFact,
    ValidatedEvaluationPackage,
    load_evaluation_package,
)
from video_demo.evaluation.dataset import EvaluationSample
from video_demo.evaluation.final_runner import cleanup_evaluation_run
from video_demo.evaluation.prediction_runner import PredictionRunner
from video_demo.persistence.models import VideoAssetModel
from video_demo.persistence.repositories import JobRepository, Scope, VideoRunRepository


def _write_runner_package(
    tmp_path: Path,
    media: bytes,
) -> tuple[Path, ValidatedEvaluationPackage]:
    import shutil

    for relative in (
        Path("src/video_demo/evaluation/prediction_runner.py"),
        Path("src/video_demo/evaluation/predictions.py"),
        Path("src/video_demo/evaluation/quality_runner.py"),
        Path("src/video_demo/application/composition.py"),
        Path("src/video_demo/application/pipeline.py"),
        Path("src/video_demo/application/queries.py"),
        Path("src/video_demo/api/app.py"),
        Path("src/video_demo/api/objects.py"),
        Path("src/video_demo/api/runs.py"),
        Path("src/video_demo/api/jobs.py"),
    ):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path.cwd() / relative, destination)
    runtime_root = tmp_path / ".codex" / "video-rag-demo"
    eval_root = runtime_root / "eval"
    media_path = eval_root / "media" / "sample.mp4"
    media_path.parent.mkdir(parents=True, exist_ok=True)
    media_path.write_bytes(media)
    media_sha256 = hashlib.sha256(media).hexdigest()
    annotation = EvaluationAnnotation(
        schema_version="1.0.0",
        sample_id="sample_001",
        media_sha256=media_sha256,
        duration_ms=500,
        language="en",
        reference_text="Hello",
        words=(ReferenceWord(word_id="word_001", text="Hello", start_ms=0, end_ms=400),),
        speaker_turns=(
            ReferenceSpeakerTurn(
                turn_id="turn_001",
                speaker_id="speaker_001",
                start_ms=0,
                end_ms=400,
            ),
        ),
        ocr_frames=(
            ReferenceOcrFrame(frame_id="ocr_001", timestamp_ms=100, text_lines=("Hello",)),
        ),
        audio_events=(
            ReferenceAudioEvent(
                event_id="audio_001",
                normalized_event="speech",
                start_ms=0,
                end_ms=400,
            ),
        ),
        scene_boundaries_ms=(100,),
        semantic_boundaries_ms=(200,),
        supported_facts=(SupportedFact(fact_id="fact_001", canonical_text="Hello"),),
        key_fact_ids=("fact_001",),
    )
    annotation_path = eval_root / "annotations" / "sample.json"
    annotation_path.parent.mkdir(parents=True, exist_ok=True)
    annotation_bytes = annotation.model_dump_json(exclude_computed_fields=True).encode("utf-8")
    annotation_path.write_bytes(annotation_bytes)
    sample = EvaluationSample(
        sample_id="sample_001",
        language="en",
        authorization_id="auth_001",
        media_relative_path="media/sample.mp4",
        media_sha256=media_sha256,
        annotations_relative_path="annotations/sample.json",
        annotations_sha256=hashlib.sha256(annotation_bytes).hexdigest(),
    )
    dataset_path = eval_root / "dataset.jsonl"
    dataset_path.write_text(sample.model_dump_json() + "\n", encoding="utf-8")
    authorization = AuthorizationFile(
        schema_version="1.0.0",
        records=(
            AuthorizationRecord(
                schema_version="1.0.0",
                authorization_id="auth_001",
                source_category="OWNED",
                allowed_purposes=("VIDEO_QUALITY_EVALUATION",),
                confirmed_at="2026-08-20T00:00:00Z",
                media_sha256=(media_sha256,),
            ),
        ),
    )
    authorization_path = eval_root / "authorization.json"
    authorization_path.write_text(authorization.model_dump_json(), encoding="utf-8")
    return runtime_root, load_evaluation_package(
        dataset_path,
        authorization_path,
        workspace_root=tmp_path,
        runtime_root=runtime_root,
    )


def _fake_worker_factory(app: FastAPI, calls: list[str]):
    class FakeWorker:
        def run_once(self) -> bool:
            calls.append("run_once")
            container = app.state.container
            scope = Scope("evaluation", "video-demo", "evaluation")
            with container.database.session() as session:
                claimed = JobRepository(session).claim("evaluation-worker", lease_seconds=60)
                assert claimed is not None
                run = VideoRunRepository(session).get(scope, claimed.resource_id)
                assert run is not None
                media_sha256 = session.execute(
                    select(VideoAssetModel.source_sha256).where(
                        VideoAssetModel.asset_id == run.asset_id,
                    )
                ).scalar_one()
            keyframe_bytes = b"\xff\xd8\xffrunner-keyframe"
            keyframe_relative = (
                Path("runs")
                / container.result_query_service.scope_key(scope)
                / claimed.resource_id
                / "keyframes/frame.jpg"
            )
            keyframe_path = container.settings.runtime_root / keyframe_relative
            keyframe_path.parent.mkdir(parents=True, exist_ok=True)
            keyframe_path.write_bytes(keyframe_bytes)
            evidence = (
                *tuple(
                    SpeechSegment(
                    evidence_id=f"asr_{index:03d}",
                    start_ms=0,
                    end_ms=400,
                    text="Hello",
                    language="en",
                    confidence=0.9,
                    is_fully_evaluated_language=True,
                    )
                    for index in range(1, 101)
                ),
                KeyframeEvidence(
                    evidence_id="keyframe_ev_001",
                    start_ms=0,
                    end_ms=400,
                    keyframe_id="keyframe_001",
                    timestamp_ms=200,
                    relative_path=keyframe_relative.as_posix(),
                    mime_type="image/jpeg",
                    sha256=hashlib.sha256(keyframe_bytes).hexdigest(),
                    perceptual_hash="abcdef12",
                ),
            )
            segment_text = "类型：VIDEO_SEGMENT"
            summary_text = "类型：VIDEO_SUMMARY"
            segment = VideoSegment(
                segment_id="segment_001",
                start_ms=0,
                end_ms=400,
                title="问候",
                summary_zh="讲者问好",
                languages=("en",),
                evidence_refs=("asr_001", "keyframe_ev_001"),
                retrieval_text=segment_text,
                retrieval_hash=hashlib.sha256(segment_text.encode()).hexdigest(),
            )
            result = VideoUnderstandingResult(
                run_id=claimed.resource_id,
                asset_sha256=media_sha256,
                segments=(segment,),
                summary=VideoSummary(
                    title="测试视频",
                    summary_zh="视频包含问候",
                    duration_ms=500,
                    chapters=(
                        SummaryChapter(
                            title="问候",
                            start_ms=0,
                            end_ms=400,
                            segment_ids=(segment.segment_id,),
                        ),
                    ),
                    languages=("en",),
                    retrieval_text=summary_text,
                    retrieval_hash=hashlib.sha256(summary_text.encode()).hexdigest(),
                ),
            )
            container.result_query_service.persist(
                scope,
                result,
                evidence=evidence,
                stage_metrics={"RESULT": 1},
                status=RunStatus.SUCCEEDED,
                transcript_source="ASR",
                fence=ResultWriteFence(
                    job_pk=claimed.id,
                    worker_id=claimed.worker_id,
                    attempt_count=claimed.attempt_count,
                ),
            )
            return True

        def close(self) -> None:
            calls.append("close")

    return FakeWorker


def test_prediction_runner_drives_api_worker_queries_and_model_free_score(
    tmp_path: Path,
) -> None:
    media = b"\x00\x00\x00\x18ftypisom" + b"m" * 128
    runtime_root, package = _write_runner_package(tmp_path, media)
    settings = Settings(
        workspace_root=tmp_path,
        runtime_root=runtime_root,
        max_video_bytes=1024 * 1024,
    )
    app = create_app(settings)
    calls: list[str] = []
    fake_worker_type = _fake_worker_factory(app, calls)
    runner = PredictionRunner(
        settings,
        app_factory=lambda _settings: app,
        worker_factory=lambda _settings, worker_id: fake_worker_type(),
        preflight=lambda: None,
    )
    # 组合测试只替换 Worker 内部生产端口，HTTP 上传/创建/查询仍走真实应用。
    report = runner.predict(package, evaluation_run_id="eval_001")

    assert report.status.value == "PASS", report.predictions
    assert [item.sample_id for item in report.predictions] == ["sample_001"]
    assert calls == ["run_once", "close"]
    prediction_root = runtime_root / "eval" / "predictions" / "eval_001" / "sample_001"
    assert {
        path.name for path in prediction_root.iterdir()
    } >= {"run.json", "result.json", "evidence.jsonl", "artifact-manifest.json", "index.json"}
    quality = __import__(
        "video_demo.evaluation.prediction_runner",
        fromlist=["score_prediction_run"],
    ).score_prediction_run("eval_001", eval_root=runtime_root / "eval")
    assert quality.evaluation_run_id == "eval_001"
    assert calls == ["run_once", "close"]
    scope = Scope("evaluation", "video-demo", "evaluation")
    scope_key = app.state.container.result_query_service.scope_key(scope)
    product_run = runtime_root / "runs" / scope_key / report.predictions[0].run_id
    foreign_run = runtime_root / "runs" / scope_key / "run_foreign"
    foreign_run.mkdir(parents=True)
    (foreign_run / "keep.txt").write_text("foreign", encoding="utf-8")
    assert product_run.is_dir()

    cleanup_evaluation_run(tmp_path, "eval_001")

    assert not product_run.exists()
    assert foreign_run.is_dir()
    assert (runtime_root / "objects").is_dir()


def test_cleanup_keeps_unbound_placeholder_product_run(tmp_path: Path) -> None:
    media = b"\x00\x00\x00\x18ftypisom" + b"m" * 128
    runtime_root, package = _write_runner_package(tmp_path, media)
    settings = Settings(
        workspace_root=tmp_path,
        runtime_root=runtime_root,
        max_video_bytes=1024 * 1024,
    )

    def fail_before_product_run(_settings: Settings) -> FastAPI:
        raise ValueError("产品 API 尚未创建 run")

    report = PredictionRunner(
        settings,
        app_factory=fail_before_product_run,
        preflight=lambda: None,
    ).predict(package, evaluation_run_id="eval_placeholder")
    placeholder = report.predictions[0]
    assert placeholder.run_id == "run_failed_sample_001"
    placeholder_run = (
        runtime_root
        / "runs"
        / hashlib.sha256(b"evaluation\x00video-demo\x00evaluation").hexdigest()[:24]
        / placeholder.run_id
    )
    placeholder_run.mkdir(parents=True)
    (placeholder_run / "keep.txt").write_text("not-owned", encoding="utf-8")

    cleanup_evaluation_run(tmp_path, "eval_placeholder")

    assert placeholder_run.is_dir()


def test_cleanup_rejects_tampered_prediction_before_deleting_anything(
    tmp_path: Path,
) -> None:
    media = b"\x00\x00\x00\x18ftypisom" + b"m" * 128
    runtime_root, package = _write_runner_package(tmp_path, media)
    settings = Settings(
        workspace_root=tmp_path,
        runtime_root=runtime_root,
        max_video_bytes=1024 * 1024,
    )
    app = create_app(settings)
    fake_worker_type = _fake_worker_factory(app, [])
    report = PredictionRunner(
        settings,
        app_factory=lambda _settings: app,
        worker_factory=lambda _settings, worker_id: fake_worker_type(),
        preflight=lambda: None,
    ).predict(package, evaluation_run_id="eval_tampered")
    prediction_report = runtime_root / "eval/reports/eval_tampered/prediction.json"
    prediction_report.write_bytes(prediction_report.read_bytes() + b" ")
    scope = Scope("evaluation", "video-demo", "evaluation")
    product_run = (
        runtime_root
        / "runs"
        / app.state.container.result_query_service.scope_key(scope)
        / report.predictions[0].run_id
    )

    with pytest.raises(ValueError, match="预测清理证据非法或损坏"):
        cleanup_evaluation_run(tmp_path, "eval_tampered")

    assert product_run.is_dir()
    assert (runtime_root / "eval/reports/eval_tampered").is_dir()


def test_prediction_runner_uses_real_production_worker_factory_for_failure_path(
    tmp_path: Path,
) -> None:
    media = b"\x00\x00\x00\x18ftypisom" + b"m" * 128
    runtime_root, package = _write_runner_package(tmp_path, media)
    settings = Settings(
        workspace_root=tmp_path,
        runtime_root=runtime_root,
        ffprobe_path=tmp_path / "missing-ffprobe",
        ffmpeg_path=tmp_path / "missing-ffmpeg",
        max_video_bytes=1024 * 1024,
    )
    runner = PredictionRunner(settings, preflight=lambda: None)

    report = runner.predict(package, evaluation_run_id="eval_real_worker_001")

    assert report.status.value == "FAIL"
    assert report.predictions[0].failure_code == "VIDEO_FFPROBE_UNAVAILABLE"
    prediction_root = (
        runtime_root
        / "eval"
        / "predictions"
        / "eval_real_worker_001"
        / "sample_001"
    )
    assert (prediction_root / "run.json").is_file()
    assert (prediction_root / "index.json").is_file()

def test_quality_score_entrypoint_is_model_free_and_requires_prediction_report(
    tmp_path: Path,
) -> None:
    from video_demo.errors import VideoDemoError
    from video_demo.evaluation.prediction_runner import score_prediction_run

    eval_root = tmp_path / ".codex" / "video-rag-demo" / "eval"
    eval_root.mkdir(parents=True)

    with pytest.raises(VideoDemoError):
        score_prediction_run("eval_001", eval_root=eval_root)
