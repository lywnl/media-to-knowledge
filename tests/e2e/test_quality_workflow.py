from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from fastapi import FastAPI

from video_demo.api.app import create_app
from video_demo.application.document_publication import ResultWriteFence
from video_demo.application.document_rendering import render_markdown
from video_demo.config import Settings
from video_demo.domain.document import (
    DocumentGenerationConfig,
    DocumentGenerationMetadata,
    GroundedClaim,
    ParagraphBlock,
    PromptVersions,
    SemanticChapter,
    VideoDocumentSummary,
    VideoUnderstandingResult,
)
from video_demo.domain.document_artifact import MODEL_METRIC_NAMES, RESULT_STAGE_NAMES
from video_demo.domain.evidence import SpeechSegment
from video_demo.evaluation.annotations import (
    AuthorizationFile,
    AuthorizationRecord,
    EvaluationAnnotation,
    ReferenceVisualFrame,
    SupportedFact,
    ValidatedEvaluationPackage,
    load_evaluation_package,
)
from video_demo.evaluation.dataset import EvaluationSample
from video_demo.evaluation.final_runner import cleanup_evaluation_run
from video_demo.evaluation.prediction_runner import PredictionRunner, score_prediction_run
from video_demo.implementation import prediction_implementation_files
from video_demo.persistence.models import VideoAssetModel
from video_demo.persistence.repositories import JobRepository, Scope, VideoRunRepository


def _sha(content: bytes | str) -> str:
    encoded = content.encode("utf-8") if isinstance(content, str) else content
    return hashlib.sha256(encoded).hexdigest()


def _write_package(
    tmp_path: Path,
    media: bytes,
) -> tuple[Path, ValidatedEvaluationPackage]:
    for relative in prediction_implementation_files(Path.cwd()):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path.cwd() / relative, destination)
    runtime_root = tmp_path / ".codex" / "video-rag-demo"
    eval_root = runtime_root / "eval"
    media_path = eval_root / "media/sample.mp4"
    media_path.parent.mkdir(parents=True, exist_ok=True)
    media_path.write_bytes(media)
    media_sha = _sha(media)
    annotation = EvaluationAnnotation(
        schema_version="2.0.0",
        sample_id="sample_001",
        media_sha256=media_sha,
        duration_ms=500,
        language="zh",
        reference_text="你好",
        visual_frames=(
            ReferenceVisualFrame(
                frame_id="frame_001", timestamp_ms=100, text_lines=("你好",)
            ),
        ),
        semantic_boundaries_ms=(250,),
        supported_facts=(SupportedFact(fact_id="fact_001", canonical_text="你好"),),
        key_fact_ids=("fact_001",),
    )
    annotation_path = eval_root / "annotations/sample.json"
    annotation_path.parent.mkdir(parents=True, exist_ok=True)
    annotation_bytes = annotation.model_dump_json(exclude_computed_fields=True).encode("utf-8")
    annotation_path.write_bytes(annotation_bytes)
    sample = EvaluationSample(
        sample_id="sample_001",
        language="zh",
        authorization_id="auth_001",
        media_relative_path="media/sample.mp4",
        media_sha256=media_sha,
        annotations_relative_path="annotations/sample.json",
        annotations_sha256=_sha(annotation_bytes),
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
                media_sha256=(media_sha,),
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


def _worker_factory(app: FastAPI, calls: list[str]):
    class Worker:
        def run_once(self) -> bool:
            calls.append("run_once")
            container = app.state.container
            scope = Scope("evaluation", "video-demo", "evaluation")
            with container.database.session() as session:
                claimed = JobRepository(session).claim("evaluation-worker", lease_seconds=60)
                assert claimed is not None
                run = VideoRunRepository(session).get(scope, claimed.resource_id)
                assert run is not None
                media_sha = session.query(VideoAssetModel.source_sha256).filter(
                    VideoAssetModel.asset_id == run.asset_id
                ).scalar()
            assert isinstance(media_sha, str)
            speech = SpeechSegment(
                evidence_id="asr_001",
                start_ms=0,
                end_ms=250,
                text="你好",
                language="zh",
                confidence=0.99,
                is_fully_evaluated_language=True,
            )
            first = SemanticChapter(
                chapter_id="chapter_001",
                start_ms=0,
                end_ms=250,
                title="问候",
                title_evidence_refs=(speech.evidence_id,),
                summary_zh="讲者问好。",
                summary_evidence_refs=(speech.evidence_id,),
                body_blocks=(
                    ParagraphBlock(text="讲者问好。", evidence_refs=(speech.evidence_id,)),
                ),
                claims=(
                    GroundedClaim(
                        text="讲者进行了问候。",
                        evidence_refs=(speech.evidence_id,),
                        certainty=0.99,
                    ),
                ),
                evidence_refs=(speech.evidence_id,),
                transcript_source="ASR",
            )
            second = SemanticChapter(
                chapter_id="chapter_002",
                start_ms=250,
                end_ms=500,
                title="本时段未提取到可验证语义内容",
                title_evidence_refs=(),
                summary_zh="本时段未提取到可验证语义内容",
                summary_evidence_refs=(),
                body_blocks=(),
                claims=(),
                content_status="NO_SEMANTIC_EVIDENCE",
                evidence_refs=(),
                transcript_source="NONE",
            )
            chapters = (first, second)
            result = VideoUnderstandingResult(
                run_id=claimed.resource_id,
                asset_sha256=media_sha,
                summary=VideoDocumentSummary(
                    title="测试视频",
                    duration_ms=500,
                    overview_zh="视频包含问候。",
                ),
                chapters=chapters,
                generation=DocumentGenerationMetadata(
                    document_config=DocumentGenerationConfig(),
                    text_model_id="text-model",
                    vlm_model_id="qwen3-vl-flash",
                    prompt_versions=PromptVersions(
                        chapter_planner="chapter-planner-v1",
                        chapter_planner_repair="chapter-planner-repair-v1",
                        chapter_vlm="chapter-vlm-v1",
                        chapter_vlm_repair="chapter-vlm-repair-v1",
                        chapter_writer="chapter-writer-v1",
                        chapter_writer_repair="chapter-writer-repair-v1",
                        global_editor="global-editor-v1",
                        global_editor_repair="global-editor-repair-v1",
                    ),
                ),
            )
            container.result_query_service.persist(
                scope,
                result,
                evidence=(speech,),
                document=render_markdown(result, (speech,)),
                stage_metrics=dict.fromkeys(RESULT_STAGE_NAMES, 0),
                model_metrics=dict.fromkeys(MODEL_METRIC_NAMES, 0),
                status="SUCCEEDED",
                transcript_source="ASR",
                fence=ResultWriteFence(
                    claimed.id, claimed.worker_id, claimed.attempt_count
                ),
            )
            return True

        def close(self) -> None:
            calls.append("close")

    return Worker


def test_prediction_quality_and_cleanup_workflow_uses_3_artifact_set(
    tmp_path: Path,
    cloud_asr_environment: None,
    monkeypatch,
) -> None:
    # 评测工作流显式驱动受控 worker；关闭 API lifespan 内的生产调度器，
    # 避免它先于测试 worker 抢占同一条任务。
    import video_demo.api.app as app_module

    monkeypatch.setattr(app_module, "build_video_scheduler", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("VIDEO_DEMO_TEXT_LLM_BASE_URL", "https://text.example.test/v1")
    monkeypatch.setenv("VIDEO_DEMO_TEXT_LLM_API_KEY", "text-key")
    monkeypatch.setenv("VIDEO_DEMO_TEXT_LLM_MODEL_ID", "text-model")
    monkeypatch.setenv("VIDEO_DEMO_VLM_BASE_URL", "https://vlm.example.test/v1")
    monkeypatch.setenv("VIDEO_DEMO_VLM_API_KEY", "vlm-key")
    runtime_root, package = _write_package(
        tmp_path, b"\x00\x00\x00\x18ftypisom" + b"m" * 128
    )
    settings = Settings(
        workspace_root=tmp_path,
        runtime_root=runtime_root,
        max_video_bytes=1024 * 1024,
    )
    app = create_app(settings)
    calls: list[str] = []
    worker = _worker_factory(app, calls)
    report = PredictionRunner(
        settings,
        app_factory=lambda _settings: app,
        worker_factory=lambda _settings, worker_id: worker(),
        preflight=lambda: None,
    ).predict(package, evaluation_run_id="eval_001")

    assert report.status.value == "PASS", [
        (item.failure_code, item.terminal_status, item.run_id)
        for item in report.predictions
    ]
    assert calls == ["run_once", "close"]
    prediction_root = runtime_root / "eval/predictions/eval_001/sample_001"
    assert {
        "run.json",
        "result.json",
        "evidence.jsonl",
        "document.md",
        "artifact-manifest.json",
        "index.json",
    }.issubset(path.name for path in prediction_root.iterdir())
    index = json.loads((prediction_root / "index.json").read_text(encoding="utf-8"))
    assert index["schema_version"] == "1.2.0"
    assert index["document_relative_path"].endswith("/document.md")
    quality = score_prediction_run("eval_001", eval_root=runtime_root / "eval")
    assert quality.evaluation_run_id == "eval_001"
    product_run = (
        runtime_root
        / "runs"
        / app.state.container.result_query_service.scope_key(
            Scope("evaluation", "video-demo", "evaluation")
        )
        / report.predictions[0].run_id
    )
    assert product_run.is_dir()

    cleanup_evaluation_run(tmp_path, "eval_001")

    assert not product_run.exists()
