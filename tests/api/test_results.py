from __future__ import annotations

import hashlib
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from video_demo.api.app import create_app
from video_demo.application.document_publication import ResultWriteFence
from video_demo.application.document_rendering import render_markdown
from video_demo.config import ApiRuntimeConfig
from video_demo.domain.document import (
    DocumentGenerationConfig,
    DocumentGenerationMetadata,
    GroundedClaim,
    ParagraphBlock,
    PromptVersions,
    SemanticChapter,
    VideoDocumentSummary,
    VideoUnderstandingResult,
    VisualBlock,
)
from video_demo.domain.document_artifact import MODEL_METRIC_NAMES, RESULT_STAGE_NAMES
from video_demo.domain.evidence import (
    KeyframeEvidence,
    SpeechSegment,
    VisualObservationEvidence,
    VisualTextContent,
)
from video_demo.persistence.models import VideoSegmentModel, VideoSummaryModel
from video_demo.persistence.repositories import JobRepository, Scope


@pytest.fixture
def client(tmp_path: Path, cloud_asr_environment: None) -> Iterator[TestClient]:
    """结果接口测试使用无视频调度器容器，避免手工发布与后台消费竞争。"""

    runtime_root = tmp_path / ".codex" / "video-rag-demo"
    settings = ApiRuntimeConfig(
        workspace_root=tmp_path,
        runtime_root=runtime_root,
        max_video_bytes=1024 * 1024,
        max_result_bundle_bytes=64 * 1024 * 1024,
        max_document_bytes=16 * 1024 * 1024,
        max_result_evidence_items=25_000,
        vlm_max_image_bytes=8 * 1024 * 1024,
        max_audio_bytes=4 * 1024 * 1024 * 1024,
        max_audio_duration_ms=7_200_000,
        max_image_bytes=8 * 1024 * 1024,
    )
    with TestClient(create_app(settings)) as test_client:
        yield test_client


def _sha256(content: bytes | str) -> str:
    encoded = content.encode("utf-8") if isinstance(content, str) else content
    return hashlib.sha256(encoded).hexdigest()


def _upload_and_create(
    client: TestClient,
    headers: dict[str, str],
    content: bytes,
) -> tuple[str, str]:
    uploaded = client.post(
        "/api/kb/knowledge-bases/kb-a/video-objects",
        headers=headers,
        files={"file": ("lesson.mp4", content, "video/mp4")},
    )
    assert uploaded.status_code == 201
    created = client.post(
        "/api/kb/knowledge-bases/kb-a/video-understanding-runs",
        headers=headers,
        json={
            "object_ref": uploaded.json()["object_ref"],
            "idempotency_key": "result-query-0001",
            "language_hints": ["zh"],
        },
    )
    assert created.status_code == 202
    return str(created.json()["run_id"]), str(uploaded.json()["sha256"])


def _publish_result(
    client: TestClient,
    run_id: str,
    asset_sha256: str,
) -> tuple[VideoUnderstandingResult, bytes]:
    container = client.app.state.container
    scope = Scope("tenant-a", "app-a", "kb-a")
    image = b"\xff\xd8\xffapi-keyframe\xff\xd9"
    image_sha = _sha256(image)
    image_relative = Path("visual/keyframes") / f"{image_sha}.jpg"
    image_path = (
        container.runtime_root
        / "runs"
        / container.result_query_service.scope_key(scope)
        / run_id
        / image_relative
    )
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(image)
    image_path.chmod(0o600)

    speech = SpeechSegment(
        evidence_id="asr_001",
        start_ms=0,
        end_ms=1_000,
        text="faster-whisper 用于语音识别。",
        language="zh",
        confidence=0.99,
        is_fully_evaluated_language=True,
    )
    keyframe = KeyframeEvidence(
        evidence_id="keyframe_evidence_001",
        start_ms=500,
        end_ms=501,
        keyframe_id="keyframe_001",
        timestamp_ms=500,
        relative_path=image_relative.as_posix(),
        mime_type="image/jpeg",
        sha256=image_sha,
        perceptual_hash="0123456789abcdef",
        size_bytes=len(image),
    )
    observation = VisualObservationEvidence(
        evidence_id="visual_001",
        chapter_id="chapter_001",
        start_ms=450,
        end_ms=550,
        target_ids=("target_001",),
        keyframe_refs=(keyframe.evidence_id,),
        transcript_evidence_refs=(speech.evidence_id,),
        visual_type="TEXT",
        caption="画面展示了模型名称。",
        content_blocks=(
            VisualTextContent(
                visual_content_id="visual_content_001",
                source_keyframe_refs=(keyframe.evidence_id,),
                text="faster-whisper",
            ),
        ),
        relation_to_transcript="SUPPORTING",
        certainty=0.98,
    )
    chapter = SemanticChapter(
        chapter_id="chapter_001",
        start_ms=0,
        end_ms=1_000,
        title="模型概览",
        title_evidence_refs=(speech.evidence_id,),
        summary_zh="介绍 faster-whisper。",
        summary_evidence_refs=(speech.evidence_id,),
        body_blocks=(
            ParagraphBlock(text="它用于语音识别。", evidence_refs=(speech.evidence_id,)),
            VisualBlock(
                visual_observation_ref=observation.evidence_id,
                visual_content_refs=("visual_content_001",),
                caption=observation.caption,
                evidence_refs=(observation.evidence_id,),
            ),
        ),
        claims=(
            GroundedClaim(
                text="faster-whisper 用于语音识别。",
                evidence_refs=(speech.evidence_id,),
                certainty=0.99,
            ),
        ),
        evidence_refs=(speech.evidence_id, keyframe.evidence_id, observation.evidence_id),
        selected_keyframe_refs=(keyframe.evidence_id,),
        transcript_source="ASR",
    )
    result = VideoUnderstandingResult(
        run_id=run_id,
        asset_sha256=asset_sha256,
        summary=VideoDocumentSummary(
            title="faster-whisper 教程",
            duration_ms=1_000,
            overview_zh="介绍模型用途。",
        ),
        chapters=(chapter,),
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
    evidence = (speech, keyframe, observation)
    with container.database.session() as session:
        claimed = JobRepository(session).claim("api-result-publisher", lease_seconds=60)
    assert claimed is not None
    container.result_query_service.persist(
        scope,
        result,
        evidence=evidence,
        document=render_markdown(result, evidence),
        stage_metrics=dict.fromkeys(RESULT_STAGE_NAMES, 0),
        model_metrics=dict.fromkeys(MODEL_METRIC_NAMES, 0),
        status="SUCCEEDED",
        transcript_source="ASR",
        published_keyframes=(keyframe,),
        fence=ResultWriteFence(claimed.id, claimed.worker_id, claimed.attempt_count),
    )
    return result, image


def test_result_document_evidence_and_keyframe_are_one_4_contract(
    client: TestClient,
    scope_headers: dict[str, str],
    mp4_content: bytes,
) -> None:
    run_id, asset_sha = _upload_and_create(client, scope_headers, mp4_content)
    result, image = _publish_result(client, run_id, asset_sha)
    root = f"/api/kb/knowledge-bases/kb-a/video-understanding-runs/{run_id}"

    result_response = client.get(f"{root}/result", headers=scope_headers)
    document_response = client.get(f"{root}/document", headers=scope_headers)
    evidence_response = client.get(f"{root}/evidence", headers=scope_headers)
    image_response = client.get(
        f"{root}/keyframes/keyframe_001/content", headers=scope_headers
    )

    assert result_response.status_code == 200
    assert result_response.json()["schema_version"] == "4.1.0"
    assert result_response.json()["chapters"][0]["chapter_id"] == "chapter_001"
    evidence = client.app.state.container.result_query_service.get_evidence(
        Scope("tenant-a", "app-a", "kb-a"), run_id
    ).items
    assert document_response.content == render_markdown(result, evidence).content
    assert document_response.headers["content-type"] == "text/markdown; charset=utf-8"
    assert document_response.headers["content-disposition"] == (
        'inline; filename="knowledge-note.md"'
    )
    items = evidence_response.json()["items"]
    assert {item["evidence_type"] for item in items} == {
        "ASR_SEGMENT",
        "KEYFRAME",
        "VISUAL_OBSERVATION",
    }
    assert "relative_path" not in next(
        item for item in items if item["evidence_type"] == "KEYFRAME"
    )
    assert image_response.status_code == 200
    assert image_response.headers["content-type"] == "image/jpeg"
    assert image_response.content == image


def test_result_routes_are_scope_isolated(
    client: TestClient,
    scope_headers: dict[str, str],
    mp4_content: bytes,
) -> None:
    run_id, asset_sha = _upload_and_create(client, scope_headers, mp4_content)
    _publish_result(client, run_id, asset_sha)
    foreign = {"X-Tenant-Id": "tenant-other", "X-Application-Id": "app-a"}

    response = client.get(
        f"/api/kb/knowledge-bases/kb-a/video-understanding-runs/{run_id}/result",
        headers=foreign,
    )

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "VIDEO_RUN_NOT_FOUND"


def test_old_successful_run_returns_409_for_all_result_routes(
    client: TestClient,
    scope_headers: dict[str, str],
    mp4_content: bytes,
) -> None:
    run_id, asset_sha = _upload_and_create(client, scope_headers, mp4_content)
    _publish_result(client, run_id, asset_sha)
    with client.app.state.container.database.session() as session:
        session.query(VideoSummaryModel).one().schema_version = "2.0.0"
        session.query(VideoSegmentModel).one().schema_version = "2.0.0"
    root = f"/api/kb/knowledge-bases/kb-a/video-understanding-runs/{run_id}"

    for suffix in (
        "/result",
        "/document",
        "/evidence",
        "/keyframes/keyframe_001/content",
    ):
        response = client.get(f"{root}{suffix}", headers=scope_headers)
        assert response.status_code == 409
        assert response.json()["error"]["code"] == "RESULT_SCHEMA_UNSUPPORTED"


def test_openapi_exposes_only_document_evidence_contract(client: TestClient) -> None:
    openapi = client.get("/openapi.json").text

    assert "PublicVisualObservationEvidence" in openapi
    assert "PublicOcrEvidence" not in openapi
    assert "SceneBoundary" not in openapi
    assert "relative_path" not in openapi
