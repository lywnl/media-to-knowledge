from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from video_demo.application.composition import ProductionPipeline
from video_demo.application.pipeline import (
    PipelineContext,
    PipelineJobHandler,
    PipelineRunConfig,
    PreparedMedia,
    ProbedAsset,
    RegisteredAsset,
    SpeechAnalysis,
    VisualAnalysis,
    VisualPreparation,
)
from video_demo.application.queries import ResultQueryService, ResultWriteFence
from video_demo.domain.evidence import KeyframeEvidence, SceneBoundary, SpeechSegment
from video_demo.domain.manifest import Rational, VideoAssetManifest, VideoStream
from video_demo.domain.result import (
    SegmentUnderstanding,
    SummaryChapter,
    SummaryUnderstanding,
    VideoSegment,
    VideoSummary,
    VideoUnderstandingResult,
)
from video_demo.domain.run import RunStatus, TimeRange
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.fusion.merge import BoundaryPoint
from video_demo.integrations.video_port import (
    WholeVideoUnderstanding,
    WholeVideoUnderstandingRequest,
    WholeVideoWindowUnderstanding,
)
from video_demo.media.probe import ProbeLimits
from video_demo.persistence.repositories import (
    JobRepository,
    Scope,
    VideoRunRepository,
)


def _upload_and_create(
    client: TestClient,
    headers: dict[str, str],
    content: bytes,
) -> tuple[dict[str, object], str]:
    uploaded = client.post(
        "/api/kb/knowledge-bases/kb-a/video-objects",
        headers=headers,
        files={"file": ("lesson.mp4", content, "video/mp4")},
    ).json()
    created = client.post(
        "/api/kb/knowledge-bases/kb-a/video-understanding-runs",
        headers=headers,
        json={
            "object_ref": uploaded["object_ref"],
            "idempotency_key": "result-query-0001",
            "language_hints": ["en"],
        },
    ).json()
    return created, str(uploaded["sha256"])


def _persist_ready_result(
    client: TestClient,
    run_id: str,
    asset_sha256: str,
) -> tuple[str, bytes]:
    scope = Scope("tenant-a", "app-a", "kb-a")
    container = client.app.state.container
    keyframe_bytes = b"\xff\xd8\xffjpeg-content"
    relative_path = (
        Path("runs")
        / container.result_query_service.scope_key(scope)
        / run_id
        / "keyframes"
        / "frame.jpg"
    )
    keyframe_path = container.settings.runtime_root / relative_path
    keyframe_path.parent.mkdir(parents=True)
    keyframe_path.write_bytes(keyframe_bytes)
    speech = SpeechSegment(
        evidence_id="asr_001",
        start_ms=0,
        end_ms=1_000,
        text="Hello",
        language="en",
        confidence=0.9,
        is_fully_evaluated_language=True,
    )
    keyframe = KeyframeEvidence(
        evidence_id="keyframe_ev_001",
        start_ms=0,
        end_ms=1_000,
        keyframe_id="keyframe_001",
        timestamp_ms=500,
        relative_path=relative_path.as_posix(),
        mime_type="image/jpeg",
        sha256=hashlib.sha256(keyframe_bytes).hexdigest(),
        perceptual_hash="abcdef12",
    )
    segment_text = "类型：VIDEO_SEGMENT"
    segment = VideoSegment(
        segment_id="segment_001",
        start_ms=0,
        end_ms=1_000,
        title="问候",
        summary_zh="讲者问好。",
        languages=("en",),
        topics=("问候",),
        keywords=("问候",),
        original_keywords=("Hello",),
        evidence_refs=(speech.evidence_id, keyframe.evidence_id),
        retrieval_text=segment_text,
        retrieval_hash=hashlib.sha256(segment_text.encode("utf-8")).hexdigest(),
    )
    summary_text = "类型：VIDEO_SUMMARY"
    summary = VideoSummary(
        title="测试视频",
        summary_zh="视频包含问候。",
        duration_ms=1_000,
        chapters=(
            SummaryChapter(
                title="问候",
                start_ms=0,
                end_ms=1_000,
                segment_ids=(segment.segment_id,),
            ),
        ),
        languages=("en",),
        topics=("问候",),
        keywords=("问候",),
        original_keywords=("Hello",),
        retrieval_text=summary_text,
        retrieval_hash=hashlib.sha256(summary_text.encode("utf-8")).hexdigest(),
    )
    result = VideoUnderstandingResult(
        run_id=run_id,
        asset_sha256=asset_sha256,
        segments=(segment,),
        summary=summary,
    )
    with container.database.session() as session:
        claimed = JobRepository(session).claim("api-test-publisher", lease_seconds=60)
    assert claimed is not None
    container.result_query_service.persist(
        scope,
        result,
        evidence=(speech, keyframe),
        stage_metrics={"RESULT": 4},
        status=RunStatus.SUCCEEDED,
        transcript_source="ASR",
        fence=ResultWriteFence(
            job_pk=claimed.id,
            worker_id=claimed.worker_id,
            attempt_count=claimed.attempt_count,
        ),
    )
    return segment.segment_id, keyframe_bytes


def _fake_production_pipeline(
    *,
    runtime_root: Path,
    scope: Scope,
    run_id: str,
    object_ref: str,
    source_bytes: bytes,
    source_sha256: str,
    keyframe_bytes: bytes,
    window_count: int = 1,
    failed_clip_ids: frozenset[str] = frozenset(),
) -> ProductionPipeline:
    if window_count not in (1, 2):
        raise ValueError("测试 fake 只支持一个或两个窗口")
    run_root = Path("runs") / ResultQueryService.scope_key(scope) / run_id
    source_path = runtime_root / run_root / "input" / "source.mp4"
    proxy_path = runtime_root / run_root / "media" / "proxy.mp4"
    keyframe_relative_path = run_root / "keyframes" / "frame.jpg"
    keyframe_path = runtime_root / keyframe_relative_path
    for path, content in (
        (source_path, source_bytes),
        (proxy_path, source_bytes),
        (keyframe_path, keyframe_bytes),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    proxy_sha256 = hashlib.sha256(source_bytes).hexdigest()

    class Registrar:
        def register(self, context: PipelineContext) -> RegisteredAsset:
            assert context.run_id == run_id
            assert context.scope == scope
            return RegisteredAsset(
                source_path=source_path,
                source_sha256=source_sha256,
                object_ref=object_ref,
                source_size_bytes=len(source_bytes),
                source_mime="video/mp4",
                run_relative_root=run_root,
                config=PipelineRunConfig(language_hints=("en",)),
            )

    class Probe:
        def probe(self, asset: RegisteredAsset) -> ProbedAsset:
            return ProbedAsset(
                asset=asset,
                manifest=VideoAssetManifest(
                    object_ref=asset.object_ref,
                    source_sha256=asset.source_sha256,
                    source_size_bytes=asset.source_size_bytes,
                    source_mime=asset.source_mime,
                    duration_ms=1_000,
                    video_stream=VideoStream(
                        index=0,
                        codec_name="h264",
                        width=640,
                        height=360,
                        average_frame_rate=Rational(numerator=25, denominator=1),
                    ),
                    format_name="mov,mp4",
                    ffprobe_version="fake-production-1.0",
                ),
                limits=ProbeLimits(),
            )

    class Transcoder:
        def transcode(
            self,
            asset: ProbedAsset,
            **_kwargs: object,
        ) -> PreparedMedia:
            return PreparedMedia(
                source=asset,
                proxy_path=proxy_path,
                proxy_sha256=proxy_sha256,
                proxy_size_bytes=proxy_path.stat().st_size,
                audio_path=None,
                audio_sha256=None,
            )

    class Speech:
        def analyze(
            self,
            _media: PreparedMedia,
            **_kwargs: object,
        ) -> SpeechAnalysis:
            return SpeechAnalysis(
                transcript_source="ASR",
                evidence=(
                    SpeechSegment(
                        evidence_id="asr_001",
                        start_ms=0,
                        end_ms=1_000,
                        text="Hello",
                        language="en",
                        confidence=0.9,
                        is_fully_evaluated_language=True,
                    ),
                ),
                warnings=("NO_AUDIO_STREAM",),
            )

    class Visual:
        def prepare(
            self,
            media: PreparedMedia,
            **_kwargs: object,
        ) -> VisualPreparation:
            ranges = (
                ((0, 1_000),)
                if window_count == 1
                else ((0, 500), (500, 1_000))
            )
            scenes = tuple(
                SceneBoundary(
                    evidence_id=f"scene_{index:03d}",
                    start_ms=start_ms,
                    end_ms=end_ms,
                    transition="candidate",
                    score=0.8,
                )
                for index, (start_ms, end_ms) in enumerate(ranges, start=1)
            )
            return VisualPreparation(
                proxy_sha256=media.proxy_sha256,
                proxy_size_bytes=media.proxy_size_bytes,
                run_relative_root=media.source.asset.run_relative_root,
                duration_ms=media.source.duration_ms,
                frame_tolerance_ms=40,
                scenes=scenes,
                preparation_sha256="a" * 64,
            )

        def finalize(
            self,
            media: PreparedMedia,
            preparation: VisualPreparation,
            **_kwargs: object,
        ) -> VisualAnalysis:
            keyframe = KeyframeEvidence(
                evidence_id="keyframe_ev_001",
                start_ms=0,
                end_ms=1_000,
                keyframe_id="keyframe_001",
                timestamp_ms=500,
                relative_path=keyframe_relative_path.as_posix(),
                mime_type="image/jpeg",
                sha256=hashlib.sha256(keyframe_bytes).hexdigest(),
                perceptual_hash="abcdef12",
            )
            ranges = (
                ((0, 1_000),)
                if window_count == 1
                else ((0, 500), (500, 1_000))
            )
            return VisualAnalysis(
                evidence=(*preparation.scenes, keyframe),
                windows=tuple(
                    TimeRange(
                        start_ms=start_ms,
                        end_ms=end_ms,
                    )
                    for start_ms, end_ms in ranges
                ),
                boundaries=tuple(
                    BoundaryPoint(
                        timestamp_ms=timestamp_ms,
                        sources=(
                            "video_start"
                            if timestamp_ms == 0
                            else "video_end"
                            if timestamp_ms == media.source.duration_ms
                            else "scene_hard"
                        ,),
                    )
                    for timestamp_ms in (
                        (0, 1_000) if window_count == 1 else (0, 500, 1_000)
                    )
                ),
            )

    class Understanding:
        def understand_video(
            self,
            request: WholeVideoUnderstandingRequest,
        ) -> WholeVideoUnderstanding:
            if len(failed_clip_ids) == len(request.windows):
                raise VideoDemoError(
                    ErrorCode.QWEN_RESPONSE_INVALID,
                    "模拟全片理解失败",
                )
            return WholeVideoUnderstanding(
                windows=tuple(
                    WholeVideoWindowUnderstanding(
                        window_id=window.window_id,
                        understanding=SegmentUnderstanding(
                            title="问候",
                            summary_zh="讲者问好。",
                            languages=("en",),
                            topics=("问候",),
                            keywords=("问候",),
                            original_keywords=("Hello",),
                            evidence_refs=tuple(
                                item.evidence_id for item in window.evidence
                            ),
                        ),
                    )
                    for index, window in enumerate(request.windows, start=1)
                    if f"clip_{index:03d}" not in failed_clip_ids
                ),
                summary=SummaryUnderstanding(
                    title="测试视频",
                    summary_zh="视频包含一段问候。",
                    languages=("en",),
                    topics=("问候",),
                    keywords=("问候",),
                    original_keywords=("Hello",),
                ),
            )

    return ProductionPipeline(
        Registrar(),
        Probe(),
        Transcoder(),
        Speech(),
        Visual(),
        Understanding(),
    )
def test_success_result_evidence_cursor_and_keyframe_mime(
    client: TestClient,
    scope_headers: dict[str, str],
    mp4_content: bytes,
) -> None:
    created, asset_sha256 = _upload_and_create(client, scope_headers, mp4_content)
    run_id = str(created["run_id"])
    segment_id, keyframe_bytes = _persist_ready_result(client, run_id, asset_sha256)
    root = f"/api/kb/knowledge-bases/kb-a/video-understanding-runs/{run_id}"

    result_response = client.get(f"{root}/result", headers=scope_headers)
    first_page = client.get(f"{root}/evidence", headers=scope_headers, params={"limit": 1})
    second_page = client.get(
        f"{root}/evidence",
        headers=scope_headers,
        params={"limit": 1, "cursor": first_page.json()["next_cursor"]},
    )
    keyframe_response = client.get(
        f"{root}/keyframes/keyframe_001/content",
        headers=scope_headers,
    )

    assert result_response.status_code == 200
    assert result_response.json()["segments"][0]["segment_id"] == segment_id
    assert first_page.status_code == 200
    assert [item["evidence_id"] for item in first_page.json()["items"]] == ["asr_001"]
    assert [item["evidence_id"] for item in second_page.json()["items"]] == [
        "keyframe_ev_001",
    ]
    assert "relative_path" not in second_page.json()["items"][0]
    assert second_page.json()["next_cursor"] is None
    assert keyframe_response.status_code == 200
    assert keyframe_response.headers["content-type"] == "image/jpeg"
    assert keyframe_response.content == keyframe_bytes


def test_production_pipeline_handler_fence_and_api_publish_complete_result(
    client: TestClient,
    scope_headers: dict[str, str],
    mp4_content: bytes,
) -> None:
    created, asset_sha256 = _upload_and_create(client, scope_headers, mp4_content)
    run_id = str(created["run_id"])
    scope = Scope("tenant-a", "app-a", "kb-a")
    container = client.app.state.container
    keyframe_bytes = b"\xff\xd8\xffpipeline-keyframe"
    with container.database.session() as session:
        run = VideoRunRepository(session).get(scope, run_id)
        assert run is not None
        object_ref = str(run.object_ref)
        claimed = JobRepository(session).claim("production-combination", lease_seconds=60)
    assert claimed is not None
    pipeline = _fake_production_pipeline(
        runtime_root=container.settings.runtime_root,
        scope=scope,
        run_id=run_id,
        object_ref=object_ref,
        source_bytes=mp4_content,
        source_sha256=asset_sha256,
        keyframe_bytes=keyframe_bytes,
    )
    try:
        PipelineJobHandler(
            container.database,
            pipeline,
            container.result_query_service,
        )(claimed)
    finally:
        pipeline.close()

    root = f"/api/kb/knowledge-bases/kb-a/video-understanding-runs/{run_id}"
    result_response = client.get(f"{root}/result", headers=scope_headers)
    evidence_response = client.get(f"{root}/evidence", headers=scope_headers)
    keyframe_response = client.get(
        f"{root}/keyframes/keyframe_001/content",
        headers=scope_headers,
    )

    assert result_response.status_code == 200
    result = result_response.json()
    segment = result["segments"][0]
    summary = result["summary"]
    assert segment["retrieval_hash"] == hashlib.sha256(
        segment["retrieval_text"].encode("utf-8"),
    ).hexdigest()
    assert summary["retrieval_hash"] == hashlib.sha256(
        summary["retrieval_text"].encode("utf-8"),
    ).hexdigest()
    assert evidence_response.status_code == 200
    assert {item["evidence_id"] for item in evidence_response.json()["items"]} == {
        "asr_001",
        "scene_001",
        "keyframe_ev_001",
    }
    assert keyframe_response.status_code == 200
    assert keyframe_response.headers["content-type"] == "image/jpeg"
    assert keyframe_response.content == keyframe_bytes


def test_production_combination_rejects_incomplete_whole_video_response(
    client: TestClient,
    scope_headers: dict[str, str],
    mp4_content: bytes,
) -> None:
    created, asset_sha256 = _upload_and_create(client, scope_headers, mp4_content)
    run_id = str(created["run_id"])
    scope = Scope("tenant-a", "app-a", "kb-a")
    container = client.app.state.container
    keyframe_bytes = b"\xff\xd8\xffpartial-keyframe"
    with container.database.session() as session:
        run = VideoRunRepository(session).get(scope, run_id)
        assert run is not None
        object_ref = str(run.object_ref)
        claimed = JobRepository(session).claim("production-partial", lease_seconds=60)
    assert claimed is not None
    pipeline = _fake_production_pipeline(
        runtime_root=container.settings.runtime_root,
        scope=scope,
        run_id=run_id,
        object_ref=object_ref,
        source_bytes=mp4_content,
        source_sha256=asset_sha256,
        keyframe_bytes=keyframe_bytes,
        window_count=2,
        failed_clip_ids=frozenset({"clip_001"}),
    )
    try:
        with pytest.raises(VideoDemoError) as raised:
            PipelineJobHandler(
                container.database,
                pipeline,
                container.result_query_service,
            )(claimed)
    finally:
        pipeline.close()

    assert raised.value.code == ErrorCode.QWEN_RESPONSE_INVALID
    root = f"/api/kb/knowledge-bases/kb-a/video-understanding-runs/{run_id}"
    run_response = client.get(root, headers=scope_headers)
    result_response = client.get(f"{root}/result", headers=scope_headers)

    assert run_response.status_code == 200
    assert run_response.json()["status"] == "FAILED"
    assert run_response.json()["warning_codes"] == []
    assert run_response.json()["error_code"] == ErrorCode.QWEN_RESPONSE_INVALID
    assert result_response.status_code == 409


def test_production_combination_all_windows_failed_leaves_no_result_or_bundle(
    client: TestClient,
    scope_headers: dict[str, str],
    mp4_content: bytes,
) -> None:
    created, asset_sha256 = _upload_and_create(client, scope_headers, mp4_content)
    run_id = str(created["run_id"])
    scope = Scope("tenant-a", "app-a", "kb-a")
    container = client.app.state.container
    with container.database.session() as session:
        run = VideoRunRepository(session).get(scope, run_id)
        assert run is not None
        object_ref = str(run.object_ref)
        claimed = JobRepository(session).claim("production-all-failed", lease_seconds=60)
    assert claimed is not None
    pipeline = _fake_production_pipeline(
        runtime_root=container.settings.runtime_root,
        scope=scope,
        run_id=run_id,
        object_ref=object_ref,
        source_bytes=mp4_content,
        source_sha256=asset_sha256,
        keyframe_bytes=b"\xff\xd8\xfffailed-keyframe",
        window_count=2,
        failed_clip_ids=frozenset({"clip_001", "clip_002"}),
    )
    try:
        with pytest.raises(VideoDemoError) as raised:
            PipelineJobHandler(
                container.database,
                pipeline,
                container.result_query_service,
            )(claimed)
    finally:
        pipeline.close()

    assert raised.value.code == ErrorCode.QWEN_RESPONSE_INVALID
    root = f"/api/kb/knowledge-bases/kb-a/video-understanding-runs/{run_id}"
    run_response = client.get(root, headers=scope_headers)
    result_response = client.get(f"{root}/result", headers=scope_headers)
    assert run_response.status_code == 200
    assert run_response.json()["status"] == "FAILED"
    assert run_response.json()["error_code"] == ErrorCode.QWEN_RESPONSE_INVALID
    assert result_response.status_code == 409
    assert result_response.json()["error"]["code"] == ErrorCode.VIDEO_RESULT_NOT_READY
    assert list(
        (container.settings.runtime_root / "runs" / ResultQueryService.scope_key(scope) / run_id)
        .rglob("bundle-*.json"),
    ) == []


def test_success_result_is_hidden_from_other_tenant(
    client: TestClient,
    scope_headers: dict[str, str],
    mp4_content: bytes,
) -> None:
    created, asset_sha256 = _upload_and_create(client, scope_headers, mp4_content)
    run_id = str(created["run_id"])
    _persist_ready_result(client, run_id, asset_sha256)

    response = client.get(
        f"/api/kb/knowledge-bases/kb-a/video-understanding-runs/{run_id}/result",
        headers={"X-Tenant-Id": "tenant-b", "X-Application-Id": "app-a"},
    )

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "VIDEO_RUN_NOT_FOUND"


def test_result_and_evidence_openapi_schemas_are_closed_and_path_free(
    client: TestClient,
) -> None:
    schema = client.get("/openapi.json").json()
    components = schema["components"]["schemas"]

    assert components["VideoUnderstandingResult"]["additionalProperties"] is False
    assert components["EvidencePageResponse"]["additionalProperties"] is False
    assert {
        "PublicSpeechSegment",
        "PublicSubtitleCue",
        "PublicSceneBoundary",
        "PublicKeyframeEvidence",
        "PublicOcrEvidence",
    }.issubset(components)
    assert {
        "PublicAlignedWord",
        "PublicSpeakerTurn",
        "PublicAudioEvent",
    }.isdisjoint(components)
    serialized = str(
        {
            "result": components["VideoUnderstandingResult"],
            "evidence": components["EvidencePageResponse"],
        },
    )
    assert "relative_path" not in serialized
    assert "local_path" not in serialized
    assert "ALIGNED_WORD" not in serialized
    assert "SPEAKER_TURN" not in serialized
    assert "AUDIO_EVENT" not in serialized


def test_error_response_does_not_expose_internal_paths(
    client: TestClient,
) -> None:
    def leak_path() -> None:
        raise VideoDemoError(
            ErrorCode.WORKSPACE_PATH_ESCAPE,
            "路径逃逸",
            {"workspace": "/Users/private/workspace", "field": "video"},
        )

    client.app.add_api_route("/__test__/path-error", leak_path)

    response = client.get("/__test__/path-error")

    assert response.status_code == 422
    assert response.json()["error"]["details"] == {"field": "video"}
    assert "/Users/private/workspace" not in response.text
