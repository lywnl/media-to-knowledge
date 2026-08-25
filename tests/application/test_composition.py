from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from pydantic import SecretStr

from video_demo.application.composition import (
    ProductionDiagnosticComponents,
    build_production_diagnostic_components,
    build_worker,
)
from video_demo.application.pipeline import (
    PipelineContext,
    PipelineRunConfig,
    PreparedMedia,
    ProbedAsset,
    RegisteredAsset,
    SpeechAnalysis,
    VisualAnalysis,
    VisualPreparation,
)
from video_demo.application.production_media import (
    ProductionAssetProbe,
    ProductionAssetRegistrar,
    ProductionMediaTranscoder,
)
from video_demo.application.production_visual import ProductionVisualAnalyzer
from video_demo.application.runs import RunService
from video_demo.application.uploads import UploadService
from video_demo.config import Settings
from video_demo.domain.evidence import SceneBoundary, SpeechSegment
from video_demo.domain.manifest import Rational, VideoAssetManifest, VideoStream
from video_demo.domain.result import SegmentUnderstanding, SummaryUnderstanding
from video_demo.domain.run import RunStatus, TimeRange
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.fusion.merge import BoundaryPoint
from video_demo.fusion.timeline import build_timeline
from video_demo.integrations.oss import PublishedVideoUnderstanding
from video_demo.integrations.qwen import DemoFallbackVideoUnderstanding, QwenVideoClient
from video_demo.integrations.video_port import (
    SegmentUnderstandingRequest,
    VideoClipInput,
    WholeVideoUnderstanding,
    WholeVideoUnderstandingRequest,
    WholeVideoWindowUnderstanding,
)
from video_demo.media.probe import ProbeLimits
from video_demo.persistence.database import Database
from video_demo.persistence.models import JobStatus, RunStatusValue
from video_demo.persistence.repositories import JobRepository, Scope, VideoRunRepository
from video_demo.speech.vad import SileroVadAdapter
from video_demo.storage.object_store import LocalVideoObjectStore


@pytest.mark.parametrize(
    "builder",
    (
        lambda settings: build_worker(settings, worker_id="worker-missing-cloud-asr"),
        lambda settings: __import__(
            "video_demo.application.composition",
            fromlist=["build_production_pipeline"],
        ).build_production_pipeline(settings, object(), object()),
        lambda settings: build_production_diagnostic_components(settings),
    ),
)
def test_production_composition_rejects_missing_cloud_asr_before_side_effects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    builder: object,
) -> None:
    import video_demo.application.composition as composition

    settings = Settings(workspace_root=tmp_path, _env_file=None)
    assert settings.runtime_root is not None
    monkeypatch.setattr(
        composition.httpx,
        "Client",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("配置失败前不得创建 HTTP 客户端")
        ),
    )

    with pytest.raises(VideoDemoError) as raised:
        builder(settings)  # type: ignore[operator]

    assert raised.value.code == ErrorCode.INVALID_CONFIGURATION
    assert not settings.runtime_root.exists()


def test_production_worker_consumes_created_job_and_records_missing_ffprobe(
    tmp_path: Path,
    cloud_asr_environment: None,
) -> None:
    settings = Settings(
        workspace_root=tmp_path,
        ffprobe_path=tmp_path / "missing-ffprobe",
        ffmpeg_path=tmp_path / "missing-ffmpeg",
    )
    assert settings.runtime_root is not None
    settings.runtime_root.mkdir(parents=True)
    database = Database(f"sqlite+pysqlite:///{settings.runtime_root / 'video-demo.db'}")
    database.create_schema()
    object_store = LocalVideoObjectStore(settings.runtime_root, max_video_bytes=1024)
    scope = Scope("tenant-a", "app-a", "kb-a")
    uploaded = UploadService(database, object_store).upload(
        BytesIO(b"\x00\x00\x00\x18ftypisomvideo"),
        "lesson.mp4",
        "video/mp4",
        scope,
    )
    run = RunService(database).create(
        scope=scope,
        object_ref=uploaded.object_ref,
        idempotency_key="production-worker-0001",
        language_hints=("en",),
    )

    worker = build_worker(settings, worker_id="worker-production-test")
    try:
        assert worker.run_once() is True
        with database.session() as session:
            job = JobRepository(session).get(scope, run.job_id)
            persisted_run = VideoRunRepository(session).get(scope, run.run_id)
            assert job is not None
            assert persisted_run is not None
            assert job.status == JobStatus.FAILED
            assert job.error_code == ErrorCode.VIDEO_FFPROBE_UNAVAILABLE
            assert persisted_run.status == RunStatusValue.FAILED
            assert persisted_run.error_code == ErrorCode.VIDEO_FFPROBE_UNAVAILABLE
    finally:
        worker.close()


def test_production_pipeline_reuses_complete_orchestration_and_builds_retrieval_text(
    tmp_path: Path,
) -> None:
    from video_demo.application.composition import ProductionPipeline

    calls: list[str] = []
    stages: list[str] = []
    source = tmp_path / "source.mp4"
    clip = tmp_path / "clip.mp4"
    source.write_bytes(b"source")
    clip.write_bytes(b"clip")
    source_digest = hashlib.sha256(source.read_bytes()).hexdigest()
    clip_digest = hashlib.sha256(clip.read_bytes()).hexdigest()

    class Registrar:
        def register(self, _context: PipelineContext) -> RegisteredAsset:
            calls.append("REGISTER")
            return RegisteredAsset(
                source_path=source,
                source_sha256=source_digest,
                object_ref="obj_001",
                source_size_bytes=source.stat().st_size,
                source_mime="video/mp4",
                run_relative_root=Path("runs/scope/run_001"),
                config=PipelineRunConfig(language_hints=("en",)),
            )

    class Probe:
        def probe(self, asset: RegisteredAsset) -> ProbedAsset:
            calls.append("PROBE")
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
                    ffprobe_version="test",
                ),
                limits=ProbeLimits(),
            )

    class Transcoder:
        def transcode(self, asset: ProbedAsset, **_kwargs: object) -> PreparedMedia:
            calls.append("TRANSCODE")
            assert _kwargs["is_cancel_requested"] is cancel
            return PreparedMedia(
                source=asset,
                proxy_path=clip,
                proxy_sha256=clip_digest,
                proxy_size_bytes=clip.stat().st_size,
                audio_path=None,
                audio_sha256=None,
            )

    class Speech:
        def analyze(self, _media: PreparedMedia, **_kwargs: object) -> SpeechAnalysis:
            calls.append("SPEECH")
            assert _kwargs["is_cancel_requested"] is cancel
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
            )

    class Visual:
        def prepare(self, media: PreparedMedia, **_kwargs: object) -> VisualPreparation:
            calls.append("VISUAL_PREPARE")
            assert _kwargs["is_cancel_requested"] is cancel
            scene = SceneBoundary(
                evidence_id="scene_001",
                start_ms=0,
                end_ms=1_000,
                transition="candidate",
                score=0.8,
            )
            return VisualPreparation(
                proxy_sha256=media.proxy_sha256,
                proxy_size_bytes=media.proxy_size_bytes,
                run_relative_root=media.source.asset.run_relative_root,
                duration_ms=1_000,
                frame_tolerance_ms=40,
                scenes=(scene,),
                preparation_sha256="a" * 64,
            )

        def finalize(
            self,
            media: PreparedMedia,
            preparation: VisualPreparation,
            **_kwargs: object,
        ) -> VisualAnalysis:
            calls.append("VISUAL")
            assert isinstance(_kwargs["speech"], SpeechAnalysis)
            assert _kwargs["is_cancel_requested"] is cancel
            return VisualAnalysis(
                evidence=preparation.scenes,
                windows=(TimeRange(start_ms=0, end_ms=1_000),),
                boundaries=(
                    BoundaryPoint(timestamp_ms=0, sources=("video_start",)),
                    BoundaryPoint(timestamp_ms=1_000, sources=("video_end",)),
                ),
            )

    class Understanding:
        def understand_video(
            self,
            request: WholeVideoUnderstandingRequest,
        ) -> WholeVideoUnderstanding:
            calls.append("UNDERSTANDING")
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
                            evidence_refs=(window.evidence[0].evidence_id,),
                        ),
                    )
                    for window in request.windows
                ),
                summary=SummaryUnderstanding(
                    title="测试视频",
                    summary_zh="视频包含问候。",
                    languages=("en",),
                    topics=("问候",),
                    keywords=("问候",),
                    original_keywords=("Hello",),
                ),
            )

    def cancel() -> bool:
        return False

    pipeline = ProductionPipeline(
        Registrar(),  # type: ignore[arg-type]
        Probe(),  # type: ignore[arg-type]
        Transcoder(),  # type: ignore[arg-type]
        Speech(),  # type: ignore[arg-type]
        Visual(),  # type: ignore[arg-type]
        Understanding(),
    )

    outcome = pipeline.run(
        PipelineContext(
            run_id="run_001",
            is_cancel_requested=cancel,
            on_stage_start=stages.append,
        )
    )

    assert outcome.status == RunStatus.SUCCEEDED
    segment = outcome.result.segments[0]
    assert segment.retrieval_text.startswith("文档类型：VIDEO_SEGMENT")
    assert segment.retrieval_hash == hashlib.sha256(
        segment.retrieval_text.encode("utf-8"),
    ).hexdigest()
    assert outcome.result.summary.retrieval_hash == hashlib.sha256(
        outcome.result.summary.retrieval_text.encode("utf-8"),
    ).hexdigest()
    assert set(calls[3:5]) == {"SPEECH", "VISUAL_PREPARE"}
    assert calls[5:] == ["VISUAL", "UNDERSTANDING"]
    assert stages == [
        "REGISTER",
        "PROBE",
        "TRANSCODE",
        "SPEECH",
        "VISUAL",
        "FUSION",
        "UNDERSTANDING",
        "RESULT",
    ]


def test_production_pipeline_preserves_injectable_stage_clock() -> None:
    from video_demo.application.composition import ProductionPipeline

    def clock() -> float:
        return 1.0

    production = ProductionPipeline(
        object(),  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        clock=clock,
    )

    assert production._clock is clock  # type: ignore[attr-defined]


def test_worker_builds_real_production_media_adapters(
    tmp_path: Path,
    cloud_asr_environment: None,
) -> None:
    from video_demo.application.composition import build_production_pipeline

    settings = Settings(workspace_root=tmp_path)
    assert settings.runtime_root is not None
    settings.runtime_root.mkdir(parents=True)
    database = Database(f"sqlite+pysqlite:///{settings.runtime_root / 'video-demo.db'}")
    database.create_schema()
    object_store = LocalVideoObjectStore(settings.runtime_root, max_video_bytes=1024)

    pipeline = build_production_pipeline(settings, database, object_store)
    try:
        assert isinstance(pipeline.registrar, ProductionAssetRegistrar)
        assert isinstance(pipeline.probe, ProductionAssetProbe)
        assert isinstance(pipeline.transcoder, ProductionMediaTranscoder)
        from video_demo.speech.isolated import IsolatedSpeechAnalyzer

        assert isinstance(pipeline.speech_analyzer, IsolatedSpeechAnalyzer)
        assert isinstance(pipeline.visual_analyzer, ProductionVisualAnalyzer)
        assert pipeline.visual_analyzer._evidence_limits.max_transcript_evidence_items == (
            settings.max_transcript_evidence_items
        )
        assert pipeline.visual_analyzer._evidence_limits.max_transcript_chars == (
            settings.max_transcript_chars
        )
        assert pipeline.visual_analyzer._evidence_limits.max_scene_boundaries == (
            settings.max_scene_boundaries
        )
        assert pipeline.visual_analyzer._evidence_limits.max_base_segments == (
            settings.max_base_segments
        )
        assert isinstance(pipeline.understanding, QwenVideoClient)
    finally:
        pipeline.close()


def test_production_pipeline_does_not_construct_in_process_speech_models(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cloud_asr_environment: None,
) -> None:
    import video_demo.application.composition as composition

    settings = Settings(workspace_root=tmp_path)
    assert settings.runtime_root is not None
    settings.runtime_root.mkdir(parents=True)
    database = Database(f"sqlite+pysqlite:///{settings.runtime_root / 'video-demo.db'}")
    database.create_schema()
    object_store = LocalVideoObjectStore(settings.runtime_root, max_video_bytes=1024)
    monkeypatch.setattr(
        composition,
        "_build_speech_models",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("生产父进程不得构造重语音模型")
        ),
    )

    pipeline = composition.build_production_pipeline(settings, database, object_store)

    pipeline.close()


def test_strict_production_pipeline_rejects_configured_qwen_without_oss(
    tmp_path: Path,
    cloud_asr_environment: None,
) -> None:
    from video_demo.application.composition import build_production_pipeline

    settings = Settings(
        workspace_root=tmp_path,
        qwen_base_url="https://ai-proxy.example.test/v1",
        qwen_model_id="qwen3-vl-flash",
        qwen_api_key="qwen-secret",
    )
    assert settings.runtime_root is not None
    settings.runtime_root.mkdir(parents=True)
    database = Database(f"sqlite+pysqlite:///{settings.runtime_root / 'video-demo.db'}")
    database.create_schema()
    object_store = LocalVideoObjectStore(settings.runtime_root, max_video_bytes=1024)

    with pytest.raises(VideoDemoError) as raised:
        build_production_pipeline(settings, database, object_store)

    assert raised.value.code == ErrorCode.OSS_CONFIGURATION_INVALID


def test_production_pipeline_wraps_qwen_with_private_oss_publisher(
    tmp_path: Path,
    cloud_asr_environment: None,
) -> None:
    from video_demo.application.composition import build_production_pipeline

    settings = Settings(
        workspace_root=tmp_path,
        qwen_base_url="https://ai-proxy.example.test/v1",
        qwen_model_id="qwen3-vl-flash",
        qwen_api_key="qwen-secret",
        oss_endpoint="https://oss-cn-hangzhou.aliyuncs.com",
        oss_bucket="private-video-bucket",
        oss_access_key_id="oss-access-key",
        oss_access_key_secret="oss-secret",
    )
    assert settings.runtime_root is not None
    settings.runtime_root.mkdir(parents=True)
    database = Database(f"sqlite+pysqlite:///{settings.runtime_root / 'video-demo.db'}")
    database.create_schema()
    object_store = LocalVideoObjectStore(settings.runtime_root, max_video_bytes=1024)

    pipeline = build_production_pipeline(settings, database, object_store)
    try:
        assert isinstance(pipeline.understanding, PublishedVideoUnderstanding)
        delegate = pipeline.understanding._delegate  # type: ignore[attr-defined]
        assert isinstance(delegate, QwenVideoClient)
        assert delegate._allowed_remote_video_hosts == frozenset(  # type: ignore[attr-defined]
            {"private-video-bucket.oss-cn-hangzhou.aliyuncs.com"},
        )
    finally:
        pipeline.close()


def test_demo_pipeline_fallback_wraps_published_qwen(
    tmp_path: Path,
    cloud_asr_environment: None,
) -> None:
    from video_demo.application.composition import build_production_pipeline

    settings = Settings(
        workspace_root=tmp_path,
        demo_degraded_mode=True,
        qwen_base_url="https://ai-proxy.example.test/v1",
        qwen_model_id="qwen3-vl-flash",
        qwen_api_key="qwen-secret",
        oss_endpoint="https://oss-cn-hangzhou.aliyuncs.com",
        oss_bucket="private-video-bucket",
        oss_access_key_id="oss-access-key",
        oss_access_key_secret="oss-secret",
    )
    assert settings.runtime_root is not None
    settings.runtime_root.mkdir(parents=True)
    database = Database(f"sqlite+pysqlite:///{settings.runtime_root / 'video-demo.db'}")
    database.create_schema()
    object_store = LocalVideoObjectStore(settings.runtime_root, max_video_bytes=1024)

    pipeline = build_production_pipeline(settings, database, object_store)
    try:
        assert isinstance(pipeline.understanding, DemoFallbackVideoUnderstanding)
        assert isinstance(
            pipeline.understanding._delegate,  # type: ignore[attr-defined]
            PublishedVideoUnderstanding,
        )
    finally:
        pipeline.close()


def test_production_pipeline_passes_nullable_qwen_configuration_and_dedicated_limits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cloud_asr_environment: None,
) -> None:
    import video_demo.application.composition as composition

    captured: dict[str, object] = {}

    class Qwen:
        def __init__(self, _client: object, **kwargs: object) -> None:
            captured.update(kwargs)

        def understand_segment(self, _request: object) -> object:
            raise AssertionError("构造测试不得调用 Qwen")

        def summarize_video(self, _request: object) -> object:
            raise AssertionError("构造测试不得调用 Qwen")

    monkeypatch.setattr(composition, "QwenVideoClient", Qwen)
    settings = Settings(
        workspace_root=tmp_path,
        qwen_base_url=None,
        qwen_model_id=None,
        qwen_api_key=None,
        qwen_max_video_bytes=1234,
        qwen_max_video_duration_ms=12_345,
        qwen_timeout_seconds=45.0,
    )
    assert settings.runtime_root is not None
    settings.runtime_root.mkdir(parents=True)
    database = Database(f"sqlite+pysqlite:///{settings.runtime_root / 'video-demo.db'}")
    database.create_schema()
    object_store = LocalVideoObjectStore(settings.runtime_root, max_video_bytes=1024)

    pipeline = composition.build_production_pipeline(settings, database, object_store)
    try:
        api_key_provider = captured.pop("api_key_provider")
        assert callable(api_key_provider)
        assert api_key_provider() is None
        assert captured == {
            "base_url": None,
            "api_key": None,
            "model_id": None,
            "allowed_video_root": settings.runtime_root,
            "max_video_bytes": 1234,
            "max_video_duration_ms": 12_345,
            "timeout_seconds": 45.0,
            "allowed_remote_video_hosts": None,
        }
        assert pipeline.understanding.__class__ is Qwen
    finally:
        pipeline.close()


def test_diagnostic_builder_uses_public_qwen_api_key_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cloud_asr_environment: None,
) -> None:
    import video_demo.application.composition as composition

    captured: dict[str, object] = {}

    class Qwen:
        __slots__ = ()

        def __init__(self, _client: object, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(composition, "QwenVideoClient", Qwen)
    settings = Settings(
        workspace_root=tmp_path,
        qwen_api_key="qwen-secret",
    )
    assert settings.runtime_root is not None
    settings.runtime_root.mkdir(parents=True)

    diagnostics = build_production_diagnostic_components(settings)
    try:
        api_key_provider = captured["api_key_provider"]
        assert callable(api_key_provider)
        assert captured["api_key"] is None
        assert api_key_provider() is settings.qwen_api_key
    finally:
        diagnostics.close()


def test_worker_starts_without_qwen_configuration_and_fails_at_first_clip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cloud_asr_environment: None,
) -> None:
    import video_demo.application.composition as composition

    settings = Settings(workspace_root=tmp_path)
    assert settings.runtime_root is not None
    settings.runtime_root.mkdir(parents=True)
    database = Database(f"sqlite+pysqlite:///{settings.runtime_root / 'video-demo.db'}")
    database.create_schema()
    object_store = LocalVideoObjectStore(settings.runtime_root, max_video_bytes=1024)
    scope = Scope("tenant-a", "app-a", "kb-a")
    uploaded = UploadService(database, object_store).upload(
        BytesIO(b"\x00\x00\x00\x18ftypisomvideo"),
        "lesson.mp4",
        "video/mp4",
        scope,
    )
    run = RunService(database).create(
        scope=scope,
        object_ref=uploaded.object_ref,
        idempotency_key="missing-qwen-at-first-clip",
        language_hints=("en",),
    )
    clip_path = settings.runtime_root / "clip.mp4"
    clip_path.write_bytes(b"clip")
    speech = SpeechSegment(
        evidence_id="asr_001",
        start_ms=0,
        end_ms=1_000,
        text="Hello",
        language="en",
        confidence=0.9,
        is_fully_evaluated_language=True,
    )
    request = SegmentUnderstandingRequest(
        clip=VideoClipInput(
            clip_id="clip_001",
            start_ms=0,
            end_ms=1_000,
            path=clip_path,
            mime_type="video/mp4",
            sha256=hashlib.sha256(b"clip").hexdigest(),
        ),
        window=TimeRange(start_ms=0, end_ms=1_000),
        timeline=build_timeline((speech,)),
        evidence=(speech,),
    )
    qwen_http_client = httpx.Client(
        transport=httpx.MockTransport(
            lambda _request: (_ for _ in ()).throw(
                AssertionError("缺配置不得外发"),
            ),
        ),
    )
    qwen = QwenVideoClient(
        qwen_http_client,
        base_url=None,
        api_key=None,
        model_id=None,
        allowed_video_root=settings.runtime_root,
    )
    closes: list[str] = []

    class Pipeline:
        def run(self, _context: PipelineContext) -> object:
            return qwen.understand_segment(request)

        def close(self) -> None:
            closes.append("pipeline")

    monkeypatch.setattr(
        composition,
        "build_production_pipeline",
        lambda *_args: Pipeline(),
    )

    worker = composition.build_worker(settings, worker_id="worker-missing-qwen")
    try:
        assert worker.run_once() is True
        with database.session() as session:
            job = JobRepository(session).get(scope, run.job_id)
            persisted_run = VideoRunRepository(session).get(scope, run.run_id)
            assert job is not None
            assert persisted_run is not None
            assert job.status == JobStatus.FAILED
            assert job.error_code == ErrorCode.QWEN_CAPABILITY_UNAVAILABLE
            assert persisted_run.status == RunStatusValue.FAILED
            assert persisted_run.error_code == ErrorCode.QWEN_CAPABILITY_UNAVAILABLE
    finally:
        worker.close()
        qwen_http_client.close()

    assert closes == ["pipeline"]


def test_demo_mode_accepts_unrecognized_qwen_model_for_deterministic_fallback(
    tmp_path: Path,
    cloud_asr_environment: None,
) -> None:
    from video_demo.application.composition import build_production_model_identity_report

    report = build_production_model_identity_report(
        Settings(
            workspace_root=tmp_path,
            demo_degraded_mode=True,
            qwen_model_id="provider-video-model",
        )
    )

    qwen_models = tuple(item for item in report.models if item.component == "qwen")
    assert len(qwen_models) == 1
    assert qwen_models[0].model_id == "provider-video-model"


def test_speech_fingerprint_inputs_bind_cloud_model_and_silero_version(
    tmp_path: Path,
    cloud_asr_environment: None,
) -> None:
    from video_demo.application.composition import _speech_fingerprint_inputs

    inputs = _speech_fingerprint_inputs(Settings(workspace_root=tmp_path))

    identities = {identity.component: identity for identity in inputs.model_identities}
    assert identities["silero_vad"].revision
    assert identities["cloud_whisper"].provider == "openai_compatible"
    assert identities["cloud_whisper"].model_id == "openai/whisper"
    assert inputs.cloud_asr_base_url == "https://ai-proxy.example.test/v1"
    assert (inputs.max_window_ms, inputs.overlap_ms) == (600_000, 1_000)


def test_speech_content_fingerprint_ignores_cloud_credentials_and_delivery_policy(
    tmp_path: Path,
) -> None:
    from video_demo.application.composition import _speech_fingerprint_inputs
    from video_demo.speech.snapshots import asr_fingerprint

    common: dict[str, object] = {
        "workspace_root": tmp_path,
        "openai_base_url": "https://asr.example/v1",
        "openai_model": "openai/whisper",
        "_env_file": None,
    }

    def content_fingerprint(settings: Settings) -> str:
        return asr_fingerprint(
            audio_sha256="a" * 64,
            duration_ms=60_000,
            language_hints=("zh",),
            hotwords=("Milvus",),
            core_context="向量数据库课程",
            inputs=_speech_fingerprint_inputs(settings),
        )

    fingerprints = {
        content_fingerprint(
            Settings(
                **common,
                openai_api_key="first-test-key",
                openai_asr_timeout_seconds=300,
                openai_asr_max_attempts=3,
            )
        ),
        content_fingerprint(
            Settings(
                **common,
                openai_api_key="second-test-key",
                openai_asr_timeout_seconds=120,
                openai_asr_max_attempts=5,
            )
        ),
    }

    assert len(fingerprints) == 1


def test_production_model_identity_report_uses_only_current_cloud_speech_stack(
    tmp_path: Path,
    cloud_asr_environment: None,
) -> None:
    from video_demo.application.composition import build_production_model_identity_report

    report = build_production_model_identity_report(Settings(workspace_root=tmp_path))
    identities = {item.component: item for item in report.models}

    assert report.schema_version == "2.0.0"
    assert {"silero_vad", "cloud_whisper"}.issubset(identities)
    assert {
        "faster_whisper",
        "whisperx",
        "pyannote",
        "yamnet",
    }.isdisjoint(identities)
    cloud = identities["cloud_whisper"]
    assert cloud.provider == "openai_compatible"
    assert cloud.model_id == "openai/whisper"
    assert cloud.revision is None
    assert cloud.device is None


def test_production_speech_factory_reuses_lifecycle_models_across_tasks(
    tmp_path: Path,
) -> None:
    from video_demo.application.composition import _build_speech_component_factory
    from video_demo.speech.runtime import ProductionSpeechModels
    from video_demo.speech.vad import NativeSileroBackend

    settings = Settings(workspace_root=tmp_path)
    clients: list[object] = []

    def ffmpeg_factory(_cancel: object) -> object:
        client = object()
        clients.append(client)
        return client

    models = ProductionSpeechModels(
        vad=SileroVadAdapter(NativeSileroBackend()),
        recognizer=object(),  # type: ignore[arg-type]
    )
    factory = _build_speech_component_factory(
        settings,
        ffmpeg_factory,
        models=models,
    )
    first = factory(  # type: ignore[arg-type]
        SimpleNamespace(source=SimpleNamespace(duration_ms=1_000)),
        lambda: False,
    )
    second = factory(  # type: ignore[arg-type]
        SimpleNamespace(source=SimpleNamespace(duration_ms=2_000)),
        lambda: False,
    )

    assert isinstance(first.vad, SileroVadAdapter)
    assert first.vad is second.vad
    assert first.recognizer is second.recognizer
    assert first.vad._backend is second.vad._backend  # type: ignore[attr-defined]
    assert first.slicer is not second.slicer
    assert len(clients) == 2


def test_production_visual_factory_reuses_ocr_but_not_task_clip_client(
    tmp_path: Path,
) -> None:
    from video_demo.application.composition import _build_visual_component_factory

    settings = Settings(workspace_root=tmp_path)
    callbacks: list[object] = []

    def ffmpeg_factory(cancel: object) -> object:
        callbacks.append(cancel)
        return object()

    factory = _build_visual_component_factory(settings, ffmpeg_factory)

    def first_cancel() -> bool:
        return False

    def second_cancel() -> bool:
        return False

    try:
        first = factory(SimpleNamespace(), first_cancel)  # type: ignore[arg-type]
        second = factory(SimpleNamespace(), second_cancel)  # type: ignore[arg-type]

        assert first.scene_detector is second.scene_detector
        assert first.frame_extractor is second.frame_extractor
        assert first.keyframe_selector is second.keyframe_selector
        assert first.ocr_client is second.ocr_client
        assert first.clip_client is not second.clip_client
        assert callbacks == [first_cancel, second_cancel]
    finally:
        factory.http_client.close()


def test_production_pipeline_owns_http_client_and_close_is_idempotent() -> None:
    from video_demo.application.composition import ProductionPipeline

    closes: list[str] = []

    class Client:
        def close(self) -> None:
            closes.append("close")

    pipeline = ProductionPipeline(  # type: ignore[arg-type]
        object(),
        object(),
        object(),
        object(),
        object(),
        object(),
        owned_resources=(Client(),),
    )

    pipeline.close()
    pipeline.close()

    assert closes == ["close"]


def test_production_pipeline_close_attempts_all_resources_when_later_close_raises() -> None:
    from video_demo.application.composition import ProductionPipeline

    closes: list[str] = []

    class Resource:
        def __init__(self, name: str, *, raises: bool = False) -> None:
            self._name = name
            self._raises = raises

        def close(self) -> None:
            closes.append(self._name)
            if self._raises:
                raise RuntimeError(f"{self._name} close failed")

    pipeline = ProductionPipeline(  # type: ignore[arg-type]
        object(),
        object(),
        object(),
        object(),
        object(),
        object(),
        owned_resources=(
            Resource("visual"),
            Resource("qwen", raises=True),
        ),
    )

    with pytest.raises(RuntimeError, match="qwen close failed"):
        pipeline.close()
    pipeline.close()

    assert closes == ["qwen", "visual"]


def test_production_pipeline_build_closes_http_client_when_owner_construction_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cloud_asr_environment: None,
) -> None:
    import video_demo.application.composition as composition

    closes: list[str] = []

    class Client:
        def close(self) -> None:
            closes.append("close")

    def fail_pipeline(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("pipeline construction failed")

    monkeypatch.setattr(composition.httpx, "Client", Client)
    monkeypatch.setattr(composition, "ProductionPipeline", fail_pipeline)
    settings = Settings(workspace_root=tmp_path)
    assert settings.runtime_root is not None
    settings.runtime_root.mkdir(parents=True)

    with pytest.raises(RuntimeError, match="pipeline construction failed"):
        composition.build_production_pipeline(settings, object(), object())  # type: ignore[arg-type]

    assert closes == ["close", "close"]


def test_qwen_adapter_construction_failure_closes_visual_and_qwen_http_clients(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cloud_asr_environment: None,
) -> None:
    import video_demo.application.composition as composition

    closes: list[int] = []
    created = 0

    class Client:
        def __init__(self) -> None:
            nonlocal created
            created += 1
            self._identity = created

        def close(self) -> None:
            closes.append(self._identity)

    def fail_qwen(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("qwen adapter construction failed")

    monkeypatch.setattr(composition.httpx, "Client", Client)
    monkeypatch.setattr(composition, "QwenVideoClient", fail_qwen)
    settings = Settings(workspace_root=tmp_path)
    assert settings.runtime_root is not None
    settings.runtime_root.mkdir(parents=True)

    with pytest.raises(RuntimeError, match="qwen adapter construction failed"):
        composition.build_production_pipeline(settings, object(), object())  # type: ignore[arg-type]

    assert closes == [2, 1]


def test_qwen_http_client_construction_failure_closes_visual_http_client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cloud_asr_environment: None,
) -> None:
    import video_demo.application.composition as composition

    closes: list[int] = []
    created = 0

    class Client:
        def __init__(self) -> None:
            nonlocal created
            created += 1
            self._identity = created
            if created == 2:
                raise RuntimeError("qwen http construction failed")

        def close(self) -> None:
            closes.append(self._identity)

    monkeypatch.setattr(composition.httpx, "Client", Client)
    settings = Settings(workspace_root=tmp_path)
    assert settings.runtime_root is not None
    settings.runtime_root.mkdir(parents=True)

    with pytest.raises(RuntimeError, match="qwen http construction failed"):
        composition.build_production_pipeline(settings, object(), object())  # type: ignore[arg-type]

    assert closes == [1]


def test_visual_factory_closes_http_client_when_component_construction_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import video_demo.application.composition as composition

    closes: list[str] = []

    class Client:
        def close(self) -> None:
            closes.append("close")

    def fail_extractor(_runtime_root: Path) -> object:
        raise RuntimeError("extractor construction failed")

    monkeypatch.setattr(composition.httpx, "Client", Client)
    monkeypatch.setattr(composition, "OpenCvFrameExtractor", fail_extractor)

    with pytest.raises(RuntimeError, match="extractor construction failed"):
        composition.ProductionVisualComponentFactory(
            tmp_path,
            object(),
            lambda: (None, None),
            endpoint="https://example.invalid/ocr",
        )

    assert closes == ["close"]


@pytest.mark.parametrize("failure", ["handler", "worker"])
def test_build_worker_closes_pipeline_when_downstream_construction_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
    cloud_asr_environment: None,
) -> None:
    import video_demo.application.composition as composition

    closes: list[str] = []

    class Pipeline:
        def close(self) -> None:
            closes.append("close")

    monkeypatch.setattr(composition, "build_production_pipeline", lambda *_args: Pipeline())
    if failure == "handler":
        monkeypatch.setattr(
            composition,
            "PipelineJobHandler",
            lambda *_args: (_ for _ in ()).throw(RuntimeError("handler construction failed")),
        )
    else:
        monkeypatch.setattr(composition, "PipelineJobHandler", lambda *_args: object())
        monkeypatch.setattr(
            composition,
            "ReliableWorker",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("worker construction failed"),
            ),
        )

    with pytest.raises(RuntimeError, match=f"{failure} construction failed"):
        composition.build_worker(Settings(workspace_root=tmp_path), worker_id="worker-test")

    assert closes == ["close"]


def test_build_worker_transfers_pipeline_to_successful_worker_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cloud_asr_environment: None,
) -> None:
    import video_demo.application.composition as composition

    closes: list[str] = []

    class Pipeline:
        def close(self) -> None:
            closes.append("close")

    pipeline = Pipeline()
    monkeypatch.setattr(composition, "build_production_pipeline", lambda *_args: pipeline)
    monkeypatch.setattr(composition, "PipelineJobHandler", lambda *_args: lambda _job: None)

    worker = composition.build_worker(
        Settings(workspace_root=tmp_path),
        worker_id="worker-test",
    )

    assert closes == []
    worker.close()
    worker.close()
    assert closes == ["close"]


def test_production_pipeline_builds_lightweight_dependencies_without_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cloud_asr_environment: None,
) -> None:
    import video_demo.application.composition as composition

    settings = Settings(workspace_root=tmp_path)
    assert settings.runtime_root is not None
    settings.runtime_root.mkdir(parents=True)
    database = Database(f"sqlite+pysqlite:///{settings.runtime_root / 'video-demo.db'}")
    object_store = LocalVideoObjectStore(settings.runtime_root, max_video_bytes=1024)
    monkeypatch.setattr(
        composition,
        "build_production_diagnostic_components",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("生产组合根不得构造完整诊断组件")
        ),
    )

    pipeline = composition.build_production_pipeline(settings, database, object_store)
    try:
        from video_demo.speech.isolated import IsolatedSpeechAnalyzer

        assert isinstance(pipeline.speech_analyzer, IsolatedSpeechAnalyzer)
        assert isinstance(pipeline.visual_analyzer, ProductionVisualAnalyzer)
    finally:
        pipeline.close()
        pipeline.close()


def test_diagnostic_builder_keeps_all_secrets_lazy_until_adapter_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cloud_asr_environment: None,
) -> None:
    import video_demo.application.composition as composition

    settings = Settings(
        workspace_root=tmp_path,
        qwen_base_url="https://dashscope.example/compatible-mode/v1",
        qwen_model_id="qwen3-vl-plus",
        qwen_api_key="  qwen-secret  ",
        baidu_api_key="baidu-api-secret",
        baidu_secret_key="baidu-secret-secret",
    )
    assert settings.runtime_root is not None
    settings.runtime_root.mkdir(parents=True)
    revealed: list[str] = []
    authorizations: list[str] = []
    original_get_secret_value = SecretStr.get_secret_value

    def reveal(secret: SecretStr) -> str:
        value = original_get_secret_value(secret)
        revealed.append(value)
        return value

    class Client:
        def post(self, *_args: object, **_kwargs: object) -> httpx.Response:
            headers = _kwargs["headers"]
            assert isinstance(headers, dict)
            authorizations.append(str(headers["Authorization"]))
            return httpx.Response(200, json={})

        def close(self) -> None:
            return None

    class LazyBaidu:
        def __init__(
            self,
            _client: object,
            credentials_provider: object,
            *,
            endpoint: str,
        ) -> None:
            assert callable(credentials_provider)
            self._credentials_provider = credentials_provider
            self.endpoint = endpoint

        def reveal_credentials(self) -> tuple[str | None, str | None]:
            return self._credentials_provider()  # type: ignore[no-any-return,operator]

    monkeypatch.setattr(SecretStr, "get_secret_value", reveal)
    monkeypatch.setattr(composition.httpx, "Client", Client)
    monkeypatch.setattr(
        composition,
        "build_ffmpeg_factory",
        lambda *_args: lambda _cancel: object(),
    )
    monkeypatch.setattr(composition, "LazyBaiduOcrClient", LazyBaidu)
    diagnostics = build_production_diagnostic_components(settings)
    try:
        assert revealed == []
        diagnostics.qwen_client._post_with_retry({})  # type: ignore[attr-defined]
        assert revealed == ["  qwen-secret  "]
        assert authorizations == ["Bearer qwen-secret"]
        lazy_baidu = diagnostics.visual_component_factory._ocr_client  # type: ignore[attr-defined]
        assert lazy_baidu.reveal_credentials() == (
            "baidu-api-secret",
            "baidu-secret-secret",
        )
        assert revealed == [
            "  qwen-secret  ",
            "baidu-api-secret",
            "baidu-secret-secret",
        ]
        cloud_configuration = diagnostics.speech_models.recognizer._configuration  # type: ignore[attr-defined]
        assert cloud_configuration.api_key.get_secret_value() == "test-openai-key"
        assert revealed == [
            "  qwen-secret  ",
            "baidu-api-secret",
            "baidu-secret-secret",
            "test-openai-key",
        ]
    finally:
        diagnostics.close()


def test_diagnostic_components_close_resources_in_reverse_order_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cloud_asr_environment: None,
) -> None:
    import video_demo.application.composition as composition

    closes: list[int] = []
    created = 0

    class Client:
        def __init__(self) -> None:
            nonlocal created
            created += 1
            self.identity = created

        def close(self) -> None:
            closes.append(self.identity)

    monkeypatch.setattr(composition.httpx, "Client", Client)
    settings = Settings(workspace_root=tmp_path)
    assert settings.runtime_root is not None
    settings.runtime_root.mkdir(parents=True)

    diagnostics = build_production_diagnostic_components(settings)

    assert isinstance(diagnostics, ProductionDiagnosticComponents)
    assert diagnostics.baidu_http_client is diagnostics.visual_component_factory.http_client
    assert diagnostics.baidu_ocr_client is diagnostics.visual_component_factory.ocr_client
    assert diagnostics.speech_models.vad is not None
    assert diagnostics.speech_models.recognizer is not None
    diagnostics.close()
    diagnostics.close()

    assert closes == [3, 2, 1]


@pytest.mark.parametrize(
    ("failure", "expected_closes"),
    [
        ("qwen_http", [2, 1]),
        ("qwen_adapter", [3, 2, 1]),
        ("speech_factory", [3, 2, 1]),
        ("identity_report", [3, 2, 1]),
        ("owner", [3, 2, 1]),
    ],
)
def test_diagnostic_builder_closes_created_resources_in_reverse_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
    expected_closes: list[int],
    cloud_asr_environment: None,
) -> None:
    import video_demo.application.composition as composition

    closes: list[int] = []
    created = 0

    class Client:
        def __init__(self) -> None:
            nonlocal created
            created += 1
            self.identity = created
            if failure == "qwen_http" and created == 3:
                raise RuntimeError("qwen http construction failed")

        def close(self) -> None:
            closes.append(self.identity)

    def fail(message: str) -> object:
        raise RuntimeError(message)

    monkeypatch.setattr(composition.httpx, "Client", Client)
    if failure == "qwen_adapter":
        monkeypatch.setattr(
            composition,
            "QwenVideoClient",
            lambda *_args, **_kwargs: fail("qwen adapter construction failed"),
        )
    elif failure == "speech_factory":
        monkeypatch.setattr(
            composition,
            "_build_speech_component_factory",
            lambda *_args, **_kwargs: fail("speech factory construction failed"),
        )
    elif failure == "identity_report":
        monkeypatch.setattr(
            composition,
            "build_production_model_identity_report",
            lambda *_args: fail("identity report construction failed"),
        )
    elif failure == "owner":
        monkeypatch.setattr(
            composition,
            "ProductionDiagnosticComponents",
            lambda **_kwargs: fail("diagnostic owner construction failed"),
        )

    settings = Settings(workspace_root=tmp_path)
    assert settings.runtime_root is not None
    settings.runtime_root.mkdir(parents=True)

    with pytest.raises(RuntimeError, match="construction failed"):
        composition.build_production_diagnostic_components(settings)

    assert closes == expected_closes
