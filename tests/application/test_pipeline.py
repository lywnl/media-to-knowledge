from __future__ import annotations

import hashlib
import threading
from pathlib import Path

import pytest
from pydantic import ValidationError

import video_demo.application.pipeline as pipeline_module
from video_demo.application.pipeline import (
    PipelineContext,
    PipelineRunConfig,
    PreparedMedia,
    ProbedAsset,
    RegisteredAsset,
    SpeechAnalysis,
    VideoUnderstandingPipeline,
    VisualAnalysis,
    VisualPreparation,
    pipeline_run_config_from_snapshot,
)
from video_demo.domain.evidence import (
    BoundingBox,
    KeyframeEvidence,
    OcrEvidence,
    OcrLine,
    SceneBoundary,
    SpeechSegment,
)
from video_demo.domain.manifest import Rational, VideoAssetManifest, VideoStream
from video_demo.domain.result import SegmentUnderstanding, SummaryUnderstanding
from video_demo.domain.run import RunStatus, TimeRange
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.fusion.merge import BoundaryPoint
from video_demo.integrations.oss import PublishedVideo, PublishedVideoUnderstanding
from video_demo.integrations.video_port import (
    VideoClipInput,
    WholeVideoUnderstanding,
    WholeVideoUnderstandingRequest,
    WholeVideoWindowUnderstanding,
)
from video_demo.media.probe import ProbeLimits


class FakeRegistrar:
    def __init__(self, source: Path, digest: str, calls: list[str]) -> None:
        self._source = source
        self._digest = digest
        self._calls = calls

    def register(self, _context: PipelineContext) -> RegisteredAsset:
        self._calls.append("REGISTER")
        return RegisteredAsset(
            source_path=self._source,
            source_sha256=self._digest,
            object_ref="obj_001",
            source_size_bytes=self._source.stat().st_size,
            source_mime="video/mp4",
            run_relative_root=Path("runs/scope/run_001"),
            config=PipelineRunConfig(language_hints=("en",)),
        )


class FakeProbe:
    def __init__(self, calls: list[str]) -> None:
        self._calls = calls

    def probe(self, asset: RegisteredAsset) -> ProbedAsset:
        self._calls.append("PROBE")
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
                ffprobe_version="ffprobe test",
            ),
            limits=ProbeLimits(),
        )


class FakeTranscoder:
    def __init__(self, clip: Path, calls: list[str], *, audio: bool = True) -> None:
        self._clip = clip
        self._calls = calls
        self._audio = audio

    def transcode(
        self,
        probed: ProbedAsset,
        **_kwargs: object,
    ) -> PreparedMedia:
        self._calls.append("TRANSCODE")
        return PreparedMedia(
            source=probed,
            proxy_path=self._clip,
            proxy_sha256=hashlib.sha256(self._clip.read_bytes()).hexdigest(),
            proxy_size_bytes=self._clip.stat().st_size,
            audio_path=self._clip.with_suffix(".wav") if self._audio else None,
            audio_sha256="b" * 64 if self._audio else None,
        )


class FakeSpeech:
    def __init__(self, calls: list[str], *, has_speech: bool = True) -> None:
        self._calls = calls
        self._has_speech = has_speech

    def analyze(
        self,
        media: PreparedMedia,
        **_kwargs: object,
    ) -> SpeechAnalysis:
        self._calls.append("SPEECH")
        if media.audio_path is None:
            return SpeechAnalysis(transcript_source="NONE", warnings=("NO_AUDIO_STREAM",))
        if not self._has_speech:
            return SpeechAnalysis(transcript_source="ASR", warnings=("NO_SPEECH_DETECTED",))
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


class FakeVisual:
    def __init__(self, clip: Path, calls: list[str]) -> None:
        self._clip = clip
        self._calls = calls

    def prepare(
        self,
        media: PreparedMedia,
        **_kwargs: object,
    ) -> VisualPreparation:
        scene = SceneBoundary(
            evidence_id="scene_001",
            start_ms=0,
            end_ms=media.source.duration_ms,
            transition="candidate",
            score=0.8,
        )
        return VisualPreparation(
            proxy_sha256=media.proxy_sha256,
            proxy_size_bytes=media.proxy_size_bytes,
            run_relative_root=media.source.asset.run_relative_root,
            duration_ms=media.source.duration_ms,
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
        self._calls.append("VISUAL")
        return VisualAnalysis(
            evidence=preparation.scenes,
            windows=(TimeRange(start_ms=0, end_ms=media.source.duration_ms),),
            boundaries=(
                BoundaryPoint(timestamp_ms=0, sources=("video_start",)),
                BoundaryPoint(timestamp_ms=media.source.duration_ms, sources=("video_end",)),
            ),
        )


class FakeUnderstanding:
    def __init__(self, calls: list[str], *, fail_segment: bool = False) -> None:
        self._calls = calls
        self._fail_segment = fail_segment

    def understand_video(
        self,
        request: WholeVideoUnderstandingRequest,
    ) -> WholeVideoUnderstanding:
        self._calls.append("UNDERSTANDING")
        if self._fail_segment:
            raise VideoDemoError(ErrorCode.QWEN_RESPONSE_INVALID, "模拟全片失败")
        return WholeVideoUnderstanding(
            windows=tuple(
                WholeVideoWindowUnderstanding(
                    window_id=window.window_id,
                    understanding=SegmentUnderstanding(
                        title="问候",
                        summary_zh="视频包含问候。",
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
                summary_zh="视频包含一段问候。",
                languages=("en",),
                topics=("问候",),
                keywords=("问候",),
                original_keywords=("Hello",),
            ),
        )


def _pipeline(
    tmp_path: Path,
    *,
    audio: bool = True,
    has_speech: bool = True,
    fail_segment: bool = False,
) -> tuple[VideoUnderstandingPipeline, PipelineContext, list[str]]:
    calls: list[str] = []
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"video")
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    pipeline = VideoUnderstandingPipeline(
        FakeRegistrar(source, hashlib.sha256(b"source").hexdigest(), calls),
        FakeProbe(calls),
        FakeTranscoder(clip, calls, audio=audio),
        FakeSpeech(calls, has_speech=has_speech),
        FakeVisual(clip, calls),
        FakeUnderstanding(calls, fail_segment=fail_segment),
    )
    return pipeline, PipelineContext(run_id="run_001"), calls


def test_pipeline_runs_all_stages_and_returns_metrics(tmp_path: Path) -> None:
    pipeline, context, calls = _pipeline(tmp_path)

    outcome = pipeline.run(context)

    assert calls[:3] == ["REGISTER", "PROBE", "TRANSCODE"]
    assert set(calls[3:5]) == {"SPEECH", "VISUAL"}
    assert calls[5:] == ["UNDERSTANDING"]
    assert outcome.status == RunStatus.SUCCEEDED
    assert outcome.result.segments[0].retrieval_text.startswith("文档类型：VIDEO_SEGMENT")
    assert [metric.stage for metric in outcome.stage_metrics] == [
        "REGISTER",
        "PROBE",
        "TRANSCODE",
        "SPEECH",
        "VISUAL_WAIT_SPEECH",
        "VISUAL_FUSION",
        "VISUAL",
        "FUSION",
        "UNDERSTANDING",
        "RESULT",
    ]
    assert all(metric.duration_ms >= 0 for metric in outcome.stage_metrics)


def test_pipeline_rejects_oversized_full_video_before_understanding(
    tmp_path: Path,
) -> None:
    pipeline, context, calls = _pipeline(tmp_path)
    original_transcoder = pipeline._transcoder  # type: ignore[attr-defined]

    class OversizedTranscoder:
        def transcode(
            self,
            probed: ProbedAsset,
            **kwargs: object,
        ) -> PreparedMedia:
            media = original_transcoder.transcode(probed, **kwargs)
            return PreparedMedia(
                source=media.source,
                proxy_path=media.proxy_path,
                proxy_sha256=media.proxy_sha256,
                proxy_size_bytes=128 * 1024 * 1024 + 1,
                audio_path=media.audio_path,
                audio_sha256=media.audio_sha256,
                warnings=media.warnings,
            )

    pipeline._transcoder = OversizedTranscoder()  # type: ignore[attr-defined]

    with pytest.raises(VideoDemoError) as raised:
        pipeline.run(context)

    assert raised.value.code == ErrorCode.QWEN_CAPABILITY_UNAVAILABLE
    assert "UNDERSTANDING" not in calls


def test_pipeline_fuses_asr_ocr_keyframe_scene_with_published_qwen_result(
    tmp_path: Path,
) -> None:
    pipeline, context, calls = _pipeline(tmp_path)
    clip = tmp_path / "clip.mp4"
    clip_sha256 = hashlib.sha256(clip.read_bytes()).hexdigest()
    published: list[VideoClipInput] = []
    qwen_requests: list[WholeVideoUnderstandingRequest] = []

    class Visual(FakeVisual):
        def finalize(
            self,
            media: PreparedMedia,
            preparation: VisualPreparation,
            **_kwargs: object,
        ) -> VisualAnalysis:
            self._calls.append("VISUAL")
            keyframe = KeyframeEvidence(
                evidence_id="keyframe_001",
                start_ms=0,
                end_ms=1_000,
                keyframe_id="frame_001",
                timestamp_ms=500,
                relative_path="runs/scope/run_001/visual/keyframes/frame.jpg",
                mime_type="image/jpeg",
                sha256="c" * 64,
                perceptual_hash="abcdef12",
                size_bytes=1,
            )
            ocr = OcrEvidence(
                evidence_id="ocr_001",
                start_ms=0,
                end_ms=1_000,
                keyframe_id=keyframe.keyframe_id,
                timestamp_ms=500,
                language="en",
                lines=(
                    OcrLine(
                        text="ChatGPT",
                        bounding_box=BoundingBox(x=10, y=10, width=100, height=20),
                        confidence=0.95,
                    ),
                ),
                provider_request_id="ocr-request-001",
            )
            return VisualAnalysis(
                evidence=(*preparation.scenes, keyframe, ocr),
                windows=(TimeRange(start_ms=0, end_ms=media.source.duration_ms),),
                boundaries=(
                    BoundaryPoint(timestamp_ms=0, sources=("video_start",)),
                    BoundaryPoint(timestamp_ms=1_000, sources=("video_end",)),
                ),
            )

    class Publisher:
        @property
        def remote_host(self) -> str:
            return "private-video-bucket.oss-cn-hangzhou.aliyuncs.com"

        def publish(self, local_clip: VideoClipInput) -> PublishedVideo:
            published.append(local_clip)
            return PublishedVideo(
                published_clip=VideoClipInput(
                    clip_id=local_clip.clip_id,
                    start_ms=local_clip.start_ms,
                    end_ms=local_clip.end_ms,
                    source_url=(
                        "https://private-video-bucket.oss-cn-hangzhou.aliyuncs.com/"
                        "video-demo/qwen-clips/clip.mp4?Signature=redacted"
                    ),
                    mime_type=local_clip.mime_type,
                    sha256=local_clip.sha256,
                ),
                object_key="video-demo/qwen-clips/owner/publish-clip.mp4",
            )

        def delete(self, _object_key: str) -> None:
            return None

    class Qwen:
        def understand_video(
            self,
            request: WholeVideoUnderstandingRequest,
        ) -> WholeVideoUnderstanding:
            qwen_requests.append(request)
            assert request.video.path is None
            assert request.video.source_url is not None
            return WholeVideoUnderstanding(
                windows=tuple(
                    WholeVideoWindowUnderstanding(
                        window_id=window.window_id,
                        understanding=SegmentUnderstanding(
                            title="ChatGPT 操作演示",
                            summary_zh="画面显示 ChatGPT 界面，语音提到上传图片。",
                            languages=("en",),
                            topics=("AI 工具",),
                            entities=("ChatGPT",),
                            actions=("上传图片",),
                            keywords=("ChatGPT",),
                            original_keywords=("ChatGPT",),
                            evidence_refs=tuple(
                                item.evidence_id for item in window.evidence
                            ),
                        ),
                    )
                    for window in request.windows
                ),
                summary=SummaryUnderstanding(
                    title="AI 工具演示",
                    summary_zh="视频演示了 ChatGPT 图片操作。",
                    languages=("en",),
                    topics=("AI 工具",),
                    entities=("ChatGPT",),
                    actions=("上传图片",),
                    keywords=("ChatGPT",),
                    original_keywords=("ChatGPT",),
                ),
            )

    pipeline._visual_analyzer = Visual(clip, calls)  # type: ignore[attr-defined]
    pipeline._understanding = PublishedVideoUnderstanding(  # type: ignore[attr-defined]
        Qwen(),
        Publisher(),
    )

    outcome = pipeline.run(context)

    assert outcome.status == RunStatus.SUCCEEDED
    assert len(published) == 1
    assert published[0].path == clip
    assert published[0].start_ms == 0
    assert published[0].end_ms == 1_000
    assert qwen_requests[0].video.sha256 == clip_sha256
    assert set(outcome.result.segments[0].evidence_refs) == {
        "asr_001",
        "scene_001",
        "keyframe_001",
        "ocr_001",
    }
    assert "ChatGPT" in outcome.result.segments[0].retrieval_text
    for item in (*outcome.result.segments, outcome.result.summary):
        assert item.retrieval_hash == hashlib.sha256(
            item.retrieval_text.encode("utf-8"),
        ).hexdigest()
    assert "Signature=redacted" not in outcome.result.model_dump_json()


@pytest.mark.parametrize(
    ("audio", "has_speech", "warning"),
    [(False, True, "NO_AUDIO_STREAM"), (True, False, "NO_SPEECH_DETECTED")],
)
def test_no_audio_or_no_speech_keeps_visual_pipeline_running(
    tmp_path: Path,
    audio: bool,
    has_speech: bool,
    warning: str,
) -> None:
    pipeline, context, _calls = _pipeline(tmp_path, audio=audio, has_speech=has_speech)

    outcome = pipeline.run(context)

    assert warning in outcome.warnings
    assert outcome.result.segments


def test_speech_and_visual_branches_really_run_in_parallel(tmp_path: Path) -> None:
    pipeline, context, _calls = _pipeline(tmp_path)
    speech_started = threading.Event()
    visual_started = threading.Event()

    class Speech(FakeSpeech):
        def analyze(self, media: PreparedMedia, **kwargs: object) -> SpeechAnalysis:
            speech_started.set()
            assert visual_started.wait(timeout=1)
            return super().analyze(media, **kwargs)

    class Visual(FakeVisual):
        def prepare(self, media: PreparedMedia, **kwargs: object) -> VisualPreparation:
            visual_started.set()
            assert speech_started.wait(timeout=1)
            return super().prepare(media, **kwargs)

    pipeline._speech_analyzer = Speech(_calls)  # type: ignore[attr-defined]
    pipeline._visual_analyzer = Visual(tmp_path / "clip.mp4", _calls)  # type: ignore[attr-defined]

    assert pipeline.run(context).status == RunStatus.SUCCEEDED


def test_scene_prepare_runs_in_parallel_with_speech_and_finalize_receives_real_speech(
    tmp_path: Path,
) -> None:
    pipeline, context, calls = _pipeline(tmp_path)
    speech_started = threading.Event()
    prepare_started = threading.Event()
    speech_finished = threading.Event()
    prepare_finished = threading.Event()
    expected_boundary = pipeline_module.SpeechBoundaryCandidate(
        timestamp_ms=500,
        source="sentence_end",
    )
    preparation = pipeline_module.VisualPreparation(
        proxy_sha256="a" * 64,
        proxy_size_bytes=5,
        run_relative_root=Path("runs/scope/run_001"),
        duration_ms=1_000,
        frame_tolerance_ms=40,
        scenes=(
            SceneBoundary(
                evidence_id="scene_prepare_001",
                start_ms=0,
                end_ms=1_000,
                transition="candidate",
                score=0.8,
            ),
        ),
        preparation_sha256="b" * 64,
    )

    class Speech:
        def analyze(
            self,
            _media: PreparedMedia,
            **_kwargs: object,
        ) -> SpeechAnalysis:
            speech_started.set()
            assert prepare_started.wait(timeout=1), "visual.prepare 未与语音并行进入"
            speech_finished.set()
            return SpeechAnalysis(
                transcript_source="ASR",
                boundary_candidates=(expected_boundary,),
            )

    class Visual:
        def prepare(
            self,
            _media: PreparedMedia,
            **_kwargs: object,
        ) -> object:
            prepare_started.set()
            assert speech_started.wait(timeout=1), "语音未与 visual.prepare 并行进入"
            prepare_finished.set()
            return preparation

        def finalize(
            self,
            media: PreparedMedia,
            actual_preparation: object,
            **kwargs: object,
        ) -> VisualAnalysis:
            assert speech_finished.is_set()
            assert prepare_finished.is_set()
            assert actual_preparation is preparation
            speech = kwargs["speech"]
            assert isinstance(speech, SpeechAnalysis)
            assert speech.boundary_candidates == (expected_boundary,)
            return FakeVisual(tmp_path / "clip.mp4", calls).finalize(
                media,
                preparation,
            )

        def analyze(self, _media: PreparedMedia, **_kwargs: object) -> VisualAnalysis:
            raise AssertionError("完整流水线不得调用旧 visual.analyze 接口")

    pipeline._speech_analyzer = Speech()  # type: ignore[attr-defined]
    pipeline._visual_analyzer = Visual()  # type: ignore[attr-defined]

    assert pipeline.run(context).status == RunStatus.SUCCEEDED


def test_pipeline_propagates_same_cancellation_callback_to_media_and_visual_phases(
    tmp_path: Path,
) -> None:
    pipeline, _context, calls = _pipeline(tmp_path)
    received: list[tuple[str, object]] = []

    def cancel() -> bool:
        return False

    class Transcoder(FakeTranscoder):
        def transcode(
            self,
            probed: ProbedAsset,
            **kwargs: object,
        ) -> PreparedMedia:
            received.append(("TRANSCODE", kwargs["is_cancel_requested"]))
            return super().transcode(probed, **kwargs)

    class Speech(FakeSpeech):
        def analyze(
            self,
            media: PreparedMedia,
            **kwargs: object,
        ) -> SpeechAnalysis:
            received.append(("SPEECH", kwargs["is_cancel_requested"]))
            return super().analyze(media, **kwargs)

    class Visual(FakeVisual):
        def prepare(
            self,
            media: PreparedMedia,
            **kwargs: object,
        ) -> VisualPreparation:
            received.append(("VISUAL_PREPARE", kwargs["is_cancel_requested"]))
            return super().prepare(media, **kwargs)

        def finalize(
            self,
            media: PreparedMedia,
            preparation: VisualPreparation,
            **kwargs: object,
        ) -> VisualAnalysis:
            received.append(("VISUAL_FINALIZE", kwargs["is_cancel_requested"]))
            return super().finalize(media, preparation, **kwargs)

    clip = tmp_path / "clip.mp4"
    pipeline._transcoder = Transcoder(clip, calls)  # type: ignore[attr-defined]
    pipeline._speech_analyzer = Speech(calls)  # type: ignore[attr-defined]
    pipeline._visual_analyzer = Visual(clip, calls)  # type: ignore[attr-defined]

    outcome = pipeline.run(
        PipelineContext(run_id="run_001", is_cancel_requested=cancel),
    )

    assert outcome.status == RunStatus.SUCCEEDED
    assert {phase for phase, _callback in received} == {
        "TRANSCODE",
        "SPEECH",
        "VISUAL_PREPARE",
        "VISUAL_FINALIZE",
    }
    assert all(callback is cancel for _phase, callback in received)


def test_cancellation_before_parallel_submission_starts_no_audio_visual_work(
    tmp_path: Path,
) -> None:
    pipeline, _context, _calls = _pipeline(tmp_path)
    cancelled = False
    branch_calls: list[str] = []

    def cancel() -> bool:
        return cancelled

    def on_stage_start(stage: str) -> None:
        nonlocal cancelled
        if stage == "VISUAL":
            cancelled = True

    class Speech:
        def analyze(self, _media: PreparedMedia, **_kwargs: object) -> SpeechAnalysis:
            branch_calls.append("SPEECH")
            return SpeechAnalysis(transcript_source="NONE")

    class Visual:
        def prepare(
            self,
            _media: PreparedMedia,
            **_kwargs: object,
        ) -> VisualPreparation:
            branch_calls.append("VISUAL_PREPARE")
            raise AssertionError("取消后不得提交视觉准备")

    pipeline._speech_analyzer = Speech()  # type: ignore[attr-defined]
    pipeline._visual_analyzer = Visual()  # type: ignore[attr-defined]

    with pytest.raises(VideoDemoError) as raised:
        pipeline.run(
            PipelineContext(
                run_id="run_001",
                is_cancel_requested=cancel,
                on_stage_start=on_stage_start,
            ),
        )

    assert raised.value.code == ErrorCode.JOB_CANCELLED
    assert branch_calls == []


def test_cancellation_after_parallel_join_prevents_visual_finalize(
    tmp_path: Path,
) -> None:
    pipeline, _context, _calls = _pipeline(tmp_path)
    speech_finished = threading.Event()
    prepare_finished = threading.Event()
    finalize_calls: list[str] = []

    def cancel() -> bool:
        return speech_finished.is_set() and prepare_finished.is_set()

    class Speech:
        def analyze(self, _media: PreparedMedia, **_kwargs: object) -> SpeechAnalysis:
            speech_finished.set()
            return SpeechAnalysis(transcript_source="NONE")

    class Visual(FakeVisual):
        def prepare(
            self,
            media: PreparedMedia,
            **kwargs: object,
        ) -> VisualPreparation:
            preparation = super().prepare(media, **kwargs)
            prepare_finished.set()
            return preparation

        def finalize(
            self,
            media: PreparedMedia,
            preparation: VisualPreparation,
            **kwargs: object,
        ) -> VisualAnalysis:
            finalize_calls.append("VISUAL_FINALIZE")
            return super().finalize(media, preparation, **kwargs)

    pipeline._speech_analyzer = Speech()  # type: ignore[attr-defined]
    pipeline._visual_analyzer = Visual(tmp_path / "clip.mp4", [])  # type: ignore[attr-defined]

    with pytest.raises(VideoDemoError) as raised:
        pipeline.run(
            PipelineContext(run_id="run_001", is_cancel_requested=cancel),
        )

    assert raised.value.code == ErrorCode.JOB_CANCELLED
    assert finalize_calls == []


def test_visual_metric_covers_prepare_through_finalize_while_speech_is_independent(
    tmp_path: Path,
) -> None:
    pipeline, context, calls = _pipeline(tmp_path)
    prepared = threading.Event()
    speech_finished = threading.Event()
    finalized = threading.Event()

    def clock() -> float:
        if threading.current_thread().name.startswith("video-understanding"):
            return 0.025 if speech_finished.is_set() else 0.0
        if finalized.is_set():
            return 0.180
        if prepared.is_set():
            return 0.150
        return 0.100

    class Speech(FakeSpeech):
        def analyze(self, media: PreparedMedia, **kwargs: object) -> SpeechAnalysis:
            result = super().analyze(media, **kwargs)
            speech_finished.set()
            return result

    class Visual(FakeVisual):
        def prepare(
            self,
            media: PreparedMedia,
            **kwargs: object,
        ) -> VisualPreparation:
            result = super().prepare(media, **kwargs)
            prepared.set()
            return result

        def finalize(
            self,
            media: PreparedMedia,
            preparation: VisualPreparation,
            **kwargs: object,
        ) -> VisualAnalysis:
            result = super().finalize(media, preparation, **kwargs)
            finalized.set()
            return result

    pipeline._clock = clock  # type: ignore[attr-defined]
    pipeline._speech_analyzer = Speech(calls)  # type: ignore[attr-defined]
    pipeline._visual_analyzer = Visual(tmp_path / "clip.mp4", calls)  # type: ignore[attr-defined]

    outcome = pipeline.run(context)
    metrics = {item.stage: item.duration_ms for item in outcome.stage_metrics}

    assert metrics["SPEECH"] == 25
    assert metrics["VISUAL"] == 80


def test_all_understanding_windows_failed_closes_pipeline(tmp_path: Path) -> None:
    pipeline, context, _calls = _pipeline(tmp_path, fail_segment=True)

    with pytest.raises(VideoDemoError) as raised:
        pipeline.run(context)

    assert raised.value.code == ErrorCode.QWEN_RESPONSE_INVALID


def test_all_temporary_window_failures_preserve_retryable_error(tmp_path: Path) -> None:
    pipeline, context, calls = _pipeline(tmp_path)

    class TemporarilyUnavailableUnderstanding(FakeUnderstanding):
        def understand_video(
            self,
            request: WholeVideoUnderstandingRequest,
        ) -> WholeVideoUnderstanding:
            raise VideoDemoError(ErrorCode.DEPENDENCY_TEMPORARY_FAILURE, "服务暂时不可用")

    pipeline._understanding = TemporarilyUnavailableUnderstanding(  # type: ignore[attr-defined]
        calls,
    )

    with pytest.raises(VideoDemoError) as raised:
        pipeline.run(context)

    assert raised.value.code == ErrorCode.DEPENDENCY_TEMPORARY_FAILURE


def test_global_understanding_error_is_not_downgraded_to_window_failure(
    tmp_path: Path,
) -> None:
    pipeline, context, _calls = _pipeline(tmp_path)

    class UnauthorizedUnderstanding(FakeUnderstanding):
        def understand_video(
            self,
            request: WholeVideoUnderstandingRequest,
        ) -> WholeVideoUnderstanding:
            raise VideoDemoError(ErrorCode.QWEN_AUTHENTICATION_FAILED, "鉴权失败")

    pipeline._understanding = UnauthorizedUnderstanding(_calls)  # type: ignore[attr-defined]

    with pytest.raises(VideoDemoError) as raised:
        pipeline.run(context)

    assert raised.value.code == ErrorCode.QWEN_AUTHENTICATION_FAILED


def test_incomplete_whole_video_window_set_fails_the_entire_pipeline(
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"video")
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    class Visual:
        def prepare(
            self,
            media: PreparedMedia,
            **_kwargs: object,
        ) -> VisualPreparation:
            scenes = (
                SceneBoundary(
                    evidence_id="scene_001",
                    start_ms=0,
                    end_ms=500,
                    transition="candidate",
                    score=0.8,
                ),
                SceneBoundary(
                    evidence_id="scene_002",
                    start_ms=500,
                    end_ms=1_000,
                    transition="candidate",
                    score=0.8,
                ),
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
            calls.append("VISUAL")
            return VisualAnalysis(
                evidence=preparation.scenes,
                windows=(
                    TimeRange(start_ms=0, end_ms=500),
                    TimeRange(start_ms=500, end_ms=1_000),
                ),
                boundaries=(
                    BoundaryPoint(timestamp_ms=0, sources=("video_start",)),
                    BoundaryPoint(timestamp_ms=500, sources=("scene_hard",)),
                    BoundaryPoint(timestamp_ms=1_000, sources=("video_end",)),
                ),
            )

    class Understanding(FakeUnderstanding):
        def understand_video(
            self,
            request: WholeVideoUnderstandingRequest,
        ) -> WholeVideoUnderstanding:
            calls.append("UNDERSTANDING")
            second = request.windows[1]
            return WholeVideoUnderstanding(
                windows=(
                    WholeVideoWindowUnderstanding(
                        window_id=second.window_id,
                        understanding=SegmentUnderstanding(
                            title="第二部分",
                            summary_zh="第二个窗口成功。",
                            languages=("en",),
                            topics=("演示",),
                            keywords=("成功",),
                            original_keywords=("success",),
                            evidence_refs=("scene_002",),
                        ),
                    ),
                ),
                summary=SummaryUnderstanding(title="测试视频", summary_zh="摘要。"),
            )

    pipeline = VideoUnderstandingPipeline(
        FakeRegistrar(source, hashlib.sha256(b"source").hexdigest(), calls),
        FakeProbe(calls),
        FakeTranscoder(clip, calls),
        FakeSpeech(calls, has_speech=False),
        Visual(),
        Understanding(calls),
    )

    with pytest.raises(VideoDemoError) as raised:
        pipeline.run(PipelineContext(run_id="run_001"))

    assert raised.value.code == ErrorCode.QWEN_RESPONSE_INVALID
@pytest.mark.parametrize(
    "retired_fields",
    [
        {"speech_enrichment_mode": "text"},
        {"speech_enrichment_mode": "full"},
        {"min_speakers": 1, "max_speakers": 2},
        {"min_speakers": None, "max_speakers": None},
    ],
)
def test_pipeline_run_config_loads_retired_snapshot_fields_without_mutation(
    retired_fields: dict[str, object],
) -> None:
    snapshot = {
        "language_hints": ["zh"],
        "hotwords": ["Milvus"],
        "core_context": "向量数据库课程",
        **retired_fields,
    }
    original = snapshot.copy()

    config = pipeline_run_config_from_snapshot(snapshot)

    assert config == PipelineRunConfig(
        language_hints=("zh",),
        hotwords=("Milvus",),
        core_context="向量数据库课程",
    )
    assert snapshot == original


def test_pipeline_run_config_rejects_unknown_historical_snapshot_field() -> None:
    with pytest.raises(ValidationError):
        pipeline_run_config_from_snapshot(
            {
                "language_hints": [],
                "hotwords": [],
                "core_context": None,
                "unexpected": True,
            }
        )
