from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path

import httpx
import pytest

from video_demo.application.pipeline import (
    PipelineRunConfig,
    PreparedMedia,
    ProbedAsset,
    RegisteredAsset,
    SpeechAnalysis,
    SpeechBoundaryCandidate,
    VisualPreparation,
)
from video_demo.application.production_visual import (
    LazyBaiduOcrClient,
    ProductionVisualAnalyzer,
    VisualComponents,
    WindowFrameCandidates,
    _limit_keyframes,
    frame_tolerance_ms_for_rate,
)
from video_demo.domain.evidence import (
    BoundingBox,
    KeyframeEvidence,
    OcrLine,
    SceneBoundary,
    SpeechSegment,
)
from video_demo.domain.manifest import Rational, VideoAssetManifest, VideoStream
from video_demo.domain.run import TimeRange
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.media.probe import ProbeLimits
from video_demo.media.transcode import ClipArtifact
from video_demo.visual.keyframes import FrameCandidate, KeyframeSelector
from video_demo.visual.ocr import OcrProviderResponse

_JPEG_PREFIX = b"\xff\xd8\xff\xe0"
_JPEG_SUFFIX = b"\xff\xd9"
_MP4 = b"\x00\x00\x00\x18ftypisom\x00\x02\x00\x00isomiso2"


@pytest.mark.parametrize(
    ("frame_rate", "expected_ms"),
    [
        (Rational(numerator=25, denominator=1), 40),
        (Rational(numerator=60, denominator=1), 17),
        (Rational(numerator=5, denominator=1), 100),
        (Rational(numerator=0, denominator=1), 100),
        (None, 100),
    ],
)
def test_frame_tolerance_uses_rational_rate_with_conservative_fallback(
    frame_rate: Rational | None,
    expected_ms: int,
) -> None:
    assert frame_tolerance_ms_for_rate(frame_rate) == expected_ms


def test_frame_tolerance_uses_task_cap_for_variable_frame_rate() -> None:
    assert frame_tolerance_ms_for_rate(
        Rational(numerator=1_200, denominator=59),
        is_variable_frame_rate=True,
    ) == 100


def test_visual_preparation_uses_task_cap_for_vfr_manifest(tmp_path: Path) -> None:
    media = _media(tmp_path, duration_ms=2_000, is_variable_frame_rate=True)

    class Scenes:
        def detect(
            self,
            _proxy: Path,
            *,
            duration_ms: int,
            source_sha256: str,
            frame_tolerance_ms: int,
        ) -> tuple[SceneBoundary, ...]:
            assert len(source_sha256) == 64
            assert frame_tolerance_ms == 100
            return (_scene(0, duration_ms, transition="candidate"),)

    analyzer = ProductionVisualAnalyzer(
        tmp_path,
        lambda _media, _cancel: VisualComponents(
            scene_detector=Scenes(),
            frame_extractor=object(),  # type: ignore[arg-type]
            keyframe_selector=KeyframeSelector(),
            ocr_client=object(),  # type: ignore[arg-type]
            clip_client=object(),  # type: ignore[arg-type]
        ),
    )

    preparation = analyzer.prepare(media)

    assert preparation.frame_tolerance_ms == 100


def test_complete_visual_chain_returns_merged_windows_without_creating_clips(
    tmp_path: Path,
) -> None:
    media = _media(tmp_path, duration_ms=20_000)
    calls: list[str] = []

    class Scenes:
        def detect(
            self,
            proxy: Path,
            *,
            duration_ms: int,
            source_sha256: str,
            frame_tolerance_ms: int,
        ) -> tuple[SceneBoundary, ...]:
            calls.append("scene")
            assert proxy == media.proxy_path
            assert source_sha256 == media.proxy_sha256
            assert frame_tolerance_ms == 40
            return (_scene(0, duration_ms, transition="candidate"),)

    class Frames:
        def extract(
            self,
            proxy: Path,
            run_relative_root: Path,
            windows: Sequence[TimeRange],
            *,
            is_cancel_requested: Callable[[], bool],
            frame_tolerance_ms: int,
        ) -> tuple[WindowFrameCandidates, ...]:
            calls.append("frames")
            assert len(windows) == 1
            assert frame_tolerance_ms == 40
            return (
                WindowFrameCandidates(
                    window=windows[0],
                    candidates=(
                        _write_frame(
                            tmp_path,
                            run_relative_root,
                            4_000,
                            b"page-one",
                            perceptual_hash="0000000000000000",
                        ),
                        _write_frame(
                            tmp_path,
                            run_relative_root,
                            12_000,
                            b"page-two",
                            perceptual_hash="ffffffffffffffff",
                        ),
                    ),
                ),
            )

    class Ocr:
        def recognize(self, image: bytes, language: str) -> OcrProviderResponse:
            calls.append(f"ocr:{language}")
            text = "第一页课程标题" if b"page-one" in image else "第二页完全不同内容"
            return _ocr(text)

    clips = _Clips(tmp_path, calls)
    analyzer = ProductionVisualAnalyzer(
        tmp_path,
        lambda _media, _cancel: VisualComponents(
            scene_detector=Scenes(),
            frame_extractor=Frames(),
            keyframe_selector=KeyframeSelector(),
            ocr_client=Ocr(),
            clip_client=clips,
        ),
    )

    result = analyzer.analyze(media, speech=SpeechAnalysis())

    assert calls == ["scene", "frames", "ocr:zh", "ocr:zh"]
    assert result.clips == ()
    assert [(item.start_ms, item.end_ms) for item in result.windows] == [(0, 20_000)]
    keyframes = [item for item in result.evidence if item.evidence_type == "KEYFRAME"]
    ocr = [item for item in result.evidence if item.evidence_type == "OCR"]
    assert [(item.start_ms, item.end_ms, item.timestamp_ms) for item in keyframes] == [
        (0, 12_000, 4_000),
        (12_000, 20_000, 12_000),
    ]
    assert [(item.start_ms, item.end_ms) for item in ocr] == [(0, 12_000), (12_000, 20_000)]
    assert keyframes[0].evidence_id.startswith("keyframe_evidence_")
    assert keyframes[0].keyframe_id.startswith("keyframe_")
    assert "OCR_LANGUAGE_FALLBACK:zh" in result.warnings
    boundary = next(item for item in result.boundaries if item.timestamp_ms == 12_000)
    assert set(boundary.sources) == {"clip_edge", "ocr_change"}


def test_visual_chain_evenly_limits_keyframes_and_ocr_to_thirty(tmp_path: Path) -> None:
    duration_ms = 198_000
    media = _media(tmp_path, duration_ms=duration_ms)
    ocr_calls: list[str] = []

    class Frames:
        def extract(
            self,
            _proxy: Path,
            run_relative_root: Path,
            windows: Sequence[TimeRange],
            **_kwargs: object,
        ) -> tuple[WindowFrameCandidates, ...]:
            return tuple(
                WindowFrameCandidates(
                    window=window,
                    candidates=(
                        _write_frame(
                            tmp_path,
                            run_relative_root,
                            window.start_ms + 1_000,
                            f"page-{window.start_ms}".encode(),
                        ),
                    ),
                )
                for window in windows
            )

    class Ocr:
        def recognize(self, _image: bytes, language: str) -> OcrProviderResponse:
            ocr_calls.append(language)
            return _ocr("页面")

    speech = SpeechAnalysis(
        boundary_candidates=tuple(
            SpeechBoundaryCandidate(timestamp_ms=timestamp_ms, source="silence")
            for timestamp_ms in range(2_000, duration_ms, 2_000)
        ),
    )
    result = ProductionVisualAnalyzer(
        tmp_path,
        lambda _media, _cancel: VisualComponents(
            scene_detector=_Scenes(duration_ms),
            frame_extractor=Frames(),
            keyframe_selector=KeyframeSelector(),
            ocr_client=Ocr(),
            clip_client=_Clips(tmp_path, []),
        ),
    ).analyze(media, speech=speech)

    keyframes = [item for item in result.evidence if item.evidence_type == "KEYFRAME"]
    ocr = [item for item in result.evidence if item.evidence_type == "OCR"]
    timestamps = [item.timestamp_ms for item in keyframes]
    assert len(keyframes) == 30
    assert len(ocr) == 30
    assert len(ocr_calls) == 30
    assert timestamps == sorted(set(timestamps))
    assert timestamps[0] == 1_000
    assert timestamps[-1] == 197_000


def test_keyframe_limit_accepts_one_frame(tmp_path: Path) -> None:
    media = _media(tmp_path, duration_ms=4_000)
    windows = (
        TimeRange(start_ms=0, end_ms=1_000),
        TimeRange(start_ms=1_000, end_ms=2_000),
        TimeRange(start_ms=2_000, end_ms=4_000),
    )
    frames = tuple(
        _write_frame(
            tmp_path,
            media.source.asset.run_relative_root,
            timestamp_ms,
            f"page-{timestamp_ms}".encode(),
        )
        for timestamp_ms in (500, 1_500, 3_000)
    )
    selected, _warnings = ProductionVisualAnalyzer(tmp_path, lambda *_args: None)._select_keyframes(
        tuple(
            WindowFrameCandidates(window=window, candidates=(frame,))
            for window, frame in zip(windows, frames, strict=True)
        ),
        windows,
        media.source.asset.run_relative_root,
        media.proxy_sha256,
        KeyframeSelector(),
        lambda: False,
    )

    limited = _limit_keyframes(selected, 1)

    assert len(limited) == 1
    assert limited[0].timestamp_ms == 1_500


def test_keyframe_limit_uses_video_time_instead_of_candidate_density() -> None:
    keyframes = tuple(
        KeyframeEvidence(
            evidence_id=f"keyframe_evidence_{timestamp_ms}",
            start_ms=0,
            end_ms=1_001,
            keyframe_id=f"keyframe_{timestamp_ms}",
            timestamp_ms=timestamp_ms,
            relative_path=f"visual/keyframes/frame_{timestamp_ms}.jpg",
            mime_type="image/jpeg",
            sha256=f"{timestamp_ms:064x}",
            perceptual_hash=f"{timestamp_ms:016x}",
        )
        for timestamp_ms in (*range(91), 500, 1_000)
    )

    limited = _limit_keyframes(keyframes, 3)

    assert [item.timestamp_ms for item in limited] == [0, 500, 1_000]


def test_visual_preparation_rejects_replaced_scenes_before_finalization_components(
    tmp_path: Path,
) -> None:
    media = _media(tmp_path, duration_ms=4_000)
    factory_calls: list[str] = []

    def factory(
        _media: PreparedMedia,
        _cancel: Callable[[], bool],
    ) -> VisualComponents:
        factory_calls.append("factory")
        return VisualComponents(
            scene_detector=_Scenes(4_000),
            frame_extractor=object(),  # type: ignore[arg-type]
            keyframe_selector=KeyframeSelector(),
            ocr_client=object(),  # type: ignore[arg-type]
            clip_client=object(),  # type: ignore[arg-type]
        )

    analyzer = ProductionVisualAnalyzer(tmp_path, factory)
    preparation = analyzer.prepare(media)
    replaced = replace(
        preparation,
        scenes=(
            _scene(0, 2_000, transition="candidate"),
            _scene(2_000, 4_000, transition="hard_cut"),
        ),
    )

    with pytest.raises(VideoDemoError) as raised:
        analyzer.finalize(media, replaced, speech=SpeechAnalysis())

    assert raised.value.code == ErrorCode.VISUAL_RESULT_INVALID
    assert isinstance(preparation, VisualPreparation)
    assert factory_calls == ["factory"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("proxy_sha256", "0" * 64),
        ("proxy_size_bytes", len(_MP4) + 1),
        ("run_relative_root", Path("runs/scope/run_other")),
        ("duration_ms", 3_999),
        ("frame_tolerance_ms", 41),
    ],
)
def test_visual_preparation_binds_every_media_and_tolerance_field(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    media = _media(tmp_path, duration_ms=4_000)
    factory_calls: list[str] = []

    def factory(
        _media: PreparedMedia,
        _cancel: Callable[[], bool],
    ) -> VisualComponents:
        factory_calls.append("factory")
        return VisualComponents(
            scene_detector=_Scenes(4_000),
            frame_extractor=object(),  # type: ignore[arg-type]
            keyframe_selector=KeyframeSelector(),
            ocr_client=object(),  # type: ignore[arg-type]
            clip_client=object(),  # type: ignore[arg-type]
        )

    analyzer = ProductionVisualAnalyzer(tmp_path, factory)
    preparation = analyzer.prepare(media)

    with pytest.raises(VideoDemoError) as raised:
        analyzer.finalize(
            media,
            replace(preparation, **{field: value}),
            speech=SpeechAnalysis(),
        )

    assert raised.value.code == ErrorCode.VISUAL_RESULT_INVALID
    assert factory_calls == ["factory"]


def test_visual_preparation_cannot_be_reused_for_another_run(
    tmp_path: Path,
) -> None:
    first = _media(tmp_path, duration_ms=4_000)
    second_root = Path("runs/scope/run_002")
    second_proxy = tmp_path / second_root / "media/proxy.mp4"
    second_proxy.parent.mkdir(parents=True)
    second_proxy.write_bytes(_MP4)
    second_asset = replace(
        first.source.asset,
        run_relative_root=second_root,
    )
    second = replace(
        first,
        source=replace(first.source, asset=second_asset),
        proxy_path=second_proxy,
    )
    factory_calls: list[str] = []

    def factory(
        _media: PreparedMedia,
        _cancel: Callable[[], bool],
    ) -> VisualComponents:
        factory_calls.append("factory")
        return VisualComponents(
            scene_detector=_Scenes(4_000),
            frame_extractor=object(),  # type: ignore[arg-type]
            keyframe_selector=KeyframeSelector(),
            ocr_client=object(),  # type: ignore[arg-type]
            clip_client=object(),  # type: ignore[arg-type]
        )

    analyzer = ProductionVisualAnalyzer(tmp_path, factory)
    preparation = analyzer.prepare(first)

    with pytest.raises(VideoDemoError) as raised:
        analyzer.finalize(second, preparation, speech=SpeechAnalysis())

    assert raised.value.code == ErrorCode.VISUAL_RESULT_INVALID
    assert factory_calls == ["factory"]


def test_speech_candidates_build_hybrid_windows_but_scene_alone_does_not(
    tmp_path: Path,
) -> None:
    media = _media(tmp_path, duration_ms=20_000, language_hints=("en",))
    seen_windows: list[tuple[int, int]] = []

    class Scenes:
        def detect(
            self,
            _proxy: Path,
            *,
            duration_ms: int,
            source_sha256: str,
            frame_tolerance_ms: int,
        ) -> tuple[SceneBoundary, ...]:
            assert source_sha256 == media.proxy_sha256
            assert frame_tolerance_ms == 40
            return (
                _scene(0, 8_000, transition="candidate"),
                _scene(8_000, duration_ms, transition="hard_cut"),
            )

    class Frames:
        def extract(
            self,
            _proxy: Path,
            run_relative_root: Path,
            windows: Sequence[TimeRange],
            **_kwargs: object,
        ) -> tuple[WindowFrameCandidates, ...]:
            seen_windows.extend((item.start_ms, item.end_ms) for item in windows)
            return tuple(
                WindowFrameCandidates(
                    window=window,
                    candidates=(
                        _write_frame(
                            tmp_path,
                            run_relative_root,
                            window.start_ms + 1_000,
                            f"same-{window.start_ms}".encode(),
                            perceptual_hash=f"{window.start_ms + 1:016x}",
                        ),
                    ),
                )
                for window in windows
            )

    speech = SpeechAnalysis(
        boundary_candidates=(
            SpeechBoundaryCandidate(timestamp_ms=10_000, source="silence", score=1.0),
        ),
    )
    result = ProductionVisualAnalyzer(
        tmp_path,
        lambda _media, _cancel: VisualComponents(
            scene_detector=Scenes(),
            frame_extractor=Frames(),
            keyframe_selector=KeyframeSelector(),
            ocr_client=_StaticOcr("相同页面"),
            clip_client=_Clips(tmp_path, []),
        ),
    ).analyze(media, speech=speech)

    assert seen_windows == [(0, 10_000), (10_000, 20_000)]
    assert result.clips == ()
    assert [(item.start_ms, item.end_ms) for item in result.windows] == [(0, 20_000)]
    hard_scene = next(item for item in result.boundaries if item.timestamp_ms == 8_000)
    assert "scene_hard" in hard_scene.sources
    assert not any(item.end_ms == 8_000 for item in result.clips)


def test_no_frames_does_not_fabricate_evidence_or_construct_ocr_credentials(
    tmp_path: Path,
) -> None:
    media = _media(tmp_path, duration_ms=4_000)
    ocr_calls: list[str] = []

    class NoFrames:
        def extract(
            self,
            _proxy: Path,
            _root: Path,
            windows: Sequence[TimeRange],
            **_kwargs: object,
        ) -> tuple[WindowFrameCandidates, ...]:
            return tuple(WindowFrameCandidates(window=item, candidates=()) for item in windows)

    class Ocr:
        def recognize(self, _image: bytes, _language: str) -> OcrProviderResponse:
            ocr_calls.append("credentials")
            raise AssertionError("无关键帧时不得加载 OCR 凭据")

    result = ProductionVisualAnalyzer(
        tmp_path,
        lambda _media, _cancel: VisualComponents(
            scene_detector=_Scenes(4_000),
            frame_extractor=NoFrames(),
            keyframe_selector=KeyframeSelector(),
            ocr_client=Ocr(),
            clip_client=_Clips(tmp_path, []),
        ),
    ).analyze(media)

    assert ocr_calls == []
    assert [item.evidence_type for item in result.evidence] == ["SCENE"]
    assert result.warnings == ("NO_KEYFRAME:0:4000",)


@pytest.mark.parametrize(
    ("speech_language", "hints", "expected_language", "warning"),
    [
        ("ja", ("en",), "ja", None),
        (None, ("es",), "es", None),
        (None, (), "zh", "OCR_LANGUAGE_FALLBACK:zh"),
        ("fr", ("en",), None, "OCR_LANGUAGE_UNSUPPORTED:fr"),
    ],
)
def test_ocr_language_selection_and_warnings(
    tmp_path: Path,
    speech_language: str | None,
    hints: tuple[str, ...],
    expected_language: str | None,
    warning: str | None,
) -> None:
    media = _media(tmp_path, duration_ms=4_000, language_hints=hints)
    languages: list[str] = []
    speech = SpeechAnalysis(
            evidence=(
                (
                    SpeechSegment(
                    evidence_id="asr_001",
                    start_ms=0,
                    end_ms=4_000,
                    text="文本",
                    language=speech_language,
                    confidence=0.9,
                    is_fully_evaluated_language=True,
                    ),
                )
            if speech_language is not None
            else ()
        ),
    )

    class Frames:
        def extract(
            self,
            _proxy: Path,
            root: Path,
            windows: Sequence[TimeRange],
            **_kwargs: object,
        ) -> tuple[WindowFrameCandidates, ...]:
            return (
                WindowFrameCandidates(
                    window=windows[0],
                    candidates=(_write_frame(tmp_path, root, 2_000, b"page"),),
                ),
            )

    class Ocr:
        def recognize(self, _image: bytes, language: str) -> OcrProviderResponse:
            languages.append(language)
            return _ocr("页面")

    result = ProductionVisualAnalyzer(
        tmp_path,
        lambda _media, _cancel: VisualComponents(
            scene_detector=_Scenes(4_000),
            frame_extractor=Frames(),
            keyframe_selector=KeyframeSelector(),
            ocr_client=Ocr(),
            clip_client=_Clips(tmp_path, []),
        ),
    ).analyze(media, speech=speech)

    assert languages == ([] if expected_language is None else [expected_language])
    if warning is not None:
        assert warning in result.warnings


@pytest.mark.parametrize(
    "mutation",
    ["digest", "outside", "symlink", "directory", "empty", "fake_mp4", "size"],
)
def test_proxy_is_verified_before_pixel_or_external_processing(
    tmp_path: Path,
    mutation: str,
) -> None:
    media = _media(tmp_path, duration_ms=4_000)
    if mutation == "digest":
        media = replace(media, proxy_sha256="0" * 64)
    elif mutation == "outside":
        outside = tmp_path / "outside.mp4"
        outside.write_bytes(b"proxy")
        media = replace(media, proxy_path=outside)
    elif mutation == "symlink":
        target = media.proxy_path
        link = target.with_name("proxy-link.mp4")
        link.symlink_to(target)
        media = replace(media, proxy_path=link)
    elif mutation == "directory":
        media.proxy_path.unlink()
        media.proxy_path.mkdir()
    elif mutation == "empty":
        media.proxy_path.write_bytes(b"")
        media = replace(media, proxy_sha256=hashlib.sha256(b"").hexdigest(), proxy_size_bytes=0)
    elif mutation == "fake_mp4":
        media.proxy_path.write_bytes(b"not-mp4")
        media = replace(
            media,
            proxy_sha256=hashlib.sha256(b"not-mp4").hexdigest(),
            proxy_size_bytes=7,
        )
    else:
        media = replace(media, proxy_size_bytes=media.proxy_size_bytes + 1)

    calls: list[str] = []

    class Scenes:
        def detect(
            self,
            _proxy: Path,
            *,
            duration_ms: int,
            source_sha256: str,
            frame_tolerance_ms: int,
        ) -> tuple[SceneBoundary, ...]:
            assert source_sha256 == media.proxy_sha256
            assert frame_tolerance_ms == 40
            calls.append("scene")
            return (_scene(0, duration_ms, transition="candidate"),)

    analyzer = ProductionVisualAnalyzer(
        tmp_path,
        lambda _media, _cancel: VisualComponents(
            scene_detector=Scenes(),
            frame_extractor=object(),  # type: ignore[arg-type]
            keyframe_selector=KeyframeSelector(),
            ocr_client=object(),  # type: ignore[arg-type]
            clip_client=object(),  # type: ignore[arg-type]
        ),
    )

    with pytest.raises(VideoDemoError) as raised:
        analyzer.analyze(media)

    assert raised.value.code in {
        ErrorCode.WORKSPACE_PATH_ESCAPE,
        ErrorCode.VIDEO_DIGEST_MISMATCH,
        ErrorCode.VIDEO_INPUT_INVALID,
    }
    assert calls == []


def test_configured_video_size_limit_is_applied_to_proxy_before_downstream_use(
    tmp_path: Path,
) -> None:
    media = _media(tmp_path, duration_ms=4_000)
    calls: list[str] = []

    class Frames:
        def extract(
            self,
            _proxy: Path,
            _root: Path,
            windows: Sequence[TimeRange],
            **_kwargs: object,
        ) -> tuple[WindowFrameCandidates, ...]:
            calls.append("frames")
            return tuple(WindowFrameCandidates(window=item, candidates=()) for item in windows)

    analyzer = ProductionVisualAnalyzer(
        tmp_path,
        lambda _media, _cancel: VisualComponents(
            scene_detector=_Scenes(4_000),
            frame_extractor=Frames(),
            keyframe_selector=KeyframeSelector(),
            ocr_client=_StaticOcr(""),
            clip_client=_Clips(tmp_path, calls),
        ),
        max_video_bytes=23,
    )

    with pytest.raises(VideoDemoError) as raised:
        analyzer.analyze(media)

    assert raised.value.code == ErrorCode.VIDEO_INPUT_INVALID
    assert calls == []


def test_visual_analysis_never_calls_clip_adapter(tmp_path: Path) -> None:
    media = _media(tmp_path, duration_ms=4_000)

    class Frames:
        def extract(
            self,
            _proxy: Path,
            _root: Path,
            windows: Sequence[TimeRange],
            **_kwargs: object,
        ) -> tuple[WindowFrameCandidates, ...]:
            return tuple(WindowFrameCandidates(window=item, candidates=()) for item in windows)

    class BadClips:
        def create_clip(
            self,
            _source: Path,
            _root: Path,
            clip_id: str,
            time_range: TimeRange,
        ) -> ClipArtifact:
            raise AssertionError("全片链路不得创建窗口短片")

    analyzer = ProductionVisualAnalyzer(
        tmp_path,
        lambda _media, _cancel: VisualComponents(
            scene_detector=_Scenes(4_000),
            frame_extractor=Frames(),
            keyframe_selector=KeyframeSelector(),
            ocr_client=_StaticOcr(""),
            clip_client=BadClips(),
        ),
    )

    result = analyzer.analyze(media)

    assert result.clips == ()
    assert result.windows == (TimeRange(start_ms=0, end_ms=4_000),)


def test_lazy_baidu_client_reads_credentials_only_on_first_ocr() -> None:
    credential_calls: list[str] = []

    def credentials() -> tuple[str | None, str | None]:
        credential_calls.append("credentials")
        return None, None

    client = LazyBaiduOcrClient(
        httpx.Client(transport=httpx.MockTransport(lambda _request: httpx.Response(500))),
        credentials,
        endpoint="https://example.invalid/ocr",
    )

    assert credential_calls == []
    with pytest.raises(VideoDemoError) as raised:
        client.recognize(b"image", "zh")

    assert raised.value.code == ErrorCode.OCR_AUTHENTICATION_FAILED
    assert credential_calls == ["credentials"]


class _Scenes:
    def __init__(self, duration_ms: int) -> None:
        self._duration_ms = duration_ms

    def detect(
        self,
        _proxy: Path,
        *,
        duration_ms: int,
        source_sha256: str,
        frame_tolerance_ms: int,
    ) -> tuple[SceneBoundary, ...]:
        assert duration_ms == self._duration_ms
        assert len(source_sha256) == 64
        assert frame_tolerance_ms == 40
        return (_scene(0, duration_ms, transition="candidate"),)


class _StaticOcr:
    def __init__(self, text: str) -> None:
        self._text = text

    def recognize(self, _image: bytes, _language: str) -> OcrProviderResponse:
        return _ocr(self._text)


class _Clips:
    def __init__(self, runtime_root: Path, calls: list[str]) -> None:
        self._runtime_root = runtime_root
        self._calls = calls

    def create_clip(
        self,
        _source: Path,
        run_relative_root: Path,
        clip_id: str,
        time_range: TimeRange,
    ) -> ClipArtifact:
        self._calls.append(f"clip:{time_range.start_ms}-{time_range.end_ms}")
        relative = run_relative_root / "visual/clips" / f"{clip_id}.mp4"
        output = self._runtime_root / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = _MP4 + f"clip:{time_range.start_ms}:{time_range.end_ms}".encode()
        output.write_bytes(payload)
        return ClipArtifact(
            clip_id=clip_id,
            start_ms=time_range.start_ms,
            end_ms=time_range.end_ms,
            relative_path=relative.as_posix(),
            sha256=hashlib.sha256(payload).hexdigest(),
            size_bytes=len(payload),
        )


def _ocr(text: str) -> OcrProviderResponse:
    lines = (
        (
            OcrLine(
                text=text,
                bounding_box=BoundingBox(x=1, y=2, width=100, height=20),
                confidence=0.95,
            ),
        )
        if text
        else ()
    )
    return OcrProviderResponse(request_id="request-safe", lines=lines)


def _scene(start_ms: int, end_ms: int, *, transition: str) -> SceneBoundary:
    return SceneBoundary(
        evidence_id=f"scene_{start_ms}_{end_ms}",
        start_ms=start_ms,
        end_ms=end_ms,
        transition=transition,  # type: ignore[arg-type]
        score=0.9,
    )


def _write_frame(
    runtime_root: Path,
    run_relative_root: Path,
    timestamp_ms: int,
    marker: bytes,
    *,
    perceptual_hash: str | None = None,
) -> FrameCandidate:
    relative = run_relative_root / "visual/keyframes" / f"frame_{timestamp_ms:012d}.jpg"
    path = runtime_root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_JPEG_PREFIX + marker + _JPEG_SUFFIX)
    return FrameCandidate(
        timestamp_ms=timestamp_ms,
        sharpness=100.0 + timestamp_ms,
        black_ratio=0.0,
        perceptual_hash=perceptual_hash or f"{timestamp_ms + 1:016x}",
        relative_path=relative,
    )


def _media(
    tmp_path: Path,
    *,
    duration_ms: int,
    language_hints: tuple[str, ...] = (),
    is_variable_frame_rate: bool = False,
) -> PreparedMedia:
    run_root = Path("runs/scope/run_001")
    source_path = tmp_path / run_root / "input/source.mp4"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(b"source")
    registered = RegisteredAsset(
        source_path=source_path,
        source_sha256=hashlib.sha256(b"source").hexdigest(),
        object_ref="obj_001",
        source_size_bytes=6,
        source_mime="video/mp4",
        run_relative_root=run_root,
        config=PipelineRunConfig(language_hints=language_hints),  # type: ignore[arg-type]
    )
    manifest = VideoAssetManifest(
        object_ref="obj_001",
        source_sha256=registered.source_sha256,
        source_size_bytes=6,
        source_mime="video/mp4",
        duration_ms=duration_ms,
        video_stream=VideoStream(
            index=0,
            codec_name="h264",
            width=640,
            height=360,
            average_frame_rate=Rational(numerator=25, denominator=1),
            is_variable_frame_rate=is_variable_frame_rate,
        ),
        format_name="mov,mp4",
        ffprobe_version="test",
    )
    proxy = tmp_path / run_root / "media/proxy.mp4"
    proxy.parent.mkdir(parents=True, exist_ok=True)
    proxy.write_bytes(_MP4)
    return PreparedMedia(
        source=ProbedAsset(registered, manifest, ProbeLimits()),
        proxy_path=proxy,
        proxy_sha256=hashlib.sha256(_MP4).hexdigest(),
        proxy_size_bytes=len(_MP4),
        audio_path=None,
        audio_sha256=None,
    )
