from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path

import httpx
import pytest

from video_demo.application import adaptive_ocr as adaptive_ocr_module
from video_demo.application.pipeline import (
    PipelineRunConfig,
    PreparedMedia,
    ProbedAsset,
    RegisteredAsset,
    SpeechAnalysis,
    SpeechBoundaryCandidate,
    VisualAnalysis,
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
    SubtitleCue,
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
        def recognize(
            self,
            image: bytes,
            language: str,
            *,
            deadline: float | None = None,
        ) -> OcrProviderResponse:
            calls.append(f"ocr:{language}")
            assert deadline is not None
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

    result = analyzer.analyze(media, speech=SpeechAnalysis(transcript_source="NONE"))

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


def test_low_text_visual_chain_stops_after_probe_budget(tmp_path: Path) -> None:
    duration_ms = 300_000
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
        def recognize(
            self,
            _image: bytes,
            language: str,
            *,
            deadline: float | None = None,
        ) -> OcrProviderResponse:
            ocr_calls.append(language)
            assert deadline is not None
            return _ocr("")

    speech = SpeechAnalysis(
        transcript_source="ASR",
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
    assert len(keyframes) == 6
    assert len(ocr) == 6
    assert len(ocr_calls) == 6
    assert timestamps == sorted(set(timestamps))
    assert timestamps[0] == 25_000
    assert timestamps[-1] < duration_ms


def test_normal_text_visual_chain_stops_at_base_budget(tmp_path: Path) -> None:
    result, calls = _run_budget_visual(
        tmp_path,
        text_for_call=lambda index: f"相同课程页面主体内容仅修改版本编号{index % 2}",
    )

    assert calls == 13
    assert len([item for item in result.evidence if item.evidence_type == "KEYFRAME"]) == 13
    assert len([item for item in result.evidence if item.evidence_type == "OCR"]) == 13


def test_dense_text_visual_chain_adds_three_at_a_time_until_hard_limit(tmp_path: Path) -> None:
    result, calls = _run_budget_visual(
        tmp_path,
        text_for_call=lambda index: chr(0x4E00 + index) * 30,
    )

    assert calls == 20
    assert len([item for item in result.evidence if item.evidence_type == "KEYFRAME"]) == 20
    assert len([item for item in result.evidence if item.evidence_type == "OCR"]) == 20


def test_dense_text_visual_chain_stops_when_full_batch_has_less_than_two_new_pages(
    tmp_path: Path,
) -> None:
    result, calls = _run_budget_visual(
        tmp_path,
        text_for_call=lambda index: (
            chr(0x4E00 + index) * 30
            if index < 6
            else "龠" * 30
        ),
    )

    assert calls == 9
    assert len([item for item in result.evidence if item.evidence_type == "KEYFRAME"]) == 9


def test_ocr_budget_timeout_keeps_completed_evidence_and_emits_warning(tmp_path: Path) -> None:
    clock_values = iter((0.0, 0.0, 0.1, 0.2, 0.3, 31.0))
    result, calls = _run_budget_visual(
        tmp_path,
        text_for_call=lambda index: f"第{index}页包含完全不同且足够长的有效文字内容",
        clock=lambda: next(clock_values, 31.0),
    )

    assert calls == 1
    assert len([item for item in result.evidence if item.evidence_type == "KEYFRAME"]) == 1
    assert "OCR_BUDGET_TIME_LIMIT_REACHED" in result.warnings


def test_deadline_reached_during_probe_scoring_wins_over_text_classification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = [0.0]
    original_assess = adaptive_ocr_module.assess_probe_text

    def assess_then_reach_deadline(*args: object, **kwargs: object) -> object:
        assessment = original_assess(*args, **kwargs)  # type: ignore[arg-type]
        now[0] = 31.0
        return assessment

    monkeypatch.setattr(
        adaptive_ocr_module,
        "assess_probe_text",
        assess_then_reach_deadline,
    )

    result, calls = _run_budget_visual(
        tmp_path,
        text_for_call=lambda _index: "",
        clock=lambda: now[0],
    )

    assert calls == 6
    assert "OCR_BUDGET_TIME_LIMIT_REACHED" in result.warnings


def test_deadline_reached_during_batch_value_scoring_wins_over_marginal_stop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = [0.0]

    def score_then_reach_deadline(*_args: object, **_kwargs: object) -> int:
        now[0] = 31.0
        return 0

    monkeypatch.setattr(
        adaptive_ocr_module,
        "batch_new_text_count",
        score_then_reach_deadline,
    )

    result, calls = _run_budget_visual(
        tmp_path,
        text_for_call=lambda index: chr(0x4E00 + index) * 30,
        clock=lambda: now[0],
    )

    assert calls == 9
    assert "OCR_BUDGET_TIME_LIMIT_REACHED" in result.warnings


def test_deadline_reached_after_normal_batch_wins_over_base_budget_stop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = [0.0]
    original_append = adaptive_ocr_module._append_batch

    def append_then_reach_deadline(*args: object, **kwargs: object) -> None:
        original_append(*args, **kwargs)  # type: ignore[arg-type]
        now[0] = 31.0

    monkeypatch.setattr(
        adaptive_ocr_module,
        "_append_batch",
        append_then_reach_deadline,
    )

    result, calls = _run_budget_visual(
        tmp_path,
        text_for_call=lambda index: f"相同课程页面主体内容仅修改版本编号{index % 2}",
        clock=lambda: now[0],
    )

    assert calls == 13
    assert "OCR_BUDGET_TIME_LIMIT_REACHED" in result.warnings


def test_deadline_reached_during_normal_candidate_selection_wins_over_exhaustion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = [0.0]

    def exhaust_then_reach_deadline(
        _keyframes: Sequence[KeyframeEvidence],
        selected: Sequence[KeyframeEvidence],
        *,
        count: int,
    ) -> tuple[KeyframeEvidence, ...]:
        assert count == 7
        now[0] = 31.0
        return tuple(selected)

    monkeypatch.setattr(
        adaptive_ocr_module,
        "extend_keyframes",
        exhaust_then_reach_deadline,
    )

    result, calls = _run_budget_visual(
        tmp_path,
        text_for_call=lambda index: f"相同课程页面主体内容仅修改版本编号{index % 2}",
        clock=lambda: now[0],
    )

    assert calls == 6
    assert "OCR_BUDGET_TIME_LIMIT_REACHED" in result.warnings


def test_deadline_reached_during_last_dense_batch_scoring_wins_over_hard_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = [0.0]
    score_calls = 0
    original_effective_texts = adaptive_ocr_module.effective_frame_texts

    def score_then_reach_deadline(*args: object, **kwargs: object) -> tuple[str, ...]:
        nonlocal score_calls
        texts = original_effective_texts(*args, **kwargs)  # type: ignore[arg-type]
        score_calls += 1
        if score_calls == 5:
            now[0] = 31.0
        return texts

    monkeypatch.setattr(
        adaptive_ocr_module,
        "effective_frame_texts",
        score_then_reach_deadline,
    )

    result, calls = _run_budget_visual(
        tmp_path,
        text_for_call=lambda index: chr(0x4E00 + index) * 30,
        clock=lambda: now[0],
    )

    assert calls == 20
    assert "OCR_BUDGET_TIME_LIMIT_REACHED" in result.warnings


def test_deadline_reached_during_dense_candidate_selection_wins_over_exhaustion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = [0.0]

    def exhaust_then_reach_deadline(
        _keyframes: Sequence[KeyframeEvidence],
        selected: Sequence[KeyframeEvidence],
        *,
        count: int,
    ) -> tuple[KeyframeEvidence, ...]:
        assert count == 3
        now[0] = 31.0
        return tuple(selected)

    monkeypatch.setattr(
        adaptive_ocr_module,
        "extend_keyframes",
        exhaust_then_reach_deadline,
    )

    result, calls = _run_budget_visual(
        tmp_path,
        text_for_call=lambda index: chr(0x4E00 + index) * 30,
        clock=lambda: now[0],
    )

    assert calls == 6
    assert "OCR_BUDGET_TIME_LIMIT_REACHED" in result.warnings


def test_cancellation_after_dense_batch_wins_over_budget_decision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cancelled = [False]
    original_append = adaptive_ocr_module._append_batch

    def append_then_cancel(*args: object, **kwargs: object) -> None:
        original_append(*args, **kwargs)  # type: ignore[arg-type]
        cancelled[0] = True

    monkeypatch.setattr(adaptive_ocr_module, "_append_batch", append_then_cancel)

    with pytest.raises(VideoDemoError) as raised:
        _run_budget_visual(
            tmp_path,
            text_for_call=lambda index: chr(0x4E00 + index) * 30,
            is_cancel_requested=lambda: cancelled[0],
        )

    assert raised.value.code == ErrorCode.JOB_CANCELLED


@pytest.mark.parametrize(
    "error_code",
    [ErrorCode.OCR_AUTHENTICATION_FAILED, ErrorCode.OCR_RESPONSE_INVALID],
)
def test_ocr_external_error_remains_fail_closed_in_visual_chain(
    tmp_path: Path,
    error_code: ErrorCode,
) -> None:
    with pytest.raises(VideoDemoError) as raised:
        _run_budget_visual(
            tmp_path,
            text_for_call=lambda _index: "",
            ocr_error_code=error_code,
        )

    assert raised.value.code == error_code


def test_unsupported_language_batch_stops_at_deadline_without_provider_calls(
    tmp_path: Path,
) -> None:
    now = [0.0]
    language_checks = 0

    def unsupported_then_reach_deadline(*_args: object, **_kwargs: object) -> object:
        nonlocal language_checks
        language_checks += 1
        if language_checks == 1:
            now[0] = 31.0
        return None, "OCR_LANGUAGE_UNSUPPORTED:fr"

    result, calls = _run_budget_visual(
        tmp_path,
        text_for_call=lambda _index: "",
        clock=lambda: now[0],
        before_analyze=lambda monkeypatch: monkeypatch.setattr(
            adaptive_ocr_module,
            "ocr_language",
            unsupported_then_reach_deadline,
        ),
    )

    assert calls == 0
    assert language_checks == 1
    assert len(
        [item for item in result.evidence if item.evidence_type == "KEYFRAME"],
    ) == 1
    assert "OCR_BUDGET_TIME_LIMIT_REACHED" in result.warnings


def test_cancellation_during_ocr_call_wins_over_deadline_and_discards_partial_result(
    tmp_path: Path,
) -> None:
    duration_ms = 4_000
    media = _media(tmp_path, duration_ms=duration_ms)
    cancelled = [False]

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
        def recognize(
            self,
            _image: bytes,
            _language: str,
            *,
            deadline: float | None = None,
        ) -> OcrProviderResponse:
            cancelled[0] = True
            return _ocr("这一帧已经识别成功但任务同时收到取消请求")

    analyzer = ProductionVisualAnalyzer(
        tmp_path,
        lambda _media, _cancel: VisualComponents(
            scene_detector=_Scenes(duration_ms),
            frame_extractor=Frames(),
            keyframe_selector=KeyframeSelector(),
            ocr_client=Ocr(),
            clip_client=_Clips(tmp_path, []),
        ),
    )

    with pytest.raises(VideoDemoError) as raised:
        analyzer.analyze(media, is_cancel_requested=lambda: cancelled[0])

    assert raised.value.code == ErrorCode.JOB_CANCELLED


def test_ocr_budget_log_contains_only_aggregate_diagnostics(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    secret_text = "绝不能出现在日志中的OCR正文"
    with caplog.at_level(logging.INFO, logger="video_demo.application.production_visual"):
        _result, _calls = _run_budget_visual(
            tmp_path,
            text_for_call=lambda _index: secret_text,
        )

    messages = [record.getMessage() for record in caplog.records]
    assert any("ocr_budget_complete" in message for message in messages)
    assert all(secret_text not in message for message in messages)
    record = next(
        record
        for record in caplog.records
        if "ocr_budget_complete" in record.getMessage()
    )
    assert record.ocr_classification in {"LOW_TEXT", "NORMAL_TEXT", "DENSE_TEXT"}
    assert isinstance(record.ocr_selected_keyframe_count, int)


def test_ocr_budget_log_separates_image_requests_from_provider_attempts(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO, logger="video_demo.application.production_visual"):
        _run_budget_visual(
            tmp_path,
            text_for_call=lambda _index: "同一页面内容" * 4,
        )

    record = next(
        record
        for record in caplog.records
        if "ocr_budget_complete" in record.getMessage()
    )
    assert isinstance(record.ocr_image_request_count, int)
    assert record.ocr_image_request_count <= record.ocr_selected_keyframe_count
    assert record.ocr_provider_attempt_count >= record.ocr_image_request_count


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
            size_bytes=1,
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
        analyzer.finalize(
            media,
            replaced,
            speech=SpeechAnalysis(transcript_source="NONE"),
        )

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
            speech=SpeechAnalysis(transcript_source="NONE"),
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
        analyzer.finalize(
            second,
            preparation,
            speech=SpeechAnalysis(transcript_source="NONE"),
        )

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
        transcript_source="ASR",
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
        def recognize(
            self,
            _image: bytes,
            _language: str,
            *,
            deadline: float | None = None,
        ) -> OcrProviderResponse:
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
        transcript_source="ASR",
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
        def recognize(
            self,
            _image: bytes,
            language: str,
            *,
            deadline: float | None = None,
        ) -> OcrProviderResponse:
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


def test_ocr_language_can_follow_subtitle_cue(tmp_path: Path) -> None:
    from video_demo.application.production_visual import _ocr_language

    media = _media(tmp_path, duration_ms=4_000, language_hints=("en",))
    subtitle = SubtitleCue(
        evidence_id="subtitle_001",
        start_ms=0,
        end_ms=4_000,
        text="字幕正文",
        language="ja",
        stream_index=2,
    )

    language, warning = _ocr_language(
        2_000,
        SpeechAnalysis(transcript_source="SUBTITLE", evidence=(subtitle,)),
        media,
    )

    assert language == "ja"
    assert warning is None


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

    def recognize(
        self,
        _image: bytes,
        _language: str,
        *,
        deadline: float | None = None,
    ) -> OcrProviderResponse:
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


def _run_budget_visual(
    tmp_path: Path,
    *,
    text_for_call: Callable[[int], str],
    clock: Callable[[], float] | None = None,
    ocr_error_code: ErrorCode | None = None,
    is_cancel_requested: Callable[[], bool] = lambda: False,
    before_analyze: Callable[[pytest.MonkeyPatch], None] | None = None,
) -> tuple[VisualAnalysis, int]:
    duration_ms = 300_000
    media = _media(tmp_path, duration_ms=duration_ms)
    call_count = 0

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
                            perceptual_hash=f"{window.start_ms + 1:016x}",
                        ),
                    ),
                )
                for window in windows
            )

    class Ocr:
        def recognize(
            self,
            _image: bytes,
            _language: str,
            *,
            deadline: float | None = None,
        ) -> OcrProviderResponse:
            nonlocal call_count
            assert deadline is not None
            if ocr_error_code is not None:
                raise VideoDemoError(ocr_error_code, "模拟 OCR 外部错误")
            text = text_for_call(call_count)
            call_count += 1
            return _ocr(text)

    speech = SpeechAnalysis(
        transcript_source="ASR",
        boundary_candidates=tuple(
            SpeechBoundaryCandidate(timestamp_ms=timestamp_ms, source="silence")
            for timestamp_ms in range(10_000, duration_ms, 10_000)
        ),
    )
    analyzer_kwargs: dict[str, object] = {}
    if clock is not None:
        analyzer_kwargs["clock"] = clock
    with pytest.MonkeyPatch.context() as monkeypatch:
        if before_analyze is not None:
            before_analyze(monkeypatch)
        result = ProductionVisualAnalyzer(
            tmp_path,
            lambda _media, _cancel: VisualComponents(
                scene_detector=_Scenes(duration_ms),
                frame_extractor=Frames(),
                keyframe_selector=KeyframeSelector(),
                ocr_client=Ocr(),
                clip_client=_Clips(tmp_path, []),
            ),
            **analyzer_kwargs,
        ).analyze(
            media,
            speech=speech,
            is_cancel_requested=is_cancel_requested,
        )
    return result, call_count


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
