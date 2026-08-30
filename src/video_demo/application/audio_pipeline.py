from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Literal, Protocol, cast

from video_demo.application.audio_chapter_planning import AudioChapterPlanner
from video_demo.application.audio_contracts import (
    AudioEvidencePreparationLimits,
    AudioSpeechAnalysis,
    AudioSpeechBoundaryCandidate,
    AudioStageMetric,
)
from video_demo.application.audio_document_writing import AudioDocumentWriter, AudioWritingContext
from video_demo.application.audio_rendering import RenderedAudioDocument, render_audio_markdown
from video_demo.application.audio_run_config import AudioRunConfig
from video_demo.application.audio_segments import build_audio_segments
from video_demo.domain.audio_document import AudioUnderstandingResult
from video_demo.domain.audio_plan import AudioDocumentConfig
from video_demo.domain.base import Sha256, stable_identifier
from video_demo.domain.evidence import SpeechSegment, SubtitleCue
from video_demo.domain.run import TimeRange
from video_demo.errors import ErrorCode, VideoDemoError, is_cancelled_error_code
from video_demo.speech.asr_contracts import WindowRecognizerPort
from video_demo.speech.audio_asr import (
    build_cloud_asr_windows,
    project_cloud_asr_window,
    remove_adjacent_cloud_asr_duplicates,
)
from video_demo.speech.audio_snapshots import (
    AudioAsrWindowSnapshotPayload,
    audio_asr_window_fingerprint,
)
from video_demo.speech.vad import VadResult
from video_demo.storage.audio_snapshots import AudioAsrWindowSnapshotStore
from video_demo.storage.document_cache import DocumentModelCache

_LOGGER = logging.getLogger(__name__)


class VadPort(Protocol):
    def detect(self, audio: Path, *, duration_ms: int) -> VadResult: ...


class AudioSlicerPort(Protocol):
    def create(
        self,
        audio: Path,
        run_relative_root: Path,
        slice_id: str,
        time_range: TimeRange,
    ) -> Path: ...


@dataclass(frozen=True, slots=True)
class AudioPipelineOutcome:
    result: AudioUnderstandingResult
    document: RenderedAudioDocument
    evidence: tuple[SpeechSegment | SubtitleCue, ...]
    warnings: tuple[str, ...]
    status: Literal["SUCCEEDED", "PARTIAL_SUCCEEDED"]


class AudioSpeechPort:
    def analyze(
        self,
        source: Path,
        *,
        duration_ms: int,
        config: AudioRunConfig,
        run_root: Path,
        is_cancel_requested: Callable[[], bool],
    ) -> AudioSpeechAnalysis:
        raise NotImplementedError


class AudioSpeechAnalyzer(AudioSpeechPort):
    """独立音频 ASR：VAD 分窗、串行 Whisper 识别和窗口级失败隔离。"""

    def __init__(
        self,
        vad: object,
        recognizer: object,
        slicer: object,
        *,
        max_window_ms: int,
        overlap_ms: int,
        max_upload_bytes: int,
        window_store: AudioAsrWindowSnapshotStore | None = None,
    ) -> None:
        self._vad = cast(VadPort, vad)
        self._recognizer = cast(WindowRecognizerPort, recognizer)
        self._slicer = cast(AudioSlicerPort, slicer)
        self._max_window_ms = max_window_ms
        self._overlap_ms = overlap_ms
        self._max_upload_bytes = max_upload_bytes
        self._window_store = window_store

    def analyze(
        self,
        source: Path,
        *,
        duration_ms: int,
        config: AudioRunConfig,
        run_root: Path,
        is_cancel_requested: Callable[[], bool],
    ) -> AudioSpeechAnalysis:
        started_at = time.monotonic()
        vad_started_at = time.monotonic()
        vad_result = self._vad.detect(source, duration_ms=duration_ms)
        windows = build_cloud_asr_windows(
            vad_result.speech,
            max_window_ms=self._max_window_ms,
            overlap_ms=self._overlap_ms,
            max_upload_bytes=self._max_upload_bytes,
        )
        _LOGGER.info(
            "音频 ASR 分窗完成: windows=%d duration_ms=%d vad_elapsed=%.3fs",
            len(windows),
            duration_ms,
            time.monotonic() - vad_started_at,
        )
        transcript: list[SpeechSegment] = []
        language_spans = []
        warnings = list(vad_result.warnings)
        language_hint = config.language_hints[0] if len(config.language_hints) == 1 else None
        prompt = _audio_asr_prompt(config)
        for index, window in enumerate(windows, start=1):
            if is_cancel_requested():
                raise VideoDemoError(ErrorCode.JOB_CANCELLED, "任务已请求取消")
            window_started_at = time.monotonic()
            _LOGGER.info(
                "音频 ASR 窗口开始: index=%d/%d upload=%d-%dms owned=%d-%dms",
                index,
                len(windows),
                window.upload_range.start_ms,
                window.upload_range.end_ms,
                window.owned_range.start_ms,
                window.owned_range.end_ms,
            )
            slice_id = stable_identifier(
                "audio-slice",
                {
                    "run_root": run_root.as_posix(),
                    "index": index,
                    "start_ms": window.upload_range.start_ms,
                },
            )
            try:
                fingerprint = audio_asr_window_fingerprint(
                    run_root=run_root.as_posix(),
                    window=window,
                    language_hint=language_hint,
                    prompt=prompt,
                    max_window_ms=self._max_window_ms,
                    overlap_ms=self._overlap_ms,
                    max_upload_bytes=self._max_upload_bytes,
                )
                cached = (
                    self._window_store.load(run_root, fingerprint)
                    if self._window_store is not None
                    else None
                )
                if cached is not None:
                    language_spans.append(cached[0].language_span)
                    transcript.extend(cached[0].segments)
                    warnings.extend(cached[0].warnings)
                    _LOGGER.info(
                        "音频 ASR 窗口命中快照: index=%d/%d segments=%d elapsed=%.3fs",
                        index,
                        len(windows),
                        len(cached[0].segments),
                        time.monotonic() - window_started_at,
                    )
                    continue
                audio_slice = self._slicer.create(
                    source,
                    run_root,
                    slice_id,
                    window.upload_range,
                )
                try:
                    result = self._recognizer.transcribe_window(
                        audio_slice,
                        language_hint=language_hint,
                        prompt=prompt,
                    )
                    projection = project_cloud_asr_window(
                        window,
                        language=result.language,
                        raw_segments=result.segments,
                        warnings=result.warnings,
                    )
                    if self._window_store is not None:
                        self._window_store.publish(
                            run_root,
                            fingerprint,
                            AudioAsrWindowSnapshotPayload(
                                upload_range=window.upload_range,
                                owned_range=window.owned_range,
                                speech_interval=window.speech_interval,
                                source_intervals=window.source_intervals,
                                language_span=projection.language_span,
                                segments=projection.segments,
                                warnings=projection.warnings,
                            ),
                        )
                    language_spans.append(projection.language_span)
                    transcript.extend(projection.segments)
                    warnings.extend(projection.warnings)
                    _LOGGER.info(
                        "音频 ASR 窗口完成: index=%d/%d segments=%d elapsed=%.3fs",
                        index,
                        len(windows),
                        len(projection.segments),
                        time.monotonic() - window_started_at,
                    )
                finally:
                    audio_slice.unlink(missing_ok=True)
            except VideoDemoError as error:
                if is_cancelled_error_code(error.code):
                    if error.code == ErrorCode.JOB_CANCELLED:
                        raise
                    raise VideoDemoError(ErrorCode.JOB_CANCELLED, "音频分析已取消") from error
                _LOGGER.warning(
                    "音频 ASR 窗口降级: index=%d/%d code=%s elapsed=%.3fs",
                    index,
                    len(windows),
                    error.code.value,
                    time.monotonic() - window_started_at,
                )
                warnings.append(f"AUDIO_ASR_WINDOW_DEGRADED:{index}")
        ordered = remove_adjacent_cloud_asr_duplicates(tuple(transcript))
        ordered_language_spans = tuple(
            sorted(
                language_spans,
                key=lambda item: (item.start_ms, item.end_ms, item.evidence_id),
            ),
        )
        transcript_source: Literal["ASR", "NONE"] = "ASR" if ordered else "NONE"
        boundaries = _boundary_candidates(
            duration_ms,
            silence=vad_result.long_silence_boundaries_ms,
            sentence_ends=tuple(item.end_ms for item in ordered),
            language_changes=tuple(
                current.start_ms
                for previous, current in pairwise(ordered_language_spans)
                if previous.language != current.language
            ),
        )
        return AudioSpeechAnalysis(
            transcript_source=transcript_source,
            evidence=ordered,
            warnings=tuple(dict.fromkeys(warnings)),
            boundary_candidates=boundaries,
            stage_metrics=(
                AudioStageMetric("SPEECH_ASR", round((time.monotonic() - started_at) * 1_000)),
            ),
        )


def _audio_asr_prompt(config: AudioRunConfig) -> str | None:
    parts: list[str] = []
    if config.core_context:
        parts.append(config.core_context)
    if config.hotwords:
        parts.append(" ".join(config.hotwords))
    return "\n".join(parts) or None


def _boundary_candidates(
    duration_ms: int,
    *,
    silence: tuple[int, ...],
    sentence_ends: tuple[int, ...],
    language_changes: tuple[int, ...],
) -> tuple[AudioSpeechBoundaryCandidate, ...]:
    candidates: set[tuple[int, Literal["silence", "sentence_end", "language_change"], float]] = {
        (timestamp_ms, "silence", 1.0) for timestamp_ms in silence if 0 < timestamp_ms < duration_ms
    }
    candidates.update(
        (timestamp_ms, "sentence_end", 0.8)
        for timestamp_ms in sentence_ends
        if 0 < timestamp_ms < duration_ms
    )
    candidates.update(
        (timestamp_ms, "language_change", 1.0)
        for timestamp_ms in language_changes
        if 0 < timestamp_ms < duration_ms
    )
    return tuple(
        AudioSpeechBoundaryCandidate(timestamp, source, score)
        for timestamp, source, score in sorted(candidates)
    )


class AudioPipeline:
    """独立音频文字流水线：ASR、章节规划和章节写作。"""

    def __init__(
        self,
        speech: AudioSpeechPort,
        planner: AudioChapterPlanner,
        writer: AudioDocumentWriter,
        *,
        evidence_limits: AudioEvidencePreparationLimits,
    ) -> None:
        self._speech = speech
        self._planner = planner
        self._writer = writer
        self._limits = evidence_limits

    def run(
        self,
        *,
        run_id: str,
        asset_sha256: Sha256,
        source: Path,
        duration_ms: int,
        title_hint: str,
        config: AudioRunConfig,
        run_root: Path,
        cache: DocumentModelCache,
        is_cancel_requested: Callable[[], bool],
    ) -> AudioPipelineOutcome:
        speech = self._speech.analyze(
            source,
            duration_ms=duration_ms,
            config=config,
            run_root=run_root,
            is_cancel_requested=is_cancel_requested,
        )
        if not speech.transcript_evidence:
            raise VideoDemoError(ErrorCode.AUDIO_ASR_UNAVAILABLE, "音频没有可验证的语音内容")
        base_segments = build_audio_segments(
            asset_sha256,
            duration_ms,
            speech.transcript_evidence,
            speech.boundary_candidates,
            self._limits,
        )
        planning = self._planner.plan(
            cache=cache,
            asset_sha256=asset_sha256,
            title_hint=title_hint,
            duration_ms=duration_ms,
            segments=base_segments,
            transcript_evidence=speech.transcript_evidence,
            document_config=AudioDocumentConfig(
                document_title=config.document_config.document_title,
                detail_level=config.document_config.detail_level,
                chapter_granularity=config.document_config.chapter_granularity,
                include_verbatim_quotes=config.document_config.include_verbatim_quotes,
            ),
            is_cancel_requested=is_cancel_requested,
        )
        writing = self._writer.write(
            AudioWritingContext(
                run_id=run_id,
                asset_sha256=asset_sha256,
                title_hint=title_hint,
                duration_ms=duration_ms,
                transcript_source=speech.transcript_source,
                document_config=config.document_config,
            ),
            planning.plans,
            speech.transcript_evidence,
            cache=cache,
            is_cancel_requested=is_cancel_requested,
        )
        warnings = tuple(dict.fromkeys((*speech.warnings, *planning.warnings, *writing.warnings)))
        result = writing.result
        return AudioPipelineOutcome(
            result=result,
            document=render_audio_markdown(result),
            evidence=speech.transcript_evidence,
            warnings=warnings,
            status="PARTIAL_SUCCEEDED" if warnings else "SUCCEEDED",
        )
