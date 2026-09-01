from __future__ import annotations

import logging
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
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
    AudioTranscriptionCheckpoint,
)
from video_demo.application.audio_document_writing import AudioDocumentWriter, AudioWritingContext
from video_demo.application.audio_rendering import RenderedAudioDocument, render_audio_markdown
from video_demo.application.audio_run_config import AudioRunConfig
from video_demo.application.audio_segments import build_audio_segments
from video_demo.application.audio_speech import discard_audio_slice
from video_demo.domain.audio_document import AudioUnderstandingResult
from video_demo.domain.audio_plan import AudioDocumentConfig
from video_demo.domain.base import Sha256, stable_identifier
from video_demo.domain.evidence import SpeechSegment, SubtitleCue
from video_demo.domain.run import TimeRange
from video_demo.errors import ErrorCode, VideoDemoError, is_cancelled_error_code
from video_demo.speech.asr_contracts import WindowRecognizerPort
from video_demo.speech.audio_fixed_asr import (
    AUDIO_ASR_CHUNK_DURATION_MS,
    AUDIO_ASR_CONCURRENCY,
    AudioFixedAsrProjection,
    AudioFixedAsrWindow,
    build_fixed_audio_asr_windows,
    project_fixed_audio_asr_window,
    remove_adjacent_audio_asr_duplicates,
)
from video_demo.speech.audio_snapshots import (
    AudioAsrFingerprintInputs,
    AudioAsrWindowSnapshotPayload,
    audio_asr_fingerprint,
    audio_asr_window_fingerprint,
)
from video_demo.storage.audio_snapshots import AudioAsrWindowSnapshotStore
from video_demo.storage.document_cache import DocumentModelCache

_LOGGER = logging.getLogger(__name__)


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
        asset_sha256: Sha256,
        duration_ms: int,
        config: AudioRunConfig,
        run_root: Path,
        is_cancel_requested: Callable[[], bool],
    ) -> AudioSpeechAnalysis:
        raise NotImplementedError


class AudioSpeechAnalyzer(AudioSpeechPort):
    """独立音频 ASR：固定十分钟分块、串行识别和可恢复快照。"""

    def __init__(
        self,
        recognizer: object,
        slicer: object,
        *,
        max_upload_bytes: int = 25 * 1024 * 1024,
        window_store: AudioAsrWindowSnapshotStore | None = None,
        fingerprint_inputs: AudioAsrFingerprintInputs | None = None,
    ) -> None:
        self._recognizer = cast(WindowRecognizerPort, recognizer)
        self._slicer = cast(AudioSlicerPort, slicer)
        self._max_upload_bytes = max_upload_bytes
        self._window_store = window_store
        self._fingerprint_inputs = fingerprint_inputs or AudioAsrFingerprintInputs(
            model_id="unspecified",
            base_url="unspecified",
            timeout_seconds=1,
            max_attempts=1,
            max_upload_bytes=max_upload_bytes,
        )

    def analyze(
        self,
        source: Path,
        *,
        asset_sha256: Sha256,
        duration_ms: int,
        config: AudioRunConfig,
        run_root: Path,
        is_cancel_requested: Callable[[], bool],
    ) -> AudioSpeechAnalysis:
        started_at = time.monotonic()
        windows = build_fixed_audio_asr_windows(duration_ms)
        _LOGGER.info(
            "音频 ASR 固定分块完成: chunks=%d chunk_duration_ms=%d concurrency=%d",
            len(windows),
            AUDIO_ASR_CHUNK_DURATION_MS,
            AUDIO_ASR_CONCURRENCY,
        )
        language_hint = _single_language_hint(config.language_hints)
        prompt = _audio_asr_prompt(config)
        parent_fingerprint = audio_asr_fingerprint(
            asset_sha256=asset_sha256,
            duration_ms=duration_ms,
            language_hints=config.language_hints,
            hotwords=config.hotwords,
            core_context=config.core_context,
            inputs=self._fingerprint_inputs,
        )
        results = self._recognize_windows_concurrently(
            source,
            run_root,
            windows,
            parent_fingerprint=parent_fingerprint,
            language_hint=language_hint,
            prompt=prompt,
            is_cancel_requested=is_cancel_requested,
        )
        language_spans = tuple(item.language_span for item in results)
        warnings = tuple(warning for item in results for warning in item.warnings)
        ordered = remove_adjacent_audio_asr_duplicates(
            tuple(
                sorted(
                    (segment for item in results for segment in item.segments),
                    key=lambda item: (item.start_ms, item.end_ms, item.evidence_id),
                ),
            ),
        )
        if windows and not ordered:
            warnings = (*warnings, "AUDIO_ASR_NO_VALID_SEGMENTS")
        ordered_language_spans = tuple(
            sorted(
                language_spans,
                key=lambda item: (item.start_ms, item.end_ms, item.evidence_id),
            ),
        )
        transcript_source: Literal["ASR", "NONE"] = "ASR" if ordered else "NONE"
        boundaries = _boundary_candidates(
            duration_ms,
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

    def _recognize_windows_concurrently(
        self,
        source: Path,
        run_root: Path,
        windows: tuple[AudioFixedAsrWindow, ...],
        *,
        parent_fingerprint: Sha256,
        language_hint: str | None,
        prompt: str | None,
        is_cancel_requested: Callable[[], bool],
    ) -> tuple[AudioAsrWindowSnapshotPayload, ...]:
        results: dict[int, AudioAsrWindowSnapshotPayload] = {}
        futures: dict[Future[AudioAsrWindowSnapshotPayload], AudioFixedAsrWindow] = {}
        first_error: Exception | None = None
        with ThreadPoolExecutor(
            max_workers=AUDIO_ASR_CONCURRENCY,
            thread_name_prefix="audio-asr",
        ) as executor:
            for window in windows:
                futures[executor.submit(
                    self._recognize_window,
                    source,
                    run_root,
                    window,
                    parent_fingerprint=parent_fingerprint,
                    total_chunks=len(windows),
                    language_hint=language_hint,
                    prompt=prompt,
                    is_cancel_requested=is_cancel_requested,
                )] = window
            for future in as_completed(futures):
                window = futures[future]
                try:
                    results[window.chunk_index] = future.result()
                except Exception as error:
                    first_error = error
                    _LOGGER.warning(
                        "音频 ASR 块失败 chunk=%d/%d error=%s",
                        window.chunk_index + 1,
                        len(windows),
                        getattr(error, "code", type(error).__name__),
                    )
                    for pending in futures:
                        if pending is not future:
                            pending.cancel()
                    break
        if first_error is not None:
            if isinstance(first_error, VideoDemoError) and is_cancelled_error_code(
                first_error.code,
            ):
                raise VideoDemoError(ErrorCode.JOB_CANCELLED, "音频分析已取消") from first_error
            raise first_error
        if len(results) != len(windows):
            raise VideoDemoError(ErrorCode.AUDIO_ASR_UNAVAILABLE, "音频 ASR 块结果不完整")
        return tuple(results[index] for index in range(len(windows)))

    def _recognize_window(
        self,
        source: Path,
        run_root: Path,
        window: AudioFixedAsrWindow,
        *,
        parent_fingerprint: Sha256,
        total_chunks: int,
        language_hint: str | None,
        prompt: str | None,
        is_cancel_requested: Callable[[], bool],
    ) -> AudioAsrWindowSnapshotPayload:
        if is_cancel_requested():
            raise VideoDemoError(ErrorCode.JOB_CANCELLED, "任务已请求取消")
        fingerprint = audio_asr_window_fingerprint(
            asr_fingerprint=parent_fingerprint,
            window=window,
        )
        if self._window_store is not None:
            cached = self._window_store.load(run_root, fingerprint)
            if cached is not None:
                _LOGGER.info(
                    "音频 ASR 块命中快照 chunk=%d/%d segments=%d",
                    window.chunk_index + 1,
                    total_chunks,
                    len(cached[0].segments),
                )
                return cached[0]
        started_at = time.monotonic()
        _LOGGER.info(
            "音频 ASR 块开始 chunk=%d range=%d-%dms",
            window.chunk_index + 1,
            window.upload_range.start_ms,
            window.upload_range.end_ms,
        )
        slice_id = stable_identifier(
            "audio-slice",
            {
                "run_root": run_root.as_posix(),
                "chunk_index": window.chunk_index,
                "window_fingerprint": fingerprint,
            },
        )
        audio_slice = self._slicer.create(source, run_root, slice_id, window.upload_range)
        try:
            if audio_slice.stat().st_size > self._max_upload_bytes:
                raise VideoDemoError(ErrorCode.AUDIO_OUTPUT_TOO_LARGE, "ASR 音频切片超过大小限制")
            result = self._recognizer.transcribe_window(
                audio_slice,
                language_hint=language_hint,
                prompt=prompt,
                chunk_index=window.chunk_index,
            )
            projection: AudioFixedAsrProjection = project_fixed_audio_asr_window(
                window,
                language=result.language,
                raw_segments=result.segments,
                warnings=result.warnings,
            )
            payload = AudioAsrWindowSnapshotPayload(
                chunk_index=window.chunk_index,
                upload_range=window.upload_range,
                owned_range=window.owned_range,
                language_span=projection.language_span,
                segments=projection.segments,
                warnings=projection.warnings,
            )
            if self._window_store is not None:
                self._window_store.publish(run_root, fingerprint, payload)
            _LOGGER.info(
                "音频 ASR 块完成 chunk=%d segments=%d elapsed=%.3fs",
                window.chunk_index + 1,
                len(payload.segments),
                time.monotonic() - started_at,
            )
            return payload
        finally:
            discard_audio_slice(audio_slice)


def _audio_asr_prompt(config: AudioRunConfig) -> str | None:
    parts: list[str] = []
    if config.core_context:
        parts.append(config.core_context)
    if config.hotwords:
        parts.append(" ".join(config.hotwords))
    return "\n".join(parts) or None


def _single_language_hint(hints: tuple[str, ...]) -> str | None:
    return hints[0] if len(hints) == 1 and hints[0] != "und" else None


def _boundary_candidates(
    duration_ms: int,
    *,
    sentence_ends: tuple[int, ...],
    language_changes: tuple[int, ...],
) -> tuple[AudioSpeechBoundaryCandidate, ...]:
    candidates: set[tuple[int, Literal["sentence_end", "language_change"], float]] = {
        (timestamp_ms, "sentence_end", 0.8)
        for timestamp_ms in sentence_ends
        if 0 < timestamp_ms < duration_ms
    }
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
        checkpoint = self.run_transcription(
            run_id=run_id,
            asset_sha256=asset_sha256,
            source=source,
            duration_ms=duration_ms,
            title_hint=title_hint,
            config=config,
            run_root=run_root,
            is_cancel_requested=is_cancel_requested,
        )
        return self.run_llm(
            checkpoint,
            config=config,
            cache=cache,
            is_cancel_requested=is_cancel_requested,
        )

    def run_transcription(
        self,
        *,
        run_id: str,
        asset_sha256: Sha256,
        source: Path,
        duration_ms: int,
        title_hint: str,
        config: AudioRunConfig,
        run_root: Path,
        is_cancel_requested: Callable[[], bool],
    ) -> AudioTranscriptionCheckpoint:
        speech = self._speech.analyze(
            source,
            asset_sha256=asset_sha256,
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
        return AudioTranscriptionCheckpoint(
            run_id=run_id,
            asset_sha256=asset_sha256,
            duration_ms=duration_ms,
            title_hint=title_hint,
            transcript_source=speech.transcript_source,
            transcript_evidence=speech.transcript_evidence,
            base_segments=base_segments,
            warnings=speech.warnings,
            stage_metrics=speech.stage_metrics,
        ).validate_consistency()

    def run_llm(
        self,
        checkpoint: AudioTranscriptionCheckpoint,
        *,
        config: AudioRunConfig,
        cache: DocumentModelCache,
        is_cancel_requested: Callable[[], bool],
    ) -> AudioPipelineOutcome:
        checkpoint.validate_consistency()
        planning = self._planner.plan(
            cache=cache,
            asset_sha256=checkpoint.asset_sha256,
            title_hint=checkpoint.title_hint,
            duration_ms=checkpoint.duration_ms,
            segments=checkpoint.base_segments,
            transcript_evidence=checkpoint.transcript_evidence,
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
                run_id=checkpoint.run_id,
                asset_sha256=checkpoint.asset_sha256,
                title_hint=checkpoint.title_hint,
                duration_ms=checkpoint.duration_ms,
                transcript_source=checkpoint.transcript_source,
                document_config=config.document_config,
            ),
            planning.plans,
            checkpoint.transcript_evidence,
            cache=cache,
            is_cancel_requested=is_cancel_requested,
        )
        warnings = tuple(
            dict.fromkeys((*checkpoint.warnings, *planning.warnings, *writing.warnings))
        )
        result = writing.result
        return AudioPipelineOutcome(
            result=result,
            document=render_audio_markdown(result),
            evidence=checkpoint.transcript_evidence,
            warnings=warnings,
            status="PARTIAL_SUCCEEDED" if warnings else "SUCCEEDED",
        )
