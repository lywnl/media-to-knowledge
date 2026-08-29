from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, cast

from video_demo.application.audio_rendering import RenderedAudioDocument, render_audio_markdown
from video_demo.application.base_segments import build_base_segments
from video_demo.application.chapter_planning import ChapterPlanner
from video_demo.application.document_writing import DocumentWriter
from video_demo.application.pipeline_contracts import (
    DocumentWritingContext,
    EvidencePreparationLimits,
    PipelineRunConfig,
    SpeechAnalysis,
    SpeechBoundaryCandidate,
    StageMetric,
)
from video_demo.domain.audio_document import (
    AudioChapter,
    AudioDocumentSummary,
    AudioUnderstandingResult,
)
from video_demo.domain.base import Sha256, stable_identifier
from video_demo.domain.evidence import SpeechSegment, SubtitleCue
from video_demo.domain.run import TimeRange
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.speech.asr import WindowRecognizerPort
from video_demo.speech.vad import VadResult
from video_demo.storage.document_cache import DocumentModelCache


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
        config: PipelineRunConfig,
        run_root: Path,
        is_cancel_requested: Callable[[], bool],
    ) -> SpeechAnalysis:
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
    ) -> None:
        self._vad = cast(VadPort, vad)
        self._recognizer = cast(WindowRecognizerPort, recognizer)
        self._slicer = cast(AudioSlicerPort, slicer)
        self._max_window_ms = max_window_ms
        self._overlap_ms = overlap_ms
        self._max_upload_bytes = max_upload_bytes

    def analyze(
        self,
        source: Path,
        *,
        duration_ms: int,
        config: PipelineRunConfig,
        run_root: Path,
        is_cancel_requested: Callable[[], bool],
    ) -> SpeechAnalysis:
        from video_demo.speech.asr import (
            build_cloud_asr_windows,
            project_cloud_asr_window,
            remove_adjacent_cloud_asr_duplicates,
        )

        started_at = time.monotonic()
        vad_result = self._vad.detect(source, duration_ms=duration_ms)
        windows = build_cloud_asr_windows(
            vad_result.speech,
            max_window_ms=self._max_window_ms,
            overlap_ms=self._overlap_ms,
            max_upload_bytes=self._max_upload_bytes,
        )
        language_spans = []
        transcript: list[SpeechSegment] = []
        warnings = list(vad_result.warnings)
        language_hint = config.language_hints[0] if len(config.language_hints) == 1 else None
        for index, window in enumerate(windows, start=1):
            if is_cancel_requested():
                raise VideoDemoError(ErrorCode.JOB_CANCELLED, "任务已请求取消")
            slice_id = stable_identifier(
                "audio-slice",
                {
                    "run_root": run_root.as_posix(),
                    "index": index,
                    "start_ms": window.upload_range.start_ms,
                },
            )
            try:
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
                        prompt=None,
                    )
                    projection = project_cloud_asr_window(
                        window,
                        language=result.language,
                        raw_segments=result.segments,
                        warnings=result.warnings,
                    )
                    language_spans.append(projection.language_span)
                    transcript.extend(projection.segments)
                    warnings.extend(projection.warnings)
                finally:
                    audio_slice.unlink(missing_ok=True)
            except VideoDemoError as error:
                if error.code in {
                    ErrorCode.JOB_CANCELLED,
                    ErrorCode.VIDEO_PROCESS_CANCELLED,
                }:
                    raise
                warnings.append(f"AUDIO_ASR_WINDOW_DEGRADED:{index}")
        ordered = remove_adjacent_cloud_asr_duplicates(tuple(transcript))
        transcript_source: Literal["ASR", "NONE"] = "ASR" if ordered else "NONE"
        boundaries = tuple(
            SpeechBoundaryCandidate(item.end_ms, "sentence_end", 1.0)
            for item in ordered
            if 0 < item.end_ms < duration_ms
        )
        return SpeechAnalysis(
            transcript_source=transcript_source,
            evidence=ordered,
            warnings=tuple(dict.fromkeys(warnings)),
            boundary_candidates=boundaries,
            stage_metrics=(
                StageMetric("SPEECH_ASR", round((time.monotonic() - started_at) * 1_000)),
            ),
        )


class AudioPipeline:
    """独立音频文字流水线：ASR、章节规划、章节写作，不包含任何视觉阶段。"""

    def __init__(
        self,
        speech: AudioSpeechPort,
        planner: ChapterPlanner,
        writer: DocumentWriter,
        *,
        evidence_limits: EvidencePreparationLimits,
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
        config: PipelineRunConfig,
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
        base_segments = build_base_segments(
            asset_sha256,
            duration_ms,
            speech.transcript_evidence,
            (),
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
            scenes=(),
            document_config=config.document_config,
            is_cancel_requested=is_cancel_requested,
        )
        writing = self._writer.write(
            DocumentWritingContext(
                run_id=run_id,
                asset_sha256=asset_sha256,
                title_hint=title_hint,
                duration_ms=duration_ms,
                transcript_source=speech.transcript_source,
                document_config=config.document_config,
            ),
            planning.plans,
            speech.transcript_evidence,
            (),
            (),
            cache=cache,
            is_cancel_requested=is_cancel_requested,
        )
        warnings = tuple(
            dict.fromkeys((*speech.warnings, *planning.warnings, *writing.warnings))
        )
        result = _audio_result(
            run_id,
            asset_sha256,
            writing.result,
            transcript_ids={item.evidence_id for item in speech.transcript_evidence},
        )
        return AudioPipelineOutcome(
            result=result,
            document=render_audio_markdown(result),
            evidence=speech.transcript_evidence,
            warnings=warnings,
            status="PARTIAL_SUCCEEDED" if warnings else "SUCCEEDED",
        )


def _audio_result(
    run_id: str,
    asset_sha256: Sha256,
    result: object,
    *,
    transcript_ids: set[str],
) -> AudioUnderstandingResult:
    from video_demo.domain.document import VideoUnderstandingResult

    assert isinstance(result, VideoUnderstandingResult)
    chapters = tuple(_audio_chapter(chapter, transcript_ids) for chapter in result.chapters)
    return AudioUnderstandingResult(
        run_id=run_id,
        asset_sha256=asset_sha256,
        summary=AudioDocumentSummary(
            title=result.summary.title,
            duration_ms=result.summary.duration_ms,
            overview_zh=result.summary.overview_zh,
        ),
        chapters=chapters,
    )


def _audio_chapter(chapter: object, transcript_ids: set[str]) -> AudioChapter:
    from video_demo.domain.document import SemanticChapter

    assert isinstance(chapter, SemanticChapter)
    allowed = transcript_ids.intersection(chapter.evidence_refs)
    title_refs = tuple(ref for ref in chapter.title_evidence_refs if ref in allowed)
    summary_refs = tuple(ref for ref in chapter.summary_evidence_refs if ref in allowed)
    body_blocks = tuple(
        block.model_copy(
            update={"evidence_refs": tuple(ref for ref in block.evidence_refs if ref in allowed)}
        )
        for block in chapter.body_blocks
        if block.block_type != "VISUAL"
        and any(ref in allowed for ref in block.evidence_refs)
    )
    claims = tuple(
        claim.model_copy(
            update={
                "evidence_refs": tuple(
                    ref for ref in claim.evidence_refs if ref in allowed
                )
            }
        )
        for claim in chapter.claims
        if any(ref in allowed for ref in claim.evidence_refs)
    )
    if not title_refs or not summary_refs or not allowed:
        return AudioChapter(
            start_ms=chapter.start_ms,
            end_ms=chapter.end_ms,
            chapter_id=chapter.chapter_id,
            title=chapter.title,
            title_evidence_refs=(),
            summary_zh="",
            summary_evidence_refs=(),
            body_blocks=(),
            claims=(),
            content_status="NO_SEMANTIC_EVIDENCE",
            evidence_refs=(),
            transcript_source="NONE",
        )
    return AudioChapter(
        start_ms=chapter.start_ms,
        end_ms=chapter.end_ms,
        chapter_id=chapter.chapter_id,
        title=chapter.title,
        title_evidence_refs=title_refs,
        summary_zh=chapter.summary_zh,
        summary_evidence_refs=summary_refs,
        body_blocks=body_blocks,
        claims=claims,
        content_status="GROUNDED",
        evidence_refs=tuple(ref for ref in chapter.evidence_refs if ref in allowed),
        transcript_source=chapter.transcript_source,
    )
