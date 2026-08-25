from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from video_demo.application.pipeline import PreparedMedia, SpeechAnalysis
from video_demo.domain.evidence import (
    KeyframeEvidence,
    OcrEvidence,
    SpeechSegment,
    SubtitleCue,
)
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.storage.workspace import safe_runtime_path
from video_demo.visual.ocr import (
    KeyframeForOcr,
    OcrClient,
    OcrDeadlineExceeded,
    OcrProcessor,
)
from video_demo.visual.ocr_budget import (
    OcrClassification,
    OcrFrameObservation,
    OcrTextAssessment,
    assess_probe_text,
    batch_new_text_count,
    calculate_ocr_budget,
    effective_frame_texts,
    extend_keyframes,
    select_probe_keyframes,
)

_OCR_LANGUAGES = frozenset({"zh", "en", "ja", "ko", "es"})
_OCR_BUDGET_TIME_LIMIT_WARNING = "OCR_BUDGET_TIME_LIMIT_REACHED"


@dataclass(frozen=True, slots=True)
class AdaptiveOcrResult:
    keyframes: tuple[KeyframeEvidence, ...]
    evidence: tuple[OcrEvidence, ...]
    warnings: tuple[str, ...]
    assessment: OcrTextAssessment
    provider_attempt_count: int
    image_request_count: int
    stop_reason: str
    elapsed_ms: int


@dataclass(frozen=True, slots=True)
class _OcrBatchResult:
    keyframes: tuple[KeyframeEvidence, ...]
    evidence: tuple[OcrEvidence, ...]
    observations: tuple[OcrFrameObservation, ...]
    warnings: tuple[str, ...]
    provider_attempt_count: int
    image_request_count: int
    reached_deadline: bool


class AdaptiveOcrRunner:
    """按文字价值分批执行 OCR，并把成本决策隔离在视觉主编排之外。"""

    def __init__(
        self,
        runtime_root: Path,
        *,
        timeout_seconds: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("OCR 预算超时必须大于 0")
        self._runtime_root = runtime_root.expanduser().resolve(strict=False)
        self._timeout_seconds = timeout_seconds
        self._clock = clock

    def run(
        self,
        keyframes: Sequence[KeyframeEvidence],
        *,
        speech: SpeechAnalysis | None,
        media: PreparedMedia,
        run_relative_root: Path,
        client: OcrClient,
        is_cancel_requested: Callable[[], bool],
    ) -> AdaptiveOcrResult:
        budget = calculate_ocr_budget(media.source.duration_ms)
        probes = select_probe_keyframes(
            keyframes,
            duration_ms=media.source.duration_ms,
            count=budget.probe,
        )
        started_at = self._clock()
        deadline = started_at + self._timeout_seconds
        processor = OcrProcessor(
            client,
            allowed_root=safe_runtime_path(self._runtime_root, run_relative_root),
            clock=self._clock,
        )
        probe_result = self._run_batch(
            probes,
            speech=speech,
            media=media,
            processor=processor,
            deadline=deadline,
            is_cancel_requested=is_cancel_requested,
        )
        assessment = assess_probe_text(
            probe_result.observations,
            duration_ms=media.source.duration_ms,
            has_subtitle_track=speech is not None and speech.transcript_source == "SUBTITLE",
        )
        _check_cancelled(is_cancel_requested)
        selected = list(probe_result.keyframes)
        evidence = list(probe_result.evidence)
        observations = list(probe_result.observations)
        warnings = list(probe_result.warnings)
        provider_attempt_count = probe_result.provider_attempt_count
        image_request_count = probe_result.image_request_count
        if probe_result.reached_deadline or self._clock() >= deadline:
            warnings.append(_OCR_BUDGET_TIME_LIMIT_WARNING)
            stop_reason = "TIME_LIMIT_REACHED"
        elif len(selected) == len(keyframes):
            stop_reason = "CANDIDATES_EXHAUSTED"
        elif assessment.classification == OcrClassification.LOW_TEXT:
            stop_reason = "LOW_TEXT"
        elif assessment.classification == OcrClassification.DENSE_TEXT:
            stop_reason, provider_attempt_count, image_request_count = self._extend_dense(
                keyframes,
                selected=selected,
                evidence=evidence,
                observations=observations,
                warnings=warnings,
                assessment=assessment,
                hard_limit=budget.hard_limit,
                speech=speech,
                media=media,
                processor=processor,
                deadline=deadline,
                is_cancel_requested=is_cancel_requested,
                provider_attempt_count=provider_attempt_count,
                image_request_count=image_request_count,
            )
        else:
            stop_reason, provider_attempt_count, image_request_count = self._extend_normal(
                keyframes,
                selected=selected,
                evidence=evidence,
                observations=observations,
                warnings=warnings,
                base_limit=budget.base,
                speech=speech,
                media=media,
                processor=processor,
                deadline=deadline,
                is_cancel_requested=is_cancel_requested,
                provider_attempt_count=provider_attempt_count,
                image_request_count=image_request_count,
            )
        elapsed_ms = max(0, round((self._clock() - started_at) * 1_000))
        return AdaptiveOcrResult(
            keyframes=tuple(selected),
            evidence=tuple(evidence),
            warnings=tuple(dict.fromkeys(warnings)),
            assessment=assessment,
            provider_attempt_count=provider_attempt_count,
            image_request_count=image_request_count,
            stop_reason=stop_reason,
            elapsed_ms=elapsed_ms,
        )

    def log_result(
        self,
        logger: logging.Logger,
        *,
        duration_ms: int,
        candidate_count: int,
        result: AdaptiveOcrResult,
    ) -> None:
        budget = calculate_ocr_budget(duration_ms)
        logger.info(
            "ocr_budget_complete",
            extra={
                "ocr_duration_ms": duration_ms,
                "ocr_candidate_count": candidate_count,
                "ocr_probe_budget": budget.probe,
                "ocr_base_budget": budget.base,
                "ocr_hard_limit": budget.hard_limit,
                "ocr_selected_keyframe_count": len(result.keyframes),
                "ocr_successful_image_count": len(result.evidence),
                "ocr_provider_attempt_count": result.provider_attempt_count,
                "ocr_image_request_count": result.image_request_count,
                "ocr_valid_text_ratio": float(result.assessment.valid_text_ratio),
                "ocr_text_change_ratio": float(result.assessment.text_change_ratio),
                "ocr_median_effective_chars": result.assessment.median_effective_chars,
                "ocr_time_coverage_ratio": float(result.assessment.time_coverage_ratio),
                "ocr_classification": result.assessment.classification.value,
                "ocr_stop_reason": result.stop_reason,
                "ocr_elapsed_ms": result.elapsed_ms,
            },
        )

    def _run_batch(
        self,
        keyframes: Sequence[KeyframeEvidence],
        *,
        speech: SpeechAnalysis | None,
        media: PreparedMedia,
        processor: OcrProcessor,
        deadline: float,
        is_cancel_requested: Callable[[], bool],
    ) -> _OcrBatchResult:
        completed: list[KeyframeEvidence] = []
        evidence: list[OcrEvidence] = []
        observations: list[OcrFrameObservation] = []
        warnings: list[str] = []
        provider_attempt_count = 0
        image_request_count = 0
        for keyframe in keyframes:
            _check_cancelled(is_cancel_requested)
            if self._clock() >= deadline:
                return _OcrBatchResult(
                    keyframes=tuple(completed),
                    evidence=tuple(evidence),
                    observations=tuple(observations),
                    warnings=tuple(warnings),
                    provider_attempt_count=provider_attempt_count,
                    image_request_count=image_request_count,
                    reached_deadline=True,
                )
            language, warning = ocr_language(keyframe.timestamp_ms, speech, media)
            if warning is not None:
                warnings.append(warning)
            if language is None:
                completed.append(keyframe)
                observations.append(_empty_observation(keyframe.timestamp_ms))
                continue
            item = KeyframeForOcr(
                keyframe_id=keyframe.keyframe_id,
                source_sha256=keyframe.sha256,
                start_ms=keyframe.start_ms,
                end_ms=keyframe.end_ms,
                timestamp_ms=keyframe.timestamp_ms,
                path=self._runtime_root / keyframe.relative_path,
                language=language,
            )
            try:
                processed = processor.process_with_diagnostics(
                    (item,), deadline=deadline, reuse_cache=True
                )
            except OcrDeadlineExceeded as error:
                _check_cancelled(is_cancel_requested)
                return _OcrBatchResult(
                    keyframes=tuple(completed),
                    evidence=tuple(evidence),
                    observations=tuple(observations),
                    warnings=tuple(warnings),
                    provider_attempt_count=(
                        provider_attempt_count + error.provider_attempt_count
                    ),
                    image_request_count=(
                        image_request_count + error.image_request_count
                    ),
                    reached_deadline=True,
                )
            _check_cancelled(is_cancel_requested)
            completed.append(keyframe)
            evidence.extend(processed.evidence)
            provider_attempt_count += processed.provider_attempt_count
            image_request_count += processed.image_request_count
            width, height = processed.image_sizes[0]
            observations.append(
                OcrFrameObservation(
                    timestamp_ms=keyframe.timestamp_ms,
                    lines=processed.evidence[0].lines,
                    image_width=width,
                    image_height=height,
                ),
            )
        return _OcrBatchResult(
            keyframes=tuple(completed),
            evidence=tuple(evidence),
            observations=tuple(observations),
            warnings=tuple(warnings),
            provider_attempt_count=provider_attempt_count,
            image_request_count=image_request_count,
            reached_deadline=False,
        )

    def _extend_normal(
        self,
        keyframes: Sequence[KeyframeEvidence],
        *,
        selected: list[KeyframeEvidence],
        evidence: list[OcrEvidence],
        observations: list[OcrFrameObservation],
        warnings: list[str],
        base_limit: int,
        speech: SpeechAnalysis | None,
        media: PreparedMedia,
        processor: OcrProcessor,
        deadline: float,
        is_cancel_requested: Callable[[], bool],
        provider_attempt_count: int,
        image_request_count: int,
    ) -> tuple[str, int, int]:
        target = extend_keyframes(
            keyframes,
            selected,
            count=max(0, base_limit - len(selected)),
        )
        batch = self._run_batch(
            target[len(selected) :],
            speech=speech,
            media=media,
            processor=processor,
            deadline=deadline,
            is_cancel_requested=is_cancel_requested,
        )
        _append_batch(batch, selected, evidence, observations, warnings)
        provider_attempt_count += batch.provider_attempt_count
        image_request_count += batch.image_request_count
        _check_cancelled(is_cancel_requested)
        if batch.reached_deadline or self._clock() >= deadline:
            warnings.append(_OCR_BUDGET_TIME_LIMIT_WARNING)
            return "TIME_LIMIT_REACHED", provider_attempt_count, image_request_count
        if len(selected) < base_limit:
            return "CANDIDATES_EXHAUSTED", provider_attempt_count, image_request_count
        return "BASE_BUDGET_REACHED", provider_attempt_count, image_request_count

    def _extend_dense(
        self,
        keyframes: Sequence[KeyframeEvidence],
        *,
        selected: list[KeyframeEvidence],
        evidence: list[OcrEvidence],
        observations: list[OcrFrameObservation],
        warnings: list[str],
        assessment: OcrTextAssessment,
        hard_limit: int,
        speech: SpeechAnalysis | None,
        media: PreparedMedia,
        processor: OcrProcessor,
        deadline: float,
        is_cancel_requested: Callable[[], bool],
        provider_attempt_count: int,
        image_request_count: int,
    ) -> tuple[str, int, int]:
        seen_texts = list(assessment.frame_texts)
        has_subtitle_track = speech is not None and speech.transcript_source == "SUBTITLE"
        while len(selected) < hard_limit:
            batch_size = min(3, hard_limit - len(selected))
            target = extend_keyframes(keyframes, selected, count=batch_size)
            _check_cancelled(is_cancel_requested)
            if self._clock() >= deadline:
                warnings.append(_OCR_BUDGET_TIME_LIMIT_WARNING)
                return "TIME_LIMIT_REACHED", provider_attempt_count, image_request_count
            candidates = target[len(selected) :]
            if not candidates:
                return "CANDIDATES_EXHAUSTED", provider_attempt_count, image_request_count
            batch = self._run_batch(
                candidates,
                speech=speech,
                media=media,
                processor=processor,
                deadline=deadline,
                is_cancel_requested=is_cancel_requested,
            )
            _append_batch(batch, selected, evidence, observations, warnings)
            provider_attempt_count += batch.provider_attempt_count
            image_request_count += batch.image_request_count
            if batch.reached_deadline:
                warnings.append(_OCR_BUDGET_TIME_LIMIT_WARNING)
                return "TIME_LIMIT_REACHED", provider_attempt_count, image_request_count
            if len(batch.keyframes) < batch_size:
                return "CANDIDATES_EXHAUSTED", provider_attempt_count, image_request_count
            new_text_count = batch_new_text_count(
                batch.observations,
                previous_texts=seen_texts,
                fixed_line_keys=assessment.fixed_line_keys,
                has_subtitle_track=has_subtitle_track,
            )
            _check_cancelled(is_cancel_requested)
            if self._clock() >= deadline:
                warnings.append(_OCR_BUDGET_TIME_LIMIT_WARNING)
                return "TIME_LIMIT_REACHED", provider_attempt_count, image_request_count
            seen_texts.extend(
                effective_frame_texts(
                    batch.observations,
                    fixed_line_keys=assessment.fixed_line_keys,
                    has_subtitle_track=has_subtitle_track,
                ),
            )
            _check_cancelled(is_cancel_requested)
            if self._clock() >= deadline:
                warnings.append(_OCR_BUDGET_TIME_LIMIT_WARNING)
                return "TIME_LIMIT_REACHED", provider_attempt_count, image_request_count
            if batch_size == 3 and new_text_count < 2:
                return "MARGINAL_VALUE_LOW", provider_attempt_count, image_request_count
        return "HARD_LIMIT_REACHED", provider_attempt_count, image_request_count


def ocr_language(
    timestamp_ms: int,
    speech: SpeechAnalysis | None,
    media: PreparedMedia,
) -> tuple[str | None, str | None]:
    timed_languages: list[SubtitleCue | SpeechSegment] = []
    if speech is not None:
        timed_languages.extend(
            item
            for item in speech.evidence
            if isinstance(item, (SubtitleCue, SpeechSegment))
            and item.start_ms <= timestamp_ms < item.end_ms
        )
    if media.subtitle is not None:
        timed_languages.extend(
            item
            for item in media.subtitle.cues
            if item.start_ms <= timestamp_ms < item.end_ms
        )
    timed_languages.sort(key=lambda item: (item.start_ms, item.end_ms, item.evidence_id))
    language = timed_languages[0].language if timed_languages else None
    if language is None:
        language = next(
            (item for item in media.source.asset.config.language_hints if item in _OCR_LANGUAGES),
            None,
        )
    if language is None:
        return "zh", "OCR_LANGUAGE_FALLBACK:zh"
    if language not in _OCR_LANGUAGES:
        return None, f"OCR_LANGUAGE_UNSUPPORTED:{language}"
    return language, None


def _append_batch(
    batch: _OcrBatchResult,
    selected: list[KeyframeEvidence],
    evidence: list[OcrEvidence],
    observations: list[OcrFrameObservation],
    warnings: list[str],
) -> None:
    selected.extend(batch.keyframes)
    evidence.extend(batch.evidence)
    observations.extend(batch.observations)
    warnings.extend(batch.warnings)


def _empty_observation(timestamp_ms: int) -> OcrFrameObservation:
    return OcrFrameObservation(
        timestamp_ms=timestamp_ms,
        lines=(),
        image_width=None,
        image_height=None,
    )


def _check_cancelled(is_cancel_requested: Callable[[], bool]) -> None:
    if is_cancel_requested():
        raise VideoDemoError(ErrorCode.JOB_CANCELLED, "任务已请求取消")


__all__ = ["AdaptiveOcrResult", "AdaptiveOcrRunner", "ocr_language"]
