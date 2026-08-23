from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Protocol

from video_demo.application.pipeline import (
    PreparedMedia,
    SpeechAnalysis,
    SpeechBoundaryCandidate,
)
from video_demo.domain.base import stable_identifier
from video_demo.domain.evidence import (
    AlignedWord,
    AudioEvent,
    EvidenceItem,
    SpeakerTurn,
    SpeechSegment,
)
from video_demo.domain.run import TimeRange
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.media.transcode import AudioSliceArtifact
from video_demo.speech.alignment import AlignmentResult
from video_demo.speech.asr import RawAsrSegment, build_speech_segments
from video_demo.speech.language import LanguageIdentificationResult, LanguageSpan
from video_demo.speech.snapshots import (
    AsrSnapshotPayload,
    SpeechAnalysisSnapshotPayload,
    SpeechFingerprintInputs,
    _speech_fingerprint_v1,
    asr_fingerprint,
    speech_fingerprint,
    subtitle_transcript_payload_sha256,
)
from video_demo.speech.speaker_assignment import assign_speakers
from video_demo.speech.vad import SpeechInterval, VadResult
from video_demo.storage.snapshots import SnapshotStore
from video_demo.storage.workspace import safe_runtime_path


class VadPort(Protocol):
    def detect(self, audio: Path, *, duration_ms: int) -> VadResult: ...


class LanguagePort(Protocol):
    def identify(
        self,
        audio: Path,
        speech: Sequence[SpeechInterval],
        hints: tuple[str, ...],
    ) -> LanguageIdentificationResult: ...


class RecognizerPort(Protocol):
    def transcribe_slice(
        self,
        audio_slice: Path,
        language_span: LanguageSpan,
        *,
        hotwords: tuple[str, ...] = (),
        core_context: str | None = None,
    ) -> tuple[RawAsrSegment, ...]: ...


class AlignerPort(Protocol):
    def align(self, audio: Path, segments: Sequence[SpeechSegment]) -> AlignmentResult: ...


class DiarizerPort(Protocol):
    def diarize(
        self,
        audio: Path,
        *,
        min_speakers: int | None,
        max_speakers: int | None,
    ) -> tuple[SpeakerTurn, ...]: ...


class AudioEventPort(Protocol):
    def detect(self, audio: Path, *, duration_ms: int) -> tuple[AudioEvent, ...]: ...


class AudioSlicer(Protocol):
    def create(
        self,
        audio: Path,
        run_relative_root: Path,
        slice_id: str,
        time_range: TimeRange,
    ) -> Path: ...


@dataclass(frozen=True)
class SpeechComponents:
    vad: VadPort
    language_identifier: LanguagePort
    recognizer: RecognizerPort
    aligner: AlignerPort
    diarizer: DiarizerPort
    audio_events: AudioEventPort
    slicer: AudioSlicer


ComponentFactory = Callable[[PreparedMedia, Callable[[], bool]], SpeechComponents]


class ProductionSpeechAnalyzer:
    """只编排生产语音 Port，不在构造时导入或加载重模型。"""

    def __init__(
        self,
        component_factory: ComponentFactory,
        *,
        snapshot_store: SnapshotStore,
        fingerprint_inputs: SpeechFingerprintInputs,
        allow_speaker_fallback: bool = False,
    ) -> None:
        self._component_factory = component_factory
        self._snapshot_store = snapshot_store
        self._fingerprint_inputs = fingerprint_inputs
        self._allow_speaker_fallback = allow_speaker_fallback

    def analyze(
        self,
        media: PreparedMedia,
        *,
        is_cancel_requested: Callable[[], bool] = lambda: False,
    ) -> SpeechAnalysis:
        if media.subtitle is not None:
            analysis = SpeechAnalysis(
                transcript_source="SUBTITLE",
                enrichment_mode="text",
                evidence=media.subtitle.cues,
                warnings=tuple(
                    dict.fromkeys((*media.warnings, "TRANSCRIPT_SOURCE_SUBTITLE"))
                ),
                boundary_candidates=tuple(
                    SpeechBoundaryCandidate(cue.end_ms, "sentence_end", 1.0)
                    for cue in media.subtitle.cues
                    if 0 < cue.end_ms < media.source.duration_ms
                ),
            )
            fingerprint = speech_fingerprint(
                processing_mode="SUBTITLE",
                transcript_payload_sha256=subtitle_transcript_payload_sha256(
                    analysis,
                    media.subtitle.artifact.sha256,
                ),
                media_warnings=media.warnings,
                min_speakers=media.source.asset.config.min_speakers,
                max_speakers=media.source.asset.config.max_speakers,
                allow_speaker_fallback=self._allow_speaker_fallback,
                inputs=self._fingerprint_inputs,
                enrichment_mode="text",
            )
            cached = self._snapshot_store.load(
                media.source.asset.run_relative_root,
                "speech",
                fingerprint,
                SpeechAnalysisSnapshotPayload,
            )
            if cached is None:
                legacy_fingerprint = _speech_fingerprint_v1(
                    processing_mode="SUBTITLE",
                    transcript_payload_sha256=subtitle_transcript_payload_sha256(
                        analysis,
                        media.subtitle.artifact.sha256,
                    ),
                    media_warnings=media.warnings,
                    min_speakers=media.source.asset.config.min_speakers,
                    max_speakers=media.source.asset.config.max_speakers,
                    allow_speaker_fallback=self._allow_speaker_fallback,
                    inputs=self._fingerprint_inputs,
                )
                cached = self._snapshot_store.load(
                    media.source.asset.run_relative_root,
                    "speech",
                    legacy_fingerprint,
                    SpeechAnalysisSnapshotPayload,
                )
            if cached is not None:
                cached_analysis = cached[0].to_analysis()
                return SpeechAnalysis(
                    transcript_source=cached_analysis.transcript_source,
                    enrichment_mode="text",
                    evidence=cached_analysis.evidence,
                    warnings=cached_analysis.warnings,
                    boundary_candidates=cached_analysis.boundary_candidates,
                )
            self._snapshot_store.publish(
                media.source.asset.run_relative_root,
                "speech",
                fingerprint,
                SpeechAnalysisSnapshotPayload.from_analysis(analysis),
            )
            return analysis
        if media.audio_path is None:
            return SpeechAnalysis(
                transcript_source="NONE",
                enrichment_mode="text",
                evidence=(),
                warnings=tuple(dict.fromkeys((*media.warnings, "NO_AUDIO_TRACK"))),
            )
        try:
            return self._analyze_asr(
                media,
                is_cancel_requested,
            )
        except VideoDemoError:
            raise
        except (ModuleNotFoundError, ImportError):
            raise VideoDemoError(
                ErrorCode.SPEECH_DEPENDENCY_UNAVAILABLE,
                "语音分析可选依赖不可用",
            ) from None
        except PermissionError:
            raise VideoDemoError(
                ErrorCode.SPEECH_AUTHENTICATION_FAILED,
                "语音模型鉴权失败",
            ) from None
        except (LookupError, OSError, RuntimeError):
            raise VideoDemoError(
                ErrorCode.SPEECH_MODEL_UNAVAILABLE,
                "语音模型不可用",
            ) from None

    def _analyze_asr(
        self,
        media: PreparedMedia,
        is_cancel_requested: Callable[[], bool],
    ) -> SpeechAnalysis:
        assert media.audio_sha256 is not None
        run_root = media.source.asset.run_relative_root
        config = media.source.asset.config
        asr_key = asr_fingerprint(
            audio_sha256=media.audio_sha256,
            duration_ms=media.source.duration_ms,
            language_hints=config.language_hints,
            hotwords=config.hotwords,
            core_context=config.core_context,
            inputs=self._fingerprint_inputs,
        )
        cached_asr = self._snapshot_store.load(
            run_root,
            "asr",
            asr_key,
            AsrSnapshotPayload,
        )
        components: SpeechComponents | None = None
        if cached_asr is None:
            components = self._component_factory(media, is_cancel_requested)
            asr_payload = self.run_asr_stage(media, components)
            asr_receipt = self._snapshot_store.publish(
                run_root,
                "asr",
                asr_key,
                asr_payload,
            )
        else:
            asr_payload, asr_receipt = cached_asr

        speech_key = speech_fingerprint(
            processing_mode="ASR",
            transcript_payload_sha256=asr_receipt.sha256,
            media_warnings=media.warnings,
            min_speakers=config.min_speakers,
            max_speakers=config.max_speakers,
            allow_speaker_fallback=self._allow_speaker_fallback,
            inputs=self._fingerprint_inputs,
            enrichment_mode=config.speech_enrichment_mode,
        )
        cached_speech = self._snapshot_store.load(
            run_root,
            "speech",
            speech_key,
            SpeechAnalysisSnapshotPayload,
        )
        if cached_speech is None and config.speech_enrichment_mode == "full":
            legacy_key = _speech_fingerprint_v1(
                processing_mode="ASR",
                transcript_payload_sha256=asr_receipt.sha256,
                media_warnings=media.warnings,
                min_speakers=config.min_speakers,
                max_speakers=config.max_speakers,
                allow_speaker_fallback=self._allow_speaker_fallback,
                inputs=self._fingerprint_inputs,
            )
            cached_speech = self._snapshot_store.load(
                run_root,
                "speech",
                legacy_key,
                SpeechAnalysisSnapshotPayload,
            )
        if cached_speech is not None:
            return cached_speech[0].to_analysis()
        if config.speech_enrichment_mode == "text":
            analysis = self.analysis_from_asr_snapshot(
                media,
                asr_payload,
                enrichment_mode="text",
            )
            self._snapshot_store.publish(
                run_root,
                "speech",
                speech_key,
                SpeechAnalysisSnapshotPayload.from_analysis(analysis),
            )
            return analysis
        if components is None:
            components = self._component_factory(media, is_cancel_requested)
        analysis = self.enrich_from_asr_snapshot(
            media,
            components,
            asr_payload,
            allow_speaker_fallback=self._allow_speaker_fallback,
        )
        self._snapshot_store.publish(
            run_root,
            "speech",
            speech_key,
            SpeechAnalysisSnapshotPayload.from_analysis(analysis),
        )
        return analysis

    @staticmethod
    def run_asr_stage(
        media: PreparedMedia,
        components: SpeechComponents,
    ) -> AsrSnapshotPayload:
        assert media.audio_path is not None
        audio = media.audio_path
        duration_ms = media.source.duration_ms
        vad = components.vad.detect(audio, duration_ms=duration_ms)
        if not vad.speech:
            return AsrSnapshotPayload(
                language_spans=(),
                segments=(),
                vad_warnings=vad.warnings,
                silence_boundaries_ms=vad.long_silence_boundaries_ms,
                language_change_boundaries_ms=(),
            )

        languages = _identify_languages(media, components, vad.speech)
        segments: list[SpeechSegment] = []
        for language_span in languages.spans:
            audio_slice = components.slicer.create(
                audio,
                media.source.asset.run_relative_root,
                f"asr_{language_span.evidence_id}",
                language_span,
            )
            raw_segments = components.recognizer.transcribe_slice(
                audio_slice,
                language_span,
                hotwords=media.source.asset.config.hotwords,
                core_context=media.source.asset.config.core_context,
            )
            segments.extend(build_speech_segments(language_span, raw_segments))
        return AsrSnapshotPayload(
            language_spans=languages.spans,
            segments=tuple(segments),
            vad_warnings=vad.warnings,
            silence_boundaries_ms=vad.long_silence_boundaries_ms,
            language_change_boundaries_ms=languages.change_boundaries_ms,
        )

    @staticmethod
    def analysis_from_asr_snapshot(
        media: PreparedMedia,
        asr_payload: AsrSnapshotPayload,
        *,
        enrichment_mode: str = "text",
    ) -> SpeechAnalysis:
        """将 ASR 快照投影为基础文本分析，不触发任何可选增强模型。"""
        if enrichment_mode != "text":
            raise ValueError("基础 ASR 分析必须使用 text 模式")
        evidence: tuple[EvidenceItem, ...] = tuple(asr_payload.segments)
        warnings = tuple(
            dict.fromkeys((*media.warnings, *asr_payload.vad_warnings))
        )
        if not asr_payload.language_spans:
            warnings = tuple(dict.fromkeys((*warnings, "NO_SPEECH_DETECTED")))
        return SpeechAnalysis(
            transcript_source="ASR",
            enrichment_mode="text",
            evidence=_sort_evidence(evidence),
            warnings=warnings,
            boundary_candidates=_boundary_candidates(
                media.source.duration_ms,
                silence=asr_payload.silence_boundaries_ms,
                sentence_ends=tuple(item.end_ms for item in asr_payload.segments),
                language_changes=asr_payload.language_change_boundaries_ms,
            ),
        )

    @staticmethod
    def enrich_from_asr_snapshot(
        media: PreparedMedia,
        components: SpeechComponents,
        asr_payload: AsrSnapshotPayload,
        *,
        allow_speaker_fallback: bool = False,
    ) -> SpeechAnalysis:
        assert media.audio_path is not None
        audio = media.audio_path
        duration_ms = media.source.duration_ms
        if not asr_payload.language_spans:
            audio_events = components.audio_events.detect(audio, duration_ms=duration_ms)
            return SpeechAnalysis(
                transcript_source="ASR",
                enrichment_mode="full",
                evidence=_sort_evidence(audio_events),
                warnings=tuple(
                    dict.fromkeys(
                        (
                            *media.warnings,
                            *asr_payload.vad_warnings,
                            "NO_SPEECH_DETECTED",
                        )
                    )
                ),
                boundary_candidates=_boundary_candidates(
                    duration_ms,
                    silence=asr_payload.silence_boundaries_ms,
                ),
            )

        alignment = components.aligner.align(audio, asr_payload.segments)
        speaker_warnings: tuple[str, ...] = ()
        try:
            turns = components.diarizer.diarize(
                audio,
                min_speakers=media.source.asset.config.min_speakers,
                max_speakers=media.source.asset.config.max_speakers,
            )
        except VideoDemoError as error:
            if not allow_speaker_fallback or error.code not in {
                ErrorCode.PYANNOTE_AUTHENTICATION_FAILED,
                ErrorCode.PYANNOTE_MODEL_UNAVAILABLE,
                ErrorCode.SPEECH_DEPENDENCY_UNAVAILABLE,
            }:
                raise
            turns = ()
            speaker_warnings = ("DEMO_DEGRADED_SPEAKER_UNKNOWN",)
        assigned_words = assign_speakers(alignment.words, turns)
        audio_events = components.audio_events.detect(audio, duration_ms=duration_ms)
        evidence: tuple[EvidenceItem, ...] = (
            *alignment.preserved_segments,
            *assigned_words,
            *turns,
            *audio_events,
        )
        return SpeechAnalysis(
            transcript_source="ASR",
            enrichment_mode="full",
            evidence=_sort_evidence(evidence),
            warnings=tuple(
                dict.fromkeys(
                    (
                        *media.warnings,
                        *asr_payload.vad_warnings,
                        *alignment.warning_codes,
                        *speaker_warnings,
                    )
                )
            ),
            boundary_candidates=_boundary_candidates(
                duration_ms,
                silence=asr_payload.silence_boundaries_ms,
                sentence_ends=tuple(item.end_ms for item in alignment.preserved_segments),
                speaker_changes=tuple(item.start_ms for item in turns),
                language_changes=asr_payload.language_change_boundaries_ms,
            ),
        )


class VerifiedAudioSlicer:
    """复用 FFmpeg Port，并对当前 run 归属与摘要做二次校验。"""

    def __init__(self, runtime_root: Path, client: AudioSliceClient, duration_ms: int) -> None:
        self._runtime_root = runtime_root.expanduser().resolve(strict=False)
        self._client = client
        self._duration_ms = duration_ms

    def create(
        self,
        audio: Path,
        run_relative_root: Path,
        slice_id: str,
        time_range: TimeRange,
    ) -> Path:
        artifact = self._client.create_audio_slice(
            audio,
            run_relative_root,
            slice_id,
            time_range,
            source_duration_ms=self._duration_ms,
        )
        run_root = safe_runtime_path(self._runtime_root, run_relative_root)
        output = safe_runtime_path(self._runtime_root, Path(artifact.relative_path))
        if not output.is_relative_to(run_root) or output.is_symlink() or not output.is_file():
            raise VideoDemoError(
                ErrorCode.WORKSPACE_PATH_ESCAPE,
                "音频切片必须位于当前运行目录内",
            )
        if _sha256_file(output) != artifact.sha256:
            raise VideoDemoError(ErrorCode.VIDEO_DIGEST_MISMATCH, "音频切片摘要校验失败")
        return output


class AudioSliceClient(Protocol):
    def create_audio_slice(
        self,
        source: Path,
        run_relative_root: Path,
        slice_id: str,
        time_range: TimeRange,
        *,
        source_duration_ms: int,
    ) -> AudioSliceArtifact: ...


def _identify_languages(
    media: PreparedMedia,
    components: SpeechComponents,
    speech_intervals: Sequence[SpeechInterval],
) -> LanguageIdentificationResult:
    assert media.audio_path is not None
    hints = tuple(media.source.asset.config.language_hints)
    if len(hints) == 1 and hints[0] != "und":
        hint_spans = tuple(
            LanguageSpan(
                evidence_id=stable_identifier(
                    "lid_hint",
                    {"speech_evidence_id": interval.evidence_id, "language": hints[0]},
                ),
                start_ms=interval.start_ms,
                end_ms=interval.end_ms,
                language=hints[0],
                confidence=None,
                detection_source="HINT",
                is_fully_evaluated_language=hints[0] in {"zh", "en", "ja", "ko", "es"},
            )
            for interval in speech_intervals
        )
        return LanguageIdentificationResult(spans=hint_spans, change_boundaries_ms=())
    spans: list[LanguageSpan] = []
    for interval in speech_intervals:
        slice_path = components.slicer.create(
            media.audio_path,
            media.source.asset.run_relative_root,
            f"lid_{interval.evidence_id}",
            interval,
        )
        local_speech = SpeechInterval(
            evidence_id=interval.evidence_id,
            start_ms=0,
            end_ms=interval.duration_ms,
            confidence=interval.confidence,
        )
        local = components.language_identifier.identify(
            slice_path,
            (local_speech,),
            hints,
        )
        for span in local.spans:
            spans.append(
                span.model_copy(
                    update={
                        "start_ms": interval.start_ms + span.start_ms,
                        "end_ms": interval.start_ms + span.end_ms,
                    },
                ),
            )
    ordered = tuple(sorted(spans, key=lambda item: (item.start_ms, item.end_ms, item.evidence_id)))
    boundaries = tuple(
        current.start_ms
        for previous, current in pairwise(ordered)
        if previous.language != current.language
    )
    return LanguageIdentificationResult(spans=ordered, change_boundaries_ms=boundaries)


def _sort_evidence(items: Sequence[EvidenceItem]) -> tuple[EvidenceItem, ...]:
    rank = {AudioEvent: 0, SpeakerTurn: 1, SpeechSegment: 2, AlignedWord: 3}
    allowed = (SpeechSegment, AlignedWord, SpeakerTurn, AudioEvent)
    if any(not isinstance(item, allowed) for item in items):
        raise VideoDemoError(ErrorCode.SPEECH_MODEL_UNAVAILABLE, "语音阶段返回了非法证据类型")
    return tuple(
        sorted(
            items,
            key=lambda item: (
                item.start_ms,
                rank[type(item)],
                item.end_ms,
                item.evidence_id,
            ),
        ),
    )


def _boundary_candidates(
    duration_ms: int,
    *,
    silence: Sequence[int] = (),
    sentence_ends: Sequence[int] = (),
    speaker_changes: Sequence[int] = (),
    language_changes: Sequence[int] = (),
) -> tuple[SpeechBoundaryCandidate, ...]:
    values = (
        *((timestamp, "silence") for timestamp in silence),
        *((timestamp, "sentence_end") for timestamp in sentence_ends),
        *((timestamp, "speaker_change") for timestamp in speaker_changes),
        *((timestamp, "language_change") for timestamp in language_changes),
    )
    return tuple(
        SpeechBoundaryCandidate(timestamp_ms=timestamp, source=source, score=1.0)  # type: ignore[arg-type]
        for timestamp, source in sorted(
            {(timestamp, source) for timestamp, source in values if 0 < timestamp < duration_ms},
            key=lambda item: (item[0], item[1]),
        )
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
