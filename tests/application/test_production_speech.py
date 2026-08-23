from __future__ import annotations

import hashlib
import traceback
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest

from video_demo.application.pipeline import (
    PipelineRunConfig,
    PreparedMedia,
    ProbedAsset,
    RegisteredAsset,
    SpeechAnalysis,
    SpeechBoundaryCandidate,
)
from video_demo.application.production_speech import ProductionSpeechAnalyzer, SpeechComponents
from video_demo.domain.evidence import AlignedWord, AudioEvent, SpeakerTurn, SpeechSegment
from video_demo.domain.manifest import AudioStream, Rational, VideoAssetManifest, VideoStream
from video_demo.domain.run import TimeRange
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.media.probe import ProbeLimits
from video_demo.media.subtitles import ParsedSubtitle
from video_demo.media.transcode import SubtitleArtifact
from video_demo.speech.alignment import AlignmentResult
from video_demo.speech.asr import RawAsrSegment
from video_demo.speech.language import LanguageIdentificationResult, LanguageSpan
from video_demo.speech.snapshots import (
    AsrSnapshotPayload,
    SpeechAnalysisSnapshotPayload,
    SpeechFingerprintInputs,
    _speech_fingerprint_v1,
    asr_fingerprint,
    subtitle_transcript_payload_sha256,
)
from video_demo.speech.vad import SpeechInterval, VadResult
from video_demo.storage.artifacts import AtomicArtifactStore
from video_demo.storage.snapshots import SnapshotStore


def test_no_audio_skips_every_speech_component(tmp_path: Path) -> None:
    calls: list[str] = []
    analyzer = _analyzer(
        tmp_path,
        lambda _media, _cancel: calls.append("components")  # type: ignore[arg-type]
    )

    result = analyzer.analyze(
        _media(
            tmp_path,
            has_audio=False,
            config=PipelineRunConfig(speech_enrichment_mode="full"),
        )
    )

    assert calls == []
    assert result.evidence == ()
    assert result.warnings == ("NO_AUDIO_TRACK",)
    assert result.transcript_source == "NONE"
    assert result.enrichment_mode == "text"


def test_text_mode_skips_optional_speech_enrichment_components(tmp_path: Path) -> None:
    speech = SpeechInterval(evidence_id="vad_text", start_ms=0, end_ms=2_000, confidence=0.9)
    language = LanguageSpan(
        evidence_id="lid_text",
        start_ms=0,
        end_ms=2_000,
        language="zh",
        confidence=0.9,
        is_fully_evaluated_language=True,
    )
    segment = SpeechSegment(
        evidence_id="asr_text",
        start_ms=0,
        end_ms=1_000,
        text="文本模式只保留转写",
        language="zh",
        confidence=0.9,
        is_fully_evaluated_language=True,
    )
    calls: list[str] = []

    class Forbidden:
        def __getattr__(self, name: str) -> Any:
            calls.append(name)
            raise AssertionError(f"text 模式不应调用可选语音增强组件: {name}")

    components = _successful_components(tmp_path, speech, language, segment)
    components = SpeechComponents(
        **{
            **components.__dict__,
            "aligner": Forbidden(),
            "diarizer": Forbidden(),
            "audio_events": Forbidden(),
        }
    )
    media = _media(tmp_path, config=PipelineRunConfig(speech_enrichment_mode="text"))

    result = _analyzer(tmp_path, lambda _media, _cancel: components).analyze(media)

    assert result.transcript_source == "ASR"
    assert len(result.evidence) == 1
    assert isinstance(result.evidence[0], SpeechSegment)
    assert result.evidence[0].text == segment.text
    assert not calls


def test_public_asr_stage_skips_optional_enrichment_components(tmp_path: Path) -> None:
    speech = SpeechInterval(evidence_id="vad_stage", start_ms=0, end_ms=2_000, confidence=0.9)
    language = LanguageSpan(
        evidence_id="lid_stage",
        start_ms=0,
        end_ms=2_000,
        language="zh",
        confidence=0.9,
        is_fully_evaluated_language=True,
    )
    segment = SpeechSegment(
        evidence_id="asr_stage",
        start_ms=0,
        end_ms=1_000,
        text="公开 ASR 阶段",
        language="zh",
        confidence=0.9,
        is_fully_evaluated_language=True,
    )
    calls: list[str] = []

    class Forbidden:
        def __getattr__(self, name: str) -> Any:
            calls.append(name)
            raise AssertionError(f"ASR 阶段不得访问增强组件：{name}")

    components = replace(
        _successful_components(tmp_path, speech, language, segment),
        aligner=Forbidden(),
        diarizer=Forbidden(),
        audio_events=Forbidden(),
    )

    payload = ProductionSpeechAnalyzer.run_asr_stage(_media(tmp_path), components)

    assert payload.segments[0].text == segment.text
    assert not calls


def test_single_language_hint_skips_lid_and_marks_hint_source(tmp_path: Path) -> None:
    speech = SpeechInterval(evidence_id="vad_hint_mode", start_ms=0, end_ms=2_000, confidence=0.9)
    segment = SpeechSegment(
        evidence_id="asr_hint_mode",
        start_ms=0,
        end_ms=1_000,
        text="配置提示语言",
        language="zh",
        confidence=0.9,
        is_fully_evaluated_language=True,
    )
    calls: list[str] = []

    class Vad:
        def detect(self, _audio: Path, *, duration_ms: int) -> VadResult:
            return VadResult(speech=(speech,), silence=(), long_silence_boundaries_ms=())

    class ForbiddenLid:
        def identify(self, *_args: object, **_kwargs: object) -> object:
            calls.append("lid")
            raise AssertionError("单语言提示不应执行 LID")

    components = _successful_components(tmp_path, speech, LanguageSpan(
        evidence_id="unused",
        start_ms=0,
        end_ms=2_000,
        language="zh",
        confidence=0.9,
        is_fully_evaluated_language=True,
    ), segment)
    components = SpeechComponents(
        **{
            **components.__dict__,
            "vad": Vad(),
            "language_identifier": ForbiddenLid(),
        }
    )
    media = _media(
        tmp_path,
        config=PipelineRunConfig(language_hints=("zh",), speech_enrichment_mode="text"),
    )

    result = _analyzer(tmp_path, lambda _media, _cancel: components).analyze(media)

    assert len(result.evidence) == 1
    assert isinstance(result.evidence[0], SpeechSegment)
    assert result.evidence[0].text == segment.text
    assert result.enrichment_mode == "text"
    assert calls == []


def test_eligible_subtitle_skips_every_speech_component(tmp_path: Path) -> None:
    component_factory_calls = 0

    def component_factory(
        _media: PreparedMedia,
        _cancel: object,
    ) -> SpeechComponents:
        nonlocal component_factory_calls
        component_factory_calls += 1
        raise AssertionError("字幕命中时不得构造语音组件")

    media = replace(
        _media(tmp_path),
        audio_path=None,
        audio_sha256=None,
        subtitle=_parsed_subtitle(),
        warnings=("SUBTITLE_TRACK_REJECTED:3:INCOMPLETE",),
    )
    result = _analyzer(tmp_path, component_factory).analyze(media)

    assert result.transcript_source == "SUBTITLE"
    assert result.evidence == _parsed_subtitle().cues
    assert result.warnings == (
        "SUBTITLE_TRACK_REJECTED:3:INCOMPLETE",
        "TRANSCRIPT_SOURCE_SUBTITLE",
    )
    assert result.boundary_candidates == (
        SpeechBoundaryCandidate(1_000, "sentence_end", 1.0),
    )
    assert component_factory_calls == 0


def test_legacy_subtitle_snapshot_is_projected_to_text_semantics(tmp_path: Path) -> None:
    media = replace(_media(tmp_path), subtitle=_parsed_subtitle())
    inputs = _fingerprint_inputs()
    store = SnapshotStore(AtomicArtifactStore(tmp_path))
    analysis = SpeechAnalysis(
        transcript_source="SUBTITLE",
        enrichment_mode="text",
        evidence=media.subtitle.cues if media.subtitle is not None else (),
        warnings=("LEGACY",),
    )
    legacy_key = _speech_fingerprint_v1(
        processing_mode="SUBTITLE",
        transcript_payload_sha256=subtitle_transcript_payload_sha256(
            analysis, media.subtitle.artifact.sha256 if media.subtitle is not None else "a" * 64
        ),
        media_warnings=media.warnings,
        min_speakers=None,
        max_speakers=None,
        allow_speaker_fallback=False,
        inputs=inputs,
    )
    store.publish(
        media.source.asset.run_relative_root,
        "speech",
        legacy_key,
        SpeechAnalysisSnapshotPayload(
            schema_version="1.0.0",
            enrichment_mode="full",
            evidence=analysis.evidence,
            warnings=analysis.warnings,
            boundary_candidates=(),
            transcript_source="SUBTITLE",
        ),
    )

    result = ProductionSpeechAnalyzer(
        lambda _media, _cancel: (_ for _ in ()).throw(AssertionError("字幕不得构造组件")),
        snapshot_store=store,
        fingerprint_inputs=inputs,
    ).analyze(media)

    assert result.enrichment_mode == "text"
    assert result.warnings == ("LEGACY",)


def test_no_speech_still_runs_yamnet_and_skips_remaining_models(tmp_path: Path) -> None:
    calls: list[str] = []
    event = _event(0, 960)

    class Events:
        def detect(self, audio: Path, *, duration_ms: int) -> tuple[AudioEvent, ...]:
            calls.append("yamnet")
            assert duration_ms == 4_000
            return (event,)

    class Vad:
        def detect(self, audio: Path, *, duration_ms: int) -> VadResult:
            calls.append("vad")
            return VadResult(
                speech=(),
                silence=(),
                long_silence_boundaries_ms=(),
                warnings=("NO_SPEECH_DETECTED",),
            )

    components = _components(calls, vad=Vad(), events=Events())
    result = _analyzer(tmp_path, lambda _media, _cancel: components).analyze(
        replace(
            _media(tmp_path),
            warnings=("SUBTITLE_TRACK_REJECTED:2:INCOMPLETE",),
        ),
    )

    assert calls == ["vad", "yamnet"]
    assert result.evidence == (event,)
    assert result.warnings == (
        "SUBTITLE_TRACK_REJECTED:2:INCOMPLETE",
        "NO_SPEECH_DETECTED",
    )
    assert result.transcript_source == "ASR"


def test_complete_chain_preserves_order_config_absolute_time_speakers_and_sorting(
    tmp_path: Path,
) -> None:
    calls: list[object] = []
    speech = SpeechInterval(evidence_id="vad_001", start_ms=1_000, end_ms=3_000, confidence=0.9)
    language = LanguageSpan(
        evidence_id="lid_001",
        start_ms=0,
        end_ms=2_000,
        language="en",
        confidence=0.95,
        is_fully_evaluated_language=True,
    )
    word = AlignedWord(
        evidence_id="word_001",
        start_ms=1_100,
        end_ms=1_500,
        text="hello",
        language="en",
        probability=0.8,
    )
    turn = SpeakerTurn(
        evidence_id="speaker_001",
        start_ms=1_000,
        end_ms=3_000,
        speaker="SPEAKER_01",
    )
    event = _event(0, 960)

    class Vad:
        def detect(self, audio: Path, *, duration_ms: int) -> VadResult:
            calls.append("vad")
            return VadResult(
                speech=(speech,), silence=(), long_silence_boundaries_ms=(), warnings=()
            )

    class Events:
        def detect(self, audio: Path, *, duration_ms: int) -> tuple[AudioEvent, ...]:
            calls.append("yamnet")
            return (event,)

    class Lid:
        def identify(
            self, audio: Path, intervals: object, hints: tuple[str, ...]
        ) -> LanguageIdentificationResult:
            calls.append(("lid", hints, tuple(intervals)))  # type: ignore[arg-type]
            return LanguageIdentificationResult(spans=(language,), change_boundaries_ms=())

    class Slicer:
        def create(
            self, audio: Path, run_root: Path, slice_id: str, time_range: TimeRange
        ) -> Path:
            calls.append(("slice", run_root, slice_id, time_range.start_ms, time_range.end_ms))
            output = tmp_path / f"{slice_id}.wav"
            output.write_bytes(b"wav")
            return output

    class Recognizer:
        def transcribe_slice(
            self,
            audio_slice: Path,
            language_span: LanguageSpan,
            *,
            hotwords: tuple[str, ...] = (),
            core_context: str | None = None,
        ) -> tuple[RawAsrSegment, ...]:
            calls.append(
                (
                    "asr",
                    audio_slice.name,
                    language_span.language,
                    hotwords,
                    core_context,
                )
            )
            return (RawAsrSegment(100, 1_500, " hello ", 0.9),)

    class Aligner:
        def align(self, audio: Path, segments: object) -> AlignmentResult:
            built = tuple(segments)  # type: ignore[arg-type]
            calls.append(("align", audio.name, built))
            assert [(item.start_ms, item.end_ms, item.text) for item in built] == [
                (1_100, 2_500, "hello")
            ]
            return AlignmentResult(words=(word,), preserved_segments=built, warning_codes=())

    class Diarizer:
        def diarize(
            self, audio: Path, *, min_speakers: int | None, max_speakers: int | None
        ) -> tuple[SpeakerTurn, ...]:
            calls.append(("diarize", min_speakers, max_speakers))
            return (turn,)

    components = SpeechComponents(
        vad=Vad(),  # type: ignore[arg-type]
        language_identifier=Lid(),  # type: ignore[arg-type]
        recognizer=Recognizer(),  # type: ignore[arg-type]
        aligner=Aligner(),  # type: ignore[arg-type]
        diarizer=Diarizer(),  # type: ignore[arg-type]
        audio_events=Events(),  # type: ignore[arg-type]
        slicer=Slicer(),  # type: ignore[arg-type]
    )
    result = _analyzer(tmp_path, lambda _media, _cancel: components).analyze(
        replace(
            _media(
                tmp_path,
                config=PipelineRunConfig(
                    language_hints=("en", "zh"),
                    min_speakers=1,
                    max_speakers=3,
                    hotwords=("Milvus", "WhisperX"),
                    core_context="这是向量检索课程。",
                    speech_enrichment_mode="full",
                ),
            ),
            warnings=("SUBTITLE_TRACK_REJECTED:2:TIMELINE_INVALID",),
        )
    )

    assert calls[0] == "vad"
    assert calls[1] == (
        "slice",
        Path("runs/scope/run_001"),
        "lid_vad_001",
        1_000,
        3_000,
    )
    local_speech = SpeechInterval(
        evidence_id="vad_001", start_ms=0, end_ms=2_000, confidence=0.9
    )
    assert calls[2] == ("lid", ("en", "zh"), (local_speech,))
    assert calls[3] == ("slice", Path("runs/scope/run_001"), "asr_lid_001", 1_000, 3_000)
    assert calls[4] == (
        "asr",
        "asr_lid_001.wav",
        "en",
        ("Milvus", "WhisperX"),
        "这是向量检索课程。",
    )
    assert calls[5][0:2] == ("align", "audio.wav")  # type: ignore[index]
    assert calls[6] == ("diarize", 1, 3)
    assert calls[7] == "yamnet"
    assert tuple(type(item) for item in result.evidence) == (
        AudioEvent,
        SpeakerTurn,
        SpeechSegment,
        AlignedWord,
    )
    assigned_word = result.evidence[-1]
    assert isinstance(assigned_word, AlignedWord)
    assert assigned_word.speaker == "SPEAKER_01"
    assert result.transcript_source == "ASR"
    assert result.warnings == ("SUBTITLE_TRACK_REJECTED:2:TIMELINE_INVALID",)


def test_speech_analysis_exports_deduplicated_hybrid_boundary_candidates(
    tmp_path: Path,
) -> None:
    speech = SpeechInterval(evidence_id="vad_001", start_ms=0, end_ms=4_000, confidence=0.9)
    first_language = LanguageSpan(
        evidence_id="lid_001",
        start_ms=0,
        end_ms=2_000,
        language="en",
        confidence=0.9,
        is_fully_evaluated_language=True,
    )
    second_language = first_language.model_copy(
        update={"evidence_id": "lid_002", "start_ms": 2_000, "end_ms": 4_000, "language": "zh"}
    )
    segment = SpeechSegment(
        evidence_id="asr_001",
        start_ms=0,
        end_ms=2_000,
        text="hello.",
        language="en",
        confidence=0.9,
        is_fully_evaluated_language=True,
    )
    turn = SpeakerTurn(
        evidence_id="speaker_001",
        start_ms=2_000,
        end_ms=4_000,
        speaker="SPEAKER_01",
    )
    components = _successful_components(tmp_path, speech, first_language, segment)

    class Vad:
        def detect(self, _audio: Path, *, duration_ms: int) -> VadResult:
            return VadResult(
                speech=(speech,),
                silence=(),
                long_silence_boundaries_ms=(2_000, 2_000, duration_ms),
            )

    class Lid:
        def identify(self, _audio: Path, _speech: object, _hints: tuple[str, ...]) -> object:
            return LanguageIdentificationResult(
                spans=(first_language, second_language),
                change_boundaries_ms=(2_000, 2_000),
            )

    class Diarizer:
        def diarize(self, _audio: Path, **_kwargs: object) -> object:
            return (turn,)

    components = SpeechComponents(
        **{
            **components.__dict__,
            "vad": Vad(),
            "language_identifier": Lid(),
            "diarizer": Diarizer(),
        }
    )

    result = _analyzer(tmp_path, lambda _media, _cancel: components).analyze(
        _media(tmp_path)
    )

    assert result.boundary_candidates == (
        SpeechBoundaryCandidate(2_000, "language_change", 1.0),
        SpeechBoundaryCandidate(2_000, "sentence_end", 1.0),
        SpeechBoundaryCandidate(2_000, "silence", 1.0),
        SpeechBoundaryCandidate(2_000, "speaker_change", 1.0),
    )


def test_every_asr_slice_receives_same_hints_without_passing_them_to_lid(
    tmp_path: Path,
) -> None:
    speech = SpeechInterval(
        evidence_id="vad_hints",
        start_ms=0,
        end_ms=4_000,
        confidence=0.9,
    )
    languages = (
        LanguageSpan(
            evidence_id="lid_hint_zh",
            start_ms=0,
            end_ms=2_000,
            language="zh",
            confidence=0.9,
            is_fully_evaluated_language=True,
        ),
        LanguageSpan(
            evidence_id="lid_hint_en",
            start_ms=2_000,
            end_ms=4_000,
            language="en",
            confidence=0.9,
            is_fully_evaluated_language=True,
        ),
    )
    lid_calls: list[tuple[str, ...]] = []
    asr_calls: list[tuple[tuple[str, ...], str | None]] = []

    class Vad:
        def detect(self, _audio: Path, *, duration_ms: int) -> VadResult:
            return VadResult(speech=(speech,), silence=(), long_silence_boundaries_ms=())

    class Lid:
        def identify(
            self,
            _audio: Path,
            _speech: object,
            hints: tuple[str, ...],
        ) -> LanguageIdentificationResult:
            lid_calls.append(hints)
            return LanguageIdentificationResult(spans=languages, change_boundaries_ms=(2_000,))

    class Recognizer:
        def transcribe_slice(
            self,
            _audio: Path,
            _language: LanguageSpan,
            *,
            hotwords: tuple[str, ...] = (),
            core_context: str | None = None,
        ) -> tuple[RawAsrSegment, ...]:
            asr_calls.append((hotwords, core_context))
            return (RawAsrSegment(0, 1_000, "模型输出", 0.9),)

    class Slicer:
        def create(
            self,
            _audio: Path,
            _root: Path,
            slice_id: str,
            _range: TimeRange,
        ) -> Path:
            output = tmp_path / f"{slice_id}.wav"
            output.write_bytes(b"wav")
            return output

    class Empty:
        def align(self, _audio: Path, segments: object) -> AlignmentResult:
            return AlignmentResult(
                words=(),
                preserved_segments=tuple(segments),  # type: ignore[arg-type]
                warning_codes=(),
            )

        def diarize(self, _audio: Path, **_kwargs: object) -> tuple[SpeakerTurn, ...]:
            return ()

        def detect(self, _audio: Path, *, duration_ms: int) -> tuple[AudioEvent, ...]:
            return ()

    components = SpeechComponents(
        vad=Vad(),  # type: ignore[arg-type]
        language_identifier=Lid(),  # type: ignore[arg-type]
        recognizer=Recognizer(),  # type: ignore[arg-type]
        aligner=Empty(),  # type: ignore[arg-type]
        diarizer=Empty(),  # type: ignore[arg-type]
        audio_events=Empty(),  # type: ignore[arg-type]
        slicer=Slicer(),  # type: ignore[arg-type]
    )
    config = PipelineRunConfig(
        language_hints=("zh", "en"),
        hotwords=("Milvus", "WhisperX"),
        core_context="这是向量检索课程。",
    )

    result = _analyzer(tmp_path, lambda _media, _cancel: components).analyze(
        _media(tmp_path, config=config)
    )

    assert result.transcript_source == "ASR"
    assert lid_calls == [("zh", "en")]
    assert asr_calls == [
        (("Milvus", "WhisperX"), "这是向量检索课程。"),
        (("Milvus", "WhisperX"), "这是向量检索课程。"),
    ]


def test_alignment_unavailable_preserves_segment_without_fabricating_word(tmp_path: Path) -> None:
    speech = SpeechInterval(evidence_id="vad_001", start_ms=0, end_ms=2_000, confidence=0.9)
    language = LanguageSpan(
        evidence_id="lid_ja",
        start_ms=0,
        end_ms=2_000,
        language="ja",
        confidence=0.9,
        is_fully_evaluated_language=True,
    )
    segment = SpeechSegment(
        evidence_id="asr_ja",
        start_ms=0,
        end_ms=1_000,
        text="こんにちは",
        language="ja",
        confidence=0.9,
        is_fully_evaluated_language=True,
    )
    components = _successful_components(tmp_path, speech, language, segment)
    components = SpeechComponents(
        **{
            **components.__dict__,
            "aligner": _StaticAligner(
                AlignmentResult(
                    words=(),
                    preserved_segments=(segment,),
                    warning_codes=("ALIGNMENT_MODEL_UNAVAILABLE:ja",),
                )
            ),
        }
    )

    result = _analyzer(tmp_path, lambda _media, _cancel: components).analyze(
        _media(tmp_path)
    )

    assert segment in result.evidence
    assert not any(isinstance(item, AlignedWord) for item in result.evidence)
    assert result.warnings == ("ALIGNMENT_MODEL_UNAVAILABLE:ja",)


def test_component_failure_is_stable_and_never_leaks_secret(tmp_path: Path) -> None:
    secret = "hf_super_secret_for_test"

    def factory(_media: PreparedMedia, _cancel: object) -> SpeechComponents:
        raise RuntimeError(secret)

    with pytest.raises(VideoDemoError) as raised:
        _analyzer(tmp_path, factory).analyze(_media(tmp_path))

    rendered = f"{raised.value.message} {raised.value.details}"
    assert raised.value.code == ErrorCode.SPEECH_MODEL_UNAVAILABLE
    assert secret not in rendered
    assert raised.value.__cause__ is None
    assert secret not in "".join(traceback.format_exception(raised.value))


def test_demo_mode_keeps_asr_when_pyannote_is_unavailable_and_marks_unknown_speaker(
    tmp_path: Path,
) -> None:
    speech = SpeechInterval(evidence_id="vad_demo", start_ms=0, end_ms=2_000, confidence=0.9)
    language = LanguageSpan(
        evidence_id="lid_demo",
        start_ms=0,
        end_ms=2_000,
        language="zh",
        confidence=0.9,
        is_fully_evaluated_language=True,
    )
    segment = SpeechSegment(
        evidence_id="asr_demo",
        start_ms=0,
        end_ms=1_000,
        text="你好",
        language="zh",
        confidence=0.9,
        is_fully_evaluated_language=True,
    )

    class Diarizer:
        def diarize(self, _audio: Path, **_kwargs: object) -> tuple[SpeakerTurn, ...]:
            raise VideoDemoError(
                ErrorCode.PYANNOTE_AUTHENTICATION_FAILED,
                "pyannote 鉴权失败",
            )

    components = _successful_components(tmp_path, speech, language, segment)
    word = AlignedWord(
        evidence_id="word_demo",
        start_ms=0,
        end_ms=500,
        text="你好",
        language="zh",
        probability=0.9,
    )
    components = SpeechComponents(
        **{
            **components.__dict__,
            "diarizer": Diarizer(),
            "aligner": _StaticAligner(
                AlignmentResult(
                    words=(word,),
                    preserved_segments=(segment,),
                    warning_codes=(),
                )
            ),
        }
    )

    result = _analyzer(
        tmp_path,
        lambda _media, _cancel: components,
        allow_speaker_fallback=True,
    ).analyze(_media(tmp_path))

    word = next(item for item in result.evidence if isinstance(item, AlignedWord))
    assert word.speaker == "SPEAKER_UNKNOWN"
    assert "DEMO_DEGRADED_SPEAKER_UNKNOWN" in result.warnings


def test_alignment_failure_retry_reuses_asr_snapshot(tmp_path: Path) -> None:
    speech = SpeechInterval(evidence_id="vad_retry", start_ms=0, end_ms=2_000, confidence=0.9)
    language = LanguageSpan(
        evidence_id="lid_retry",
        start_ms=0,
        end_ms=2_000,
        language="zh",
        confidence=0.9,
        is_fully_evaluated_language=True,
    )
    segment = SpeechSegment(
        evidence_id="asr_retry",
        start_ms=0,
        end_ms=1_000,
        text="重试文本",
        language="zh",
        confidence=0.9,
        is_fully_evaluated_language=True,
    )
    calls = {"vad": 0, "lid": 0, "recognizer": 0, "aligner": 0}

    class Vad:
        def detect(self, _audio: Path, *, duration_ms: int) -> VadResult:
            calls["vad"] += 1
            return VadResult(speech=(speech,), silence=(), long_silence_boundaries_ms=())

    class Lid:
        def identify(self, _audio: Path, _speech: object, _hints: tuple[str, ...]) -> object:
            calls["lid"] += 1
            return LanguageIdentificationResult(spans=(language,), change_boundaries_ms=())

    class Recognizer:
        def transcribe_slice(
            self,
            _audio: Path,
            _language: LanguageSpan,
            **_kwargs: object,
        ) -> object:
            calls["recognizer"] += 1
            return (RawAsrSegment(0, 1_000, segment.text, segment.confidence),)

    class Aligner:
        def align(self, _audio: Path, segments: object) -> AlignmentResult:
            calls["aligner"] += 1
            if calls["aligner"] == 1:
                raise VideoDemoError(ErrorCode.SPEECH_MODEL_UNAVAILABLE, "模拟首次失败")
            return AlignmentResult(
                words=(),
                preserved_segments=tuple(segments),  # type: ignore[arg-type]
                warning_codes=(),
            )

    class Slicer:
        def create(self, _audio: Path, _root: Path, slice_id: str, _range: TimeRange) -> Path:
            path = tmp_path / f"{slice_id}.wav"
            path.write_bytes(b"wav")
            return path

    class Empty:
        def diarize(self, _audio: Path, **_kwargs: object) -> object:
            return ()

        def detect(self, _audio: Path, *, duration_ms: int) -> object:
            return ()

    components = SpeechComponents(
        vad=Vad(),  # type: ignore[arg-type]
        language_identifier=Lid(),  # type: ignore[arg-type]
        recognizer=Recognizer(),  # type: ignore[arg-type]
        aligner=Aligner(),  # type: ignore[arg-type]
        diarizer=Empty(),  # type: ignore[arg-type]
        audio_events=Empty(),  # type: ignore[arg-type]
        slicer=Slicer(),  # type: ignore[arg-type]
    )
    analyzer = ProductionSpeechAnalyzer(
        lambda _media, _cancel: components,
        snapshot_store=SnapshotStore(AtomicArtifactStore(tmp_path)),
        fingerprint_inputs=_fingerprint_inputs(),
    )
    media = _media(tmp_path)

    with pytest.raises(VideoDemoError) as raised:
        analyzer.analyze(media)
    result = analyzer.analyze(media)

    assert raised.value.code == ErrorCode.SPEECH_MODEL_UNAVAILABLE
    assert result.transcript_source == "ASR"
    assert calls == {"vad": 1, "lid": 1, "recognizer": 1, "aligner": 2}


def test_complete_speech_snapshot_hit_skips_component_factory(tmp_path: Path) -> None:
    speech = SpeechInterval(evidence_id="vad_cached", start_ms=0, end_ms=2_000, confidence=0.9)
    language = LanguageSpan(
        evidence_id="lid_cached",
        start_ms=0,
        end_ms=2_000,
        language="zh",
        confidence=0.9,
        is_fully_evaluated_language=True,
    )
    segment = SpeechSegment(
        evidence_id="asr_cached",
        start_ms=0,
        end_ms=1_000,
        text="缓存文本",
        language="zh",
        confidence=0.9,
        is_fully_evaluated_language=True,
    )
    components = _successful_components(tmp_path, speech, language, segment)
    factory_calls = 0

    def factory(_media: PreparedMedia, _cancel: object) -> SpeechComponents:
        nonlocal factory_calls
        factory_calls += 1
        return components

    analyzer = _analyzer(tmp_path, factory)
    media = _media(tmp_path)

    first = analyzer.analyze(media)
    second = analyzer.analyze(media)

    assert second == first
    assert factory_calls == 1


def test_production_speech_reads_legacy_full_snapshot_when_current_pointer_is_legacy(
    tmp_path: Path,
) -> None:
    language = LanguageSpan(
        evidence_id="legacy_lid",
        start_ms=0,
        end_ms=2_000,
        language="zh",
        confidence=0.9,
        is_fully_evaluated_language=True,
    )
    segment = SpeechSegment(
        evidence_id="legacy_asr",
        start_ms=0,
        end_ms=1_000,
        text="历史快照",
        language="zh",
        confidence=0.9,
        is_fully_evaluated_language=True,
    )
    media = _media(tmp_path, config=PipelineRunConfig(speech_enrichment_mode="full"))
    inputs = _fingerprint_inputs()
    store = SnapshotStore(AtomicArtifactStore(tmp_path))
    asr_key = asr_fingerprint(
        audio_sha256=media.audio_sha256 or "",
        duration_ms=media.source.duration_ms,
        language_hints=(),
        hotwords=(),
        core_context=None,
        inputs=inputs,
    )
    asr_receipt = store.publish(
        media.source.asset.run_relative_root,
        "asr",
        asr_key,
        AsrSnapshotPayload(
            language_spans=(language,),
            segments=(segment,),
            vad_warnings=(),
            silence_boundaries_ms=(),
            language_change_boundaries_ms=(),
        ),
    )
    legacy_key = _speech_fingerprint_v1(
        processing_mode="ASR",
        transcript_payload_sha256=asr_receipt.sha256,
        media_warnings=(),
        min_speakers=None,
        max_speakers=None,
        allow_speaker_fallback=False,
        inputs=inputs,
    )
    store.publish(
        media.source.asset.run_relative_root,
        "speech",
        legacy_key,
        SpeechAnalysisSnapshotPayload(
            schema_version="1.0.0",
            evidence=(segment,),
            warnings=("LEGACY",),
            boundary_candidates=(),
            transcript_source="ASR",
        ),
    )

    result = ProductionSpeechAnalyzer(
        lambda _media, _cancel: (_ for _ in ()).throw(
            AssertionError("命中历史快照不应构造语音组件")
        ),
        snapshot_store=store,
        fingerprint_inputs=inputs,
    ).analyze(media)

    assert result.warnings == ("LEGACY",)
    assert result.evidence == (segment,)
    assert result.enrichment_mode == "full"


def test_text_mode_does_not_reuse_full_speech_snapshot(tmp_path: Path) -> None:
    """text 结果必须只投影 ASR 快照，不能把历史 full 增强证据带入。"""
    speech = SpeechInterval(evidence_id="vad_mode", start_ms=0, end_ms=2_000, confidence=0.9)
    language = LanguageSpan(
        evidence_id="lid_mode",
        start_ms=0,
        end_ms=2_000,
        language="zh",
        confidence=0.9,
        is_fully_evaluated_language=True,
    )
    segment = SpeechSegment(
        evidence_id="asr_mode",
        start_ms=0,
        end_ms=1_000,
        text="只保留原始转写",
        language="zh",
        confidence=0.9,
        is_fully_evaluated_language=True,
    )
    store = SnapshotStore(AtomicArtifactStore(tmp_path))
    components = _successful_components(tmp_path, speech, language, segment)
    media_full = _media(
        tmp_path,
        config=PipelineRunConfig(speech_enrichment_mode="full"),
    )
    ProductionSpeechAnalyzer(
        lambda _media, _cancel: components,
        snapshot_store=store,
        fingerprint_inputs=_fingerprint_inputs(),
    ).analyze(media_full)
    media_text = replace(
        media_full,
        source=replace(
            media_full.source,
            asset=replace(
                media_full.source.asset,
                config=PipelineRunConfig(speech_enrichment_mode="text"),
            ),
        ),
    )
    result = ProductionSpeechAnalyzer(
        lambda _media, _cancel: (_ for _ in ()).throw(
            AssertionError("已有 ASR 快照时 text 不应构造组件")
        ),
        snapshot_store=store,
        fingerprint_inputs=_fingerprint_inputs(),
    ).analyze(media_text)

    assert result.enrichment_mode == "text"
    assert tuple(type(item) for item in result.evidence) == (SpeechSegment,)


@dataclass
class _StaticAligner:
    result: AlignmentResult

    def align(self, _audio: Path, _segments: object) -> AlignmentResult:
        return self.result


def _components(calls: list[str], *, vad: object, events: object) -> SpeechComponents:
    class Forbidden:
        def __getattr__(self, _name: str) -> Any:
            calls.append("forbidden")
            raise AssertionError("无语音时不应调用")

    return SpeechComponents(
        vad=vad,  # type: ignore[arg-type]
        language_identifier=Forbidden(),  # type: ignore[arg-type]
        recognizer=Forbidden(),  # type: ignore[arg-type]
        aligner=Forbidden(),  # type: ignore[arg-type]
        diarizer=Forbidden(),  # type: ignore[arg-type]
        audio_events=events,  # type: ignore[arg-type]
        slicer=Forbidden(),  # type: ignore[arg-type]
    )


def _successful_components(
    tmp_path: Path,
    speech: SpeechInterval,
    language: LanguageSpan,
    segment: SpeechSegment,
) -> SpeechComponents:
    class Vad:
        def detect(self, _audio: Path, *, duration_ms: int) -> VadResult:
            return VadResult(speech=(speech,), silence=(), long_silence_boundaries_ms=())

    class Lid:
        def identify(self, _audio: Path, _speech: object, _hints: tuple[str, ...]) -> object:
            return LanguageIdentificationResult(spans=(language,), change_boundaries_ms=())

    class Slicer:
        def create(self, _audio: Path, _root: Path, slice_id: str, _range: TimeRange) -> Path:
            path = tmp_path / f"{slice_id}.wav"
            path.write_bytes(b"wav")
            return path

    class Recognizer:
        def transcribe_slice(
            self,
            _audio: Path,
            _language: LanguageSpan,
            **_kwargs: object,
        ) -> object:
            return (RawAsrSegment(0, 1_000, segment.text, segment.confidence),)

    class Events:
        def detect(self, _audio: Path, *, duration_ms: int) -> object:
            return ()

    class Diarizer:
        def diarize(self, _audio: Path, **_kwargs: object) -> object:
            return ()

    return SpeechComponents(
        vad=Vad(),  # type: ignore[arg-type]
        language_identifier=Lid(),  # type: ignore[arg-type]
        recognizer=Recognizer(),  # type: ignore[arg-type]
        aligner=_StaticAligner(
            AlignmentResult(words=(), preserved_segments=(segment,), warning_codes=())
        ),
        diarizer=Diarizer(),  # type: ignore[arg-type]
        audio_events=Events(),  # type: ignore[arg-type]
        slicer=Slicer(),  # type: ignore[arg-type]
    )


def _event(start_ms: int, end_ms: int) -> AudioEvent:
    return AudioEvent(
        evidence_id="audio_001",
        start_ms=start_ms,
        end_ms=end_ms,
        audioset_class="Music",
        normalized_event="音乐",
        confidence=0.8,
        threshold_version="test-v1",
    )


def _fingerprint_inputs() -> SpeechFingerprintInputs:
    from video_demo.domain.run import ModelIdentity

    return SpeechFingerprintInputs(
        model_identities=(
            ModelIdentity(component="silero_vad", provider="local", model_id="silero"),
            ModelIdentity(component="faster_whisper", provider="local", model_id="large-v3"),
            ModelIdentity(component="whisperx", provider="local", model_id="align"),
            ModelIdentity(component="pyannote", provider="local", model_id="diarize"),
            ModelIdentity(component="yamnet", provider="local", model_id="yamnet"),
        ),
        asr_compute_type="int8",
        yamnet_class_map_sha256="d" * 64,
        yamnet_thresholds_sha256="e" * 64,
    )


def _analyzer(
    tmp_path: Path,
    factory: object,
    *,
    allow_speaker_fallback: bool = False,
) -> ProductionSpeechAnalyzer:
    return ProductionSpeechAnalyzer(
        factory,  # type: ignore[arg-type]
        snapshot_store=SnapshotStore(AtomicArtifactStore(tmp_path)),
        fingerprint_inputs=_fingerprint_inputs(),
        allow_speaker_fallback=allow_speaker_fallback,
    )


def _parsed_subtitle() -> ParsedSubtitle:
    from video_demo.domain.evidence import SubtitleCue

    cue = SubtitleCue(
        evidence_id="subtitle_001",
        start_ms=0,
        end_ms=1_000,
        text="字幕正文",
        language="zh",
        stream_index=2,
    )
    artifact = SubtitleArtifact(
        relative_path="runs/scope/run_001/media/subtitles/2.vtt",
        sha256="c" * 64,
        size_bytes=32,
        stream_index=2,
        language="zh",
        codec_name="mov_text",
    )
    return ParsedSubtitle(
        artifact=artifact,
        cues=(cue,),
        normalized_char_count=4,
        timeline_span_ratio=1.0,
    )


def _media(
    tmp_path: Path,
    *,
    has_audio: bool = True,
    config: PipelineRunConfig | None = None,
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
        config=config or PipelineRunConfig(speech_enrichment_mode="full"),
    )
    manifest = VideoAssetManifest(
        object_ref="obj_001",
        source_sha256=registered.source_sha256,
        source_size_bytes=6,
        source_mime="video/mp4",
        duration_ms=4_000,
        video_stream=VideoStream(
            index=0,
            codec_name="h264",
            width=640,
            height=360,
            average_frame_rate=Rational(numerator=25, denominator=1),
        ),
        audio_streams=(
            AudioStream(
                index=1,
                codec_name="pcm_s16le",
                sample_rate_hz=16_000,
                channels=1,
            ),
        )
        if has_audio
        else (),
        format_name="mov,mp4",
        ffprobe_version="test",
    )
    audio = tmp_path / run_root / "media/audio.wav"
    proxy = tmp_path / run_root / "media/proxy.mp4"
    proxy.parent.mkdir(parents=True, exist_ok=True)
    proxy.write_bytes(b"proxy")
    if has_audio:
        audio.write_bytes(b"wav")
    return PreparedMedia(
        source=ProbedAsset(registered, manifest, ProbeLimits()),
        proxy_path=proxy,
        proxy_sha256=hashlib.sha256(b"proxy").hexdigest(),
        proxy_size_bytes=5,
        audio_path=audio if has_audio else None,
        audio_sha256=hashlib.sha256(b"wav").hexdigest() if has_audio else None,
        warnings=() if has_audio else ("NO_AUDIO_TRACK",),
    )
