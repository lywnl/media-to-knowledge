from __future__ import annotations

import hashlib
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from video_demo.application.pipeline import (
    PipelineRunConfig,
    PreparedMedia,
    ProbedAsset,
    RegisteredAsset,
    SpeechBoundaryCandidate,
)
from video_demo.application.production_speech import ProductionSpeechAnalyzer, SpeechComponents
from video_demo.domain.evidence import AlignedWord, AudioEvent, SpeakerTurn, SpeechSegment
from video_demo.domain.manifest import AudioStream, Rational, VideoAssetManifest, VideoStream
from video_demo.domain.run import TimeRange
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.media.probe import ProbeLimits
from video_demo.speech.alignment import AlignmentResult
from video_demo.speech.asr import RawAsrSegment
from video_demo.speech.language import LanguageIdentificationResult, LanguageSpan
from video_demo.speech.vad import SpeechInterval, VadResult


def test_no_audio_skips_every_speech_component(tmp_path: Path) -> None:
    calls: list[str] = []
    analyzer = ProductionSpeechAnalyzer(
        lambda _media, _cancel: calls.append("components")  # type: ignore[arg-type]
    )

    result = analyzer.analyze(_media(tmp_path, has_audio=False))

    assert calls == []
    assert result.evidence == ()
    assert result.warnings == ("NO_AUDIO_TRACK",)


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
    result = ProductionSpeechAnalyzer(lambda _media, _cancel: components).analyze(_media(tmp_path))

    assert calls == ["vad", "yamnet"]
    assert result.evidence == (event,)
    assert result.warnings == ("NO_SPEECH_DETECTED",)


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
            self, audio_slice: Path, language_span: LanguageSpan
        ) -> tuple[RawAsrSegment, ...]:
            calls.append(("asr", audio_slice.name, language_span.language))
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
    result = ProductionSpeechAnalyzer(lambda _media, _cancel: components).analyze(
        _media(
            tmp_path,
            config=PipelineRunConfig(language_hints=("en", "zh"), min_speakers=1, max_speakers=3),
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
    assert calls[4] == ("asr", "asr_lid_001.wav", "en")
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

    result = ProductionSpeechAnalyzer(lambda _media, _cancel: components).analyze(
        _media(tmp_path)
    )

    assert result.boundary_candidates == (
        SpeechBoundaryCandidate(2_000, "language_change", 1.0),
        SpeechBoundaryCandidate(2_000, "sentence_end", 1.0),
        SpeechBoundaryCandidate(2_000, "silence", 1.0),
        SpeechBoundaryCandidate(2_000, "speaker_change", 1.0),
    )


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

    result = ProductionSpeechAnalyzer(lambda _media, _cancel: components).analyze(_media(tmp_path))

    assert segment in result.evidence
    assert not any(isinstance(item, AlignedWord) for item in result.evidence)
    assert result.warnings == ("ALIGNMENT_MODEL_UNAVAILABLE:ja",)


def test_component_failure_is_stable_and_never_leaks_secret(tmp_path: Path) -> None:
    secret = "hf_super_secret_for_test"

    def factory(_media: PreparedMedia, _cancel: object) -> SpeechComponents:
        raise RuntimeError(secret)

    with pytest.raises(VideoDemoError) as raised:
        ProductionSpeechAnalyzer(factory).analyze(_media(tmp_path))

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

    result = ProductionSpeechAnalyzer(
        lambda _media, _cancel: components,
        allow_speaker_fallback=True,
    ).analyze(_media(tmp_path))

    word = next(item for item in result.evidence if isinstance(item, AlignedWord))
    assert word.speaker == "SPEAKER_UNKNOWN"
    assert "DEMO_DEGRADED_SPEAKER_UNKNOWN" in result.warnings


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
        def transcribe_slice(self, _audio: Path, _language: LanguageSpan) -> object:
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
        config=config or PipelineRunConfig(),
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
