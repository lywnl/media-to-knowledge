from __future__ import annotations

import traceback
from pathlib import Path

import pytest

from video_demo.domain.evidence import SpeechSegment
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.speech.alignment import (
    AlignmentUnavailableError,
    NativeWhisperXBackend,
    WhisperXAligner,
)


def _segment(language: str = "ja") -> SpeechSegment:
    return SpeechSegment(
        evidence_id="asr_001",
        start_ms=1_000,
        end_ms=3_000,
        text="こんにちは",
        language=language,
        confidence=0.9,
        is_fully_evaluated_language=True,
    )


def test_alignment_unavailable_keeps_segments_without_fabricating_words(
    tmp_path: Path,
) -> None:
    class Backend:
        def load_model(self, language: str, device: str, model_dir: Path) -> object:
            raise LookupError(language)

        def align(
            self,
            segments: list[dict[str, object]],
            model: object,
            metadata: object,
            audio: Path,
            device: str,
        ) -> list[dict[str, object]]:
            raise AssertionError("不可用时不应调用 align")

    result = WhisperXAligner(Backend(), tmp_path / "models").align(
        tmp_path / "audio.wav",
        (_segment(),),
    )

    assert result.words == ()
    assert result.preserved_segments == (_segment(),)
    assert result.warning_codes == ("ALIGNMENT_MODEL_UNAVAILABLE:ja",)


def test_alignment_maps_word_times_to_milliseconds_and_preserves_language(
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, object]] = []

    class Backend:
        def load_model(self, language: str, device: str, model_dir: Path) -> object:
            calls.append(("load", (language, device, model_dir)))
            return ("model", "metadata")

        def align(
            self,
            segments: list[dict[str, object]],
            model: object,
            metadata: object,
            audio: Path,
            device: str,
        ) -> list[dict[str, object]]:
            calls.append(("align", (segments, model, metadata, audio, device)))
            return [
                {"word": "こん", "start": 1.0, "end": 1.5, "score": 0.8},
                {"word": "にちは", "start": 1.5, "end": 3.0, "score": 0.9},
            ]

    result = WhisperXAligner(Backend(), tmp_path / "models").align(
        tmp_path / "audio.wav",
        (_segment(),),
    )

    assert [(word.start_ms, word.end_ms, word.text) for word in result.words] == [
        (1_000, 1_500, "こん"),
        (1_500, 3_000, "にちは"),
    ]
    assert all(word.language == "ja" for word in result.words)
    assert result.warning_codes == ()
    assert calls[0] == ("load", ("ja", "cpu", tmp_path / "models" / "whisperx" / "ja"))


def test_alignment_skips_unaligned_word_without_fabricating_time(tmp_path: Path) -> None:
    class Backend:
        def load_model(self, language: str, device: str, model_dir: Path) -> object:
            return (object(), object())

        def align(
            self,
            segments: list[dict[str, object]],
            model: object,
            metadata: object,
            audio: Path,
            device: str,
        ) -> list[dict[str, object]]:
            return [
                {"word": "こん", "start": 1.0, "end": 1.5, "score": 0.8},
                {"word": "未对齐"},
            ]

    result = WhisperXAligner(Backend(), tmp_path / "models").align(
        tmp_path / "audio.wav",
        (_segment(),),
    )

    assert [(word.start_ms, word.end_ms, word.text) for word in result.words] == [
        (1_000, 1_500, "こん"),
    ]
    assert result.preserved_segments == (_segment(),)
    assert result.warning_codes == ("ALIGNMENT_WORD_UNALIGNED:ja",)


def test_alignment_probes_each_validation_language_independently(tmp_path: Path) -> None:
    available = {"zh", "en", "ja", "es"}

    class Backend:
        def load_model(self, language: str, device: str, model_dir: Path) -> object:
            if language not in available:
                raise LookupError(language)
            return (object(), object())

        def align(
            self,
            segments: list[dict[str, object]],
            model: object,
            metadata: object,
            audio: Path,
            device: str,
        ) -> list[dict[str, object]]:
            return []

    report = WhisperXAligner(Backend(), tmp_path / "models").probe_languages(
        ("zh", "en", "ja", "ko", "es"),
    )

    assert report == {"zh": True, "en": True, "ja": True, "ko": False, "es": True}


def test_alignment_skips_word_time_outside_asr_segment(tmp_path: Path) -> None:
    class Backend:
        def load_model(self, language: str, device: str, model_dir: Path) -> object:
            return (object(), object())

        def align(
            self,
            segments: list[dict[str, object]],
            model: object,
            metadata: object,
            audio: Path,
            device: str,
        ) -> list[dict[str, object]]:
            return [{"word": "越界", "start": 0.0, "end": 4.0, "score": 0.9}]

    result = WhisperXAligner(Backend(), tmp_path / "models").align(
        tmp_path / "audio.wav",
        (_segment(),),
    )

    assert result.words == ()
    assert result.preserved_segments == (_segment(),)
    assert result.warning_codes == ("ALIGNMENT_WORD_UNALIGNED:ja",)


def test_alignment_rejects_incomplete_word_fields_with_stable_error(tmp_path: Path) -> None:
    class Backend:
        def load_model(self, language: str, device: str, model_dir: Path) -> object:
            return (object(), object())

        def align(
            self,
            segments: list[dict[str, object]],
            model: object,
            metadata: object,
            audio: Path,
            device: str,
        ) -> list[dict[str, object]]:
            return [{"word": "字段不完整", "start": 1.0, "score": 0.9}]

    with pytest.raises(VideoDemoError) as raised:
        WhisperXAligner(Backend(), tmp_path / "models").align(
            tmp_path / "audio.wav",
            (_segment(),),
        )

    assert raised.value.code == ErrorCode.SPEECH_MODEL_UNAVAILABLE
    assert raised.value.__cause__ is None


@pytest.mark.parametrize(
    ("start", "end", "score"),
    [
        (float("nan"), 2.0, 0.9),
        (1e308, 2.0, 0.9),
        (2.0, 1.0, 0.9),
        (1.0, 2.0, 1.1),
    ],
)
def test_alignment_rejects_invalid_word_range_or_score_with_stable_error(
    tmp_path: Path,
    start: float,
    end: float,
    score: float,
) -> None:
    class Backend:
        def load_model(self, language: str, device: str, model_dir: Path) -> object:
            return (object(), object())

        def align(
            self,
            segments: list[dict[str, object]],
            model: object,
            metadata: object,
            audio: Path,
            device: str,
        ) -> list[dict[str, object]]:
            return [{"word": "非法词", "start": start, "end": end, "score": score}]

    with pytest.raises(VideoDemoError) as raised:
        WhisperXAligner(Backend(), tmp_path / "models").align(
            tmp_path / "audio.wav",
            (_segment(),),
        )

    assert raised.value.code == ErrorCode.SPEECH_MODEL_UNAVAILABLE
    assert raised.value.__cause__ is None


def test_native_whisperx_is_lazy_and_adapts_word_segments(tmp_path: Path) -> None:
    calls: list[object] = []

    def importer(name: str) -> object:
        calls.append(("import", name))

        class Module:
            @staticmethod
            def load_align_model(**kwargs: object) -> tuple[object, object]:
                calls.append(("load", kwargs))
                return ("model", "metadata")

            @staticmethod
            def align(*args: object, **kwargs: object) -> dict[str, object]:
                calls.append(("align", args, kwargs))
                return {"word_segments": [{"word": "hello", "start": 0.0, "end": 1.0}]}

        return Module()

    backend = NativeWhisperXBackend(importer=importer)
    assert calls == []

    model, metadata = backend.load_model("en", "cpu", tmp_path / "models/whisperx/en")
    words = backend.align([], model, metadata, tmp_path / "audio.wav", "cpu")

    assert words == [{"word": "hello", "start": 0.0, "end": 1.0}]
    assert calls[0] == ("import", "whisperx")
    load_call = calls[1]
    assert isinstance(load_call, tuple)
    assert load_call[1] == {
        "language_code": "en",
        "device": "cpu",
        "model_dir": str(tmp_path / "models/whisperx/en"),
    }


def test_native_whisperx_missing_dependency_is_stable(tmp_path: Path) -> None:
    secret = "whisperx-import-secret"
    backend = NativeWhisperXBackend(
        importer=lambda _name: (_ for _ in ()).throw(ModuleNotFoundError(secret))
    )

    with pytest.raises(AlignmentUnavailableError) as raised:
        backend.load_model("en", "cpu", tmp_path / "models")

    assert secret not in "".join(traceback.format_exception(raised.value))
    assert raised.value.__cause__ is None


def test_native_whisperx_missing_dependency_degrades_per_language(tmp_path: Path) -> None:
    backend = NativeWhisperXBackend(
        importer=lambda _name: (_ for _ in ()).throw(ModuleNotFoundError("whisperx"))
    )

    result = WhisperXAligner(backend, tmp_path / "models").align(
        tmp_path / "full-audio.wav",
        (_segment("en"),),
    )

    assert result.preserved_segments == (_segment("en"),)
    assert result.words == ()
    assert result.warning_codes == ("ALIGNMENT_MODEL_UNAVAILABLE:en",)


def test_native_whisperx_full_audio_absolute_segments_and_language_model_cache(
    tmp_path: Path,
) -> None:
    calls: list[object] = []

    def importer(_name: str) -> object:
        class Module:
            @staticmethod
            def load_align_model(**kwargs: object) -> tuple[object, object]:
                calls.append(("load", kwargs))
                return ("model", "metadata")

            @staticmethod
            def align(*args: object, **kwargs: object) -> dict[str, object]:
                calls.append(("align", args, kwargs))
                return {
                    "word_segments": [
                        {"word": "hello", "start": 1.0, "end": 2.0, "score": 0.9}
                    ]
                }

        return Module()

    full_audio = tmp_path / "runs/scope/run_001/media/audio.wav"
    aligner = WhisperXAligner(NativeWhisperXBackend(importer=importer), tmp_path / "models")

    first = aligner.align(full_audio, (_segment("en"),))
    second = aligner.align(full_audio, (_segment("en"),))

    assert first.words == second.words
    assert sum(call[0] == "load" for call in calls) == 1  # type: ignore[index]
    align_calls = [call for call in calls if call[0] == "align"]  # type: ignore[index]
    assert len(align_calls) == 2
    args = align_calls[0][1]  # type: ignore[index]
    assert args[0] == [{"start": 1.0, "end": 3.0, "text": "こんにちは"}]
    assert args[3] == str(full_audio)


def test_native_whisperx_invalid_response_still_fails_closed(tmp_path: Path) -> None:
    def importer(_name: str) -> object:
        class Module:
            @staticmethod
            def load_align_model(**_kwargs: object) -> tuple[object, object]:
                return (object(), object())

            @staticmethod
            def align(*_args: object, **_kwargs: object) -> dict[str, object]:
                return {"unexpected": []}

        return Module()

    with pytest.raises(VideoDemoError) as raised:
        WhisperXAligner(
            NativeWhisperXBackend(importer=importer),
            tmp_path / "models",
        ).align(tmp_path / "audio.wav", (_segment("en"),))

    assert raised.value.code == ErrorCode.SPEECH_MODEL_UNAVAILABLE


def test_whisperx_invalid_word_value_drops_sensitive_traceback(tmp_path: Path) -> None:
    secret = "whisperx-word-secret"

    class Backend:
        def load_model(self, *_args: object) -> object:
            return (object(), object())

        def align(self, *_args: object) -> list[dict[str, object]]:
            return [
                {
                    "word": "hello",
                    "start": secret,
                    "end": 2.0,
                    "score": 0.9,
                }
            ]

    with pytest.raises(VideoDemoError) as raised:
        WhisperXAligner(Backend(), tmp_path / "models").align(
            tmp_path / "audio.wav",
            (_segment("en"),),
        )

    assert raised.value.code == ErrorCode.SPEECH_MODEL_UNAVAILABLE
    assert raised.value.__cause__ is None
    assert secret not in "".join(traceback.format_exception(raised.value))


def test_whisperx_word_text_drops_sensitive_traceback(tmp_path: Path) -> None:
    secret = "whisperx-word-text-secret"

    class Text:
        def __str__(self) -> str:
            raise RuntimeError(secret)

    class Backend:
        def load_model(self, *_args: object) -> object:
            return (object(), object())

        def align(self, *_args: object) -> list[dict[str, object]]:
            return [{"word": Text(), "start": 1.0, "end": 2.0, "score": 0.9}]

    with pytest.raises(VideoDemoError) as raised:
        WhisperXAligner(Backend(), tmp_path / "models").align(
            tmp_path / "audio.wav",
            (_segment("en"),),
        )

    assert raised.value.code == ErrorCode.SPEECH_MODEL_UNAVAILABLE
    assert raised.value.__cause__ is None
    assert secret not in "".join(traceback.format_exception(raised.value))
