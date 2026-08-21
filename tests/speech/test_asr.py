from __future__ import annotations

import tomllib
import traceback
from pathlib import Path

import pytest

from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.speech.asr import (
    FasterWhisperAdapter,
    NativeFasterWhisperBackend,
    RawAsrSegment,
    build_speech_segments,
)
from video_demo.speech.language import LanguageSpan


def test_speech_extra_declares_compatible_runtime_dependencies() -> None:
    project_root = Path(__file__).resolve().parents[2]
    with (project_root / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)

    speech_dependencies = project["project"]["optional-dependencies"]["speech"]

    assert "faster-whisper>=1.2,<2" in speech_dependencies
    assert "huggingface-hub>=0.28,<1" in speech_dependencies
    assert "pyannote.audio>=4,<5" in speech_dependencies
    assert any(item.startswith("requests") for item in speech_dependencies)
    assert "torch>=2.8,<2.9" in speech_dependencies
    assert "torchaudio>=2.8,<2.9" in speech_dependencies
    assert "transformers>=4.48,<5" in speech_dependencies
    assert "whisperx==3.4.2" in speech_dependencies


def test_build_asr_segments_preserves_original_language_text_and_absolute_time() -> None:
    language_span = LanguageSpan(
        evidence_id="lid_001",
        start_ms=10_000,
        end_ms=20_000,
        language="en",
        confidence=0.9,
        is_fully_evaluated_language=True,
    )

    segments = build_speech_segments(
        language_span,
        (
            RawAsrSegment(start_ms=0, end_ms=1_500, text=" Hello world ", confidence=0.85),
        ),
    )

    assert len(segments) == 1
    assert segments[0].start_ms == 10_000
    assert segments[0].end_ms == 11_500
    assert segments[0].text == "Hello world"
    assert segments[0].language == "en"
    assert segments[0].is_fully_evaluated_language is True


def test_build_asr_segments_clamps_one_whisper_timestamp_tick_at_slice_end() -> None:
    language_span = LanguageSpan(
        evidence_id="lid_001",
        start_ms=25_890,
        end_ms=30_000,
        language="zh",
        confidence=0.9,
        is_fully_evaluated_language=True,
    )

    segments = build_speech_segments(
        language_span,
        (
            RawAsrSegment(
                start_ms=3_000,
                end_ms=4_120,
                text="量化到下一个时间刻度",
                confidence=0.85,
            ),
        ),
    )

    assert [(item.start_ms, item.end_ms) for item in segments] == [(28_890, 30_000)]


def test_build_asr_segments_clamps_timestamp_after_unaligned_slice_end() -> None:
    language_span = LanguageSpan(
        evidence_id="lid_001",
        start_ms=0,
        end_ms=8_668,
        language="zh",
        confidence=0.9,
        is_fully_evaluated_language=True,
    )

    segments = build_speech_segments(
        language_span,
        (
            RawAsrSegment(
                start_ms=6_140,
                end_ms=8_700,
                text="真实失败切片的末段",
                confidence=0.85,
            ),
        ),
    )

    assert [(item.start_ms, item.end_ms) for item in segments] == [(6_140, 8_668)]


def test_build_asr_segments_rejects_overrun_beyond_one_timestamp_tick() -> None:
    language_span = LanguageSpan(
        evidence_id="lid_001",
        start_ms=25_890,
        end_ms=30_000,
        language="zh",
        confidence=0.9,
        is_fully_evaluated_language=True,
    )

    with pytest.raises(ValueError, match="超出语言窗口"):
        build_speech_segments(
            language_span,
            (
                RawAsrSegment(
                    start_ms=3_000,
                    end_ms=4_141,
                    text="超出量化容差",
                    confidence=0.85,
                ),
            ),
        )


def test_faster_whisper_adapter_disables_second_vad_and_translation(tmp_path: Path) -> None:
    captured: list[dict[str, object]] = []

    class Backend:
        def transcribe(self, audio: Path, **kwargs: object) -> tuple[object, object]:
            captured.append({"audio": audio, **kwargs})

            class Segment:
                start = 0.0
                end = 1.0
                text = "こんにちは"
                avg_logprob = -0.2
                no_speech_prob = 0.1

            return iter((Segment(),)), object()

    audio_slice = tmp_path / "slice.wav"
    audio_slice.write_bytes(b"wav")
    language_span = LanguageSpan(
        evidence_id="lid_001",
        start_ms=0,
        end_ms=2_000,
        language="ja",
        confidence=0.9,
        is_fully_evaluated_language=True,
    )
    adapter = FasterWhisperAdapter(Backend())

    result = adapter.transcribe_slice(audio_slice, language_span)

    assert result[0].text == "こんにちは"
    assert captured[0]["language"] == "ja"
    assert captured[0]["task"] == "transcribe"
    assert captured[0]["vad_filter"] is False
    assert captured[0]["word_timestamps"] is False
    assert captured[0]["condition_on_previous_text"] is False


def test_faster_whisper_model_is_downloaded_flat_then_loaded_offline(tmp_path: Path) -> None:
    from video_demo.speech.asr import load_faster_whisper_model

    calls: list[tuple[str, object, dict[str, object]]] = []

    def downloader(model_id: str, **kwargs: object) -> str:
        calls.append(("download", model_id, kwargs))
        return str(tmp_path / "unexpected-snapshot")

    def factory(model_id: str, **kwargs: object) -> object:
        calls.append(("load", model_id, kwargs))
        return object()

    model = load_faster_whisper_model(
        factory,
        tmp_path / "models",
        downloader=downloader,
    )

    assert model is not None
    assert calls == [
        (
            "download",
            "large-v3",
            {
                "output_dir": str(tmp_path / "models" / "faster-whisper"),
                "cache_dir": str(tmp_path / "cache" / "huggingface"),
            },
        ),
        (
            "load",
            str(tmp_path / "models" / "faster-whisper"),
            {
                "device": "cpu",
                "compute_type": "int8",
                "local_files_only": True,
            },
        ),
    ]


def test_faster_whisper_complete_local_model_skips_network_download(tmp_path: Path) -> None:
    from video_demo.speech.asr import load_faster_whisper_model

    model_dir = tmp_path / "models" / "faster-whisper"
    model_dir.mkdir(parents=True)
    for filename in (
        "config.json",
        "model.bin",
        "preprocessor_config.json",
        "tokenizer.json",
        "vocabulary.json",
    ):
        (model_dir / filename).write_bytes(b"model")
    loaded: list[tuple[str, dict[str, object]]] = []

    def fail_download(_model_id: str, **_kwargs: object) -> str:
        raise AssertionError("完整本地模型不应访问网络")

    def factory(model_id: str, **kwargs: object) -> object:
        loaded.append((model_id, kwargs))
        return object()

    load_faster_whisper_model(
        factory,
        tmp_path / "models",
        downloader=fail_download,
    )

    assert loaded == [
        (
            str(model_dir),
            {
                "device": "cpu",
                "compute_type": "int8",
                "local_files_only": True,
            },
        )
    ]


@pytest.mark.parametrize("linked_name", ("models", "cache"))
def test_faster_whisper_rejects_symlinked_model_or_cache_parent(
    tmp_path: Path,
    linked_name: str,
) -> None:
    from video_demo.speech.asr import load_faster_whisper_model

    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (runtime_root / linked_name).symlink_to(outside, target_is_directory=True)

    with pytest.raises(VideoDemoError) as raised:
        load_faster_whisper_model(
            lambda *_args, **_kwargs: object(),
            runtime_root / "models",
            downloader=lambda *_args, **_kwargs: "unused",
        )

    assert raised.value.code == ErrorCode.WORKSPACE_PATH_ESCAPE
    assert tuple(outside.iterdir()) == ()


def test_native_faster_whisper_loads_lazily_with_configured_cpu_compute_type(
    tmp_path: Path,
) -> None:
    calls: list[object] = []

    class Model:
        def transcribe(self, audio: str, **kwargs: object) -> tuple[object, object]:
            calls.append(("transcribe", audio, kwargs))
            return ((), object())

    def importer(name: str) -> object:
        calls.append(("import", name))

        if name == "faster_whisper.utils":
            class UtilsModule:
                download_model = staticmethod(
                    lambda model_id, **kwargs: calls.append(
                        ("download", model_id, kwargs)
                    )
                )

            return UtilsModule()

        class Module:
            WhisperModel = staticmethod(
                lambda model_id, **kwargs: calls.append(("load", model_id, kwargs)) or Model()
            )

        return Module()

    backend = NativeFasterWhisperBackend(
        tmp_path / "models",
        device="cpu",
        compute_type="float32",
        importer=importer,
    )
    assert calls == []

    backend.transcribe(tmp_path / "slice.wav", language="en")
    backend.transcribe(tmp_path / "slice.wav", language="zh")

    assert calls[0] == ("import", "faster_whisper")
    assert calls[1] == ("import", "faster_whisper.utils")
    assert calls[2] == (
        "download",
        "large-v3",
        {
            "output_dir": str(tmp_path / "models" / "faster-whisper"),
            "cache_dir": str(tmp_path / "cache" / "huggingface"),
        },
    )
    assert calls[3] == (
        "load",
        str(tmp_path / "models" / "faster-whisper"),
        {
            "device": "cpu",
            "compute_type": "float32",
            "local_files_only": True,
        },
    )
    assert sum(item[0] == "load" for item in calls if isinstance(item, tuple)) == 1


def test_native_faster_whisper_translates_missing_dependency(tmp_path: Path) -> None:
    secret = "faster-whisper-import-secret"

    def missing(_name: str) -> object:
        raise ModuleNotFoundError(secret)

    backend = NativeFasterWhisperBackend(tmp_path / "models", importer=missing)

    with pytest.raises(VideoDemoError) as raised:
        backend.transcribe(tmp_path / "slice.wav")

    assert raised.value.code == ErrorCode.SPEECH_DEPENDENCY_UNAVAILABLE
    assert raised.value.details == {}
    assert raised.value.__cause__ is None
    assert secret not in "".join(traceback.format_exception(raised.value))


def test_native_faster_whisper_inference_drops_sensitive_traceback(tmp_path: Path) -> None:
    secret = "faster-whisper-inference-secret"

    class Model:
        def transcribe(self, *_args: object, **_kwargs: object) -> object:
            raise RuntimeError(secret)

    def importer(_name: str) -> object:
        class Module:
            WhisperModel = staticmethod(lambda *_args, **_kwargs: Model())

        return Module()

    backend = NativeFasterWhisperBackend(tmp_path / "models", importer=importer)

    with pytest.raises(VideoDemoError) as raised:
        backend.transcribe(tmp_path / "slice.wav")

    assert raised.value.code == ErrorCode.SPEECH_MODEL_UNAVAILABLE
    assert raised.value.__cause__ is None
    assert secret not in "".join(traceback.format_exception(raised.value))


def test_faster_whisper_adapter_drops_lazy_generator_sensitive_traceback(
    tmp_path: Path,
) -> None:
    secret = "faster-whisper-generator-secret"

    class Backend:
        def transcribe(self, _audio: Path, **_kwargs: object) -> tuple[object, object]:
            def segments() -> object:
                raise RuntimeError(secret)
                yield object()

            return (segments(), object())

    language_span = LanguageSpan(
        evidence_id="lid_001",
        start_ms=0,
        end_ms=1_000,
        language="en",
        confidence=0.9,
        is_fully_evaluated_language=True,
    )

    with pytest.raises(VideoDemoError) as raised:
        FasterWhisperAdapter(Backend()).transcribe_slice(
            tmp_path / "slice.wav",
            language_span,
        )

    assert raised.value.code == ErrorCode.SPEECH_MODEL_UNAVAILABLE
    assert raised.value.__cause__ is None
    assert secret not in "".join(traceback.format_exception(raised.value))
