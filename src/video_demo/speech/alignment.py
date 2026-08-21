from __future__ import annotations

import importlib
import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Never, Protocol, cast

from video_demo.domain.base import FrozenModel, stable_identifier
from video_demo.domain.evidence import AlignedWord, SpeechSegment
from video_demo.errors import ErrorCode, VideoDemoError


class AlignmentUnavailableError(Exception):
    """仅表示 alignment 依赖或某语言模型不可用，可按语言降级。"""


class AlignmentBackend(Protocol):
    def load_model(self, language: str, device: str, model_dir: Path) -> object: ...

    def align(
        self,
        segments: list[dict[str, object]],
        model: object,
        metadata: object,
        audio: Path,
        device: str,
    ) -> list[dict[str, object]]: ...


class AlignmentResult(FrozenModel):
    words: tuple[AlignedWord, ...]
    preserved_segments: tuple[SpeechSegment, ...]
    warning_codes: tuple[str, ...]


class NativeWhisperXBackend:
    """WhisperX 官方 alignment API 的懒加载适配器。"""

    def __init__(
        self,
        *,
        importer: Callable[[str], object] = importlib.import_module,
    ) -> None:
        self._importer = importer
        self._module: Any | None = None

    def load_model(
        self,
        language: str,
        device: str,
        model_dir: Path,
    ) -> tuple[object, object]:
        module = self._load_module()
        model_dir.mkdir(parents=True, exist_ok=True)
        try:
            return cast(
                tuple[object, object],
                module.load_align_model(
                    language_code=language,
                    device=device,
                    model_dir=str(model_dir),
                ),
            )
        except Exception:
            raise AlignmentUnavailableError from None

    def align(
        self,
        segments: list[dict[str, object]],
        model: object,
        metadata: object,
        audio: Path,
        device: str,
    ) -> list[dict[str, object]]:
        module = self._load_module()
        try:
            result = module.align(
                segments,
                model,
                metadata,
                str(audio),
                device,
                return_char_alignments=False,
            )
        except Exception:
            raise VideoDemoError(
                ErrorCode.SPEECH_MODEL_UNAVAILABLE,
                "WhisperX 对齐失败",
            ) from None
        if not isinstance(result, dict) or not isinstance(result.get("word_segments"), list):
            raise VideoDemoError(
                ErrorCode.SPEECH_MODEL_UNAVAILABLE,
                "WhisperX 返回结构非法",
            )
        return cast(list[dict[str, object]], result["word_segments"])

    def _load_module(self) -> Any:
        if self._module is not None:
            return self._module
        try:
            self._module = self._importer("whisperx")
        except (ModuleNotFoundError, ImportError):
            raise AlignmentUnavailableError from None
        return self._module


class WhisperXAligner:
    def __init__(self, backend: AlignmentBackend, model_root: Path) -> None:
        self._backend = backend
        self._model_root = model_root
        self._models: dict[str, tuple[object, object]] = {}

    def probe_languages(self, languages: Sequence[str]) -> dict[str, bool]:
        return {language: self._probe_language(language) for language in languages}

    def align(
        self,
        audio: Path,
        segments: Sequence[SpeechSegment],
    ) -> AlignmentResult:
        words: list[AlignedWord] = []
        warnings: list[str] = []
        by_language: dict[str, list[SpeechSegment]] = {}
        for segment in segments:
            by_language.setdefault(segment.language, []).append(segment)

        for language, language_segments in by_language.items():
            if language == "und":
                warnings.append("ALIGNMENT_MODEL_UNAVAILABLE:und")
                continue
            try:
                model, metadata = self._load_language_model(language)
            except (AlignmentUnavailableError, LookupError):
                warnings.append(f"ALIGNMENT_MODEL_UNAVAILABLE:{language}")
                continue
            raw_words = self._backend.align(
                [_whisperx_segment(segment) for segment in language_segments],
                model,
                metadata,
                audio,
                "cpu",
            )
            built_words, has_unaligned_words = _build_words(
                language,
                language_segments,
                raw_words,
            )
            words.extend(built_words)
            if has_unaligned_words:
                warnings.append(f"ALIGNMENT_WORD_UNALIGNED:{language}")

        return AlignmentResult(
            words=tuple(sorted(words, key=lambda item: (item.start_ms, item.end_ms))),
            preserved_segments=tuple(segments),
            warning_codes=tuple(warnings),
        )

    def _probe_language(self, language: str) -> bool:
        try:
            self._load_language_model(language)
        except (AlignmentUnavailableError, LookupError):
            return False
        return True

    def _load_language_model(self, language: str) -> tuple[object, object]:
        cached = self._models.get(language)
        if cached is not None:
            return cached
        model_dir = self._model_root / "whisperx" / language
        model_dir.mkdir(parents=True, exist_ok=True)
        loaded = self._backend.load_model(language, "cpu", model_dir)
        if not isinstance(loaded, tuple) or len(loaded) != 2:
            raise VideoDemoError(
                ErrorCode.SPEECH_MODEL_UNAVAILABLE,
                "WhisperX alignment loader 返回结构非法",
            )
        self._models[language] = loaded
        return loaded


def _whisperx_segment(segment: SpeechSegment) -> dict[str, object]:
    return {
        "start": segment.start_ms / 1000,
        "end": segment.end_ms / 1000,
        "text": segment.text,
    }


def _build_words(
    language: str,
    segments: Sequence[SpeechSegment],
    raw_words: Sequence[dict[str, object]],
) -> tuple[list[AlignedWord], bool]:
    words: list[AlignedWord] = []
    has_unaligned_words = False
    for raw in raw_words:
        try:
            text = str(raw.get("word", "")).strip()
        except Exception:
            raise VideoDemoError(
                ErrorCode.SPEECH_MODEL_UNAVAILABLE,
                "WhisperX 返回结构非法",
            ) from None
        if not text:
            continue
        required_fields = ("start", "end", "score")
        present_fields = tuple(field in raw for field in required_fields)
        if not any(present_fields):
            has_unaligned_words = True
            continue
        if not all(present_fields):
            _raise_invalid_word()
        try:
            start_ms = round(_number(raw["start"], "start") * 1000)
            end_ms = round(_number(raw["end"], "end") * 1000)
            score = _number(raw["score"], "score")
        except (KeyError, OverflowError, TypeError, ValueError):
            _raise_invalid_word()
        if start_ms < 0 or end_ms <= start_ms or not 0 <= score <= 1:
            _raise_invalid_word()
        if not any(
            segment.start_ms <= start_ms < end_ms <= segment.end_ms for segment in segments
        ):
            has_unaligned_words = True
            continue
        words.append(
            AlignedWord(
                evidence_id=stable_identifier(
                    "word",
                    {
                        "language": language,
                        "start_ms": start_ms,
                        "end_ms": end_ms,
                        "text": text,
                    },
                ),
                start_ms=start_ms,
                end_ms=end_ms,
                text=text,
                language=language,
                probability=score,
            ),
        )
    return words, has_unaligned_words


def _number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(f"WhisperX {field} 必须是数值")
    try:
        number = float(value)
    except (OverflowError, ValueError):
        raise ValueError(f"WhisperX {field} 必须是数值") from None
    if not math.isfinite(number):
        raise ValueError(f"WhisperX {field} 必须是有限数值")
    return number


def _raise_invalid_word() -> Never:
    raise VideoDemoError(
        ErrorCode.SPEECH_MODEL_UNAVAILABLE,
        "WhisperX 返回了非法词时间或置信度",
    ) from None
