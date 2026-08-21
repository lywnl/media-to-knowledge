from __future__ import annotations

import csv
import importlib
import sys
import warnings
import wave
from array import array
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

from pydantic import Field

from video_demo.domain.base import FrozenModel, Probability, Sha256, stable_identifier
from video_demo.domain.evidence import AudioEvent
from video_demo.errors import ErrorCode, VideoDemoError


@dataclass(frozen=True, slots=True)
class RawAudioEventFrame:
    start_ms: int
    end_ms: int
    audioset_class: str
    confidence: float


class YamnetThresholds(FrozenModel):
    schema_version: str
    threshold_version: str
    default_threshold: Probability
    class_thresholds: dict[str, Probability]
    excluded_classes: tuple[str, ...]
    normalized_classes: dict[str, str]
    merge_gap_ms: int = Field(ge=0, le=10_000)
    calibration_dataset_sha256: Sha256 | None


class YamnetBackend(Protocol):
    def predict(self, samples: array[float]) -> Sequence[Sequence[float]]: ...


def import_tensorflow_hub(
    *,
    importer: Callable[[str], object] = importlib.import_module,
) -> object:
    """只屏蔽 TensorFlow Hub 仍依赖 pkg_resources 的已知上游警告。"""

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"pkg_resources is deprecated as an API\..*",
            category=UserWarning,
        )
        return importer("tensorflow_hub")


class NativeYamnetBackend:
    """TensorFlow Hub YAMNet 的懒加载边界。"""

    def __init__(
        self,
        model_path: Path,
        *,
        importer: Callable[[str], object] = importlib.import_module,
    ) -> None:
        self._model_path = model_path
        self._importer = importer
        self._model: Any | None = None

    def predict(self, samples: array[float]) -> Sequence[Sequence[float]]:
        model = self._load_model()
        try:
            output = model(samples)
            scores = output[0]
            return cast(Sequence[Sequence[float]], scores.numpy().tolist())
        except Exception:
            raise VideoDemoError(
                ErrorCode.SPEECH_MODEL_UNAVAILABLE,
                "YAMNet 推理失败",
            ) from None

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        if not self._model_path.is_dir():
            raise VideoDemoError(ErrorCode.SPEECH_MODEL_UNAVAILABLE, "工作区内缺少 YAMNet 模型")
        try:
            hub: Any = import_tensorflow_hub(importer=self._importer)
        except (ModuleNotFoundError, ImportError):
            raise VideoDemoError(
                ErrorCode.SPEECH_DEPENDENCY_UNAVAILABLE,
                "未安装 YAMNet 可选依赖",
            ) from None
        try:
            self._model = hub.load(str(self._model_path))
        except Exception:
            raise VideoDemoError(
                ErrorCode.SPEECH_MODEL_UNAVAILABLE,
                "YAMNet 模型不可用",
            ) from None
        return self._model


class NativeYamnetDetector:
    def __init__(
        self,
        *,
        backend_factory: Callable[[], YamnetBackend],
        class_map_path: Path,
        thresholds_path: Path,
    ) -> None:
        self._backend_factory = backend_factory
        self._class_map_path = class_map_path
        self._thresholds_path = thresholds_path
        self._backend: YamnetBackend | None = None

    def detect(self, audio: Path, *, duration_ms: int) -> tuple[AudioEvent, ...]:
        samples = _read_pcm16_mono(audio)
        class_names = _load_class_names(self._class_map_path)
        thresholds = load_thresholds(self._thresholds_path)
        if self._backend is None:
            self._backend = self._backend_factory()
        scores = self._backend.predict(samples)
        return aggregate_audio_events(
            scores_to_frames(scores=scores, class_names=class_names, duration_ms=duration_ms),
            thresholds,
        )


def load_thresholds(path: Path) -> YamnetThresholds:
    try:
        return YamnetThresholds.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception:
        raise VideoDemoError(
            ErrorCode.SPEECH_MODEL_UNAVAILABLE,
            "YAMNet 阈值配置不可用",
        ) from None


def scores_to_frames(
    *,
    scores: Sequence[Sequence[float]],
    class_names: Sequence[str],
    duration_ms: int,
) -> tuple[RawAudioEventFrame, ...]:
    if duration_ms <= 0:
        raise ValueError("duration_ms 必须大于 0")
    frames: list[RawAudioEventFrame] = []
    for class_index, class_name in enumerate(class_names):
        for frame_index, frame_scores in enumerate(scores):
            if len(frame_scores) != len(class_names):
                raise ValueError("YAMNet score 列数与类别数不一致")
            start_ms = frame_index * 480
            if start_ms >= duration_ms:
                continue
            end_ms = min(start_ms + 960, duration_ms)
            confidence = float(frame_scores[class_index])
            if not 0 <= confidence <= 1:
                raise ValueError("YAMNet score 必须在 0 到 1 之间")
            frames.append(
                RawAudioEventFrame(
                    start_ms=start_ms,
                    end_ms=end_ms,
                    audioset_class=class_name,
                    confidence=confidence,
                ),
            )
    return tuple(frames)


def aggregate_audio_events(
    frames: Sequence[RawAudioEventFrame],
    thresholds: YamnetThresholds,
) -> tuple[AudioEvent, ...]:
    excluded = frozenset(thresholds.excluded_classes)
    accepted = [
        frame
        for frame in frames
        if frame.audioset_class not in excluded
        and frame.audioset_class in thresholds.normalized_classes
        and frame.confidence
        >= thresholds.class_thresholds.get(
            frame.audioset_class,
            thresholds.default_threshold,
        )
    ]
    accepted.sort(key=lambda item: (item.audioset_class, item.start_ms, item.end_ms))
    grouped: list[RawAudioEventFrame] = []
    for frame in accepted:
        if (
            grouped
            and grouped[-1].audioset_class == frame.audioset_class
            and frame.start_ms - grouped[-1].end_ms <= thresholds.merge_gap_ms
        ):
            previous = grouped[-1]
            grouped[-1] = RawAudioEventFrame(
                start_ms=previous.start_ms,
                end_ms=max(previous.end_ms, frame.end_ms),
                audioset_class=previous.audioset_class,
                confidence=max(previous.confidence, frame.confidence),
            )
        else:
            grouped.append(frame)

    events = [
        AudioEvent(
            evidence_id=stable_identifier(
                "audio",
                {
                    "audioset_class": item.audioset_class,
                    "start_ms": item.start_ms,
                    "end_ms": item.end_ms,
                },
            ),
            start_ms=item.start_ms,
            end_ms=item.end_ms,
            audioset_class=item.audioset_class,
            normalized_event=thresholds.normalized_classes[item.audioset_class],
            confidence=item.confidence,
            threshold_version=thresholds.threshold_version,
        )
        for item in grouped
    ]
    return tuple(sorted(events, key=lambda item: (item.start_ms, item.end_ms, item.audioset_class)))


def _read_pcm16_mono(path: Path) -> array[float]:
    try:
        with wave.open(str(path), "rb") as stream:
            if (
                stream.getnchannels() != 1
                or stream.getsampwidth() != 2
                or stream.getframerate() != 16_000
                or stream.getcomptype() != "NONE"
            ):
                raise VideoDemoError(
                    ErrorCode.SPEECH_AUDIO_INVALID,
                    "YAMNet 输入必须是 16 kHz 单声道 PCM WAV",
                )
            frames = stream.readframes(stream.getnframes())
    except (OSError, EOFError, wave.Error):
        raise VideoDemoError(ErrorCode.SPEECH_AUDIO_INVALID, "YAMNet WAV 输入非法") from None
    if not frames:
        raise VideoDemoError(ErrorCode.SPEECH_AUDIO_INVALID, "YAMNet WAV 输入为空")
    values = array("h")
    values.frombytes(frames)
    if sys.byteorder != "little":
        values.byteswap()
    return array("f", (value / 32768.0 for value in values))


def _load_class_names(path: Path) -> tuple[str, ...]:
    try:
        with path.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
    except OSError:
        raise VideoDemoError(
            ErrorCode.SPEECH_MODEL_UNAVAILABLE,
            "YAMNet 类别映射不可用",
        ) from None
    try:
        ordered = sorted(rows, key=lambda row: int(row["index"]))
        names = tuple(row["display_name"].strip() for row in ordered)
    except (KeyError, TypeError, ValueError):
        raise VideoDemoError(
            ErrorCode.SPEECH_MODEL_UNAVAILABLE,
            "YAMNet 类别映射非法",
        ) from None
    if not names or any(not name for name in names):
        raise VideoDemoError(ErrorCode.SPEECH_MODEL_UNAVAILABLE, "YAMNet 类别映射为空")
    return names
