from __future__ import annotations

import importlib
import wave
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

from video_demo.domain.base import stable_identifier
from video_demo.domain.evidence import SpeakerId, SpeakerTurn
from video_demo.errors import ErrorCode, VideoDemoError


@dataclass(frozen=True, slots=True)
class RawSpeakerTurn:
    start_ms: int
    end_ms: int
    raw_speaker: str


class DiarizationBackend(Protocol):
    def diarize(
        self,
        audio: Path,
        *,
        min_speakers: int | None,
        max_speakers: int | None,
    ) -> tuple[RawSpeakerTurn, ...]: ...


class PyannoteDiarizer:
    def __init__(self, backend: DiarizationBackend) -> None:
        self._backend = backend

    def diarize(
        self,
        audio: Path,
        *,
        min_speakers: int | None,
        max_speakers: int | None,
    ) -> tuple[SpeakerTurn, ...]:
        raw = self._backend.diarize(
            audio,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )
        return normalize_speaker_turns(raw)


class NativePyannoteBackend:
    """pyannote community pipeline 的懒加载、Secret 隔离边界。"""

    def __init__(
        self,
        token: str | Callable[[], str | None] | None,
        *,
        model_root: Path | None = None,
        importer: Callable[[str], object] = importlib.import_module,
        audio_loader: Callable[[Path], Mapping[str, object]] | None = None,
    ) -> None:
        self._token_provider = token if callable(token) else lambda: token
        self._model_root = model_root
        self._importer = importer
        self._audio_loader = audio_loader or _load_pcm16_waveform
        self._pipeline: Any | None = None

    def diarize(
        self,
        audio: Path,
        *,
        min_speakers: int | None,
        max_speakers: int | None,
    ) -> tuple[RawSpeakerTurn, ...]:
        pipeline = self._load_pipeline()
        kwargs = {
            name: value
            for name, value in (
                ("min_speakers", min_speakers),
                ("max_speakers", max_speakers),
            )
            if value is not None
        }
        try:
            output = pipeline(self._audio_loader(audio), **kwargs)
            annotation = getattr(output, "speaker_diarization", output)
            return tuple(
                RawSpeakerTurn(
                    start_ms=round(float(segment.start) * 1000),
                    end_ms=round(float(segment.end) * 1000),
                    raw_speaker=str(speaker),
                )
                for segment, _track, speaker in self._iter_tracks(annotation)
            )
        except VideoDemoError:
            raise
        except Exception:
            raise VideoDemoError(
                ErrorCode.PYANNOTE_MODEL_UNAVAILABLE,
                "pyannote 说话人分离失败",
            ) from None

    @staticmethod
    def _iter_tracks(annotation: object) -> Iterable[tuple[Any, object, object]]:
        iterator = getattr(annotation, "itertracks", None)
        if not callable(iterator):
            raise VideoDemoError(ErrorCode.SPEECH_MODEL_UNAVAILABLE, "pyannote 返回结构非法")
        return cast(Iterable[tuple[Any, object, object]], iterator(yield_label=True))

    def _load_pipeline(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline
        token = self._token_provider()
        if not token:
            raise VideoDemoError(
                ErrorCode.PYANNOTE_AUTHENTICATION_FAILED,
                "pyannote 缺少 Hugging Face Token",
            )
        try:
            module: Any = self._importer("pyannote.audio")
        except (ModuleNotFoundError, ImportError):
            raise VideoDemoError(
                ErrorCode.SPEECH_DEPENDENCY_UNAVAILABLE,
                "未安装 pyannote 可选依赖",
            ) from None
        try:
            self._pipeline = load_pyannote_pipeline(
                module.Pipeline.from_pretrained,
                token,
                self._model_root,
            )
        except Exception as error:
            if _is_pyannote_auth_failure(error):
                translated = VideoDemoError(
                    ErrorCode.PYANNOTE_AUTHENTICATION_FAILED,
                    "pyannote 鉴权失败或模型条款不可用",
                )
            else:
                translated = VideoDemoError(
                    ErrorCode.PYANNOTE_MODEL_UNAVAILABLE,
                    "pyannote 模型不可用",
                )
            raise translated from None
        return self._pipeline


def normalize_speaker_turns(raw_turns: Sequence[RawSpeakerTurn]) -> tuple[SpeakerTurn, ...]:
    ordered = sorted(raw_turns, key=lambda item: (item.start_ms, item.end_ms, item.raw_speaker))
    speaker_mapping: dict[str, SpeakerId] = {}
    for raw_turn in ordered:
        if raw_turn.raw_speaker not in speaker_mapping:
            next_index = len(speaker_mapping) + 1
            if next_index > 10:
                raise ValueError("说话人数超过契约上限 10")
            speaker_mapping[raw_turn.raw_speaker] = f"SPEAKER_{next_index:02d}"  # type: ignore[assignment]

    normalized = [
        SpeakerTurn(
            evidence_id=stable_identifier(
                "speaker",
                {
                    "speaker": speaker_mapping[raw_turn.raw_speaker],
                    "start_ms": raw_turn.start_ms,
                    "end_ms": raw_turn.end_ms,
                },
            ),
            start_ms=raw_turn.start_ms,
            end_ms=raw_turn.end_ms,
            speaker=speaker_mapping[raw_turn.raw_speaker],
        )
        for raw_turn in ordered
    ]
    with_overlap: list[SpeakerTurn] = []
    for index, normalized_turn in enumerate(normalized):
        overlaps = tuple(
            other.speaker
            for other_index, other in enumerate(normalized)
            if other_index != index
            and normalized_turn.overlaps(other)
            and other.speaker != normalized_turn.speaker
        )
        unique_overlaps = tuple(dict.fromkeys(overlaps))
        with_overlap.append(
            normalized_turn.model_copy(update={"overlap_speakers": unique_overlaps}),
        )
    return tuple(with_overlap)


def load_pyannote_pipeline(
    factory: Callable[..., object],
    token: str,
    model_root: Path | None = None,
) -> object:
    if not token:
        raise ValueError("Hugging Face Token 不能为空")
    kwargs: dict[str, object] = {"token": token}
    if model_root is not None:
        cache_dir = model_root / "pyannote"
        cache_dir.mkdir(parents=True, exist_ok=True)
        kwargs["cache_dir"] = str(cache_dir)
    return factory("pyannote/speaker-diarization-community-1", **kwargs)


def _load_pcm16_waveform(audio: Path) -> Mapping[str, object]:
    torch: Any = importlib.import_module("torch")
    with wave.open(str(audio), "rb") as stream:
        channels = stream.getnchannels()
        sample_rate = stream.getframerate()
        sample_width = stream.getsampwidth()
        compression = stream.getcomptype()
        frames = stream.readframes(stream.getnframes())
    if channels < 1 or sample_rate < 1 or sample_width != 2 or compression != "NONE":
        raise ValueError("pyannote 仅接受 PCM16 WAV 音频")
    waveform = torch.frombuffer(bytearray(frames), dtype=torch.int16)
    if waveform.numel() % channels != 0:
        raise ValueError("pyannote PCM16 WAV 帧长度非法")
    waveform = waveform.reshape(-1, channels).transpose(0, 1).contiguous()
    return {
        "waveform": waveform.to(dtype=torch.float32).div_(32_768),
        "sample_rate": sample_rate,
        "uri": audio.stem,
    }


def _is_pyannote_auth_failure(error: Exception) -> bool:
    if isinstance(error, PermissionError):
        return True
    status = getattr(error, "status_code", None)
    response = getattr(error, "response", None)
    if status is None and response is not None:
        status = getattr(response, "status_code", None)
    return status in {401, 403} or error.__class__.__name__ in {
        "GatedRepoError",
        "RepositoryNotFoundError",
    }
