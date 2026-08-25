from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Protocol

from video_demo.application.pipeline_contracts import (
    PreparedMedia,
    SpeechAnalysis,
    SpeechBoundaryCandidate,
)
from video_demo.domain.base import StableId
from video_demo.domain.evidence import SpeechSegment
from video_demo.domain.run import TimeRange
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.media.transcode import AudioSliceArtifact
from video_demo.speech.asr import (
    WindowRecognizerPort,
    build_cloud_asr_windows,
    project_cloud_asr_window,
    remove_adjacent_cloud_asr_duplicates,
)
from video_demo.speech.snapshots import (
    AsrSnapshotPayload,
    AsrWindowSnapshotPayload,
    asr_window_fingerprint,
)
from video_demo.speech.vad import VadResult
from video_demo.storage.snapshots import AsrWindowSnapshotStore
from video_demo.storage.workspace import safe_runtime_path


class VadPort(Protocol):
    def detect(self, audio: Path, *, duration_ms: int) -> VadResult: ...


class AudioSlicer(Protocol):
    def create(
        self,
        audio: Path,
        run_relative_root: Path,
        slice_id: str,
        time_range: TimeRange,
    ) -> Path: ...


@dataclass(frozen=True, slots=True)
class AsrComponents:
    vad: VadPort
    recognizer: WindowRecognizerPort
    slicer: AudioSlicer
    slice_namespace: StableId


AsrComponentFactory = Callable[[PreparedMedia, Callable[[], bool]], AsrComponents]


def transcript_shortcut(media: PreparedMedia) -> SpeechAnalysis | None:
    """处理无需模型的字幕优先和无音频路径。"""

    if media.subtitle is not None:
        return SpeechAnalysis(
            transcript_source="SUBTITLE",
            evidence=media.subtitle.cues,
            warnings=tuple(dict.fromkeys((*media.warnings, "TRANSCRIPT_SOURCE_SUBTITLE"))),
            boundary_candidates=tuple(
                SpeechBoundaryCandidate(cue.end_ms, "sentence_end", 1.0)
                for cue in media.subtitle.cues
                if 0 < cue.end_ms < media.source.duration_ms
            ),
        )
    if media.audio_path is None:
        return SpeechAnalysis(
            transcript_source="NONE",
            evidence=(),
            warnings=tuple(dict.fromkeys((*media.warnings, "NO_AUDIO_TRACK"))),
        )
    return None


def run_asr_stage(
    media: PreparedMedia,
    components: AsrComponents,
    *,
    window_store: AsrWindowSnapshotStore,
    asr_fingerprint: str,
    max_window_ms: int,
    overlap_ms: int,
) -> AsrSnapshotPayload:
    """严格串行识别 VAD 窗口，并在每个成功窗口后立即持久化。"""

    if media.audio_path is None:
        raise VideoDemoError(ErrorCode.SPEECH_AUDIO_INVALID, "云端 ASR 缺少音频")
    vad = components.vad.detect(media.audio_path, duration_ms=media.source.duration_ms)
    windows = build_cloud_asr_windows(
        vad.speech,
        max_window_ms=max_window_ms,
        overlap_ms=overlap_ms,
    )
    language_spans = []
    segments: list[SpeechSegment] = []
    warnings: list[str] = []
    prompt = cloud_asr_prompt(
        media.source.asset.config.hotwords,
        media.source.asset.config.core_context,
    )
    language_hint = _single_language_hint(media.source.asset.config.language_hints)
    for window in windows:
        fingerprint = asr_window_fingerprint(
            asr_fingerprint=asr_fingerprint,
            window=window,
        )
        cached = window_store.load(media.source.asset.run_relative_root, fingerprint)
        if cached is None:
            payload = _recognize_window(
                media,
                components,
                window_store,
                window,
                fingerprint,
                language_hint=language_hint,
                prompt=prompt,
            )
        else:
            payload = cached[0]
        language_spans.append(payload.language_span)
        segments.extend(payload.segments)
        warnings.extend(payload.warnings)
    ordered_spans = tuple(
        sorted(language_spans, key=lambda item: (item.start_ms, item.end_ms, item.evidence_id))
    )
    ordered_segments = tuple(
        sorted(segments, key=lambda item: (item.start_ms, item.end_ms, item.evidence_id))
    )
    deduplicated = remove_adjacent_cloud_asr_duplicates(ordered_segments)
    if windows and not deduplicated:
        warnings.append("ASR_NO_VALID_SEGMENTS")
    return AsrSnapshotPayload(
        language_spans=ordered_spans,
        segments=deduplicated,
        vad_warnings=vad.warnings,
        silence_boundaries_ms=vad.long_silence_boundaries_ms,
        language_change_boundaries_ms=tuple(
            current.start_ms
            for previous, current in pairwise(ordered_spans)
            if previous.language != current.language
        ),
        asr_warnings=tuple(dict.fromkeys(warnings)),
    )


def _recognize_window(
    media: PreparedMedia,
    components: AsrComponents,
    window_store: AsrWindowSnapshotStore,
    window: object,
    fingerprint: str,
    *,
    language_hint: str | None,
    prompt: str | None,
) -> AsrWindowSnapshotPayload:
    from video_demo.speech.asr import CloudAsrWindow

    if not isinstance(window, CloudAsrWindow):
        raise TypeError("云端 ASR 窗口类型非法")
    assert media.audio_path is not None
    audio_slice = components.slicer.create(
        media.audio_path,
        media.source.asset.run_relative_root,
        f"{components.slice_namespace}_{fingerprint[:24]}",
        window.upload_range,
    )
    try:
        result = components.recognizer.transcribe_window(
            audio_slice,
            language_hint=language_hint,
            prompt=prompt,
        )
        projection = project_cloud_asr_window(
            window,
            language=result.language,
            raw_segments=result.segments,
            warnings=result.warnings,
        )
        payload = AsrWindowSnapshotPayload(
            upload_range=window.upload_range,
            owned_range=window.owned_range,
            speech_interval=window.speech_interval,
            language_span=projection.language_span,
            segments=projection.segments,
            warnings=projection.warnings,
        )
        window_store.publish(media.source.asset.run_relative_root, fingerprint, payload)
        return payload
    finally:
        _discard_audio_slice(audio_slice)


def analysis_from_asr_snapshot(
    media: PreparedMedia,
    payload: AsrSnapshotPayload,
) -> SpeechAnalysis:
    warnings = tuple(
        dict.fromkeys((*media.warnings, *payload.vad_warnings, *payload.asr_warnings))
    )
    if not payload.language_spans:
        warnings = tuple(dict.fromkeys((*warnings, "NO_SPEECH_DETECTED")))
    return SpeechAnalysis(
        transcript_source="ASR",
        evidence=tuple(payload.segments),
        warnings=warnings,
        boundary_candidates=_boundary_candidates(
            media.source.duration_ms,
            silence=payload.silence_boundaries_ms,
            sentence_ends=tuple(item.end_ms for item in payload.segments),
            language_changes=payload.language_change_boundaries_ms,
        ),
    )


def cloud_asr_prompt(
    hotwords: tuple[str, ...],
    core_context: str | None,
) -> str | None:
    parts: list[str] = []
    if core_context:
        parts.append(core_context)
    if hotwords:
        parts.append(" ".join(hotwords))
    return "\n".join(parts) or None


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


def _single_language_hint(hints: Sequence[str]) -> str | None:
    return hints[0] if len(hints) == 1 and hints[0] != "und" else None


def _discard_audio_slice(path: Path) -> None:
    try:
        if path.is_symlink():
            raise VideoDemoError(
                ErrorCode.WORKSPACE_PATH_ESCAPE,
                "云端 ASR 音频切片不能是符号链接",
            )
        path.unlink(missing_ok=True)
    except OSError:
        raise VideoDemoError(
            ErrorCode.VIDEO_PROCESS_FAILED,
            "云端 ASR 音频切片清理失败",
        ) from None


def _boundary_candidates(
    duration_ms: int,
    *,
    silence: Sequence[int] = (),
    sentence_ends: Sequence[int] = (),
    language_changes: Sequence[int] = (),
) -> tuple[SpeechBoundaryCandidate, ...]:
    candidates = {
        (timestamp_ms, "silence", 1.0)
        for timestamp_ms in silence
        if 0 < timestamp_ms < duration_ms
    }
    candidates.update(
        (timestamp_ms, "sentence_end", 0.8)
        for timestamp_ms in sentence_ends
        if 0 < timestamp_ms < duration_ms
    )
    candidates.update(
        (timestamp_ms, "language_change", 1.0)
        for timestamp_ms in language_changes
        if 0 < timestamp_ms < duration_ms
    )
    return tuple(
        SpeechBoundaryCandidate(timestamp_ms, source, score)  # type: ignore[arg-type]
        for timestamp_ms, source, score in sorted(candidates)
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
