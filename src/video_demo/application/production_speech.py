from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
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
from video_demo.domain.run import TimeRange
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.media.transcode import AudioSliceArtifact
from video_demo.speech.asr import (
    WindowRecognizerPort,
    remove_adjacent_cloud_asr_duplicates,
)
from video_demo.speech.snapshots import (
    AsrSnapshotPayload,
    AsrWindowSnapshotPayload,
    asr_window_fingerprint,
)
from video_demo.speech.video_asr import (
    VIDEO_ASR_CHUNK_DURATION_MS,
    VIDEO_ASR_CONCURRENCY,
    FixedAsrWindow,
    build_fixed_asr_windows,
    project_fixed_asr_window,
)
from video_demo.storage.snapshots import AsrWindowSnapshotStore
from video_demo.storage.workspace import safe_runtime_path

_LOGGER = logging.getLogger(__name__)


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
    recognizer: WindowRecognizerPort
    slicer: AudioSlicer
    slice_namespace: StableId
    is_cancel_requested: Callable[[], bool] = lambda: False


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
    chunk_duration_ms: int = VIDEO_ASR_CHUNK_DURATION_MS,
    concurrency: int = VIDEO_ASR_CONCURRENCY,
    max_upload_bytes: int = 25 * 1024 * 1024,
) -> AsrSnapshotPayload:
    """固定十分钟分块并发识别；所有块成功后才发布完整 ASR 结果。"""

    if media.audio_path is None:
        raise VideoDemoError(ErrorCode.SPEECH_AUDIO_INVALID, "云端 ASR 缺少音频")
    if concurrency != VIDEO_ASR_CONCURRENCY:
        raise ValueError("视频 ASR 并发数必须固定为 1")
    windows = build_fixed_asr_windows(
        media.source.duration_ms,
        chunk_duration_ms=chunk_duration_ms,
    )
    _LOGGER.info(
        "视频 ASR 固定分块完成 chunks=%d chunk_duration_ms=%d concurrency=%d",
        len(windows),
        chunk_duration_ms,
        concurrency,
    )
    prompt = cloud_asr_prompt(
        media.source.asset.config.hotwords,
        media.source.asset.config.core_context,
    )
    language_hint = _single_language_hint(media.source.asset.config.language_hints)
    results = _recognize_windows_concurrently(
        media,
        components,
        windows,
        window_store,
        asr_fingerprint,
        language_hint=language_hint,
        prompt=prompt,
        max_upload_bytes=max_upload_bytes,
    )
    language_spans = tuple(payload.language_span for payload in results)
    segments = tuple(segment for payload in results for segment in payload.segments)
    warnings = tuple(
        warning for payload in results for warning in payload.warnings
    )
    deduplicated = remove_adjacent_cloud_asr_duplicates(
        tuple(sorted(segments, key=lambda item: (item.start_ms, item.end_ms, item.evidence_id)))
    )
    if windows and not deduplicated:
        warnings = (*warnings, "ASR_NO_VALID_SEGMENTS")
    _LOGGER.info("视频 ASR 全部块完成 chunks=%d", len(windows))
    return AsrSnapshotPayload(
        language_spans=tuple(
            sorted(language_spans, key=lambda item: (item.start_ms, item.end_ms, item.evidence_id))
        ),
        segments=deduplicated,
        language_change_boundaries_ms=tuple(
            current.start_ms
            for previous, current in pairwise(
                sorted(
                    language_spans,
                    key=lambda item: (item.start_ms, item.end_ms, item.evidence_id),
                )
            )
            if previous.language != current.language
        ),
        asr_warnings=tuple(dict.fromkeys(warnings)),
    )


def _recognize_windows_concurrently(
    media: PreparedMedia,
    components: AsrComponents,
    windows: tuple[FixedAsrWindow, ...],
    window_store: AsrWindowSnapshotStore,
    asr_fingerprint: str,
    *,
    language_hint: str | None,
    prompt: str | None,
    max_upload_bytes: int,
) -> tuple[AsrWindowSnapshotPayload, ...]:
    results: dict[int, AsrWindowSnapshotPayload] = {}
    futures: dict[Future[AsrWindowSnapshotPayload], FixedAsrWindow] = {}
    first_error: Exception | None = None
    with ThreadPoolExecutor(
        max_workers=VIDEO_ASR_CONCURRENCY,
        thread_name_prefix="video-asr",
    ) as executor:
        for window in windows:
            futures[executor.submit(
                _recognize_window,
                media,
                components,
                window_store,
                window,
                asr_fingerprint,
                language_hint=language_hint,
                prompt=prompt,
                max_upload_bytes=max_upload_bytes,
            )] = window
        for future in as_completed(futures):
            window = futures[future]
            try:
                results[window.chunk_index] = future.result()
            except Exception as error:
                first_error = error
                _LOGGER.warning(
                    "视频 ASR 块失败 chunk=%d/%d code=%s",
                    window.chunk_index + 1,
                    len(windows),
                    getattr(error, "code", type(error).__name__),
                )
                for pending in futures:
                    if pending is not future:
                        pending.cancel()
                break
    if first_error is not None:
        raise first_error
    if len(results) != len(windows):
        raise VideoDemoError(ErrorCode.SPEECH_SUBPROCESS_RESPONSE_INVALID, "视频 ASR 块结果不完整")
    return tuple(results[index] for index in range(len(windows)))


def _recognize_window(
    media: PreparedMedia,
    components: AsrComponents,
    window_store: AsrWindowSnapshotStore,
    window: FixedAsrWindow,
    fingerprint: str,
    *,
    language_hint: str | None,
    prompt: str | None,
    max_upload_bytes: int,
) -> AsrWindowSnapshotPayload:
    assert media.audio_path is not None
    if components.is_cancel_requested():
        raise VideoDemoError(ErrorCode.JOB_CANCELLED, "任务已请求取消")
    window_fingerprint = asr_window_fingerprint(
        asr_fingerprint=fingerprint,
        window=window,
    )
    cached = window_store.load(
        media.source.asset.run_relative_root,
        window_fingerprint,
    )
    if cached is not None:
        _LOGGER.info(
            "视频 ASR 块完成 chunk=%d cached=true segments=%d",
            window.chunk_index + 1,
            len(cached[0].segments),
        )
        return cached[0]
    _LOGGER.info(
        "视频 ASR 块开始 chunk=%d range=%d-%dms",
        window.chunk_index + 1,
        window.upload_range.start_ms,
        window.upload_range.end_ms,
    )
    audio_slice = components.slicer.create(
        media.audio_path,
        media.source.asset.run_relative_root,
        f"{components.slice_namespace}_{window_fingerprint[:24]}",
        window.upload_range,
    )
    try:
        if audio_slice.stat().st_size > max_upload_bytes:
            raise VideoDemoError(ErrorCode.VIDEO_OUTPUT_TOO_LARGE, "ASR 音频切片超过大小限制")
        result = components.recognizer.transcribe_window(
            audio_slice,
            language_hint=language_hint,
            prompt=prompt,
        )
        projection = project_fixed_asr_window(
            window,
            language=result.language,
            raw_segments=result.segments,
            warnings=result.warnings,
        )
        payload = AsrWindowSnapshotPayload(
            chunk_index=window.chunk_index,
            upload_range=window.upload_range,
            owned_range=window.owned_range,
            language_span=projection.language_span,
            segments=projection.segments,
            warnings=projection.warnings,
        )
        window_store.publish(
            media.source.asset.run_relative_root,
            window_fingerprint,
            payload,
        )
        _LOGGER.info(
            "视频 ASR 块完成 chunk=%d cached=false segments=%d",
            window.chunk_index + 1,
            len(payload.segments),
        )
        return payload
    finally:
        _discard_audio_slice(audio_slice)


def analysis_from_asr_snapshot(
    media: PreparedMedia,
    payload: AsrSnapshotPayload,
) -> SpeechAnalysis:
    warnings = tuple(dict.fromkeys((*media.warnings, *payload.asr_warnings)))
    if not payload.language_spans:
        warnings = tuple(dict.fromkeys((*warnings, "NO_SPEECH_DETECTED")))
    return SpeechAnalysis(
        transcript_source="ASR",
        evidence=tuple(payload.segments),
        warnings=warnings,
        boundary_candidates=_boundary_candidates(
            media.source.duration_ms,
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
