from __future__ import annotations

import hashlib
import math
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from video_demo.domain.run import TimeRange
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.storage.workspace import (
    atomic_replace,
    reject_symlink_components,
    safe_runtime_path,
)


@dataclass(frozen=True, slots=True)
class FrameCandidate:
    timestamp_ms: int
    sharpness: float
    black_ratio: float
    perceptual_hash: str
    relative_path: Path


@dataclass(frozen=True, slots=True)
class KeyframeSelection:
    frames: tuple[FrameCandidate, ...]


@dataclass(frozen=True, slots=True)
class WindowFrameCandidates:
    window: TimeRange
    candidates: tuple[FrameCandidate, ...]


class FrameExtractor(Protocol):
    def extract(
        self,
        proxy: Path,
        run_relative_root: Path,
        windows: Sequence[TimeRange],
        *,
        is_cancel_requested: Callable[[], bool],
        frame_tolerance_ms: int,
    ) -> tuple[WindowFrameCandidates, ...]: ...


class OpenCvFrameExtractor:
    """复用单个 VideoCapture，计算真实像素指标并安全写入 JPEG。"""

    def __init__(
        self,
        runtime_root: Path,
        *,
        module_loader: Callable[[], Any] | None = None,
        samples_per_window: int | None = None,
        max_frame_bytes: int = 20 * 1024 * 1024,
        jpeg_quality: int = 90,
    ) -> None:
        if samples_per_window is not None and samples_per_window < 1:
            raise ValueError("帧采样数必须大于 0")
        if max_frame_bytes < 1:
            raise ValueError("帧采样数与字节上限必须大于 0")
        self._runtime_root = runtime_root.expanduser().resolve(strict=False)
        self._module_loader = module_loader or _load_cv2
        self._samples_per_window = samples_per_window
        self._max_frame_bytes = max_frame_bytes
        self._jpeg_quality = jpeg_quality

    def extract(
        self,
        proxy: Path,
        run_relative_root: Path,
        windows: Sequence[TimeRange],
        *,
        is_cancel_requested: Callable[[], bool],
        frame_tolerance_ms: int = 100,
        input_fd: int | None = None,
        write_jpeg: Callable[[Path, bytes, int], None] | None = None,
    ) -> tuple[WindowFrameCandidates, ...]:
        if not 0 <= frame_tolerance_ms <= 100:
            raise ValueError("帧时间容差必须在 0 到 100 毫秒")
        try:
            cv2 = self._module_loader()
        except (ModuleNotFoundError, ImportError, OSError):
            raise VideoDemoError(
                ErrorCode.VISUAL_DEPENDENCY_UNAVAILABLE,
                "视觉分析可选依赖不可用",
            ) from None
        try:
            capture_path = f"/dev/fd/{input_fd}" if input_fd is not None else str(proxy)
            capture = cv2.VideoCapture(capture_path)
            try:
                if not capture.isOpened():
                    raise VideoDemoError(
                        ErrorCode.VISUAL_MEDIA_INVALID,
                        "OpenCV 无法打开代理视频",
                    )
                groups: list[WindowFrameCandidates] = []
                seen_actual_timestamps: set[int] = set()
                for window in windows:
                    self._check_cancelled(is_cancel_requested)
                    frames: list[FrameCandidate] = []
                    for timestamp_ms in _sample_timestamps(window, self._samples_per_window):
                        self._check_cancelled(is_cancel_requested)
                        if not capture.set(cv2.CAP_PROP_POS_MSEC, float(timestamp_ms)):
                            continue
                        ok, pixels = capture.read()
                        if not ok or pixels is None:
                            continue
                        actual_ms_raw = float(capture.get(cv2.CAP_PROP_POS_MSEC))
                        if not math.isfinite(actual_ms_raw):
                            continue
                        actual_timestamp_ms = round(actual_ms_raw)
                        if (
                            abs(actual_timestamp_ms - timestamp_ms) > frame_tolerance_ms
                            or not window.start_ms <= actual_timestamp_ms < window.end_ms
                            or actual_timestamp_ms in seen_actual_timestamps
                        ):
                            continue
                        seen_actual_timestamps.add(actual_timestamp_ms)
                        gray = cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY)
                        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                        black_ratio = float((gray <= 16).mean())
                        perceptual_hash = _perceptual_hash(cv2, gray)
                        relative = (
                            run_relative_root
                            / "visual"
                            / "keyframes"
                            / f"frame_{actual_timestamp_ms:012d}.jpg"
                        )
                        self._write_jpeg(
                            cv2,
                            pixels,
                            relative,
                            writer=write_jpeg,
                        )
                        frames.append(
                            FrameCandidate(
                                timestamp_ms=actual_timestamp_ms,
                                sharpness=sharpness,
                                black_ratio=black_ratio,
                                perceptual_hash=perceptual_hash,
                                relative_path=relative,
                            ),
                        )
                    groups.append(WindowFrameCandidates(window=window, candidates=tuple(frames)))
                return tuple(groups)
            finally:
                capture.release()
        except VideoDemoError:
            raise
        except Exception:
            raise VideoDemoError(
                ErrorCode.VISUAL_MEDIA_INVALID,
                "OpenCV 帧解码失败",
            ) from None

    def _write_jpeg(
        self,
        cv2: Any,
        pixels: object,
        relative_path: Path,
        *,
        writer: Callable[[Path, bytes, int], None] | None = None,
    ) -> None:
        ok, encoded = cv2.imencode(
            ".jpg",
            pixels,
            [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality],
        )
        if not ok:
            raise VideoDemoError(ErrorCode.VISUAL_RESULT_INVALID, "OpenCV JPEG 编码失败")
        payload = bytes(encoded.tobytes())
        if (
            not payload
            or len(payload) > self._max_frame_bytes
            or not _is_jpeg(payload)
        ):
            raise VideoDemoError(ErrorCode.VISUAL_RESULT_INVALID, "关键帧 JPEG 产物非法")
        if writer is not None:
            writer(relative_path, payload, self._max_frame_bytes)
            return
        destination = safe_runtime_path(self._runtime_root, relative_path)
        reject_symlink_components(
            self._runtime_root,
            destination,
            message="关键帧输出路径不能包含符号链接",
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        reject_symlink_components(
            self._runtime_root,
            destination,
            message="关键帧输出路径不能包含符号链接",
        )
        temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.part")
        try:
            temporary.write_bytes(payload)
            reject_symlink_components(
                self._runtime_root,
                destination,
                message="关键帧输出路径不能包含符号链接",
            )
            atomic_replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        if (
            not destination.is_file()
            or destination.stat().st_size != len(payload)
            or _sha256_file(destination) != hashlib.sha256(payload).hexdigest()
        ):
            raise VideoDemoError(ErrorCode.VISUAL_RESULT_INVALID, "关键帧写入校验失败")

    @staticmethod
    def _check_cancelled(is_cancel_requested: Callable[[], bool]) -> None:
        if is_cancel_requested():
            raise VideoDemoError(ErrorCode.JOB_CANCELLED, "任务已请求取消")


class KeyframeSelector:
    def __init__(
        self,
        *,
        maximum_black_ratio: float = 0.95,
        max_hash_distance_for_duplicate: int = 4,
    ) -> None:
        self._maximum_black_ratio = maximum_black_ratio
        self._max_hash_distance_for_duplicate = max_hash_distance_for_duplicate

    def select(
        self,
        window: TimeRange,
        candidates: Sequence[FrameCandidate],
    ) -> KeyframeSelection:
        for candidate in candidates:
            if not window.start_ms <= candidate.timestamp_ms < window.end_ms:
                raise ValueError("候选帧超出窗口")
        eligible = [
            candidate
            for candidate in candidates
            if candidate.black_ratio <= self._maximum_black_ratio
        ]
        if not eligible:
            return KeyframeSelection(frames=())

        if window.duration_ms <= 10_000:
            target_count = 1
        elif window.duration_ms <= 20_000:
            target_count = 2
        else:
            target_count = 3
        unique = self._deduplicate(eligible)
        bucket_width = window.duration_ms / target_count
        selected: list[FrameCandidate] = []
        for bucket_index in range(target_count):
            bucket_start = window.start_ms + bucket_index * bucket_width
            bucket_end = window.start_ms + (bucket_index + 1) * bucket_width
            bucket = [
                candidate
                for candidate in unique
                if bucket_start <= candidate.timestamp_ms < bucket_end
            ]
            if bucket:
                selected.append(max(bucket, key=lambda item: (item.sharpness, item.timestamp_ms)))
        if len(selected) < target_count:
            remaining = [candidate for candidate in unique if candidate not in selected]
            remaining.sort(key=lambda item: (-item.sharpness, item.timestamp_ms))
            selected.extend(remaining[: target_count - len(selected)])
        selected.sort(key=lambda item: item.timestamp_ms)
        return KeyframeSelection(frames=tuple(selected))

    def _deduplicate(self, candidates: Sequence[FrameCandidate]) -> list[FrameCandidate]:
        ranked = sorted(candidates, key=lambda item: (-item.sharpness, item.timestamp_ms))
        kept: list[FrameCandidate] = []
        for candidate in ranked:
            if any(
                _hamming_distance(candidate.perceptual_hash, existing.perceptual_hash)
                <= self._max_hash_distance_for_duplicate
                for existing in kept
            ):
                continue
            kept.append(candidate)
        return kept


def _hamming_distance(left: str, right: str) -> int:
    try:
        return (int(left, 16) ^ int(right, 16)).bit_count()
    except ValueError as error:
        raise ValueError("感知哈希必须是十六进制字符串") from error


def _load_cv2() -> Any:
    import cv2

    return cv2


def _sample_timestamps(window: TimeRange, count: int | None = None) -> tuple[int, ...]:
    if count is None:
        if window.duration_ms <= 8_000:
            return (window.start_ms + window.duration_ms // 2,)
        return (
            window.start_ms + window.duration_ms // 3,
            window.start_ms + (window.duration_ms * 2) // 3,
        )
    actual_count = min(count, max(1, window.duration_ms))
    return tuple(
        min(
            window.end_ms - 1,
            window.start_ms + round((index + 0.5) * window.duration_ms / actual_count),
        )
        for index in range(actual_count)
    )


def _perceptual_hash(cv2: Any, gray: object) -> str:
    resized = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA)
    values = [int(value) for value in resized.reshape(64)]
    average = sum(values) / len(values)
    bits = 0
    for value in values:
        bits = (bits << 1) | int(value >= average)
    return f"{bits:016x}"


def _is_jpeg(payload: bytes) -> bool:
    return payload.startswith(b"\xff\xd8\xff") and payload.endswith(b"\xff\xd9")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
