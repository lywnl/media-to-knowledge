from __future__ import annotations

import hashlib
import math
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol

from video_demo.domain.run import TimeRange
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.storage.workspace import (
    atomic_replace,
    reject_symlink_components,
    safe_runtime_path,
)

if TYPE_CHECKING:
    from video_demo.visual.candidate_artifacts import CandidateArtifactSession


@dataclass(frozen=True, slots=True)
class FrameCandidate:
    timestamp_ms: int
    sharpness: float
    black_ratio: float
    perceptual_hash: str
    relative_path: Path
    created_by_call: bool = False


@dataclass(frozen=True, slots=True)
class KeyframeSelection:
    frames: tuple[FrameCandidate, ...]


@dataclass(frozen=True, slots=True)
class WindowFrameCandidates:
    window: TimeRange
    candidates: tuple[FrameCandidate, ...]


FrameSampleStatus = Literal[
    "SUCCEEDED",
    "QUALITY_REJECTED",
    "SEEK_FAILED",
    "DECODE_FAILED",
    "INVALID_TIMESTAMP",
    "OUT_OF_TOLERANCE",
]
FrameArtifactStatus = Literal["PUBLISHED", "BUDGET_REJECTED"]
FrameAdmissionTier = Literal[
    "SEMANTIC_PRIMARY",
    "BASE_PRIMARY",
    "SEMANTIC_SUPPLEMENT",
    "BASE_SUPPLEMENT",
]
_FRAME_SAMPLE_STATUSES = frozenset(
    {
        "SUCCEEDED",
        "QUALITY_REJECTED",
        "SEEK_FAILED",
        "DECODE_FAILED",
        "INVALID_TIMESTAMP",
        "OUT_OF_TOLERANCE",
    },
)
_FRAME_ADMISSION_TIERS: tuple[FrameAdmissionTier, ...] = (
    "SEMANTIC_PRIMARY",
    "BASE_PRIMARY",
    "SEMANTIC_SUPPLEMENT",
    "BASE_SUPPLEMENT",
)
_FRAME_ADMISSION_RANK = {
    tier: rank for rank, tier in enumerate(_FRAME_ADMISSION_TIERS)
}

_MAX_SEQUENTIAL_READS_PER_SAMPLE = 256
_MAXIMUM_BLACK_RATIO = 0.95


@dataclass(frozen=True, slots=True)
class FrameSample:
    sample_id: str
    timestamp_ms: int
    admission_tier: FrameAdmissionTier = "BASE_SUPPLEMENT"

    def __post_init__(self) -> None:
        if (
            not self.sample_id
            or self.timestamp_ms < 0
            or self.admission_tier not in _FRAME_ADMISSION_RANK
        ):
            raise ValueError("精确采样标识不能为空且时间戳不得为负数")


@dataclass(frozen=True, slots=True)
class ExactFrameSampleResult:
    sample_id: str
    requested_timestamp_ms: int
    status: FrameSampleStatus
    candidate: FrameCandidate | None = None
    artifact_status: FrameArtifactStatus | None = None

    def __post_init__(self) -> None:
        if self.status not in _FRAME_SAMPLE_STATUSES:
            raise ValueError("精确采样状态非法")
        if self.status != "SUCCEEDED" and (
            self.candidate is not None or self.artifact_status is not None
        ):
            raise ValueError("精确采样状态与候选帧不一致")
        if self.status == "SUCCEEDED" and self.artifact_status == "PUBLISHED":
            if self.candidate is None:
                raise ValueError("已发布的精确采样必须包含候选帧")
        elif self.status == "SUCCEEDED" and self.artifact_status == "BUDGET_REJECTED":
            if self.candidate is not None:
                raise ValueError("预算拒绝的精确采样不得包含候选帧")
        elif self.status == "SUCCEEDED":
            if self.candidate is None:
                raise ValueError("成功的精确采样必须说明制品状态")
            object.__setattr__(self, "artifact_status", "PUBLISHED")


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

    def extract_samples(
        self,
        proxy: Path,
        run_relative_root: Path,
        samples: Sequence[FrameSample],
        *,
        is_cancel_requested: Callable[[], bool],
        frame_tolerance_ms: int,
        artifact_session: CandidateArtifactSession | None = None,
    ) -> tuple[ExactFrameSampleResult, ...]: ...


class OpenCvFrameExtractor:
    """复用单个 VideoCapture，计算真实像素指标并安全写入 JPEG。"""

    def __init__(
        self,
        runtime_root: Path,
        *,
        module_loader: Callable[[], Any] | None = None,
        samples_per_window: int = 6,
        max_frame_bytes: int = 20 * 1024 * 1024,
        jpeg_quality: int = 90,
    ) -> None:
        if samples_per_window < 1 or max_frame_bytes < 1:
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
                        "OpenCV 无法打开视觉输入",
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

    def extract_samples(
        self,
        proxy: Path,
        run_relative_root: Path,
        samples: Sequence[FrameSample],
        *,
        is_cancel_requested: Callable[[], bool],
        frame_tolerance_ms: int,
        input_fd: int | None = None,
        artifact_session: CandidateArtifactSession | None = None,
    ) -> tuple[ExactFrameSampleResult, ...]:
        """按全 Run 精确计划抽帧，并把同一时间的单次解码回绑到全部采样 ID。"""

        if not 0 <= frame_tolerance_ms <= 100:
            raise ValueError("帧时间容差必须在 0 到 100 毫秒")
        ordered_input = tuple(samples)
        sample_ids = tuple(sample.sample_id for sample in ordered_input)
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError("精确采样标识不得重复")
        if not ordered_input:
            return ()
        cv2 = self._load_module()
        grouped = _group_samples_by_admission(ordered_input)
        capture_path = f"/dev/fd/{input_fd}" if input_fd is not None else str(proxy)
        capture = cv2.VideoCapture(capture_path)
        try:
            if not capture.isOpened():
                raise VideoDemoError(
                    ErrorCode.VISUAL_MEDIA_INVALID,
                    "OpenCV 无法打开视觉输入",
                )
            by_sample_id = self._extract_grouped_samples(
                cv2,
                capture,
                run_relative_root,
                grouped,
                frame_tolerance_ms=frame_tolerance_ms,
                is_cancel_requested=is_cancel_requested,
                artifact_session=artifact_session,
            )
            return tuple(by_sample_id[sample.sample_id] for sample in ordered_input)
        except VideoDemoError:
            raise
        except Exception:
            raise VideoDemoError(
                ErrorCode.VISUAL_MEDIA_INVALID,
                "OpenCV 帧解码失败",
            ) from None
        finally:
            capture.release()

    def _load_module(self) -> Any:
        try:
            return self._module_loader()
        except (ModuleNotFoundError, ImportError, OSError):
            raise VideoDemoError(
                ErrorCode.VISUAL_DEPENDENCY_UNAVAILABLE,
                "视觉分析可选依赖不可用",
            ) from None

    def _extract_grouped_samples(
        self,
        cv2: Any,
        capture: Any,
        run_relative_root: Path,
        grouped: tuple[tuple[FrameAdmissionTier, int, tuple[FrameSample, ...]], ...],
        *,
        frame_tolerance_ms: int,
        is_cancel_requested: Callable[[], bool],
        artifact_session: CandidateArtifactSession | None,
    ) -> dict[str, ExactFrameSampleResult]:
        results: dict[str, ExactFrameSampleResult] = {}
        last_actual_ms: int | None = None
        lookahead: tuple[object, int] | None = None
        previous_tier: FrameAdmissionTier | None = None
        for admission_tier, requested_ms, bound_samples in grouped:
            self._check_cancelled(is_cancel_requested)
            should_seek = (
                last_actual_ms is None
                or admission_tier != previous_tier
                or requested_ms < last_actual_ms
                or requested_ms - last_actual_ms > 1_000
            )
            status, candidate, artifact_status, actual_ms, lookahead = self._extract_one_sample(
                cv2,
                capture,
                run_relative_root,
                requested_ms,
                should_seek=should_seek,
                frame_tolerance_ms=frame_tolerance_ms,
                lookahead=None if should_seek else lookahead,
                artifact_session=artifact_session,
            )
            if actual_ms is not None:
                last_actual_ms = actual_ms
            previous_tier = admission_tier
            for sample in bound_samples:
                results[sample.sample_id] = ExactFrameSampleResult(
                    sample_id=sample.sample_id,
                    requested_timestamp_ms=requested_ms,
                    status=status,
                    candidate=candidate,
                    artifact_status=artifact_status,
                )
        return results

    def _extract_one_sample(
        self,
        cv2: Any,
        capture: Any,
        run_relative_root: Path,
        requested_ms: int,
        *,
        should_seek: bool,
        frame_tolerance_ms: int,
        lookahead: tuple[object, int] | None,
        artifact_session: CandidateArtifactSession | None,
    ) -> tuple[
        FrameSampleStatus,
        FrameCandidate | None,
        FrameArtifactStatus | None,
        int | None,
        tuple[object, int] | None,
    ]:
        if should_seek and not capture.set(cv2.CAP_PROP_POS_MSEC, float(requested_ms)):
            return "SEEK_FAILED", None, None, None, None
        last_actual_ms: int | None = None
        for _attempt in range(_MAX_SEQUENTIAL_READS_PER_SAMPLE):
            if lookahead is not None:
                pixels, actual_ms = lookahead
                lookahead = None
            else:
                try:
                    ok, pixels = capture.read()
                except Exception:
                    return "DECODE_FAILED", None, None, last_actual_ms, None
                if not ok or pixels is None:
                    return "DECODE_FAILED", None, None, last_actual_ms, None
                try:
                    raw_timestamp = float(capture.get(cv2.CAP_PROP_POS_MSEC))
                except (TypeError, ValueError, OverflowError):
                    return "INVALID_TIMESTAMP", None, None, last_actual_ms, None
                if not math.isfinite(raw_timestamp) or raw_timestamp < 0:
                    return "INVALID_TIMESTAMP", None, None, last_actual_ms, None
                actual_ms = round(raw_timestamp)
            if last_actual_ms is not None and actual_ms <= last_actual_ms:
                return "INVALID_TIMESTAMP", None, None, actual_ms, None
            last_actual_ms = actual_ms
            if abs(actual_ms - requested_ms) <= frame_tolerance_ms:
                sample_status, candidate, artifact_status = self._candidate_from_pixels(
                    cv2,
                    pixels,
                    run_relative_root,
                    actual_ms,
                    artifact_session,
                )
                return sample_status, candidate, artifact_status, actual_ms, None
            if actual_ms > requested_ms + frame_tolerance_ms:
                return "OUT_OF_TOLERANCE", None, None, actual_ms, (pixels, actual_ms)
        return "OUT_OF_TOLERANCE", None, None, last_actual_ms, None

    def _candidate_from_pixels(
        self,
        cv2: Any,
        pixels: object,
        run_relative_root: Path,
        actual_timestamp_ms: int,
        artifact_session: CandidateArtifactSession | None,
    ) -> tuple[FrameSampleStatus, FrameCandidate | None, FrameArtifactStatus | None]:
        gray = cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        black_ratio = float((gray <= 16).mean())
        if (
            not math.isfinite(sharpness)
            or sharpness < 0
            or not math.isfinite(black_ratio)
            or not 0.0 <= black_ratio <= 1.0
        ):
            raise VideoDemoError(ErrorCode.VISUAL_RESULT_INVALID, "候选帧像素指标非法")
        if black_ratio > _MAXIMUM_BLACK_RATIO:
            return "QUALITY_REJECTED", None, None
        payload = self._encode_jpeg(cv2, pixels)
        digest = hashlib.sha256(payload).hexdigest()
        relative_path = run_relative_root / "visual" / "candidates" / f"{digest}.jpg"
        if artifact_session is not None:
            artifact_session.prepare_run(run_relative_root)
            publication = artifact_session.publish_jpeg(relative_path, payload, digest)
            if publication.status == "BUDGET_REJECTED":
                return "SUCCEEDED", None, "BUDGET_REJECTED"
            created_by_call = publication.created_by_call
        else:
            raise VideoDemoError(
                ErrorCode.INVALID_CONFIGURATION,
                "精确抽帧必须提供候选制品会话",
            )
        return (
            "SUCCEEDED",
            FrameCandidate(
                timestamp_ms=actual_timestamp_ms,
                sharpness=sharpness,
                black_ratio=black_ratio,
                perceptual_hash=_perceptual_hash(cv2, gray),
                relative_path=relative_path,
                created_by_call=created_by_call,
            ),
            "PUBLISHED",
        )

    def _encode_jpeg(self, cv2: Any, pixels: object) -> bytes:
        ok, encoded = cv2.imencode(
            ".jpg",
            pixels,
            [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality],
        )
        if not ok:
            raise VideoDemoError(ErrorCode.VISUAL_RESULT_INVALID, "OpenCV JPEG 编码失败")
        payload = bytes(encoded.tobytes())
        if not payload or len(payload) > self._max_frame_bytes or not _is_jpeg(payload):
            raise VideoDemoError(ErrorCode.VISUAL_RESULT_INVALID, "候选帧 JPEG 产物非法")
        return payload

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
        if not payload or len(payload) > self._max_frame_bytes or not _is_jpeg(payload):
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


def _sample_timestamps(window: TimeRange, count: int) -> tuple[int, ...]:
    actual_count = min(count, max(1, window.duration_ms))
    return tuple(
        min(
            window.end_ms - 1,
            window.start_ms + round((index + 0.5) * window.duration_ms / actual_count),
        )
        for index in range(actual_count)
    )


def _group_samples_by_admission(
    samples: Sequence[FrameSample],
) -> tuple[tuple[FrameAdmissionTier, int, tuple[FrameSample, ...]], ...]:
    grouped: dict[int, list[FrameSample]] = {}
    for sample in samples:
        grouped.setdefault(sample.timestamp_ms, []).append(sample)
    normalized = tuple(
        (
            min(
                (sample.admission_tier for sample in bound_samples),
                key=_FRAME_ADMISSION_RANK.__getitem__,
            ),
            timestamp_ms,
            tuple(bound_samples),
        )
        for timestamp_ms, bound_samples in grouped.items()
    )
    return tuple(
        sorted(
            normalized,
            key=lambda item: (_FRAME_ADMISSION_RANK[item[0]], item[1], item[2][0].sample_id),
        ),
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
