from __future__ import annotations

import hashlib
import logging
import os
import tempfile
from collections.abc import Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from video_demo.domain.base import Sha256
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.media.process import ProcessResult, SafeProcessRunner
from video_demo.storage.workspace import reject_symlink_components, safe_runtime_path
from video_demo.visual.candidate_artifacts import CandidateArtifactSession

_LOGGER = logging.getLogger(__name__)


class ProcessRunner(Protocol):
    def run(
        self,
        args: Sequence[str],
        *,
        timeout_seconds: int,
        output_paths: tuple[Path, ...] = (),
    ) -> ProcessResult: ...


FrameAdmissionTier = Literal[
    "SEMANTIC_PRIMARY",
    "BASE_PRIMARY",
    "SEMANTIC_SUPPLEMENT",
    "BASE_SUPPLEMENT",
]
FrameSampleStatus = Literal["SUCCEEDED", "SEEK_FAILED", "DECODE_FAILED", "INVALID_TIMESTAMP"]


@dataclass(frozen=True, slots=True)
class FrameSample:
    sample_id: str
    timestamp_ms: int
    admission_tier: FrameAdmissionTier = "BASE_SUPPLEMENT"

    def __post_init__(self) -> None:
        if not self.sample_id or self.timestamp_ms < 0:
            raise ValueError("精确采样标识不能为空且时间戳不得为负数")


@dataclass(frozen=True, slots=True)
class FrameCandidate:
    timestamp_ms: int
    relative_path: Path
    sha256: Sha256
    size_bytes: int
    target_ids: tuple[str, ...] = ()
    created_by_call: bool = False


@dataclass(frozen=True, slots=True)
class ExactFrameSampleResult:
    sample_id: str
    requested_timestamp_ms: int
    status: FrameSampleStatus
    candidate: FrameCandidate | None = None
    artifact_status: Literal["PUBLISHED", "BUDGET_REJECTED"] | None = None

    def __post_init__(self) -> None:
        if (
            self.status == "SUCCEEDED"
            and self.candidate is None
            and self.artifact_status != "BUDGET_REJECTED"
        ):
            raise ValueError("成功采样必须包含候选帧或预算拒绝状态")
        if self.status != "SUCCEEDED" and (
            self.candidate is not None or self.artifact_status is not None
        ):
            raise ValueError("失败采样不得包含候选帧")


class FrameExtractor(Protocol):
    def extract_samples(
        self,
        source: Path,
        run_relative_root: Path,
        samples: Sequence[FrameSample],
        *,
        is_cancel_requested: Callable[[], bool],
        artifact_session: CandidateArtifactSession,
    ) -> tuple[ExactFrameSampleResult, ...]: ...


class FFmpegFrameExtractor:
    """使用 FFmpeg 按程序指定时间点抽取单帧 JPEG。"""

    def __init__(
        self,
        executable: Path,
        runner: ProcessRunner,
        runtime_root: Path,
        *,
        max_frame_bytes: int,
        frame_width: int = 1280,
        jpeg_quality: int = 5,
        timeout_seconds: int = 300,
    ) -> None:
        if max_frame_bytes < 1 or frame_width < 1 or not 1 <= jpeg_quality <= 100:
            raise ValueError("视觉抽帧参数非法")
        self._executable = executable
        self._runner = runner
        self._runtime_root = runtime_root.expanduser().resolve(strict=False)
        self._max_frame_bytes = max_frame_bytes
        self._frame_width = frame_width
        self._jpeg_quality = jpeg_quality
        self._timeout_seconds = timeout_seconds

    @classmethod
    def from_path(
        cls,
        executable: Path,
        runtime_root: Path,
        *,
        workspace_root: Path,
        max_frame_bytes: int,
        frame_width: int = 1280,
        jpeg_quality: int = 5,
        timeout_seconds: int = 300,
    ) -> FFmpegFrameExtractor:
        runner = SafeProcessRunner(
            max_output_bytes=2 * 1024 * 1024,
            workspace_root=workspace_root,
        )
        return cls(
            executable,
            runner,
            runtime_root,
            max_frame_bytes=max_frame_bytes,
            frame_width=frame_width,
            jpeg_quality=jpeg_quality,
            timeout_seconds=timeout_seconds,
        )

    def extract_samples(
        self,
        source: Path,
        run_relative_root: Path,
        samples: Sequence[FrameSample],
        *,
        is_cancel_requested: Callable[[], bool],
        artifact_session: CandidateArtifactSession,
    ) -> tuple[ExactFrameSampleResult, ...]:
        allowed_root = safe_runtime_path(self._runtime_root, run_relative_root)
        source_path = reject_symlink_components(
            self._runtime_root,
            source,
            message="视觉输入必须位于当前 Run 且不能包含符号链接",
        )
        if not source_path.is_relative_to(allowed_root) or not source_path.is_file():
            raise VideoDemoError(ErrorCode.WORKSPACE_PATH_ESCAPE, "视觉输入必须位于当前 Run")
        artifact_session.prepare_run(run_relative_root)
        results: list[ExactFrameSampleResult] = []
        for sample in samples:
            if is_cancel_requested():
                raise VideoDemoError(ErrorCode.JOB_CANCELLED, "视觉抽帧已取消")
            if sample.timestamp_ms < 0:
                results.append(
                    ExactFrameSampleResult(
                        sample.sample_id, sample.timestamp_ms, "INVALID_TIMESTAMP"
                    )
                )
                continue
            _LOGGER.info("视觉关键帧抽取开始 timestamp_ms=%s", sample.timestamp_ms)
            temporary = self._temporary_path(allowed_root)
            try:
                result = self._extract_one(
                    source_path, temporary, sample.timestamp_ms, is_cancel_requested
                )
                if result is None:
                    results.append(
                        ExactFrameSampleResult(
                            sample.sample_id, sample.timestamp_ms, "DECODE_FAILED"
                        )
                    )
                    _LOGGER.warning(
                        "视觉关键帧抽取失败 timestamp_ms=%s code=DECODE_FAILED", sample.timestamp_ms
                    )
                    continue
                payload = temporary.read_bytes()
                digest = hashlib.sha256(payload).hexdigest()
                publication = artifact_session.publish_jpeg(
                    run_relative_root / "visual" / "candidates" / f"{digest}.jpg",
                    payload,
                    digest,
                )
                candidate = None
                if publication.status == "PUBLISHED":
                    candidate = FrameCandidate(
                        timestamp_ms=sample.timestamp_ms,
                        relative_path=Path("visual/candidates") / f"{digest}.jpg",
                        sha256=digest,
                        size_bytes=len(payload),
                        created_by_call=publication.created_by_call,
                    )
                results.append(
                    ExactFrameSampleResult(
                        sample.sample_id,
                        sample.timestamp_ms,
                        "SUCCEEDED",
                        candidate=candidate,
                        artifact_status=publication.status,
                    ),
                )
                _LOGGER.info(
                    "视觉关键帧抽取完成 timestamp_ms=%s sha256=%s size_bytes=%s",
                    sample.timestamp_ms,
                    digest,
                    len(payload),
                )
            except VideoDemoError:
                raise
            except (OSError, ValueError):
                results.append(
                    ExactFrameSampleResult(sample.sample_id, sample.timestamp_ms, "DECODE_FAILED")
                )
                _LOGGER.warning(
                    "视觉关键帧抽取失败 timestamp_ms=%s code=DECODE_FAILED", sample.timestamp_ms
                )
            finally:
                with suppress(OSError):
                    temporary.unlink(missing_ok=True)
        return tuple(results)

    def _extract_one(
        self,
        source: Path,
        destination: Path,
        timestamp_ms: int,
        is_cancel_requested: Callable[[], bool],
    ) -> ProcessResult | None:
        result = self._runner.run(
            [
                str(self._executable),
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-ss",
                f"{timestamp_ms / 1000:.3f}",
                "-i",
                str(source),
                "-frames:v",
                "1",
                "-vf",
                f"scale=min({self._frame_width},iw):-2",
                "-q:v",
                str(self._ffmpeg_quality()),
                str(destination),
            ],
            timeout_seconds=self._timeout_seconds,
            output_paths=(destination,),
        )
        if is_cancel_requested():
            raise VideoDemoError(ErrorCode.JOB_CANCELLED, "视觉抽帧已取消")
        if result.returncode != 0 or not destination.is_file():
            return None
        payload = destination.read_bytes()
        if not 4 <= len(payload) <= self._max_frame_bytes or not _is_jpeg(payload):
            return None
        return result

    def _ffmpeg_quality(self) -> int:
        """将项目 1~100 的 JPEG 质量映射为 FFmpeg q:v 的 2~31。"""

        return max(2, min(31, round(31 - (self._jpeg_quality - 1) * 29 / 99)))

    @staticmethod
    def _temporary_path(allowed_root: Path) -> Path:
        directory = allowed_root / "visual" / ".frame-staging"
        directory.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix="frame-", suffix=".jpg", dir=directory)
        os.close(fd)
        return Path(name)


def _is_jpeg(payload: bytes) -> bool:
    return payload.startswith(b"\xff\xd8\xff") and payload.endswith(b"\xff\xd9")
