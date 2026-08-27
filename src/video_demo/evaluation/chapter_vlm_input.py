"""章节 VLM 评测输入的可重验契约与准备器。"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from itertools import pairwise
from pathlib import Path
from typing import Literal

from pydantic import ConfigDict, Field, StrictBool, StrictInt, model_validator

from video_demo.application.production_scene import frame_tolerance_ms_for_rate
from video_demo.domain.base import FrozenModel, Sha256, StableId, stable_identifier
from video_demo.domain.document_plan import FrameCandidateArtifact, frame_candidate_id
from video_demo.domain.manifest import Rational, VideoAssetManifest
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.evaluation.annotations import (
    EvaluationAnnotation,
    ValidatedEvaluationPackage,
    VerifiedAnnotation,
)
from video_demo.evaluation.dataset import EvaluationSample
from video_demo.media.probe import FFprobeClient, ProbeLimits, SupportedMime
from video_demo.media.transcode import FFmpegTranscoder
from video_demo.storage.workspace import (
    atomic_replace,
    reject_symlink_components,
    safe_runtime_path,
    verified_mp4_file,
)
from video_demo.visual.candidate_artifacts import (
    CandidateArtifactSession,
    CandidateDirectoryLease,
    read_verified_candidate_jpeg,
)
from video_demo.visual.keyframes import FrameSample, OpenCvFrameExtractor

_MAX_CHAPTER_SPAN_MS = 300_000
_MAX_MANIFEST_BYTES = 4 * 1024 * 1024
_DEFAULT_DURATION_TOLERANCE_MS = 100
_VISUAL_TARGET_TEXT = "识别并结构化提取这些画面中实际可见的文字、代码、表格、公式和界面状态"
_PREPARATION_CODES = frozenset(
    {
        ErrorCode.INVALID_CONFIGURATION,
        ErrorCode.VIDEO_FFMPEG_UNAVAILABLE,
        ErrorCode.VIDEO_FFPROBE_UNAVAILABLE,
        ErrorCode.VISUAL_DEPENDENCY_UNAVAILABLE,
        ErrorCode.VIDEO_DISK_SPACE_INSUFFICIENT,
        ErrorCode.LIVE_AUTHORIZED_CHAPTER_FRAMES_UNAVAILABLE,
        ErrorCode.DEPENDENCY_TEMPORARY_FAILURE,
        ErrorCode.JOB_CANCELLED,
        ErrorCode.INPUT_BUDGET_EXCEEDED,
        ErrorCode.ARTIFACT_SCHEMA_INVALID,
        ErrorCode.IDEMPOTENCY_CONFLICT,
    }
)
_PREFLIGHT_CODES = frozenset(
    {
        ErrorCode.INVALID_CONFIGURATION,
        ErrorCode.VIDEO_FFMPEG_UNAVAILABLE,
        ErrorCode.VIDEO_FFPROBE_UNAVAILABLE,
        ErrorCode.VISUAL_DEPENDENCY_UNAVAILABLE,
        ErrorCode.VIDEO_DISK_SPACE_INSUFFICIENT,
        ErrorCode.LIVE_AUTHORIZED_CHAPTER_FRAMES_UNAVAILABLE,
    }
)


def _stable_preparation_error_code(code: ErrorCode) -> ErrorCode:
    """把底层依赖错误收敛到准备结果允许的稳定错误码集合。"""

    if code in _PREPARATION_CODES:
        return code
    return ErrorCode.ARTIFACT_SCHEMA_INVALID


class ChapterVlmInputFrame(FrozenModel):
    model_config = ConfigDict(revalidate_instances="always")

    reference_frame_id: StableId
    frame_id: StableId
    requested_timestamp_ms: StrictInt = Field(ge=0)
    actual_timestamp_ms: StrictInt = Field(ge=0)
    relative_path: str = Field(min_length=1, max_length=1024)
    mime_type: Literal["image/jpeg"] = "image/jpeg"
    sha256: Sha256
    size_bytes: StrictInt = Field(gt=0, le=5 * 1024 * 1024)
    perceptual_hash: str = Field(pattern=r"^[0-9a-f]{16}$")
    target_ids: tuple[StableId, ...] = Field(min_length=1, max_length=1)


class ChapterVlmInputManifest(FrozenModel):
    model_config = ConfigDict(revalidate_instances="always")

    schema_version: Literal["1.0.0"]
    parent_evaluation_run_id: StableId
    evaluation_run_id: StableId
    sample_id: StableId
    source_media_sha256: Sha256
    source_duration_ms: StrictInt = Field(gt=0, le=7_200_000)
    annotation_sha256: Sha256
    proxy_max_edge: StrictInt = Field(ge=1_280, le=2_560)
    proxy_width: StrictInt = Field(gt=0)
    proxy_height: StrictInt = Field(gt=0)
    proxy_frame_rate: Rational
    proxy_is_variable_frame_rate: StrictBool
    proxy_duration_ms: StrictInt = Field(gt=0, le=7_200_000)
    proxy_relative_path: Literal["media/proxy.mp4"]
    duration_tolerance_ms: StrictInt = Field(ge=0, le=1_000)
    jpeg_quality: StrictInt = Field(ge=1, le=100)
    proxy_sha256: Sha256
    proxy_size_bytes: StrictInt = Field(gt=0)
    frame_tolerance_ms: StrictInt = Field(ge=1, le=100)
    requested_reference_frame_ids: tuple[StableId, ...] = Field(min_length=2, max_length=4)
    requested_image_sha256s: tuple[Sha256, ...] = Field(min_length=2, max_length=4)
    retained_reference_frame_ids: tuple[StableId, ...] = Field(min_length=2, max_length=4)
    duplicate_frame_count: StrictInt = Field(ge=0, le=2)
    frames: tuple[ChapterVlmInputFrame, ...] = Field(min_length=2, max_length=4)

    @model_validator(mode="after")
    def validate_manifest_shape(self) -> ChapterVlmInputManifest:
        requested = self.requested_reference_frame_ids
        images = self.requested_image_sha256s
        if self.parent_evaluation_run_id == self.evaluation_run_id:
            raise ValueError("父评测 Run 与章节输入产品 Run 必须不同")
        if len(set(requested)) != len(requested):
            raise ValueError("requested_reference_frame_ids 不得重复")
        if len(images) != len(requested):
            raise ValueError("requested_image_sha256s 必须与请求参考帧一一对应")
        first_ids = tuple(
            frame_id
            for index, frame_id in enumerate(requested)
            if images[index] not in images[:index]
        )
        if self.retained_reference_frame_ids != first_ids:
            raise ValueError("保留参考帧必须按图片 SHA 首次出现顺序排列")
        if self.duplicate_frame_count != len(requested) - len(first_ids):
            raise ValueError("duplicate_frame_count 与图片去重结果不一致")
        frame_ids = tuple(frame.reference_frame_id for frame in self.frames)
        if frame_ids != self.retained_reference_frame_ids:
            raise ValueError("frames 顺序必须等于 retained_reference_frame_ids")
        retained_images = tuple(
            images[index]
            for index, frame_id in enumerate(requested)
            if frame_id in self.retained_reference_frame_ids and requested.index(frame_id) == index
        )
        if tuple(frame.sha256 for frame in self.frames) != retained_images:
            raise ValueError("frames 的图片摘要必须等于请求 SHA 首次出现项")
        if self.frames != tuple(
            sorted(
                self.frames, key=lambda item: (item.requested_timestamp_ms, item.reference_frame_id)
            )
        ):
            raise ValueError("评测帧必须按请求时间和参考帧 ID 有序")
        if any(
            current.requested_timestamp_ms <= previous.requested_timestamp_ms
            or current.actual_timestamp_ms <= previous.actual_timestamp_ms
            for previous, current in zip(self.frames[:-1], self.frames[1:], strict=True)
        ):
            raise ValueError("评测帧时间必须严格递增")
        if (
            self.frames[-1].requested_timestamp_ms - self.frames[0].requested_timestamp_ms
            > _MAX_CHAPTER_SPAN_MS
        ):
            raise ValueError("评测帧不得跨越 5 分钟章节范围")
        target_id = base_coverage_target_id(self)
        for frame in self.frames:
            if frame.target_ids != (target_id,):
                raise ValueError("评测输入帧必须绑定唯一 BASE_COVERAGE 目标")
            if frame.frame_id != frame_candidate_id(
                self.source_media_sha256, frame.actual_timestamp_ms, frame.sha256
            ):
                raise ValueError("评测 frame_id 与媒体、时间和图片摘要不匹配")
            if (
                frame.actual_timestamp_ms >= self.proxy_duration_ms
                or abs(frame.actual_timestamp_ms - frame.requested_timestamp_ms)
                > self.frame_tolerance_ms
            ):
                raise ValueError("评测帧实际时间超出容差或代理时长")
        if self.proxy_frame_rate.numerator <= 0 or self.proxy_frame_rate.denominator <= 0:
            raise ValueError("代理帧率必须是正有理数")
        if max(self.proxy_width, self.proxy_height) > self.proxy_max_edge:
            raise ValueError("代理实际长边超过请求上限")
        return self


def evaluation_run_id_for_input(
    parent_evaluation_run_id: StableId,
    sample_id: StableId,
    media_sha256: Sha256,
    annotation_sha256: Sha256,
    proxy_max_edge: int,
    jpeg_quality: int,
    requested_reference_frame_ids: tuple[StableId, ...],
) -> StableId:
    return stable_identifier(
        "chapter_vlm",
        {
            "parent_evaluation_run_id": parent_evaluation_run_id,
            "sample_id": sample_id,
            "media_sha256": media_sha256,
            "annotation_sha256": annotation_sha256,
            "proxy_max_edge": proxy_max_edge,
            "jpeg_quality": jpeg_quality,
            "requested_reference_frame_ids": requested_reference_frame_ids,
        },
    )


def chapter_vlm_chapter_id(manifest: ChapterVlmInputManifest) -> StableId:
    return stable_identifier(
        "chapter",
        {
            "asset_sha256": manifest.source_media_sha256,
            "evaluation_run_id": manifest.evaluation_run_id,
            "sample_id": manifest.sample_id,
            "requested_reference_frame_ids": manifest.requested_reference_frame_ids,
        },
    )


def base_coverage_target_id(manifest: ChapterVlmInputManifest) -> StableId:
    return stable_identifier(
        "visual_target",
        {
            "asset_sha256": manifest.source_media_sha256,
            "chapter_id": chapter_vlm_chapter_id(manifest),
            "purpose": "BASE_COVERAGE",
            "ordinal": 0,
            "target": {"query_zh": _VISUAL_TARGET_TEXT},
        },
    )


class ValidatedChapterVlmInputContext(FrozenModel):
    parent_evaluation_run_id: StableId
    evaluation_run_id: StableId
    sample_id: StableId
    source_media_sha256: Sha256
    annotation_sha256: Sha256
    source_duration_ms: StrictInt = Field(gt=0, le=7_200_000)
    source_display_width: StrictInt = Field(gt=0)
    source_display_height: StrictInt = Field(gt=0)
    allowed_run_root: Path
    proxy_relative_path: Literal["media/proxy.mp4"]
    proxy_sha256: Sha256
    proxy_size_bytes: StrictInt = Field(gt=0)
    proxy_max_edge: StrictInt = Field(ge=1_280, le=2_560)
    proxy_width: StrictInt = Field(gt=0)
    proxy_height: StrictInt = Field(gt=0)
    proxy_frame_rate: Rational
    proxy_is_variable_frame_rate: StrictBool
    proxy_duration_ms: StrictInt = Field(gt=0, le=7_200_000)
    duration_tolerance_ms: StrictInt = Field(ge=0, le=1_000)
    frame_tolerance_ms: StrictInt = Field(ge=1, le=100)
    jpeg_quality: StrictInt = Field(ge=1, le=100)
    vlm_max_image_bytes: StrictInt = Field(gt=0, le=5 * 1024 * 1024)
    max_candidate_frame_bytes_per_run: StrictInt = Field(gt=0, le=512 * 1024 * 1024)
    max_candidate_frame_files_per_run: StrictInt = Field(gt=0, le=20_000)


def chapter_vlm_input_manifest_sha256(manifest: ChapterVlmInputManifest) -> Sha256:
    encoded = json.dumps(
        manifest.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class ChapterVlmInputPreparation(FrozenModel):
    model_config = ConfigDict(revalidate_instances="always")
    status: Literal["READY", "NOT_RUN", "FAIL"]
    execution_started: StrictBool
    manifest: ChapterVlmInputManifest | None = None
    manifest_sha256: Sha256 | None = None
    error_code: ErrorCode | None = None

    @model_validator(mode="after")
    def validate_status_shape(self) -> ChapterVlmInputPreparation:
        if self.error_code is not None and self.error_code not in _PREPARATION_CODES:
            raise ValueError("评测输入准备错误码不在稳定白名单内")
        if self.status == "READY" and (
            not self.execution_started
            or self.manifest is None
            or self.error_code is not None
            or self.manifest_sha256 != chapter_vlm_input_manifest_sha256(self.manifest)
        ):
            raise ValueError("READY 必须包含已校验 manifest 且无错误码")
        if self.status == "NOT_RUN" and (
            self.execution_started
            or self.manifest is not None
            or self.manifest_sha256 is not None
            or self.error_code not in _PREFLIGHT_CODES
        ):
            raise ValueError("NOT_RUN 必须是未开始执行的前置失败")
        if self.status == "FAIL" and (not self.execution_started or self.error_code is None):
            raise ValueError("FAIL 必须已开始执行且有稳定错误码")
        if self.manifest is None and self.manifest_sha256 is not None:
            raise ValueError("没有 manifest 时不得保存 manifest_sha256")
        if self.manifest is not None and self.manifest_sha256 != chapter_vlm_input_manifest_sha256(
            self.manifest
        ):
            raise ValueError("manifest_sha256 与 manifest 不匹配")
        return self


def validate_chapter_vlm_input_manifest(
    manifest: ChapterVlmInputManifest,
    *,
    context: ValidatedChapterVlmInputContext,
) -> None:
    pairs = (
        (manifest.parent_evaluation_run_id, context.parent_evaluation_run_id),
        (manifest.evaluation_run_id, context.evaluation_run_id),
        (manifest.sample_id, context.sample_id),
        (manifest.source_media_sha256, context.source_media_sha256),
        (manifest.annotation_sha256, context.annotation_sha256),
        (manifest.source_duration_ms, context.source_duration_ms),
        (manifest.proxy_relative_path, context.proxy_relative_path),
        (manifest.proxy_sha256, context.proxy_sha256),
        (manifest.proxy_size_bytes, context.proxy_size_bytes),
        (manifest.proxy_max_edge, context.proxy_max_edge),
        (manifest.proxy_width, context.proxy_width),
        (manifest.proxy_height, context.proxy_height),
        (manifest.proxy_frame_rate, context.proxy_frame_rate),
        (manifest.proxy_is_variable_frame_rate, context.proxy_is_variable_frame_rate),
        (manifest.proxy_duration_ms, context.proxy_duration_ms),
        (manifest.duration_tolerance_ms, context.duration_tolerance_ms),
        (manifest.frame_tolerance_ms, context.frame_tolerance_ms),
        (manifest.jpeg_quality, context.jpeg_quality),
    )
    if any(actual != expected for actual, expected in pairs):
        raise VideoDemoError(
            ErrorCode.ARTIFACT_SCHEMA_INVALID, "章节 VLM Manifest 与已验证上下文不一致"
        )
    if (
        abs(manifest.proxy_duration_ms - manifest.source_duration_ms)
        > context.duration_tolerance_ms
    ):
        raise VideoDemoError(ErrorCode.ARTIFACT_SCHEMA_INVALID, "源视频与代理时长超出容差")
    run_root = context.allowed_run_root.expanduser().resolve(strict=True)
    runtime_root = run_root.parents[2]
    relative_run = run_root.relative_to(runtime_root)
    if relative_run.parts[:1] != ("runs",) or len(relative_run.parts) != 3:
        raise VideoDemoError(ErrorCode.ARTIFACT_SCHEMA_INVALID, "评测 Run 根层级非法")
    proxy = safe_runtime_path(run_root, Path(manifest.proxy_relative_path))
    verified_mp4_file(
        runtime_root,
        relative_run,
        proxy,
        expected_sha256=manifest.proxy_sha256,
        expected_size_bytes=manifest.proxy_size_bytes,
        max_size_bytes=manifest.proxy_size_bytes,
        message="评测代理必须是当前 Run 内安全 MP4",
    )
    seen_paths: set[str] = set()
    seen_sha: set[str] = set()
    for frame in manifest.frames:
        if frame.relative_path in seen_paths or frame.sha256 in seen_sha:
            raise VideoDemoError(ErrorCode.ARTIFACT_SCHEMA_INVALID, "评测候选帧不得重复")
        seen_paths.add(frame.relative_path)
        seen_sha.add(frame.sha256)
        read_verified_candidate_jpeg(
            run_root,
            FrameCandidateArtifact(
                frame_id=frame.frame_id,
                timestamp_ms=frame.actual_timestamp_ms,
                sha256=frame.sha256,
                size_bytes=frame.size_bytes,
                relative_path=frame.relative_path,
                mime_type=frame.mime_type,
                perceptual_hash=frame.perceptual_hash,
                target_ids=frame.target_ids,
            ),
            max_bytes=context.vlm_max_image_bytes,
        )


def _choose_reference_frames(annotation: EvaluationAnnotation) -> tuple[tuple[str, int], ...]:
    buckets: dict[int, list[str]] = {}
    for frame in sorted(
        annotation.visual_frames, key=lambda item: (item.timestamp_ms, item.frame_id)
    ):
        buckets.setdefault(frame.timestamp_ms, []).append(frame.frame_id)
    values = tuple((timestamp, min(ids)) for timestamp, ids in sorted(buckets.items()))
    best: tuple[tuple[int, str], ...] = ()
    for start in range(len(values)):
        for end in range(start + 1, len(values)):
            candidate = values[start : end + 1]
            if candidate[-1][0] - candidate[0][0] > _MAX_CHAPTER_SPAN_MS:
                break
            if len(candidate) > len(best) or (len(candidate) == len(best) and candidate < best):
                best = candidate
    if len(best) > 4:
        indexes = (0, round((len(best) - 1) / 3), round(2 * (len(best) - 1) / 3), len(best) - 1)
        best = tuple(best[index] for index in indexes)
    return tuple((frame_id, timestamp) for timestamp, frame_id in best)


def _mime_for_path(path: Path) -> SupportedMime:
    mapping: dict[str, SupportedMime] = {
        ".mp4": "video/mp4",
        ".mov": "video/quicktime",
        ".mkv": "video/x-matroska",
        ".webm": "video/webm",
    }
    try:
        return mapping[path.suffix.casefold()]
    except KeyError:
        raise VideoDemoError(ErrorCode.VIDEO_FORMAT_UNSUPPORTED, "评测媒体后缀不受支持") from None


def _find_sample(
    package: ValidatedEvaluationPackage, sample_id: StableId
) -> tuple[EvaluationSample, VerifiedAnnotation]:
    sample = next((item for item in package.dataset.samples if item.sample_id == sample_id), None)
    verified = next(
        (item for item in package.annotations if item.annotation.sample_id == sample_id), None
    )
    if sample is None or verified is None:
        raise VideoDemoError(ErrorCode.LIVE_AUTHORIZED_CHAPTER_FRAMES_UNAVAILABLE, "授权样本不存在")
    return sample, verified


def _source_display_size(manifest: VideoAssetManifest) -> tuple[int, int]:
    stream = manifest.video_stream
    return (
        (stream.height, stream.width)
        if stream.rotation_degrees in (90, 270)
        else (stream.width, stream.height)
    )


def _context_from_parts(
    manifest: ChapterVlmInputManifest,
    run_root: Path,
    source_manifest: VideoAssetManifest,
    proxy_manifest: VideoAssetManifest,
    proxy_sha256: Sha256,
    proxy_size_bytes: int,
    proxy_max_edge: int,
    jpeg_quality: int,
    frame_tolerance_ms: int,
    vlm_max_image_bytes: int,
    max_candidate_frame_bytes_per_run: int,
    max_candidate_frame_files_per_run: int,
) -> ValidatedChapterVlmInputContext:
    width, height = _source_display_size(source_manifest)
    return ValidatedChapterVlmInputContext(
        parent_evaluation_run_id=manifest.parent_evaluation_run_id,
        evaluation_run_id=manifest.evaluation_run_id,
        sample_id=manifest.sample_id,
        source_media_sha256=manifest.source_media_sha256,
        annotation_sha256=manifest.annotation_sha256,
        source_duration_ms=manifest.source_duration_ms,
        source_display_width=width,
        source_display_height=height,
        allowed_run_root=run_root,
        proxy_relative_path=manifest.proxy_relative_path,
        proxy_sha256=proxy_sha256,
        proxy_size_bytes=proxy_size_bytes,
        proxy_max_edge=proxy_max_edge,
        proxy_width=proxy_manifest.video_stream.width,
        proxy_height=proxy_manifest.video_stream.height,
        proxy_frame_rate=proxy_manifest.video_stream.average_frame_rate,
        proxy_is_variable_frame_rate=proxy_manifest.video_stream.is_variable_frame_rate,
        proxy_duration_ms=proxy_manifest.duration_ms,
        duration_tolerance_ms=_DEFAULT_DURATION_TOLERANCE_MS,
        frame_tolerance_ms=frame_tolerance_ms,
        jpeg_quality=jpeg_quality,
        vlm_max_image_bytes=vlm_max_image_bytes,
        max_candidate_frame_bytes_per_run=max_candidate_frame_bytes_per_run,
        max_candidate_frame_files_per_run=max_candidate_frame_files_per_run,
    )


def _read_manifest(path: Path) -> ChapterVlmInputManifest:
    try:
        if path.is_symlink() or not path.is_file() or path.stat().st_size > _MAX_MANIFEST_BYTES:
            raise OSError
        return ChapterVlmInputManifest.model_validate_json(path.read_bytes())
    except (OSError, ValueError):
        raise VideoDemoError(ErrorCode.IDEMPOTENCY_CONFLICT, "既有评测 Manifest 不可复用") from None


def _write_manifest(
    runtime: Path, run_relative_root: Path, manifest: ChapterVlmInputManifest
) -> None:
    destination = safe_runtime_path(runtime, run_relative_root / "visual/chapter-vlm-input.json")
    reject_symlink_components(runtime, destination, message="评测 Manifest 路径不能包含符号链接")
    encoded = (
        json.dumps(
            manifest.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    if len(encoded) > _MAX_MANIFEST_BYTES:
        raise VideoDemoError(ErrorCode.INPUT_BUDGET_EXCEEDED, "评测 Manifest 超过大小上限")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise VideoDemoError(ErrorCode.IDEMPOTENCY_CONFLICT, "评测 Manifest 已存在")
    temporary = destination.with_name(f".{destination.name}.part")
    try:
        with temporary.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            import os

            os.fsync(stream.fileno())
        atomic_replace(temporary, destination)
    except FileExistsError:
        raise VideoDemoError(ErrorCode.IDEMPOTENCY_CONFLICT, "评测 Manifest 并发冲突") from None
    finally:
        temporary.unlink(missing_ok=True)


def prepare_chapter_vlm_input(
    package: ValidatedEvaluationPackage,
    *,
    parent_evaluation_run_id: StableId,
    proxy_max_edge: int,
    jpeg_quality: int,
    max_video_bytes: int,
    vlm_max_image_bytes: int,
    max_candidate_frame_bytes_per_run: int,
    max_candidate_frame_files_per_run: int,
    ffprobe: FFprobeClient,
    transcoder: FFmpegTranscoder,
    frame_extractor: OpenCvFrameExtractor,
    runtime_root: Path,
    sample_id: StableId | None = None,
    requested_reference_frame_ids: tuple[StableId, ...] | None = None,
    is_cancel_requested: Callable[[], bool] = lambda: False,
) -> ChapterVlmInputPreparation:
    """同一排他租约内完成探测、代理、抽帧和 Manifest 发布。"""
    started = False
    proxy_identity: tuple[Path, tuple[int, int]] | None = None
    try:
        if sample_id is None:
            for candidate in package.dataset.samples:
                verified = next(
                    item
                    for item in package.annotations
                    if item.annotation.sample_id == candidate.sample_id
                )
                selected = _choose_reference_frames(verified.annotation)
                if len(selected) >= 2:
                    sample_id = candidate.sample_id
                    requested_reference_frame_ids = tuple(item[0] for item in selected)
                    break
        if sample_id is None or requested_reference_frame_ids is None:
            return ChapterVlmInputPreparation(
                status="NOT_RUN",
                execution_started=False,
                error_code=ErrorCode.LIVE_AUTHORIZED_CHAPTER_FRAMES_UNAVAILABLE,
            )
        sample, verified = _find_sample(package, sample_id)
        references = {frame.frame_id: frame for frame in verified.annotation.visual_frames}
        if len(requested_reference_frame_ids) not in {2, 3, 4} or any(
            frame_id not in references for frame_id in requested_reference_frame_ids
        ):
            raise VideoDemoError(
                ErrorCode.LIVE_AUTHORIZED_CHAPTER_FRAMES_UNAVAILABLE, "请求参考帧不满足章节输入条件"
            )
        ordered = tuple(references[frame_id] for frame_id in requested_reference_frame_ids)
        if (
            any(
                current.timestamp_ms <= previous.timestamp_ms
                for previous, current in pairwise(ordered)
            )
            or ordered[-1].timestamp_ms - ordered[0].timestamp_ms > _MAX_CHAPTER_SPAN_MS
        ):
            raise VideoDemoError(
                ErrorCode.LIVE_AUTHORIZED_CHAPTER_FRAMES_UNAVAILABLE, "请求参考帧时间簇非法"
            )
        source_path = package.dataset.eval_root / sample.media_relative_path
        source_size = source_path.stat().st_size
        if source_size > max_video_bytes:
            raise VideoDemoError(ErrorCode.INPUT_BUDGET_EXCEEDED, "评测媒体超过大小预算")
        source_probe = ffprobe.probe(
            source_path,
            object_ref=sample.sample_id,
            source_sha256=sample.media_sha256,
            source_size_bytes=source_size,
            source_mime=_mime_for_path(source_path),
            limits=ProbeLimits(max_duration_ms=7_200_000),
        )
        if source_probe.manifest.duration_ms != verified.annotation.duration_ms:
            raise VideoDemoError(ErrorCode.ARTIFACT_SCHEMA_INVALID, "媒体时长与授权标注不一致")
        evaluation_run_id = evaluation_run_id_for_input(
            parent_evaluation_run_id,
            sample.sample_id,
            sample.media_sha256,
            verified.sha256,
            proxy_max_edge,
            jpeg_quality,
            requested_reference_frame_ids,
        )
        run_relative_root = Path("runs") / "evaluation" / evaluation_run_id
        runtime = runtime_root.expanduser().resolve(strict=False)
        run_root = safe_runtime_path(runtime, run_relative_root)
        run_root.mkdir(parents=True, exist_ok=True)
        started = True
        manifest_path = run_root / "visual/chapter-vlm-input.json"
        with CandidateDirectoryLease(
            runtime_root=runtime,
            run_relative_root=run_relative_root,
            mode="EXCLUSIVE",
            is_cancel_requested=is_cancel_requested,
        ) as lease:
            if manifest_path.exists() or manifest_path.is_symlink():
                existing = _read_manifest(manifest_path)
                proxy_path = safe_runtime_path(run_root, Path("media/proxy.mp4"))
                proxy_sha, proxy_size = _file_digest(proxy_path, max_bytes=max_video_bytes)
                proxy_probe = ffprobe.probe(
                    proxy_path,
                    object_ref=sample.sample_id,
                    source_sha256=proxy_sha,
                    source_size_bytes=proxy_size,
                    source_mime="video/mp4",
                    limits=ProbeLimits(max_duration_ms=7_200_000),
                )
                tolerance = frame_tolerance_ms_for_rate(
                    proxy_probe.manifest.video_stream.average_frame_rate,
                    is_variable_frame_rate=proxy_probe.manifest.video_stream.is_variable_frame_rate,
                )
                context = _context_from_parts(
                    manifest=existing,
                    run_root=run_root,
                    source_manifest=source_probe.manifest,
                    proxy_manifest=proxy_probe.manifest,
                    proxy_sha256=proxy_sha,
                    proxy_size_bytes=proxy_size,
                    proxy_max_edge=proxy_max_edge,
                    jpeg_quality=jpeg_quality,
                    frame_tolerance_ms=tolerance,
                    vlm_max_image_bytes=vlm_max_image_bytes,
                    max_candidate_frame_bytes_per_run=max_candidate_frame_bytes_per_run,
                    max_candidate_frame_files_per_run=max_candidate_frame_files_per_run,
                )
                if (
                    existing.evaluation_run_id != evaluation_run_id
                    or existing.proxy_sha256 != proxy_sha
                    or existing.proxy_size_bytes != proxy_size
                    or existing.frame_tolerance_ms != tolerance
                ):
                    raise VideoDemoError(ErrorCode.IDEMPOTENCY_CONFLICT, "既有评测输入上下文不一致")
                validate_chapter_vlm_input_manifest(existing, context=context)
                return ChapterVlmInputPreparation(
                    status="READY",
                    execution_started=True,
                    manifest=existing,
                    manifest_sha256=chapter_vlm_input_manifest_sha256(existing),
                )
            proxy_path = safe_runtime_path(run_root, Path("media/proxy.mp4"))
            if proxy_path.exists() or proxy_path.is_symlink():
                raise VideoDemoError(ErrorCode.IDEMPOTENCY_CONFLICT, "代理存在但 Manifest 缺失")
            proxy = transcoder.create_proxy(
                source_path, run_relative_root, duration_ms=source_probe.manifest.duration_ms
            )
            if proxy.max_edge != proxy_max_edge:
                raise VideoDemoError(
                    ErrorCode.INVALID_CONFIGURATION,
                    "转码器代理长边配置与评测请求不一致",
                )
            proxy_path = safe_runtime_path(runtime, Path(proxy.relative_path))
            status = proxy_path.stat()
            proxy_identity = (proxy_path, (status.st_dev, status.st_ino))
            proxy_sha, proxy_size = _file_digest(proxy_path, max_bytes=max_video_bytes)
            if proxy_sha != proxy.sha256 or proxy_size != proxy.size_bytes:
                raise VideoDemoError(ErrorCode.ARTIFACT_SCHEMA_INVALID, "代理摘要或大小不一致")
            proxy_probe = ffprobe.probe(
                proxy_path,
                object_ref=sample.sample_id,
                source_sha256=proxy_sha,
                source_size_bytes=proxy_size,
                source_mime="video/mp4",
                limits=ProbeLimits(max_duration_ms=7_200_000),
            )
            tolerance = frame_tolerance_ms_for_rate(
                proxy_probe.manifest.video_stream.average_frame_rate,
                is_variable_frame_rate=proxy_probe.manifest.video_stream.is_variable_frame_rate,
            )
            session = CandidateArtifactSession(
                runtime_root=runtime,
                max_unique_bytes=max_candidate_frame_bytes_per_run,
                max_files=max_candidate_frame_files_per_run,
                max_file_bytes=vlm_max_image_bytes,
                is_cancel_requested=is_cancel_requested,
            )
            try:
                session.prepare_run(run_relative_root, lease=lease)
                samples = tuple(
                    FrameSample(
                        sample_id=frame.frame_id,
                        timestamp_ms=frame.timestamp_ms,
                        admission_tier="BASE_PRIMARY",
                    )
                    for frame in ordered
                )
                extracted = frame_extractor.extract_samples(
                    proxy_path,
                    run_relative_root,
                    samples,
                    is_cancel_requested=is_cancel_requested,
                    frame_tolerance_ms=tolerance,
                    artifact_session=session,
                )
                by_id = {item.sample_id: item for item in extracted}
                if len(by_id) != len(samples) or any(
                    item.candidate is None or item.artifact_status != "PUBLISHED"
                    for item in extracted
                ):
                    code = (
                        ErrorCode.INPUT_BUDGET_EXCEEDED
                        if any(item.artifact_status == "BUDGET_REJECTED" for item in extracted)
                        else ErrorCode.ARTIFACT_SCHEMA_INVALID
                    )
                    raise VideoDemoError(code, "章节评测抽帧未形成完整候选")
                candidates = {
                    item.sample_id: item.candidate
                    for item in extracted
                    if item.candidate is not None
                }
                if len(candidates) != len(ordered):
                    raise VideoDemoError(ErrorCode.ARTIFACT_SCHEMA_INVALID, "抽帧候选回绑不完整")
                requested_sha = tuple(
                    _candidate_sha(candidates[frame.frame_id]) for frame in ordered
                )
                retained = tuple(
                    frame.frame_id
                    for index, frame in enumerate(ordered)
                    if requested_sha[index] not in requested_sha[:index]
                )
                if len(retained) < 2:
                    raise VideoDemoError(
                        ErrorCode.ARTIFACT_SCHEMA_INVALID, "相同 SHA 去重后少于 2 张"
                    )
                provisional = ChapterVlmInputManifest.model_construct(
                    schema_version="1.0.0",
                    parent_evaluation_run_id=parent_evaluation_run_id,
                    evaluation_run_id=evaluation_run_id,
                    sample_id=sample.sample_id,
                    source_media_sha256=sample.media_sha256,
                    source_duration_ms=source_probe.manifest.duration_ms,
                    annotation_sha256=verified.sha256,
                    proxy_max_edge=proxy_max_edge,
                    proxy_width=proxy_probe.manifest.video_stream.width,
                    proxy_height=proxy_probe.manifest.video_stream.height,
                    proxy_frame_rate=proxy_probe.manifest.video_stream.average_frame_rate,
                    proxy_is_variable_frame_rate=proxy_probe.manifest.video_stream.is_variable_frame_rate,
                    proxy_duration_ms=proxy_probe.manifest.duration_ms,
                    proxy_relative_path="media/proxy.mp4",
                    duration_tolerance_ms=_DEFAULT_DURATION_TOLERANCE_MS,
                    jpeg_quality=jpeg_quality,
                    proxy_sha256=proxy_sha,
                    proxy_size_bytes=proxy_size,
                    frame_tolerance_ms=tolerance,
                    requested_reference_frame_ids=requested_reference_frame_ids,
                    requested_image_sha256s=requested_sha,
                    retained_reference_frame_ids=retained,
                    duplicate_frame_count=len(requested_sha) - len(retained),
                    frames=(),
                )
                target_id = base_coverage_target_id(provisional)
                manifest_frames = tuple(
                    ChapterVlmInputFrame(
                        reference_frame_id=reference_id,
                        frame_id=frame_candidate_id(
                            sample.media_sha256,
                            candidates[reference_id].timestamp_ms,
                            _candidate_sha(candidates[reference_id]),
                        ),
                        requested_timestamp_ms=references[reference_id].timestamp_ms,
                        actual_timestamp_ms=candidates[reference_id].timestamp_ms,
                        relative_path=candidates[reference_id]
                        .relative_path.relative_to(run_relative_root)
                        .as_posix(),
                        sha256=_candidate_sha(candidates[reference_id]),
                        size_bytes=safe_runtime_path(
                            runtime,
                            candidates[reference_id].relative_path,
                        )
                        .stat()
                        .st_size,
                        perceptual_hash=candidates[reference_id].perceptual_hash,
                        target_ids=(target_id,),
                    )
                    for reference_id in retained
                )
                manifest = ChapterVlmInputManifest.model_validate(
                    provisional.model_copy(update={"frames": manifest_frames}).model_dump(
                        mode="python"
                    )
                )
                context = _context_from_parts(
                    manifest=manifest,
                    run_root=run_root,
                    source_manifest=source_probe.manifest,
                    proxy_manifest=proxy_probe.manifest,
                    proxy_sha256=proxy_sha,
                    proxy_size_bytes=proxy_size,
                    proxy_max_edge=proxy_max_edge,
                    jpeg_quality=jpeg_quality,
                    frame_tolerance_ms=tolerance,
                    vlm_max_image_bytes=vlm_max_image_bytes,
                    max_candidate_frame_bytes_per_run=max_candidate_frame_bytes_per_run,
                    max_candidate_frame_files_per_run=max_candidate_frame_files_per_run,
                )
                validate_chapter_vlm_input_manifest(manifest, context=context)
                _write_manifest(runtime, run_relative_root, manifest)
                return ChapterVlmInputPreparation(
                    status="READY",
                    execution_started=True,
                    manifest=manifest,
                    manifest_sha256=chapter_vlm_input_manifest_sha256(manifest),
                )
            except BaseException:
                session.cleanup_unretained(frozenset())
                raise
            finally:
                session.close()
    except VideoDemoError as error:
        if proxy_identity is not None:
            _remove_owned_file(*proxy_identity)
        stable_code = _stable_preparation_error_code(error.code)
        preflight = not started and stable_code in _PREFLIGHT_CODES
        return ChapterVlmInputPreparation(
            status="NOT_RUN" if preflight else "FAIL",
            execution_started=not preflight,
            error_code=stable_code,
        )
    except (OSError, ValueError, TypeError):
        if proxy_identity is not None:
            _remove_owned_file(*proxy_identity)
        return ChapterVlmInputPreparation(
            status="FAIL",
            execution_started=True,
            error_code=ErrorCode.ARTIFACT_SCHEMA_INVALID,
        )


def _file_digest(path: Path, *, max_bytes: int) -> tuple[str, int]:
    if path.is_symlink() or not path.is_file():
        raise VideoDemoError(ErrorCode.ARTIFACT_SCHEMA_INVALID, "产物不是安全普通文件")
    digest = hashlib.sha256()
    total = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            total += len(chunk)
            if total > max_bytes:
                raise VideoDemoError(ErrorCode.INPUT_BUDGET_EXCEEDED, "产物超过字节预算")
            digest.update(chunk)
    return digest.hexdigest(), total


def _candidate_sha(candidate: object) -> Sha256:
    path = getattr(candidate, "relative_path", None)
    if not isinstance(path, Path):
        raise VideoDemoError(ErrorCode.ARTIFACT_SCHEMA_INVALID, "候选帧路径类型非法")
    digest = path.stem
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise VideoDemoError(ErrorCode.ARTIFACT_SCHEMA_INVALID, "候选帧路径摘要非法")
    return digest


def _remove_owned_file(path: Path, identity: tuple[int, int]) -> None:
    try:
        current = path.lstat()
        if (
            not path.is_symlink()
            and path.is_file()
            and (current.st_dev, current.st_ino) == identity
        ):
            path.unlink()
    except OSError:
        pass
