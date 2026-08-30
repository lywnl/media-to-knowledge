from __future__ import annotations

import hashlib
import logging
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

from pydantic import ValidationError

from video_demo.application.pipeline import PipelineContext
from video_demo.application.pipeline_contracts import (
    PipelineRunConfig,
    PreparedMedia,
    ProbedAsset,
    RegisteredAsset,
    pipeline_run_config_from_snapshot,
)
from video_demo.domain.manifest import SubtitleStream
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.media.probe import FFprobeClient, ProbeLimits, ProbeResult, SupportedMime
from video_demo.media.subtitles import (
    ParsedSubtitle,
    SubtitleTrackRejected,
    is_subtitle_eligible,
    parse_webvtt,
    rank_text_subtitle_streams,
)
from video_demo.media.transcode import (
    AudioArtifact,
    FFmpegTranscoder,
    NoAudioArtifact,
    SubtitleArtifact,
    TranscodeLimits,
)
from video_demo.persistence.database import Database
from video_demo.persistence.repositories import VideoObjectRepository, VideoRunRepository
from video_demo.storage.object_store import LocalVideoObjectStore, VideoObjectRecord
from video_demo.storage.workspace import safe_runtime_path, verified_run_file

_SUPPORTED_MIME_TYPES = frozenset(
    {"video/mp4", "video/quicktime", "video/x-matroska", "video/webm"},
)
_LOGGER = logging.getLogger(__name__)


class ProbeClient(Protocol):
    def probe(
        self,
        source: Path,
        *,
        object_ref: str,
        source_sha256: str,
        source_size_bytes: int,
        source_mime: SupportedMime,
        limits: ProbeLimits,
    ) -> ProbeResult: ...


class TranscodeClient(Protocol):
    def extract_audio(
        self,
        source: Path,
        run_relative_root: Path,
        *,
        has_audio: bool,
        duration_ms: int,
    ) -> AudioArtifact | NoAudioArtifact: ...

    def extract_subtitle(
        self,
        source: Path,
        run_relative_root: Path,
        stream: SubtitleStream,
        *,
        duration_ms: int | None = None,
    ) -> SubtitleArtifact: ...


@dataclass(frozen=True, slots=True)
class DocumentCapacityProfile:
    """输入已物化后，知识文档流水线在当前卷上的最坏容量画像。"""

    proxy_output_bytes: int
    proxy_temporary_bytes: int
    pcm_audio_bytes: int
    max_asr_slice_bytes: int
    candidate_frame_bytes: int
    published_keyframe_bytes: int
    model_cache_bytes: int
    result_bundle_bytes: int
    document_bytes: int
    reserve_bytes: int
    required_free_bytes: int

    def __post_init__(self) -> None:
        components = (
            self.proxy_output_bytes,
            self.proxy_temporary_bytes,
            self.pcm_audio_bytes,
            self.max_asr_slice_bytes,
            self.candidate_frame_bytes,
            self.published_keyframe_bytes,
            self.model_cache_bytes,
            self.result_bundle_bytes,
            self.document_bytes,
            self.reserve_bytes,
        )
        if any(type(value) is not int or value < 0 for value in components):
            raise ValueError("容量画像的组成项必须是非负整数")
        if type(self.required_free_bytes) is not int or self.required_free_bytes != sum(
            components
        ):
            raise ValueError("容量画像总量必须等于各组成项之和")


def build_document_capacity_profile(
    *,
    duration_ms: int,
    proxy_estimated_bytes_per_second: int,
    max_proxy_bytes: int,
    max_asr_window_ms: int,
    candidate_frame_bytes: int,
    published_keyframe_bytes: int,
    model_cache_bytes: int,
    result_bundle_bytes: int,
    document_bytes: int,
    reserve_bytes: int,
    proxy_transcode_required: bool = True,
) -> DocumentCapacityProfile:
    """计算 Probe 后容量；已经落盘的 source 不在此重复计入。"""

    positive_values = {
        "duration_ms": duration_ms,
        "proxy_estimated_bytes_per_second": proxy_estimated_bytes_per_second,
        "max_proxy_bytes": max_proxy_bytes,
        "max_asr_window_ms": max_asr_window_ms,
    }
    if any(type(value) is not int or value < 1 for value in positive_values.values()):
        raise ValueError("容量画像的时长、代理速率和上限必须是正整数")
    byte_budgets = {
        "candidate_frame_bytes": candidate_frame_bytes,
        "published_keyframe_bytes": published_keyframe_bytes,
        "model_cache_bytes": model_cache_bytes,
        "result_bundle_bytes": result_bundle_bytes,
        "document_bytes": document_bytes,
        "reserve_bytes": reserve_bytes,
    }
    if any(type(value) is not int or value < 0 for value in byte_budgets.values()):
        raise ValueError("容量画像的字节预算必须是非负整数")
    if not isinstance(proxy_transcode_required, bool):
        raise ValueError("代理转码开关必须是布尔值")

    duration_seconds = (duration_ms + 999) // 1_000
    proxy_output_bytes = (
        min(
            duration_seconds * proxy_estimated_bytes_per_second,
            max_proxy_bytes,
        )
        if proxy_transcode_required
        else 0
    )
    pcm_audio_bytes = duration_seconds * 32_000
    max_asr_slice_bytes = ((max_asr_window_ms + 999) // 1_000) * 32_000
    parts = (
        proxy_output_bytes,
        proxy_output_bytes,
        pcm_audio_bytes,
        max_asr_slice_bytes,
        *byte_budgets.values(),
    )
    return DocumentCapacityProfile(
        proxy_output_bytes=proxy_output_bytes,
        proxy_temporary_bytes=proxy_output_bytes,
        pcm_audio_bytes=pcm_audio_bytes,
        max_asr_slice_bytes=max_asr_slice_bytes,
        candidate_frame_bytes=candidate_frame_bytes,
        published_keyframe_bytes=published_keyframe_bytes,
        model_cache_bytes=model_cache_bytes,
        result_bundle_bytes=result_bundle_bytes,
        document_bytes=document_bytes,
        reserve_bytes=reserve_bytes,
        required_free_bytes=sum(parts),
    )


def require_document_capacity(
    path: Path,
    profile: DocumentCapacityProfile,
    *,
    available_bytes: Callable[[Path], int] | None = None,
) -> None:
    """在 Probe 后、转码前对当前目标卷执行容量门禁。"""

    candidate = path.expanduser().resolve(strict=False)
    free_bytes = (
        shutil.disk_usage(candidate).free
        if available_bytes is None
        else available_bytes(candidate)
    )
    if free_bytes < profile.required_free_bytes:
        raise VideoDemoError(
            ErrorCode.VIDEO_DISK_SPACE_INSUFFICIENT,
            "可用磁盘空间不足",
        )


class ProductionAssetRegistrar:
    def __init__(self, database: Database, object_store: LocalVideoObjectStore) -> None:
        self._database = database
        self._object_store = object_store

    def register(self, context: PipelineContext) -> RegisteredAsset:
        if context.scope is None:
            raise VideoDemoError(ErrorCode.INVALID_CONFIGURATION, "生产流水线缺少任务作用域")
        with self._database.session() as session:
            run = VideoRunRepository(session).get(context.scope, context.run_id)
            if run is None:
                raise VideoDemoError(ErrorCode.VIDEO_RUN_NOT_FOUND, "视频理解运行不存在")
            object_model = VideoObjectRepository(session).get_ready(context.scope, run.object_ref)
            if object_model is None:
                raise VideoDemoError(ErrorCode.VIDEO_OBJECT_NOT_FOUND, "视频对象不存在")
            record = VideoObjectRecord(
                object_ref=object_model.object_ref,
                original_filename=object_model.original_filename,
                declared_mime=object_model.declared_mime,
                detected_mime=object_model.detected_mime,
                size_bytes=object_model.size_bytes,
                sha256=object_model.sha256,
                relative_path=object_model.relative_path,
                scope_key=LocalVideoObjectStore.scope_key(context.scope),
            )
            source = self._object_store.materialize(
                context.scope,
                record,
                context.run_id,
                object_model.sha256,
            )
            config = self._run_config(run.config_snapshot)
        return RegisteredAsset(
            source_path=source,
            source_sha256=record.sha256,
            object_ref=record.object_ref,
            source_size_bytes=record.size_bytes,
            source_mime=_supported_mime(record.detected_mime),
            run_relative_root=Path("runs") / record.scope_key / context.run_id,
            config=config,
        )

    @staticmethod
    def _run_config(snapshot: dict[str, object]) -> PipelineRunConfig:
        try:
            return pipeline_run_config_from_snapshot(snapshot)
        except ValidationError as error:
            raise VideoDemoError(
                ErrorCode.INVALID_CONFIGURATION,
                "视频理解运行配置快照非法",
            ) from error


class ProductionAssetProbe:
    def __init__(
        self,
        client_factory: Callable[[], ProbeClient],
        *,
        limits: ProbeLimits | None = None,
    ) -> None:
        self._client_factory = client_factory
        self._limits = limits or ProbeLimits()

    def probe(self, asset: RegisteredAsset) -> ProbedAsset:
        result = self._client_factory().probe(
            asset.source_path,
            object_ref=asset.object_ref,
            source_sha256=asset.source_sha256,
            source_size_bytes=asset.source_size_bytes,
            source_mime=asset.source_mime,
            limits=self._limits,
        )
        return ProbedAsset(
            asset=asset,
            manifest=result.manifest,
            limits=self._limits,
            warnings=result.warnings,
            timeline_duration_ms=result.timeline_duration_ms,
        )


class ProductionMediaTranscoder:
    def __init__(
        self,
        runtime_root: Path,
        client_factory: Callable[[Callable[[], bool]], TranscodeClient],
        *,
        max_proxy_bytes: int = 4 * 1024 * 1024 * 1024,
    ) -> None:
        self._runtime_root = runtime_root.expanduser().resolve(strict=False)
        self._client_factory = client_factory
        self._max_proxy_bytes = max_proxy_bytes

    def transcode(
        self,
        probed: ProbedAsset,
        *,
        is_cancel_requested: Callable[[], bool] = lambda: False,
    ) -> PreparedMedia:
        client = self._client_factory(is_cancel_requested)
        asset = probed.asset
        source = self._verified_source(asset)
        proxy_path, proxy_sha256, proxy_size_bytes = self._visual_input(probed, source)
        warnings = list(probed.warnings)
        subtitle = self._select_subtitle(client, probed, warnings)
        if subtitle is not None:
            audio_path = None
            audio_sha256 = None
        else:
            audio = client.extract_audio(
                source,
                asset.run_relative_root,
                has_audio=bool(probed.manifest.audio_streams),
                duration_ms=probed.duration_ms,
            )
            if isinstance(audio, NoAudioArtifact):
                warnings.append(audio.warning_code)
                audio_path = None
                audio_sha256 = None
            else:
                audio_path = self._verified_run_artifact(
                    asset.run_relative_root,
                    audio.relative_path,
                    audio.sha256,
                )
                audio_sha256 = audio.sha256
        return PreparedMedia(
            source=probed,
            proxy_path=proxy_path,
            proxy_sha256=proxy_sha256,
            proxy_size_bytes=proxy_size_bytes,
            audio_path=audio_path,
            audio_sha256=audio_sha256,
            subtitle=subtitle,
            warnings=tuple(dict.fromkeys(warnings)),
        )

    def _verified_source(self, asset: RegisteredAsset) -> Path:
        source = verified_run_file(
            self._runtime_root,
            asset.run_relative_root,
            asset.source_path,
            expected_sha256=asset.source_sha256,
            digest=_sha256_file,
            message="原始视频必须位于当前 Run 根目录内",
        )
        if source.stat().st_size != asset.source_size_bytes:
            raise VideoDemoError(ErrorCode.VIDEO_DIGEST_MISMATCH, "原始视频大小校验失败")
        return source

    def _visual_input(self, probed: ProbedAsset, source: Path) -> tuple[Path, str, int]:
        asset = probed.asset
        _LOGGER.info(
            "视觉输入选择 mode=SOURCE source_mime=%s codec=%s reason=NONE",
            asset.source_mime,
            probed.manifest.video_stream.codec_name,
        )
        return source, asset.source_sha256, asset.source_size_bytes

    def _select_subtitle(
        self,
        client: TranscodeClient,
        probed: ProbedAsset,
        warnings: list[str],
    ) -> ParsedSubtitle | None:
        asset = probed.asset
        candidates = rank_text_subtitle_streams(
            probed.manifest.subtitle_streams,
            asset.config.language_hints,
        )
        for stream in candidates:
            try:
                artifact = client.extract_subtitle(
                    asset.source_path,
                    asset.run_relative_root,
                    stream,
                    duration_ms=probed.duration_ms,
                )
                parsed = parse_webvtt(
                    self._runtime_root,
                    asset.run_relative_root,
                    artifact,
                    duration_ms=probed.duration_ms,
                )
            except SubtitleTrackRejected as error:
                warnings.append(
                    f"SUBTITLE_TRACK_REJECTED:{stream.index}:{error.reason}"
                )
                continue
            except VideoDemoError as error:
                if error.code != ErrorCode.VIDEO_PROCESS_FAILED:
                    raise
                warnings.append(
                    f"SUBTITLE_TRACK_REJECTED:{stream.index}:DECODE_FAILED"
                )
                continue
            if is_subtitle_eligible(parsed, duration_ms=probed.duration_ms):
                return parsed
            warnings.append(f"SUBTITLE_TRACK_REJECTED:{stream.index}:INCOMPLETE")
        return None

    def _verified_run_artifact(
        self,
        run_relative_root: Path,
        relative_path: str,
        expected_sha256: str,
    ) -> Path:
        run_root = safe_runtime_path(self._runtime_root, run_relative_root)
        artifact = safe_runtime_path(self._runtime_root, Path(relative_path))
        if not artifact.is_relative_to(run_root) or artifact.is_symlink() or not artifact.is_file():
            raise VideoDemoError(
                ErrorCode.WORKSPACE_PATH_ESCAPE,
                "媒体产物必须位于当前运行目录内",
            )
        if _sha256_file(artifact) != expected_sha256:
            raise VideoDemoError(ErrorCode.VIDEO_DIGEST_MISMATCH, "媒体产物摘要校验失败")
        return artifact


def build_ffprobe_factory(
    settings_workspace_root: Path,
    executable: Path,
) -> Callable[[], ProbeClient]:
    return lambda: FFprobeClient.from_path(
        executable,
        workspace_root=settings_workspace_root,
    )


def build_ffmpeg_factory(
    settings_workspace_root: Path,
    runtime_root: Path,
    executable: Path,
    *,
    max_output_bytes: int = 4 * 1024 * 1024 * 1024,
    required_free_bytes: int = 512 * 1024 * 1024,
    timeout_seconds: int = 1_800,
    visual_proxy_max_edge: int = 1_280,
) -> Callable[[Callable[[], bool]], TranscodeClient]:
    return lambda is_cancel_requested: FFmpegTranscoder.from_path(
        executable,
        runtime_root,
        workspace_root=settings_workspace_root,
        is_cancel_requested=is_cancel_requested,
        limits=TranscodeLimits(
            max_output_bytes=max_output_bytes,
            required_free_bytes=required_free_bytes,
            timeout_seconds=timeout_seconds,
        ),
        proxy_max_edge=visual_proxy_max_edge,
    )


def _supported_mime(value: str) -> SupportedMime:
    if value not in _SUPPORTED_MIME_TYPES:
        raise VideoDemoError(ErrorCode.VIDEO_FORMAT_UNSUPPORTED, "不支持该视频格式")
    return cast(SupportedMime, value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
