from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import Protocol, cast

from pydantic import ValidationError

from video_demo.application.pipeline import (
    PipelineContext,
    PipelineRunConfig,
    PreparedMedia,
    ProbedAsset,
    RegisteredAsset,
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
    ProxyVideoArtifact,
    SubtitleArtifact,
)
from video_demo.persistence.database import Database
from video_demo.persistence.repositories import VideoObjectRepository, VideoRunRepository
from video_demo.storage.object_store import LocalVideoObjectStore, VideoObjectRecord
from video_demo.storage.workspace import safe_runtime_path, verified_mp4_file

_SUPPORTED_MIME_TYPES = frozenset(
    {"video/mp4", "video/quicktime", "video/x-matroska", "video/webm"},
)


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
    def create_proxy(self, source: Path, run_relative_root: Path) -> ProxyVideoArtifact: ...

    def extract_audio(
        self,
        source: Path,
        run_relative_root: Path,
        *,
        has_audio: bool,
    ) -> AudioArtifact | NoAudioArtifact: ...

    def extract_subtitle(
        self,
        source: Path,
        run_relative_root: Path,
        stream: SubtitleStream,
    ) -> SubtitleArtifact: ...


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
            return PipelineRunConfig.model_validate(snapshot)
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
        proxy = client.create_proxy(asset.source_path, asset.run_relative_root)
        proxy_path = verified_mp4_file(
            self._runtime_root,
            asset.run_relative_root,
            Path(proxy.relative_path),
            expected_sha256=proxy.sha256,
            expected_size_bytes=proxy.size_bytes,
            max_size_bytes=self._max_proxy_bytes,
            message="代理视频必须位于当前运行目录内",
        )
        warnings = list(probed.warnings)
        subtitle = self._select_subtitle(client, probed, warnings)
        if subtitle is not None:
            audio_path = None
            audio_sha256 = None
        else:
            audio = client.extract_audio(
                asset.source_path,
                asset.run_relative_root,
                has_audio=bool(probed.manifest.audio_streams),
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
            proxy_sha256=proxy.sha256,
            proxy_size_bytes=proxy.size_bytes,
            audio_path=audio_path,
            audio_sha256=audio_sha256,
            subtitle=subtitle,
            warnings=tuple(dict.fromkeys(warnings)),
        )

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
) -> Callable[[Callable[[], bool]], TranscodeClient]:
    return lambda is_cancel_requested: FFmpegTranscoder.from_path(
        executable,
        runtime_root,
        workspace_root=settings_workspace_root,
        is_cancel_requested=is_cancel_requested,
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
