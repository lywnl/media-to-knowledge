"""音频结果的独立原子发布与读取。"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import ValidationError

from video_demo.application.audio_rendering import (
    RenderedAudioDocument,
    render_audio_markdown,
)
from video_demo.application.publication_contracts import ResultWriteFence, scope_key
from video_demo.domain.audio_document import AudioUnderstandingResult
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.persistence.audio_document_repository import AudioResultRepository
from video_demo.persistence.database import Database
from video_demo.persistence.media_repositories import MediaRunRepository
from video_demo.persistence.models import AudioUnderstandingRunModel, RunStatusValue
from video_demo.persistence.repositories import JobRepository
from video_demo.persistence.scope import Scope
from video_demo.storage.artifacts import (
    ArtifactBytesReceipt,
    ArtifactReceipt,
    AtomicArtifactStore,
)


@dataclass(frozen=True, slots=True)
class AudioResultPublication:
    result: AudioUnderstandingResult
    document: bytes


class AudioPublicationService:
    """发布并读取音频结果、音频结果行和 Markdown 三方一致事实。"""

    def __init__(
        self,
        database: Database,
        artifact_store: AtomicArtifactStore,
        *,
        max_document_bytes: int = 16 * 1024 * 1024,
        max_bundle_bytes: int = 64 * 1024 * 1024,
    ) -> None:
        self._database = database
        self._store = artifact_store
        self._max_document_bytes = max_document_bytes
        self._max_bundle_bytes = max_bundle_bytes

    @property
    def database(self) -> Database:
        return self._database

    def persist(
        self,
        scope: Scope,
        result: AudioUnderstandingResult,
        *,
        document: RenderedAudioDocument,
        status: Literal["SUCCEEDED", "PARTIAL_SUCCEEDED"],
        warnings: tuple[str, ...],
        fence: ResultWriteFence,
    ) -> None:
        validated = AudioUnderstandingResult.model_validate(
            result.model_dump(mode="json", exclude_computed_fields=True),
        )
        rendered = render_audio_markdown(validated)
        if rendered != document:
            raise ValueError("音频 Markdown 必须由同一结果确定性渲染")
        if len(warnings) != len(set(warnings)):
            raise ValueError("音频 warnings 不得重复")
        result_root = Path("runs") / scope_key(scope) / validated.run_id / "result"
        document_receipt: ArtifactBytesReceipt | None = None
        bundle_receipt: ArtifactReceipt | None = None
        try:
            bundle_receipt = self._store.write_json(
                result_root / f"audio-result-{uuid.uuid4().hex}.json",
                {
                    "result": validated.model_dump(mode="json", exclude_computed_fields=True),
                    "status": status,
                    "warnings": list(warnings),
                },
                schema_version="1.0.0",
                upstream_sha256=validated.asset_sha256,
                file_mode=0o600,
                exclusive=True,
                max_bytes=self._max_bundle_bytes,
            )
            document_receipt = self._store.write_bytes(
                result_root / f"audio-document-{uuid.uuid4().hex}.md",
                rendered.content,
                max_bytes=self._max_document_bytes,
                exclusive=True,
            )
            with self._database.session() as session:
                run = MediaRunRepository(session, AudioUnderstandingRunModel).get(
                    scope,
                    validated.run_id,
                )
                if run is None:
                    raise VideoDemoError(ErrorCode.AUDIO_RUN_NOT_FOUND, "音频运行不存在")
                AudioResultRepository(session).replace(scope, validated)
                run.status = RunStatusValue(status)
                run.current_stage = "RESULT"
                run.warning_codes = list(warnings)
                run.error_code = None
                run.artifact_relative_path = bundle_receipt.relative_path
                run.artifact_sha256 = bundle_receipt.sha256
                run.document_relative_path = document_receipt.relative_path
                run.document_sha256 = document_receipt.sha256
                run.document_size_bytes = document_receipt.size_bytes
                published = JobRepository(session).mark_result_published(
                    fence.job_pk,
                    fence.worker_id,
                    attempt_count=fence.attempt_count,
                    scope=scope,
                    run_id=validated.run_id,
                    resource_type="AUDIO_UNDERSTANDING_RUN",
                )
                if not published:
                    raise VideoDemoError(ErrorCode.JOB_LEASE_LOST, "音频结果已由其他发布者写入")
        except BaseException:
            if bundle_receipt is not None:
                self._store.discard_artifact(bundle_receipt, max_bytes=self._max_bundle_bytes)
            if document_receipt is not None:
                self._store.discard_bytes(document_receipt)
            raise

    def get(self, scope: Scope, run_id: str) -> AudioResultPublication:
        with self._database.session() as session:
            run = MediaRunRepository(session, AudioUnderstandingRunModel).get(scope, run_id)
            if run is None:
                raise VideoDemoError(ErrorCode.AUDIO_RUN_NOT_FOUND, "音频运行不存在")
            if run.status not in {RunStatusValue.SUCCEEDED, RunStatusValue.PARTIAL_SUCCEEDED}:
                raise VideoDemoError(ErrorCode.AUDIO_RESULT_NOT_READY, "音频理解结果尚未就绪")
            if not all(
                (
                    run.artifact_relative_path,
                    run.artifact_sha256,
                    run.document_relative_path,
                    run.document_sha256,
                    run.document_size_bytes,
                ),
            ):
                raise VideoDemoError(ErrorCode.ARTIFACT_SCHEMA_INVALID, "音频制品指针不完整")
            artifact_path = run.artifact_relative_path
            artifact_sha256 = run.artifact_sha256
            document = ArtifactBytesReceipt(
                relative_path=run.document_relative_path,
                sha256=run.document_sha256,
                size_bytes=run.document_size_bytes,
            )
        try:
            encoded_bundle, bundle_receipt = self._store.inspect_artifact_bytes(
                _validated_result_path(scope, run_id, artifact_path, ".json"),
                max_bytes=self._max_bundle_bytes,
            )
            if bundle_receipt.sha256 != artifact_sha256:
                raise VideoDemoError(ErrorCode.ARTIFACT_DIGEST_MISMATCH, "音频 bundle 摘要不一致")
            envelope = json.loads(encoded_bundle)
            if not isinstance(envelope, dict) or envelope.get("schema_version") != "1.0.0":
                raise ValueError("音频 bundle envelope 非法")
            upstream_sha256 = envelope.get("upstream_sha256")
            payload = envelope.get("payload")
            if not isinstance(upstream_sha256, str) or not isinstance(payload, dict):
                raise ValueError("音频 bundle 必须是对象")
            if payload.get("status") not in {"SUCCEEDED", "PARTIAL_SUCCEEDED"}:
                raise ValueError("音频 bundle 状态非法")
            bundle_warnings = payload.get("warnings")
            if not isinstance(bundle_warnings, list) or any(
                not isinstance(item, str) for item in bundle_warnings
            ):
                raise ValueError("音频 bundle 告警非法")
            result = AudioUnderstandingResult.model_validate(payload.get("result"))
            if result.run_id != run_id:
                raise ValueError("音频结果运行标识与请求不一致")
            if result.asset_sha256 != upstream_sha256:
                raise ValueError("音频结果上游摘要不一致")
            encoded_document = self._store.read_verified_bytes(
                _validated_document_receipt(scope, run_id, document),
                max_bytes=self._max_document_bytes,
            )
            if encoded_document != render_audio_markdown(result).content:
                raise ValueError("音频 Markdown 与结果不一致")
            with self._database.session() as session:
                run = MediaRunRepository(session, AudioUnderstandingRunModel).get(scope, run_id)
                if run is None or run.status.value != payload["status"]:
                    raise ValueError("音频运行状态与 bundle 不一致")
                if tuple(run.warning_codes) != tuple(bundle_warnings):
                    raise ValueError("音频运行告警与 bundle 不一致")
            return AudioResultPublication(result=result, document=encoded_document)
        except VideoDemoError:
            raise
        except (ValidationError, ValueError, TypeError, OSError) as error:
            raise VideoDemoError(ErrorCode.ARTIFACT_SCHEMA_INVALID, "音频结果制品非法") from error


def _validated_result_path(
    scope: Scope,
    run_id: str,
    relative_path: str,
    suffix: str,
) -> Path:
    path = Path(relative_path)
    expected_parent = Path("runs") / scope_key(scope) / run_id / "result"
    if path.is_absolute() or ".." in path.parts or path.suffix != suffix:
        raise VideoDemoError(ErrorCode.ARTIFACT_SCHEMA_INVALID, "音频 bundle 路径非法")
    if path.parent != expected_parent or len(path.name) > 256:
        raise VideoDemoError(ErrorCode.ARTIFACT_SCHEMA_INVALID, "音频 bundle 路径非法")
    return path


def _validated_document_receipt(
    scope: Scope,
    run_id: str,
    receipt: ArtifactBytesReceipt,
) -> ArtifactBytesReceipt:
    _validated_result_path(scope, run_id, receipt.relative_path, ".md")
    return receipt
