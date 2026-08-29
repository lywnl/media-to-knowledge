from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ValidationError

from video_demo.application.document_publication import ResultWriteFence, scope_key
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.persistence.database import Database
from video_demo.persistence.media_repositories import MediaRunRepository
from video_demo.persistence.models import (
    AudioUnderstandingRunModel,
    ImageUnderstandingRunModel,
    RunStatusValue,
)
from video_demo.persistence.repositories import JobRepository, Scope
from video_demo.storage.artifacts import (
    ArtifactBytesReceipt,
    AtomicArtifactStore,
)

MediaRunModel = type[AudioUnderstandingRunModel] | type[ImageUnderstandingRunModel]


MediaResult = BaseModel


@dataclass(frozen=True, slots=True)
class MediaResultPublication:
    result: MediaResult
    document: bytes


class MediaPublicationService:
    """音频/图片独立结果的原子双制品发布与读取。"""

    def __init__(
        self,
        database: Database,
        artifact_store: AtomicArtifactStore,
        *,
        run_model: MediaRunModel,
        result_type: type[BaseModel],
        render: Callable[[Any], Any],
        resource_type: Literal["AUDIO_UNDERSTANDING_RUN", "IMAGE_UNDERSTANDING_RUN"],
        not_found_code: ErrorCode,
        max_document_bytes: int = 16 * 1024 * 1024,
        max_bundle_bytes: int = 64 * 1024 * 1024,
    ) -> None:
        self._database = database
        self._store = artifact_store
        self._run_model = run_model
        self._result_type = result_type
        self._render = render
        self._resource_type = resource_type
        self._not_found_code = not_found_code
        self._max_document_bytes = max_document_bytes
        self._max_bundle_bytes = max_bundle_bytes

    def persist(
        self,
        scope: Scope,
        result: BaseModel,
        *,
        document: Any,
        status: Literal["SUCCEEDED", "PARTIAL_SUCCEEDED"],
        warnings: tuple[str, ...],
        fence: ResultWriteFence,
    ) -> None:
        validated: Any = self._result_type.model_validate(
            result.model_dump(mode="json", exclude_computed_fields=True),
        )
        rendered = self._render(validated)
        if rendered != document:
            raise ValueError("媒体 Markdown 必须由同一结果确定性渲染")
        run_root = Path("runs") / scope_key(scope) / validated.run_id
        result_root = run_root / "result"
        prefix = "audio" if self._resource_type.startswith("AUDIO") else "image"
        document_path = result_root / f"{prefix}-document-{uuid.uuid4().hex}.md"
        payload_path = result_root / f"{prefix}-result-{uuid.uuid4().hex}.json"
        payload = {
            "result": validated.model_dump(mode="json", exclude_computed_fields=True),
            "status": status,
            "warnings": list(dict.fromkeys(warnings)),
        }
        upstream_sha = str(validated.asset_sha256)
        document_receipt = self._store.write_bytes(
            document_path,
            rendered.content,
            max_bytes=self._max_document_bytes,
            exclusive=True,
        )
        bundle_receipt = self._store.write_json(
            payload_path,
            payload,
            schema_version="1.0.0",
            upstream_sha256=upstream_sha,
            file_mode=0o600,
            exclusive=True,
            max_bytes=self._max_bundle_bytes,
        )
        try:
            with self._database.session() as session:
                published = JobRepository(session).mark_result_published(
                    fence.job_pk,
                    fence.worker_id,
                    attempt_count=fence.attempt_count,
                    scope=scope,
                    run_id=validated.run_id,
                    resource_type=self._resource_type,
                )
                if not published:
                    raise VideoDemoError(ErrorCode.JOB_LEASE_LOST, "结果已由其他发布者写入")
                run = MediaRunRepository(session, self._run_model).get(scope, validated.run_id)
                if run is None:
                    raise VideoDemoError(self._not_found_code, "媒体运行不存在")
                run.status = RunStatusValue(status)
                run.current_stage = "RESULT"
                run.warning_codes = list(dict.fromkeys(warnings))
                run.error_code = None
                run.artifact_relative_path = bundle_receipt.relative_path
                run.artifact_sha256 = bundle_receipt.sha256
                run.document_relative_path = document_receipt.relative_path
                run.document_sha256 = document_receipt.sha256
                run.document_size_bytes = document_receipt.size_bytes
        except BaseException:
            self._store.discard_artifact(bundle_receipt, max_bytes=self._max_bundle_bytes)
            self._store.discard_bytes(document_receipt)
            raise

    def get(self, scope: Scope, run_id: str) -> MediaResultPublication:
        with self._database.session() as session:
            run = MediaRunRepository(session, self._run_model).get(scope, run_id)
            if run is None:
                raise VideoDemoError(self._not_found_code, "媒体运行不存在")
            if run.status not in {RunStatusValue.SUCCEEDED, RunStatusValue.PARTIAL_SUCCEEDED}:
                raise VideoDemoError(ErrorCode.VIDEO_RESULT_NOT_READY, "媒体理解结果尚未就绪")
            if not all(
                (
                    run.artifact_relative_path,
                    run.artifact_sha256,
                    run.document_relative_path,
                    run.document_sha256,
                    run.document_size_bytes,
                )
            ):
                raise VideoDemoError(ErrorCode.ARTIFACT_SCHEMA_INVALID, "媒体制品指针不完整")
            document = ArtifactBytesReceipt(
                relative_path=run.document_relative_path,
                sha256=run.document_sha256,
                size_bytes=run.document_size_bytes,
            )
        try:
            encoded_bundle, bundle_receipt = self._store.inspect_artifact_bytes(
                self._validated_result_path(scope, run_id, run.artifact_relative_path, ".json"),
                max_bytes=self._max_bundle_bytes,
            )
            if bundle_receipt.sha256 != run.artifact_sha256:
                raise VideoDemoError(ErrorCode.ARTIFACT_DIGEST_MISMATCH, "媒体 bundle 摘要不一致")
            envelope = json.loads(encoded_bundle)
            if not isinstance(envelope, dict) or envelope.get("schema_version") != "1.0.0":
                raise ValueError("媒体 bundle envelope 非法")
            upstream_sha = envelope.get("upstream_sha256")
            payload = envelope.get("payload")
            if not isinstance(upstream_sha, str) or not isinstance(payload, dict):
                raise ValueError("媒体 bundle 必须是对象")
            result: Any = self._result_type.model_validate(payload.get("result"))
            if getattr(result, "asset_sha256", None) != upstream_sha:
                raise ValueError("媒体结果上游摘要不一致")
            encoded_document = self._store.read_verified_bytes(
                self._validated_document_receipt(scope, run_id, document),
                max_bytes=self._max_document_bytes,
            )
            rendered = self._render(result)
            if encoded_document != rendered.content:
                raise ValueError("媒体 Markdown 与结果不一致")
            return MediaResultPublication(result=result, document=encoded_document)
        except VideoDemoError:
            raise
        except (ValidationError, ValueError, TypeError, OSError) as error:
            raise VideoDemoError(ErrorCode.ARTIFACT_SCHEMA_INVALID, "媒体结果制品非法") from error

    @staticmethod
    def _validated_result_path(
        scope: Scope,
        run_id: str,
        relative_path: str,
        suffix: str,
    ) -> Path:
        path = Path(relative_path)
        expected_parent = Path("runs") / scope_key(scope) / run_id / "result"
        if path.is_absolute() or ".." in path.parts or path.suffix != suffix:
            raise VideoDemoError(ErrorCode.ARTIFACT_SCHEMA_INVALID, "媒体 bundle 路径非法")
        if path.parent != expected_parent or len(path.name) > 256:
            raise VideoDemoError(ErrorCode.ARTIFACT_SCHEMA_INVALID, "媒体 bundle 路径非法")
        return path

    @classmethod
    def _validated_document_receipt(
        cls,
        scope: Scope,
        run_id: str,
        receipt: ArtifactBytesReceipt,
    ) -> ArtifactBytesReceipt:
        cls._validated_result_path(scope, run_id, receipt.relative_path, ".md")
        return receipt
