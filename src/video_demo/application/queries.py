from __future__ import annotations

import base64
import hashlib
import json
import uuid
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from pydantic import TypeAdapter
from sqlalchemy import select

from video_demo.domain.evidence import EvidenceItem, KeyframeEvidence
from video_demo.domain.result import VideoUnderstandingResult, validate_evidence_references
from video_demo.domain.result_artifact import ResultArtifactPayload
from video_demo.domain.run import RunStatus
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.persistence.database import Database
from video_demo.persistence.models import JobModel, JobStatus, RunStatusValue, VideoAssetModel
from video_demo.persistence.repositories import (
    JobRepository,
    ResultRepository,
    Scope,
    VideoRunRepository,
)
from video_demo.storage.artifacts import ArtifactReceipt, AtomicArtifactStore
from video_demo.storage.workspace import safe_runtime_path

_EVIDENCE_ADAPTER = TypeAdapter(tuple[EvidenceItem, ...])


@dataclass(frozen=True, slots=True)
class EvidencePage:
    items: tuple[EvidenceItem, ...]
    next_cursor: str | None


@dataclass(frozen=True, slots=True)
class KeyframeContent:
    content: bytes
    mime_type: str


@dataclass(frozen=True, slots=True)
class ResultRunMetadata:
    status: RunStatus
    warnings: tuple[str, ...]
    stage_metrics: dict[str, int]


@dataclass(frozen=True, slots=True)
class ResultWriteFence:
    job_pk: int
    worker_id: str
    attempt_count: int


class ResultQueryService:
    def __init__(
        self,
        database: Database,
        artifact_store: AtomicArtifactStore,
        *,
        max_evidence_items: int = 100_000,
    ) -> None:
        if max_evidence_items < 1:
            raise ValueError("max_evidence_items 必须大于等于 1")
        self._database = database
        self._artifact_store = artifact_store
        self._max_evidence_items = max_evidence_items

    @staticmethod
    def scope_key(scope: Scope) -> str:
        encoded = "\x00".join(
            (scope.tenant_id, scope.application_id, scope.knowledge_base_id),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:24]

    def persist(
        self,
        scope: Scope,
        result: VideoUnderstandingResult,
        *,
        evidence: tuple[EvidenceItem, ...],
        stage_metrics: dict[str, int],
        status: RunStatus,
        fence: ResultWriteFence,
        warnings: tuple[str, ...] = (),
    ) -> None:
        if fence is None:
            raise ValueError("fence 不能为空")
        if status not in (RunStatus.SUCCEEDED, RunStatus.PARTIAL_SUCCEEDED):
            raise ValueError("只能持久化成功或部分成功结果")
        self._validate_evidence_count(evidence)
        validate_evidence_references(result, evidence)
        self._validate_result_target(scope, result)
        payload = ResultArtifactPayload(
            result=result,
            evidence=evidence,
            stage_metrics=stage_metrics,
            status=status.value,
            warnings=warnings,
        ).model_dump(mode="json", exclude_computed_fields=True)
        payload_digest = hashlib.sha256(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8"),
        ).hexdigest()[:24]
        relative_path = (
            Path("runs")
            / self.scope_key(scope)
            / result.run_id
            / "result"
            / f"bundle-{payload_digest}-{uuid.uuid4().hex}.json"
        )
        with self._database.session() as session:
            self._require_active_fence(session, fence)
        receipt = self._artifact_store.write_json(
            relative_path,
            payload,
            schema_version=result.schema_version,
            upstream_sha256=result.asset_sha256,
        )
        try:
            with self._database.session() as session:
                published = JobRepository(session).mark_result_published(
                    fence.job_pk,
                    fence.worker_id,
                    attempt_count=fence.attempt_count,
                    scope=scope,
                    run_id=result.run_id,
                )
                if not published:
                    raise VideoDemoError(ErrorCode.JOB_LEASE_LOST, "结果已由其他发布者写入")
                run = VideoRunRepository(session).get(scope, result.run_id)
                assert run is not None
                ResultRepository(session).replace(scope, result)
                run.status = RunStatusValue(status.value)
                run.current_stage = "RESULT"
                run.warning_codes = list(dict.fromkeys(warnings))
                run.error_code = None
                run.artifact_manifest_relative_path = receipt.relative_path
                run.artifact_manifest_sha256 = receipt.sha256
        except BaseException:
            with suppress(OSError, VideoDemoError):
                self._artifact_store.discard(receipt)
            raise

    @staticmethod
    def _require_active_fence(session: object, fence: ResultWriteFence) -> None:
        job = session.get(JobModel, fence.job_pk)  # type: ignore[attr-defined]
        if (
            job is not None
            and job.attempt_count == fence.attempt_count
            and job.status == JobStatus.CANCELLED
            and job.cancel_requested
        ):
            raise VideoDemoError(ErrorCode.JOB_CANCELLED, "任务已请求取消")
        if (
            job is None
            or job.worker_id != fence.worker_id
            or job.attempt_count != fence.attempt_count
            or job.status != JobStatus.RUNNING
            or job.lease_expires_at is None
            or job.lease_expires_at <= datetime.now(UTC)
        ):
            raise VideoDemoError(ErrorCode.JOB_LEASE_LOST, "结果写入租约已丢失")
        if job.cancel_requested:
            raise VideoDemoError(ErrorCode.JOB_CANCELLED, "任务已请求取消")

    def _validate_result_target(
        self,
        scope: Scope,
        result: VideoUnderstandingResult,
    ) -> None:
        with self._database.session() as session:
            run = VideoRunRepository(session).get(scope, result.run_id)
            if run is None:
                raise VideoDemoError(ErrorCode.VIDEO_RUN_NOT_FOUND, "视频理解运行不存在")
            asset_sha = self._asset_sha_for_run(session, scope, run.asset_id)
            if asset_sha != result.asset_sha256:
                raise VideoDemoError(
                    ErrorCode.VIDEO_DIGEST_MISMATCH,
                    "结果资产摘要与运行不匹配",
                )

    def get_result(self, scope: Scope, run_id: str) -> VideoUnderstandingResult:
        with self._database.session() as session:
            run = VideoRunRepository(session).get(scope, run_id)
            if run is None:
                raise VideoDemoError(ErrorCode.VIDEO_RUN_NOT_FOUND, "视频理解运行不存在")
            asset_sha = self._asset_sha_for_run(session, scope, run.asset_id)
            result = ResultRepository(session).get(scope, run_id, asset_sha)
            if result is None:
                raise VideoDemoError(ErrorCode.VIDEO_RESULT_NOT_READY, "视频理解结果尚未就绪")
            return result

    def get_evidence(
        self,
        scope: Scope,
        run_id: str,
        *,
        evidence_type: str | None = None,
        start_ms: int | None = None,
        cursor: str | None = None,
        limit: int = 50,
    ) -> EvidencePage:
        if not 1 <= limit <= 100:
            raise ValueError("limit 必须在 1 到 100 之间")
        bundle = self._read_bundle(scope, run_id)
        evidence_payload = bundle.get("evidence")
        self._validate_evidence_count(evidence_payload)
        evidence = _EVIDENCE_ADAPTER.validate_python(evidence_payload)
        filtered = tuple(
            item
            for item in evidence
            if (evidence_type is None or item.evidence_type == evidence_type)
            and (start_ms is None or item.start_ms >= start_ms)
        )
        ordered = tuple(
            sorted(
                filtered,
                key=lambda item: (item.start_ms, item.end_ms, item.evidence_type, item.evidence_id),
            ),
        )
        filter_key = self._filter_key(run_id, evidence_type, start_ms)
        offset = self._decode_cursor(cursor, filter_key) if cursor else 0
        if offset > len(ordered):
            raise VideoDemoError(ErrorCode.INVALID_EVIDENCE_CURSOR, "证据游标超出范围")
        items = ordered[offset : offset + limit]
        next_offset = offset + len(items)
        next_cursor = (
            self._encode_cursor(next_offset, filter_key)
            if next_offset < len(ordered)
            else None
        )
        return EvidencePage(items=items, next_cursor=next_cursor)

    def get_keyframe(self, scope: Scope, run_id: str, keyframe_id: str) -> KeyframeContent:
        bundle = self._read_bundle(scope, run_id)
        evidence_payload = bundle.get("evidence")
        self._validate_evidence_count(evidence_payload)
        evidence = _EVIDENCE_ADAPTER.validate_python(evidence_payload)
        keyframe = next(
            (
                item
                for item in evidence
                if isinstance(item, KeyframeEvidence) and item.keyframe_id == keyframe_id
            ),
            None,
        )
        if keyframe is None:
            raise VideoDemoError(ErrorCode.KEYFRAME_NOT_FOUND, "关键帧不存在")
        relative_path = Path(keyframe.relative_path)
        runtime_root = self._artifact_store.runtime_root.resolve(strict=True)
        expected_root = (
            runtime_root / "runs" / self.scope_key(scope) / run_id
        ).resolve(strict=False)
        resolved_path = (runtime_root / relative_path).resolve(strict=False)
        if not resolved_path.is_relative_to(expected_root):
            raise VideoDemoError(ErrorCode.WORKSPACE_PATH_ESCAPE, "关键帧不属于当前运行")
        path = safe_runtime_path(self._artifact_store.runtime_root, relative_path)
        if path.is_symlink() or not path.is_file():
            raise VideoDemoError(ErrorCode.KEYFRAME_NOT_FOUND, "关键帧不存在")
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != keyframe.sha256:
            raise VideoDemoError(ErrorCode.ARTIFACT_DIGEST_MISMATCH, "关键帧摘要不匹配")
        if not _matches_mime_signature(content, keyframe.mime_type):
            raise VideoDemoError(ErrorCode.ARTIFACT_SCHEMA_INVALID, "关键帧 MIME 与内容不一致")
        return KeyframeContent(content=content, mime_type=keyframe.mime_type)

    def get_run_metadata(self, scope: Scope, run_id: str) -> ResultRunMetadata:
        bundle = self._read_bundle(scope, run_id)
        status = bundle.get("status")
        warnings = bundle.get("warnings")
        metrics = bundle.get("stage_metrics")
        if (
            not isinstance(status, str)
            or not isinstance(warnings, list)
            or not isinstance(metrics, dict)
        ):
            raise VideoDemoError(ErrorCode.ARTIFACT_SCHEMA_INVALID, "结果 bundle 元数据非法")
        return ResultRunMetadata(
            status=RunStatus(status),
            warnings=tuple(str(item) for item in warnings),
            stage_metrics={str(key): int(value) for key, value in metrics.items()},
        )

    def _read_bundle(self, scope: Scope, run_id: str) -> dict[str, object]:
        with self._database.session() as session:
            run = VideoRunRepository(session).get(scope, run_id)
            if run is None:
                raise VideoDemoError(ErrorCode.VIDEO_RUN_NOT_FOUND, "视频理解运行不存在")
            if run.artifact_manifest_relative_path is None or run.artifact_manifest_sha256 is None:
                raise VideoDemoError(ErrorCode.VIDEO_RESULT_NOT_READY, "视频理解结果尚未就绪")
            asset_sha = self._asset_sha_for_run(session, scope, run.asset_id)
            receipt = ArtifactReceipt(
                relative_path=run.artifact_manifest_relative_path,
                schema_version="1.0.0",
                sha256=run.artifact_manifest_sha256,
                upstream_sha256=asset_sha,
            )
        payload = self._artifact_store.read_verified_json(receipt)
        if not isinstance(payload, dict):
            raise VideoDemoError(ErrorCode.ARTIFACT_SCHEMA_INVALID, "结果 bundle 必须是对象")
        return payload

    @staticmethod
    def _asset_sha_for_run(session: object, scope: Scope, asset_id: str) -> str:
        asset = session.scalar(  # type: ignore[attr-defined]
            select(VideoAssetModel).where(
                VideoAssetModel.tenant_id == scope.tenant_id,
                VideoAssetModel.application_id == scope.application_id,
                VideoAssetModel.knowledge_base_id == scope.knowledge_base_id,
                VideoAssetModel.asset_id == asset_id,
            ),
        )
        if asset is None:
            raise VideoDemoError(ErrorCode.VIDEO_RUN_NOT_FOUND, "视频资产不存在")
        return str(asset.source_sha256)

    @staticmethod
    def _filter_key(run_id: str, evidence_type: str | None, start_ms: int | None) -> str:
        encoded = json.dumps(
            {"run_id": run_id, "evidence_type": evidence_type, "start_ms": start_ms},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:16]

    def _validate_evidence_count(self, evidence: object) -> None:
        if not isinstance(evidence, (list, tuple)):
            raise VideoDemoError(ErrorCode.ARTIFACT_SCHEMA_INVALID, "证据集合必须是数组")
        if len(evidence) > self._max_evidence_items:
            raise VideoDemoError(
                ErrorCode.ARTIFACT_SCHEMA_INVALID,
                "证据数量超过配置上限",
                {"max_evidence_items": self._max_evidence_items},
            )

    @staticmethod
    def _encode_cursor(offset: int, filter_key: str) -> str:
        data = json.dumps(
            {"offset": offset, "filter": filter_key},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        checksum = hashlib.sha256(data).hexdigest()[:16]
        return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=") + "." + checksum

    @staticmethod
    def _decode_cursor(cursor: str, filter_key: str) -> int:
        try:
            encoded, checksum = cursor.split(".", 1)
            data = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
            if hashlib.sha256(data).hexdigest()[:16] != checksum:
                raise ValueError("摘要不匹配")
            payload = json.loads(data)
            if payload.get("filter") != filter_key:
                raise ValueError("筛选条件不匹配")
            offset = payload.get("offset")
            if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
                raise ValueError("偏移非法")
            return cast(int, offset)
        except (UnicodeDecodeError, ValueError, json.JSONDecodeError, AttributeError) as error:
            raise VideoDemoError(ErrorCode.INVALID_EVIDENCE_CURSOR, "证据游标非法") from error


def _matches_mime_signature(content: bytes, mime_type: str) -> bool:
    if mime_type == "image/png":
        return content.startswith(b"\x89PNG\r\n\x1a\n")
    if mime_type == "image/jpeg":
        return content.startswith(b"\xff\xd8\xff")
    return False
