from __future__ import annotations

import base64
import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from pydantic import TypeAdapter, ValidationError

from video_demo.application.document_publication import (
    DocumentPublicationService,
    VisualCleaner,
    scope_key,
)
from video_demo.application.document_publication import (
    ResultWriteFence as ResultWriteFence,
)
from video_demo.application.document_rendering import RenderedDocument
from video_demo.domain.document import TranscriptSource, VideoUnderstandingResult
from video_demo.domain.document_artifact import DocumentArtifactPayload
from video_demo.domain.evidence import DocumentEvidenceItem, KeyframeEvidence
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.persistence.database import Database
from video_demo.persistence.document_repository import ResultRepository
from video_demo.persistence.repositories import Scope
from video_demo.storage.artifacts import AtomicArtifactStore
from video_demo.storage.workspace import safe_runtime_path

_EVIDENCE_ADAPTER = TypeAdapter(tuple[DocumentEvidenceItem, ...])
_JPEG_PREFIX = b"\xff\xd8\xff"
_JPEG_SUFFIX = b"\xff\xd9"


@dataclass(frozen=True, slots=True)
class EvidencePage:
    items: tuple[DocumentEvidenceItem, ...]
    next_cursor: str | None


@dataclass(frozen=True, slots=True)
class KeyframeContent:
    content: bytes
    mime_type: str


@dataclass(frozen=True, slots=True)
class ResultRunMetadata:
    status: str
    warnings: tuple[str, ...]
    stage_metrics: dict[str, int]
    model_metrics: dict[str, int]
    stage_cache_hits: tuple[str, ...] = ()


class ResultQueryService:
    """唯一 3.0 结果发布与四接口查询服务。"""

    def __init__(
        self,
        database: Database,
        artifact_store: AtomicArtifactStore,
        *,
        max_evidence_items: int = 25_000,
        max_keyframe_bytes: int = 5 * 1024 * 1024,
        max_document_bytes: int = 16 * 1024 * 1024,
        max_bundle_bytes: int = 64 * 1024 * 1024,
        visual_cleaner: VisualCleaner | None = None,
    ) -> None:
        if max_evidence_items < 1 or max_keyframe_bytes < 1:
            raise ValueError("结果查询预算必须大于 0")
        self._database = database
        self._store = artifact_store
        self._publication = DocumentPublicationService(
            database,
            artifact_store,
            visual_cleaner=visual_cleaner,
            max_document_bytes=max_document_bytes,
            max_bundle_bytes=max_bundle_bytes,
        )
        self._max_evidence_items = max_evidence_items
        self._max_keyframe_bytes = max_keyframe_bytes

    scope_key = staticmethod(scope_key)

    def persist(
        self,
        scope: Scope,
        result: VideoUnderstandingResult,
        *,
        evidence: tuple[DocumentEvidenceItem, ...],
        document: RenderedDocument,
        stage_metrics: dict[str, int],
        model_metrics: dict[str, int],
        status: Literal["SUCCEEDED", "PARTIAL_SUCCEEDED"],
        transcript_source: TranscriptSource,
        fence: ResultWriteFence,
        warnings: tuple[str, ...] = (),
        stage_cache_hits: tuple[str, ...] = (),
        published_keyframes: tuple[KeyframeEvidence, ...] = (),
    ) -> None:
        self._validate_evidence_count(evidence)
        self._publication.persist(
            scope,
            result,
            evidence=evidence,
            document=document,
            stage_metrics=stage_metrics,
            model_metrics=model_metrics,
            status=status,
            transcript_source=transcript_source,
            fence=fence,
            warnings=warnings,
            stage_cache_hits=stage_cache_hits,
            published_keyframes=published_keyframes,
        )

    def get_artifact(
        self,
        scope: Scope,
        run_id: str,
    ) -> tuple[DocumentArtifactPayload, bytes]:
        """受限读取已发布的 3.0 bundle 与确定性 Markdown。"""

        return self._publication.get_artifact(scope, run_id)

    def require_compatible_result(
        self,
        scope: Scope,
        run_id: str,
        schema_version: str = "3.0.0",
    ) -> DocumentArtifactPayload:
        if schema_version != "3.0.0":
            raise ValueError("只支持 3.0.0")
        artifact, _ = self.get_artifact(scope, run_id)
        return artifact

    def get_result(self, scope: Scope, run_id: str) -> VideoUnderstandingResult:
        return self.require_compatible_result(scope, run_id).result

    def get_document(self, scope: Scope, run_id: str) -> bytes:
        self.require_compatible_result(scope, run_id)
        return self._publication.get_document(scope, run_id)

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
        evidence = self.require_compatible_result(scope, run_id).evidence
        self._validate_evidence_count(evidence)
        filtered = tuple(
            item
            for item in evidence
            if (evidence_type is None or item.evidence_type == evidence_type)
            and (start_ms is None or item.start_ms >= start_ms)
        )
        ordered = tuple(
            sorted(
                filtered,
                key=lambda item: (
                    item.start_ms,
                    item.end_ms,
                    item.evidence_type,
                    item.evidence_id,
                ),
            )
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
        artifact = self.require_compatible_result(scope, run_id)
        keyframe = next(
            (
                item
                for item in artifact.evidence
                if isinstance(item, KeyframeEvidence) and item.keyframe_id == keyframe_id
            ),
            None,
        )
        if keyframe is None:
            raise VideoDemoError(ErrorCode.KEYFRAME_NOT_FOUND, "关键帧不存在")
        content = self._read_keyframe(scope, run_id, keyframe)
        return KeyframeContent(content=content, mime_type="image/jpeg")

    def get_run_metadata(self, scope: Scope, run_id: str) -> ResultRunMetadata:
        artifact = self.require_compatible_result(scope, run_id)
        return ResultRunMetadata(
            status=artifact.status,
            warnings=artifact.warnings,
            stage_metrics=dict(artifact.stage_metrics),
            model_metrics=dict(artifact.model_metrics),
            stage_cache_hits=artifact.stage_cache_hits,
        )

    def _read_keyframe(
        self,
        scope: Scope,
        run_id: str,
        keyframe: KeyframeEvidence,
    ) -> bytes:
        expected_parent = Path("visual/keyframes")
        relative = Path(keyframe.relative_path)
        if (
            keyframe.mime_type != "image/jpeg"
            or relative.suffix != ".jpg"
            or relative.parent != expected_parent
            or relative.name != f"{keyframe.sha256}.jpg"
            or keyframe.size_bytes > self._max_keyframe_bytes
        ):
            raise VideoDemoError(ErrorCode.ARTIFACT_SCHEMA_INVALID, "关键帧契约非法")
        runtime_relative = Path("runs") / scope_key(scope) / run_id / relative
        path = safe_runtime_path(self._store.runtime_root, runtime_relative)
        descriptor = -1
        try:
            before = os.lstat(path)
            descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            opened = os.fstat(descriptor)
            if (
                not stat.S_ISREG(opened.st_mode)
                or opened.st_nlink != 1
                or opened.st_size != keyframe.size_bytes
                or (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino)
            ):
                raise OSError
            content = os.read(descriptor, self._max_keyframe_bytes + 1)
            after = os.fstat(descriptor)
            current = os.lstat(path)
            if (
                (opened.st_dev, opened.st_ino, opened.st_size)
                != (after.st_dev, after.st_ino, after.st_size)
                or (after.st_dev, after.st_ino) != (current.st_dev, current.st_ino)
                or len(content) != keyframe.size_bytes
                or not content.startswith(_JPEG_PREFIX)
                or not content.endswith(_JPEG_SUFFIX)
                or hashlib.sha256(content).hexdigest() != keyframe.sha256
            ):
                raise OSError
            return content
        except OSError:
            raise VideoDemoError(
                ErrorCode.ARTIFACT_SCHEMA_INVALID,
                "关键帧内容完整性校验失败",
            ) from None
        finally:
            if descriptor >= 0:
                os.close(descriptor)

    def _validate_evidence_count(self, evidence: object) -> None:
        if not isinstance(evidence, (list, tuple)) or len(evidence) > self._max_evidence_items:
            raise VideoDemoError(ErrorCode.ARTIFACT_SCHEMA_INVALID, "证据数量超过配置上限")

    @staticmethod
    def _filter_key(run_id: str, evidence_type: str | None, start_ms: int | None) -> str:
        encoded = json.dumps(
            {"run_id": run_id, "evidence_type": evidence_type, "start_ms": start_ms},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:16]

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
                raise ValueError
            payload = json.loads(data)
            offset = payload.get("offset")
            if (
                payload.get("filter") != filter_key
                or isinstance(offset, bool)
                or not isinstance(offset, int)
                or offset < 0
            ):
                raise ValueError
            return cast(int, offset)
        except (UnicodeDecodeError, ValueError, json.JSONDecodeError, AttributeError) as error:
            raise VideoDemoError(ErrorCode.INVALID_EVIDENCE_CURSOR, "证据游标非法") from error


def read_result_rows(
    database: Database,
    scope: Scope,
    run_id: str,
    asset_sha256: str,
) -> VideoUnderstandingResult:
    """供闭包校验复用生产 3.0 行映射。"""
    try:
        with database.session() as session:
            return ResultRepository(session).get(scope, run_id, asset_sha256)
    except (ValidationError, TypeError, ValueError) as error:
        raise VideoDemoError(ErrorCode.ARTIFACT_SCHEMA_INVALID, "3.0 结果行非法") from error
