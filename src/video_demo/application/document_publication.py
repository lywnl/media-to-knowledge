from __future__ import annotations

import hashlib
import json
import re
import uuid
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, NoReturn, Protocol

from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session

from video_demo.application.document_rendering import RenderedDocument, render_markdown
from video_demo.application.publication_contracts import (
    ResultWriteFence as ResultWriteFence,
)
from video_demo.application.publication_contracts import (
    scope_key as scope_key,
)
from video_demo.domain.document import (
    TranscriptSource,
    VideoUnderstandingResult,
    validate_evidence_references,
)
from video_demo.domain.document_artifact import DocumentArtifactPayload
from video_demo.domain.evidence import (
    DocumentEvidenceItem,
    KeyframeEvidence,
    SpeechSegment,
    SubtitleCue,
)
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.persistence.database import Database
from video_demo.persistence.document_repository import ResultRepository
from video_demo.persistence.models import (
    JobModel,
    JobStatus,
    RunStatusValue,
    VideoAssetModel,
    VideoUnderstandingRunModel,
)
from video_demo.persistence.repositories import JobRepository, VideoRunRepository
from video_demo.persistence.scope import Scope
from video_demo.storage.artifacts import (
    RESULT_BUNDLE_ENVELOPE_SCHEMA_VERSION,
    ArtifactBytesReceipt,
    ArtifactReceipt,
    AtomicArtifactStore,
    canonical_artifact_envelope_bytes,
)

_MAX_DOCUMENT_BYTES = 16 * 1024 * 1024
_MAX_BUNDLE_BYTES = 64 * 1024 * 1024
_PUBLICATION_NAME = re.compile(
    r"(?:(knowledge-note)-([0-9a-f]{64})|bundle-([0-9a-f]{64}))-[0-9a-f]{32}\.(md|json)\Z"
)


class VisualCleaner(Protocol):
    def cleanup(
        self,
        run_relative_root: Path,
        keyframes: tuple[KeyframeEvidence, ...],
    ) -> bool: ...

    def mark_pending(self, run_relative_root: Path) -> None: ...


class DocumentPublicationService:
    """原子发布正式 4.1 结构化结果、证据和确定性 Markdown。"""

    def __init__(
        self,
        database: Database,
        artifact_store: AtomicArtifactStore,
        *,
        visual_cleaner: VisualCleaner | None = None,
        max_document_bytes: int = _MAX_DOCUMENT_BYTES,
        max_bundle_bytes: int = _MAX_BUNDLE_BYTES,
    ) -> None:
        if not 1 <= max_document_bytes <= _MAX_DOCUMENT_BYTES:
            raise ValueError("Markdown 预算超过首版硬上限")
        if not 1 <= max_bundle_bytes <= _MAX_BUNDLE_BYTES:
            raise ValueError("结果 bundle 预算超过首版硬上限")
        self._database = database
        self._store = artifact_store
        self._visual_cleaner = visual_cleaner
        self._max_document_bytes = max_document_bytes
        self._max_bundle_bytes = max_bundle_bytes

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
        if fence is None:
            raise ValueError("fence 不能为空")
        if status not in ("SUCCEEDED", "PARTIAL_SUCCEEDED"):
            raise ValueError("只能发布成功或部分成功知识文档")
        if len(warnings) != len(set(warnings)):
            raise ValueError("warnings 不得重复")
        if len(stage_cache_hits) != len(set(stage_cache_hits)):
            raise ValueError("stage_cache_hits 不得重复")
        if any(name not in stage_metrics or stage_metrics[name] != 0 for name in stage_cache_hits):
            raise ValueError("缓存命中阶段必须存在且耗时为 0")
        validate_evidence_references(result, evidence)
        self._validate_target(scope, result)
        rendered = render_markdown(result, evidence)
        if document != rendered:
            raise ValueError("document 必须来自同一结果证据闭包")
        payload = DocumentArtifactPayload(
            result=result,
            evidence=evidence,
            stage_metrics=stage_metrics,
            model_metrics=model_metrics,
            stage_cache_hits=stage_cache_hits,
            status=status,
            warnings=warnings,
            transcript_source=transcript_source,
            document_sha256=rendered.sha256,
            document_size_bytes=rendered.size_bytes,
        )
        payload_keyframes = tuple(
            item for item in payload.evidence if isinstance(item, KeyframeEvidence)
        )
        if payload_keyframes != published_keyframes:
            raise VideoDemoError(
                ErrorCode.ARTIFACT_SCHEMA_INVALID,
                "发布视觉闭包与待发布 bundle 不一致",
            )
        run_root = Path("runs") / scope_key(scope) / result.run_id
        root = run_root / "result"
        document_path = root / f"knowledge-note-{rendered.sha256}-{uuid.uuid4().hex}.md"
        digest = hashlib.sha256(_canonical_payload(payload)).hexdigest()
        bundle_path = root / f"bundle-{digest}-{uuid.uuid4().hex}.json"
        self._require_active_fence(fence)
        document_receipt: ArtifactBytesReceipt | None = None
        bundle_receipt: ArtifactReceipt | None = None
        try:
            document_receipt = self._store.write_bytes(
                document_path,
                rendered.content,
                max_bytes=self._max_document_bytes,
                exclusive=True,
            )
            bundle_receipt = self._store.write_json(
                bundle_path,
                payload.model_dump(mode="json", exclude_computed_fields=True),
                schema_version=RESULT_BUNDLE_ENVELOPE_SCHEMA_VERSION,
                upstream_sha256=result.asset_sha256,
                file_mode=0o600,
                exclusive=True,
                max_bytes=self._max_bundle_bytes,
            )
            self._commit(scope, result, fence, status, warnings, document_receipt, bundle_receipt)
        except BaseException:
            if bundle_receipt is not None and not self._bundle_is_referenced(bundle_receipt):
                with suppress(OSError, VideoDemoError):
                    self._store.discard_artifact(
                        bundle_receipt,
                        max_bytes=self._max_bundle_bytes,
                    )
            if document_receipt is not None and not self._document_is_referenced(document_receipt):
                with suppress(OSError, VideoDemoError):
                    self._discard_document(document_receipt)
            raise
        try:
            artifact, _ = self.get_artifact(scope, result.run_id)
            bundle_keyframes = tuple(
                item for item in artifact.evidence if isinstance(item, KeyframeEvidence)
            )
            if bundle_keyframes != published_keyframes:
                raise VideoDemoError(
                    ErrorCode.ARTIFACT_SCHEMA_INVALID,
                    "发布视觉闭包与已提交 bundle 不一致",
                )
            if self._visual_cleaner is not None:
                self._visual_cleaner.cleanup(run_root, bundle_keyframes)
        except Exception:
            # 发布已经赢得 fence 并提交；清理异常只能由 pending/启动恢复收敛。
            if self._visual_cleaner is not None:
                with suppress(Exception):
                    self._visual_cleaner.mark_pending(run_root)
            return

    def get_document(self, scope: Scope, run_id: str) -> bytes:
        _, document = self.get_artifact(scope, run_id)
        return document

    def get_artifact(
        self,
        scope: Scope,
        run_id: str,
    ) -> tuple[DocumentArtifactPayload, bytes]:
        try:
            with self._database.session() as session:
                run = VideoRunRepository(session).get(scope, run_id)
                if run is None:
                    raise VideoDemoError(ErrorCode.VIDEO_RUN_NOT_FOUND, "视频理解运行不存在")
                asset_sha = self._asset_sha(session, scope, run.asset_id)
                result = ResultRepository(session).get(scope, run_id, asset_sha)
                bundle_receipt, document_receipt = self._receipts(scope, run_id, asset_sha, run)
                run_facts = (
                    str(run.status),
                    tuple(run.warning_codes),
                    run.current_stage,
                    run.error_code,
                )
            payload = self._store.read_verified_json_limited(
                bundle_receipt, max_bytes=self._max_bundle_bytes
            )
            artifact = DocumentArtifactPayload.model_validate(payload)
            validate_evidence_references(artifact.result, artifact.evidence)
            document = self._store.read_verified_bytes(
                document_receipt, max_bytes=self._max_document_bytes
            )
            expected = render_markdown(artifact.result, artifact.evidence)
            if (
                artifact.result != result
                or (artifact.document_sha256, artifact.document_size_bytes)
                != (document_receipt.sha256, document_receipt.size_bytes)
                or not _publication_paths_match(bundle_receipt, document_receipt, artifact)
                or expected.content != document
                or run_facts != (artifact.status, artifact.warnings, "RESULT", None)
                or not _transcript_source_matches(artifact)
            ):
                _invalid("数据库、bundle 与 Markdown 事实不一致")
            return artifact, document
        except VideoDemoError as error:
            if error.code in {
                ErrorCode.VIDEO_RUN_NOT_FOUND,
                ErrorCode.VIDEO_RESULT_NOT_READY,
                ErrorCode.RESULT_SCHEMA_UNSUPPORTED,
            }:
                raise
            raise VideoDemoError(
                ErrorCode.ARTIFACT_SCHEMA_INVALID, "知识文档发布闭包非法"
            ) from error
        except (ValidationError, ValueError, TypeError, OSError) as error:
            raise VideoDemoError(
                ErrorCode.ARTIFACT_SCHEMA_INVALID, "知识文档发布闭包非法"
            ) from error

    def recover_current_run_orphans(
        self,
        scope: Scope,
        run_id: str,
        *,
        publishers_stopped: bool,
        max_entries: int = 128,
    ) -> int:
        """在调用方已停止该 Run 发布者时，有界清理已验证且未被当前指针引用的制品。"""

        if not publishers_stopped:
            raise ValueError("孤儿恢复前必须停止当前 Run 的全部发布者")
        root = Path("runs") / scope_key(scope) / run_id / "result"
        with self._database.session() as session:
            run = VideoRunRepository(session).get(scope, run_id)
            if run is None:
                raise VideoDemoError(ErrorCode.VIDEO_RUN_NOT_FOUND, "视频理解运行不存在")
            asset_sha = self._asset_sha(session, scope, run.asset_id)
        candidates: list[ArtifactBytesReceipt] = []
        for name in self._store.list_regular_artifacts(root, max_entries=max_entries):
            relative_path = root / name
            if self._artifact_is_referenced_by_any_run(relative_path.as_posix()):
                continue
            match = _PUBLICATION_NAME.fullmatch(name)
            if match is None or (match.group(1) == "knowledge-note") != (match.group(4) == "md"):
                _invalid("当前 Run 制品目录包含未知文件")
            max_bytes = (
                self._max_document_bytes
                if match.group(4) == "md"
                else self._max_bundle_bytes
            )
            encoded, receipt = self._store.inspect_artifact_bytes(
                relative_path,
                max_bytes=max_bytes,
            )
            if match.group(4) == "md":
                expected_digest = match.group(2)
            else:
                expected_digest = _validate_orphan_bundle(encoded, run_id, asset_sha)
                if expected_digest != match.group(3):
                    _invalid("孤儿 bundle 文件名与 payload 摘要不一致")
            if match.group(4) == "md" and receipt.sha256 != expected_digest:
                _invalid("孤儿 Markdown 文件名与内容摘要不一致")
            candidates.append(receipt)
        for receipt in candidates:
            if not self._store.discard_bytes(receipt):
                _invalid("孤儿制品在恢复期间发生变化")
        return len(candidates)

    def _artifact_is_referenced_by_any_run(self, relative_path: str) -> bool:
        with self._database.session() as session:
            reference = session.execute(
                select(
                    VideoUnderstandingRunModel.artifact_manifest_relative_path,
                    VideoUnderstandingRunModel.artifact_manifest_sha256,
                    VideoUnderstandingRunModel.document_relative_path,
                    VideoUnderstandingRunModel.document_sha256,
                    VideoUnderstandingRunModel.document_size_bytes,
                )
                .where(
                    (VideoUnderstandingRunModel.artifact_manifest_relative_path == relative_path)
                    | (VideoUnderstandingRunModel.document_relative_path == relative_path)
                )
                .limit(1)
            ).one_or_none()
        return reference is not None

    def _commit(
        self,
        scope: Scope,
        result: VideoUnderstandingResult,
        fence: ResultWriteFence,
        status: str,
        warnings: tuple[str, ...],
        document: ArtifactBytesReceipt,
        bundle: ArtifactReceipt,
    ) -> None:
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
            if run is None:
                raise VideoDemoError(ErrorCode.VIDEO_RUN_NOT_FOUND, "视频理解运行不存在")
            ResultRepository(session).replace(scope, result)
            run.status = RunStatusValue(status)
            run.current_stage = "RESULT"
            run.warning_codes = list(dict.fromkeys(warnings))
            run.error_code = None
            run.artifact_manifest_relative_path = bundle.relative_path
            run.artifact_manifest_sha256 = bundle.sha256
            run.document_relative_path = document.relative_path
            run.document_sha256 = document.sha256
            run.document_size_bytes = document.size_bytes

    def _validate_target(self, scope: Scope, result: VideoUnderstandingResult) -> None:
        with self._database.session() as session:
            run = VideoRunRepository(session).get(scope, result.run_id)
            if run is None or self._asset_sha(session, scope, run.asset_id) != result.asset_sha256:
                raise VideoDemoError(ErrorCode.VIDEO_DIGEST_MISMATCH, "结果目标与运行资产不匹配")

    def _require_active_fence(self, fence: ResultWriteFence) -> None:
        with self._database.session() as session:
            job = session.get(JobModel, fence.job_pk)
            if job is not None and job.cancel_requested:
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

    @staticmethod
    def _asset_sha(session: Session, scope: Scope, asset_id: str) -> str:
        asset = session.scalar(
            select(VideoAssetModel).where(
                VideoAssetModel.tenant_id == scope.tenant_id,
                VideoAssetModel.application_id == scope.application_id,
                VideoAssetModel.knowledge_base_id == scope.knowledge_base_id,
                VideoAssetModel.asset_id == asset_id,
            )
        )
        if asset is None:
            raise VideoDemoError(ErrorCode.VIDEO_RUN_NOT_FOUND, "视频资产不存在")
        return str(asset.source_sha256)

    def _receipts(
        self,
        scope: Scope, run_id: str, asset_sha: str, run: object
    ) -> tuple[ArtifactReceipt, ArtifactBytesReceipt]:
        bundle_path = getattr(run, "artifact_manifest_relative_path", None)
        bundle_sha = getattr(run, "artifact_manifest_sha256", None)
        document_path = getattr(run, "document_relative_path", None)
        document_sha = getattr(run, "document_sha256", None)
        document_size = getattr(run, "document_size_bytes", None)
        expected_parent = Path("runs") / scope_key(scope) / run_id / "result"
        if not all((bundle_path, bundle_sha, document_path, document_sha, document_size)):
            _invalid("双制品元数据不完整")
        bundle_match = _PUBLICATION_NAME.fullmatch(Path(str(bundle_path)).name)
        document_match = _PUBLICATION_NAME.fullmatch(Path(str(document_path)).name)
        if (
            Path(str(bundle_path)).parent != expected_parent
            or Path(str(document_path)).parent != expected_parent
            or len(str(bundle_path)) > 1024
            or len(str(document_path)) > 1024
            or not str(bundle_path).endswith(".json")
            or not str(document_path).endswith(".md")
            or bundle_match is None
            or bundle_match.group(3) is None
            or bundle_match.group(4) != "json"
            or document_match is None
            or document_match.group(2) != str(document_sha)
            or document_match.group(4) != "md"
            or not isinstance(document_size, int)
            or isinstance(document_size, bool)
            or not 1 <= document_size <= self._max_document_bytes
        ):
            _invalid("双制品路径或大小非法")
        return (
            ArtifactReceipt(
                relative_path=str(bundle_path),
                schema_version=RESULT_BUNDLE_ENVELOPE_SCHEMA_VERSION,
                sha256=str(bundle_sha),
                upstream_sha256=asset_sha,
            ),
            ArtifactBytesReceipt(
                relative_path=str(document_path), sha256=str(document_sha), size_bytes=document_size
            ),
        )

    def _discard_document(self, receipt: ArtifactBytesReceipt) -> None:
        self._store.discard_bytes(receipt)

    def _bundle_is_referenced(self, receipt: ArtifactReceipt) -> bool:
        try:
            with self._database.session() as session:
                return (
                    session.scalar(
                        select(VideoUnderstandingRunModel.id).where(
                            VideoUnderstandingRunModel.artifact_manifest_relative_path
                            == receipt.relative_path,
                            VideoUnderstandingRunModel.artifact_manifest_sha256 == receipt.sha256,
                        )
                    )
                    is not None
                )
        except BaseException:
            return True

    def _document_is_referenced(self, receipt: ArtifactBytesReceipt) -> bool:
        try:
            with self._database.session() as session:
                return (
                    session.scalar(
                        select(VideoUnderstandingRunModel.id).where(
                            VideoUnderstandingRunModel.document_relative_path
                            == receipt.relative_path,
                            VideoUnderstandingRunModel.document_sha256 == receipt.sha256,
                            VideoUnderstandingRunModel.document_size_bytes == receipt.size_bytes,
                        )
                    )
                    is not None
                )
        except BaseException:
            return True


def _canonical_payload(payload: DocumentArtifactPayload) -> bytes:
    return json.dumps(
        payload.model_dump(mode="json", exclude_computed_fields=True),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _validate_orphan_bundle(encoded: bytes, run_id: str, asset_sha: str) -> str:
    try:
        envelope = json.loads(encoded)
        if not isinstance(envelope, dict) or set(envelope) != {
            "schema_version",
            "upstream_sha256",
            "payload",
        }:
            _invalid("孤儿 bundle envelope 非法")
        if (
            envelope["schema_version"] != RESULT_BUNDLE_ENVELOPE_SCHEMA_VERSION
            or envelope["upstream_sha256"] != asset_sha
        ):
            _invalid("孤儿 bundle envelope 版本或上游摘要非法")
        payload = envelope["payload"]
        if not isinstance(payload, dict):
            _invalid("孤儿 bundle payload 非法")
        artifact = DocumentArtifactPayload.model_validate(payload)
        validate_evidence_references(artifact.result, artifact.evidence)
        if (
            artifact.result.run_id != run_id
            or artifact.result.asset_sha256 != asset_sha
            or not _transcript_source_matches(artifact)
        ):
            _invalid("孤儿 bundle 不属于目标 Run 或 Asset")
        if encoded != canonical_artifact_envelope_bytes(
            payload,
            RESULT_BUNDLE_ENVELOPE_SCHEMA_VERSION,
            asset_sha,
        ):
            _invalid("孤儿 bundle 不是规范 envelope")
        canonical = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
        ValidationError,
    ) as error:
        raise VideoDemoError(ErrorCode.ARTIFACT_SCHEMA_INVALID, "孤儿 bundle 非法") from error
    return hashlib.sha256(canonical).hexdigest()


def _publication_paths_match(
    bundle: ArtifactReceipt,
    document: ArtifactBytesReceipt,
    artifact: DocumentArtifactPayload,
) -> bool:
    bundle_match = _PUBLICATION_NAME.fullmatch(Path(bundle.relative_path).name)
    document_match = _PUBLICATION_NAME.fullmatch(Path(document.relative_path).name)
    if bundle_match is None or document_match is None:
        return False
    return (
        bundle_match.group(3) == hashlib.sha256(_canonical_payload(artifact)).hexdigest()
        and bundle_match.group(4) == "json"
        and document_match.group(2) == document.sha256
        and document_match.group(4) == "md"
    )


def _invalid(message: str) -> NoReturn:
    raise VideoDemoError(ErrorCode.ARTIFACT_SCHEMA_INVALID, message)


def _transcript_source_matches(artifact: DocumentArtifactPayload) -> bool:
    chapter_sources = {
        chapter.transcript_source
        for chapter in artifact.result.chapters
        if chapter.transcript_source != "NONE"
    }
    evidence_sources = {
        "ASR" if isinstance(item, SpeechSegment) else "SUBTITLE"
        for item in artifact.evidence
        if isinstance(item, (SpeechSegment, SubtitleCue))
    }
    if artifact.transcript_source == "NONE":
        return not chapter_sources and not evidence_sources
    expected = {artifact.transcript_source}
    return chapter_sources == expected and evidence_sources == expected
