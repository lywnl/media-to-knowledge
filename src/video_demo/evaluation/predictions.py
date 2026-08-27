from __future__ import annotations

import hashlib
import json
import stat
from datetime import datetime
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, TypeAdapter, ValidationError, field_validator, model_validator

from video_demo.application.document_rendering import render_markdown
from video_demo.domain.base import FrozenModel, Sha256, StableId, stable_identifier
from video_demo.domain.evidence import DocumentEvidenceItem, KeyframeEvidence
from video_demo.domain.result import VideoUnderstandingResult, validate_evidence_references
from video_demo.domain.result_artifact import (
    DocumentArtifactPayload,
    TranscriptSource,
)
from video_demo.domain.run import ModelIdentity
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.evaluation.dataset import (
    EvaluationSample,
    _read_json_file,
    _safe_relative_file,
    _safe_root,
    _safe_runtime_root,
)
from video_demo.storage.artifacts import RESULT_BUNDLE_ENVELOPE_SCHEMA_VERSION

_EVIDENCE_ADAPTER = TypeAdapter(tuple[DocumentEvidenceItem, ...])
_TERMINAL_SUCCESS = ("SUCCEEDED", "PARTIAL_SUCCEEDED")
_TERMINAL_FAILURE = ("FAILED", "CANCELLED")


class PredictionRunSnapshot(FrozenModel):
    schema_version: Literal["1.0.0"]
    run_id: StableId
    job_id: StableId
    terminal_status: Literal["SUCCEEDED", "PARTIAL_SUCCEEDED", "FAILED", "CANCELLED"]
    current_stage: str = Field(min_length=1, max_length=64)
    warning_codes: tuple[str, ...] = ()
    error_code: str | None = Field(default=None, min_length=3, max_length=128)
    models: tuple[ModelIdentity, ...] = Field(min_length=1)

    @field_validator("warning_codes")
    @classmethod
    def reject_duplicate_warning_codes(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not code for code in value) or len(value) != len(set(value)):
            raise ValueError("运行警告码不得为空或重复")
        return value


class EvaluationPrediction(FrozenModel):
    schema_version: Literal["1.2.0"]
    evaluation_run_id: StableId
    sample_id: StableId
    media_sha256: Sha256
    run_id: StableId
    job_id: StableId
    terminal_status: Literal["SUCCEEDED", "PARTIAL_SUCCEEDED", "FAILED", "CANCELLED"]
    run_relative_path: str = Field(min_length=1, max_length=1024)
    run_sha256: Sha256
    result_relative_path: str | None = Field(default=None, max_length=1024)
    result_sha256: Sha256 | None = None
    evidence_relative_path: str | None = Field(default=None, max_length=1024)
    evidence_sha256: Sha256 | None = None
    document_relative_path: str | None = Field(default=None, max_length=1024)
    document_sha256: Sha256 | None = None
    document_size_bytes: int | None = Field(default=None, gt=0, le=16 * 1024 * 1024)
    artifact_manifest_relative_path: str | None = Field(default=None, max_length=1024)
    artifact_manifest_sha256: Sha256 | None = None
    failure_code: str | None = Field(default=None, min_length=3, max_length=128)
    transcript_source: TranscriptSource | None = None
    started_at: datetime
    finished_at: datetime

    @field_validator(
        "run_relative_path",
        "result_relative_path",
        "evidence_relative_path",
        "document_relative_path",
        "artifact_manifest_relative_path",
    )
    @classmethod
    def reject_unsafe_paths(cls, value: str | None) -> str | None:
        if value is None:
            return None
        path = Path(value)
        if path.is_absolute() or ".." in path.parts or not path.parts:
            raise ValueError("预测路径必须是无穿越的相对路径")
        return value

    @model_validator(mode="after")
    def validate_terminal_artifact_contract(self) -> Self:
        if self.started_at.tzinfo is None or self.finished_at.tzinfo is None:
            raise ValueError("预测时间必须包含时区")
        if self.finished_at < self.started_at:
            raise ValueError("预测结束时间不得早于开始时间")
        artifact_fields = (
            self.result_relative_path,
            self.result_sha256,
            self.evidence_relative_path,
            self.evidence_sha256,
            self.document_relative_path,
            self.document_sha256,
            self.document_size_bytes,
            self.artifact_manifest_relative_path,
            self.artifact_manifest_sha256,
        )
        if self.terminal_status in _TERMINAL_SUCCESS:
            if (
                any(field is None for field in artifact_fields)
                or self.failure_code is not None
                or self.transcript_source is None
            ):
                raise ValueError("成功预测必须完整绑定生产产物且不得有失败码")
        elif (
            any(field is not None for field in artifact_fields)
            or self.failure_code is None
            or self.transcript_source is not None
        ):
            raise ValueError("失败预测不得携带成功产物且必须有失败码")
        return self


class PredictionClaim(FrozenModel):
    claim_id: StableId
    source_kind: Literal["CHAPTER_CLAIM"] = "CHAPTER_CLAIM"
    source_id: StableId
    text: str = Field(min_length=1)
    evidence_refs: tuple[StableId, ...] = Field(min_length=1, max_length=32)
    certainty: float = Field(ge=0, le=1)


class VerifiedPrediction(FrozenModel):
    index: EvaluationPrediction
    index_sha256: Sha256
    run: PredictionRunSnapshot
    result: VideoUnderstandingResult | None
    evidence: tuple[DocumentEvidenceItem, ...]
    claims: tuple[PredictionClaim, ...]
    artifact_manifest_sha256: Sha256 | None
    eval_root: Path = Field(exclude=True)
    index_path: Path | None = Field(default=None, exclude=True)
    workspace_root: Path | None = Field(default=None, exclude=True)
    runtime_root: Path | None = Field(default=None, exclude=True)


def load_verified_prediction(
    index_path: Path,
    *,
    eval_root: Path,
    workspace_root: Path,
    runtime_root: Path,
    sample: EvaluationSample,
) -> VerifiedPrediction:
    """从真实生产导出产物重建预测；不信任索引声称的状态或摘要。"""

    try:
        safe_runtime_root = _safe_runtime_root(workspace_root, runtime_root)
        safe_eval_root = _safe_root(eval_root, safe_runtime_root)
        index_file = _absolute_safe_file(index_path, eval_root, safe_runtime_root)
        index_bytes = _read_json_file(index_file)
        index = EvaluationPrediction.model_validate_json(index_bytes)
        if index.sample_id != sample.sample_id or index.media_sha256 != sample.media_sha256:
            raise ValueError("预测索引与样本不匹配")
        run = _load_json_model(
            safe_eval_root,
            safe_runtime_root,
            index.run_relative_path,
            index.run_sha256,
            PredictionRunSnapshot,
        )
        if (
            run.run_id != index.run_id
            or run.job_id != index.job_id
            or run.terminal_status != index.terminal_status
        ):
            raise ValueError("运行快照与预测索引不匹配")
        if index.terminal_status in _TERMINAL_SUCCESS and run.error_code is not None:
            raise ValueError("成功运行快照不得有失败码")
        if index.terminal_status in _TERMINAL_FAILURE:
            if run.error_code != index.failure_code:
                raise ValueError("失败运行快照与预测失败码不匹配")
            return VerifiedPrediction(
                index=index,
                index_sha256=hashlib.sha256(index_bytes).hexdigest(),
                run=run,
                result=None,
                evidence=(),
                claims=(),
                artifact_manifest_sha256=None,
                eval_root=safe_eval_root,
                index_path=index_file,
                workspace_root=workspace_root.resolve(strict=True),
                runtime_root=safe_runtime_root,
            )
        assert index.result_relative_path is not None and index.result_sha256 is not None
        assert index.evidence_relative_path is not None and index.evidence_sha256 is not None
        assert index.document_relative_path is not None and index.document_sha256 is not None
        assert index.document_size_bytes is not None
        assert index.artifact_manifest_relative_path is not None
        assert index.artifact_manifest_sha256 is not None
        result_file = _safe_relative_file(
            safe_eval_root, index.result_relative_path, safe_runtime_root
        )
        result_bytes = _read_json_file(result_file)
        _require_digest(result_bytes, index.result_sha256)
        result = VideoUnderstandingResult.model_validate_json(result_bytes)
        if result.run_id != index.run_id or result.asset_sha256 != sample.media_sha256:
            raise ValueError("结果与预测索引不匹配")
        evidence_file = _safe_relative_file(
            safe_eval_root, index.evidence_relative_path, safe_runtime_root
        )
        evidence_bytes = _read_json_file(evidence_file)
        _require_digest(evidence_bytes, index.evidence_sha256)
        evidence = _parse_evidence_jsonl(evidence_bytes)
        validate_evidence_references(result, evidence)
        document_file = _safe_relative_file(
            safe_eval_root,
            index.document_relative_path,
            safe_runtime_root,
        )
        document_bytes = _read_json_file(document_file)
        _require_digest(document_bytes, index.document_sha256)
        if len(document_bytes) != index.document_size_bytes:
            raise ValueError("Markdown 大小与预测索引不一致")
        rendered = render_markdown(result, evidence)
        if rendered.content != document_bytes:
            raise ValueError("Markdown 不是结构化结果的确定性渲染")
        _validate_manifest(
            safe_eval_root,
            safe_runtime_root,
            index,
            result=result,
            evidence=evidence,
            document_bytes=document_bytes,
            run=run,
        )
        _validate_keyframe_closure(
            _safe_relative_file(
                safe_eval_root,
                index.artifact_manifest_relative_path,
                safe_runtime_root,
            ).parent,
            evidence,
        )
        return VerifiedPrediction(
            index=index,
            index_sha256=hashlib.sha256(index_bytes).hexdigest(),
            run=run,
            result=result,
            evidence=evidence,
            claims=_extract_claims(result),
            artifact_manifest_sha256=index.artifact_manifest_sha256,
            eval_root=safe_eval_root,
            index_path=index_file,
            workspace_root=workspace_root.resolve(strict=True),
            runtime_root=safe_runtime_root,
        )
    except (OSError, ValueError, ValidationError, VideoDemoError):
        raise VideoDemoError(
            ErrorCode.EVALUATION_ARTIFACT_INVALID,
            "评测预测产物非法或绑定不匹配",
        ) from None


def reverify_verified_prediction(
    prediction: VerifiedPrediction,
    *,
    sample: EvaluationSample,
) -> VerifiedPrediction:
    """从真实 index 路径重新严格加载，并比较全部解析后生产内容。"""

    try:
        if (
            prediction.index_path is None
            or prediction.workspace_root is None
            or prediction.runtime_root is None
        ):
            raise ValueError("预测缺少真实来源")
        reloaded = load_verified_prediction(
            prediction.index_path,
            eval_root=prediction.eval_root,
            workspace_root=prediction.workspace_root,
            runtime_root=prediction.runtime_root,
            sample=sample,
        )
        if (
            reloaded.index != prediction.index
            or reloaded.index_sha256 != prediction.index_sha256
            or reloaded.run != prediction.run
            or reloaded.result != prediction.result
            or reloaded.evidence != prediction.evidence
            or reloaded.claims != prediction.claims
            or reloaded.artifact_manifest_sha256 != prediction.artifact_manifest_sha256
        ):
            raise ValueError("预测内容与真实来源不一致")
        return reloaded
    except (OSError, ValueError, ValidationError, VideoDemoError):
        raise VideoDemoError(
            ErrorCode.EVALUATION_ARTIFACT_INVALID,
            "评测预测真实来源缺失或重验不匹配",
        ) from None


def _absolute_safe_file(path: Path, root: Path, runtime_root: Path) -> Path:
    _safe_root(root, runtime_root)
    if path.is_symlink() or not path.is_file():
        raise ValueError("预测索引不存在或不是普通文件")
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise ValueError("预测索引不在评测根目录") from error
    return _safe_relative_file(root, relative.as_posix(), runtime_root)


def _load_json_model(
    root: Path,
    runtime_root: Path,
    relative_path: str,
    expected_sha256: str,
    model: type[PredictionRunSnapshot],
) -> PredictionRunSnapshot:
    encoded = _read_json_file(_safe_relative_file(root, relative_path, runtime_root))
    _require_digest(encoded, expected_sha256)
    return model.model_validate_json(encoded)


def _require_digest(content: bytes, expected: str) -> None:
    if hashlib.sha256(content).hexdigest() != expected:
        raise ValueError("产物摘要不匹配")


def _parse_evidence_jsonl(content: bytes) -> tuple[DocumentEvidenceItem, ...]:
    values = [json.loads(line) for line in content.decode("utf-8").splitlines() if line.strip()]
    if not values:
        raise ValueError("证据 JSONL 不能为空")
    return _EVIDENCE_ADAPTER.validate_python(values)


def _validate_manifest(
    root: Path,
    runtime_root: Path,
    index: EvaluationPrediction,
    *,
    result: VideoUnderstandingResult,
    evidence: tuple[DocumentEvidenceItem, ...],
    document_bytes: bytes,
    run: PredictionRunSnapshot,
) -> None:
    assert index.artifact_manifest_relative_path is not None
    assert index.artifact_manifest_sha256 is not None
    encoded = _read_json_file(
        _safe_relative_file(root, index.artifact_manifest_relative_path, runtime_root)
    )
    _require_digest(encoded, index.artifact_manifest_sha256)
    envelope = json.loads(encoded.decode("utf-8"))
    if not isinstance(envelope, dict) or set(envelope) != {
        "schema_version",
        "upstream_sha256",
        "payload",
    }:
        raise ValueError("生产产物 Manifest envelope 非法")
    if (
        envelope["schema_version"] != RESULT_BUNDLE_ENVELOPE_SCHEMA_VERSION
        or envelope["upstream_sha256"] != index.media_sha256
    ):
        raise ValueError("生产产物 Manifest 上游绑定不匹配")
    payload = DocumentArtifactPayload.model_validate(envelope["payload"])
    if payload.status != index.terminal_status:
        raise ValueError("生产产物 Manifest 状态不匹配")
    if payload.result != result or payload.evidence != evidence:
        raise ValueError("生产产物 Manifest 内容与导出产物不一致")
    if (
        payload.document_sha256 != hashlib.sha256(document_bytes).hexdigest()
        or payload.document_size_bytes != len(document_bytes)
    ):
        raise ValueError("生产产物 Manifest 的 Markdown 绑定不匹配")
    if payload.warnings != run.warning_codes:
        raise ValueError("生产产物 Manifest 警告与运行快照不匹配")
    if payload.transcript_source != index.transcript_source:
        raise ValueError("生产产物 Manifest 文本来源与预测索引不匹配")


def _extract_claims(result: VideoUnderstandingResult) -> tuple[PredictionClaim, ...]:
    return tuple(
        _claim(chapter.chapter_id, index, claim)
        for chapter in result.chapters
        for index, claim in enumerate(chapter.claims)
    )


def _claim(
    chapter_id: str,
    chapter_claim_index: int,
    claim: object,
) -> PredictionClaim:
    from video_demo.domain.document import GroundedClaim

    if not isinstance(claim, GroundedClaim):
        raise TypeError("章节 claim 类型非法")
    identity = {
        "source_kind": "CHAPTER_CLAIM",
        "source_id": chapter_id,
        "chapter_claim_index": chapter_claim_index,
        "claim": claim.model_dump(mode="json"),
    }
    return PredictionClaim(
        claim_id=stable_identifier("claim", identity),
        source_id=chapter_id,
        text=claim.text,
        evidence_refs=claim.evidence_refs,
        certainty=claim.certainty,
    )


def _validate_keyframe_closure(
    sample_root: Path,
    evidence: tuple[DocumentEvidenceItem, ...],
) -> None:
    keyframes = tuple(item for item in evidence if isinstance(item, KeyframeEvidence))
    expected_names = {f"{item.sha256}.jpg" for item in keyframes}
    keyframe_root = sample_root / "visual" / "keyframes"
    if not keyframe_root.exists():
        if keyframes:
            raise ValueError("评测关键帧目录缺失")
        return
    if keyframe_root.is_symlink() or not keyframe_root.is_dir():
        raise ValueError("评测关键帧目录非法")
    actual_names = {entry.name for entry in keyframe_root.iterdir()}
    if actual_names != expected_names:
        raise ValueError("评测关键帧副本闭包不精确")
    for item in keyframes:
        path = keyframe_root / f"{item.sha256}.jpg"
        metadata = path.lstat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_size != item.size_bytes
        ):
            raise ValueError("评测关键帧副本类型、链接数或大小非法")
        content = path.read_bytes()
        if (
            not content.startswith(b"\xff\xd8\xff")
            or not content.endswith(b"\xff\xd9")
            or hashlib.sha256(content).hexdigest() != item.sha256
        ):
            raise ValueError("评测关键帧副本不是声明的 JPEG 内容")
