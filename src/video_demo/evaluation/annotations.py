from __future__ import annotations

import hashlib
import json
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, ValidationError, field_validator, model_validator

from video_demo.domain.base import FrozenModel, Sha256, StableId
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.evaluation.dataset import (
    EvaluationDataset,
    _read_json_file,
    _safe_relative_file,
    _safe_runtime_root,
    _sha256_media,
)

_DEFAULT_MAX_VIDEO_BYTES = 4 * 1024 * 1024 * 1024


class ReferenceOcrFrame(FrozenModel):
    frame_id: StableId
    timestamp_ms: int = Field(ge=0)
    text_lines: tuple[str, ...] = Field(min_length=1)

    @field_validator("text_lines")
    @classmethod
    def reject_blank_lines(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not line for line in value):
            raise ValueError("OCR 行不得为空")
        return value


class SupportedFact(FrozenModel):
    fact_id: StableId
    canonical_text: str = Field(min_length=1)


class KnownPerson(FrozenModel):
    person_id: StableId
    allowed_names: tuple[str, ...] = Field(min_length=1)

    @field_validator("allowed_names")
    @classmethod
    def reject_empty_or_duplicate_names(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not name for name in value) or len(value) != len(set(value)):
            raise ValueError("允许姓名不得为空或重复")
        return value


class AuthorizationRecord(FrozenModel):
    schema_version: Literal["1.0.0"]
    authorization_id: StableId
    source_category: Literal["OWNED", "LICENSED", "PUBLIC_DOMAIN"]
    allowed_purposes: tuple[Literal["VIDEO_QUALITY_EVALUATION"], ...] = Field(min_length=1)
    confirmed_at: datetime
    media_sha256: tuple[Sha256, ...] = Field(min_length=1)

    @field_validator("confirmed_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("授权确认时间必须包含时区")
        return value

    @field_validator("media_sha256")
    @classmethod
    def reject_duplicate_media(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("授权媒体摘要不得重复")
        return value

    @field_validator("allowed_purposes")
    @classmethod
    def reject_duplicate_purposes(
        cls,
        value: tuple[Literal["VIDEO_QUALITY_EVALUATION"], ...],
    ) -> tuple[Literal["VIDEO_QUALITY_EVALUATION"], ...]:
        if len(value) != len(set(value)):
            raise ValueError("授权用途不得重复")
        return value


class AuthorizationFile(FrozenModel):
    schema_version: Literal["1.0.0"]
    records: tuple[AuthorizationRecord, ...] = Field(min_length=1)

    @field_validator("records")
    @classmethod
    def reject_duplicate_record_ids(
        cls, value: tuple[AuthorizationRecord, ...]
    ) -> tuple[AuthorizationRecord, ...]:
        if len({record.authorization_id for record in value}) != len(value):
            raise ValueError("授权记录 ID 不得重复")
        return value


class EvaluationAnnotation(FrozenModel):
    schema_version: Literal["1.0.0"]
    sample_id: StableId
    media_sha256: Sha256
    duration_ms: int = Field(gt=0)
    language: Literal["zh", "en", "ja", "ko", "es"]
    reference_text: str = Field(min_length=1)
    ocr_frames: tuple[ReferenceOcrFrame, ...] = Field(min_length=1)
    scene_boundaries_ms: tuple[int, ...] = Field(min_length=1)
    semantic_boundaries_ms: tuple[int, ...] = Field(min_length=1)
    supported_facts: tuple[SupportedFact, ...] = Field(min_length=1)
    key_fact_ids: tuple[StableId, ...] = Field(min_length=1)
    known_people: tuple[KnownPerson, ...] = ()
    terms: tuple[str, ...] = ()

    @field_validator("terms")
    @classmethod
    def reject_blank_or_duplicate_terms(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(" ".join(term.split()) for term in value)
        comparison_keys = tuple(
            unicodedata.normalize("NFKC", term).casefold() for term in normalized
        )
        if any(not term for term in normalized) or len(comparison_keys) != len(
            set(comparison_keys)
        ):
            raise ValueError("明确标注术语不得为空或重复")
        return normalized

    @model_validator(mode="after")
    def validate_annotation_references(self) -> Self:
        self._validate_unique_ids()
        if any(timestamp >= self.duration_ms for timestamp in self.ocr_timestamps):
            raise ValueError("OCR 帧时间不得超过媒体时长")
        self._validate_boundaries(self.scene_boundaries_ms)
        self._validate_boundaries(self.semantic_boundaries_ms)
        fact_ids = {fact.fact_id for fact in self.supported_facts}
        if not set(self.key_fact_ids).issubset(fact_ids):
            raise ValueError("关键事实必须引用已支持事实")
        if len(self.key_fact_ids) != len(set(self.key_fact_ids)):
            raise ValueError("关键事实 ID 不得重复")
        return self

    @property
    def ocr_timestamps(self) -> tuple[int, ...]:
        return tuple(frame.timestamp_ms for frame in self.ocr_frames)

    def _validate_unique_ids(self) -> None:
        collections = (
            tuple(frame.frame_id for frame in self.ocr_frames),
            tuple(fact.fact_id for fact in self.supported_facts),
            tuple(person.person_id for person in self.known_people),
        )
        if any(len(values) != len(set(values)) for values in collections):
            raise ValueError("标注稳定 ID 不得重复")

    def _validate_boundaries(self, boundaries: tuple[int, ...]) -> None:
        if any(timestamp < 0 or timestamp >= self.duration_ms for timestamp in boundaries):
            raise ValueError("边界时间不得超过媒体时长")
        if tuple(sorted(boundaries)) != boundaries or len(boundaries) != len(set(boundaries)):
            raise ValueError("边界时间必须严格递增且不重复")


class ClaimJudgment(FrozenModel):
    claim_id: StableId
    verdict: Literal["SUPPORTED", "UNSUPPORTED"]


class SemanticJudgment(FrozenModel):
    schema_version: Literal["1.0.0"]
    sample_id: StableId
    annotation_sha256: Sha256
    prediction_sha256: Sha256
    claim_judgments: tuple[ClaimJudgment, ...]
    matched_key_fact_ids: tuple[StableId, ...]
    fabricated_names: tuple[str, ...]
    reviewer_id: StableId
    reviewed_at: datetime
    rubric_version: str = Field(min_length=1, max_length=64)

    @field_validator("reviewed_at")
    @classmethod
    def require_review_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("审阅时间必须包含时区")
        return value

    @field_validator("fabricated_names")
    @classmethod
    def reject_blank_or_duplicate_names(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized_names = tuple(_normalize_person_name(name) for name in value)
        if any(not name for name in normalized_names) or len(normalized_names) != len(
            set(normalized_names)
        ):
            raise ValueError("虚构姓名不得为空或重复")
        return value

    @field_validator("claim_judgments")
    @classmethod
    def reject_duplicate_claims(cls, value: tuple[ClaimJudgment, ...]) -> tuple[ClaimJudgment, ...]:
        if len({judgment.claim_id for judgment in value}) != len(value):
            raise ValueError("claim ID 不得重复")
        return value

    @field_validator("matched_key_fact_ids")
    @classmethod
    def reject_duplicate_key_facts(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("关键事实 ID 不得重复")
        return value


class VerifiedAnnotation(FrozenModel):
    annotation: EvaluationAnnotation
    sha256: Sha256


def pair_reference_sha256(annotation: EvaluationAnnotation) -> str:
    """计算忽略样本 ID 的规范参考摘要，供 NONE/CORRECT 配对绑定。"""

    payload = annotation.model_dump(
        mode="json",
        exclude={"sample_id"},
        exclude_computed_fields=True,
    )
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class ValidatedEvaluationPackage(FrozenModel):
    dataset: EvaluationDataset
    authorization: AuthorizationFile
    annotations: tuple[VerifiedAnnotation, ...]
    dataset_sha256: Sha256
    authorization_sha256: Sha256
    dataset_path: Path | None = Field(default=None, exclude=True)
    authorization_path: Path | None = Field(default=None, exclude=True)
    workspace_root: Path | None = Field(default=None, exclude=True)
    runtime_root: Path | None = Field(default=None, exclude=True)
    max_video_bytes: int | None = Field(default=None, exclude=True)


def load_evaluation_package(
    dataset_path: Path,
    authorization_path: Path,
    *,
    workspace_root: Path,
    runtime_root: Path,
    max_video_bytes: int = _DEFAULT_MAX_VIDEO_BYTES,
) -> ValidatedEvaluationPackage:
    """加载并交叉校验数据集、授权与人工标注，任何不可信输入一律关闭。"""

    try:
        if type(max_video_bytes) is not int or max_video_bytes <= 0:
            raise ValueError("媒体大小上限必须是正整数")
        safe_runtime_root = _safe_runtime_root(workspace_root, runtime_root)
        dataset = EvaluationDataset.load(
            dataset_path,
            workspace_root=workspace_root,
            runtime_root=safe_runtime_root,
        )
        dataset_sha256 = _sha256(dataset_path)
        authorization_relative_path = authorization_path.relative_to(dataset.eval_root)
        authorization_bytes = _read_json_file(
            _safe_relative_file(
                dataset.eval_root,
                authorization_relative_path.as_posix(),
                safe_runtime_root,
            )
        )
        authorization = AuthorizationFile.model_validate_json(authorization_bytes)
        authorization_sha256 = hashlib.sha256(authorization_bytes).hexdigest()
        records = {record.authorization_id: record for record in authorization.records}
        annotations: list[VerifiedAnnotation] = []
        for sample in dataset.samples:
            record = records.get(sample.authorization_id)
            if record is None or sample.media_sha256 not in record.media_sha256:
                raise ValueError("授权记录未覆盖样本媒体")
            annotation_path = _safe_relative_file(
                dataset.eval_root,
                sample.annotations_relative_path,
                safe_runtime_root,
            )
            annotation_bytes = _read_json_file(annotation_path)
            annotation = EvaluationAnnotation.model_validate_json(annotation_bytes)
            if (
                annotation.sample_id != sample.sample_id
                or annotation.media_sha256 != sample.media_sha256
                or annotation.language != sample.language
                or hashlib.sha256(annotation_bytes).hexdigest() != sample.annotations_sha256
            ):
                raise ValueError("样本和标注绑定不匹配")
            if (
                sample.pair_reference_sha256 is not None
                and pair_reference_sha256(annotation) != sample.pair_reference_sha256
            ):
                raise ValueError("配对参考摘要与规范标注不匹配")
            media_path = _safe_relative_file(
                dataset.eval_root,
                sample.media_relative_path,
                safe_runtime_root,
            )
            if _sha256_media(media_path, max_video_bytes) != sample.media_sha256:
                raise ValueError("媒体摘要不匹配")
            annotations.append(
                VerifiedAnnotation(
                    annotation=annotation,
                    sha256=hashlib.sha256(annotation_bytes).hexdigest(),
                )
            )
        return ValidatedEvaluationPackage(
            dataset=dataset,
            authorization=authorization,
            annotations=tuple(annotations),
            dataset_sha256=dataset_sha256,
            authorization_sha256=authorization_sha256,
            dataset_path=dataset_path.resolve(strict=True),
            authorization_path=authorization_path.resolve(strict=True),
            workspace_root=workspace_root.resolve(strict=True),
            runtime_root=safe_runtime_root,
            max_video_bytes=max_video_bytes,
        )
    except (OSError, ValueError, ValidationError, VideoDemoError):
        raise VideoDemoError(
            ErrorCode.EVALUATION_DATASET_INVALID,
            "评测数据集、授权或标注非法",
        ) from None


def reverify_evaluation_package(
    package: ValidatedEvaluationPackage,
) -> ValidatedEvaluationPackage:
    """从 loader 保存的真实路径重载 package，并全量比较评分所依赖内容。"""

    try:
        if (
            package.dataset_path is None
            or package.authorization_path is None
            or package.workspace_root is None
            or package.runtime_root is None
            or package.max_video_bytes is None
        ):
            raise ValueError("评测包缺少真实来源")
        reloaded = load_evaluation_package(
            package.dataset_path,
            package.authorization_path,
            workspace_root=package.workspace_root,
            runtime_root=package.runtime_root,
            max_video_bytes=package.max_video_bytes,
        )
        if (
            reloaded.dataset.samples != package.dataset.samples
            or reloaded.authorization != package.authorization
            or reloaded.annotations != package.annotations
            or reloaded.dataset_sha256 != package.dataset_sha256
            or reloaded.authorization_sha256 != package.authorization_sha256
        ):
            raise ValueError("评测包内容与真实来源不一致")
        return reloaded
    except (OSError, ValueError, ValidationError, VideoDemoError):
        raise VideoDemoError(
            ErrorCode.EVALUATION_DATASET_INVALID,
            "评测包真实来源缺失或重验不匹配",
        ) from None


def load_semantic_judgment(
    path: Path,
    *,
    workspace_root: Path,
    runtime_root: Path,
    annotation: VerifiedAnnotation,
    prediction: object,
) -> SemanticJudgment:
    """读取人工审阅，使用鸭子类型避免 annotation/predictions 的循环导入。"""

    try:
        from video_demo.evaluation.predictions import VerifiedPrediction

        if not isinstance(prediction, VerifiedPrediction):
            raise ValueError("预测类型非法")
        safe_runtime_root = _safe_runtime_root(workspace_root, runtime_root)
        judgment_relative_path = path.relative_to(prediction.eval_root)
        encoded = _read_json_file(
            _safe_relative_file(
                prediction.eval_root,
                judgment_relative_path.as_posix(),
                safe_runtime_root,
            )
        )
        judgment = SemanticJudgment.model_validate_json(encoded)
        if (
            judgment.sample_id != annotation.annotation.sample_id
            or judgment.sample_id != prediction.index.sample_id
            or judgment.annotation_sha256 != annotation.sha256
            or judgment.prediction_sha256 != prediction.index_sha256
        ):
            raise ValueError("审阅绑定摘要或样本不匹配")
        claim_ids = {claim.claim_id for claim in prediction.claims}
        if {item.claim_id for item in judgment.claim_judgments} != claim_ids:
            raise ValueError("审阅必须恰好覆盖当前预测 claims")
        fact_ids = set(annotation.annotation.key_fact_ids)
        if not set(judgment.matched_key_fact_ids).issubset(fact_ids):
            raise ValueError("审阅引用了未知关键事实")
        allowed_names = {
            _normalize_person_name(name)
            for person in annotation.annotation.known_people
            for name in person.allowed_names
        }
        if any(_normalize_person_name(name) in allowed_names for name in judgment.fabricated_names):
            raise ValueError("虚构姓名不得属于已知人物别名")
        return judgment
    except (OSError, ValueError, ValidationError, VideoDemoError):
        raise VideoDemoError(
            ErrorCode.EVALUATION_ARTIFACT_INVALID,
            "人工语义审阅非法或绑定失效",
        ) from None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_person_name(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())
