from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import NoReturn

from pydantic import ValidationError
from sqlalchemy import Select, delete, select
from sqlalchemy.orm import Session

from video_demo.domain.document import (
    DocumentGenerationMetadata,
    SemanticChapter,
    VideoDocumentSummary,
    VideoUnderstandingResult,
)
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.persistence.models import VideoSegmentModel, VideoSummaryModel

_SENSITIVE_KEY_PARTS = (
    "secret",
    "token",
    "api_key",
    "apikey",
    "authorization",
    "password",
)


@dataclass(frozen=True, slots=True)
class Scope:
    tenant_id: str
    application_id: str
    knowledge_base_id: str


def _reject_sensitive_json(value: object, path: str = "$") -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            normalized = str(key).lower().replace("-", "_")
            if any(part in normalized for part in _SENSITIVE_KEY_PARTS):
                raise ValueError(f"检测到敏感字段: {path}.{key}")
            _reject_sensitive_json(nested, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _reject_sensitive_json(nested, f"{path}[{index}]")


class ResultRepository:
    """生产 4.1 结果行的唯一映射。"""

    def __init__(self, session: Session) -> None:
        self._session = session

    def replace(self, scope: Scope, result: VideoUnderstandingResult) -> None:
        validated = VideoUnderstandingResult.model_validate(
            result.model_dump(mode="json", exclude_computed_fields=True)
        )
        segment_payloads: list[tuple[SemanticChapter, dict[str, object]]] = []
        for chapter in validated.chapters:
            payload = chapter.model_dump(mode="json", exclude_computed_fields=True)
            _reject_sensitive_json(payload)
            segment_payloads.append((chapter, payload))
        summary_payload = {
            "summary": validated.summary.model_dump(mode="json", exclude_computed_fields=True),
            "generation": validated.generation.model_dump(
                mode="json", exclude_computed_fields=True
            ),
        }
        _reject_sensitive_json(summary_payload)

        self._delete_existing(scope, validated.run_id)
        for chapter, payload in segment_payloads:
            self._session.add(
                VideoSegmentModel(
                    tenant_id=scope.tenant_id,
                    application_id=scope.application_id,
                    knowledge_base_id=scope.knowledge_base_id,
                    run_id=validated.run_id,
                    segment_id=chapter.chapter_id,
                    start_ms=chapter.start_ms,
                    end_ms=chapter.end_ms,
                    schema_version="4.1.0",
                    payload_json=payload,
                )
            )
        self._session.add(
            VideoSummaryModel(
                tenant_id=scope.tenant_id,
                application_id=scope.application_id,
                knowledge_base_id=scope.knowledge_base_id,
                run_id=validated.run_id,
                schema_version="4.1.0",
                payload_json=summary_payload,
            )
        )
        self._session.flush()

    def get(self, scope: Scope, run_id: str, asset_sha256: str) -> VideoUnderstandingResult:
        segments = self._session.scalars(self._segments(scope, run_id)).all()
        summaries = self._session.scalars(self._summaries(scope, run_id)).all()
        _require_supported_rows(segments, summaries)
        try:
            summary_row = summaries[0]
            payload = summary_row.payload_json
            if not isinstance(payload, dict):
                _invalid("Summary payload 必须是对象")
            summary = VideoDocumentSummary.model_validate(payload.get("summary"))
            generation = DocumentGenerationMetadata.model_validate(payload.get("generation"))
            chapters = tuple(SemanticChapter.model_validate(item.payload_json) for item in segments)
            _validate_column_projections(segments, chapters)
            return VideoUnderstandingResult(
                run_id=run_id,
                asset_sha256=asset_sha256,
                summary=summary,
                chapters=chapters,
                generation=generation,
            )
        except (ValidationError, TypeError, KeyError, AttributeError) as error:
            raise VideoDemoError(ErrorCode.ARTIFACT_SCHEMA_INVALID, "4.1 结果行内容非法") from error

    def _delete_existing(self, scope: Scope, run_id: str) -> None:
        for model in (VideoSegmentModel, VideoSummaryModel):
            self._session.execute(
                delete(model).where(
                    model.tenant_id == scope.tenant_id,
                    model.application_id == scope.application_id,
                    model.knowledge_base_id == scope.knowledge_base_id,
                    model.run_id == run_id,
                )
            )

    @staticmethod
    def _segments(scope: Scope, run_id: str) -> Select[tuple[VideoSegmentModel]]:
        return (
            select(VideoSegmentModel)
            .where(
                VideoSegmentModel.tenant_id == scope.tenant_id,
                VideoSegmentModel.application_id == scope.application_id,
                VideoSegmentModel.knowledge_base_id == scope.knowledge_base_id,
                VideoSegmentModel.run_id == run_id,
            )
            .order_by(VideoSegmentModel.start_ms, VideoSegmentModel.end_ms, VideoSegmentModel.id)
        )

    @staticmethod
    def _summaries(scope: Scope, run_id: str) -> Select[tuple[VideoSummaryModel]]:
        return select(VideoSummaryModel).where(
            VideoSummaryModel.tenant_id == scope.tenant_id,
            VideoSummaryModel.application_id == scope.application_id,
            VideoSummaryModel.knowledge_base_id == scope.knowledge_base_id,
            VideoSummaryModel.run_id == run_id,
        )


def _require_supported_rows(
    segments: Sequence[VideoSegmentModel], summaries: Sequence[VideoSummaryModel]
) -> None:
    if not segments and not summaries:
        raise VideoDemoError(ErrorCode.VIDEO_RESULT_NOT_READY, "知识文档结果尚未就绪")
    if not segments or len(summaries) != 1:
        _invalid("4.1 结果行缺失或重复")
    versions = {str(item.schema_version) for item in segments} | {
        str(item.schema_version) for item in summaries
    }
    if len(versions) == 1 and versions <= {"1.0.0", "2.0.0", "3.0.0", "4.0.0"}:
        raise VideoDemoError(
            ErrorCode.RESULT_SCHEMA_UNSUPPORTED,
            "结果 Schema 4.0.0 及更早版本已停用，请重新处理视频 (当前支持 4.1.0)",
        )
    if versions != {"4.1.0"}:
        _invalid("结果行 Schema 版本非法或混杂")


def _validate_column_projections(
    segments: Sequence[VideoSegmentModel],
    chapters: tuple[SemanticChapter, ...],
) -> None:
    if len(segments) != len(chapters):
        _invalid("章节行数量不一致")
    for row, chapter in zip(segments, chapters, strict=True):
        if (row.segment_id, row.start_ms, row.end_ms) != (
            chapter.chapter_id,
            chapter.start_ms,
            chapter.end_ms,
        ):
            _invalid("章节列投影与 payload 不一致")


def _invalid(message: str) -> NoReturn:
    raise VideoDemoError(ErrorCode.ARTIFACT_SCHEMA_INVALID, message)
