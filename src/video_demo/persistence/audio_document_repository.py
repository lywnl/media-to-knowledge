"""音频 1.0 结果行的唯一映射。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NoReturn

from pydantic import ValidationError
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from video_demo.domain.audio_document import (
    AudioChapter,
    AudioDocumentSummary,
    AudioUnderstandingResult,
)
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.persistence.models import AudioAssetModel, AudioSegmentModel, AudioSummaryModel
from video_demo.persistence.scope import Scope


class AudioResultRepository:
    """仅读写音频结果表。"""

    def __init__(self, session: Session) -> None:
        self._session = session

    def replace(self, scope: Scope, result: AudioUnderstandingResult) -> None:
        validated = AudioUnderstandingResult.model_validate(
            result.model_dump(mode="json", exclude_computed_fields=True),
        )
        self._delete_existing(scope, validated.run_id)
        self._session.execute(
            delete(AudioAssetModel).where(
                AudioAssetModel.tenant_id == scope.tenant_id,
                AudioAssetModel.application_id == scope.application_id,
                AudioAssetModel.knowledge_base_id == scope.knowledge_base_id,
                AudioAssetModel.run_id == validated.run_id,
            ),
        )
        self._session.add(
            AudioAssetModel(
                tenant_id=scope.tenant_id,
                application_id=scope.application_id,
                knowledge_base_id=scope.knowledge_base_id,
                run_id=validated.run_id,
                asset_id=validated.asset_sha256,
                object_ref=validated.run_id,
                source_sha256=validated.asset_sha256,
                schema_version=validated.schema_version,
            ),
        )
        for chapter in validated.chapters:
            self._session.add(
                AudioSegmentModel(
                    tenant_id=scope.tenant_id,
                    application_id=scope.application_id,
                    knowledge_base_id=scope.knowledge_base_id,
                    run_id=validated.run_id,
                    segment_id=chapter.chapter_id,
                    start_ms=chapter.start_ms,
                    end_ms=chapter.end_ms,
                    schema_version=validated.schema_version,
                    payload_json=chapter.model_dump(
                        mode="json", exclude_computed_fields=True
                    ),
                ),
            )
        self._session.add(
            AudioSummaryModel(
                tenant_id=scope.tenant_id,
                application_id=scope.application_id,
                knowledge_base_id=scope.knowledge_base_id,
                run_id=validated.run_id,
                schema_version=validated.schema_version,
                payload_json={
                    "summary": validated.summary.model_dump(
                        mode="json", exclude_computed_fields=True
                    ),
                },
            ),
        )
        self._session.flush()

    def get(self, scope: Scope, run_id: str, asset_sha256: str) -> AudioUnderstandingResult:
        segments = self._session.scalars(
            select(AudioSegmentModel)
            .where(
                AudioSegmentModel.tenant_id == scope.tenant_id,
                AudioSegmentModel.application_id == scope.application_id,
                AudioSegmentModel.knowledge_base_id == scope.knowledge_base_id,
                AudioSegmentModel.run_id == run_id,
            )
            .order_by(AudioSegmentModel.start_ms, AudioSegmentModel.end_ms, AudioSegmentModel.id),
        ).all()
        summaries = self._session.scalars(
            select(AudioSummaryModel).where(
                AudioSummaryModel.tenant_id == scope.tenant_id,
                AudioSummaryModel.application_id == scope.application_id,
                AudioSummaryModel.knowledge_base_id == scope.knowledge_base_id,
                AudioSummaryModel.run_id == run_id,
            ),
        ).all()
        _require_supported_rows(segments, summaries)
        try:
            summary_payload = summaries[0].payload_json
            if not isinstance(summary_payload, dict):
                _invalid("音频摘要 payload 必须是对象")
            summary = AudioDocumentSummary.model_validate(summary_payload.get("summary"))
            chapters = tuple(AudioChapter.model_validate(item.payload_json) for item in segments)
            _validate_column_projections(segments, chapters)
            return AudioUnderstandingResult(
                run_id=run_id,
                asset_sha256=asset_sha256,
                summary=summary,
                chapters=chapters,
            )
        except (ValidationError, TypeError, KeyError, AttributeError) as error:
            raise VideoDemoError(ErrorCode.ARTIFACT_SCHEMA_INVALID, "音频结果行内容非法") from error

    def _delete_existing(self, scope: Scope, run_id: str) -> None:
        for model in (AudioSegmentModel, AudioSummaryModel):
            self._session.execute(
                delete(model).where(
                    model.tenant_id == scope.tenant_id,
                    model.application_id == scope.application_id,
                    model.knowledge_base_id == scope.knowledge_base_id,
                    model.run_id == run_id,
                ),
            )


def _require_supported_rows(
    segments: Sequence[AudioSegmentModel],
    summaries: Sequence[AudioSummaryModel],
) -> None:
    if not segments and not summaries:
        raise VideoDemoError(ErrorCode.AUDIO_RESULT_NOT_READY, "音频结果尚未就绪")
    if not segments or len(summaries) != 1:
        _invalid("音频结果行缺失或重复")
    versions = {str(item.schema_version) for item in segments} | {
        str(item.schema_version) for item in summaries
    }
    if versions != {"1.0.0"}:
        raise VideoDemoError(
            ErrorCode.RESULT_SCHEMA_UNSUPPORTED,
            "音频结果版本不受支持，请重新处理音频",
        )


def _validate_column_projections(
    segments: Sequence[AudioSegmentModel],
    chapters: tuple[AudioChapter, ...],
) -> None:
    if len(segments) != len(chapters):
        _invalid("音频章节行数量不一致")
    for row, chapter in zip(segments, chapters, strict=True):
        if (row.segment_id, row.start_ms, row.end_ms) != (
            chapter.chapter_id,
            chapter.start_ms,
            chapter.end_ms,
        ):
            _invalid("音频章节列投影与 payload 不一致")


def _invalid(message: str) -> NoReturn:
    raise VideoDemoError(ErrorCode.ARTIFACT_SCHEMA_INVALID, message)
