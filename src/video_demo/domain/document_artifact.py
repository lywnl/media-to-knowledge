from __future__ import annotations

from typing import Literal

from pydantic import Field, StrictInt, field_validator

from video_demo.domain.base import FrozenModel, Sha256
from video_demo.domain.document import TranscriptSource, VideoUnderstandingResult
from video_demo.domain.evidence import DocumentEvidenceItem

MAX_METRIC_VALUE = 2**63 - 1
RESULT_STAGE_NAMES = frozenset(
    {
        "REGISTER",
        "PROBE",
        "TRANSCODE",
        "EVIDENCE_PREP",
        "SPEECH",
        "SPEECH_ASR",
        "SCENE_DETECT",
        "CHAPTER_PLAN",
        "FRAME_SEARCH",
        "VISUAL_EVIDENCE",
        "CHAPTER_WRITE",
        "DOCUMENT_ASSEMBLY",
        "RESULT",
    },
)
MODEL_METRIC_NAMES = frozenset(
    {
        "chapter_planner_logical_calls",
        "chapter_planner_provider_attempts",
        "chapter_planner_structure_repairs",
        "chapter_planner_cache_hits",
        "chapter_planner_fallback_chapters",
        "visual_disabled_chapters",
        "visual_no_candidate_chapters",
        "visual_collapsed_same_frame_chapters",
        "visual_frame_degraded_chapters",
        "visual_candidate_budget_degraded_chapters",
        "visual_published_budget_degraded_chapters",
        "vlm_logical_analyses",
        "vlm_provider_attempts",
        "vlm_structure_repairs",
        "vlm_cache_hits",
        "vlm_no_value_chapters",
        "vlm_fallback_chapters",
        "chapter_writer_logical_calls",
        "chapter_writer_provider_attempts",
        "chapter_writer_structure_repairs",
        "chapter_writer_cache_hits",
        "chapter_writer_fallback_chapters",
        "global_editor_logical_calls",
        "global_editor_provider_attempts",
        "global_editor_structure_repairs",
        "global_editor_cache_hits",
        "global_editor_fallbacks",
    },
)


class DocumentArtifactPayload(FrozenModel):
    artifact_schema_version: Literal["4.1.0"] = "4.1.0"
    result: VideoUnderstandingResult
    evidence: tuple[DocumentEvidenceItem, ...]
    stage_metrics: dict[str, StrictInt]
    model_metrics: dict[str, StrictInt]
    stage_cache_hits: tuple[str, ...] = ()
    status: Literal["SUCCEEDED", "PARTIAL_SUCCEEDED"]
    warnings: tuple[str, ...]
    transcript_source: TranscriptSource
    document_sha256: Sha256
    document_size_bytes: int = Field(gt=0, le=16 * 1024 * 1024)

    @field_validator("stage_metrics")
    @classmethod
    def validate_stage_metrics(cls, value: dict[str, int]) -> dict[str, int]:
        return _validate_metric_map(value, RESULT_STAGE_NAMES, "阶段")

    @field_validator("model_metrics")
    @classmethod
    def validate_model_metrics(cls, value: dict[str, int]) -> dict[str, int]:
        return _validate_metric_map(value, MODEL_METRIC_NAMES, "模型")


def _validate_metric_map(
    value: dict[str, int],
    expected_names: frozenset[str],
    display_name: str,
) -> dict[str, int]:
    actual_names = set(value)
    if actual_names != expected_names:
        if expected_names - actual_names:
            raise ValueError(f"{display_name}指标缺失白名单键")
        raise ValueError(f"{display_name}指标包含未知白名单键")
    if any(
        type(metric) is not int or not 0 <= metric <= MAX_METRIC_VALUE
        for metric in value.values()
    ):
        raise ValueError(f"{display_name}指标必须是 0 到 2^63-1 的严格整数")
    return value
