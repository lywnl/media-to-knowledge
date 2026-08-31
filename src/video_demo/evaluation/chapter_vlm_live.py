"""章节 VLM 评测适配器与脱敏视觉文字评分事实。"""

from __future__ import annotations

import hashlib
import unicodedata
from collections.abc import Iterable
from typing import Literal

from pydantic import Field, StrictInt, model_validator

from video_demo.domain.base import FrozenModel, Sha256, StableId, stable_identifier
from video_demo.domain.document import DocumentGenerationConfig
from video_demo.domain.document_plan import FrameCandidateArtifact, VisualSearchTarget
from video_demo.domain.evidence import (
    VisualCodeContentDraft,
    VisualDiagramContentDraft,
    VisualFormulaContentDraft,
    VisualStateContentDraft,
    VisualTableContentDraft,
    VisualTextContentDraft,
)
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.evaluation.annotations import EvaluationAnnotation, VerifiedAnnotation
from video_demo.evaluation.chapter_vlm_input import (
    ChapterVlmInputFrame,
    ChapterVlmInputManifest,
    ValidatedChapterVlmInputContext,
    base_coverage_target_id,
    chapter_vlm_chapter_id,
    chapter_vlm_input_manifest_sha256,
    validate_chapter_vlm_input_manifest,
)
from video_demo.evaluation.metrics import nfkc_character_edit_counts
from video_demo.integrations.document_port import ChapterVisionRequest, ChapterVisionResponse
from video_demo.integrations.document_validation import (
    chapter_vision_response_bytes,
    chapter_vision_response_sha256,
    validate_chapter_vision_response,
)
from video_demo.integrations.qwen_vl import (
    QwenVisionClient,
    QwenVisionProviderReceipt,
)

VisualTextCategory = Literal[
    "GENERAL_TEXT",
    "CODE",
    "TABLE",
    "FORMULA",
    "DIAGRAM",
    "UI_SMALL_TEXT",
]

__all__ = [
    "ChapterVlmCallReceipt",
    "VisualTextScoreFact",
    "build_visual_text_score_fact",
    "chapter_vision_response_bytes",
    "execute_chapter_vlm_live",
    "has_selected_frame_selection",
    "has_visual_text_projection",
]


class ChapterVlmCallReceipt(FrozenModel):
    logical_analysis_count: Literal[1]
    parent_evaluation_run_id: StableId
    evaluation_run_id: StableId
    sample_id: StableId
    manifest_sha256: Sha256
    provider: QwenVisionProviderReceipt
    ordered_input_frame_ids: tuple[StableId, ...] = Field(min_length=2, max_length=4)
    request_json_bytes: StrictInt = Field(ge=0)
    encoded_request_bytes: StrictInt = Field(ge=0)
    vlm_elapsed_ms: StrictInt = Field(ge=0)
    response_sha256: Sha256


class VisualTextScoreFact(FrozenModel):
    schema_version: Literal["1.0.0"]
    parent_evaluation_run_id: StableId
    evaluation_run_id: StableId
    sample_id: StableId
    manifest_sha256: Sha256
    response_sha256: Sha256
    reference_sha256: Sha256
    hypothesis_sha256: Sha256
    errors: StrictInt = Field(ge=0)
    reference_units: StrictInt = Field(ge=0)
    key_field_matches: StrictInt = Field(ge=0)
    key_field_reference_units: StrictInt = Field(ge=0)
    quality_categories: tuple[VisualTextCategory, ...] = Field(min_length=1)
    selected_reference_frame_count: StrictInt = Field(ge=1, le=2)

    @model_validator(mode="after")
    def validate_counts(self) -> VisualTextScoreFact:
        if self.key_field_matches > self.key_field_reference_units:
            raise ValueError("key_field_matches 不得超过参考关键字段数")
        if self.reference_units == 0 and self.errors != 0:
            raise ValueError("零参考字符时 errors 必须为 0")
        return self


def execute_chapter_vlm_live(
    manifest: ChapterVlmInputManifest,
    *,
    context: ValidatedChapterVlmInputContext,
    expected_parent_evaluation_run_id: StableId,
    expected_evaluation_run_id: StableId,
    vision_client: QwenVisionClient,
) -> tuple[ChapterVisionResponse, ChapterVlmCallReceipt]:
    """重验输入并执行一次正式章节多图 VLM 调用。"""

    if manifest.parent_evaluation_run_id != expected_parent_evaluation_run_id:
        raise VideoDemoError(ErrorCode.ARTIFACT_SCHEMA_INVALID, "父评测 Run 不匹配")
    if manifest.evaluation_run_id != expected_evaluation_run_id:
        raise VideoDemoError(ErrorCode.ARTIFACT_SCHEMA_INVALID, "章节输入产品 Run 不匹配")
    validate_chapter_vlm_input_manifest(manifest, context=context)
    request = _live_request(manifest)
    frames = request.frames
    response, provider = vision_client.analyze_chapter_with_receipt(
        request,
        allowed_run_root=context.allowed_run_root,
    )
    try:
        validate_chapter_vision_response(
            response,
            request,
            max_selected_frames=2,
        )
    except ValueError as error:
        raise VideoDemoError(
            ErrorCode.QWEN_RESPONSE_INVALID,
            "Qwen 返回内容不符合章节评测视觉契约",
        ) from error
    response_sha = chapter_vision_response_sha256(response)
    return response, ChapterVlmCallReceipt(
        logical_analysis_count=1,
        parent_evaluation_run_id=manifest.parent_evaluation_run_id,
        evaluation_run_id=manifest.evaluation_run_id,
        sample_id=manifest.sample_id,
        manifest_sha256=chapter_vlm_input_manifest_sha256(manifest),
        provider=provider,
        ordered_input_frame_ids=tuple(frame.frame_id for frame in frames),
        request_json_bytes=provider.request_json_bytes,
        encoded_request_bytes=provider.encoded_request_bytes,
        vlm_elapsed_ms=provider.elapsed_ms,
        response_sha256=response_sha,
    )


def build_visual_text_score_fact(
    manifest: ChapterVlmInputManifest,
    annotation: VerifiedAnnotation,
    response: ChapterVisionResponse,
    *,
    response_sha256: Sha256,
) -> VisualTextScoreFact:
    """仅在正式调用成功后，按选中帧计算脱敏视觉文字事实。"""

    _validate_annotation_binding(manifest, annotation)
    if response_sha256 != chapter_vision_response_sha256(response):
        raise VideoDemoError(ErrorCode.ARTIFACT_SCHEMA_INVALID, "视觉响应摘要绑定不一致")
    try:
        validate_chapter_vision_response(
            response,
            _live_request(manifest),
            max_selected_frames=2,
        )
    except ValueError as error:
        raise VideoDemoError(
            ErrorCode.VISUAL_RESULT_INVALID,
            "视觉评分响应引用不符合章节输入闭包",
        ) from error
    frame_order = {frame.frame_id: index for index, frame in enumerate(manifest.frames)}
    selected = _selected_frame_ids(response, frame_order)
    if not selected:
        raise VideoDemoError(ErrorCode.VISUAL_RESULT_INVALID, "响应没有选中输入帧")
    if len(selected) > 2:
        raise VideoDemoError(ErrorCode.VISUAL_RESULT_INVALID, "响应选中帧超过评测上限")
    selected_references = _selected_reference_ids(manifest, selected)
    reference_by_id = {
        frame.frame_id: frame for frame in annotation.annotation.visual_frames
    }
    reference_text = "\n".join(
        line
        for reference_id in selected_references
        for line in reference_by_id[reference_id].text_lines
    )
    hypothesis_text = "\n".join(_visual_text_units(response, frame_order))
    edit = nfkc_character_edit_counts(hypothesis_text, reference_text)
    key_fields = _unique_normalized(
        field
        for frame in annotation.annotation.visual_frames
        if frame.frame_id in {item.reference_frame_id for item in manifest.frames}
        for field in frame.key_fields
    )
    normalized_hypothesis = unicodedata.normalize("NFKC", hypothesis_text)
    key_matches = sum(
        1 for field in key_fields if unicodedata.normalize("NFKC", field) in normalized_hypothesis
    )
    categories = _unique_categories(
        category
        for reference_id in selected_references
        for category in reference_by_id[reference_id].quality_categories
    )
    return VisualTextScoreFact(
        schema_version="1.0.0",
        parent_evaluation_run_id=manifest.parent_evaluation_run_id,
        evaluation_run_id=manifest.evaluation_run_id,
        sample_id=manifest.sample_id,
        manifest_sha256=chapter_vlm_input_manifest_sha256(manifest),
        response_sha256=response_sha256,
        reference_sha256=_sha256_text(reference_text),
        hypothesis_sha256=_sha256_text(hypothesis_text),
        errors=edit.errors,
        reference_units=edit.reference_units,
        key_field_matches=key_matches,
        key_field_reference_units=len(key_fields),
        quality_categories=categories or ("GENERAL_TEXT",),
        selected_reference_frame_count=len(selected),
    )


def has_visual_text_projection(
    response: ChapterVisionResponse,
    manifest: ChapterVlmInputManifest,
) -> bool:
    """判断响应是否至少包含一个绑定当前输入帧的非空文字投影。"""

    frame_order = {frame.frame_id: index for index, frame in enumerate(manifest.frames)}
    try:
        return bool(_visual_text_units(response, frame_order))
    except (KeyError, ValueError):
        return False


def has_selected_frame_selection(
    response: ChapterVisionResponse,
    manifest: ChapterVlmInputManifest,
) -> bool:
    """判断响应是否至少选中了一个当前 Manifest 中的输入帧。"""

    frame_order = {frame.frame_id: index for index, frame in enumerate(manifest.frames)}
    try:
        return bool(_selected_frame_ids(response, frame_order))
    except (KeyError, ValueError):
        return False


def _frame_descriptor(frame: ChapterVlmInputFrame) -> FrameCandidateArtifact:
    return FrameCandidateArtifact(
        frame_id=frame.frame_id,
        timestamp_ms=frame.actual_timestamp_ms,
        sha256=frame.sha256,
        size_bytes=frame.size_bytes,
        relative_path=frame.relative_path,
        mime_type=frame.mime_type,
        target_ids=frame.target_ids,
    )


def _live_request(manifest: ChapterVlmInputManifest) -> ChapterVisionRequest:
    """构造评测与正式客户端共用的无人工答案章节请求。"""

    target_id = base_coverage_target_id(manifest)
    timestamps = tuple(
        sorted({frame.requested_timestamp_ms for frame in manifest.frames})
    )
    if len(timestamps) > 2:
        timestamps = (timestamps[0], timestamps[-1])
    return ChapterVisionRequest(
        chapter_id=chapter_vlm_chapter_id(manifest),
        targets=(
            VisualSearchTarget(
                target_id=target_id,
                purpose="BASE_COVERAGE",
                query_zh="识别并结构化提取这些画面中实际可见的文字、代码、表格、公式和界面状态",
                sample_timestamps_ms=timestamps,
            ),
        ),
        frames=tuple(_frame_descriptor(frame) for frame in manifest.frames),
        transcript_evidence=(),
        document_config=DocumentGenerationConfig(max_visuals_per_chapter=2),
        prompt_version="chapter-vlm-v1",
    )


def _validate_annotation_binding(
    manifest: ChapterVlmInputManifest,
    annotation: VerifiedAnnotation,
) -> None:
    value: EvaluationAnnotation = annotation.annotation
    if (
        value.sample_id != manifest.sample_id
        or value.media_sha256 != manifest.source_media_sha256
        or annotation.sha256 != manifest.annotation_sha256
    ):
        raise VideoDemoError(ErrorCode.ARTIFACT_SCHEMA_INVALID, "视觉评分标注绑定不一致")


def _selected_frame_ids(
    response: ChapterVisionResponse,
    frame_order: dict[str, int],
) -> tuple[str, ...]:
    values = {
        frame_id
        for observation in response.observations
        for frame_id in observation.selected_frame_ids
    }
    try:
        return tuple(sorted(values, key=lambda item: frame_order[item]))
    except KeyError:
        raise VideoDemoError(ErrorCode.VISUAL_RESULT_INVALID, "响应引用了未知输入帧") from None


def _selected_reference_ids(
    manifest: ChapterVlmInputManifest,
    selected_frame_ids: tuple[str, ...],
) -> tuple[str, ...]:
    selected = set(selected_frame_ids)
    return tuple(
        frame.reference_frame_id for frame in manifest.frames if frame.frame_id in selected
    )


def _visual_text_units(
    response: ChapterVisionResponse,
    frame_order: dict[str, int],
) -> tuple[str, ...]:
    blocks: list[tuple[int, int, str]] = []
    seen_blocks: set[str] = set()
    for observation_index, observation in enumerate(response.observations):
        for block_index, block in enumerate(observation.content_blocks):
            block_id = stable_identifier(
                "visual_content_block",
                block.model_dump(mode="json"),
            )
            if block_id in seen_blocks:
                continue
            seen_blocks.add(block_id)
            source = min(frame_order[item] for item in block.source_frame_ids)
            text = _content_block_text(block)
            if text:
                blocks.append((source, observation_index * 1000 + block_index, text))
    return tuple(item[2] for item in sorted(blocks, key=lambda item: (item[0], item[1])))


def _content_block_text(block: object) -> str:
    if isinstance(block, VisualTextContentDraft):
        return block.text
    if isinstance(block, VisualCodeContentDraft):
        return block.code
    if isinstance(block, VisualTableContentDraft):
        return "\n".join(("\t".join(block.columns), *("\t".join(row) for row in block.rows)))
    if isinstance(block, VisualFormulaContentDraft):
        return block.latex
    if isinstance(block, VisualDiagramContentDraft):
        return "\n".join(block.labels)
    if isinstance(block, VisualStateContentDraft):
        return "\n".join(f"{key}={value}" for key, value in block.key_values)
    return ""


def _unique_normalized(values: Iterable[str]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = unicodedata.normalize("NFKC", value).strip()
        if normalized and normalized not in seen:
            result.append(normalized)
            seen.add(normalized)
    return tuple(result)


def _unique_categories(values: Iterable[str]) -> tuple[VisualTextCategory, ...]:
    allowed = {
        "GENERAL_TEXT",
        "CODE",
        "TABLE",
        "FORMULA",
        "DIAGRAM",
        "UI_SMALL_TEXT",
    }
    result: list[VisualTextCategory] = []
    for value in _unique_normalized(values):
        if value in allowed:
            result.append(value)  # type: ignore[arg-type]
    return tuple(result)


def _sha256_text(value: str) -> Sha256:
    return hashlib.sha256(unicodedata.normalize("NFKC", value).encode("utf-8")).hexdigest()
