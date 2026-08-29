from __future__ import annotations

import logging

from video_demo.domain.document import ChapterBodyBlock, VisualBlock
from video_demo.domain.evidence import VisualObservationEvidence
from video_demo.integrations.document_port import ChapterWritingResponse

_LOGGER = logging.getLogger(__name__)


def normalize_optional_visual_blocks(
    response: ChapterWritingResponse,
    visual_observations: tuple[VisualObservationEvidence, ...],
) -> ChapterWritingResponse:
    """清理可选视觉块的错误子内容引用，保留同一响应中的文字内容。

    视觉观察本身是已经验证的输入证据，但视觉子内容并非每次都存在。
    因此无法确认的视觉块可以被删除或置空；标题、摘要、普通正文和
    claims 不在此处放宽，继续由写作响应校验负责拒绝未知引用。
    """

    observations = {item.evidence_id: item for item in visual_observations}
    blocks: list[ChapterBodyBlock] = []
    for block in response.body_blocks:
        if not isinstance(block, VisualBlock):
            blocks.append(block)
            continue

        observation = observations.get(block.visual_observation_ref)
        if observation is None:
            _log_discarded_visual_block(
                block,
                reason="unknown_observation",
            )
            continue

        if block.evidence_refs != (block.visual_observation_ref,):
            _log_discarded_visual_block(block, reason="invalid_visual_evidence_refs")
            continue

        allowed_content_ids = {
            *(item.visual_content_id for item in observation.content_blocks),
            *(item.visual_fact_id for item in observation.visual_facts),
        }
        if not allowed_content_ids:
            if block.visual_content_refs:
                block = block.model_copy(update={"visual_content_refs": ()})
            blocks.append(block)
            continue

        if not block.visual_content_refs:
            _log_discarded_visual_block(block, reason="missing_content_reference")
            continue
        if not set(block.visual_content_refs).issubset(allowed_content_ids):
            _log_discarded_visual_block(block, reason="unknown_content_reference")
            continue
        blocks.append(block)

    normalized_blocks = tuple(blocks)
    if normalized_blocks == response.body_blocks:
        return response
    return response.model_copy(update={"body_blocks": normalized_blocks})


def _log_discarded_visual_block(block: VisualBlock, *, reason: str) -> None:
    _LOGGER.warning(
        "章节写作可选视觉块已丢弃 observation_ref=%s reason=%s",
        block.visual_observation_ref,
        reason,
    )
