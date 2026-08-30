from __future__ import annotations

import json
import math
from typing import Any

from video_demo.integrations.document_port import (
    ChapterPlanningRequest,
    ChapterPlanRepairRequest,
    ChapterVisionRepairRequest,
    ChapterVisionRequest,
    ChapterWritingRepairRequest,
    ChapterWritingRequest,
    GlobalWritingRepairRequest,
    GlobalWritingRequest,
)

VisionPrompt = tuple[str, str, str]
VisionPayload = dict[str, object]


def prompt_for_planning(request: ChapterPlanningRequest) -> tuple[str, str, str]:
    return _prompt(
        request.prompt_version,
        (
            "你只能根据输入的基础片段和转写证据规划连续章节。所有 segment_refs "
            "必须按输入顺序完整分区且恰好使用一次；每个语义目标必须绑定当前章节的 "
            "1~3 个按时间排序的转写 evidence_id，首个锚点起点到末个锚点终点不得超过 "
            "30 秒。章节粒度优先目标为 fine 60~120 秒、standard 60~180 秒、"
            "coarse 120~300 秒；证据不可拆分时允许偏离目标范围，但任何章节不得超过 "
            "300 秒。不得生成时间、章节 ID 或未知引用。segments 数组每项依次为 "
            "[segment_id, duration_ms, transcript_evidence_indexes]；索引对应 "
            "transcript_evidence 数组，按输入顺序解释。"
        ),
        _planning_context(request),
    )


def prompt_for_compact_planning(request: ChapterPlanningRequest) -> tuple[str, str, str]:
    """使用数组索引规划章节，避免模型重复生成长稳定标识。"""

    return _prompt(
        request.prompt_version,
        (
            "你只能根据输入的基础片段和转写证据规划连续章节。使用数组索引返回范围："
            "start_segment_index 包含、end_segment_index 不包含，所有范围必须按顺序从 0 开始；"
            "普通章节 end_segment_index 不得超过 segments 长度，"
            "最后一个章节允许暂时使用 segments 长度加 1，"
            "程序会归一化为 segments 长度；"
            "end_segment_index 只能引用 segments 数组位置，不得使用毫秒时间或全片位置。"
            "segments 每项为 [segment_index, start_ms, end_ms, transcript_evidence_indexes]；"
            "segment_transcript_index_ranges 与 segments 一一对应，给出每个基础片段所拥有的"
            "全局转写索引首尾(无转写片段为 null)；语义锚点只能从章节所含片段的这些范围中选择；"
            "visual_mode=NONE 时 semantic_targets 必须为空；"
            "否则语义目标的 anchor_transcript_indexes 必须是当前章节内；"
            "如果无法确认章节边界，semantic_targets 必须返回空数组；"
            "title_hint 必须概括当前章节自己的转写内容，优先使用该章节内明确出现的主题、"
            "账号或操作名称；不得把相邻章节的主题、赛道或编号写入当前章节标题。"
            "COMPARISON/MULTI_STEP 至少需要 2 个互不重叠的语义目标；"
            "当转写明确提到截图、界面、图文、漫画或操作步骤时，优先返回"
            "visual_mode=SINGLE 并绑定 1 个最相关语义目标，不要无理由返回 NONE；"
            "transcript_evidence_indexes 是当前请求 transcript_evidence 数组的位置；"
            "不得使用整部视频的全局转写编号。请先根据每个 segment 的"
            "transcript_evidence_indexes 判断章节实际拥有的索引，"
            "不得把相邻章节或空片段的索引绑定到当前章节；"
            "返回前逐章执行集合校验：把该章节所有 segment 的 transcript_evidence_indexes 合并，"
            "每个 semantic_targets.anchor_transcript_indexes 必须是这个集合的子集；"
            "校验不通过就删除该 semantic target(必要时将 visual_mode 改为 NONE)，"
            "绝对不要复制上一章或下一章的锚点。"
            "1~3 个按时间排序且不重复的 transcript_evidence 索引，跨度不得超过 30 秒。"
            "章节粒度优先目标为 fine 60~120 秒、standard 60~180 秒、coarse 120~300 秒；"
            "任何章节不得超过 300 秒。不得生成时间、ID 或未知索引。transcript_evidence 每项为"
            "[transcript_index, start_ms, end_ms, text]。每条转写证据的 transcript_index "
            "必须使用输入数组位置；只返回 JSON。"
        ),
        _compact_planning_context(request),
    )


def prompt_for_compact_plan_repair(request: ChapterPlanRepairRequest) -> tuple[str, str, str]:
    context = {
        "invalid_response": request.invalid_response.model_dump(mode="json"),
        "request": _compact_planning_context(request.request),
    }
    return _prompt(
        request.prompt_version,
        "只修复结构和索引；chapter_drafts 必须按顺序完整覆盖 request.segments，"
        "普通章节 end_segment_index 不得超过 request.segments 长度，"
        "最后一个章节最多允许 request.segments 长度加 1，"
        "程序最终归一化，"
        "anchor_transcript_indexes 必须属于对应章节的 transcript_evidence_indexes；"
        "必须保留原请求的事实边界，不得添加新事实。只返回 JSON。",
        context,
    )


def _planning_context(request: ChapterPlanningRequest) -> dict[str, object]:
    """只向章节规划模型发送决策所需字段，完整请求仍由程序保留并校验。

    语言、置信度、场景引用和视觉配置不会改变章节边界，却会随每条转写重复
    占用上下文。删掉这些冗余字段可以让更多视频在一次规划请求内完成，同时不
    改变缓存指纹或程序侧的证据闭包校验。
    """

    evidence_ids = {
        evidence.evidence_id: index
        for index, evidence in enumerate(request.transcript_evidence)
    }
    return {
        "title_hint": request.title_hint,
        "duration_ms": request.duration_ms,
        "chapter_granularity": request.document_config.chapter_granularity,
        "segments": tuple(
            (
                segment.segment_id,
                segment.duration_ms,
                tuple(evidence_ids[ref] for ref in segment.evidence_refs),
            )
            for segment in request.segments
        ),
        "transcript_evidence": tuple(
            (
                evidence.evidence_id,
                evidence.start_ms,
                evidence.end_ms,
                evidence.text,
            )
            for evidence in request.transcript_evidence
        ),
    }


def _compact_planning_context(request: ChapterPlanningRequest) -> dict[str, object]:
    transcript_index_by_id = {
        evidence.evidence_id: index
        for index, evidence in enumerate(request.transcript_evidence)
    }
    segments = tuple(
        (
            index,
            segment.start_ms,
            segment.end_ms,
            tuple(
                transcript_index_by_id[ref]
                for ref in segment.evidence_refs
            ),
        )
        for index, segment in enumerate(request.segments)
    )
    segment_transcript_index_ranges = tuple(
        (
            min(indexes) if indexes else None,
            max(indexes) if indexes else None,
        )
        for indexes in (
            tuple(
                transcript_index_by_id[ref]
                for ref in segment.evidence_refs
            )
            for segment in request.segments
        )
    )
    return {
        "title_hint": request.title_hint,
        "duration_ms": request.duration_ms,
        "chapter_granularity": request.document_config.chapter_granularity,
        "segments": segments,
        "segment_transcript_index_ranges": segment_transcript_index_ranges,
        "transcript_evidence": tuple(
            (index, evidence.start_ms, evidence.end_ms, evidence.text)
            for index, evidence in enumerate(request.transcript_evidence)
        ),
    }


def prompt_for_plan_repair(request: ChapterPlanRepairRequest) -> tuple[str, str, str]:
    context = request.model_dump(mode="json", exclude={"request"})
    context["request"] = _planning_context(request.request)
    return _prompt(
        request.prompt_version,
        "只修复结构和引用；必须保留原请求的事实边界，不得添加新事实。",
        context,
    )


def prompt_for_vision(request: ChapterVisionRequest) -> tuple[str, str, str]:
    frame_limit = request.max_selected_frames
    return _prompt(
        request.prompt_version,
        (
            "你只能比较输入的本地图片，并且只能返回输入中的 frame_id、target_id 和 "
            f"evidence_id。每个 observation 最多选择 {frame_limit} 张图片；"
            f"整份响应最多使用 {frame_limit} 张不同图片。"
            "如果图片无法确认目标，返回 observations=[]，不要猜测或选择额外图片。"
            "先选择 selected_frame_ids，再从这些帧的 target_ids 交集表中复制 target_ids；"
            "selected_frame_ids 必须是实际最能支持该 observation 的最少图片。"
            "先逐字复制输入中的 evidence_id：INDEPENDENT 时 "
            "transcript_evidence_refs 必须为空；"
            "其他音画关系必须至少引用 1 条当前转写证据。"
        ),
        _vision_context(request),
    )


def prompt_for_vision_repair(request: ChapterVisionRepairRequest) -> tuple[str, str, str]:
    frame_limit = request.request.max_selected_frames
    context = request.model_dump(mode="json", exclude={"request"})
    context["request"] = _vision_context(request.request)
    return _prompt(
        request.prompt_version,
        (
            "只修复结构和引用；图片顺序和原始取证问题不得改变，不得添加新事实。"
            f"每个 observation 最多选择 {frame_limit} 张图片，"
            f"整份响应最多使用 {frame_limit} 张不同图片；"
            "先选择 selected_frame_ids，再从这些帧的 target_ids 交集表中复制 target_ids；"
            "无法确认时返回 observations=[]。先逐字复制输入中的 evidence_id："
            "INDEPENDENT 时 transcript_evidence_refs 必须为空；"
            "其他音画关系必须至少引用 1 条当前转写证据。"
        ),
        context,
    )




def build_vision_payload(
    prompt: VisionPrompt,
    *,
    model_id: str,
    schema_name: str,
    response_schema: dict[str, object],
    ordered_encoded_frames: tuple[tuple[str, str], ...],
) -> VisionPayload:
    version, instruction, data = prompt
    return {
        "model": model_id,
        "temperature": 0,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "strict": True,
                "schema": response_schema,
            },
        },
        "messages": [
            {
                "role": "system",
                "content": f"PROMPT_VERSION={version}\n{instruction}",
            },
            {
                "role": "user",
                "content": _vision_content(data, ordered_encoded_frames),
            },
        ],
    }


def vision_payload_size_upper_bound(
    prompt: VisionPrompt,
    *,
    model_id: str,
    schema_name: str,
    response_schema: dict[str, object],
    ordered_frames: tuple[tuple[str, int], ...],
) -> int:
    if any(size_bytes < 1 for _frame_id, size_bytes in ordered_frames):
        raise ValueError("视觉图片大小必须大于 0")
    payload_without_base64 = build_vision_payload(
        prompt,
        model_id=model_id,
        schema_name=schema_name,
        response_schema=response_schema,
        ordered_encoded_frames=tuple((frame_id, "") for frame_id, _size in ordered_frames),
    )
    encoded_image_bytes = sum(
        4 * math.ceil(size_bytes / 3)
        for _frame_id, size_bytes in ordered_frames
    )
    return len(vision_payload_json_bytes(payload_without_base64)) + encoded_image_bytes


def vision_payload_json_bytes(payload: VisionPayload) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def prompt_for_writing(request: ChapterWritingRequest) -> tuple[str, str, str]:
    return _prompt(
        request.prompt_version,
        (
            "你只能使用输入证据写作；不得发明事实、时间、章节 ID 或关键帧 ID。"
            "每个 body_blocks 项都必须包含 block_type，且只能是 PARAGRAPH、BULLET_LIST、"
            "QUOTE、CODE、TABLE、FORMULA 或 VISUAL，并填写该类型要求的字段。"
            "title_evidence_refs、summary_evidence_refs、普通正文 evidence_refs 和 "
            "claims.evidence_refs 只能引用 ASR/字幕 evidence_id 或 "
            "visual_observation.evidence_id。不能引用 keyframe_refs、keyframe_id、"
            "visual_content_id、visual_fact_id、source_keyframe_refs；只有 "
            "VisualBlock.visual_content_refs 才能引用视觉内容 ID，并且必须属于该块的 "
            "visual_observation_ref。"
            "所有 evidence_id 必须从输入白名单逐字复制，不得自行编造、截断或替换；"
            "输出前逐项检查每个引用都在上述白名单中，无法确认时删除该引用或对应事实。"
            "视觉观察可能没有 content_blocks 或 visual_facts；此时 "
            "visual_content_refs 必须为空数组，不能编造 ID。"
            "视觉观察缺失或不可用时，不要输出 VISUAL block，但必须继续根据 "
            "ASR/字幕生成正常正文和 claims。"
            "视觉块引用错误时只删除视觉块，不要删除同一响应中的 ASR 正文、摘要和 claims。"
            "只要本章存在可验证语义且 summary_zh 非空，必须返回 1-2 条 claims；"
            "每条 claim 都应是有证据支持的独立关键结论，不要只重复章节标题。"
        ),
        request.model_dump(mode="json"),
    )


def prompt_for_writing_repair(request: ChapterWritingRepairRequest) -> tuple[str, str, str]:
    return _prompt(
        request.prompt_version,
        (
            "只修复章节正文结构和证据引用；不得扩展原请求证据范围。"
            "每个 body_blocks 项都必须包含 block_type，且只能是 PARAGRAPH、BULLET_LIST、"
            "QUOTE、CODE、TABLE、FORMULA 或 VISUAL，并填写该类型要求的字段。"
            "title_evidence_refs、summary_evidence_refs、普通正文 evidence_refs 和 "
            "claims.evidence_refs 只能引用 ASR/字幕 evidence_id 或 "
            "visual_observation.evidence_id。不能引用 keyframe_refs、keyframe_id、"
            "visual_content_id、visual_fact_id、source_keyframe_refs；只有 "
            "VisualBlock.visual_content_refs 才能引用视觉内容 ID，并且必须属于该块的 "
            "visual_observation_ref。"
            "所有 evidence_id 必须从输入白名单逐字复制，不得自行编造、截断或替换；"
            "输出前逐项检查每个引用都在上述白名单中，无法确认时删除该引用或对应事实。"
            "视觉观察可能没有 content_blocks 或 visual_facts；此时 "
            "visual_content_refs 必须为空数组，不能编造 ID。"
            "视觉观察缺失或不可用时，不要输出 VISUAL block，但必须继续根据 "
            "ASR/字幕生成正常正文和 claims。"
            "视觉块引用错误时只删除视觉块，不要删除同一响应中的 ASR 正文、摘要和 claims。"
            "只要本章存在可验证语义且 summary_zh 非空，必须返回 1-2 条 claims；"
            "每条 claim 都应是有证据支持的独立关键结论，不要只重复章节标题。"
        ),
        request.model_dump(mode="json"),
    )


def prompt_for_global_editing(request: GlobalWritingRequest) -> tuple[str, str, str]:
    return _prompt(
        request.prompt_version,
        "你只能整理已验证章节；不得新增章节、证据或关键帧 ID；"
        "只返回 overview_zh；必须返回非空的中文核心概览，概括输入中的事实章节。",
        request.model_dump(mode="json"),
    )


def prompt_for_global_repair(request: GlobalWritingRepairRequest) -> tuple[str, str, str]:
    return _prompt(
        request.prompt_version,
        "只修复全局摘要结构；不得新增、遗漏、重复或重排章节；"
        "只返回 overview_zh；必须返回非空的中文核心概览，概括输入中的事实章节。",
        request.model_dump(mode="json"),
    )


def _vision_context(request: ChapterVisionRequest) -> dict[str, object]:
    frame_descriptors = [
        {
            "frame_id": frame.frame_id,
            "timestamp_ms": frame.timestamp_ms,
            "target_ids": frame.target_ids,
        }
        for frame in sorted(request.frames, key=lambda item: (item.timestamp_ms, item.frame_id))
    ]
    context = request.model_dump(mode="json", exclude={"frames"})
    context["frames"] = frame_descriptors
    context["target_frame_bindings"] = [
        {
            "target_id": target.target_id,
            "eligible_frame_ids": [
                frame["frame_id"]
                for frame in frame_descriptors
                if target.target_id in frame["target_ids"]  # type: ignore[operator]
            ],
        }
        for target in request.targets
    ]
    return context


def _vision_content(
    data: str,
    ordered_encoded_frames: tuple[tuple[str, str], ...],
) -> list[dict[str, object]]:
    content: list[dict[str, object]] = [
        {"type": "text", "text": "UNTRUSTED_VISION_CONTEXT_JSON\n" + data},
    ]
    for frame_id, encoded in ordered_encoded_frames:
        content.append({"type": "text", "text": f"FRAME_ID={frame_id}"})
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
            },
        )
    return content


def _prompt(version: str, instruction: str, value: Any) -> tuple[str, str, str]:
    return (
        version,
        instruction,
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
    )
