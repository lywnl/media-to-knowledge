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
            "300 秒。不得生成时间、章节 ID 或未知引用。"
        ),
        _planning_context(request),
    )


def _planning_context(request: ChapterPlanningRequest) -> dict[str, object]:
    """只向章节规划模型发送决策所需字段，完整请求仍由程序保留并校验。

    语言、置信度、场景引用和视觉配置不会改变章节边界，却会随每条转写重复
    占用上下文。删掉这些冗余字段可以让更多视频在一次规划请求内完成，同时不
    改变缓存指纹或程序侧的证据闭包校验。
    """

    return {
        "title_hint": request.title_hint,
        "duration_ms": request.duration_ms,
        "chapter_granularity": request.document_config.chapter_granularity,
        "segments": tuple(
            {
                "segment_id": segment.segment_id,
                "start_ms": segment.start_ms,
                "end_ms": segment.end_ms,
                "evidence_refs": segment.evidence_refs,
            }
            for segment in request.segments
        ),
        "transcript_evidence": tuple(
            {
                "evidence_id": evidence.evidence_id,
                "start_ms": evidence.start_ms,
                "end_ms": evidence.end_ms,
                "text": evidence.text,
            }
            for evidence in request.transcript_evidence
        ),
    }


def prompt_for_plan_repair(request: ChapterPlanRepairRequest) -> tuple[str, str, str]:
    return _prompt(
        request.prompt_version,
        "只修复结构和引用；必须保留原请求的事实边界，不得添加新事实。",
        request.model_dump(mode="json"),
    )


def prompt_for_vision(request: ChapterVisionRequest) -> tuple[str, str, str]:
    return _prompt(
        request.prompt_version,
        "你只能比较输入的本地图片，并且只能返回输入中的 frame_id、target_id 和 evidence_id。",
        _vision_context(request),
    )


def prompt_for_vision_repair(request: ChapterVisionRepairRequest) -> tuple[str, str, str]:
    context = request.model_dump(mode="json", exclude={"request"})
    context["request"] = _vision_context(request.request)
    return _prompt(
        request.prompt_version,
        "只修复结构和引用；图片顺序和原始取证问题不得改变，不得添加新事实。",
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
        "你只能使用输入证据写作；不得发明事实、时间、章节 ID 或关键帧 ID。",
        request.model_dump(mode="json"),
    )


def prompt_for_writing_repair(request: ChapterWritingRepairRequest) -> tuple[str, str, str]:
    return _prompt(
        request.prompt_version,
        "只修复章节正文结构和证据引用；不得扩展原请求证据范围。",
        request.model_dump(mode="json"),
    )


def prompt_for_global_editing(request: GlobalWritingRequest) -> tuple[str, str, str]:
    return _prompt(
        request.prompt_version,
        "你只能整理已验证章节分组；不得新增章节、证据或关键帧 ID。",
        request.model_dump(mode="json"),
    )


def prompt_for_global_repair(request: GlobalWritingRepairRequest) -> tuple[str, str, str]:
    return _prompt(
        request.prompt_version,
        "只修复全局摘要和 Section 结构；不得新增、遗漏、重复或重排章节。",
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
