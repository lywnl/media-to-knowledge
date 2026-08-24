from __future__ import annotations

import json
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


def prompt_for_planning(request: ChapterPlanningRequest) -> tuple[str, str, str]:
    return _prompt(
        request.prompt_version,
        "你只能根据输入的基础片段和转写证据规划连续章节；不得生成时间、章节 ID 或未知引用。",
        request.model_dump(mode="json"),
    )


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
