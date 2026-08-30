from __future__ import annotations

import json
from typing import Any

from video_demo.integrations.audio_document_port import (
    AudioChapterPlanningRequest,
    AudioChapterPlanRepairRequest,
    AudioChapterWritingRepairRequest,
    AudioChapterWritingRequest,
    AudioGlobalWritingRepairRequest,
    AudioGlobalWritingRequest,
)


def _prompt(version: str, instruction: str, value: Any) -> tuple[str, str, str]:
    return (
        version,
        instruction,
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
    )


def prompt_for_audio_planning(request: AudioChapterPlanningRequest) -> tuple[str, str, str]:
    return _prompt(
        request.prompt_version,
        (
            "只根据连续音频片段和 ASR/字幕证据规划章节。使用数组索引返回范围："
            "start_segment_index 包含、end_segment_index 不包含，所有范围必须按顺序从 0 开始；"
            "普通章节 end_segment_index 不得超过 segments 长度，最后一个章节允许暂时使用"
            "segments 长度加 1，程序会归一化为 segments 长度。"
            "例如 6 个 segments 必须返回 [0,2]、[2,4]、[4,6] 这样的连续范围，"
            "不能返回 [0,1]、[2,3]、[4,5]，不能跳过任何下标，不能把 end_segment_index"
            "当作最后一个元素下标。每章不得超过 300 秒；标题只概括本章证据。"
            "返回前逐章检查范围连续、完整覆盖且每个片段只使用一次。只返回 JSON。"
        ),
        _planning_context(request),
    )


def prompt_for_audio_plan_repair(request: AudioChapterPlanRepairRequest) -> tuple[str, str, str]:
    return _prompt(
        request.prompt_version,
        "只修复音频章节范围和结构，必须按原片段顺序完整覆盖，不添加原证据之外的事实。只返回 JSON。",
        {
            "request": _planning_context(request.request),
            "invalid_response": request.invalid_response.model_dump(mode="json"),
        },
    )


def prompt_for_audio_writing(request: AudioChapterWritingRequest) -> tuple[str, str, str]:
    return _prompt(
        request.prompt_version,
        (
            "只使用输入的 ASR/字幕证据写作音频章节。正文只能使用 PARAGRAPH、"
            "BULLET_LIST、QUOTE、CODE、TABLE、FORMULA；所有引用必须逐字来自输入 "
            "evidence_id，不得编造事实。证据 ID 只能放在 *_evidence_refs 字段，"
            "不得写入标题、摘要、正文文本或结论文本；正文按语义合并为自然段，"
            "不要把每条 ASR 句子逐条复制成列表。返回标题、摘要、正文和本章关键结论。只返回 JSON。"
        ),
        request.model_dump(mode="json"),
    )


def prompt_for_audio_writing_repair(
    request: AudioChapterWritingRepairRequest,
) -> tuple[str, str, str]:
    return _prompt(
        request.prompt_version,
        (
            "只修复音频章节文字结构和 ASR/字幕引用，不得添加事实。"
            "证据 ID 只能放在 *_evidence_refs 字段，不得写入正文或结论文本。只返回 JSON。"
        ),
        {
            "request": request.request.model_dump(mode="json"),
            "invalid_response": request.invalid_response.model_dump(mode="json"),
        },
    )


def prompt_for_audio_global(request: AudioGlobalWritingRequest) -> tuple[str, str, str]:
    return _prompt(
        request.prompt_version,
        (
            "根据已验证的音频章节标题和摘要生成中文核心概览；不得添加未出现的事实，"
            "只返回 overview_zh。"
        ),
        {
            "title_hint": request.title_hint,
            "duration_ms": request.duration_ms,
            "chapters": [item.model_dump(mode="json") for item in request.chapters],
        },
    )


def prompt_for_audio_global_repair(
    request: AudioGlobalWritingRepairRequest,
) -> tuple[str, str, str]:
    return _prompt(
        request.prompt_version,
        (
            "只修复音频核心概览的文字结构；只能依据已验证的章节标题和摘要，"
            "不得添加事实。只返回 overview_zh。"
        ),
        {
            "title_hint": request.request.title_hint,
            "duration_ms": request.request.duration_ms,
            "chapters": [
                item.model_dump(mode="json") for item in request.request.chapters
            ],
            "invalid_response": request.invalid_response.model_dump(mode="json"),
        },
    )


def _planning_context(request: AudioChapterPlanningRequest) -> dict[str, object]:
    evidence_index = {
        item.evidence_id: index for index, item in enumerate(request.transcript_evidence)
    }
    return {
        "title_hint": request.title_hint,
        "duration_ms": request.duration_ms,
        "chapter_granularity": request.document_config.chapter_granularity,
        "segments": [
            [
                index,
                item.start_ms,
                item.end_ms,
                [evidence_index[ref] for ref in item.evidence_refs],
            ]
            for index, item in enumerate(request.segments)
        ],
        "transcript_evidence": [
            [index, item.start_ms, item.end_ms, item.text]
            for index, item in enumerate(request.transcript_evidence)
        ],
    }
