from __future__ import annotations

import json
from typing import Any

from video_demo.integrations.audio_document_port import (
    AudioChapterPlanningRequest,
    AudioChapterPlanRepairRequest,
    AudioChapterWritingRepairRequest,
    AudioChapterWritingRequest,
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
            "只根据连续音频片段和 ASR/字幕证据规划章节。segment_refs 必须按输入顺序"
            "完整覆盖且恰好使用一次；每章不得超过 300 秒；标题只概括本章证据。只返回 JSON。"
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
            "evidence_id，不得编造事实。返回标题、摘要、正文和本章关键结论。只返回 JSON。"
        ),
        request.model_dump(mode="json"),
    )


def prompt_for_audio_writing_repair(
    request: AudioChapterWritingRepairRequest,
) -> tuple[str, str, str]:
    return _prompt(
        request.prompt_version,
        "只修复音频章节文字结构和 ASR/字幕引用，不得添加事实。只返回 JSON。",
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


def _planning_context(request: AudioChapterPlanningRequest) -> dict[str, object]:
    evidence_index = {
        item.evidence_id: index for index, item in enumerate(request.transcript_evidence)
    }
    return {
        "title_hint": request.title_hint,
        "duration_ms": request.duration_ms,
        "chapter_granularity": request.document_config.chapter_granularity,
        "segments": [
            [index, item.start_ms, item.end_ms, [evidence_index[ref] for ref in item.evidence_refs]]
            for index, item in enumerate(request.segments)
        ],
        "transcript_evidence": [
            [index, item.start_ms, item.end_ms, item.text]
            for index, item in enumerate(request.transcript_evidence)
        ],
    }
