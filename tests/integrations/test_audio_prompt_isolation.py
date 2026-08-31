from __future__ import annotations

from video_demo.domain.audio_plan import AudioBaseSegment, AudioDocumentConfig
from video_demo.domain.evidence import SpeechSegment
from video_demo.integrations.audio_document_port import (
    AudioChapterPlanningRequest,
    AudioChapterPlanRepairRequest,
    AudioInvalidModelResponse,
)
from video_demo.integrations.audio_document_prompts import (
    prompt_for_audio_plan_repair,
    prompt_for_audio_planning,
)


def test_audio_prompt_contains_only_transcript_partition_data() -> None:
    request = AudioChapterPlanningRequest(
        title_hint="音频",
        duration_ms=1_000,
        segments=(
            AudioBaseSegment(
                segment_id="audio_segment_001",
                start_ms=0,
                end_ms=1_000,
                evidence_refs=("asr_001",),
                transcript_source="ASR",
            ),
        ),
        transcript_evidence=(
            SpeechSegment(
                evidence_id="asr_001",
                start_ms=0,
                end_ms=1_000,
                text="音频内容",
                language="zh",
                confidence=0.9,
                is_fully_evaluated_language=True,
            ),
        ),
        document_config=AudioDocumentConfig(),
        prompt_version="audio-chapter-planner-v1",
    )

    version, instruction, data = prompt_for_audio_planning(request)
    combined = f"{version}\n{instruction}\n{data}".lower()

    for forbidden in (
        "visual_mode",
        "semantic_targets",
        "base_coverage_targets",
        "keyframe",
        "scene",
        "vlm",
    ):
        assert forbidden not in combined


def test_audio_plan_repair_prompt_repeats_complete_range_contract() -> None:
    request = AudioChapterPlanningRequest(
        title_hint="音频",
        duration_ms=1_000,
        segments=(
            AudioBaseSegment(
                segment_id="audio_segment_001",
                start_ms=0,
                end_ms=1_000,
                evidence_refs=("asr_001",),
                transcript_source="ASR",
            ),
        ),
        transcript_evidence=(
            SpeechSegment(
                evidence_id="asr_001",
                start_ms=0,
                end_ms=1_000,
                text="音频内容",
                language="zh",
                confidence=0.9,
                is_fully_evaluated_language=True,
            ),
        ),
        document_config=AudioDocumentConfig(),
        prompt_version="audio-chapter-planner-v1",
    )
    repair = AudioChapterPlanRepairRequest(
        request=request,
        invalid_response=AudioInvalidModelResponse(
            content_sha256="a" * 64,
            validation_errors=("segment_range:out_of_bounds",),
        ),
        allowed_segment_ids=("audio_segment_001",),
        prompt_version="audio-chapter-planner-repair-v1",
    )

    _, instruction, _ = prompt_for_audio_plan_repair(repair)

    assert "普通章节 end_segment_index 不得超过 segments 长度" in instruction
    assert "最后一个章节最多允许 segments 长度加 1" in instruction
    assert "归一化为 segments 长度" in instruction
    assert "0 ≤ start_segment_index < end_segment_index" in instruction
    assert "连续、完整覆盖" in instruction
    assert "不可重复消费" in instruction
