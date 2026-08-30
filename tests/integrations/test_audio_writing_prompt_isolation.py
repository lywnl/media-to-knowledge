from __future__ import annotations

from video_demo.domain.audio_plan import AudioChapterPlan, AudioDocumentConfig
from video_demo.domain.evidence import SpeechSegment
from video_demo.integrations.audio_document_port import AudioChapterWritingRequest
from video_demo.integrations.audio_document_prompts import prompt_for_audio_writing


def test_audio_writing_prompt_has_no_non_audio_fields() -> None:
    request = AudioChapterWritingRequest(
        run_id="run_audio_001",
        asset_sha256="a" * 64,
        title_hint="音频",
        duration_ms=1_000,
        transcript_source="ASR",
        document_config=AudioDocumentConfig(),
        chapter=AudioChapterPlan(
            chapter_id="audio_chapter_001",
            start_ms=0,
            end_ms=1_000,
            segment_refs=("audio_segment_001",),
            title_hint="主题",
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
        prompt_version="audio-chapter-writer-v1",
    )

    _version, instruction, data = prompt_for_audio_writing(request)
    combined = f"{instruction}\n{data}".lower()
    for forbidden in ("visual", "keyframe", "scene", "vlm", "video"):
        assert forbidden not in combined
