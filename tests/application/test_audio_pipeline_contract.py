from video_demo.application.audio_rendering import render_audio_markdown
from video_demo.domain.audio_document import (
    AudioChapter,
    AudioDocumentSummary,
    AudioUnderstandingResult,
)
from video_demo.domain.document import GroundedClaim, ParagraphBlock


def _result() -> AudioUnderstandingResult:
    evidence_id = "asr_evidence_001"
    return AudioUnderstandingResult(
        run_id="run_audio_001",
        asset_sha256="a" * 64,
        summary=AudioDocumentSummary(
            title="音频标题",
            duration_ms=1_000,
            overview_zh="音频概览",
        ),
        chapters=(
            AudioChapter(
                start_ms=0,
                end_ms=1_000,
                chapter_id="audio_chapter_001",
                title="第一章",
                title_evidence_refs=(evidence_id,),
                summary_zh="章节摘要",
                summary_evidence_refs=(evidence_id,),
                body_blocks=(
                    ParagraphBlock(text="语音正文", evidence_refs=(evidence_id,)),
                ),
                claims=(
                    GroundedClaim(
                        text="关键结论",
                        evidence_refs=(evidence_id,),
                        certainty=0.9,
                    ),
                ),
                evidence_refs=(evidence_id,),
                transcript_source="ASR",
            ),
        ),
    )


def test_audio_markdown_is_text_only_and_keeps_chapter_claim_heading() -> None:
    rendered = render_audio_markdown(_result())
    text = rendered.content.decode("utf-8")

    assert "## 核心概览" in text
    assert "## 目录" in text
    assert "## 第一章：第一章" in text
    assert "视觉" not in text
    assert "关键帧" not in text
    assert "retrieval_text" not in text


def test_audio_result_contract_has_no_visual_or_rag_fields() -> None:
    payload = _result().model_dump(mode="json")
    serialized = str(payload)
    assert "retrieval_text" not in serialized
    assert "retrieval_hash" not in serialized
    assert "keyframe" not in serialized.lower()


def test_media_worker_entrypoints_keep_video_job_type_isolated() -> None:
    from video_demo.application.composition import build_audio_worker, build_image_worker

    assert callable(build_audio_worker)
    assert callable(build_image_worker)
