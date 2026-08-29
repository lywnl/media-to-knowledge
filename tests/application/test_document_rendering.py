from __future__ import annotations

import hashlib

import pytest

from video_demo.application.document_rendering import render_markdown
from video_demo.domain.document import (
    BulletListBlock,
    CodeBlock,
    DocumentGenerationConfig,
    DocumentGenerationMetadata,
    FormulaBlock,
    GroundedClaim,
    ParagraphBlock,
    PromptVersions,
    QuoteBlock,
    SemanticChapter,
    TableBlock,
    VideoDocumentSummary,
    VideoUnderstandingResult,
    VisualBlock,
)
from video_demo.domain.evidence import (
    DocumentEvidenceItem,
    KeyframeEvidence,
    SpeechSegment,
    VisualObservationEvidence,
    VisualTextContent,
)
from video_demo.errors import ErrorCode, VideoDemoError

ASSET_SHA256 = "a" * 64
KEYFRAME_SHA256 = "b" * 64


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _metadata() -> DocumentGenerationMetadata:
    return DocumentGenerationMetadata(
        document_config=DocumentGenerationConfig(),
        text_model_id="text-model",
        vlm_model_id="qwen3-vl-flash",
        prompt_versions=PromptVersions(
            chapter_planner="chapter-planner-v1",
            chapter_planner_repair="chapter-planner-repair-v1",
            chapter_vlm="chapter-vlm-v1",
            chapter_vlm_repair="chapter-vlm-repair-v1",
            chapter_writer="chapter-writer-v1",
            chapter_writer_repair="chapter-writer-repair-v1",
            global_editor="global-editor-v1",
            global_editor_repair="global-editor-repair-v1",
        ),
    )


def _document_fixture() -> tuple[VideoUnderstandingResult, tuple[DocumentEvidenceItem, ...]]:
    speech = SpeechSegment(
        evidence_id="asr_001",
        start_ms=0,
        end_ms=8_000,
        text="讲解模型参数。",
        language="zh",
        confidence=0.95,
        is_fully_evaluated_language=True,
    )
    keyframe = KeyframeEvidence(
        evidence_id="keyframe_evidence_001",
        start_ms=5_000,
        end_ms=5_001,
        keyframe_id="keyframe_001",
        timestamp_ms=5_000,
        relative_path=f"visual/keyframes/{KEYFRAME_SHA256}.jpg",
        mime_type="image/jpeg",
        sha256=KEYFRAME_SHA256,
        perceptual_hash="0123456789abcdef",
        size_bytes=1_024,
    )
    observation = VisualObservationEvidence(
        evidence_id="visual_001",
        chapter_id="chapter_001",
        start_ms=4_500,
        end_ms=5_500,
        target_ids=("target_001",),
        keyframe_refs=(keyframe.evidence_id,),
        transcript_evidence_refs=(speech.evidence_id,),
        visual_type="TEXT",
        caption="画面给出了更精确的参数。",
        content_blocks=(
            VisualTextContent(
                visual_content_id="visual_content_001",
                source_keyframe_refs=(keyframe.evidence_id,),
                text="beam_size=5",
            ),
        ),
        relation_to_transcript="CONFLICTING",
        certainty=0.65,
    )
    chapter = SemanticChapter(
        chapter_id="chapter_001",
        start_ms=0,
        end_ms=10_000,
        title="安装 #1 <script>",
        title_evidence_refs=(speech.evidence_id,),
        summary_zh="本章说明安装与参数。",
        summary_evidence_refs=(speech.evidence_id,),
        body_blocks=(
            ParagraphBlock(
                text="<script>alert(1)</script>\n# 不能成为标题 | 保留竖线",
                evidence_refs=(speech.evidence_id,),
            ),
            BulletListBlock(
                items=("安装依赖", "<b>不要执行原始 HTML</b>"),
                evidence_refs=(speech.evidence_id,),
            ),
            QuoteBlock(
                text="原话第一行\n> 不能嵌套注入",
                evidence_refs=(speech.evidence_id,),
            ),
            CodeBlock(
                language="python",
                code='print("```")\nbeam_size = 5',
                evidence_refs=(speech.evidence_id,),
            ),
            TableBlock(
                columns=("参数|名", "<script>值</script>"),
                rows=(("beam_size", "5|推荐\n第二行"),),
                evidence_refs=(speech.evidence_id,),
            ),
            FormulaBlock(
                latex="x = y + 1",
                explanation="简单关系式。",
                evidence_refs=(speech.evidence_id,),
            ),
            VisualBlock(
                visual_observation_ref=observation.evidence_id,
                visual_content_refs=("visual_content_001",),
                caption=observation.caption,
                evidence_refs=(observation.evidence_id,),
            ),
        ),
        claims=(
            GroundedClaim(
                text="默认参数可以调整。",
                evidence_refs=(speech.evidence_id,),
                certainty=0.9,
            ),
        ),
        evidence_refs=(speech.evidence_id, keyframe.evidence_id, observation.evidence_id),
        selected_keyframe_refs=(keyframe.evidence_id,),
        transcript_source="ASR",
    )
    result = VideoUnderstandingResult(
        run_id="run_document_rendering_001",
        asset_sha256=ASSET_SHA256,
        summary=VideoDocumentSummary(
            title="faster-whisper <入门>",
            duration_ms=chapter.end_ms,
            overview_zh="介绍 faster-whisper 的安装和参数。",
        ),
        chapters=(chapter,),
        generation=_metadata(),
    )
    return result, (speech, keyframe, observation)


def test_render_markdown_is_deterministic_utf8_and_has_fixed_structure() -> None:
    result, evidence = _document_fixture()

    first = render_markdown(result, evidence)
    second = render_markdown(result, evidence)
    markdown = first.content.decode("utf-8")

    assert first == second
    assert first.media_type == "text/markdown; charset=utf-8"
    assert first.sha256 == hashlib.sha256(first.content).hexdigest()
    assert first.size_bytes == len(first.content)
    assert markdown.endswith("\n") and not markdown.endswith("\n\n")

    ordered_markers = (
        "# faster-whisper",
        "## 核心概览",
        "## 目录",
        (
            "- 第一章：安装 \\#1 &lt;script&gt; "
            "\N{FULLWIDTH LEFT PARENTHESIS}00:00:00 - 00:00:10"
            "\N{FULLWIDTH RIGHT PARENTHESIS}"
        ),
        "## 第一章：安装 \\#1 &lt;script&gt;",
        "不能成为标题",
    )
    positions = tuple(markdown.index(marker) for marker in ordered_markers)
    assert positions == tuple(sorted(positions))
    assert "## 全文关键结论" not in markdown
    assert "关键画面与引用" not in markdown
    assert "video-demo-keyframe:keyframe_001" not in markdown
    assert markdown.count("安装 \\#1") == 2
    assert markdown.count("00:00:00 - 00:00:10") == 2


def test_render_markdown_escapes_untrusted_text_and_renders_every_block_type() -> None:
    result, evidence = _document_fixture()

    markdown = render_markdown(result, evidence).content.decode("utf-8")

    assert "<script>" not in markdown
    assert "<img" not in markdown
    assert "<b>" not in markdown
    assert "<br>" not in markdown
    assert "&lt;script&gt;" in markdown
    assert r"\# 不能成为标题 \| 保留竖线" in markdown
    assert "- 安装依赖" in markdown
    assert "> 原话第一行" in markdown
    assert '\n````python\nprint("```")\nbeam_size = 5\n````\n' in markdown
    assert r"| 参数\|名 | &lt;script&gt;值&lt;/script&gt; |" in markdown
    assert r"5\|推荐 / 第二行" in markdown
    assert "$$\nx = y + 1\n$$" in markdown
    assert "画面给出了更精确的参数。" in markdown


def test_render_markdown_keeps_visual_body_without_leaking_storage_path() -> None:
    result, evidence = _document_fixture()

    markdown = render_markdown(result, evidence).content.decode("utf-8")

    assert "**视觉补充：** 画面给出了更精确的参数。" in markdown
    assert "video-demo-keyframe:keyframe_001" not in markdown
    assert "图片处理：章节视觉模型已分析" not in markdown
    assert "视觉观察类型：TEXT" not in markdown
    assert f"visual/keyframes/{KEYFRAME_SHA256}.jpg" not in markdown
    assert KEYFRAME_SHA256 not in markdown


def test_render_markdown_deduplicates_single_chapter_section_summary() -> None:
    result, evidence = _document_fixture()
    chapter = result.chapters[0].model_copy(
        update={
            "title": "AI自动化小红书账号案例介绍",
            "summary_zh": "本章概要只应展示一次。",
        },
    )
    result = result.model_copy(update={"chapters": (chapter,)})

    markdown = render_markdown(result, evidence).content.decode("utf-8")
    body = markdown.split("## 第一章：", maxsplit=1)[1]

    assert body.count("AI自动化小红书账号案例介绍") == 1
    assert body.count("本章概要只应展示一次。") == 1
    assert "### 00:00:00 - 00:00:10 AI自动化小红书账号案例介绍" not in body
    assert "时间：00:00:00 - 00:00:10" in body
    assert "不能成为标题" in body


def test_render_markdown_renders_chapter_claims() -> None:
    result, evidence = _document_fixture()

    markdown = render_markdown(result, evidence).content.decode("utf-8")

    assert "#### 本章结论" in markdown
    assert "- 默认参数可以调整。" in markdown


def test_render_markdown_keeps_chapter_summary_when_section_has_multiple_chapters() -> None:
    result, evidence = _document_fixture()
    first = result.chapters[0].model_copy(
        update={"chapter_id": "chapter_001", "title": "第一章", "summary_zh": "第一章概要。"},
    )
    second = first.model_copy(
        update={
            "chapter_id": "chapter_002",
            "start_ms": 10_000,
            "end_ms": 20_000,
            "title": "第二章",
            "summary_zh": "第二章概要。",
            "title_evidence_refs": (),
            "summary_evidence_refs": (),
            "body_blocks": (),
            "claims": (),
            "evidence_refs": (),
            "selected_keyframe_refs": (),
            "transcript_source": "NONE",
        },
    )
    result = result.model_copy(update={"chapters": (first, second)})

    markdown = render_markdown(result, evidence).content.decode("utf-8")

    assert "## 第一章：第一章" in markdown
    assert "第一章概要。" in markdown
    assert "## 第二章：第二章" in markdown
    assert "第二章概要。" in markdown


def test_render_markdown_revalidates_selected_keyframe_evidence() -> None:
    result, evidence = _document_fixture()
    chapter = result.chapters[0].model_copy(
        update={"selected_keyframe_refs": ("keyframe_missing",)},
    )
    result_with_missing_frame = result.model_copy(update={"chapters": (chapter,)})

    with pytest.raises(VideoDemoError) as missing:
        render_markdown(result_with_missing_frame, evidence)
    assert missing.value.code == ErrorCode.UNKNOWN_EVIDENCE_REFERENCE

    keyframe = evidence[1]
    assert isinstance(keyframe, KeyframeEvidence)
    invalid_keyframe = keyframe.model_copy(update={"relative_path": "visual/keyframes/wrong.jpg"})
    with pytest.raises(VideoDemoError) as invalid:
        render_markdown(result, (evidence[0], invalid_keyframe, evidence[2]))
    assert invalid.value.code == ErrorCode.EVIDENCE_RELATION_INVALID


def test_no_semantic_evidence_placeholder_is_rendered_without_information_boundary() -> None:
    result, _evidence = _document_fixture()
    placeholder = "本时段未提取到可验证语义内容"
    chapter = SemanticChapter(
        chapter_id="chapter_empty",
        start_ms=0,
        end_ms=10_000,
        title=placeholder,
        title_evidence_refs=(),
        summary_zh=placeholder,
        summary_evidence_refs=(),
        body_blocks=(),
        claims=(),
        content_status="NO_SEMANTIC_EVIDENCE",
        evidence_refs=(),
        transcript_source="NONE",
    )
    empty_result = result.model_copy(
        update={
            "summary": result.summary,
            "chapters": (chapter,),
        },
    )

    markdown = render_markdown(empty_result, ()).content.decode("utf-8")

    assert "## 第一章" in markdown
    assert placeholder in markdown
    assert "信息边界" not in markdown
