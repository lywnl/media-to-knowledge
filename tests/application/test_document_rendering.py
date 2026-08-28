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
    SemanticSection,
    SummaryPoint,
    TableBlock,
    VideoDocumentSummary,
    VideoUnderstandingResult,
    VisualBlock,
    section_id_for,
    visual_caption_for_policy,
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
        quality_flags=("右下角轻微模糊",),
        uncertainties=("屏幕参数与口述参数可能不一致",),
    )
    retrieval_text = "章节检索正文"
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
                caption=visual_caption_for_policy(observation, DocumentGenerationConfig()),
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
        retrieval_text=retrieval_text,
        retrieval_hash=_sha256(retrieval_text),
    )
    section = SemanticSection(
        section_id=section_id_for(ASSET_SHA256, (chapter.chapter_id,)),
        title="准备 | 安装",
        summary_zh="准备开发环境。",
        chapter_refs=(chapter.chapter_id,),
    )
    summary_retrieval = "视频检索摘要"
    result = VideoUnderstandingResult(
        run_id="run_document_rendering_001",
        asset_sha256=ASSET_SHA256,
        summary=VideoDocumentSummary(
            title="faster-whisper <入门>",
            duration_ms=chapter.end_ms,
            overview_zh="介绍 faster-whisper 的安装和参数。",
            key_points=(
                SummaryPoint(text="掌握安装流程。", chapter_refs=(chapter.chapter_id,)),
            ),
            retrieval_text=summary_retrieval,
            retrieval_hash=_sha256(summary_retrieval),
        ),
        sections=(section,),
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
            "- 第一部分：准备 \\| 安装 "
            "\N{FULLWIDTH LEFT PARENTHESIS}00:00:00 - 00:00:10"
            "\N{FULLWIDTH RIGHT PARENTHESIS}"
        ),
        "## 第一部分：",
        "### 00:00:00 - 00:00:10",
        "不能成为标题",
        "## 信息边界",
    )
    positions = tuple(markdown.index(marker) for marker in ordered_markers)
    assert positions == tuple(sorted(positions))
    assert "00:00:05" in markdown
    assert "## 关键结论" not in markdown
    assert "关键画面与引用" not in markdown
    assert "video-demo-keyframe:keyframe_001" not in markdown
    assert markdown.count("准备 \\| 安装") == 2
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
    assert "音画信息存在冲突" in markdown


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
    section = result.sections[0].model_copy(
        update={
            "title": chapter.title,
            "summary_zh": chapter.summary_zh,
        },
    )
    result = result.model_copy(update={"chapters": (chapter,), "sections": (section,)})

    markdown = render_markdown(result, evidence).content.decode("utf-8")
    body = markdown.split("## 第一部分：", maxsplit=1)[1].split(
        "## 信息边界\n",
        maxsplit=1,
    )[0]

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
            "retrieval_text": "",
            "retrieval_hash": _sha256(""),
        },
    )
    section = result.sections[0].model_copy(
        update={
            "chapter_refs": (first.chapter_id, second.chapter_id),
            "section_id": section_id_for(ASSET_SHA256, (first.chapter_id, second.chapter_id)),
            "title": "合并部分",
            "summary_zh": "合并部分概要。",
        },
    )
    result = result.model_copy(
        update={"chapters": (first, second), "sections": (section,)},
    )

    markdown = render_markdown(result, evidence).content.decode("utf-8")

    assert "### 00:00:00 - 00:00:10 第一章" in markdown
    assert "第一章概要。" in markdown
    assert "### 00:00:10 - 00:00:20 第二章" in markdown
    assert "第二章概要。" in markdown


def test_render_markdown_summarizes_visual_information_boundaries() -> None:
    result, evidence = _document_fixture()
    observation = evidence[2]
    assert isinstance(observation, VisualObservationEvidence)

    markdown = render_markdown(result, evidence).content.decode("utf-8")
    boundary = markdown.split("## 信息边界\n", maxsplit=1)[1]

    assert "屏幕参数与口述参数可能不一致" in boundary
    assert "右下角轻微模糊" in boundary
    assert "画面与转写存在冲突" in boundary
    assert markdown.count(observation.caption) == 1


def test_conservative_policy_moves_low_confidence_visual_caption_to_boundary() -> None:
    result, evidence = _document_fixture()
    observation = evidence[2]
    assert isinstance(observation, VisualObservationEvidence)
    observation = observation.model_copy(
        update={
            "relation_to_transcript": "COMPLEMENTARY",
            "certainty": 0.6,
        },
    )
    chapter = result.chapters[0].model_copy(
        update={
            "body_blocks": tuple(
                block
                for block in result.chapters[0].body_blocks
                if not isinstance(block, VisualBlock)
            ),
            "selected_keyframe_refs": (),
        },
    )
    result = result.model_copy(
        update={
            "chapters": (chapter,),
            "generation": result.generation.model_copy(
                update={
                    "document_config": DocumentGenerationConfig(
                        uncertainty_policy="conservative",
                    ),
                },
            ),
        },
    )

    markdown = render_markdown(
        result,
        (evidence[0], evidence[1], observation),
    ).content.decode("utf-8")
    before_boundary, boundary = markdown.split("## 信息边界\n", maxsplit=1)

    assert observation.caption not in before_boundary
    assert f"低置信视觉观察，未纳入正文：{observation.caption}" in boundary


@pytest.mark.parametrize("relation", ["CONFLICTING", "COMPLEMENTARY"])
def test_information_boundary_includes_observation_omitted_from_chapter_evidence(
    relation: str,
) -> None:
    result, evidence = _document_fixture()
    observation = evidence[2]
    assert isinstance(observation, VisualObservationEvidence)
    observation = observation.model_copy(
        update={
            "relation_to_transcript": relation,
            "certainty": 0.6,
        },
    )
    chapter = result.chapters[0].model_copy(
        update={
            "body_blocks": tuple(
                block
                for block in result.chapters[0].body_blocks
                if not isinstance(block, VisualBlock)
            ),
            "evidence_refs": (evidence[0].evidence_id,),
            "selected_keyframe_refs": (),
        },
    )
    result = result.model_copy(
        update={
            "chapters": (chapter,),
            "generation": result.generation.model_copy(
                update={
                    "document_config": DocumentGenerationConfig(
                        uncertainty_policy="conservative",
                    ),
                },
            ),
        },
    )

    markdown = render_markdown(
        result,
        (evidence[0], evidence[1], observation),
    ).content.decode("utf-8")
    boundary = markdown.split("## 信息边界\n", maxsplit=1)[1]

    if relation == "CONFLICTING":
        assert "画面与转写存在冲突" in boundary
        assert observation.caption in boundary
    else:
        assert f"低置信视觉观察，未纳入正文：{observation.caption}" in boundary


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


def test_no_semantic_evidence_placeholder_only_appears_in_information_boundary() -> None:
    result, _evidence = _document_fixture()
    placeholder = "本时段未提取到可验证语义内容"
    boundary_message = "未提取到可验证语义内容"
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
        retrieval_text="",
        retrieval_hash=_sha256(""),
    )
    section = SemanticSection(
        section_id=section_id_for(ASSET_SHA256, (chapter.chapter_id,)),
        title="信息不足时段",
        summary_zh="",
        chapter_refs=(chapter.chapter_id,),
    )
    empty_result = result.model_copy(
        update={
            "summary": result.summary.model_copy(update={"key_points": ()}),
            "sections": (section,),
            "chapters": (chapter,),
        },
    )

    markdown = render_markdown(empty_result, ()).content.decode("utf-8")
    before_boundary, boundary = markdown.split("## 信息边界\n", maxsplit=1)

    assert placeholder not in before_boundary
    assert boundary.count(boundary_message) == 1
