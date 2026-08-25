from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, Lock
from typing import cast

import pytest

from video_demo.application.document_writing import DocumentWriter
from video_demo.application.pipeline_contracts import DocumentWritingContext
from video_demo.domain.document import (
    DocumentGenerationConfig,
    ParagraphBlock,
    PromptVersions,
    QuoteBlock,
    SectionDraft,
    SummaryPoint,
    VisualBlock,
)
from video_demo.domain.document_plan import ChapterPlan
from video_demo.domain.evidence import (
    GroundedVisualFact,
    KeyframeEvidence,
    SpeechSegment,
    VisualObservationEvidence,
    VisualTextContent,
)
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.integrations.document_port import (
    ChapterWritingRequest,
    ChapterWritingResponse,
    DocumentTextPort,
    GlobalWritingRequest,
    GlobalWritingResponse,
    InvalidModelResponse,
    ModelResponseValidationError,
)
from video_demo.storage.document_cache import DocumentModelCache, ModelInvocationIdentity

_ASSET_SHA256 = "a" * 64
_KEYFRAME_SHA256 = "b" * 64


def _context(config: DocumentGenerationConfig | None = None) -> DocumentWritingContext:
    return DocumentWritingContext(
        run_id="run_writing_001",
        asset_sha256=_ASSET_SHA256,
        title_hint="测试视频",
        duration_ms=20_000,
        transcript_source="ASR",
        document_config=config or DocumentGenerationConfig(),
    )


def _plan(index: int) -> ChapterPlan:
    return ChapterPlan(
        chapter_id=f"chapter_{index:03d}",
        start_ms=index * 10_000,
        end_ms=(index + 1) * 10_000,
        segment_refs=(f"segment_{index:03d}",),
        title_hint=f"章节 {index}",
        visual_mode="NONE",
        semantic_targets=(),
        base_coverage_targets=(),
    )


def _speech(index: int, text: str | None = None) -> SpeechSegment:
    return SpeechSegment(
        evidence_id=f"asr_{index:03d}",
        start_ms=index * 10_000,
        end_ms=index * 10_000 + 5_000,
        text=text or f"第 {index} 章转写内容。",
        language="zh",
        confidence=0.9,
        is_fully_evaluated_language=True,
    )


def _keyframe() -> KeyframeEvidence:
    return KeyframeEvidence(
        evidence_id="keyframe_evidence_001",
        start_ms=1_000,
        end_ms=1_001,
        keyframe_id="keyframe_001",
        timestamp_ms=1_000,
        relative_path=f"visual/keyframes/{_KEYFRAME_SHA256}.jpg",
        mime_type="image/jpeg",
        sha256=_KEYFRAME_SHA256,
        perceptual_hash="0123456789abcdef",
        size_bytes=100,
    )


def _observation() -> VisualObservationEvidence:
    return VisualObservationEvidence(
        evidence_id="visual_001",
        chapter_id="chapter_000",
        start_ms=900,
        end_ms=1_100,
        target_ids=("target_001",),
        keyframe_refs=("keyframe_evidence_001",),
        transcript_evidence_refs=("asr_000",),
        visual_type="TEXT",
        caption="画面显示参数 42。",
        content_blocks=(
            VisualTextContent(
                visual_content_id="visual_content_001",
                source_keyframe_refs=("keyframe_evidence_001",),
                text="42",
            ),
        ),
        relation_to_transcript="COMPLEMENTARY",
        certainty=0.9,
        uncertainties=("小字号可能识别有误",),
    )


def _chapter_response(request: ChapterWritingRequest) -> ChapterWritingResponse:
    evidence_ref = request.transcript_evidence[0].evidence_id
    blocks: tuple[ParagraphBlock | VisualBlock, ...] = (
        ParagraphBlock(text="根据转写整理正文。", evidence_refs=(evidence_ref,)),
    )
    if request.visual_observations:
        observation = request.visual_observations[0]
        observation_ref = observation.evidence_id
        content_refs = tuple(
            (
                *(item.visual_content_id for item in observation.content_blocks),
                *(item.visual_fact_id for item in observation.visual_facts),
            ),
        )
        blocks += (
            VisualBlock(
                visual_observation_ref=observation_ref,
                visual_content_refs=content_refs,
                caption="参数为 42。",
                evidence_refs=(observation_ref,),
            ),
        )
    return ChapterWritingResponse(
        title=request.chapter.title_hint,
        summary_zh="章节摘要",
        body_blocks=blocks,
        claims=(),
    )


def _global_response(request: GlobalWritingRequest) -> GlobalWritingResponse:
    refs = tuple(ref for group in request.groups for ref in group.chapter_refs)
    grounded = tuple(ref for group in request.groups for ref in group.grounded_chapter_refs)
    return GlobalWritingResponse(
        overview_zh="全局概览",
        key_points=(SummaryPoint(text="关键点", chapter_refs=(grounded[0],)),) if grounded else (),
        sections=(SectionDraft(title="完整内容", summary_zh="分节摘要", chapter_refs=refs),),
    )


class _TextPort:
    def __init__(
        self,
        chapter: Callable[[ChapterWritingRequest], ChapterWritingResponse | BaseException]
        = _chapter_response,
        repair: Callable[[ChapterWritingRequest], ChapterWritingResponse | BaseException]
        = _chapter_response,
        global_edit: Callable[[GlobalWritingRequest], GlobalWritingResponse | BaseException]
        = _global_response,
        global_repair: Callable[[GlobalWritingRequest], GlobalWritingResponse | BaseException]
        = _global_response,
    ) -> None:
        self.chapter = chapter
        self.repair = repair
        self.global_edit = global_edit
        self.global_repair = global_repair
        self.chapter_requests: list[ChapterWritingRequest] = []
        self.repair_requests: list[ChapterWritingRequest] = []
        self.global_requests: list[GlobalWritingRequest] = []
        self.global_repair_requests: list[GlobalWritingRequest] = []

    def write_chapter(
        self,
        request: ChapterWritingRequest,
        *,
        on_provider_attempt: Callable[[], None] | None = None,
    ) -> ChapterWritingResponse:
        self.chapter_requests.append(request)
        if on_provider_attempt:
            on_provider_attempt()
        return _raise_or_return(self.chapter(request))

    def repair_chapter_writing(
        self,
        request: object,
        *,
        on_provider_attempt: Callable[[], None] | None = None,
    ) -> ChapterWritingResponse:
        original = request.request  # type: ignore[attr-defined]
        self.repair_requests.append(original)
        if on_provider_attempt:
            on_provider_attempt()
        return _raise_or_return(self.repair(original))

    def organize_document(
        self,
        request: GlobalWritingRequest,
        *,
        on_provider_attempt: Callable[[], None] | None = None,
    ) -> GlobalWritingResponse:
        self.global_requests.append(request)
        if on_provider_attempt:
            on_provider_attempt()
        return _raise_or_return(self.global_edit(request))

    def repair_global_writing(
        self,
        request: object,
        *,
        on_provider_attempt: Callable[[], None] | None = None,
    ) -> GlobalWritingResponse:
        original = request.request  # type: ignore[attr-defined]
        self.global_repair_requests.append(original)
        if on_provider_attempt:
            on_provider_attempt()
        return _raise_or_return(self.global_repair(original))


def _raise_or_return(value: object) -> object:
    if isinstance(value, BaseException):
        raise value
    return value


def _identity(operation: str) -> ModelInvocationIdentity:
    return ModelInvocationIdentity(
        logical_operation=operation,
        provider_config_fingerprint="c" * 64,
        model_id="text-model",
        generation_config=(("temperature", "0"),),
        main_response_schema_name=f"{operation}_v1",
        main_prompt_version=(
            "chapter-writer-v1" if operation == "chapter_writing" else "global-editor-v1"
        ),
        repair_response_schema_name=f"{operation}_repair_v1",
        repair_prompt_version=(
            "chapter-writer-repair-v1"
            if operation == "chapter_writing"
            else "global-editor-repair-v1"
        ),
    )


def _writer(port: _TextPort, **overrides: object) -> DocumentWriter:
    values: dict[str, object] = {
        "text_model_id": "text-model",
        "vlm_model_id": "qwen3-vl-flash",
        "prompt_versions": PromptVersions(
            chapter_planner="chapter-planner-v1",
            chapter_planner_repair="chapter-planner-repair-v1",
            chapter_vlm="chapter-vlm-v1",
            chapter_vlm_repair="chapter-vlm-repair-v1",
            chapter_writer="chapter-writer-v1",
            chapter_writer_repair="chapter-writer-repair-v1",
            global_editor="global-editor-v1",
            global_editor_repair="global-editor-repair-v1",
        ),
        "chapter_identity": _identity("chapter_writing"),
        "global_identity": _identity("global_editing"),
        "chapter_writer_concurrency": 2,
        "max_input_chars": 60_000,
        "max_input_bytes": 1_048_576,
        "invocation_wait_timeout_seconds": 2.0,
    }
    values.update(overrides)
    return DocumentWriter(cast(DocumentTextPort, port), **values)  # type: ignore[arg-type]


def _cache(path: Path) -> DocumentModelCache:
    return DocumentModelCache(path, max_entry_bytes=1_048_576, max_run_bytes=4_194_304)


def test_writer_scopes_evidence_refills_images_and_reuses_cache(tmp_path: Path) -> None:
    port = _TextPort()
    writer = _writer(port)
    plans = (_plan(0), _plan(1))
    transcript = (_speech(0), _speech(1))

    first = writer.write(
        _context(),
        plans,
        transcript,
        (_observation(),),
        (_keyframe(),),
        cache=_cache(tmp_path),
        is_cancel_requested=lambda: False,
    )
    second = writer.write(
        _context(),
        plans,
        transcript,
        (_observation(),),
        (_keyframe(),),
        cache=_cache(tmp_path),
        is_cancel_requested=lambda: False,
    )

    actual_transcript_ids = [
        tuple(item.evidence_id for item in request.transcript_evidence)
        for request in port.chapter_requests
    ]
    assert sorted(actual_transcript_ids) == [
        ("asr_000",),
        ("asr_001",),
    ]
    assert first.result.chapters == second.result.chapters
    assert first.result.chapters[0].selected_keyframe_refs == ("keyframe_evidence_001",)
    assert "小字号可能识别有误" in first.result.chapters[0].retrieval_text
    assert "visual_001" not in first.result.chapters[0].retrieval_text
    assert first.metrics["chapter_writer_provider_attempts"] == 2
    assert second.metrics["chapter_writer_cache_hits"] == 2
    assert second.metrics["global_editor_cache_hits"] == 1


def test_visual_block_refills_only_selected_content_frame(tmp_path: Path) -> None:
    frames = tuple(
        _keyframe().model_copy(
            update={
                "evidence_id": f"keyframe_evidence_{index:03d}",
                "keyframe_id": f"keyframe_{index:03d}",
                "start_ms": 1_000 + index,
                "end_ms": 1_001 + index,
                "timestamp_ms": 1_000 + index,
                "sha256": f"{index + 1:064x}",
                "relative_path": f"visual/keyframes/{index + 1:064x}.jpg",
            },
        )
        for index in range(3)
    )
    observation = _observation().model_copy(
        update={
            "keyframe_refs": tuple(frame.evidence_id for frame in frames),
            "content_blocks": (
                VisualTextContent(
                    visual_content_id="visual_content_first",
                    source_keyframe_refs=(frames[0].evidence_id,),
                    text="第一项",
                ),
                VisualTextContent(
                    visual_content_id="visual_content_second",
                    source_keyframe_refs=(frames[1].evidence_id,),
                    text="第二项",
                ),
            ),
            "visual_facts": (
                GroundedVisualFact(
                    visual_fact_id="visual_fact_third",
                    text="第三项",
                    source_keyframe_refs=(frames[2].evidence_id,),
                ),
            ),
        },
    )

    def chapter(request: ChapterWritingRequest) -> ChapterWritingResponse:
        if request.chapter.chapter_id == "chapter_001":
            return _chapter_response(request)
        return ChapterWritingResponse(
            title="选择单项",
            summary_zh="只使用第二项",
            body_blocks=(
                VisualBlock(
                    visual_observation_ref=observation.evidence_id,
                    visual_content_refs=("visual_content_second",),
                    caption="第二项",
                    evidence_refs=(observation.evidence_id,),
                ),
            ),
            claims=(),
        )

    written = _writer(_TextPort(chapter=chapter)).write(
        _context(),
        (_plan(0), _plan(1)),
        (_speech(0), _speech(1)),
        (observation,),
        frames,
        cache=_cache(tmp_path),
        is_cancel_requested=lambda: False,
    )

    assert written.result.chapters[0].selected_keyframe_refs == (frames[1].evidence_id,)


def test_unknown_or_cross_observation_visual_content_ref_is_rejected(tmp_path: Path) -> None:
    observation = _observation()
    invalid = ChapterWritingResponse(
        title="非法",
        summary_zh="非法引用",
        body_blocks=(
            VisualBlock(
                visual_observation_ref=observation.evidence_id,
                visual_content_refs=("visual_content_other",),
                caption="非法",
                evidence_refs=(observation.evidence_id,),
            ),
        ),
        claims=(),
    )
    repaired = _chapter_response(
        ChapterWritingRequest(
            context=_context(),
            chapter=_plan(0),
            transcript_evidence=(_speech(0),),
            visual_observations=(observation,),
            prompt_version="chapter-writer-v1",
        ),
    )
    port = _TextPort(chapter=lambda _request: invalid, repair=lambda _request: repaired)

    written = _writer(port).write(
        _context(),
        (_plan(0), _plan(1)),
        (_speech(0), _speech(1)),
        (observation,),
        (_keyframe(),),
        cache=_cache(tmp_path),
        is_cancel_requested=lambda: False,
    )

    assert written.metrics["chapter_writer_structure_repairs"] == 2


def test_empty_visual_observation_content_falls_back_to_observation_frames(
    tmp_path: Path,
) -> None:
    observation = _observation().model_copy(update={"content_blocks": (), "visual_facts": ()})

    def chapter(request: ChapterWritingRequest) -> ChapterWritingResponse:
        if request.chapter.chapter_id == "chapter_001":
            return _chapter_response(request)
        return ChapterWritingResponse(
            title="空内容观察",
            summary_zh="使用观察帧",
            body_blocks=(
                VisualBlock(
                    visual_observation_ref=observation.evidence_id,
                    visual_content_refs=(),
                    caption="观察画面",
                    evidence_refs=(observation.evidence_id,),
                ),
            ),
            claims=(),
        )

    written = _writer(_TextPort(chapter=chapter)).write(
        _context(),
        (_plan(0), _plan(1)),
        (_speech(0), _speech(1)),
        (observation,),
        (_keyframe(),),
        cache=_cache(tmp_path),
        is_cancel_requested=lambda: False,
    )

    assert written.result.chapters[0].selected_keyframe_refs == ("keyframe_evidence_001",)


def test_empty_evidence_chapter_skips_all_models_and_stays_non_retrievable(
    tmp_path: Path,
) -> None:
    port = _TextPort()
    context = _context().model_copy(update={"transcript_source": "NONE"})

    written = _writer(port).write(
        context,
        (_plan(0), _plan(1)),
        (),
        (),
        (),
        cache=_cache(tmp_path),
        is_cancel_requested=lambda: False,
    )

    assert not port.chapter_requests and not port.global_requests
    assert all(
        chapter.content_status == "NO_SEMANTIC_EVIDENCE"
        for chapter in written.result.chapters
    )
    assert all(chapter.retrieval_text == "" for chapter in written.result.chapters)
    assert written.result.summary.retrieval_text == ""
    assert written.result.summary.key_points == ()


def test_quote_policy_repairs_disabled_quotes_and_downgrades_unmatched_quotes(
    tmp_path: Path,
) -> None:
    invalid = ChapterWritingResponse(
        title="章节",
        summary_zh="摘要",
        body_blocks=(QuoteBlock(text="模型伪造引文", evidence_refs=("asr_000",)),),
        claims=(),
    )
    repaired = ChapterWritingResponse(
        title="章节",
        summary_zh="摘要",
        body_blocks=(ParagraphBlock(text="普通正文", evidence_refs=("asr_000",)),),
        claims=(),
    )
    disabled_port = _TextPort(chapter=lambda _request: invalid, repair=lambda _request: repaired)
    disabled = _writer(disabled_port).write(
        _context(DocumentGenerationConfig(include_verbatim_quotes=False)),
        (_plan(0), _plan(1)),
        (_speech(0), _speech(1)),
        (),
        (),
        cache=_cache(tmp_path / "disabled"),
        is_cancel_requested=lambda: False,
    )
    assert disabled.metrics["chapter_writer_structure_repairs"] == 2
    assert all(
        isinstance(chapter.body_blocks[0], ParagraphBlock)
        for chapter in disabled.result.chapters
    )

    enabled_port = _TextPort(chapter=lambda _request: invalid)
    enabled = _writer(enabled_port).write(
        _context(DocumentGenerationConfig(include_verbatim_quotes=True)),
        (_plan(0), _plan(1)),
        (_speech(0), _speech(1)),
        (),
        (),
        cache=_cache(tmp_path / "enabled"),
        is_cancel_requested=lambda: False,
    )
    assert all(
        isinstance(chapter.body_blocks[0], ParagraphBlock)
        for chapter in enabled.result.chapters
    )


def test_invalid_main_and_repair_fall_back_but_authentication_fails_closed(
    tmp_path: Path,
) -> None:
    invalid = ModelResponseValidationError(
        ErrorCode.TEXT_LLM_RESPONSE_INVALID,
        "非法响应",
        InvalidModelResponse(content_sha256="d" * 64, validation_errors=("root:invalid",)),
    )
    port = _TextPort(chapter=lambda _request: invalid, repair=lambda _request: invalid)
    fallback = _writer(port).write(
        _context(),
        (_plan(0), _plan(1)),
        (_speech(0), _speech(1)),
        (),
        (),
        cache=_cache(tmp_path / "fallback"),
        is_cancel_requested=lambda: False,
    )
    assert fallback.status == "PARTIAL_SUCCEEDED"
    assert fallback.metrics["chapter_writer_fallback_chapters"] == 2
    assert fallback.warnings == (
        "CHAPTER_WRITING_DEGRADED:chapter_000",
        "CHAPTER_WRITING_DEGRADED:chapter_001",
    )

    auth = VideoDemoError(ErrorCode.TEXT_LLM_AUTHENTICATION_FAILED, "鉴权失败")
    with pytest.raises(VideoDemoError) as raised:
        _writer(_TextPort(chapter=lambda _request: auth)).write(
            _context(),
            (_plan(0), _plan(1)),
            (_speech(0), _speech(1)),
            (),
            (),
            cache=_cache(tmp_path / "auth"),
            is_cancel_requested=lambda: False,
        )
    assert raised.value.code == ErrorCode.TEXT_LLM_AUTHENTICATION_FAILED


def test_repair_programming_error_is_not_hidden_as_rule_fallback(tmp_path: Path) -> None:
    invalid = ModelResponseValidationError(
        ErrorCode.TEXT_LLM_RESPONSE_INVALID,
        "非法响应",
        InvalidModelResponse(content_sha256="d" * 64, validation_errors=("root:invalid",)),
    )
    port = _TextPort(
        chapter=lambda _request: invalid,
        repair=lambda _request: TypeError("修复适配器内部错误"),
    )

    with pytest.raises(TypeError, match="修复适配器内部错误"):
        _writer(port).write(
            _context(),
            (_plan(0), _plan(1)),
            (_speech(0), _speech(1)),
            (),
            (),
            cache=_cache(tmp_path),
            is_cancel_requested=lambda: False,
        )


def test_global_reorder_is_repaired_once_and_input_contains_only_digest_groups(
    tmp_path: Path,
) -> None:
    invalid = GlobalWritingResponse(
        overview_zh="错误顺序",
        key_points=(),
        sections=(
            SectionDraft(
                title="倒序",
                summary_zh="",
                chapter_refs=("chapter_001", "chapter_000"),
            ),
        ),
    )
    port = _TextPort(global_edit=lambda _request: invalid)

    written = _writer(port).write(
        _context(),
        (_plan(0), _plan(1)),
        (_speech(0), _speech(1)),
        (),
        (),
        cache=_cache(tmp_path),
        is_cancel_requested=lambda: False,
    )

    assert written.metrics["global_editor_structure_repairs"] == 1
    assert tuple(ref for section in written.result.sections for ref in section.chapter_refs) == (
        "chapter_000",
        "chapter_001",
    )
    sent = port.global_requests[0].model_dump_json()
    assert "body_blocks" not in sent and "evidence_id" not in sent


def test_chapter_input_budget_is_checked_before_any_provider_attempt(tmp_path: Path) -> None:
    port = _TextPort()
    with pytest.raises(VideoDemoError) as raised:
        _writer(port, max_input_chars=100, max_input_bytes=100).write(
            _context(),
            (_plan(0), _plan(1)),
            (_speech(0), _speech(1)),
            (),
            (),
            cache=_cache(tmp_path),
            is_cancel_requested=lambda: False,
        )
    assert raised.value.code == ErrorCode.INPUT_BUDGET_EXCEEDED
    assert not port.chapter_requests and not port.repair_requests


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"max_input_chars": 60_001}, "60000"),
        ({"max_input_bytes": 1_048_577}, "1048576"),
    ),
)
def test_writer_rejects_configured_input_budget_above_global_hard_limit(
    overrides: dict[str, int],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _writer(_TextPort(), **overrides)


def test_invalid_visual_evidence_closure_fails_before_provider_attempt(
    tmp_path: Path,
) -> None:
    port = _TextPort()
    cross_chapter_transcript = _observation().model_copy(
        update={"transcript_evidence_refs": ("asr_001",)},
    )

    with pytest.raises(VideoDemoError) as raised:
        _writer(port).write(
            _context(),
            (_plan(0), _plan(1)),
            (_speech(0), _speech(1)),
            (cross_chapter_transcript,),
            (_keyframe(),),
            cache=_cache(tmp_path),
            is_cancel_requested=lambda: False,
        )

    assert raised.value.code == ErrorCode.EVIDENCE_OUTSIDE_CHAPTER
    assert not port.chapter_requests and not port.global_requests


def test_orphan_keyframe_fails_before_provider_attempt(tmp_path: Path) -> None:
    port = _TextPort()

    with pytest.raises(VideoDemoError) as raised:
        _writer(port).write(
            _context(),
            (_plan(0), _plan(1)),
            (_speech(0), _speech(1)),
            (),
            (_keyframe(),),
            cache=_cache(tmp_path),
            is_cancel_requested=lambda: False,
        )

    assert raised.value.code == ErrorCode.UNKNOWN_EVIDENCE_REFERENCE
    assert not port.chapter_requests and not port.global_requests


def test_two_concurrent_writes_with_same_fingerprint_call_provider_once(tmp_path: Path) -> None:
    barrier = Barrier(2)
    lock = Lock()
    calls = 0

    def chapter(request: ChapterWritingRequest) -> ChapterWritingResponse:
        nonlocal calls
        with lock:
            calls += 1
        return _chapter_response(request)

    port = _TextPort(chapter=chapter)
    writer = _writer(port, chapter_writer_concurrency=1)
    plans = (_plan(0), _plan(1))
    transcript = (_speech(0), _speech(1))

    def run() -> object:
        barrier.wait()
        return writer.write(
            _context(),
            plans,
            transcript,
            (),
            (),
            cache=_cache(tmp_path),
            is_cancel_requested=lambda: False,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(executor.map(lambda _index: run(), range(2)))

    assert calls == 2
    assert results[0].result == results[1].result  # type: ignore[union-attr]
