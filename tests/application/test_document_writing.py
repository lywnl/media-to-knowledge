from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue
from threading import Barrier, Event, Lock, Thread
from typing import cast

import pytest

from video_demo.application.document_writing import (
    DocumentWriter,
    _global_request,
    _global_request_fits,
)
from video_demo.application.pipeline_contracts import DocumentWritingContext
from video_demo.domain.document import (
    DocumentGenerationConfig,
    GroundedClaim,
    ParagraphBlock,
    PromptVersions,
    QuoteBlock,
    SemanticChapter,
    VisualBlock,
)
from video_demo.domain.document_plan import ChapterPlan
from video_demo.domain.evidence import (
    GroundedVisualFact,
    KeyframeEvidence,
    SpeechSegment,
    VisualFrameRelation,
    VisualObservationEvidence,
    VisualTextContent,
)
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.integrations.document_port import (
    ChapterWritingRequest,
    ChapterWritingResponse,
    DocumentTextPort,
    GlobalChapterInput,
    GlobalWritingRepairRequest,
    GlobalWritingRequest,
    GlobalWritingResponse,
    InvalidModelResponse,
    ModelResponseValidationError,
)
from video_demo.integrations.document_prompts import prompt_for_global_repair
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
        title_evidence_refs=(evidence_ref,),
        summary_zh="章节摘要",
        summary_evidence_refs=(evidence_ref,),
        body_blocks=blocks,
        claims=(),
    )


def _global_response(request: GlobalWritingRequest) -> GlobalWritingResponse:
    return GlobalWritingResponse(overview_zh="全局概览")


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
    writing = operation == "chapter_writing"
    global_editing = operation == "global_editing"
    return ModelInvocationIdentity(
        logical_operation=operation,
        provider_config_fingerprint="c" * 64,
        model_id="text-model",
        generation_config=(("temperature", "0"),),
        main_response_schema_name=(
            "chapter_writing_v2"
            if writing
            else "global_writing_v1" if global_editing else f"{operation}_v1"
        ),
        main_prompt_version=(
            "chapter-writer-v1" if operation == "chapter_writing" else "global-editor-v1"
        ),
        repair_response_schema_name=(
            "chapter_writing_repair_v2"
            if writing
            else "global_writing_repair_v1"
            if global_editing
            else f"{operation}_repair_v1"
        ),
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


class _BlockingPutCache(DocumentModelCache):
    def __init__(self, path: Path, *, blocked_operation: str) -> None:
        super().__init__(path, max_entry_bytes=1_048_576, max_run_bytes=4_194_304)
        self._blocked_operation = blocked_operation
        self.entered = Event()
        self.release = Event()
        self.published = Event()

    def put(  # type: ignore[override]
        self,
        identity: ModelInvocationIdentity,
        *args: object,
        **kwargs: object,
    ) -> object:
        if identity.logical_operation == self._blocked_operation:
            self.entered.set()
            self.release.wait(timeout=5)
        result = super().put(identity, *args, **kwargs)  # type: ignore[arg-type]
        if identity.logical_operation == self._blocked_operation:
            self.published.set()
        return result


class _SiblingCommitFailureCache(_BlockingPutCache):
    def __init__(self, path: Path, *, cancelled: Event) -> None:
        super().__init__(path, blocked_operation="chapter_writing")
        self._cancelled = cancelled
        self._chapter_gets: dict[str, int] = {}

    def get(  # type: ignore[override]
        self,
        identity: ModelInvocationIdentity,
        canonical_input: object,
        *args: object,
        **kwargs: object,
    ) -> object:
        chapter = canonical_input.chapter  # type: ignore[attr-defined]
        count = self._chapter_gets.get(chapter.chapter_id, 0) + 1
        self._chapter_gets[chapter.chapter_id] = count
        if chapter.chapter_id == "chapter_000" and count == 2:
            assert self.entered.wait(timeout=1)
            self._cancelled.set()
            raise VideoDemoError(ErrorCode.ARTIFACT_SCHEMA_INVALID, "模拟缓存损坏")
        return super().get(identity, canonical_input, *args, **kwargs)  # type: ignore[arg-type]

    def put(  # type: ignore[override]
        self,
        identity: ModelInvocationIdentity,
        canonical_input: object,
        *args: object,
        **kwargs: object,
    ) -> object:
        chapter = canonical_input.chapter  # type: ignore[attr-defined]
        if chapter.chapter_id == "chapter_001":
            return super().put(identity, canonical_input, *args, **kwargs)
        return DocumentModelCache.put(  # type: ignore[arg-type]
            self,
            identity,
            canonical_input,
            *args,
            **kwargs,
        )


class _RetryAfterCancellationPort(_TextPort):
    def __init__(self, blocked_call: str) -> None:
        super().__init__()
        self._blocked_call = blocked_call
        self.entered = Event()
        self.release = Event()
        self.provider_finished = Event()
        self.second_attempt_started = Event()

    def _attempt_twice(self, on_provider_attempt: Callable[[], None] | None) -> None:
        if on_provider_attempt:
            on_provider_attempt()
        self.entered.set()
        self.release.wait(timeout=5)
        try:
            if on_provider_attempt:
                on_provider_attempt()
            self.second_attempt_started.set()
        finally:
            self.provider_finished.set()

    def write_chapter(
        self,
        request: ChapterWritingRequest,
        *,
        on_provider_attempt: Callable[[], None] | None = None,
    ) -> ChapterWritingResponse:
        self.chapter_requests.append(request)
        if self._blocked_call == "chapter_main":
            self._attempt_twice(on_provider_attempt)
        elif self._blocked_call == "chapter_repair":
            if on_provider_attempt:
                on_provider_attempt()
            raise ModelResponseValidationError(
                ErrorCode.TEXT_LLM_RESPONSE_INVALID,
                "非法章节响应",
                InvalidModelResponse(
                    content_sha256="d" * 64,
                    validation_errors=("root:invalid",),
                ),
            )
        elif on_provider_attempt:
            on_provider_attempt()
        return _chapter_response(request)

    def repair_chapter_writing(
        self,
        request: object,
        *,
        on_provider_attempt: Callable[[], None] | None = None,
    ) -> ChapterWritingResponse:
        original = request.request  # type: ignore[attr-defined]
        self.repair_requests.append(original)
        if self._blocked_call == "chapter_repair":
            self._attempt_twice(on_provider_attempt)
        elif on_provider_attempt:
            on_provider_attempt()
        return _chapter_response(original)

    def organize_document(
        self,
        request: GlobalWritingRequest,
        *,
        on_provider_attempt: Callable[[], None] | None = None,
    ) -> GlobalWritingResponse:
        self.global_requests.append(request)
        if self._blocked_call == "global_main":
            self._attempt_twice(on_provider_attempt)
        elif self._blocked_call == "global_repair":
            if on_provider_attempt:
                on_provider_attempt()
            raise ModelResponseValidationError(
                ErrorCode.TEXT_LLM_RESPONSE_INVALID,
                "非法全局响应",
                InvalidModelResponse(
                    content_sha256="e" * 64,
                    validation_errors=("root:invalid",),
                ),
            )
        elif on_provider_attempt:
            on_provider_attempt()
        return _global_response(request)

    def repair_global_writing(
        self,
        request: object,
        *,
        on_provider_attempt: Callable[[], None] | None = None,
    ) -> GlobalWritingResponse:
        original = request.request  # type: ignore[attr-defined]
        self.global_repair_requests.append(original)
        if self._blocked_call == "global_repair":
            self._attempt_twice(on_provider_attempt)
        elif on_provider_attempt:
            on_provider_attempt()
        return _global_response(original)


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
    assert "retrieval_text" not in first.result.model_dump(mode="json")
    assert first.metrics["chapter_writer_provider_attempts"] == 2
    assert second.metrics["chapter_writer_cache_hits"] == 2
    assert second.metrics["global_editor_cache_hits"] == 1


def test_writer_adds_visual_block_when_model_omits_valid_visual_observation(
    tmp_path: Path,
) -> None:
    observation = _observation()

    def chapter_without_visual(request: ChapterWritingRequest) -> ChapterWritingResponse:
        evidence_ref = request.transcript_evidence[0].evidence_id
        return ChapterWritingResponse(
            title=request.chapter.title_hint,
            title_evidence_refs=(evidence_ref,),
            summary_zh="章节摘要",
            summary_evidence_refs=(evidence_ref,),
            body_blocks=(
                ParagraphBlock(text="根据转写整理正文。", evidence_refs=(evidence_ref,)),
            ),
            claims=(),
        )

    written = _writer(_TextPort(chapter=chapter_without_visual)).write(
        _context(),
        (_plan(0), _plan(1)),
        (_speech(0), _speech(1)),
        (observation,),
        (_keyframe(),),
        cache=_cache(tmp_path),
        is_cancel_requested=lambda: False,
    )

    chapter = written.result.chapters[0]
    assert chapter.selected_keyframe_refs == ("keyframe_evidence_001",)
    assert any(isinstance(block, VisualBlock) for block in chapter.body_blocks)


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
            title_evidence_refs=(observation.evidence_id,),
            summary_zh="只使用第二项",
            summary_evidence_refs=(observation.evidence_id,),
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
        title_evidence_refs=(observation.evidence_id,),
        summary_zh="非法引用",
        summary_evidence_refs=(observation.evidence_id,),
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
            title_evidence_refs=(observation.evidence_id,),
            summary_zh="使用观察帧",
            summary_evidence_refs=(observation.evidence_id,),
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
    assert "retrieval_text" not in written.result.model_dump(mode="json")
    assert "key_points" not in written.result.model_dump(mode="json")


def test_quote_policy_repairs_disabled_quotes_and_downgrades_unmatched_quotes(
    tmp_path: Path,
) -> None:
    invalid = ChapterWritingResponse(
        title="章节",
        title_evidence_refs=("asr_000",),
        summary_zh="摘要",
        summary_evidence_refs=("asr_000",),
        body_blocks=(QuoteBlock(text="模型伪造引文", evidence_refs=("asr_000",)),),
        claims=(),
    )
    repaired = ChapterWritingResponse(
        title="章节",
        title_evidence_refs=("asr_000",),
        summary_zh="摘要",
        summary_evidence_refs=("asr_000",),
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
    caplog: pytest.LogCaptureFixture,
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
    messages = "\n".join(caplog.messages)
    assert "章节写作响应校验失败" in messages
    assert "chapter_id=chapter_000" in messages
    assert "phase=main" in messages
    assert "phase=repair" in messages
    assert "root:invalid" in messages

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


def test_visual_only_fallback_without_transcript_uses_visual_evidence(
    tmp_path: Path,
) -> None:
    payload = _observation().model_dump(exclude={"duration_ms"})
    payload.update(
        {
            "relation_to_transcript": "INDEPENDENT",
            "certainty": 0.6,
            "transcript_evidence_refs": (),
        },
    )
    observation = VisualObservationEvidence.model_validate(payload)
    temporary = VideoDemoError(ErrorCode.DEPENDENCY_TEMPORARY_FAILURE, "临时失败")

    written = _writer(
        _TextPort(chapter=lambda _request: temporary),
        chapter_writer_concurrency=1,
    ).write(
        _context(
            DocumentGenerationConfig(),
        ).model_copy(update={"transcript_source": "NONE"}),
        (_plan(0), _plan(1)),
        (),
        (observation,),
        (_keyframe(),),
        cache=_cache(tmp_path),
        is_cancel_requested=lambda: False,
    )

    first = written.result.chapters[0]
    assert written.status == "PARTIAL_SUCCEEDED"
    assert first.content_status == "GROUNDED"
    assert first.title_evidence_refs == (observation.evidence_id,)
    assert first.summary_evidence_refs == (observation.evidence_id,)


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


def test_global_editor_uses_chapter_facts_and_repairs_invalid_response(
    tmp_path: Path,
) -> None:
    invalid = GlobalWritingResponse(overview_zh="")
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
    assert tuple(chapter.chapter_id for chapter in written.result.chapters) == (
        "chapter_000",
        "chapter_001",
    )
    sent = port.global_requests[0].model_dump_json()
    assert "body_blocks" not in sent and "evidence_id" not in sent and "chapters" in sent


def test_writer_falls_back_to_chapter_summary_when_global_overview_is_empty(
    tmp_path: Path,
) -> None:
    def empty_overview(_request: GlobalWritingRequest) -> GlobalWritingResponse:
        return GlobalWritingResponse(overview_zh="")

    written = _writer(
        _TextPort(global_edit=empty_overview, global_repair=empty_overview),
    ).write(
        _context(),
        (_plan(0), _plan(1)),
        (_speech(0), _speech(1)),
        (),
        (),
        cache=_cache(tmp_path),
        is_cancel_requested=lambda: False,
    )

    assert written.result.summary.overview_zh == "章节摘要；章节摘要"


def test_writer_adds_grounded_claim_when_model_omits_claims(tmp_path: Path) -> None:
    written = _writer(_TextPort()).write(
        _context(),
        (_plan(0), _plan(1)),
        (_speech(0), _speech(1)),
        (),
        (),
        cache=_cache(tmp_path),
        is_cancel_requested=lambda: False,
    )

    claim = written.result.chapters[0].claims[0]
    assert claim.text == "章节摘要"
    assert claim.evidence_refs == ("asr_000",)


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


@pytest.mark.parametrize(
    ("identity_field", "invalid_schema"),
    (
        ("main_response_schema_name", "global_editing_v1"),
        ("repair_response_schema_name", "global_editing_repair_v1"),
    ),
)
def test_writer_rejects_global_cache_identity_with_wrong_response_schema(
    identity_field: str,
    invalid_schema: str,
) -> None:
    identity = _identity("global_editing").model_copy(
        update={identity_field: invalid_schema},
    )

    with pytest.raises(ValueError, match=r"全局编辑缓存身份与.*Schema 不一致"):
        _writer(_TextPort(), global_identity=identity)


@pytest.mark.parametrize(
    ("identity_name", "invalid_operation"),
    (
        ("chapter_identity", "global_editing"),
        ("global_identity", "chapter_writing"),
    ),
)
def test_writer_rejects_cache_identity_with_wrong_logical_operation(
    identity_name: str,
    invalid_operation: str,
) -> None:
    source_operation = (
        "chapter_writing" if identity_name == "chapter_identity" else "global_editing"
    )
    identity = _identity(source_operation).model_copy(
        update={"logical_operation": invalid_operation},
    )

    with pytest.raises(ValueError, match="缓存身份与逻辑操作不一致"):
        _writer(_TextPort(), **{identity_name: identity})


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


@pytest.mark.parametrize(
    ("invalid", "overrides"),
    (
        (
            InvalidModelResponse(
                content_sha256="e" * 64,
                validation_errors=tuple(
                    f"{index:02d}:" + '"' * 497 for index in range(32)
                ),
                safe_json_excerpt='"' * 8_000,
            ),
            {"max_input_chars": 30_000},
        ),
        (
            InvalidModelResponse(
                content_sha256="e" * 64,
                validation_errors=tuple(
                    f"{index:02d}:" + "\U00010000" * 497 for index in range(32)
                ),
                safe_json_excerpt="\U00010000" * 8_000,
            ),
            {"max_input_bytes": 40_000},
        ),
    ),
)
def test_worst_repair_budget_is_rejected_before_paid_chapter_call(
    tmp_path: Path,
    invalid: InvalidModelResponse,
    overrides: dict[str, int],
) -> None:
    model_error = ModelResponseValidationError(
        ErrorCode.TEXT_LLM_RESPONSE_INVALID,
        "合法上界修复上下文",
        invalid,
    )
    port = _TextPort(chapter=lambda _request: model_error)

    with pytest.raises(VideoDemoError) as raised:
        _writer(port, **overrides).write(
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


def test_chapter_budget_preserves_visual_observation_before_transcript(
    tmp_path: Path,
) -> None:
    observation = _observation()

    def visual_first_response(request: ChapterWritingRequest) -> ChapterWritingResponse:
        evidence_ref = (
            request.transcript_evidence[0].evidence_id
            if request.transcript_evidence
            else request.visual_observations[0].evidence_id
        )
        return ChapterWritingResponse(
            title=request.chapter.title_hint,
            title_evidence_refs=(evidence_ref,),
            summary_zh="视觉证据优先",
            summary_evidence_refs=(evidence_ref,),
            body_blocks=(),
            claims=(),
        )

    port = _TextPort(chapter=visual_first_response)

    written = _writer(port, max_input_chars=49_525).write(
        _context(),
        (_plan(0), _plan(1)),
        (_speech(0), _speech(1)),
        (observation,),
        (_keyframe(),),
        cache=_cache(tmp_path),
        is_cancel_requested=lambda: False,
    )

    first_request = next(
        request for request in port.chapter_requests if request.chapter.chapter_id == "chapter_000"
    )
    assert written.status == "SUCCEEDED"
    assert first_request.visual_observations == (observation,)
    assert first_request.transcript_evidence == ()


def test_chapter_budget_does_not_turn_the_only_visual_evidence_into_an_empty_request(
    tmp_path: Path,
) -> None:
    payload = _observation().model_dump(exclude={"duration_ms"})
    payload.update(
        {
            "relation_to_transcript": "INDEPENDENT",
            "transcript_evidence_refs": (),
        },
    )
    observation = VisualObservationEvidence.model_validate(payload)
    port = _TextPort()

    with pytest.raises(VideoDemoError) as raised:
        _writer(port, max_input_chars=49_300).write(
            _context().model_copy(update={"transcript_source": "NONE"}),
            (_plan(0), _plan(1)),
            (),
            (observation,),
            (_keyframe(),),
            cache=_cache(tmp_path),
            is_cancel_requested=lambda: False,
        )

    assert raised.value.code == ErrorCode.INPUT_BUDGET_EXCEEDED
    assert port.chapter_requests == []


def test_global_chapter_fact_trimming_reaches_a_non_empty_solution_below_one_percent() -> None:
    chapters = tuple(
        _materialized_chapter_for_global_test(index, count=105)
        for index in range(105)
    )
    context = _context().model_copy(update={"duration_ms": 1_050_000})
    chapter_inputs = tuple(
        GlobalChapterInput(
            chapter_id=chapter.chapter_id,
            start_ms=chapter.start_ms,
            end_ms=chapter.end_ms,
            title=chapter.title,
            summary_zh=chapter.summary_zh[:20],
            content_status=chapter.content_status,
        )
        for chapter in chapters
    )
    candidate = GlobalWritingRequest(
        context=context,
        chapters=chapter_inputs,
        prompt_version="global-editor-v1",
    )
    repair = GlobalWritingRepairRequest(
        request=candidate,
        invalid_response=_maximal_escaped_invalid_response(),
        allowed_chapter_ids=tuple(
            chapter.chapter_id for chapter in candidate.chapters
        ),
        prompt_version="global-editor-repair-v1",
    )
    maximum = len(prompt_for_global_repair(repair)[2])
    assert _global_request_fits(candidate, maximum, 1_048_576)

    request = _global_request(
        context,
        chapters,
        "global-editor-v1",
        maximum,
        1_048_576,
    )

    assert all(1 <= len(chapter.summary_zh) <= 20 for chapter in request.chapters)


def test_cancellation_does_not_wait_for_blocked_chapter_provider(tmp_path: Path) -> None:
    entered = Event()
    release = Event()
    cancelled = Event()
    outcome: Queue[BaseException | object] = Queue()

    def blocked(request: ChapterWritingRequest) -> ChapterWritingResponse:
        entered.set()
        release.wait(timeout=5)
        return _chapter_response(request)

    def run() -> None:
        try:
            outcome.put(
                _writer(_TextPort(chapter=blocked), chapter_writer_concurrency=1).write(
                    _context(),
                    (_plan(0), _plan(1)),
                    (_speech(0), _speech(1)),
                    (),
                    (),
                    cache=_cache(tmp_path),
                    is_cancel_requested=cancelled.is_set,
                ),
            )
        except BaseException as error:
            outcome.put(error)

    worker = Thread(target=run, daemon=True)
    worker.start()
    try:
        assert entered.wait(timeout=1)
        cancelled.set()
        worker.join(timeout=0.5)
        assert not worker.is_alive()
        error = outcome.get_nowait()
        assert isinstance(error, VideoDemoError)
        assert error.code == ErrorCode.JOB_CANCELLED
    finally:
        release.set()
        worker.join(timeout=2)


def test_cancelled_chapter_provider_does_not_start_repair(tmp_path: Path) -> None:
    entered = Event()
    release = Event()
    cancelled = Event()
    repair_started = Event()
    outcome: Queue[BaseException | object] = Queue()
    invalid = ModelResponseValidationError(
        ErrorCode.TEXT_LLM_RESPONSE_INVALID,
        "非法响应",
        InvalidModelResponse(
            content_sha256="d" * 64,
            validation_errors=("root:invalid",),
        ),
    )

    def blocked(_request: ChapterWritingRequest) -> ChapterWritingResponse:
        entered.set()
        release.wait(timeout=5)
        raise invalid

    def repair(request: ChapterWritingRequest) -> ChapterWritingResponse:
        repair_started.set()
        return _chapter_response(request)

    port = _TextPort(chapter=blocked, repair=repair)

    def run() -> None:
        try:
            outcome.put(
                _writer(port, chapter_writer_concurrency=1).write(
                    _context(),
                    (_plan(0), _plan(1)),
                    (_speech(0), _speech(1)),
                    (),
                    (),
                    cache=_cache(tmp_path),
                    is_cancel_requested=cancelled.is_set,
                ),
            )
        except BaseException as error:
            outcome.put(error)

    worker = Thread(target=run, daemon=True)
    worker.start()
    assert entered.wait(timeout=1)
    cancelled.set()
    worker.join(timeout=0.5)
    assert not worker.is_alive()
    release.set()
    worker.join(timeout=2)

    error = outcome.get_nowait()
    assert isinstance(error, VideoDemoError)
    assert error.code == ErrorCode.JOB_CANCELLED
    assert not repair_started.wait(timeout=0.5)
    assert port.repair_requests == []


def test_pre_cancelled_empty_document_does_not_return_success(tmp_path: Path) -> None:
    with pytest.raises(VideoDemoError) as raised:
        _writer(_TextPort()).write(
            _context().model_copy(update={"transcript_source": "NONE"}),
            (_plan(0), _plan(1)),
            (),
            (),
            (),
            cache=_cache(tmp_path),
            is_cancel_requested=lambda: True,
        )

    assert raised.value.code == ErrorCode.JOB_CANCELLED


def test_cancellation_does_not_wait_for_blocked_global_provider_or_publish_cache(
    tmp_path: Path,
) -> None:
    entered = Event()
    release = Event()
    cancelled = Event()
    outcome: Queue[BaseException | object] = Queue()
    port = _TextPort()

    def blocked(request: GlobalWritingRequest) -> GlobalWritingResponse:
        entered.set()
        release.wait(timeout=5)
        return _global_response(request)

    port.global_edit = blocked
    cache = _cache(tmp_path)

    def run() -> None:
        try:
            outcome.put(
                _writer(port).write(
                    _context(),
                    (_plan(0), _plan(1)),
                    (_speech(0), _speech(1)),
                    (),
                    (),
                    cache=cache,
                    is_cancel_requested=cancelled.is_set,
                ),
            )
        except BaseException as error:
            outcome.put(error)

    worker = Thread(target=run, daemon=True)
    worker.start()
    try:
        assert entered.wait(timeout=1)
        cancelled.set()
        worker.join(timeout=0.5)
        assert not worker.is_alive()
        error = outcome.get_nowait()
        assert isinstance(error, VideoDemoError)
        assert error.code == ErrorCode.JOB_CANCELLED
    finally:
        release.set()
        worker.join(timeout=2)

    second = _writer(port).write(
        _context(),
        (_plan(0), _plan(1)),
        (_speech(0), _speech(1)),
        (),
        (),
        cache=cache,
        is_cancel_requested=lambda: False,
    )
    assert second.metrics["global_editor_cache_hits"] == 0
    assert len(port.global_requests) == 2


@pytest.mark.parametrize(
    ("blocked_operation", "writer_concurrency"),
    (
        ("chapter_writing", 1),
        ("global_editing", 2),
    ),
)
def test_cancellation_waits_for_in_progress_cache_commit(
    tmp_path: Path,
    blocked_operation: str,
    writer_concurrency: int,
) -> None:
    cancelled = Event()
    outcome: Queue[BaseException | object] = Queue()
    cache = _BlockingPutCache(tmp_path, blocked_operation=blocked_operation)

    def run() -> None:
        try:
            outcome.put(
                _writer(
                    _TextPort(),
                    chapter_writer_concurrency=writer_concurrency,
                ).write(
                    _context(),
                    (_plan(0), _plan(1)),
                    (_speech(0), _speech(1)),
                    (),
                    (),
                    cache=cache,
                    is_cancel_requested=cancelled.is_set,
                ),
            )
        except BaseException as error:
            outcome.put(error)

    worker = Thread(target=run, daemon=True)
    worker.start()
    try:
        assert cache.entered.wait(timeout=1)
        cancelled.set()
        worker.join(timeout=0.5)
        assert worker.is_alive(), "缓存提交已经开始时，调用方不得提前返回取消"
        assert not cache.published.is_set()
    finally:
        cache.release.set()
        worker.join(timeout=2)

    assert cache.published.is_set()
    error = outcome.get_nowait()
    assert isinstance(error, VideoDemoError)
    assert error.code == ErrorCode.JOB_CANCELLED


def test_cancelled_chapter_future_waits_for_sibling_cache_commit(tmp_path: Path) -> None:
    cancelled = Event()
    provider_entered = Event()
    provider_release = Event()
    outcome: Queue[BaseException | object] = Queue()
    cache = _BlockingPutCache(tmp_path, blocked_operation="chapter_writing")

    def chapter(request: ChapterWritingRequest) -> ChapterWritingResponse:
        if request.chapter.chapter_id == "chapter_000":
            provider_entered.set()
            provider_release.wait(timeout=5)
        return _chapter_response(request)

    def run() -> None:
        try:
            outcome.put(
                _writer(_TextPort(chapter=chapter), chapter_writer_concurrency=2).write(
                    _context(),
                    (_plan(0), _plan(1)),
                    (_speech(0), _speech(1)),
                    (),
                    (),
                    cache=cache,
                    is_cancel_requested=cancelled.is_set,
                ),
            )
        except BaseException as error:
            outcome.put(error)

    worker = Thread(target=run, daemon=True)
    worker.start()
    try:
        assert provider_entered.wait(timeout=1)
        assert cache.entered.wait(timeout=1)
        cancelled.set()
        provider_release.set()
        worker.join(timeout=0.5)
        assert worker.is_alive(), "同批其他章节正在提交缓存时不得提前返回取消"
        assert not cache.published.is_set()
    finally:
        provider_release.set()
        cache.release.set()
        worker.join(timeout=2)

    assert cache.published.is_set()
    error = outcome.get_nowait()
    assert isinstance(error, VideoDemoError)
    assert error.code == ErrorCode.JOB_CANCELLED


def test_cancelled_non_cancel_failure_waits_for_sibling_cache_commit(tmp_path: Path) -> None:
    cancelled = Event()
    outcome: Queue[BaseException | object] = Queue()
    cache = _SiblingCommitFailureCache(tmp_path, cancelled=cancelled)

    def run() -> None:
        try:
            outcome.put(
                _writer(
                    _TextPort(),
                    chapter_writer_concurrency=2,
                    invocation_wait_timeout_seconds=0.1,
                ).write(
                    _context(),
                    (_plan(0), _plan(1)),
                    (_speech(0), _speech(1)),
                    (),
                    (),
                    cache=cache,
                    is_cancel_requested=cancelled.is_set,
                ),
            )
        except BaseException as error:
            outcome.put(error)

    worker = Thread(target=run, daemon=True)
    worker.start()
    try:
        assert cache.entered.wait(timeout=1)
        worker.join(timeout=0.5)
        assert worker.is_alive(), "兄弟章节正在提交缓存时，非取消异常也不得提前返回"
        assert not cache.published.is_set()
    finally:
        cache.release.set()
        worker.join(timeout=2)

    assert cache.published.is_set()
    error = outcome.get_nowait()
    assert isinstance(error, VideoDemoError)
    assert error.code == ErrorCode.ARTIFACT_SCHEMA_INVALID


@pytest.mark.parametrize(
    "blocked_call",
    ("chapter_main", "chapter_repair", "global_main", "global_repair"),
)
def test_cancelled_writer_prevents_next_provider_attempt(
    tmp_path: Path,
    blocked_call: str,
) -> None:
    cancelled = Event()
    outcome: Queue[BaseException | object] = Queue()
    port = _RetryAfterCancellationPort(blocked_call)

    def run() -> None:
        try:
            outcome.put(
                _writer(port, chapter_writer_concurrency=1).write(
                    _context(),
                    (_plan(0), _plan(1)),
                    (_speech(0), _speech(1)),
                    (),
                    (),
                    cache=_cache(tmp_path),
                    is_cancel_requested=cancelled.is_set,
                ),
            )
        except BaseException as error:
            outcome.put(error)

    worker = Thread(target=run, daemon=True)
    worker.start()
    try:
        assert port.entered.wait(timeout=1)
        cancelled.set()
        worker.join(timeout=0.5)
        assert not worker.is_alive()
    finally:
        port.release.set()
        assert port.provider_finished.wait(timeout=2)
        worker.join(timeout=2)

    error = outcome.get_nowait()
    assert isinstance(error, VideoDemoError)
    assert error.code == ErrorCode.JOB_CANCELLED
    assert not port.second_attempt_started.is_set()


@pytest.mark.parametrize(
    ("relation", "certainty"),
    (("DUPLICATE", 0.9), ("CONFLICTING", 0.9)),
)
def test_visual_policy_cannot_be_bypassed_by_paragraphs_or_claims(
    tmp_path: Path,
    relation: str,
    certainty: float,
) -> None:
    observation = _observation().model_copy(
        update={
            "relation_to_transcript": relation,
            "certainty": certainty,
        },
    )
    unsafe = ChapterWritingResponse(
        title="不安全视觉正文",
        title_evidence_refs=("asr_000",),
        summary_zh="把视觉观察写成确定事实",
        summary_evidence_refs=("asr_000",),
        body_blocks=(
            ParagraphBlock(
                text="确定事实：参数就是 42。",
                evidence_refs=(observation.evidence_id,),
            ),
        ),
        claims=(
            GroundedClaim(
                text="参数就是 42。",
                evidence_refs=(observation.evidence_id,),
                certainty=1.0,
            ),
        ),
    )
    safe = ChapterWritingResponse(
        title="修复后正文",
        title_evidence_refs=("asr_000",),
        summary_zh="仅保留转写事实",
        summary_evidence_refs=("asr_000",),
        body_blocks=(ParagraphBlock(text="转写事实。", evidence_refs=("asr_000",)),),
        claims=(),
    )
    port = _TextPort(
        chapter=lambda request: (
            unsafe if request.visual_observations else _chapter_response(request)
        ),
        repair=lambda _request: safe,
    )
    config = DocumentGenerationConfig()

    written = _writer(port).write(
        _context(config),
        (_plan(0), _plan(1)),
        (_speech(0), _speech(1)),
        (observation,),
        (_keyframe(),),
        cache=_cache(tmp_path),
        is_cancel_requested=lambda: False,
    )

    assert written.metrics["chapter_writer_structure_repairs"] == 1
    assert written.result.chapters[0].body_blocks == safe.body_blocks


def test_visual_policy_cannot_be_bypassed_by_title_or_summary(tmp_path: Path) -> None:
    observation = _observation().model_copy(update={"relation_to_transcript": "DUPLICATE"})
    unsafe = ChapterWritingResponse.model_validate(
        {
            "title": "秘密参数 42",
            "title_evidence_refs": [observation.evidence_id],
            "summary_zh": "画面确认秘密参数等于 42。",
            "summary_evidence_refs": [observation.evidence_id],
            "body_blocks": [
                ParagraphBlock(
                    text="根据转写整理正文。",
                    evidence_refs=("asr_000",),
                ).model_dump(mode="json"),
            ],
            "claims": [],
        },
    )
    safe = ChapterWritingResponse.model_validate(
        {
            "title": "转写演示",
            "title_evidence_refs": ["asr_000"],
            "summary_zh": "根据转写介绍演示内容。",
            "summary_evidence_refs": ["asr_000"],
            "body_blocks": [
                ParagraphBlock(
                    text="根据转写整理正文。",
                    evidence_refs=("asr_000",),
                ).model_dump(mode="json"),
            ],
            "claims": [],
        },
    )
    port = _TextPort(
        chapter=lambda request: (
            unsafe if request.visual_observations else _chapter_response(request)
        ),
        repair=lambda _request: safe,
    )

    written = _writer(port).write(
        _context(),
        (_plan(0), _plan(1)),
        (_speech(0), _speech(1)),
        (observation,),
        (_keyframe(),),
        cache=_cache(tmp_path),
        is_cancel_requested=lambda: False,
    )

    first = written.result.chapters[0]
    assert written.metrics["chapter_writer_structure_repairs"] == 1
    assert first.title == safe.title
    assert first.summary_zh == safe.summary_zh
    assert "retrieval_text" not in first.model_dump(mode="json")
    assert "秘密参数" not in port.global_requests[0].chapters[0].summary_zh


def test_conflict_caption_normalization_cannot_exceed_final_body_budget(
    tmp_path: Path,
) -> None:
    observation = _observation().model_copy(
        update={
            "relation_to_transcript": "CONFLICTING",
            "caption": "画" * 700,
        },
    )
    unsafe = ChapterWritingResponse(
        title="冲突画面",
        title_evidence_refs=("asr_000",),
        summary_zh="冲突观察",
        summary_evidence_refs=("asr_000",),
        body_blocks=(
            VisualBlock(
                visual_observation_ref=observation.evidence_id,
                visual_content_refs=("visual_content_001",),
                caption="画" * 700,
                evidence_refs=(observation.evidence_id,),
            ),
        ),
        claims=(),
    )
    port = _TextPort(
        chapter=lambda request: (
            unsafe if request.visual_observations else _chapter_response(request)
        ),
    )

    written = _writer(port).write(
        _context(DocumentGenerationConfig(detail_level="concise")),
        (_plan(0), _plan(1)),
        (_speech(0), _speech(1)),
        (observation,),
        (_keyframe(),),
        cache=_cache(tmp_path),
        is_cancel_requested=lambda: False,
    )

    first = written.result.chapters[0]
    assert (
        sum(
            len(block.caption)
            for block in first.body_blocks
            if isinstance(block, VisualBlock)
        )
        <= 800
    )


def test_rule_fallback_handles_large_transcript_and_unrelated_multiframe_observation(
    tmp_path: Path,
) -> None:
    transcripts = (
        *(
            SpeechSegment(
                evidence_id=f"asr_many_{index:03d}",
                start_ms=index * 100,
                end_ms=index * 100 + 50,
                text=f"第 {index} 条。",
                language="zh",
                confidence=0.9,
                is_fully_evaluated_language=True,
            )
            for index in range(33)
        ),
        _speech(1),
    )
    frames = tuple(
        _keyframe().model_copy(
            update={
                "evidence_id": f"keyframe_many_{index}",
                "keyframe_id": f"frame_many_{index}",
                "start_ms": 5_000 + index,
                "end_ms": 5_001 + index,
                "timestamp_ms": 5_000 + index,
                "sha256": f"{index + 1:064x}",
                "relative_path": f"visual/keyframes/{index + 1:064x}.jpg",
            },
        )
        for index in range(2)
    )
    observation = _observation().model_copy(
        update={
            "start_ms": 4_900,
            "end_ms": 5_100,
            "keyframe_refs": tuple(frame.evidence_id for frame in frames),
            "transcript_evidence_refs": (transcripts[0].evidence_id,),
            "content_blocks": tuple(
                VisualTextContent(
                    visual_content_id=f"visual_many_{index}",
                    source_keyframe_refs=(frame.evidence_id,),
                    text=f"画面 {index}",
                )
                for index, frame in enumerate(frames)
            ),
            "frame_relations": (),
        },
    )
    temporary = VideoDemoError(ErrorCode.DEPENDENCY_TEMPORARY_FAILURE, "临时失败")

    written = _writer(_TextPort(chapter=lambda _request: temporary)).write(
        _context(),
        (_plan(0), _plan(1)),
        transcripts,
        (observation,),
        frames,
        cache=_cache(tmp_path),
        is_cancel_requested=lambda: False,
    )

    first = written.result.chapters[0]
    assert written.status == "PARTIAL_SUCCEEDED"
    assert all(len(block.evidence_refs) <= 32 for block in first.body_blocks)
    assert len(first.selected_keyframe_refs) <= 1


def test_selected_multiframe_content_requires_relations_between_the_selected_frames(
    tmp_path: Path,
) -> None:
    frames = tuple(
        _keyframe().model_copy(
            update={
                "evidence_id": f"keyframe_relation_{index}",
                "keyframe_id": f"frame_relation_{index}",
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
            "content_blocks": tuple(
                VisualTextContent(
                    visual_content_id=f"visual_relation_{index}",
                    source_keyframe_refs=(frame.evidence_id,),
                    text=f"内容 {index}",
                )
                for index, frame in enumerate(frames)
            ),
            "frame_relations": (
                VisualFrameRelation(
                    relation_type="BEFORE_AFTER",
                    from_keyframe_ref=frames[0].evidence_id,
                    to_keyframe_ref=frames[1].evidence_id,
                    description="只有前两帧存在关系",
                ),
            ),
        },
    )
    unsafe = ChapterWritingResponse(
        title="错误多图关系",
        title_evidence_refs=(observation.evidence_id,),
        summary_zh="选择了没有对应关系的两帧",
        summary_evidence_refs=(observation.evidence_id,),
        body_blocks=(
            VisualBlock(
                visual_observation_ref=observation.evidence_id,
                visual_content_refs=("visual_relation_0", "visual_relation_2"),
                caption="比较第 1 和第 3 帧",
                evidence_refs=(observation.evidence_id,),
            ),
        ),
        claims=(),
    )
    safe = _chapter_response(
        ChapterWritingRequest(
            context=_context(),
            chapter=_plan(0),
            transcript_evidence=(_speech(0),),
            visual_observations=(observation,),
            prompt_version="chapter-writer-v1",
        ),
    ).model_copy(
        update={
            "body_blocks": (
                VisualBlock(
                    visual_observation_ref=observation.evidence_id,
                    visual_content_refs=("visual_relation_0",),
                    caption="只保留第一帧",
                    evidence_refs=(observation.evidence_id,),
                ),
            ),
        },
    )
    port = _TextPort(
        chapter=lambda request: (
            unsafe if request.visual_observations else _chapter_response(request)
        ),
        repair=lambda _request: safe,
    )

    written = _writer(port).write(
        _context(),
        (_plan(0), _plan(1)),
        (_speech(0), _speech(1)),
        (observation,),
        frames,
        cache=_cache(tmp_path),
        is_cancel_requested=lambda: False,
    )

    assert written.metrics["chapter_writer_structure_repairs"] == 1
    assert written.result.chapters[0].selected_keyframe_refs == (frames[0].evidence_id,)


def _materialized_chapter_for_global_test(
    index: int,
    *,
    count: int,
) -> SemanticChapter:
    end_ms = (index + 1) * (1_050_000 // count)
    if index == count - 1:
        end_ms = 1_050_000
    return _empty_semantic_chapter_for_test(index, end_ms=end_ms)


def test_global_chapter_input_retains_content_from_every_grounded_chapter() -> None:
    first = _empty_semantic_chapter_for_test(0, end_ms=10_000).model_copy(
        update={
            "title": "首章",
            "summary_zh": "首章摘要",
        },
    )
    second = _empty_semantic_chapter_for_test(1, end_ms=20_000).model_copy(
        update={
            "start_ms": 10_000,
            "title": "末章唯一标题",
            "summary_zh": "末章唯一摘要",
        },
    )

    inputs = tuple(
        GlobalChapterInput(
            chapter_id=chapter.chapter_id,
            start_ms=chapter.start_ms,
            end_ms=chapter.end_ms,
            title=chapter.title,
            summary_zh=chapter.summary_zh,
            content_status=chapter.content_status,
        )
        for chapter in (first, second)
    )

    assert len(inputs[0].summary_zh) <= 4_000
    assert inputs[0].title == "首章"
    assert inputs[1].title == "末章唯一标题"
    assert inputs[1].summary_zh == "末章唯一摘要"


def _empty_semantic_chapter_for_test(index: int, *, end_ms: int) -> SemanticChapter:
    return SemanticChapter.model_construct(
        chapter_id=f"chapter_global_{index:03d}",
        start_ms=index * 10_000,
        end_ms=end_ms,
        title="题" * 200,
        summary_zh="摘" * 500,
        body_blocks=(),
        claims=(),
        content_status="GROUNDED",
        evidence_refs=("asr",),
        selected_keyframe_refs=(),
        transcript_source="ASR",
    )


def _maximal_escaped_invalid_response() -> InvalidModelResponse:
    return InvalidModelResponse(
        content_sha256="f" * 64,
        validation_errors=tuple(
            f"{index:02d}:" + '"' * 497 for index in range(32)
        ),
        safe_json_excerpt='"' * 8_000,
    )
