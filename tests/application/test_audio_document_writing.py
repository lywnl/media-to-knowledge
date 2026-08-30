from __future__ import annotations

from pathlib import Path

from video_demo.application.audio_document_writing import AudioDocumentWriter, AudioWritingContext
from video_demo.domain.audio_plan import AudioChapterPlan, AudioDocumentConfig
from video_demo.domain.evidence import SpeechSegment
from video_demo.storage.document_cache import DocumentModelCache, ModelInvocationIdentity


def _evidence() -> tuple[SpeechSegment, ...]:
    return (
        SpeechSegment(
            evidence_id="asr_001",
            start_ms=0,
            end_ms=1_000,
            text="介绍音频内容",
            language="zh",
            confidence=0.9,
            is_fully_evaluated_language=True,
        ),
    )


def _plan() -> AudioChapterPlan:
    return AudioChapterPlan(
        chapter_id="audio_chapter_001",
        start_ms=0,
        end_ms=1_000,
        segment_refs=("audio_segment_001",),
        title_hint="音频主题",
    )


def _identity(
    operation: str, schema: str, prompt: str, repair_schema: str, repair_prompt: str
) -> ModelInvocationIdentity:
    return ModelInvocationIdentity(
        logical_operation=operation,
        provider_config_fingerprint="a" * 64,
        model_id="text-model",
        generation_config=(("temperature", "0"),),
        main_response_schema_name=schema,
        main_prompt_version=prompt,
        repair_response_schema_name=repair_schema,
        repair_prompt_version=repair_prompt,
    )


class _Port:
    def write_chapter(self, request, *, on_provider_attempt=None):
        from video_demo.domain.audio_plan import AudioGroundedClaim, AudioParagraphBlock
        from video_demo.integrations.audio_document_port import AudioChapterWritingResponse

        return AudioChapterWritingResponse(
            title="音频主题",
            title_evidence_refs=("asr_001",),
            summary_zh="音频内容介绍。",
            summary_evidence_refs=("asr_001",),
            body_blocks=(AudioParagraphBlock(text="音频内容介绍。", evidence_refs=("asr_001",)),),
            claims=(
                AudioGroundedClaim(
                    text="音频介绍了相关内容。", evidence_refs=("asr_001",), certainty=0.9
                ),
            ),
        )

    def repair_chapter_writing(self, request, *, on_provider_attempt=None):
        return self.write_chapter(request.request, on_provider_attempt=on_provider_attempt)

    def organize_document(self, request, *, on_provider_attempt=None):
        from video_demo.integrations.audio_document_port import AudioGlobalWritingResponse

        return AudioGlobalWritingResponse(overview_zh="音频内容介绍。")


class _RepairingGlobalPort(_Port):
    def __init__(self) -> None:
        self.global_calls = 0
        self.repair_calls = 0

    def organize_document(self, request, *, on_provider_attempt=None):
        from video_demo.integrations.audio_document_port import AudioGlobalWritingResponse

        self.global_calls += 1
        return AudioGlobalWritingResponse(overview_zh="")

    def repair_global_writing(self, request, *, on_provider_attempt=None):
        from video_demo.integrations.audio_document_port import AudioGlobalWritingResponse

        self.repair_calls += 1
        return AudioGlobalWritingResponse(overview_zh="修复后的音频概览。")


class _InvalidChapterPort(_Port):
    def __init__(self) -> None:
        self.repair_calls = 0

    def write_chapter(self, request, *, on_provider_attempt=None):
        from video_demo.errors import ErrorCode, VideoDemoError

        raise VideoDemoError(ErrorCode.TEXT_LLM_RESPONSE_INVALID, "模拟模型响应非法")

    def repair_chapter_writing(self, request, *, on_provider_attempt=None):
        self.repair_calls += 1
        return super().write_chapter(request.request, on_provider_attempt=on_provider_attempt)


class _FailedGlobalRepairPort(_RepairingGlobalPort):
    def repair_global_writing(self, request, *, on_provider_attempt=None):
        self.repair_calls += 1
        from video_demo.integrations.audio_document_port import AudioGlobalWritingResponse

        return AudioGlobalWritingResponse(overview_zh="")


class _InvalidGlobalPort(_RepairingGlobalPort):
    def organize_document(self, request, *, on_provider_attempt=None):
        from video_demo.errors import ErrorCode, VideoDemoError

        raise VideoDemoError(ErrorCode.TEXT_LLM_RESPONSE_INVALID, "模拟全局模型响应非法")


def test_audio_writer_returns_audio_result_without_visual_contract(tmp_path: Path) -> None:
    writer = AudioDocumentWriter(
        _Port(),
        chapter_identity=_identity(
            "audio_chapter_writing",
            "audio_chapter_writing_v1",
            "audio-chapter-writer-v1",
            "audio_chapter_writing_repair_v1",
            "audio-chapter-writer-repair-v1",
        ),
        global_identity=_identity(
            "audio_global_editing",
            "audio_global_writing_v1",
            "audio-global-editor-v1",
            "audio_global_writing_repair_v1",
            "audio-global-editor-repair-v1",
        ),
        concurrency=2,
        max_input_chars=60_000,
        max_input_bytes=1_048_576,
        invocation_wait_timeout_seconds=2,
    )
    result = writer.write(
        AudioWritingContext(
            run_id="run_audio_001",
            asset_sha256="b" * 64,
            title_hint="音频",
            duration_ms=1_000,
            transcript_source="ASR",
            document_config=AudioDocumentConfig(),
        ),
        (_plan(),),
        _evidence(),
        cache=DocumentModelCache(tmp_path, max_entry_bytes=1_000_000, max_run_bytes=2_000_000),
        is_cancel_requested=lambda: False,
    )

    payload = result.result.model_dump(mode="json")
    assert payload["summary"]["overview_zh"] == "音频内容介绍。"
    assert "visual_mode" not in str(payload)
    assert "keyframe" not in str(payload).lower()
    assert result.result.chapters[0].claims[0].text == "音频介绍了相关内容。"


def test_audio_writer_repairs_invalid_global_overview_before_fallback(tmp_path: Path) -> None:
    port = _RepairingGlobalPort()
    writer = AudioDocumentWriter(
        port,
        chapter_identity=_identity(
            "audio_chapter_writing",
            "audio_chapter_writing_v1",
            "audio-chapter-writer-v1",
            "audio_chapter_writing_repair_v1",
            "audio-chapter-writer-repair-v1",
        ),
        global_identity=_identity(
            "audio_global_editing",
            "audio_global_writing_v1",
            "audio-global-editor-v1",
            "audio_global_writing_repair_v1",
            "audio-global-editor-repair-v1",
        ),
        concurrency=1,
        max_input_chars=60_000,
        max_input_bytes=1_048_576,
        invocation_wait_timeout_seconds=2,
    )

    result = writer.write(
        AudioWritingContext(
            run_id="run_audio_001",
            asset_sha256="b" * 64,
            title_hint="音频",
            duration_ms=1_000,
            transcript_source="ASR",
            document_config=AudioDocumentConfig(),
        ),
        (_plan(),),
        _evidence(),
        cache=DocumentModelCache(tmp_path, max_entry_bytes=1_000_000, max_run_bytes=2_000_000),
        is_cancel_requested=lambda: False,
    )

    assert port.global_calls == 1
    assert port.repair_calls == 1
    assert result.result.summary.overview_zh == "修复后的音频概览。"
    assert result.status == "SUCCEEDED"


def test_audio_writer_repairs_provider_invalid_global_response_before_fallback(
    tmp_path: Path,
) -> None:
    port = _InvalidGlobalPort()
    writer = AudioDocumentWriter(
        port,
        chapter_identity=_identity(
            "audio_chapter_writing",
            "audio_chapter_writing_v1",
            "audio-chapter-writer-v1",
            "audio_chapter_writing_repair_v1",
            "audio-chapter-writer-repair-v1",
        ),
        global_identity=_identity(
            "audio_global_editing",
            "audio_global_writing_v1",
            "audio-global-editor-v1",
            "audio_global_writing_repair_v1",
            "audio-global-editor-repair-v1",
        ),
        concurrency=1,
        max_input_chars=60_000,
        max_input_bytes=1_048_576,
        invocation_wait_timeout_seconds=2,
    )

    result = writer.write(
        AudioWritingContext(
            run_id="run_audio_001",
            asset_sha256="b" * 64,
            title_hint="音频",
            duration_ms=1_000,
            transcript_source="ASR",
            document_config=AudioDocumentConfig(),
        ),
        (_plan(),),
        _evidence(),
        cache=DocumentModelCache(tmp_path, max_entry_bytes=1_000_000, max_run_bytes=2_000_000),
        is_cancel_requested=lambda: False,
    )

    assert port.repair_calls == 1
    assert result.result.summary.overview_zh == "修复后的音频概览。"
    assert result.status == "SUCCEEDED"


def test_audio_writer_logs_provider_attempts_per_chapter(tmp_path: Path, caplog) -> None:
    from video_demo.domain.audio_plan import AudioGroundedClaim, AudioParagraphBlock
    from video_demo.integrations.audio_document_port import AudioChapterWritingResponse

    evidence = (
        SpeechSegment(
            evidence_id="asr_001",
            start_ms=0,
            end_ms=1_000,
            text="第一段音频",
            language="zh",
            confidence=0.9,
            is_fully_evaluated_language=True,
        ),
        SpeechSegment(
            evidence_id="asr_002",
            start_ms=1_000,
            end_ms=2_000,
            text="第二段音频",
            language="zh",
            confidence=0.9,
            is_fully_evaluated_language=True,
        ),
    )
    plans = (
        _plan(),
        AudioChapterPlan(
            chapter_id="audio_chapter_002",
            start_ms=1_000,
            end_ms=2_000,
            segment_refs=("audio_segment_002",),
            title_hint="第二主题",
        ),
    )

    class Port(_Port):
        def write_chapter(self, request, *, on_provider_attempt=None):
            if on_provider_attempt:
                on_provider_attempt()
            ref = request.transcript_evidence[0].evidence_id
            return AudioChapterWritingResponse(
                title=request.title_hint,
                title_evidence_refs=(ref,),
                summary_zh="音频摘要。",
                summary_evidence_refs=(ref,),
                body_blocks=(AudioParagraphBlock(text="音频正文。", evidence_refs=(ref,)),),
                claims=(
                    AudioGroundedClaim(
                        text="音频结论。", evidence_refs=(ref,), certainty=0.9
                    ),
                ),
            )

    caplog.set_level("INFO", logger="video_demo.application.audio_document_writing")
    writer = AudioDocumentWriter(
        Port(),
        chapter_identity=_identity(
            "audio_chapter_writing",
            "audio_chapter_writing_v1",
            "audio-chapter-writer-v1",
            "audio_chapter_writing_repair_v1",
            "audio-chapter-writer-repair-v1",
        ),
        global_identity=_identity(
            "audio_global_editing",
            "audio_global_writing_v1",
            "audio-global-editor-v1",
            "audio_global_writing_repair_v1",
            "audio-global-editor-repair-v1",
        ),
        concurrency=1,
        max_input_chars=60_000,
        max_input_bytes=1_048_576,
        invocation_wait_timeout_seconds=2,
    )

    writer.write(
        AudioWritingContext(
            run_id="run_audio_001",
            asset_sha256="b" * 64,
            title_hint="音频",
            duration_ms=2_000,
            transcript_source="ASR",
            document_config=AudioDocumentConfig(),
        ),
        plans,
        evidence,
        cache=DocumentModelCache(tmp_path, max_entry_bytes=1_000_000, max_run_bytes=2_000_000),
        is_cancel_requested=lambda: False,
    )

    messages = [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("音频章节写作完成")
    ]
    assert len(messages) == 2
    assert all("provider_attempts=1" in message for message in messages)


def test_audio_writer_repairs_provider_invalid_chapter_before_fallback(tmp_path: Path) -> None:
    port = _InvalidChapterPort()
    writer = AudioDocumentWriter(
        port,
        chapter_identity=_identity(
            "audio_chapter_writing",
            "audio_chapter_writing_v1",
            "audio-chapter-writer-v1",
            "audio_chapter_writing_repair_v1",
            "audio-chapter-writer-repair-v1",
        ),
        global_identity=_identity(
            "audio_global_editing",
            "audio_global_writing_v1",
            "audio-global-editor-v1",
            "audio_global_writing_repair_v1",
            "audio-global-editor-repair-v1",
        ),
        concurrency=1,
        max_input_chars=60_000,
        max_input_bytes=1_048_576,
        invocation_wait_timeout_seconds=2,
    )

    result = writer.write(
        AudioWritingContext(
            run_id="run_audio_001",
            asset_sha256="b" * 64,
            title_hint="音频",
            duration_ms=1_000,
            transcript_source="ASR",
            document_config=AudioDocumentConfig(),
        ),
        (_plan(),),
        _evidence(),
        cache=DocumentModelCache(tmp_path, max_entry_bytes=1_000_000, max_run_bytes=2_000_000),
        is_cancel_requested=lambda: False,
    )

    assert port.repair_calls == 1
    assert result.status == "SUCCEEDED"
    assert result.warnings == ()
    assert result.result.chapters[0].summary_zh == "音频内容介绍。"


def test_audio_writer_marks_global_fallback_when_repair_also_fails(tmp_path: Path) -> None:
    port = _FailedGlobalRepairPort()
    writer = AudioDocumentWriter(
        port,
        chapter_identity=_identity(
            "audio_chapter_writing",
            "audio_chapter_writing_v1",
            "audio-chapter-writer-v1",
            "audio_chapter_writing_repair_v1",
            "audio-chapter-writer-repair-v1",
        ),
        global_identity=_identity(
            "audio_global_editing",
            "audio_global_writing_v1",
            "audio-global-editor-v1",
            "audio_global_writing_repair_v1",
            "audio-global-editor-repair-v1",
        ),
        concurrency=1,
        max_input_chars=60_000,
        max_input_bytes=1_048_576,
        invocation_wait_timeout_seconds=2,
    )

    result = writer.write(
        AudioWritingContext(
            run_id="run_audio_001",
            asset_sha256="b" * 64,
            title_hint="音频",
            duration_ms=1_000,
            transcript_source="ASR",
            document_config=AudioDocumentConfig(),
        ),
        (_plan(),),
        _evidence(),
        cache=DocumentModelCache(tmp_path, max_entry_bytes=1_000_000, max_run_bytes=2_000_000),
        is_cancel_requested=lambda: False,
    )

    assert result.status == "PARTIAL_SUCCEEDED"
    assert result.warnings == ("AUDIO_GLOBAL_WRITING_FALLBACK",)
    assert result.result.summary.overview_zh == "音频内容介绍。"


def test_audio_writer_passes_actual_validation_reason_to_repair_and_counts_attempts(
    tmp_path: Path,
) -> None:
    from video_demo.domain.audio_plan import AudioGroundedClaim, AudioParagraphBlock
    from video_demo.integrations.audio_document_port import AudioChapterWritingResponse

    class Port(_Port):
        def __init__(self) -> None:
            self.repair_errors: tuple[str, ...] = ()
            self.attempts = 0

        def write_chapter(self, request, *, on_provider_attempt=None):
            if on_provider_attempt:
                on_provider_attempt()
            return AudioChapterWritingResponse(
                title="非法标题",
                title_evidence_refs=("unknown",),
                summary_zh="非法引用",
                summary_evidence_refs=("unknown",),
                body_blocks=(AudioParagraphBlock(text="正文", evidence_refs=("unknown",)),),
                claims=(
                    AudioGroundedClaim(text="结论", evidence_refs=("unknown",), certainty=0.9),
                ),
            )

        def repair_chapter_writing(self, request, *, on_provider_attempt=None):
            self.repair_errors = request.invalid_response.validation_errors
            if on_provider_attempt:
                on_provider_attempt()
            return super().write_chapter(request.request, on_provider_attempt=None)

    port = Port()
    writer = AudioDocumentWriter(
        port,
        chapter_identity=_identity(
            "audio_chapter_writing",
            "audio_chapter_writing_v1",
            "audio-chapter-writer-v1",
            "audio_chapter_writing_repair_v1",
            "audio-chapter-writer-repair-v1",
        ),
        global_identity=_identity(
            "audio_global_editing",
            "audio_global_writing_v1",
            "audio-global-editor-v1",
            "audio_global_writing_repair_v1",
            "audio-global-editor-repair-v1",
        ),
        concurrency=1,
        max_input_chars=60_000,
        max_input_bytes=1_048_576,
        invocation_wait_timeout_seconds=2,
    )

    result = writer.write(
        AudioWritingContext(
            run_id="run_audio_001",
            asset_sha256="b" * 64,
            title_hint="音频",
            duration_ms=1_000,
            transcript_source="ASR",
            document_config=AudioDocumentConfig(),
        ),
        (_plan(),),
        _evidence(),
        cache=DocumentModelCache(tmp_path, max_entry_bytes=1_000_000, max_run_bytes=2_000_000),
        is_cancel_requested=lambda: False,
    )

    assert port.repair_errors
    assert "unknown" in port.repair_errors[0]
    assert result.status == "SUCCEEDED"


def test_audio_writer_rejects_chapter_input_budget_before_provider_call(tmp_path: Path) -> None:
    port = _Port()
    writer = AudioDocumentWriter(
        port,
        chapter_identity=_identity(
            "audio_chapter_writing",
            "audio_chapter_writing_v1",
            "audio-chapter-writer-v1",
            "audio_chapter_writing_repair_v1",
            "audio-chapter-writer-repair-v1",
        ),
        global_identity=_identity(
            "audio_global_editing",
            "audio-global-writing-v1",
            "audio-global-editor-v1",
            "audio_global_writing_repair_v1",
            "audio-global-editor-repair-v1",
        ),
        concurrency=1,
        max_input_chars=100,
        max_input_bytes=100,
        invocation_wait_timeout_seconds=2,
    )

    try:
        writer.write(
            AudioWritingContext(
                run_id="run_audio_001",
                asset_sha256="b" * 64,
                title_hint="音频",
                duration_ms=1_000,
                transcript_source="ASR",
                document_config=AudioDocumentConfig(),
            ),
            (_plan(),),
            _evidence(),
            cache=DocumentModelCache(tmp_path, max_entry_bytes=1_000_000, max_run_bytes=2_000_000),
            is_cancel_requested=lambda: False,
        )
    except Exception as error:
        from video_demo.errors import ErrorCode, VideoDemoError

        assert isinstance(error, VideoDemoError)
        assert error.code == ErrorCode.INPUT_BUDGET_EXCEEDED
    else:
        raise AssertionError("预算不足时必须在模型调用前失败关闭")
