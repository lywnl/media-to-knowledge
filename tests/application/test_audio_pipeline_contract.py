from pathlib import Path

from video_demo.application.audio_rendering import render_audio_markdown
from video_demo.domain.audio_document import (
    AudioChapter,
    AudioDocumentSummary,
    AudioUnderstandingResult,
)
from video_demo.domain.audio_plan import AudioGroundedClaim, AudioParagraphBlock


def test_audio_asr_receives_core_context_and_hotwords(tmp_path: Path) -> None:
    from video_demo.application.audio_pipeline import AudioSpeechAnalyzer
    from video_demo.application.audio_run_config import AudioRunConfig
    from video_demo.speech.asr_contracts import WindowTranscriptionResult

    class Slicer:
        def create(self, _audio, _root, _slice_id, _range):
            path = tmp_path / "audio-test-slice.wav"
            path.write_bytes(b"slice")
            return path

    class Recognizer:
        def __init__(self) -> None:
            self.prompt = None

        def transcribe_window(self, _slice, *, language_hint, prompt, chunk_index=None):
            self.prompt = prompt
            return WindowTranscriptionResult(language="zh", segments=())

    recognizer = Recognizer()
    analyzer = AudioSpeechAnalyzer(
        recognizer,
        Slicer(),
    )
    analyzer.analyze(
        tmp_path / "audio.wav",
        asset_sha256="a" * 64,
        duration_ms=1_000,
        config=AudioRunConfig(hotwords=("Qwen",), core_context="课程"),
        run_root=Path("runs/audio"),
        is_cancel_requested=lambda: False,
    )
    assert recognizer.prompt == "课程\nQwen"


def test_audio_asr_resumes_completed_windows_from_audio_snapshot(
    tmp_path: Path,
    caplog,
) -> None:
    import logging

    from video_demo.application.audio_pipeline import AudioSpeechAnalyzer
    from video_demo.application.audio_run_config import AudioRunConfig
    from video_demo.speech.asr_contracts import RawAsrSegment, WindowTranscriptionResult
    from video_demo.storage.artifacts import AtomicArtifactStore
    from video_demo.storage.audio_snapshots import AudioAsrWindowSnapshotStore

    class Slicer:
        def __init__(self) -> None:
            self.count = 0

        def create(self, _audio, _root, _slice_id, _range):
            self.count += 1
            path = tmp_path / f"audio-slice-{self.count}.wav"
            path.write_bytes(b"slice")
            return path

    class Recognizer:
        def __init__(self) -> None:
            self.calls = 0
            self.chunk_indexes = []

        def transcribe_window(self, _slice, *, language_hint, prompt, chunk_index=None):
            self.calls += 1
            self.chunk_indexes.append(chunk_index)
            return WindowTranscriptionResult(
                language=language_hint or "zh",
                segments=(RawAsrSegment(0, 1_000, f"窗口 {self.calls}", 0.9),),
            )

    store = AudioAsrWindowSnapshotStore(AtomicArtifactStore(tmp_path))
    recognizer = Recognizer()
    analyzer = AudioSpeechAnalyzer(
        recognizer,
        Slicer(),
        window_store=store,
    )
    run_root = Path("runs/scope_001/run_audio_001")
    analyzer.analyze(
        tmp_path / "audio.wav",
        asset_sha256="b" * 64,
        duration_ms=1_200_000,
        config=AudioRunConfig(language_hints=("zh",)),
        run_root=run_root,
        is_cancel_requested=lambda: False,
    )
    assert recognizer.calls == 2
    assert sorted(recognizer.chunk_indexes) == [0, 1]

    caplog.set_level(logging.INFO, logger="video_demo.application.audio_pipeline")
    analyzer.analyze(
        tmp_path / "audio.wav",
        asset_sha256="b" * 64,
        duration_ms=1_200_000,
        config=AudioRunConfig(language_hints=("zh",)),
        run_root=run_root,
        is_cancel_requested=lambda: False,
    )
    assert recognizer.calls == 2
    assert "音频 ASR 块命中快照 chunk=1/2 segments=1" in caplog.text
    assert "音频 ASR 块命中快照 chunk=2/2 segments=1" in caplog.text
    assert "chunk=1/0" not in caplog.text


def test_audio_asr_fails_the_stage_when_any_fixed_chunk_fails(tmp_path: Path) -> None:
    from video_demo.application.audio_pipeline import AudioSpeechAnalyzer
    from video_demo.application.audio_run_config import AudioRunConfig
    from video_demo.errors import ErrorCode, VideoDemoError
    from video_demo.speech.asr_contracts import RawAsrSegment, WindowTranscriptionResult

    class Slicer:
        def __init__(self) -> None:
            self.count = 0

        def create(self, _audio, _root, _slice_id, _range):
            self.count += 1
            path = tmp_path / f"failed-window-{self.count}.wav"
            path.write_bytes(b"slice")
            return path

    class Recognizer:
        def __init__(self) -> None:
            self.calls = 0

        def transcribe_window(self, _slice, *, language_hint, prompt, chunk_index=None):
            self.calls += 1
            if self.calls == 1:
                raise VideoDemoError(ErrorCode.DEPENDENCY_TEMPORARY_FAILURE, "模拟窗口失败")
            return WindowTranscriptionResult(
                language="zh",
                segments=(RawAsrSegment(0, 1_000, "第二窗口", 0.9),),
            )

    recognizer = Recognizer()
    analyzer = AudioSpeechAnalyzer(
        recognizer,
        Slicer(),
    )
    import pytest

    with pytest.raises(VideoDemoError, match="模拟窗口失败"):
        analyzer.analyze(
            tmp_path / "audio.wav",
            asset_sha256="c" * 64,
            duration_ms=1_200_000,
            config=AudioRunConfig(language_hints=("zh",)),
            run_root=Path("runs/scope_001/run_audio_002"),
            is_cancel_requested=lambda: False,
        )

    assert recognizer.calls <= 2


def test_audio_asr_uses_sentence_and_language_boundaries_without_vad(tmp_path: Path) -> None:
    from video_demo.application.audio_pipeline import AudioSpeechAnalyzer
    from video_demo.application.audio_run_config import AudioRunConfig
    from video_demo.speech.asr_contracts import RawAsrSegment, WindowTranscriptionResult
    class Slicer:
        def __init__(self) -> None:
            self.count = 0

        def create(self, _audio, _root, _slice_id, _range):
            self.count += 1
            path = tmp_path / f"audio-boundary-slice-{self.count}.wav"
            path.write_bytes(b"slice")
            return path

    class Recognizer:
        def __init__(self) -> None:
            self.calls = 0

        def transcribe_window(self, _slice, *, language_hint, prompt, chunk_index=None):
            self.calls += 1
            language = "zh" if self.calls == 1 else "en"
            return WindowTranscriptionResult(
                language=language,
                segments=(RawAsrSegment(0, 1_000, f"窗口 {self.calls}", 0.9),),
            )

    result = AudioSpeechAnalyzer(
        Recognizer(),
        Slicer(),
    ).analyze(
        tmp_path / "audio.wav",
        asset_sha256="d" * 64,
        duration_ms=1_200_000,
        config=AudioRunConfig(),
        run_root=Path("runs/scope_001/run_audio_boundary"),
        is_cancel_requested=lambda: False,
    )

    assert {(item.timestamp_ms, item.source) for item in result.boundary_candidates} >= {
        (1_000, "sentence_end"),
    }
    assert {item.source for item in result.boundary_candidates} <= {
        "sentence_end",
        "language_change",
    }


def test_audio_asr_does_not_send_und_as_explicit_language_hint(tmp_path: Path) -> None:
    from video_demo.application.audio_pipeline import AudioSpeechAnalyzer
    from video_demo.application.audio_run_config import AudioRunConfig
    from video_demo.speech.asr_contracts import WindowTranscriptionResult

    class Slicer:
        def create(self, _audio, _root, _slice_id, _range):
            path = tmp_path / "audio-und-slice.wav"
            path.write_bytes(b"slice")
            return path

    class Recognizer:
        def __init__(self) -> None:
            self.language_hints = []

        def transcribe_window(self, _slice, *, language_hint, prompt, chunk_index=None):
            self.language_hints.append(language_hint)
            return WindowTranscriptionResult(language="zh", segments=())

    recognizer = Recognizer()
    AudioSpeechAnalyzer(recognizer, Slicer()).analyze(
        tmp_path / "audio.wav",
        asset_sha256="e" * 64,
        duration_ms=1_000,
        config=AudioRunConfig(language_hints=("und",)),
        run_root=Path("runs/scope_001/run_audio_und"),
        is_cancel_requested=lambda: False,
    )

    assert recognizer.language_hints == [None]


def test_audio_asr_warns_when_fixed_windows_have_no_valid_segments(tmp_path: Path) -> None:
    from video_demo.application.audio_pipeline import AudioSpeechAnalyzer
    from video_demo.application.audio_run_config import AudioRunConfig
    from video_demo.speech.asr_contracts import WindowTranscriptionResult

    class Slicer:
        def create(self, _audio, _root, _slice_id, _range):
            path = tmp_path / "audio-empty-slice.wav"
            path.write_bytes(b"slice")
            return path

    class Recognizer:
        def transcribe_window(self, _slice, *, language_hint, prompt, chunk_index=None):
            return WindowTranscriptionResult(language="zh", segments=())

    result = AudioSpeechAnalyzer(Recognizer(), Slicer()).analyze(
        tmp_path / "audio.wav",
        asset_sha256="f" * 64,
        duration_ms=1_000,
        config=AudioRunConfig(),
        run_root=Path("runs/scope_001/run_audio_empty"),
        is_cancel_requested=lambda: False,
    )

    assert "AUDIO_ASR_NO_VALID_SEGMENTS" in result.warnings


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
                body_blocks=(AudioParagraphBlock(text="语音正文", evidence_refs=(evidence_id,)),),
                claims=(
                    AudioGroundedClaim(
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


def test_audio_markdown_deduplicates_summary_and_hides_evidence_ids() -> None:
    evidence_id = "asr_evidence_001"
    result = AudioUnderstandingResult(
        run_id="run_audio_002",
        asset_sha256="b" * 64,
        summary=AudioDocumentSummary(
            title="音频标题",
            duration_ms=1_000,
            overview_zh="音频概览",
        ),
        chapters=(
            AudioChapter(
                start_ms=0,
                end_ms=1_000,
                chapter_id="audio_chapter_002",
                title="第一章",
                title_evidence_refs=(evidence_id,),
                summary_zh="同一段内容 [asr_evidence_001]",
                summary_evidence_refs=(evidence_id,),
                body_blocks=(
                    AudioParagraphBlock(
                        text="同一段内容 [asr_evidence_001]",
                        evidence_refs=(evidence_id,),
                    ),
                ),
                claims=(
                    AudioGroundedClaim(
                        text="结论 [asr_evidence_001]",
                        evidence_refs=(evidence_id,),
                        certainty=0.9,
                    ),
                ),
                evidence_refs=(evidence_id,),
                transcript_source="ASR",
            ),
        ),
    )

    text = render_audio_markdown(result).content.decode("utf-8")

    assert "asr_evidence_001" not in text
    assert text.count("同一段内容") == 1


def test_audio_result_contract_has_no_visual_or_rag_fields() -> None:
    payload = _result().model_dump(mode="json")
    serialized = str(payload)
    assert "retrieval_text" not in serialized
    assert "retrieval_hash" not in serialized
    assert "keyframe" not in serialized.lower()


def test_audio_transcription_checkpoint_round_trips_through_audio_payload() -> None:
    from video_demo.application.audio_contracts import (
        AudioStageMetric,
        AudioTranscriptionCheckpoint,
        audio_transcription_checkpoint_from_payload,
        audio_transcription_checkpoint_to_payload,
    )
    from video_demo.domain.audio_plan import AudioBaseSegment
    from video_demo.domain.evidence import SpeechSegment

    evidence = SpeechSegment(
        evidence_id="asr_checkpoint_001",
        start_ms=0,
        end_ms=1_000,
        text="音频片段",
        language="zh",
        confidence=0.9,
        is_fully_evaluated_language=True,
    )
    checkpoint = AudioTranscriptionCheckpoint(
        run_id="run_audio_checkpoint",
        asset_sha256="c" * 64,
        duration_ms=1_000,
        title_hint="音频标题",
        transcript_source="ASR",
        transcript_evidence=(evidence,),
        base_segments=(
            AudioBaseSegment(
                segment_id="audio_segment_checkpoint_001",
                start_ms=0,
                end_ms=1_000,
                evidence_refs=(evidence.evidence_id,),
                transcript_source="ASR",
            ),
        ),
        stage_metrics=(AudioStageMetric("SPEECH_ASR", 12),),
    )

    payload = audio_transcription_checkpoint_to_payload(checkpoint)
    restored = audio_transcription_checkpoint_from_payload(payload)

    assert restored == checkpoint
    assert set(payload) == {
        "schema_version",
        "run_id",
        "asset_sha256",
        "duration_ms",
        "title_hint",
        "transcript_source",
        "transcript_evidence",
        "base_segments",
        "warnings",
        "stage_metrics",
    }


def test_audio_transcription_checkpoint_rejects_unknown_evidence_kind() -> None:
    from video_demo.application.audio_contracts import (
        audio_transcription_checkpoint_from_payload,
    )

    payload = {
        "schema_version": "1.0.0",
        "run_id": "run_audio_checkpoint",
        "asset_sha256": "d" * 64,
        "duration_ms": 1_000,
        "title_hint": "音频标题",
        "transcript_source": "ASR",
        "transcript_evidence": [
            {"kind": "UNKNOWN", "payload": {}},
        ],
        "base_segments": [],
        "warnings": [],
        "stage_metrics": [],
    }

    import pytest

    with pytest.raises(ValueError, match="未知音频证据类型"):
        audio_transcription_checkpoint_from_payload(payload)


def test_audio_uses_scheduler_entrypoint_without_legacy_entrypoint() -> None:
    from pathlib import Path

    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
    assert "video-demo-audio-worker" not in pyproject
    assert not Path("src/video_demo/audio_worker_main.py").exists()


def test_image_pipeline_binds_source_evidence_and_renders_three_sections(tmp_path: Path) -> None:
    from video_demo.application.image_pipeline import run_image_pipeline
    from video_demo.domain.image_document import ImageContentBlock, ImageDocument

    source = tmp_path / "source.png"
    source.write_bytes(
        bytes.fromhex(
            "89504e470d0a1a0a0000000d4948445200000001000000010806000000"
            "1f15c4890000000d49444154789c6360f8cf00000004000101"
            "000018dd8db40000000049454e44ae426082"
        ),
    )

    class Analyzer:
        def analyze(self, *, image_data_url: str, title_hint: str) -> ImageDocument:
            assert image_data_url.startswith("data:image/png;base64,")
            return ImageDocument(
                title="架构图",
                overview_zh="图片概览",
                content_blocks=(
                    ImageContentBlock(
                        content_type="DESCRIPTION",
                        text="图片内容",
                        evidence_refs=("model_ref",),
                    ),
                ),
                claims=(),
                evidence_refs=("model_ref",),
            )

    import hashlib

    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    outcome = run_image_pipeline(
        run_id="run_image_001",
        asset_sha256=digest,
        source=source,
        relative_path="runs/scope/run_image_001/input/source.png",
        mime_type="image/png",
        title_hint="图片",
        analyzer=Analyzer(),
        runtime_root=tmp_path,
    )
    text = outcome.document.content.decode("utf-8")
    assert outcome.result.source.evidence_id in outcome.result.document.evidence_refs
    assert "## 核心概览" in text
    assert "## 图片内容" in text
    assert "## 关键结论" in text
