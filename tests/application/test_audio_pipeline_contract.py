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
    from video_demo.speech.asr import WindowTranscriptionResult
    from video_demo.speech.vad import SpeechInterval, VadResult

    class Vad:
        def detect(self, _audio: Path, *, duration_ms: int) -> VadResult:
            return VadResult(
                speech=(
                    SpeechInterval(
                        evidence_id="speech_001",
                        start_ms=0,
                        end_ms=duration_ms,
                        confidence=0.9,
                    ),
                ),
                silence=(),
                long_silence_boundaries_ms=(),
            )

    class Slicer:
        def create(self, _audio, _root, _slice_id, _range):
            path = tmp_path / "audio-test-slice.wav"
            path.write_bytes(b"slice")
            return path

    class Recognizer:
        def __init__(self) -> None:
            self.prompt = None

        def transcribe_window(self, _slice, *, language_hint, prompt):
            self.prompt = prompt
            return WindowTranscriptionResult(language="zh", segments=())

    recognizer = Recognizer()
    analyzer = AudioSpeechAnalyzer(
        Vad(), recognizer, Slicer(), max_window_ms=60_000, overlap_ms=0, max_upload_bytes=2_000_000
    )
    analyzer.analyze(
        tmp_path / "audio.wav",
        duration_ms=1_000,
        config=AudioRunConfig(hotwords=("Qwen",), core_context="课程"),
        run_root=Path("runs/audio"),
        is_cancel_requested=lambda: False,
    )
    assert recognizer.prompt == "课程\nQwen"


def test_audio_asr_resumes_completed_windows_from_audio_snapshot(tmp_path: Path) -> None:
    from video_demo.application.audio_pipeline import AudioSpeechAnalyzer
    from video_demo.application.audio_run_config import AudioRunConfig
    from video_demo.speech.asr_contracts import RawAsrSegment, WindowTranscriptionResult
    from video_demo.speech.vad import SpeechInterval, VadResult
    from video_demo.storage.artifacts import AtomicArtifactStore
    from video_demo.storage.audio_snapshots import AudioAsrWindowSnapshotStore

    class Vad:
        def detect(self, _audio: Path, *, duration_ms: int) -> VadResult:
            return VadResult(
                speech=(
                    SpeechInterval(
                        evidence_id="speech_resume",
                        start_ms=0,
                        end_ms=duration_ms,
                        confidence=0.9,
                    ),
                ),
                silence=(),
                long_silence_boundaries_ms=(),
            )

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

        def transcribe_window(self, _slice, *, language_hint, prompt):
            self.calls += 1
            return WindowTranscriptionResult(
                language=language_hint or "zh",
                segments=(RawAsrSegment(0, 1_000, f"窗口 {self.calls}", 0.9),),
            )

    store = AudioAsrWindowSnapshotStore(AtomicArtifactStore(tmp_path))
    recognizer = Recognizer()
    analyzer = AudioSpeechAnalyzer(
        Vad(),
        recognizer,
        Slicer(),
        max_window_ms=60_000,
        overlap_ms=0,
        max_upload_bytes=2_000_000,
        window_store=store,
    )
    run_root = Path("runs/scope_001/run_audio_001")
    analyzer.analyze(
        tmp_path / "audio.wav",
        duration_ms=120_000,
        config=AudioRunConfig(language_hints=("zh",)),
        run_root=run_root,
        is_cancel_requested=lambda: False,
    )
    assert recognizer.calls == 2

    analyzer.analyze(
        tmp_path / "audio.wav",
        duration_ms=120_000,
        config=AudioRunConfig(language_hints=("zh",)),
        run_root=run_root,
        is_cancel_requested=lambda: False,
    )
    assert recognizer.calls == 2


def test_audio_asr_isolates_a_failed_window_and_continues(tmp_path: Path) -> None:
    from video_demo.application.audio_pipeline import AudioSpeechAnalyzer
    from video_demo.application.audio_run_config import AudioRunConfig
    from video_demo.errors import ErrorCode, VideoDemoError
    from video_demo.speech.asr_contracts import RawAsrSegment, WindowTranscriptionResult
    from video_demo.speech.vad import SpeechInterval, VadResult

    class Vad:
        def detect(self, _audio: Path, *, duration_ms: int) -> VadResult:
            return VadResult(
                speech=(
                    SpeechInterval(
                        evidence_id="speech_one",
                        start_ms=0,
                        end_ms=60_000,
                        confidence=0.9,
                    ),
                    SpeechInterval(
                        evidence_id="speech_two",
                        start_ms=60_000,
                        end_ms=duration_ms,
                        confidence=0.9,
                    ),
                ),
                silence=(),
                long_silence_boundaries_ms=(),
            )

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

        def transcribe_window(self, _slice, *, language_hint, prompt):
            self.calls += 1
            if self.calls == 1:
                raise VideoDemoError(ErrorCode.DEPENDENCY_TEMPORARY_FAILURE, "模拟窗口失败")
            return WindowTranscriptionResult(
                language="zh",
                segments=(RawAsrSegment(0, 1_000, "第二窗口", 0.9),),
            )

    recognizer = Recognizer()
    analyzer = AudioSpeechAnalyzer(
        Vad(),
        recognizer,
        Slicer(),
        max_window_ms=60_000,
        overlap_ms=0,
        max_upload_bytes=2_000_000,
    )
    result = analyzer.analyze(
        tmp_path / "audio.wav",
        duration_ms=120_000,
        config=AudioRunConfig(language_hints=("zh",)),
        run_root=Path("runs/scope_001/run_audio_002"),
        is_cancel_requested=lambda: False,
    )

    assert recognizer.calls == 2
    assert tuple(item.text for item in result.evidence) == ("第二窗口",)
    assert result.warnings == ("AUDIO_ASR_WINDOW_DEGRADED:1",)

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
                    AudioParagraphBlock(text="语音正文", evidence_refs=(evidence_id,)),
                ),
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


def test_audio_result_contract_has_no_visual_or_rag_fields() -> None:
    payload = _result().model_dump(mode="json")
    serialized = str(payload)
    assert "retrieval_text" not in serialized
    assert "retrieval_hash" not in serialized
    assert "keyframe" not in serialized.lower()


def test_media_worker_entrypoints_keep_video_job_type_isolated() -> None:
    from video_demo.application.audio_composition import build_audio_worker
    from video_demo.application.composition import build_image_worker

    assert callable(build_audio_worker)
    assert callable(build_image_worker)


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
