from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from video_demo.application.pipeline import PipelineRunConfig, ProbedAsset, RegisteredAsset
from video_demo.application.production_media import ProductionMediaTranscoder
from video_demo.domain.evidence import (
    AlignedWord,
    AudioEvent,
    SpeakerTurn,
    SpeechSegment,
    SubtitleCue,
)
from video_demo.domain.manifest import SubtitleStream
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.media.probe import ProbeLimits, parse_ffprobe_payload
from video_demo.media.process import ProcessResult
from video_demo.media.transcode import (
    AudioArtifact,
    ProxyVideoArtifact,
    SubtitleArtifact,
)
from video_demo.speech.isolated import IsolatedSpeechAnalyzer
from video_demo.speech.snapshots import (
    AsrSnapshotPayload,
    SpeechAnalysisSnapshotPayload,
    SpeechFingerprintInputs,
    speech_fingerprint,
)
from video_demo.speech.subprocess_protocol import (
    SpeechRuntimeConfig,
    SpeechSubprocessCredentials,
    SpeechSubprocessRequest,
)
from video_demo.storage.artifacts import AtomicArtifactStore
from video_demo.storage.snapshots import SnapshotStore

_MP4 = b"\x00\x00\x00\x18ftypisom\x00\x00\x02\x00isomiso2"
_FIXTURE = Path(__file__).parents[1] / "media/fixtures/ffprobe/embedded_text_subtitles.json"
_COMPLETE_VTT = (
    "WEBVTT\n\n"
    "00:00:00.000 --> 00:00:30.000\n"
    "这是一条覆盖完整视频并且字符数量足够的有效字幕文本内容\n"
)


@pytest.mark.parametrize(
    ("subtitle_streams", "subtitle_payloads", "expected_source"),
    [
        (
            (SubtitleStream(index=2, codec_name="mov_text", language="zh"),),
            {2: _COMPLETE_VTT},
            "SUBTITLE",
        ),
        (
            (SubtitleStream(index=2, codec_name="mov_text", language="zh"),),
            {2: "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\n太短\n"},
            "ASR",
        ),
    ],
)
def test_subtitle_eligibility_controls_audio_and_isolated_speech_route(
    tmp_path: Path,
    subtitle_streams: tuple[SubtitleStream, ...],
    subtitle_payloads: dict[int, str],
    expected_source: str,
) -> None:
    runtime_root, probed = _probed(
        tmp_path,
        subtitle_streams=subtitle_streams,
    )
    transcoder = _RecordingTranscoder(runtime_root, subtitle_payloads)
    process_calls = 0

    class Runner:
        def run(self, args: list[str], **_kwargs: object) -> ProcessResult:
            nonlocal process_calls
            process_calls += 1
            _publish_successful_asr_response(runtime_root, args)
            return ProcessResult(0, b"", b"")

    media = ProductionMediaTranscoder(
        runtime_root,
        lambda _cancel: transcoder,
    ).transcode(probed)
    result = _isolated_analyzer(runtime_root, lambda _cancel: Runner()).analyze(media)

    assert result.transcript_source == expected_source
    if expected_source == "SUBTITLE":
        assert transcoder.extract_audio_calls == []
        assert media.audio_path is None
        assert process_calls == 0
        assert any(isinstance(item, SubtitleCue) for item in result.evidence)
        assert not any(
            isinstance(item, (SpeechSegment, AlignedWord, SpeakerTurn, AudioEvent))
            for item in result.evidence
        )
        assert not (runtime_root / "runs/scope/run_001/media/audio.wav").exists()
        assert not (runtime_root / "runs/scope/run_001/speech/ipc").exists()
    else:
        assert transcoder.extract_audio_calls == [True]
        assert media.audio_path is not None
        assert process_calls == 1
        assert "SUBTITLE_TRACK_REJECTED:2:INCOMPLETE" in result.warnings
        assert tuple((runtime_root / "runs/scope/run_001/speech/ipc").iterdir()) == ()


def test_video_timeline_truncates_container_length_subtitle_without_starting_asr(
    tmp_path: Path,
) -> None:
    runtime_root, registered = _registered(tmp_path)
    payload = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    payload["format"]["duration"] = "302.366"
    payload["streams"][0]["duration"] = "302.101"
    payload["streams"] = [stream for stream in payload["streams"] if stream["index"] <= 2]
    probe = parse_ffprobe_payload(
        payload,
        object_ref=registered.object_ref,
        source_sha256=registered.source_sha256,
        source_size_bytes=registered.source_size_bytes,
        source_mime=registered.source_mime,
        ffprobe_version="ffprobe test",
        limits=ProbeLimits(),
    )
    probed = ProbedAsset(
        asset=registered,
        manifest=probe.manifest,
        limits=ProbeLimits(),
        warnings=probe.warnings,
        timeline_duration_ms=probe.timeline_duration_ms,
    )
    transcoder = _RecordingTranscoder(runtime_root, {2: _duration_mismatch_vtt()})
    process_calls = 0

    class Runner:
        def run(self, _args: list[str], **_kwargs: object) -> ProcessResult:
            nonlocal process_calls
            process_calls += 1
            raise AssertionError("有效字幕不得启动 ASR 子进程")

    media = ProductionMediaTranscoder(
        runtime_root,
        lambda _cancel: transcoder,
    ).transcode(probed)
    result = _isolated_analyzer(runtime_root, lambda _cancel: Runner()).analyze(media)

    subtitle_cues = tuple(item for item in result.evidence if isinstance(item, SubtitleCue))
    assert probed.manifest.duration_ms == 302_366
    assert probed.duration_ms == 302_101
    assert subtitle_cues[-1].end_ms == 302_101
    assert result.evidence == subtitle_cues
    assert result.transcript_source == "SUBTITLE"
    assert transcoder.extract_audio_calls == []
    assert process_calls == 0
    assert not (runtime_root / "runs/scope/run_001/media/audio.wav").exists()


def test_pgs_is_detected_but_never_parsed_as_text_before_asr(tmp_path: Path) -> None:
    payload = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    payload["streams"] = [
        stream
        for stream in payload["streams"]
        if stream["codec_type"] != "subtitle" or stream["codec_name"] == "hdmv_pgs_subtitle"
    ]
    probe = parse_ffprobe_payload(
        payload,
        object_ref="obj_001",
        source_sha256="a" * 64,
        source_size_bytes=6,
        source_mime="video/mp4",
        ffprobe_version="ffprobe test",
        limits=ProbeLimits(),
    )
    runtime_root, registered = _registered(tmp_path)
    transcoder = _RecordingTranscoder(runtime_root, {})
    media = ProductionMediaTranscoder(
        runtime_root,
        lambda _cancel: transcoder,
    ).transcode(
        ProbedAsset(
            asset=registered,
            manifest=probe.manifest,
            limits=ProbeLimits(),
            warnings=probe.warnings,
        )
    )

    assert [stream.codec_name for stream in probe.manifest.subtitle_streams] == [
        "hdmv_pgs_subtitle"
    ]
    assert transcoder.extract_subtitle_calls == []
    assert transcoder.extract_audio_calls == [True]
    assert media.subtitle is None
    assert media.audio_path is not None


def test_retry_after_crash_reuses_published_asr_snapshot(tmp_path: Path) -> None:
    runtime_root, probed = _probed(tmp_path, subtitle_streams=())
    transcoder = _RecordingTranscoder(runtime_root, {})
    media = ProductionMediaTranscoder(
        runtime_root,
        lambda _cancel: transcoder,
    ).transcode(probed)
    process_calls = 0
    recognizer_calls = 0

    class Runner:
        def run(self, args: list[str], **_kwargs: object) -> ProcessResult:
            nonlocal process_calls, recognizer_calls
            process_calls += 1
            request_path = Path(args[args.index("--request") + 1])
            envelope = json.loads((runtime_root / request_path).read_text(encoding="utf-8"))
            request = SpeechSubprocessRequest.model_validate(envelope["payload"])
            snapshots = SnapshotStore(AtomicArtifactStore(runtime_root))
            cached = snapshots.load(
                Path(request.run_relative_root),
                "asr",
                request.asr_fingerprint,
                AsrSnapshotPayload,
            )
            if cached is None:
                recognizer_calls += 1
                asr_receipt = snapshots.publish(
                    Path(request.run_relative_root),
                    "asr",
                    request.asr_fingerprint,
                    _asr_payload(),
                )
                raise VideoDemoError(ErrorCode.VIDEO_PROCESS_FAILED, "模拟原生崩溃")
            asr_receipt = cached[1]
            _publish_successful_asr_response(
                runtime_root,
                args,
                request=request,
                asr_receipt_sha256=asr_receipt.sha256,
            )
            return ProcessResult(0, b"", b"")

    analyzer = _isolated_analyzer(runtime_root, lambda _cancel: Runner())

    with pytest.raises(VideoDemoError) as raised:
        analyzer.analyze(media)
    result = analyzer.analyze(media)

    assert raised.value.code == ErrorCode.SPEECH_SUBPROCESS_CRASHED
    assert result.transcript_source == "ASR"
    assert process_calls == 2
    assert recognizer_calls == 1
    assert tuple((runtime_root / "runs/scope/run_001/speech/ipc").iterdir()) == ()


def _probed(
    tmp_path: Path,
    *,
    subtitle_streams: tuple[SubtitleStream, ...],
) -> tuple[Path, ProbedAsset]:
    runtime_root, registered = _registered(tmp_path)
    payload = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    payload["streams"] = [
        stream for stream in payload["streams"] if stream["codec_type"] != "subtitle"
    ]
    payload["streams"].extend(
        {
            "index": stream.index,
            "codec_type": "subtitle",
            "codec_name": stream.codec_name,
            "tags": {"language": stream.language},
            "disposition": {"default": int(stream.is_default), "forced": int(stream.is_forced)},
        }
        for stream in subtitle_streams
    )
    payload["format"]["duration"] = "30.000"
    probe = parse_ffprobe_payload(
        payload,
        object_ref=registered.object_ref,
        source_sha256=registered.source_sha256,
        source_size_bytes=registered.source_size_bytes,
        source_mime=registered.source_mime,
        ffprobe_version="ffprobe test",
        limits=ProbeLimits(),
    )
    return runtime_root, ProbedAsset(
        asset=registered,
        manifest=probe.manifest,
        limits=ProbeLimits(),
        warnings=probe.warnings,
    )


def _registered(tmp_path: Path) -> tuple[Path, RegisteredAsset]:
    runtime_root = tmp_path / ".codex/video-rag-demo"
    run_root = Path("runs/scope/run_001")
    source = runtime_root / run_root / "input/source.mp4"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"source")
    return runtime_root, RegisteredAsset(
        source_path=source,
        source_sha256=hashlib.sha256(b"source").hexdigest(),
        object_ref="obj_001",
        source_size_bytes=6,
        source_mime="video/mp4",
        run_relative_root=run_root,
        config=PipelineRunConfig(language_hints=("zh",)),
    )


def _duration_mismatch_vtt() -> str:
    return (
        "WEBVTT\n\n"
        "00:00:00.000 --> 00:00:30.000\n"
        "第一段字幕用于验证容器时长与视频时间轴分离后的完整解析流程\n\n"
        "00:00:30.000 --> 00:01:00.000\n"
        "第二段字幕用于保证字幕数量和文本覆盖率满足有效性判断要求\n\n"
        "00:01:00.000 --> 00:01:30.000\n"
        "第三段字幕继续覆盖视频内容并保留清晰稳定的时间范围信息\n\n"
        "00:01:30.000 --> 00:02:00.000\n"
        "第四段字幕模拟真实中文内嵌字幕轨道中的连续讲解文本内容\n\n"
        "00:02:00.000 --> 00:02:30.000\n"
        "第五段字幕用于验证系统始终优先选择有效文本字幕而不是语音识别\n\n"
        "00:02:30.000 --> 00:03:00.000\n"
        "第六段字幕确保整个样本拥有足够字符并持续覆盖主要视频时间线\n\n"
        "00:03:00.000 --> 00:03:30.000\n"
        "第七段字幕验证后续片段仍然按照主视频流时间轴进行统一处理\n\n"
        "00:03:30.000 --> 00:04:00.000\n"
        "第八段字幕模拟教程视频内完整且连续出现的中文字幕轨文本\n\n"
        "00:04:00.000 --> 00:04:30.000\n"
        "第九段字幕用于覆盖视频后半部分并满足字幕完整性检查条件\n\n"
        "00:04:30.000 --> 00:05:00.000\n"
        "第十段字幕接近主视频流结束位置但仍保持合法的起止时间\n\n"
        "00:05:00.000 --> 00:05:02.366\n"
        "最后一段字幕跟随容器结尾并应被截断到主视频流实际结束位置\n"
    )


class _RecordingTranscoder:
    def __init__(self, runtime_root: Path, subtitle_payloads: dict[int, str]) -> None:
        self.runtime_root = runtime_root
        self.subtitle_payloads = subtitle_payloads
        self.extract_subtitle_calls: list[int] = []
        self.extract_audio_calls: list[bool] = []

    def create_proxy(self, _source: Path, root: Path) -> ProxyVideoArtifact:
        relative = root / "media/proxy.mp4"
        output = self.runtime_root / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(_MP4)
        return ProxyVideoArtifact(
            relative_path=relative.as_posix(),
            sha256=hashlib.sha256(_MP4).hexdigest(),
            size_bytes=len(_MP4),
            max_edge=1280,
            normalized_start_ms=0,
        )

    def extract_subtitle(
        self,
        _source: Path,
        root: Path,
        stream: SubtitleStream,
    ) -> SubtitleArtifact:
        self.extract_subtitle_calls.append(stream.index)
        payload = self.subtitle_payloads[stream.index].encode("utf-8")
        relative = root / "media/subtitles" / f"{stream.index}.vtt"
        output = self.runtime_root / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(payload)
        return SubtitleArtifact(
            relative_path=relative.as_posix(),
            sha256=hashlib.sha256(payload).hexdigest(),
            size_bytes=len(payload),
            stream_index=stream.index,
            language=stream.language,
            codec_name=stream.codec_name,
        )

    def extract_audio(
        self,
        _source: Path,
        root: Path,
        *,
        has_audio: bool,
    ) -> AudioArtifact:
        self.extract_audio_calls.append(has_audio)
        payload = b"wav"
        relative = root / "media/audio.wav"
        output = self.runtime_root / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(payload)
        return AudioArtifact(
            relative_path=relative.as_posix(),
            sha256=hashlib.sha256(payload).hexdigest(),
            size_bytes=len(payload),
            sample_rate_hz=16_000,
            channels=1,
            codec="pcm_s16le",
        )


def _isolated_analyzer(runtime_root: Path, factory: object) -> IsolatedSpeechAnalyzer:
    store = AtomicArtifactStore(runtime_root)
    inputs = _fingerprint_inputs()
    return IsolatedSpeechAnalyzer(
        workspace_root=runtime_root.parent.parent,
        runtime_root=runtime_root,
        snapshot_store=SnapshotStore(store),
        artifact_store=store,
        fingerprint_inputs=inputs,
        speech_runtime=SpeechRuntimeConfig(
            inference_device="cpu",
            whisper_compute_type="int8",
            model_identities=inputs.model_identities,
            yamnet_class_map_sha256="b" * 64,
            yamnet_thresholds_sha256="c" * 64,
        ),
        credentials=SpeechSubprocessCredentials(),
        timeout_seconds=5,
        process_runner_factory=factory,  # type: ignore[arg-type]
    )


def _publish_successful_asr_response(
    runtime_root: Path,
    args: list[str],
    *,
    request: SpeechSubprocessRequest | None = None,
    asr_receipt_sha256: str | None = None,
) -> None:
    request_path = Path(args[args.index("--request") + 1])
    response_path = Path(args[args.index("--response") + 1])
    request_sha = args[args.index("--request-sha256") + 1]
    if request is None:
        envelope = json.loads((runtime_root / request_path).read_text(encoding="utf-8"))
        request = SpeechSubprocessRequest.model_validate(envelope["payload"])
    snapshots = SnapshotStore(AtomicArtifactStore(runtime_root))
    if asr_receipt_sha256 is None:
        asr_receipt_sha256 = snapshots.publish(
            Path(request.run_relative_root),
            "asr",
            request.asr_fingerprint,
            _asr_payload(),
        ).sha256
    speech_key = speech_fingerprint(
        processing_mode="ASR",
        transcript_payload_sha256=asr_receipt_sha256,
        media_warnings=request.media_warnings,
        min_speakers=request.config.min_speakers,
        max_speakers=request.config.max_speakers,
        allow_speaker_fallback=request.allow_speaker_fallback,
        inputs=request.runtime.fingerprint_inputs(),
    )
    speech_receipt = snapshots.publish(
        Path(request.run_relative_root),
        "speech",
        speech_key,
        SpeechAnalysisSnapshotPayload(
            evidence=_asr_payload().segments,
            warnings=request.media_warnings,
            boundary_candidates=(),
            transcript_source="ASR",
        ),
    )
    response = {
        "schema_version": "1.0.0",
        "status": "SUCCEEDED",
        "request_id": request.request_id,
        "speech_fingerprint": speech_key,
        "payload_receipt": speech_receipt.model_dump(mode="json"),
    }
    AtomicArtifactStore(runtime_root).write_json(
        response_path,
        response,
        schema_version="1.0.0",
        upstream_sha256=request_sha,
        file_mode=0o600,
    )


def _asr_payload() -> AsrSnapshotPayload:
    return AsrSnapshotPayload(
        language_spans=(),
        segments=(
            SpeechSegment(
                evidence_id="asr_001",
                start_ms=0,
                end_ms=1_000,
                text="兜底文本",
                language="zh",
                confidence=0.9,
                is_fully_evaluated_language=True,
            ),
        ),
        vad_warnings=(),
        silence_boundaries_ms=(),
        language_change_boundaries_ms=(),
    )


def _fingerprint_inputs() -> SpeechFingerprintInputs:
    return SpeechFingerprintInputs(
        model_identities=(),
        asr_compute_type="int8",
        yamnet_class_map_sha256="b" * 64,
        yamnet_thresholds_sha256="c" * 64,
    )
