from __future__ import annotations

import hashlib
import json
import stat
from pathlib import Path
from typing import Any

import pytest

from video_demo.application.pipeline import (
    PipelineRunConfig,
    PreparedMedia,
    ProbedAsset,
    RegisteredAsset,
    SpeechAnalysis,
)
from video_demo.domain.manifest import AudioStream, Rational, VideoAssetManifest, VideoStream
from video_demo.domain.run import ModelIdentity
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.media.probe import ProbeLimits
from video_demo.media.process import ProcessResult
from video_demo.speech.isolated import IsolatedSpeechAnalyzer
from video_demo.speech.snapshots import (
    AsrSnapshotPayload,
    SpeechAnalysisSnapshotPayload,
    SpeechFingerprintInputs,
    asr_fingerprint,
    speech_fingerprint,
)
from video_demo.speech.subprocess_main import main as subprocess_main
from video_demo.speech.subprocess_protocol import (
    SpeechRuntimeConfig,
    SpeechSubprocessCredentials,
    SpeechSubprocessRequest,
    ipc_request_payload,
)
from video_demo.storage.artifacts import AtomicArtifactStore
from video_demo.storage.snapshots import SnapshotStore


def test_ipc_request_encoding_reveals_secret_only_in_private_file(tmp_path: Path) -> None:
    request = SpeechSubprocessRequest(
        request_id="speech_request_001",
        run_relative_root="runs/scope/run_001",
        audio_relative_path="runs/scope/run_001/media/audio.wav",
        audio_sha256="a" * 64,
        duration_ms=1_000,
        config=PipelineRunConfig(
            hotwords=("Milvus",),
            core_context="向量检索课程",
        ),
        media_warnings=("SUBTITLE_TRACK_REJECTED:2:INCOMPLETE",),
        runtime=SpeechRuntimeConfig(
            inference_device="cpu",
            whisper_compute_type="int8",
            model_identities=(
                ModelIdentity(
                    component="faster_whisper",
                    provider="local",
                    model_id="large-v3",
                ),
            ),
            yamnet_class_map_sha256="b" * 64,
            yamnet_thresholds_sha256="c" * 64,
        ),
        credentials=SpeechSubprocessCredentials(huggingface_token="hf_private_test"),
        asr_fingerprint="d" * 64,
    )
    relative = Path("runs/scope/run_001/speech/ipc/request-speech_request_001.json")
    receipt = AtomicArtifactStore(tmp_path).write_json(
        relative,
        ipc_request_payload(request),
        schema_version="1.0.0",
        upstream_sha256=request.asr_fingerprint,
        file_mode=0o600,
    )

    encoded = (tmp_path / receipt.relative_path).read_text(encoding="utf-8")

    assert "hf_private_test" in encoded
    assert "hf_private_test" not in repr(request)
    assert "Milvus" not in repr(request)
    assert "向量检索课程" not in repr(request)
    assert stat.S_IMODE((tmp_path / receipt.relative_path).stat().st_mode) == 0o600
    assert json.loads(encoded)["payload"]["media_warnings"] == [
        "SUBTITLE_TRACK_REJECTED:2:INCOMPLETE"
    ]


def test_runtime_config_reconstructs_fingerprint_inputs() -> None:
    runtime = SpeechRuntimeConfig(
        inference_device="cpu",
        whisper_compute_type="int8",
        model_identities=(),
        yamnet_class_map_sha256="a" * 64,
        yamnet_thresholds_sha256="b" * 64,
    )

    inputs = runtime.fingerprint_inputs()

    assert inputs.asr_compute_type == "int8"
    assert inputs.yamnet_class_map_sha256 == "a" * 64


def test_subprocess_entry_rejects_request_digest_mismatch_without_response(
    tmp_path: Path,
) -> None:
    runtime_root = tmp_path / ".codex/video-rag-demo"
    runtime_root.mkdir(parents=True)
    run_root = Path("runs/scope/run_001")
    request = SpeechSubprocessRequest(
        request_id="speech_request_001",
        run_relative_root=run_root.as_posix(),
        audio_relative_path=(run_root / "media/audio.wav").as_posix(),
        audio_sha256="a" * 64,
        duration_ms=1_000,
        config=PipelineRunConfig(),
        runtime=SpeechRuntimeConfig(
            inference_device="cpu",
            whisper_compute_type="int8",
            model_identities=(),
            yamnet_class_map_sha256="b" * 64,
            yamnet_thresholds_sha256="c" * 64,
        ),
        credentials=SpeechSubprocessCredentials(),
        asr_fingerprint="d" * 64,
    )
    request_relative = run_root / "speech/ipc/request-speech_request_001.json"
    response_relative = run_root / "speech/ipc/response-speech_request_001.json"
    AtomicArtifactStore(runtime_root).write_json(
        request_relative,
        ipc_request_payload(request),
        schema_version="1.0.0",
        upstream_sha256=request.asr_fingerprint,
        file_mode=0o600,
    )

    result = subprocess_main(
        [
            "--workspace-root",
            str(tmp_path),
            "--runtime-root",
            str(runtime_root),
            "--request",
            request_relative.as_posix(),
            "--request-sha256",
            "f" * 64,
            "--response",
            response_relative.as_posix(),
        ]
    )

    assert result == 2
    assert not (runtime_root / response_relative).exists()


def test_subprocess_entry_rejects_response_path_outside_current_ipc(
    tmp_path: Path,
) -> None:
    runtime_root = tmp_path / ".codex/video-rag-demo"
    runtime_root.mkdir(parents=True)
    run_root = Path("runs/scope/run_001")
    request = SpeechSubprocessRequest(
        request_id="speech_request_001",
        run_relative_root=run_root.as_posix(),
        audio_relative_path=(run_root / "media/audio.wav").as_posix(),
        audio_sha256="a" * 64,
        duration_ms=1_000,
        config=PipelineRunConfig(),
        runtime=SpeechRuntimeConfig(
            inference_device="cpu",
            whisper_compute_type="int8",
            model_identities=(),
            yamnet_class_map_sha256="b" * 64,
            yamnet_thresholds_sha256="c" * 64,
        ),
        credentials=SpeechSubprocessCredentials(),
        asr_fingerprint="d" * 64,
    )
    request_relative = run_root / "speech/ipc/request-speech_request_001.json"
    receipt = AtomicArtifactStore(runtime_root).write_json(
        request_relative,
        ipc_request_payload(request),
        schema_version="1.0.0",
        upstream_sha256=request.asr_fingerprint,
        file_mode=0o600,
    )
    outside_response = Path("runs/scope/run_002/speech/ipc/response.json")

    result = subprocess_main(
        [
            "--workspace-root",
            str(tmp_path),
            "--runtime-root",
            str(runtime_root),
            "--request",
            request_relative.as_posix(),
            "--request-sha256",
            receipt.sha256,
            "--response",
            outside_response.as_posix(),
        ]
    )

    assert result == 2
    assert not (runtime_root / outside_response).exists()


def test_isolated_analyzer_loads_subprocess_snapshot_and_cleans_private_ipc(
    tmp_path: Path,
) -> None:
    calls: list[dict[str, object]] = []

    class Runner:
        def run(self, args: list[str], **kwargs: object) -> ProcessResult:
            calls.append({"args": args, **kwargs})
            request_path = Path(args[args.index("--request") + 1])
            response_path = Path(args[args.index("--response") + 1])
            request_sha = args[args.index("--request-sha256") + 1]
            envelope = json.loads((tmp_path / request_path).read_text(encoding="utf-8"))
            request = SpeechSubprocessRequest.model_validate(envelope["payload"])
            snapshots = SnapshotStore(AtomicArtifactStore(tmp_path))
            asr_receipt = snapshots.publish(
                Path(request.run_relative_root),
                "asr",
                request.asr_fingerprint,
                AsrSnapshotPayload(
                    language_spans=(),
                    segments=(),
                    vad_warnings=("NO_SPEECH_DETECTED",),
                    silence_boundaries_ms=(),
                    language_change_boundaries_ms=(),
                ),
            )
            speech_key = speech_fingerprint(
                processing_mode="ASR",
                transcript_payload_sha256=asr_receipt.sha256,
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
                SpeechAnalysisSnapshotPayload.from_analysis(
                    SpeechAnalysis(
                        transcript_source="ASR",
                        warnings=("NO_SPEECH_DETECTED",),
                    )
                ),
            )
            response = {
                "schema_version": "1.0.0",
                "status": "SUCCEEDED",
                "request_id": request.request_id,
                "speech_fingerprint": speech_key,
                "payload_receipt": speech_receipt.model_dump(mode="json"),
            }
            AtomicArtifactStore(tmp_path).write_json(
                response_path,
                response,
                schema_version="1.0.0",
                upstream_sha256=request_sha,
                file_mode=0o600,
            )
            return ProcessResult(0, b"", b"")

    analyzer = _isolated_analyzer(tmp_path, lambda _cancel: Runner())

    result = analyzer.analyze(_media(tmp_path, hotwords=("Milvus",)))

    assert result.transcript_source == "ASR"
    assert result.warnings == ("NO_SPEECH_DETECTED",)
    assert len(calls) == 1
    command = calls[0]["args"]
    assert isinstance(command, list)
    assert "Milvus" not in " ".join(command)
    environment = calls[0]["env"]
    assert isinstance(environment, dict)
    assert all(not key.startswith("VIDEO_DEMO_") for key in environment)
    ipc = tmp_path / "runs/scope/run_001/speech/ipc"
    assert tuple(ipc.iterdir()) == ()


def test_isolated_analyzer_complete_snapshot_hit_does_not_start_process(
    tmp_path: Path,
) -> None:
    media = _media(tmp_path)
    inputs = _fingerprint_inputs()
    snapshots = SnapshotStore(AtomicArtifactStore(tmp_path))
    asr_key = asr_fingerprint(
        audio_sha256=media.audio_sha256 or "",
        duration_ms=media.source.duration_ms,
        language_hints=(),
        hotwords=(),
        core_context=None,
        inputs=inputs,
    )
    asr_receipt = snapshots.publish(
        media.source.asset.run_relative_root,
        "asr",
        asr_key,
        AsrSnapshotPayload(
            language_spans=(),
            segments=(),
            vad_warnings=(),
            silence_boundaries_ms=(),
            language_change_boundaries_ms=(),
        ),
    )
    speech_key = speech_fingerprint(
        processing_mode="ASR",
        transcript_payload_sha256=asr_receipt.sha256,
        media_warnings=(),
        min_speakers=None,
        max_speakers=None,
        allow_speaker_fallback=False,
        inputs=inputs,
    )
    snapshots.publish(
        media.source.asset.run_relative_root,
        "speech",
        speech_key,
        SpeechAnalysisSnapshotPayload.from_analysis(
            SpeechAnalysis(transcript_source="ASR", warnings=("CACHED",))
        ),
    )

    result = _isolated_analyzer(
        tmp_path,
        lambda _cancel: (_ for _ in ()).throw(AssertionError("不得启动子进程")),
    ).analyze(media)

    assert result.warnings == ("CACHED",)


@pytest.mark.parametrize(
    ("process_error", "expected"),
    [
        (ErrorCode.VIDEO_PROCESS_TIMEOUT, ErrorCode.SPEECH_SUBPROCESS_TIMEOUT),
        (ErrorCode.VIDEO_PROCESS_CANCELLED, ErrorCode.JOB_CANCELLED),
        (ErrorCode.VIDEO_PROCESS_FAILED, ErrorCode.SPEECH_SUBPROCESS_CRASHED),
    ],
)
def test_isolated_analyzer_maps_process_failures_and_cleans_request(
    tmp_path: Path,
    process_error: ErrorCode,
    expected: ErrorCode,
) -> None:
    class Runner:
        def run(self, _args: object, **_kwargs: object) -> ProcessResult:
            raise VideoDemoError(process_error, "第三方敏感错误")

    with pytest.raises(VideoDemoError) as raised:
        _isolated_analyzer(tmp_path, lambda _cancel: Runner()).analyze(_media(tmp_path))

    assert raised.value.code == expected
    assert "敏感" not in raised.value.message
    ipc = tmp_path / "runs/scope/run_001/speech/ipc"
    assert tuple(ipc.iterdir()) == ()


def test_isolated_analyzer_rejects_missing_response_as_non_retryable(
    tmp_path: Path,
) -> None:
    class Runner:
        def run(self, _args: object, **_kwargs: object) -> ProcessResult:
            return ProcessResult(0, b"", b"")

    with pytest.raises(VideoDemoError) as raised:
        _isolated_analyzer(tmp_path, lambda _cancel: Runner()).analyze(_media(tmp_path))

    assert raised.value.code == ErrorCode.SPEECH_SUBPROCESS_RESPONSE_INVALID


def _isolated_analyzer(tmp_path: Path, factory: Any) -> IsolatedSpeechAnalyzer:
    store = AtomicArtifactStore(tmp_path)
    runtime = SpeechRuntimeConfig(
        inference_device="cpu",
        whisper_compute_type="int8",
        model_identities=_fingerprint_inputs().model_identities,
        yamnet_class_map_sha256="b" * 64,
        yamnet_thresholds_sha256="c" * 64,
    )
    return IsolatedSpeechAnalyzer(
        workspace_root=tmp_path,
        runtime_root=tmp_path,
        snapshot_store=SnapshotStore(store),
        artifact_store=store,
        fingerprint_inputs=_fingerprint_inputs(),
        speech_runtime=runtime,
        credentials=SpeechSubprocessCredentials(huggingface_token="hf_private_test"),
        timeout_seconds=5,
        process_runner_factory=factory,
    )


def _fingerprint_inputs() -> SpeechFingerprintInputs:
    return SpeechFingerprintInputs(
        model_identities=(
            ModelIdentity(
                component="faster_whisper",
                provider="local",
                model_id="large-v3",
            ),
        ),
        asr_compute_type="int8",
        yamnet_class_map_sha256="b" * 64,
        yamnet_thresholds_sha256="c" * 64,
    )


def _media(tmp_path: Path, *, hotwords: tuple[str, ...] = ()) -> PreparedMedia:
    run_root = Path("runs/scope/run_001")
    source = tmp_path / run_root / "input/source.mp4"
    audio = tmp_path / run_root / "media/audio.wav"
    proxy = tmp_path / run_root / "media/proxy.mp4"
    audio.parent.mkdir(parents=True, exist_ok=True)
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"source")
    audio.write_bytes(b"wav")
    proxy.write_bytes(b"proxy")
    registered = RegisteredAsset(
        source_path=source,
        source_sha256=hashlib.sha256(b"source").hexdigest(),
        object_ref="obj_001",
        source_size_bytes=6,
        source_mime="video/mp4",
        run_relative_root=run_root,
        config=PipelineRunConfig(hotwords=hotwords),
    )
    manifest = VideoAssetManifest(
        object_ref="obj_001",
        source_sha256=registered.source_sha256,
        source_size_bytes=6,
        source_mime="video/mp4",
        duration_ms=1_000,
        video_stream=VideoStream(
            index=0,
            codec_name="h264",
            width=640,
            height=360,
            average_frame_rate=Rational(numerator=25, denominator=1),
        ),
        audio_streams=(
            AudioStream(index=1, codec_name="pcm_s16le", sample_rate_hz=16_000, channels=1),
        ),
        format_name="mov,mp4",
        ffprobe_version="test",
    )
    return PreparedMedia(
        source=ProbedAsset(registered, manifest, ProbeLimits()),
        proxy_path=proxy,
        proxy_sha256=hashlib.sha256(b"proxy").hexdigest(),
        proxy_size_bytes=5,
        audio_path=audio,
        audio_sha256=hashlib.sha256(b"wav").hexdigest(),
    )
