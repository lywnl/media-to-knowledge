from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from video_demo.application.pipeline import (
    PipelineRunConfig,
    PreparedMedia,
    ProbedAsset,
    RegisteredAsset,
)
from video_demo.domain.manifest import (
    AudioStream,
    Rational,
    VideoAssetManifest,
    VideoStream,
)
from video_demo.domain.run import ModelIdentity
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.media.probe import ProbeLimits
from video_demo.media.process import ProcessResult
from video_demo.speech import isolated as isolated_module
from video_demo.speech import subprocess_main as subprocess_main_module
from video_demo.speech.isolated import IsolatedSpeechAnalyzer
from video_demo.speech.snapshots import AsrSnapshotPayload, asr_fingerprint
from video_demo.speech.subprocess_protocol import (
    SpeechRuntimeConfig,
    SpeechSubprocessCredentials,
    SpeechSubprocessFailure,
    SpeechSubprocessRequest,
    SpeechSubprocessSuccess,
    ipc_request_payload,
)
from video_demo.storage.artifacts import ArtifactReceipt, AtomicArtifactStore
from video_demo.storage.snapshots import SnapshotStore


def test_asr_only_ipc_reveals_only_openai_key_at_private_encoding_boundary() -> None:
    request = _request(api_key="test-openai-private-key")

    encoded = ipc_request_payload(request)
    serialized = json.dumps(encoded, ensure_ascii=False)

    assert encoded["credentials"] == {"openai_api_key": "test-openai-private-key"}
    assert "test-openai-private-key" not in repr(request)
    assert "test-openai-private-key" not in repr(request.credentials)
    assert "Milvus" not in repr(request)
    assert "向量检索课程" not in repr(request)
    assert "huggingface" not in serialized.casefold()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("stage", "ENRICHMENT"),
        ("speech_fingerprint", "e" * 64),
        (
            "asr_payload_receipt",
            {
                "relative_path": "runs/scope/run_001/speech/snapshots/asr/payload.json",
                "schema_version": "1.1.0",
                "sha256": "f" * 64,
                "upstream_sha256": "d" * 64,
            },
        ),
        ("allow_speaker_fallback", False),
    ],
)
def test_asr_only_ipc_rejects_retired_request_fields(field: str, value: object) -> None:
    payload = _request().model_dump(mode="json")
    payload[field] = value

    with pytest.raises(ValidationError):
        SpeechSubprocessRequest.model_validate(payload)


def test_asr_only_ipc_success_and_failure_bind_request_and_asr_fingerprint() -> None:
    receipt = ArtifactReceipt(
        relative_path="runs/scope/run_001/speech/snapshots/asr/payload.json",
        schema_version="1.3.0",
        sha256="e" * 64,
        upstream_sha256="d" * 64,
    )
    success = SpeechSubprocessSuccess(
        request_id="speech_request_cloud_asr",
        asr_fingerprint="d" * 64,
        payload_receipt=receipt,
    )
    failure = SpeechSubprocessFailure(
        request_id="speech_request_cloud_asr",
        asr_fingerprint="d" * 64,
        error_code=ErrorCode.SPEECH_MODEL_UNAVAILABLE,
        message="语音模型不可用",
    )

    assert success.asr_fingerprint == failure.asr_fingerprint == "d" * 64
    with pytest.raises(ValidationError):
        SpeechSubprocessSuccess.model_validate({**success.model_dump(), "stage": "ASR"})
    with pytest.raises(ValidationError):
        SpeechSubprocessFailure.model_validate(
            {**failure.model_dump(), "speech_fingerprint": "e" * 64}
        )


def test_ipc_request_file_is_private_and_removed_after_success(tmp_path: Path) -> None:
    request_modes: list[int] = []

    class Runner:
        def run(self, args: list[str], **_kwargs: object) -> ProcessResult:
            request_path = tmp_path / Path(args[args.index("--request") + 1])
            request_modes.append(stat.S_IMODE(request_path.stat().st_mode))
            request = SpeechSubprocessRequest.model_validate(
                json.loads(request_path.read_text(encoding="utf-8"))["payload"]
            )
            _publish_success(tmp_path, args, request)
            return ProcessResult(0, b"", b"")

    result = _analyzer(tmp_path, lambda _cancel: Runner()).analyze(_media(tmp_path))

    assert result.transcript_source == "ASR"
    assert request_modes == [0o600]
    assert not list(tmp_path.glob("runs/*/*/speech/ipc/*.json"))


def test_snapshot_hit_does_not_start_subprocess(tmp_path: Path) -> None:
    analyzer = _analyzer(
        tmp_path,
        lambda _cancel: pytest.fail("整段 ASR 快照命中时不得启动子进程"),
    )
    media = _media(tmp_path)
    key = _asr_key(media)
    SnapshotStore(AtomicArtifactStore(tmp_path)).publish(
        media.source.asset.run_relative_root,
        "asr",
        key,
        _asr_payload(),
    )

    result = analyzer.analyze(media)

    assert result.stage_cache_hits == ("SPEECH_ASR",)
    assert result.stage_metrics[0].duration_ms == 0


@pytest.mark.parametrize(
    ("process_code", "expected"),
    [
        (ErrorCode.VIDEO_PROCESS_CANCELLED, ErrorCode.JOB_CANCELLED),
        (ErrorCode.VIDEO_PROCESS_TIMEOUT, ErrorCode.SPEECH_SUBPROCESS_TIMEOUT),
        (ErrorCode.VIDEO_PROCESS_FAILED, ErrorCode.SPEECH_SUBPROCESS_CRASHED),
    ],
)
def test_timeout_and_cancel_clean_only_current_request_slices(
    tmp_path: Path,
    process_code: ErrorCode,
    expected: ErrorCode,
) -> None:
    created: dict[str, Path] = {}

    class Runner:
        def run(self, args: list[str], **_kwargs: object) -> ProcessResult:
            request_path = tmp_path / Path(args[args.index("--request") + 1])
            request = SpeechSubprocessRequest.model_validate(
                json.loads(request_path.read_text(encoding="utf-8"))["payload"]
            )
            slices = tmp_path / request.run_relative_root / "speech/slices"
            slices.mkdir(parents=True, exist_ok=True)
            created["current"] = slices / f"{request.request_id}_{'a' * 24}.wav"
            created["other"] = slices / f"speech_other_{'b' * 24}.wav"
            created["unrelated"] = slices / "unrelated.wav"
            for path in created.values():
                path.write_bytes(b"wav")
            raise VideoDemoError(process_code, "注入进程失败")

    with pytest.raises(VideoDemoError) as raised:
        _analyzer(tmp_path, lambda _cancel: Runner()).analyze(_media(tmp_path))

    assert raised.value.code == expected
    assert not created["current"].exists()
    assert created["other"].is_file()
    assert created["unrelated"].is_file()


def test_failure_response_must_echo_request_and_asr_fingerprint(tmp_path: Path) -> None:
    class Runner:
        def run(self, args: list[str], **_kwargs: object) -> ProcessResult:
            request_path = tmp_path / Path(args[args.index("--request") + 1])
            request_sha = args[args.index("--request-sha256") + 1]
            response = Path(args[args.index("--response") + 1])
            request = SpeechSubprocessRequest.model_validate(
                json.loads(request_path.read_text(encoding="utf-8"))["payload"]
            )
            AtomicArtifactStore(tmp_path).write_json(
                response,
                SpeechSubprocessFailure(
                    request_id=request.request_id,
                    asr_fingerprint="f" * 64,
                    error_code=ErrorCode.SPEECH_MODEL_UNAVAILABLE,
                    message="语音模型不可用",
                ).model_dump(mode="json"),
                schema_version="1.0.0",
                upstream_sha256=request_sha,
                file_mode=0o600,
                exclusive=True,
            )
            return ProcessResult(0, b"", b"")

    with pytest.raises(VideoDemoError) as raised:
        _analyzer(tmp_path, lambda _cancel: Runner()).analyze(_media(tmp_path))

    assert raised.value.code == ErrorCode.SPEECH_SUBPROCESS_RESPONSE_INVALID


def test_subprocess_entry_maps_cloud_temporary_failure_without_secret(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request(api_key="secret-must-not-leak")
    request_path, request_receipt = _write_request(tmp_path, request)
    response = Path(request.run_relative_root) / "speech/ipc" / (
        f"response-{request.request_id}.json"
    )

    def fail(*_args: object, **_kwargs: object) -> Any:
        raise VideoDemoError(ErrorCode.DEPENDENCY_TEMPORARY_FAILURE, "provider secret")

    monkeypatch.setattr(subprocess_main_module, "_execute_request", fail)

    result = subprocess_main_module.main(
        [
            "--workspace-root",
            str(tmp_path),
            "--runtime-root",
            str(tmp_path),
            "--request",
            request_path.as_posix(),
            "--request-sha256",
            request_receipt.sha256,
            "--response",
            response.as_posix(),
        ]
    )

    encoded = (tmp_path / response).read_text(encoding="utf-8")
    payload = json.loads(encoded)["payload"]
    assert result == 0
    assert payload["error_code"] == "DEPENDENCY_TEMPORARY_FAILURE"
    assert payload["asr_fingerprint"] == request.asr_fingerprint
    assert "secret-must-not-leak" not in encoded
    assert "provider secret" not in encoded


def test_open_descendant_directory_closes_all_fds_when_parent_close_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened: list[int] = []
    close_calls = 0
    real_open = isolated_module.os.open
    real_close = isolated_module.os.close
    relative = Path("runs/scope/run_001/speech/ipc")
    (tmp_path / relative).mkdir(parents=True)
    safe_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC

    def tracking_open(*args: object, **kwargs: object) -> int:
        descriptor = real_open(*args, **kwargs)
        opened.append(descriptor)
        return descriptor

    def close_after_real_close(descriptor: int) -> None:
        nonlocal close_calls
        close_calls += 1
        real_close(descriptor)
        if close_calls == 1:
            raise OSError("注入的父目录 fd 关闭错误")

    monkeypatch.setattr(isolated_module.os, "open", tracking_open)
    monkeypatch.setattr(isolated_module.os, "close", close_after_real_close)
    monkeypatch.setattr(isolated_module, "_fd_directory_flags", lambda: safe_flags)

    with pytest.raises(OSError, match="注入的父目录 fd 关闭错误"):
        isolated_module._open_descendant_directory(tmp_path, relative)

    for descriptor in opened:
        with pytest.raises(OSError):
            isolated_module.os.fstat(descriptor)


def _request(*, api_key: str = "test-openai-key") -> SpeechSubprocessRequest:
    return SpeechSubprocessRequest(
        request_id="speech_request_cloud_asr",
        run_relative_root="runs/scope/run_001",
        audio_relative_path="runs/scope/run_001/media/audio.wav",
        audio_sha256=hashlib.sha256(b"wav").hexdigest(),
        duration_ms=1_000,
        config=PipelineRunConfig(hotwords=("Milvus",), core_context="向量检索课程"),
        media_warnings=("SUBTITLE_TRACK_REJECTED:2:INCOMPLETE",),
        runtime=_runtime(),
        credentials=SpeechSubprocessCredentials(openai_api_key=api_key),
        asr_fingerprint="d" * 64,
    )


def _runtime() -> SpeechRuntimeConfig:
    return SpeechRuntimeConfig(
        base_url="https://ai-proxy.example/v1",
        model="openai/whisper",
        timeout_seconds=300,
        max_attempts=3,
        max_window_ms=600_000,
        overlap_ms=1_000,
        model_identities=(
            ModelIdentity(component="silero_vad", provider="local", model_id="silero-vad"),
            ModelIdentity(
                component="cloud_whisper",
                provider="openai_compatible",
                model_id="openai/whisper",
            ),
        ),
        ffmpeg_relative_path="tools/ffmpeg",
    )


def _analyzer(tmp_path: Path, factory: Any) -> IsolatedSpeechAnalyzer:
    store = AtomicArtifactStore(tmp_path)
    return IsolatedSpeechAnalyzer(
        workspace_root=tmp_path,
        runtime_root=tmp_path,
        snapshot_store=SnapshotStore(store),
        artifact_store=store,
        speech_runtime=_runtime(),
        credentials=SpeechSubprocessCredentials(openai_api_key="test-openai-key"),
        timeout_seconds=5,
        process_runner_factory=factory,
    )


def _publish_success(
    runtime_root: Path,
    args: list[str],
    request: SpeechSubprocessRequest,
) -> None:
    response = Path(args[args.index("--response") + 1])
    request_sha = args[args.index("--request-sha256") + 1]
    receipt = SnapshotStore(AtomicArtifactStore(runtime_root)).publish(
        Path(request.run_relative_root),
        "asr",
        request.asr_fingerprint,
        _asr_payload(),
    )
    AtomicArtifactStore(runtime_root).write_json(
        response,
        SpeechSubprocessSuccess(
            request_id=request.request_id,
            asr_fingerprint=request.asr_fingerprint,
            payload_receipt=receipt,
        ).model_dump(mode="json"),
        schema_version="1.0.0",
        upstream_sha256=request_sha,
        file_mode=0o600,
        exclusive=True,
    )


def _write_request(
    runtime_root: Path,
    request: SpeechSubprocessRequest,
) -> tuple[Path, ArtifactReceipt]:
    path = Path(request.run_relative_root) / "speech/ipc" / f"request-{request.request_id}.json"
    receipt = AtomicArtifactStore(runtime_root).write_json(
        path,
        ipc_request_payload(request),
        schema_version="1.0.0",
        upstream_sha256=request.asr_fingerprint,
        file_mode=0o600,
        exclusive=True,
    )
    return path, receipt


def _asr_payload() -> AsrSnapshotPayload:
    return AsrSnapshotPayload(
        language_spans=(),
        segments=(),
        vad_warnings=(),
        silence_boundaries_ms=(),
        language_change_boundaries_ms=(),
    )


def _asr_key(media: PreparedMedia) -> str:
    return asr_fingerprint(
        audio_sha256=media.audio_sha256 or "",
        duration_ms=media.source.duration_ms,
        language_hints=media.source.asset.config.language_hints,
        hotwords=media.source.asset.config.hotwords,
        core_context=media.source.asset.config.core_context,
        inputs=_runtime().fingerprint_inputs(),
    )


def _media(tmp_path: Path) -> PreparedMedia:
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
        config=PipelineRunConfig(),
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
            AudioStream(
                index=1,
                codec_name="pcm_s16le",
                sample_rate_hz=16_000,
                channels=1,
            ),
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
