from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from video_demo.application.pipeline import (
    PipelineRunConfig,
    PreparedMedia,
    ProbedAsset,
    RegisteredAsset,
    SpeechAnalysis,
    StageMetric,
)
from video_demo.domain.manifest import AudioStream, Rational, VideoAssetManifest, VideoStream
from video_demo.domain.run import ModelIdentity
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.media.probe import ProbeLimits
from video_demo.media.process import ProcessResult
from video_demo.speech import isolated as isolated_module
from video_demo.speech import subprocess_main as subprocess_main_module
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
    ALLOWED_SPEECH_SUBPROCESS_FAILURE_CODES,
    SpeechRuntimeConfig,
    SpeechSubprocessCredentials,
    SpeechSubprocessFailure,
    SpeechSubprocessRequest,
    SpeechSubprocessSuccess,
    ipc_request_payload,
)
from video_demo.storage import artifacts as artifacts_module
from video_demo.storage.artifacts import ArtifactReceipt, AtomicArtifactStore
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
            speech_enrichment_mode="full",
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
        stage="ENRICHMENT",
        speech_fingerprint="e" * 64,
        asr_payload_receipt={
            "relative_path": "runs/scope/run_001/speech/snapshots/asr/payload.json",
            "schema_version": "1.1.0",
            "sha256": "f" * 64,
            "upstream_sha256": "d" * 64,
        },
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


def test_enrichment_request_rejects_text_mode() -> None:
    with pytest.raises(ValidationError, match="ENRICHMENT 请求必须使用 full 模式"):
        SpeechSubprocessRequest(
            request_id="speech_request_text_enrichment",
            run_relative_root="runs/scope/run_001",
            audio_relative_path="runs/scope/run_001/media/audio.wav",
            audio_sha256="a" * 64,
            duration_ms=1_000,
            config=PipelineRunConfig(speech_enrichment_mode="text"),
            runtime=SpeechRuntimeConfig(
                inference_device="cpu",
                whisper_compute_type="int8",
                model_identities=(),
                yamnet_class_map_sha256="b" * 64,
                yamnet_thresholds_sha256="c" * 64,
            ),
            credentials=SpeechSubprocessCredentials(),
            asr_fingerprint="d" * 64,
            stage="ENRICHMENT",
            speech_fingerprint="e" * 64,
            asr_payload_receipt={
                "relative_path": "runs/scope/run_001/speech/snapshots/asr/payload.json",
                "schema_version": "1.1.0",
                "sha256": "f" * 64,
                "upstream_sha256": "d" * 64,
            },
        )


def test_asr_request_rejects_huggingface_token() -> None:
    with pytest.raises(ValidationError, match="ASR 请求不得携带 Hugging Face Token"):
        SpeechSubprocessRequest(
            request_id="speech_request_asr_token",
            run_relative_root="runs/scope/run_001",
            audio_relative_path="runs/scope/run_001/media/audio.wav",
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
            credentials=SpeechSubprocessCredentials(huggingface_token="hf_private_test"),
            asr_fingerprint="d" * 64,
            stage="ASR",
        )


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


def test_subprocess_success_and_failure_bind_processing_stage() -> None:
    success = SpeechSubprocessSuccess(
        request_id="speech_request_001",
        stage="ASR",
        speech_fingerprint="d" * 64,
        payload_receipt={
            "relative_path": "runs/scope/run_001/speech/snapshots/asr/payload.json",
            "schema_version": "1.1.0",
            "sha256": "e" * 64,
            "upstream_sha256": "f" * 64,
        },
    )
    failure = SpeechSubprocessFailure(
        request_id="speech_request_001",
        stage="ENRICHMENT",
        error_code=ErrorCode.SPEECH_MODEL_UNAVAILABLE,
        message="语音模型不可用",
    )

    assert success.stage == "ASR"
    assert failure.stage == "ENRICHMENT"


@pytest.mark.parametrize(
    ("model_type", "payload"),
    (
        (
            SpeechSubprocessRequest,
            {
                "request_id": "speech_request_required_stage",
                "run_relative_root": "runs/scope/run_001",
                "audio_relative_path": "runs/scope/run_001/media/audio.wav",
                "audio_sha256": "a" * 64,
                "duration_ms": 1_000,
                "config": PipelineRunConfig(),
                "runtime": SpeechRuntimeConfig(
                    inference_device="cpu",
                    whisper_compute_type="int8",
                    model_identities=(),
                    yamnet_class_map_sha256="b" * 64,
                    yamnet_thresholds_sha256="c" * 64,
                ),
                "credentials": SpeechSubprocessCredentials(),
                "asr_fingerprint": "d" * 64,
            },
        ),
        (
            SpeechSubprocessSuccess,
            {
                "request_id": "speech_request_required_stage",
                "speech_fingerprint": "d" * 64,
                "payload_receipt": {
                    "relative_path": "runs/scope/run_001/speech/snapshots/asr/payload.json",
                    "schema_version": "1.1.0",
                    "sha256": "e" * 64,
                    "upstream_sha256": "d" * 64,
                },
            },
        ),
        (
            SpeechSubprocessFailure,
            {
                "request_id": "speech_request_required_stage",
                "error_code": ErrorCode.SPEECH_MODEL_UNAVAILABLE,
                "message": "语音模型不可用",
            },
        ),
    ),
)
def test_ipc_stage_is_required(model_type: Any, payload: dict[str, object]) -> None:
    assert model_type.model_fields["stage"].is_required()
    with pytest.raises(ValidationError):
        model_type.model_validate(payload)


def test_subprocess_request_binds_enrichment_to_asr_receipt_and_target_fingerprint() -> None:
    from video_demo.storage.artifacts import ArtifactReceipt

    request = SpeechSubprocessRequest(
        request_id="speech_request_002",
        run_relative_root="runs/scope/run_001",
        audio_relative_path="runs/scope/run_001/media/audio.wav",
        audio_sha256="a" * 64,
        duration_ms=1_000,
        config=PipelineRunConfig(speech_enrichment_mode="full"),
        runtime=SpeechRuntimeConfig(
            inference_device="cpu",
            whisper_compute_type="int8",
            model_identities=(),
            yamnet_class_map_sha256="b" * 64,
            yamnet_thresholds_sha256="c" * 64,
        ),
        credentials=SpeechSubprocessCredentials(),
        asr_fingerprint="d" * 64,
        stage="ENRICHMENT",
        speech_fingerprint="e" * 64,
        asr_payload_receipt=ArtifactReceipt(
            relative_path="runs/scope/run_001/speech/snapshots/asr/payload.json",
            schema_version="1.1.0",
            sha256="f" * 64,
            upstream_sha256="d" * 64,
        ),
    )

    assert request.stage == "ENRICHMENT"
    assert request.speech_fingerprint == "e" * 64
    assert request.asr_payload_receipt is not None
    assert request.asr_payload_receipt.upstream_sha256 == request.asr_fingerprint


def test_enrichment_request_rejects_receipt_bound_to_another_asr() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="上游回执必须绑定 ASR 指纹"):
        SpeechSubprocessRequest(
            request_id="speech_request_003",
            run_relative_root="runs/scope/run_001",
            audio_relative_path="runs/scope/run_001/media/audio.wav",
            audio_sha256="a" * 64,
            duration_ms=1_000,
            config=PipelineRunConfig(speech_enrichment_mode="full"),
            runtime=SpeechRuntimeConfig(
                inference_device="cpu",
                whisper_compute_type="int8",
                model_identities=(),
                yamnet_class_map_sha256="b" * 64,
                yamnet_thresholds_sha256="c" * 64,
            ),
            credentials=SpeechSubprocessCredentials(),
            asr_fingerprint="d" * 64,
            stage="ENRICHMENT",
            speech_fingerprint="e" * 64,
            asr_payload_receipt={
                "relative_path": "runs/scope/run_001/speech/snapshots/asr/payload.json",
                "schema_version": "1.1.0",
                "sha256": "f" * 64,
                "upstream_sha256": "0" * 64,
            },
        )


def test_text_mode_runs_only_asr_subprocess_and_projects_result(tmp_path: Path) -> None:
    calls: list[str] = []

    class Runner:
        def run(self, args: list[str], **_kwargs: object) -> ProcessResult:
            request_path = Path(args[args.index("--request") + 1])
            response_path = Path(args[args.index("--response") + 1])
            request_sha = args[args.index("--request-sha256") + 1]
            request = SpeechSubprocessRequest.model_validate(
                json.loads((tmp_path / request_path).read_text(encoding="utf-8"))["payload"],
            )
            calls.append(request.stage)
            snapshots = SnapshotStore(AtomicArtifactStore(tmp_path))
            receipt = snapshots.publish(
                Path(request.run_relative_root),
                "asr",
                request.asr_fingerprint,
                AsrSnapshotPayload(
                    language_spans=(),
                    segments=(),
                    vad_warnings=(),
                    silence_boundaries_ms=(),
                    language_change_boundaries_ms=(),
                ),
            )
            response = SpeechSubprocessSuccess(
                request_id=request.request_id,
                stage="ASR",
                speech_fingerprint=request.asr_fingerprint,
                payload_receipt=receipt,
            )
            AtomicArtifactStore(tmp_path).write_json(
                response_path,
                response.model_dump(mode="json", exclude_computed_fields=True),
                schema_version="1.0.0",
                upstream_sha256=request_sha,
                file_mode=0o600,
            )
            return ProcessResult(0, b"", b"")

    media = _media(tmp_path)
    media = replace(
        media,
        source=replace(
            media.source,
            asset=replace(
                media.source.asset,
                config=PipelineRunConfig(speech_enrichment_mode="text"),
            ),
        ),
    )
    result = _isolated_analyzer(tmp_path, lambda _cancel: Runner()).analyze(media)

    assert result.enrichment_mode == "text"
    assert calls == ["ASR"]


def test_isolated_text_mode_publishes_text_snapshot_and_reuses_it(tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []

    class Runner:
        def run(self, args: list[str], **kwargs: object) -> ProcessResult:
            calls.append({"args": args, **kwargs})
            request_path = Path(args[args.index("--request") + 1])
            response_path = Path(args[args.index("--response") + 1])
            request_sha = args[args.index("--request-sha256") + 1]
            request = SpeechSubprocessRequest.model_validate(
                json.loads((tmp_path / request_path).read_text(encoding="utf-8"))["payload"],
            )
            snapshots = SnapshotStore(AtomicArtifactStore(tmp_path))
            receipt = snapshots.publish(
                Path(request.run_relative_root),
                "asr",
                request.asr_fingerprint,
                AsrSnapshotPayload(
                    language_spans=(),
                    segments=(),
                    vad_warnings=(),
                    silence_boundaries_ms=(),
                    language_change_boundaries_ms=(),
                ),
            )
            response = SpeechSubprocessSuccess(
                request_id=request.request_id,
                stage="ASR",
                speech_fingerprint=request.asr_fingerprint,
                payload_receipt=receipt,
            )
            AtomicArtifactStore(tmp_path).write_json(
                response_path,
                response.model_dump(mode="json", exclude_computed_fields=True),
                schema_version="1.0.0",
                upstream_sha256=request_sha,
                file_mode=0o600,
            )
            return ProcessResult(0, b"", b"")

    media = replace(
        _media(tmp_path),
        source=replace(
            _media(tmp_path).source,
            asset=replace(
                _media(tmp_path).source.asset,
                config=PipelineRunConfig(speech_enrichment_mode="text"),
            ),
        ),
    )
    analyzer = _isolated_analyzer(tmp_path, lambda _cancel: Runner())

    first = analyzer.analyze(media)
    second = analyzer.analyze(media)

    assert len(calls) == 1
    assert first.stage_metrics[0].stage == "SPEECH_ASR"
    assert second.stage_metrics == (first.stage_metrics[0].__class__("SPEECH_ASR", 0),)
    assert second.stage_cache_hits == ("SPEECH_ASR",)


def test_full_mode_uses_stage_scoped_credentials_and_private_ipc(tmp_path: Path) -> None:
    requests: list[tuple[str, str | None, int]] = []

    class Runner:
        def run(self, args: list[str], **_kwargs: object) -> ProcessResult:
            request_path = tmp_path / Path(args[args.index("--request") + 1])
            response_path = Path(args[args.index("--response") + 1])
            request_sha = args[args.index("--request-sha256") + 1]
            envelope = json.loads(request_path.read_text(encoding="utf-8"))
            token = envelope["payload"]["credentials"]["huggingface_token"]
            request = SpeechSubprocessRequest.model_validate(envelope["payload"])
            requests.append((request.stage, token, stat.S_IMODE(request_path.stat().st_mode)))
            snapshots = SnapshotStore(AtomicArtifactStore(tmp_path))
            if request.stage == "ASR":
                receipt = snapshots.publish(
                    Path(request.run_relative_root),
                    "asr",
                    request.asr_fingerprint,
                    AsrSnapshotPayload(
                        language_spans=(),
                        segments=(),
                        vad_warnings=(),
                        silence_boundaries_ms=(),
                        language_change_boundaries_ms=(),
                    ),
                )
                response_fingerprint = request.asr_fingerprint
            else:
                assert request.speech_fingerprint is not None
                receipt = snapshots.publish(
                    Path(request.run_relative_root),
                    "speech",
                    request.speech_fingerprint,
                    SpeechAnalysisSnapshotPayload.from_analysis(
                        SpeechAnalysis(transcript_source="ASR", enrichment_mode="full")
                    ),
                )
                response_fingerprint = request.speech_fingerprint
            response = SpeechSubprocessSuccess(
                request_id=request.request_id,
                stage=request.stage,
                speech_fingerprint=response_fingerprint,
                payload_receipt=receipt,
            )
            AtomicArtifactStore(tmp_path).write_json(
                response_path,
                response.model_dump(mode="json", exclude_computed_fields=True),
                schema_version="1.0.0",
                upstream_sha256=request_sha,
                file_mode=0o600,
            )
            return ProcessResult(0, b"", b"")

    base_media = _media(tmp_path)
    media = replace(
        base_media,
        source=replace(
            base_media.source,
            asset=replace(
                base_media.source.asset,
                config=PipelineRunConfig(speech_enrichment_mode="full"),
            ),
        ),
    )

    result = _isolated_analyzer(tmp_path, lambda _cancel: Runner()).analyze(media)

    assert result.enrichment_mode == "full"
    assert requests == [
        ("ASR", None, 0o600),
        ("ENRICHMENT", "hf_private_test", 0o600),
    ]


def test_enrichment_missing_on_parent_second_load_includes_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_media = _media(tmp_path)
    media = replace(
        base_media,
        source=replace(
            base_media.source,
            asset=replace(
                base_media.source.asset,
                config=PipelineRunConfig(speech_enrichment_mode="full"),
            ),
        ),
    )
    store = AtomicArtifactStore(tmp_path)
    snapshots = SnapshotStore(store)
    asr_key = asr_fingerprint(
        audio_sha256=media.audio_sha256 or "",
        duration_ms=media.source.duration_ms,
        language_hints=(),
        hotwords=(),
        core_context=None,
        inputs=_fingerprint_inputs(),
    )
    snapshots.publish(
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
    original_load = snapshots.load
    successful_speech_loads = 0

    def fail_second_successful_speech_load(
        run_relative_root: Path,
        kind: str,
        fingerprint: str,
        payload_type: type[Any],
    ) -> Any:
        nonlocal successful_speech_loads
        loaded = original_load(run_relative_root, kind, fingerprint, payload_type)
        if kind == "speech" and loaded is not None:
            successful_speech_loads += 1
            if successful_speech_loads == 2:
                return None
        return loaded

    monkeypatch.setattr(snapshots, "load", fail_second_successful_speech_load)

    class Runner:
        def run(self, args: list[str], **_kwargs: object) -> ProcessResult:
            request_path = Path(args[args.index("--request") + 1])
            response_path = Path(args[args.index("--response") + 1])
            request_sha = args[args.index("--request-sha256") + 1]
            request = SpeechSubprocessRequest.model_validate(
                json.loads((tmp_path / request_path).read_text(encoding="utf-8"))["payload"]
            )
            assert request.stage == "ENRICHMENT"
            assert request.speech_fingerprint is not None
            receipt = snapshots.publish(
                Path(request.run_relative_root),
                "speech",
                request.speech_fingerprint,
                SpeechAnalysisSnapshotPayload.from_analysis(
                    SpeechAnalysis(transcript_source="ASR", enrichment_mode="full")
                ),
            )
            response = SpeechSubprocessSuccess(
                request_id=request.request_id,
                stage="ENRICHMENT",
                speech_fingerprint=request.speech_fingerprint,
                payload_receipt=receipt,
            )
            store.write_json(
                response_path,
                response.model_dump(mode="json", exclude_computed_fields=True),
                schema_version="1.0.0",
                upstream_sha256=request_sha,
                file_mode=0o600,
            )
            return ProcessResult(0, b"", b"")

    analyzer = IsolatedSpeechAnalyzer(
        workspace_root=tmp_path,
        runtime_root=tmp_path,
        snapshot_store=snapshots,
        artifact_store=store,
        fingerprint_inputs=_fingerprint_inputs(),
        speech_runtime=SpeechRuntimeConfig(
            inference_device="cpu",
            whisper_compute_type="int8",
            model_identities=_fingerprint_inputs().model_identities,
            yamnet_class_map_sha256="b" * 64,
            yamnet_thresholds_sha256="c" * 64,
        ),
        credentials=SpeechSubprocessCredentials(huggingface_token="hf_private_test"),
        process_runner_factory=lambda _cancel: Runner(),
    )

    with pytest.raises(VideoDemoError) as raised:
        analyzer.analyze(media)

    assert raised.value.code == ErrorCode.SPEECH_SUBPROCESS_RESPONSE_INVALID
    assert raised.value.details == {"stage": "ENRICHMENT"}


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
        stage="ASR",
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
        stage="ASR",
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


def test_subprocess_entry_writes_private_failure_response(tmp_path: Path) -> None:
    runtime_root = tmp_path / ".codex/video-rag-demo"
    runtime_root.mkdir(parents=True)
    run_root = Path("runs/scope/run_001")
    request = SpeechSubprocessRequest(
        request_id="speech_request_001",
        run_relative_root=run_root.as_posix(),
        audio_relative_path=(run_root / "media/missing.wav").as_posix(),
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
        stage="ASR",
    )
    request_relative = run_root / "speech/ipc/request-speech_request_001.json"
    response_relative = run_root / "speech/ipc/response-speech_request_001.json"
    receipt = AtomicArtifactStore(runtime_root).write_json(
        request_relative,
        ipc_request_payload(request),
        schema_version=request.schema_version,
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
            receipt.sha256,
            "--response",
            response_relative.as_posix(),
        ]
    )

    response_path = runtime_root / response_relative
    envelope = json.loads(response_path.read_text(encoding="utf-8"))
    assert result == 0
    assert stat.S_IMODE(response_path.stat().st_mode) == 0o600
    assert envelope["payload"]["status"] == "FAILED"
    assert envelope["payload"]["stage"] == "ASR"


@pytest.mark.parametrize(
    ("error_code", "message"),
    [
        (ErrorCode.VIDEO_DISK_SPACE_INSUFFICIENT, "可用磁盘空间不足"),
        (ErrorCode.VIDEO_PROCESS_FAILED, "FFmpeg 音频切片失败"),
        (ErrorCode.VIDEO_PROCESS_TIMEOUT, "FFmpeg 音频切片超时"),
        (ErrorCode.VIDEO_OUTPUT_INVALID, "FFmpeg 输出音频非法"),
        (ErrorCode.VIDEO_OUTPUT_TOO_LARGE, "FFmpeg 输出超过大小限制"),
        (ErrorCode.VIDEO_INPUT_INVALID, "FFmpeg 输入音频非法"),
    ],
)
def test_speech_subprocess_preserves_media_failure_codes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error_code: ErrorCode,
    message: str,
) -> None:
    runtime_root = tmp_path / ".codex/video-rag-demo"
    runtime_root.mkdir(parents=True)
    run_root = Path("runs/scope/run_001")
    audio_path = runtime_root / run_root / "media/audio.wav"
    audio_path.parent.mkdir(parents=True)
    audio_path.write_bytes(b"audio")
    request = SpeechSubprocessRequest(
        request_id="speech_request_media_failure",
        run_relative_root=run_root.as_posix(),
        audio_relative_path=(run_root / "media/audio.wav").as_posix(),
        audio_sha256=hashlib.sha256(b"audio").hexdigest(),
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
        stage="ASR",
    )
    request_relative = run_root / "speech/ipc/request-speech_request_media_failure.json"
    response_relative = run_root / "speech/ipc/response-speech_request_media_failure.json"
    receipt = AtomicArtifactStore(runtime_root).write_json(
        request_relative,
        ipc_request_payload(request),
        schema_version=request.schema_version,
        upstream_sha256=request.asr_fingerprint,
        file_mode=0o600,
    )

    def raise_media_error(*_args: object, **_kwargs: object) -> SpeechSubprocessSuccess:
        raise VideoDemoError(error_code, "底层媒体错误")

    monkeypatch.setattr(subprocess_main_module, "_execute_request", raise_media_error)
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
            response_relative.as_posix(),
        ]
    )

    assert result == 0
    envelope = json.loads(
        (runtime_root / response_relative).read_text(encoding="utf-8")
    )
    assert error_code in ALLOWED_SPEECH_SUBPROCESS_FAILURE_CODES
    assert envelope["payload"]["error_code"] == error_code
    assert envelope["payload"]["message"] == message


def test_enrichment_subprocess_rejects_receipt_other_than_bound_asr(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, asr_receipt = _enrichment_subprocess_request(tmp_path)
    wrong_receipt = ArtifactReceipt(
        relative_path=asr_receipt.relative_path,
        schema_version=asr_receipt.schema_version,
        sha256="f" * 64,
        upstream_sha256=asr_receipt.upstream_sha256,
    )
    request = SpeechSubprocessRequest.model_validate(
        {
            **request.model_dump(mode="python"),
            "asr_payload_receipt": wrong_receipt,
        }
    )
    _reject_actual_component_construction(monkeypatch)

    with pytest.raises(VideoDemoError, match="增强请求 ASR 快照校验失败") as raised:
        subprocess_main_module._execute_request(tmp_path, tmp_path, request)

    assert raised.value.code == ErrorCode.SPEECH_SUBPROCESS_RESPONSE_INVALID


def test_enrichment_subprocess_rejects_full_fingerprint_not_derived_from_asr_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, _asr_receipt = _enrichment_subprocess_request(tmp_path)
    request = SpeechSubprocessRequest.model_validate(
        {
            **request.model_dump(mode="python"),
            "speech_fingerprint": "f" * 64,
        }
    )
    _reject_actual_component_construction(monkeypatch)

    with pytest.raises(VideoDemoError, match="增强请求完整指纹不匹配") as raised:
        subprocess_main_module._execute_request(tmp_path, tmp_path, request)

    assert raised.value.code == ErrorCode.SPEECH_SUBPROCESS_RESPONSE_INVALID


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
            response = {
                "schema_version": "1.0.0",
                "status": "SUCCEEDED",
                "request_id": request.request_id,
                "stage": "ASR",
                "speech_fingerprint": request.asr_fingerprint,
                "payload_receipt": asr_receipt.model_dump(mode="json"),
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
    base_media = _media(tmp_path)
    media = replace(
        base_media,
        source=replace(
            base_media.source,
            asset=replace(
                base_media.source.asset,
                config=PipelineRunConfig(speech_enrichment_mode="full"),
            ),
        ),
    )
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
    assert {metric.stage: metric.duration_ms for metric in result.stage_metrics} == {
        "SPEECH_ASR": 0,
        "SPEECH_ENRICHMENT": 0,
    }
    assert result.stage_cache_hits == ("SPEECH_ASR", "SPEECH_ENRICHMENT")


def test_full_mode_asr_snapshot_hit_starts_only_enrichment(
    tmp_path: Path,
) -> None:
    """full 模式命中 ASR 快照时不得再次启动 ASR 子进程。"""
    media = replace(
        _media(tmp_path),
        source=replace(
            _media(tmp_path).source,
            asset=replace(
                _media(tmp_path).source.asset,
                config=PipelineRunConfig(speech_enrichment_mode="full"),
            ),
        ),
    )
    snapshots = SnapshotStore(AtomicArtifactStore(tmp_path))
    inputs = _fingerprint_inputs()
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
    calls: list[tuple[str, int]] = []

    class Runner:
        def run(self, args: list[str], **kwargs: object) -> ProcessResult:
            request_path = Path(args[args.index("--request") + 1])
            response_path = Path(args[args.index("--response") + 1])
            request_sha = args[args.index("--request-sha256") + 1]
            request = SpeechSubprocessRequest.model_validate(
                json.loads((tmp_path / request_path).read_text(encoding="utf-8"))["payload"]
            )
            calls.append((request.stage, int(kwargs["timeout_seconds"])))
            assert request.stage == "ENRICHMENT"
            speech_key = request.speech_fingerprint
            assert speech_key is not None
            speech_receipt = SnapshotStore(AtomicArtifactStore(tmp_path)).publish(
                Path(request.run_relative_root),
                "speech",
                speech_key,
                SpeechAnalysisSnapshotPayload.from_analysis(
                    SpeechAnalysis(transcript_source="ASR", enrichment_mode="full")
                ),
            )
            response = SpeechSubprocessSuccess(
                request_id=request.request_id,
                stage="ENRICHMENT",
                speech_fingerprint=speech_key,
                payload_receipt=speech_receipt,
            )
            AtomicArtifactStore(tmp_path).write_json(
                response_path,
                response.model_dump(mode="json", exclude_computed_fields=True),
                schema_version="1.0.0",
                upstream_sha256=request_sha,
                file_mode=0o600,
            )
            return ProcessResult(0, b"", b"")

    result = _isolated_analyzer(
        tmp_path,
        lambda _cancel: Runner(),
        asr_timeout_seconds=1800,
        enrichment_timeout_seconds=600,
    ).analyze(media)

    assert result.enrichment_mode == "full"
    assert calls == [("ENRICHMENT", 600)]
    assert asr_receipt.sha256
    metrics = {metric.stage: metric.duration_ms for metric in result.stage_metrics}
    assert metrics["SPEECH_ASR"] == 0
    assert metrics["SPEECH_ENRICHMENT"] >= 0
    assert result.stage_cache_hits == ("SPEECH_ASR",)


def test_asr_snapshot_is_reused_after_response_is_lost(tmp_path: Path) -> None:
    runner_calls = 0

    class Runner:
        def run(self, args: list[str], **_kwargs: object) -> ProcessResult:
            nonlocal runner_calls
            runner_calls += 1
            request_path = Path(args[args.index("--request") + 1])
            request = SpeechSubprocessRequest.model_validate(
                json.loads((tmp_path / request_path).read_text(encoding="utf-8"))["payload"]
            )
            assert request.stage == "ASR"
            SnapshotStore(AtomicArtifactStore(tmp_path)).publish(
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
            return ProcessResult(0, b"", b"")

    analyzer = _isolated_analyzer(tmp_path, lambda _cancel: Runner())
    media = _media(tmp_path)

    with pytest.raises(VideoDemoError) as raised:
        analyzer.analyze(media)
    second = analyzer.analyze(media)

    assert raised.value.code == ErrorCode.SPEECH_SUBPROCESS_RESPONSE_INVALID
    assert second.transcript_source == "ASR"
    assert second.stage_metrics == (StageMetric("SPEECH_ASR", 0),)
    assert second.stage_cache_hits == ("SPEECH_ASR",)
    assert runner_calls == 1


def test_enrichment_accepts_bound_result_after_current_asr_changes(
    tmp_path: Path,
) -> None:
    base_media = _media(tmp_path)
    media = replace(
        base_media,
        source=replace(
            base_media.source,
            asset=replace(
                base_media.source.asset,
                config=PipelineRunConfig(speech_enrichment_mode="full"),
            ),
        ),
    )
    snapshots = SnapshotStore(AtomicArtifactStore(tmp_path))
    inputs = _fingerprint_inputs()
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
    changed_asr_key = "c" * 64
    changed_receipts: list[ArtifactReceipt] = []

    class Runner:
        def run(self, args: list[str], **_kwargs: object) -> ProcessResult:
            request_path = Path(args[args.index("--request") + 1])
            response_path = Path(args[args.index("--response") + 1])
            request_sha = args[args.index("--request-sha256") + 1]
            request = SpeechSubprocessRequest.model_validate(
                json.loads((tmp_path / request_path).read_text(encoding="utf-8"))["payload"]
            )
            assert request.stage == "ENRICHMENT"
            assert request.asr_payload_receipt == asr_receipt
            assert request.speech_fingerprint is not None
            speech_receipt = snapshots.publish(
                Path(request.run_relative_root),
                "speech",
                request.speech_fingerprint,
                SpeechAnalysisSnapshotPayload.from_analysis(
                    SpeechAnalysis(transcript_source="ASR", enrichment_mode="full")
                ),
            )
            changed_receipts.append(
                snapshots.publish(
                    Path(request.run_relative_root),
                    "asr",
                    changed_asr_key,
                    AsrSnapshotPayload(
                        language_spans=(),
                        segments=(),
                        vad_warnings=("NO_SPEECH_DETECTED",),
                        silence_boundaries_ms=(),
                        language_change_boundaries_ms=(),
                    ),
                )
            )
            response = SpeechSubprocessSuccess(
                request_id=request.request_id,
                stage="ENRICHMENT",
                speech_fingerprint=request.speech_fingerprint,
                payload_receipt=speech_receipt,
            )
            AtomicArtifactStore(tmp_path).write_json(
                response_path,
                response.model_dump(mode="json", exclude_computed_fields=True),
                schema_version="1.0.0",
                upstream_sha256=request_sha,
                file_mode=0o600,
            )
            return ProcessResult(0, b"", b"")

    result = _isolated_analyzer(tmp_path, lambda _cancel: Runner()).analyze(media)

    assert result.enrichment_mode == "full"
    assert result.stage_cache_hits == ("SPEECH_ASR",)
    assert changed_receipts
    changed_receipt = changed_receipts[0]
    assert changed_receipt.upstream_sha256 == changed_asr_key
    assert changed_receipt.upstream_sha256 != asr_receipt.upstream_sha256
    assert snapshots.load(
        media.source.asset.run_relative_root,
        "asr",
        asr_key,
        AsrSnapshotPayload,
    ) is None


def test_enrichment_failure_response_must_echo_request_stage(tmp_path: Path) -> None:
    media = _media(tmp_path)
    media = replace(
        media,
        source=replace(
            media.source,
            asset=replace(
                media.source.asset,
                config=PipelineRunConfig(speech_enrichment_mode="full"),
            ),
        ),
    )
    snapshots = SnapshotStore(AtomicArtifactStore(tmp_path))
    snapshots.publish(
        media.source.asset.run_relative_root,
        "asr",
        asr_fingerprint(
            audio_sha256=media.audio_sha256 or "",
            duration_ms=media.source.duration_ms,
            language_hints=(),
            hotwords=(),
            core_context=None,
            inputs=_fingerprint_inputs(),
        ),
        AsrSnapshotPayload(
            language_spans=(),
            segments=(),
            vad_warnings=(),
            silence_boundaries_ms=(),
            language_change_boundaries_ms=(),
        ),
    )

    class Runner:
        def run(self, args: list[str], **kwargs: object) -> ProcessResult:
            request_path = Path(args[args.index("--request") + 1])
            response_path = Path(args[args.index("--response") + 1])
            request_sha = args[args.index("--request-sha256") + 1]
            request = SpeechSubprocessRequest.model_validate(
                json.loads((tmp_path / request_path).read_text(encoding="utf-8"))["payload"]
            )
            assert request.stage == "ENRICHMENT"
            response = {
                "schema_version": "1.0.0",
                "status": "FAILED",
                "request_id": request.request_id,
                "stage": "ASR",
                "error_code": ErrorCode.SPEECH_MODEL_UNAVAILABLE,
                "message": "语音模型不可用",
            }
            AtomicArtifactStore(tmp_path).write_json(
                response_path,
                response,
                schema_version="1.0.0",
                upstream_sha256=request_sha,
                file_mode=0o600,
            )
            return ProcessResult(0, b"", b"")

    with pytest.raises(VideoDemoError) as raised:
        _isolated_analyzer(
            tmp_path,
            lambda _cancel: Runner(),
            asr_timeout_seconds=1800,
            enrichment_timeout_seconds=600,
        ).analyze(media)

    assert raised.value.code == ErrorCode.SPEECH_SUBPROCESS_RESPONSE_INVALID
    assert raised.value.details == {"stage": "ENRICHMENT"}


def test_enrichment_business_failure_includes_request_stage(tmp_path: Path) -> None:
    media = _media(tmp_path)
    media = replace(
        media,
        source=replace(
            media.source,
            asset=replace(
                media.source.asset,
                config=PipelineRunConfig(speech_enrichment_mode="full"),
            ),
        ),
    )
    snapshots = SnapshotStore(AtomicArtifactStore(tmp_path))
    snapshots.publish(
        media.source.asset.run_relative_root,
        "asr",
        asr_fingerprint(
            audio_sha256=media.audio_sha256 or "",
            duration_ms=media.source.duration_ms,
            language_hints=(),
            hotwords=(),
            core_context=None,
            inputs=_fingerprint_inputs(),
        ),
        AsrSnapshotPayload(
            language_spans=(),
            segments=(),
            vad_warnings=(),
            silence_boundaries_ms=(),
            language_change_boundaries_ms=(),
        ),
    )

    class Runner:
        def run(self, args: list[str], **_kwargs: object) -> ProcessResult:
            request_path = Path(args[args.index("--request") + 1])
            response_path = Path(args[args.index("--response") + 1])
            request_sha = args[args.index("--request-sha256") + 1]
            request = SpeechSubprocessRequest.model_validate(
                json.loads((tmp_path / request_path).read_text(encoding="utf-8"))["payload"]
            )
            assert request.stage == "ENRICHMENT"
            response = {
                "schema_version": "1.0.0",
                "status": "FAILED",
                "request_id": request.request_id,
                "stage": request.stage,
                "error_code": ErrorCode.SPEECH_MODEL_UNAVAILABLE,
                "message": "语音模型不可用",
            }
            AtomicArtifactStore(tmp_path).write_json(
                response_path,
                response,
                schema_version="1.0.0",
                upstream_sha256=request_sha,
                file_mode=0o600,
            )
            return ProcessResult(0, b"", b"")

    with pytest.raises(VideoDemoError) as raised:
        _isolated_analyzer(
            tmp_path,
            lambda _cancel: Runner(),
            asr_timeout_seconds=1800,
            enrichment_timeout_seconds=600,
        ).analyze(media)

    assert raised.value.code == ErrorCode.SPEECH_MODEL_UNAVAILABLE
    assert raised.value.details == {"stage": "ENRICHMENT"}


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
    assert raised.value.details == {"stage": "ASR"}
    assert "敏感" not in raised.value.message
    ipc = tmp_path / "runs/scope/run_001/speech/ipc"
    assert tuple(ipc.iterdir()) == ()


def test_isolated_analyzer_nonzero_exit_includes_stage(tmp_path: Path) -> None:
    class Runner:
        def run(self, _args: object, **_kwargs: object) -> ProcessResult:
            return ProcessResult(17, b"", b"")

    with pytest.raises(VideoDemoError) as raised:
        _isolated_analyzer(tmp_path, lambda _cancel: Runner()).analyze(_media(tmp_path))

    assert raised.value.code == ErrorCode.SPEECH_SUBPROCESS_CRASHED
    assert raised.value.details == {"stage": "ASR"}


def test_isolated_analyzer_nonzero_exit_with_response_cleans_ipc(tmp_path: Path) -> None:
    class Runner:
        def run(self, args: list[str], **_kwargs: object) -> ProcessResult:
            request_path = Path(args[args.index("--request") + 1])
            response_path = Path(args[args.index("--response") + 1])
            request_sha = args[args.index("--request-sha256") + 1]
            request = SpeechSubprocessRequest.model_validate(
                json.loads((tmp_path / request_path).read_text(encoding="utf-8"))["payload"]
            )
            response = SpeechSubprocessFailure(
                request_id=request.request_id,
                stage=request.stage,
                error_code=ErrorCode.SPEECH_MODEL_UNAVAILABLE,
                message="语音模型不可用",
            )
            AtomicArtifactStore(tmp_path).write_json(
                response_path,
                response.model_dump(mode="json", exclude_computed_fields=True),
                schema_version="1.0.0",
                upstream_sha256=request_sha,
                file_mode=0o600,
            )
            return ProcessResult(17, b"", b"")

    with pytest.raises(VideoDemoError) as raised:
        _isolated_analyzer(tmp_path, lambda _cancel: Runner()).analyze(_media(tmp_path))

    assert raised.value.code == ErrorCode.SPEECH_SUBPROCESS_CRASHED
    assert raised.value.details == {"stage": "ASR"}
    ipc = tmp_path / "runs/scope/run_001/speech/ipc"
    assert tuple(ipc.iterdir()) == ()


def test_isolated_analyzer_oversized_response_cleans_ipc(tmp_path: Path) -> None:
    class Runner:
        def run(self, args: list[str], **_kwargs: object) -> ProcessResult:
            response_path = tmp_path / Path(args[args.index("--response") + 1])
            response_path.write_bytes(b"x" * (64 * 1024 + 1))
            return ProcessResult(0, b"", b"")

    with pytest.raises(VideoDemoError) as raised:
        _isolated_analyzer(tmp_path, lambda _cancel: Runner()).analyze(_media(tmp_path))

    assert raised.value.code == ErrorCode.SPEECH_SUBPROCESS_RESPONSE_INVALID
    assert raised.value.details == {"stage": "ASR"}
    ipc = tmp_path / "runs/scope/run_001/speech/ipc"
    assert tuple(ipc.iterdir()) == ()


def test_isolated_analyzer_cleanup_rejects_parent_directory_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    simulated_outside = tmp_path / "simulated-outside"
    simulated_outside.mkdir()
    response_names: list[str] = []

    class Runner:
        def run(self, args: list[str], **_kwargs: object) -> ProcessResult:
            response_path = tmp_path / Path(args[args.index("--response") + 1])
            response_path.write_text("原目录响应", encoding="utf-8")
            response_names.append(response_path.name)
            (simulated_outside / response_path.name).write_text(
                "不得删除",
                encoding="utf-8",
            )
            return ProcessResult(17, b"", b"")

    real_reject = isolated_module.reject_symlink_components
    swap_triggered = False

    def swap_parent_after_validation(
        root: Path,
        candidate: Path,
        *,
        message: str,
    ) -> Path:
        nonlocal swap_triggered
        validated = real_reject(root, candidate, message=message)
        if not swap_triggered and candidate.name.startswith("response-"):
            ipc = candidate.parent
            original_ipc = ipc.with_name("ipc-before-swap")
            ipc.rename(original_ipc)
            ipc.symlink_to(simulated_outside, target_is_directory=True)
            swap_triggered = True
        return validated

    monkeypatch.setattr(
        isolated_module,
        "reject_symlink_components",
        swap_parent_after_validation,
    )

    with pytest.raises(VideoDemoError) as raised:
        _isolated_analyzer(tmp_path, lambda _cancel: Runner()).analyze(_media(tmp_path))

    assert raised.value.code == ErrorCode.SPEECH_SUBPROCESS_CRASHED
    assert raised.value.details == {"stage": "ASR"}
    assert swap_triggered is True
    assert response_names
    assert (simulated_outside / response_names[0]).read_text(encoding="utf-8") == "不得删除"


def test_isolated_analyzer_cleanup_uses_held_ipc_descriptor_after_path_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    simulated_outside = tmp_path / "simulated-outside"
    simulated_outside.mkdir()
    response_names: list[str] = []

    class Runner:
        def run(self, args: list[str], **_kwargs: object) -> ProcessResult:
            response_path = tmp_path / Path(args[args.index("--response") + 1])
            response_path.write_text("原目录响应", encoding="utf-8")
            response_names.append(response_path.name)
            (simulated_outside / response_path.name).write_text(
                "不得删除",
                encoding="utf-8",
            )
            return ProcessResult(17, b"", b"")

    open_descendant = getattr(isolated_module, "_open_descendant_directory", None)
    assert callable(open_descendant), "语音 IPC 清理必须通过目录描述符定位父目录"
    held_ipc_paths: list[Path] = []
    open_calls = 0

    def swap_parent_after_open(root: Path, relative: Path) -> int:
        nonlocal open_calls
        open_calls += 1
        descriptor = open_descendant(root, relative)
        if open_calls != 2:
            return descriptor
        ipc = root / relative
        original_ipc = ipc.with_name("ipc-after-open")
        ipc.rename(original_ipc)
        ipc.symlink_to(simulated_outside, target_is_directory=True)
        held_ipc_paths.append(original_ipc)
        return descriptor

    monkeypatch.setattr(
        isolated_module,
        "_open_descendant_directory",
        swap_parent_after_open,
    )

    with pytest.raises(VideoDemoError) as raised:
        _isolated_analyzer(tmp_path, lambda _cancel: Runner()).analyze(_media(tmp_path))

    assert raised.value.code == ErrorCode.SPEECH_SUBPROCESS_CRASHED
    assert raised.value.details == {"stage": "ASR"}
    assert response_names
    assert (simulated_outside / response_names[0]).read_text(encoding="utf-8") == "不得删除"
    assert held_ipc_paths
    # 路径校验在取得目录 fd 后执行；发现新命名路径已变成符号链接时，
    # 清理可以安全放弃，不能因此改为跟随新路径。
    assert (held_ipc_paths[0] / response_names[0]).exists()


@pytest.mark.parametrize("artifact_kind", ("request", "response"))
@pytest.mark.parametrize("replacement_kind", ("symlink", "directory"))
def test_verified_ipc_cleanup_rejects_parent_directory_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact_kind: str,
    replacement_kind: str,
) -> None:
    """已取得回执后，清理不能跟随被替换的 IPC 父目录。"""

    external_root = tmp_path / "simulated-outside"
    external_root.mkdir()
    swap_triggered = False
    cleanup_armed = False
    external_target: Path | None = None
    original_target_bytes = b""

    class Runner:
        def run(self, args: list[str], **_kwargs: object) -> ProcessResult:
            request_path = Path(args[args.index("--request") + 1])
            response_path = Path(args[args.index("--response") + 1])
            request_sha = args[args.index("--request-sha256") + 1]
            request = SpeechSubprocessRequest.model_validate(
                json.loads((tmp_path / request_path).read_text(encoding="utf-8"))["payload"]
            )
            response = SpeechSubprocessFailure(
                request_id=request.request_id,
                stage=request.stage,
                error_code=ErrorCode.SPEECH_MODEL_UNAVAILABLE,
                message="语音模型不可用",
            )
            AtomicArtifactStore(tmp_path).write_json(
                response_path,
                response.model_dump(mode="json", exclude_computed_fields=True),
                schema_version="1.0.0",
                upstream_sha256=request_sha,
                file_mode=0o600,
            )
            return ProcessResult(0, b"", b"")

    real_reject = artifacts_module.reject_symlink_components

    def swap_after_validation(
        root: Path,
        candidate: Path,
        *,
        message: str,
    ) -> Path:
        nonlocal external_target, original_target_bytes, swap_triggered
        validated = real_reject(root, candidate, message=message)
        if (
            cleanup_armed
            and not swap_triggered
            and candidate.name.startswith(f"{artifact_kind}-")
        ):
            ipc = candidate.parent
            original_ipc = ipc.with_name("ipc-before-verified-swap")
            ipc.rename(original_ipc)
            target = original_ipc / candidate.name
            original_target_bytes = target.read_bytes()
            external_target = external_root / candidate.name
            external_target.write_bytes(original_target_bytes)
            if replacement_kind == "symlink":
                ipc.symlink_to(external_root, target_is_directory=True)
            else:
                replacement_ipc = ipc.with_name("ipc-replacement")
                replacement_ipc.mkdir()
                replacement_target = replacement_ipc / candidate.name
                replacement_target.write_bytes(original_target_bytes)
                replacement_ipc.rename(ipc)
                external_target = ipc / candidate.name
            swap_triggered = True
        return validated

    monkeypatch.setattr(artifacts_module, "reject_symlink_components", swap_after_validation)
    monkeypatch.setattr(isolated_module, "reject_symlink_components", swap_after_validation)

    analyzer = _isolated_analyzer(tmp_path, lambda _cancel: Runner())
    original_response_receipt = analyzer._response_receipt

    def arm_after_response_receipt(
        response_relative: Path,
        request_sha256: str,
    ) -> ArtifactReceipt:
        nonlocal cleanup_armed
        receipt = original_response_receipt(response_relative, request_sha256)
        cleanup_armed = True
        return receipt

    monkeypatch.setattr(analyzer, "_response_receipt", arm_after_response_receipt)

    with pytest.raises(VideoDemoError) as raised:
        analyzer.analyze(_media(tmp_path))

    assert raised.value.code == ErrorCode.SPEECH_MODEL_UNAVAILABLE
    assert raised.value.details == {"stage": "ASR"}
    assert cleanup_armed is True
    assert swap_triggered is True
    assert external_target is not None
    assert external_target.is_file()
    assert external_target.read_bytes() == original_target_bytes


def test_open_descendant_directory_closes_child_when_parent_close_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """逐级打开目录时，旧父 fd 关闭异常不能泄漏新 child fd。"""

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
        if close_calls == 1:
            real_close(descriptor)
            raise OSError("注入的父目录 fd 关闭错误")
        real_close(descriptor)

    monkeypatch.setattr(isolated_module.os, "open", tracking_open)
    monkeypatch.setattr(isolated_module.os, "close", close_after_real_close)
    monkeypatch.setattr(isolated_module, "_fd_directory_flags", lambda: safe_flags)

    with pytest.raises(OSError, match="注入的父目录 fd 关闭错误"):
        isolated_module._open_descendant_directory(
            tmp_path,
            relative,
        )

    assert len(opened) >= 2
    for descriptor in opened:
        with pytest.raises(OSError):
            isolated_module.os.fstat(descriptor)


@pytest.mark.parametrize("response_kind", ("symlink", "directory"))
def test_isolated_analyzer_cleanup_preserves_non_regular_response_and_original_error(
    tmp_path: Path,
    response_kind: str,
) -> None:
    target = tmp_path / "response-target.json"
    target.write_text("不得删除", encoding="utf-8")
    response_paths: list[Path] = []

    class Runner:
        def run(self, args: list[str], **_kwargs: object) -> ProcessResult:
            response_path = tmp_path / Path(args[args.index("--response") + 1])
            response_paths.append(response_path)
            if response_kind == "symlink":
                response_path.symlink_to(target)
            else:
                response_path.mkdir()
            return ProcessResult(17, b"", b"")

    with pytest.raises(VideoDemoError) as raised:
        _isolated_analyzer(tmp_path, lambda _cancel: Runner()).analyze(_media(tmp_path))

    assert raised.value.code == ErrorCode.SPEECH_SUBPROCESS_CRASHED
    assert raised.value.details == {"stage": "ASR"}
    assert target.read_text(encoding="utf-8") == "不得删除"
    assert response_paths[0].exists() or response_paths[0].is_symlink()


def test_isolated_analyzer_rejects_missing_response_as_non_retryable(
    tmp_path: Path,
) -> None:
    class Runner:
        def run(self, _args: object, **_kwargs: object) -> ProcessResult:
            return ProcessResult(0, b"", b"")

    with pytest.raises(VideoDemoError) as raised:
        _isolated_analyzer(tmp_path, lambda _cancel: Runner()).analyze(_media(tmp_path))

    assert raised.value.code == ErrorCode.SPEECH_SUBPROCESS_RESPONSE_INVALID


def _isolated_analyzer(
    tmp_path: Path,
    factory: Any,
    *,
    asr_timeout_seconds: int = 1800,
    enrichment_timeout_seconds: int = 600,
) -> IsolatedSpeechAnalyzer:
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
        asr_timeout_seconds=asr_timeout_seconds,
        enrichment_timeout_seconds=enrichment_timeout_seconds,
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


def _enrichment_subprocess_request(
    tmp_path: Path,
) -> tuple[SpeechSubprocessRequest, ArtifactReceipt]:
    media = _media(tmp_path)
    inputs = _fingerprint_inputs()
    asr_key = asr_fingerprint(
        audio_sha256=media.audio_sha256 or "",
        duration_ms=media.source.duration_ms,
        language_hints=(),
        hotwords=(),
        core_context=None,
        inputs=inputs,
    )
    asr_receipt = SnapshotStore(AtomicArtifactStore(tmp_path)).publish(
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
        enrichment_mode="full",
    )
    return (
        SpeechSubprocessRequest(
            request_id="speech_enrichment_binding",
            run_relative_root=media.source.asset.run_relative_root.as_posix(),
            audio_relative_path=(
                media.audio_path.relative_to(tmp_path).as_posix()
                if media.audio_path is not None
                else ""
            ),
            audio_sha256=media.audio_sha256 or "",
            duration_ms=media.source.duration_ms,
            config=PipelineRunConfig(speech_enrichment_mode="full"),
            runtime=SpeechRuntimeConfig(
                inference_device="cpu",
                whisper_compute_type="int8",
                model_identities=inputs.model_identities,
                yamnet_class_map_sha256=inputs.yamnet_class_map_sha256,
                yamnet_thresholds_sha256=inputs.yamnet_thresholds_sha256,
            ),
            credentials=SpeechSubprocessCredentials(),
            asr_fingerprint=asr_key,
            stage="ENRICHMENT",
            speech_fingerprint=speech_key,
            asr_payload_receipt=asr_receipt,
        ),
        asr_receipt,
    )


def _reject_actual_component_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    def build_lazy_factory(**_kwargs: object) -> Any:
        def reject_components(_media: object, _cancel: object) -> Any:
            raise AssertionError("错误增强请求不得构造语音组件")

        return reject_components

    monkeypatch.setattr(
        subprocess_main_module,
        "build_subprocess_component_factory",
        build_lazy_factory,
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
