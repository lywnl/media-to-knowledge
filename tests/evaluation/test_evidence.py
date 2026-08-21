from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

import pytest
from pydantic import ValidationError

import video_demo.evaluation.evidence as evidence_module
import video_demo.evaluation.gate as gate_module
from video_demo.domain.run import ModelIdentity
from video_demo.errors import ErrorCode
from video_demo.evaluation.evidence import (
    AuthorizedDatasetDetails,
    BaiduLiveDetails,
    BaiduLiveRawReport,
    CommandEvidenceDetails,
    CommandTrace,
    EvidenceKind,
    EvidenceLevel,
    EvidenceStore,
    FiveLanguageModelsDetails,
    FiveLanguageModelsRawReport,
    LiveInputArtifact,
    LiveSample,
    LiveServiceDetails,
    MachineEvidenceReport,
    ModelExecutionFact,
    PerformanceDetails,
    PerformanceSampleDetails,
    PreflightIssue,
    PreflightRawReport,
    ProviderResponseSummary,
    PyannoteLiveDetails,
    PyannoteLiveRawReport,
    QwenLiveDetails,
    QwenLiveRawReport,
    RealMediaCommand,
    RealMediaDetails,
    RealMediaFile,
    RealMediaRawReport,
    RealMediaSample,
    SetupMediaCommand,
    StaticAuditDetails,
    TraceArtifact,
    build_verified_gate_check,
    load_machine_evidence,
    sha256_file,
    verify_machine_artifacts,
)
from video_demo.evaluation.report import GateStatus

_NOW = "2026-08-18T01:00:00Z"
_LATER = "2026-08-18T01:00:01Z"


def _roots(tmp_path: Path) -> tuple[Path, Path, EvidenceStore]:
    runtime_root = tmp_path / ".codex" / "video-rag-demo"
    runtime_root.mkdir(parents=True)
    return tmp_path, runtime_root, EvidenceStore(tmp_path, runtime_root)


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write(runtime_root: Path, relative_path: str, content: bytes) -> Path:
    path = runtime_root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _real_media_sample(
    case_id: str,
    *,
    execution_status: str = "SUCCESS",
    failure_code: ErrorCode | None = None,
) -> RealMediaSample:
    has_audio = case_id != "no_audio"
    files = [
        RealMediaFile(
            role="SOURCE",
            format="MP4",
            relative_path=f".codex/video-rag-demo/eval/generated/run-1/{case_id}/source.mp4",
            sha256="a" * 64,
            size_bytes=1,
        ),
        RealMediaFile(
            role="PROXY",
            format="MP4",
            relative_path=f".codex/video-rag-demo/eval/generated/run-1/{case_id}/media/proxy.mp4",
            sha256="b" * 64,
            size_bytes=1,
        ),
        RealMediaFile(
            role="KEYFRAME",
            format="JPEG",
            relative_path=f".codex/video-rag-demo/eval/generated/run-1/{case_id}/visual/keyframes/000001.jpg",
            sha256="c" * 64,
            size_bytes=1,
        ),
    ]
    if has_audio:
        files.append(
            RealMediaFile(
                role="AUDIO",
                format="WAV",
                relative_path=f".codex/video-rag-demo/eval/generated/run-1/{case_id}/media/audio.wav",
                sha256="d" * 64,
                size_bytes=1,
            )
        )
    root = f".codex/video-rag-demo/eval/generated/run-1/{case_id}"
    phase_commands = (
        ("generate", "ffmpeg", (), (), (f"{root}/source.mp4",)),
        ("probe", "ffprobe", (f"{root}/source.mp4",), (f"{root}/source.mp4",), ()),
        (
            "proxy",
            "FFmpegTranscoder",
            (f"{root}/source.mp4",),
            (f"{root}/source.mp4",),
            (f"{root}/media/proxy.mp4",),
        ),
        (
            "audio",
            "FFmpegTranscoder",
            (f"{root}/source.mp4",),
            (f"{root}/source.mp4",),
            ((f"{root}/media/audio.wav",) if has_audio else ()),
        ),
        (
            "opencv_decode",
            "OpenCvFrameExtractor",
            (f"{root}/media/proxy.mp4",),
            (f"{root}/media/proxy.mp4",),
            (),
        ),
        (
            "scene_detect",
            "PySceneDetectAdapter",
            (f"{root}/media/proxy.mp4",),
            (f"{root}/media/proxy.mp4",),
            (),
        ),
        (
            "keyframe_select",
            "KeyframeSelector",
            (f"{root}/media/proxy.mp4",),
            (f"{root}/media/proxy.mp4",),
            (f"{root}/visual/keyframes/000001.jpg",),
        ),
    )
    return RealMediaSample(
        case_id=case_id,
        execution_status=execution_status,
        failure_code=failure_code,
        duration_ms=1_000,
        has_audio=has_audio,
        rotation_degrees=90 if case_id == "rotation" else 0,
        is_variable_frame_rate=case_id == "vfr",
        warnings=("NO_AUDIO_TRACK",) if case_id == "no_audio" else (),
        opencv_decoded_frame_count=1,
        scene_count=1,
        selected_keyframe_count=1,
        files=tuple(files),
        commands=tuple(
            RealMediaCommand(
                phase=phase,
                executable=executable,
                arguments=arguments,
                input_relative_paths=inputs,
                output_relative_paths=outputs,
                exit_code=(1 if execution_status == "FAILED" and phase == "generate" else 0),
                stdout_relative_path=(
                    f".codex/video-rag-demo/eval/reports/run-1/{case_id}-{phase}.stdout.txt"
                ),
                stderr_relative_path=(
                    f".codex/video-rag-demo/eval/reports/run-1/{case_id}-{phase}.stderr.txt"
                ),
                stdout_sha256="b" * 64,
                stderr_sha256="c" * 64,
            )
            for phase, executable, arguments, inputs, outputs in (
                phase_commands[:1] if execution_status == "FAILED" else phase_commands
            )
        ),
    )


def _real_media_raw_report(
    *,
    status: GateStatus = GateStatus.PASS,
    samples: tuple[RealMediaSample, ...] | None = None,
    failure_code: ErrorCode | None = None,
) -> RealMediaRawReport:
    return RealMediaRawReport(
        schema_version="1.0.0",
        status=status,
        trace_exit_code=0 if status == GateStatus.PASS else 1,
        evaluation_run_id="run-1",
        ffmpeg_version="ffmpeg version test",
        ffprobe_version="ffprobe version test",
        implementation_sha256="d" * 64,
        setup_commands=(
            SetupMediaCommand(
                phase="ffmpeg_version",
                executable="ffmpeg",
                exit_code=0,
                stdout_relative_path=".codex/video-rag-demo/eval/reports/run-1/setup-ffmpeg.stdout.txt",
                stderr_relative_path=".codex/video-rag-demo/eval/reports/run-1/setup-ffmpeg.stderr.txt",
                stdout_sha256="e" * 64,
                stderr_sha256="f" * 64,
            ),
            SetupMediaCommand(
                phase="ffprobe_version",
                executable="ffprobe",
                exit_code=0,
                stdout_relative_path=".codex/video-rag-demo/eval/reports/run-1/setup-ffprobe.stdout.txt",
                stderr_relative_path=".codex/video-rag-demo/eval/reports/run-1/setup-ffprobe.stderr.txt",
                stdout_sha256="1" * 64,
                stderr_sha256="2" * 64,
            ),
        ),
        samples=samples
        or tuple(
            _real_media_sample(case_id)
            for case_id in ("normal_audio", "no_audio", "rotation", "vfr")
        ),
        failure_code=failure_code,
    )


def _real_media_magic_bytes(media_file: RealMediaFile) -> bytes:
    if media_file.format == "MP4":
        return b"\x00\x00\x00\x18ftypisom\x00\x00\x00\x00isomiso2"
    if media_file.format == "WAV":
        return (
            b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00"
            + b"\x00" * 16
            + b"data\x00\x00\x00\x00"
        )
    return b"\xff\xd8\xff\x00\xff\xd9"


def _wave_bytes(*chunks: tuple[bytes, bytes]) -> bytes:
    body = bytearray(b"WAVE")
    for chunk_id, payload in chunks:
        body.extend(chunk_id)
        body.extend(len(payload).to_bytes(4, byteorder="little"))
        body.extend(payload)
        if len(payload) % 2:
            body.extend(b"\x00")
    return b"RIFF" + len(body).to_bytes(4, byteorder="little") + bytes(body)


_PCM_FMT = b"\x01\x00\x01\x00" + b"\x00" * 12
_EXTENSIBLE_FMT = b"\xfe\xff\x02\x00" + b"\x00" * 36


def _write_real_media_pass_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    status: GateStatus = GateStatus.PASS,
) -> tuple[Path, Path]:
    workspace, runtime_root, store = _roots(tmp_path)
    raw_template = _real_media_raw_report()
    raw_samples: list[RealMediaSample] = []
    artifacts: list[TraceArtifact] = []
    for sample in raw_template.samples:
        files: list[RealMediaFile] = []
        for media_file in sample.files:
            content = _real_media_magic_bytes(media_file)
            path = workspace / media_file.relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            files.append(
                media_file.model_copy(
                    update={
                        "sha256": _digest(content),
                        "size_bytes": len(content),
                    }
                )
            )
            artifacts.append(
                store.bind_artifact(
                    path.relative_to(runtime_root),
                    "INPUT_MEDIA" if media_file.role == "SOURCE" else "OUTPUT_MEDIA",
                )
            )
        commands: list[RealMediaCommand] = []
        for command in sample.commands:
            stdout = f"{sample.case_id}:{command.phase}:stdout\n".encode()
            stderr = f"{sample.case_id}:{command.phase}:stderr\n".encode()
            stdout_path = _write(
                runtime_root,
                f"eval/reports/run-1/{sample.case_id}-{command.phase}.stdout.txt",
                stdout,
            )
            stderr_path = _write(
                runtime_root,
                f"eval/reports/run-1/{sample.case_id}-{command.phase}.stderr.txt",
                stderr,
            )
            artifacts.extend(
                (
                    store.bind_artifact(
                        stdout_path.relative_to(runtime_root), "COMMAND_STDOUT"
                    ),
                    store.bind_artifact(
                        stderr_path.relative_to(runtime_root), "COMMAND_STDERR"
                    ),
                )
            )
            commands.append(
                command.model_copy(
                    update={
                        "stdout_relative_path": stdout_path.relative_to(workspace).as_posix(),
                        "stderr_relative_path": stderr_path.relative_to(workspace).as_posix(),
                        "stdout_sha256": _digest(stdout),
                        "stderr_sha256": _digest(stderr),
                    }
                )
            )
        raw_samples.append(
            sample.model_copy(update={"files": tuple(files), "commands": tuple(commands)})
        )
    setup_commands: list[SetupMediaCommand] = []
    for command in raw_template.setup_commands:
        stdout = _write(
            runtime_root,
            Path(command.stdout_relative_path)
            .relative_to(".codex/video-rag-demo")
            .as_posix(),
            b"setup stdout\n",
        )
        stderr = _write(
            runtime_root,
            Path(command.stderr_relative_path)
            .relative_to(".codex/video-rag-demo")
            .as_posix(),
            b"setup stderr\n",
        )
        artifacts.extend(
            (
                store.bind_artifact(stdout.relative_to(runtime_root), "COMMAND_STDOUT"),
                store.bind_artifact(stderr.relative_to(runtime_root), "COMMAND_STDERR"),
            )
        )
        setup_commands.append(
            command.model_copy(
                update={
                    "stdout_sha256": _digest(stdout.read_bytes()),
                    "stderr_sha256": _digest(stderr.read_bytes()),
                }
            )
        )
    raw = raw_template.model_copy(
        update={"samples": tuple(raw_samples), "setup_commands": tuple(setup_commands)}
    )
    if status == GateStatus.FAIL:
        failed_commands = (*raw.samples[-1].commands[:-1], raw.samples[-1].commands[-1].model_copy(
            update={"exit_code": 1}
        ))
        failed_sample = raw.samples[-1].model_copy(
            update={
                "execution_status": "FAILED",
                "failure_code": ErrorCode.VIDEO_PROCESS_FAILED,
                "commands": failed_commands,
            }
        )
        raw = raw.model_copy(
            update={
                "status": GateStatus.FAIL,
                "trace_exit_code": 1,
                "failure_code": ErrorCode.VIDEO_PROCESS_FAILED,
                "samples": (*raw.samples[:-1], failed_sample),
            }
        )
    raw = RealMediaRawReport.model_validate(raw.model_dump(mode="python"))
    raw_path = _write(
        runtime_root,
        "eval/reports/run-1/raw.json",
        raw.model_dump_json().encode("utf-8"),
    )
    trace_stdout = _write(runtime_root, "eval/reports/run-1/trace.stdout.txt", b"trace\n")
    trace_stderr = _write(runtime_root, "eval/reports/run-1/trace.stderr.txt", b"")
    artifacts.extend(
        (
            store.bind_artifact(raw_path.relative_to(runtime_root), "AUDIT_REPORT"),
            store.bind_artifact(trace_stdout.relative_to(runtime_root), "COMMAND_STDOUT"),
            store.bind_artifact(trace_stderr.relative_to(runtime_root), "COMMAND_STDERR"),
        )
    )
    report = MachineEvidenceReport(
        schema_version="1.0.0",
        check_id="real_media_chain",
        status=status,
        kind=EvidenceKind.COMMAND_REPORT,
        level=EvidenceLevel.REAL_MEDIA,
        covered_items=("real_media_chain",),
        summary="真实媒体链原始证据",
        producer="测试专用",
        started_at=_NOW,
        finished_at=_LATER,
        artifacts=tuple(artifacts),
        details=RealMediaDetails(
            type="REAL_MEDIA",
            trace=CommandTrace(
                command=("python", "-m", "video_demo.evaluation.media_runner"),
                exit_code=0 if status == GateStatus.PASS else 1,
                stdout_sha256=_digest(trace_stdout.read_bytes()),
                stderr_sha256=_digest(trace_stderr.read_bytes()),
            ),
            ffmpeg_version=raw.ffmpeg_version,
            ffprobe_version=raw.ffprobe_version,
            raw_report_sha256=_digest(raw_path.read_bytes()),
            implementation_sha256=raw.implementation_sha256,
        ),
    )
    monkeypatch.setattr(
        "video_demo.evaluation.gate._current_real_media_implementation_sha256",
        lambda _workspace: raw.implementation_sha256,
        raising=False,
    )
    report_path = runtime_root / "eval/reports/run-1/real-media.json"
    store.write_json(report_path.relative_to(runtime_root), report)
    (report_path.parent / ".real-media.commit.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "evaluation_run_id": "run-1",
                "authority_sha256": _digest(report_path.read_bytes()),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    return workspace, report_path


def test_real_media_raw_artifacts_form_a_verified_pass_only_after_revalidation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, report_path = _write_real_media_pass_report(tmp_path, monkeypatch)
    check = build_verified_gate_check(
        "real_media_chain", report_path, workspace_root=workspace
    )

    assert check.status == GateStatus.PASS


def test_real_media_raw_failure_can_only_derive_a_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, report_path = _write_real_media_pass_report(
        tmp_path, monkeypatch, status=GateStatus.FAIL
    )

    check = build_verified_gate_check(
        "real_media_chain", report_path, workspace_root=workspace
    )

    assert check.status == GateStatus.FAIL


@pytest.mark.parametrize("status", (GateStatus.PASS, GateStatus.FAIL))
@pytest.mark.parametrize(
    ("role", "cross_run"),
    (("QUALITY_DETAIL", False), ("ANNOTATION", True)),
)
def test_real_media_pass_and_failure_reject_unowned_artifact_roles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    status: GateStatus,
    role: str,
    cross_run: bool,
) -> None:
    workspace, report_path = _write_real_media_pass_report(
        tmp_path, monkeypatch, status=status
    )
    runtime = workspace / ".codex/video-rag-demo"
    report = load_machine_evidence(report_path, workspace_root=workspace)
    target_run = "run-2" if cross_run else "run-1"
    extra = EvidenceStore(workspace, runtime).write_artifact(
        Path(f"eval/reports/{target_run}/extra.json"), role, b"{}"
    )
    changed = report.model_copy(update={"artifacts": (*report.artifacts, extra)})
    report_path.write_text(changed.model_dump_json(exclude_none=True), encoding="utf-8")

    with pytest.raises(ValueError, match="可信门禁"):
        build_verified_gate_check("real_media_chain", report_path, workspace_root=workspace)


@pytest.mark.parametrize("status", (GateStatus.PASS, GateStatus.FAIL))
def test_real_media_pass_and_failure_reject_copied_authoritative_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    status: GateStatus,
) -> None:
    workspace, report_path = _write_real_media_pass_report(
        tmp_path, monkeypatch, status=status
    )
    copied = workspace / ".codex/video-rag-demo/eval/reports/run-2/real-media.json"
    copied.parent.mkdir(parents=True)
    copied.write_bytes(report_path.read_bytes())

    with pytest.raises(ValueError, match="可信门禁"):
        build_verified_gate_check("real_media_chain", copied, workspace_root=workspace)


def test_real_media_verifier_rejects_detail_artifact_and_command_output_mismatches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, report_path = _write_real_media_pass_report(tmp_path, monkeypatch)
    report = load_machine_evidence(report_path, workspace_root=workspace)
    artifacts = verify_machine_artifacts(report, workspace_root=workspace)
    details = report.details
    assert isinstance(details, RealMediaDetails)

    with pytest.raises(ValueError, match="detail"):
        gate_module._verify_real_media(
            details.model_copy(update={"ffmpeg_version": "forged"}),
            artifacts,
            workspace,
        )
    with pytest.raises(ValueError, match="精确绑定"):
        gate_module._verify_real_media(
            details,
            {**artifacts, "INPUT_MEDIA": ()},
            workspace,
        )
    duplicated = artifacts["OUTPUT_MEDIA"][0]
    with pytest.raises(ValueError, match="重复绑定"):
        gate_module._verify_real_media(
            details,
            {**artifacts, "OUTPUT_MEDIA": (*artifacts["OUTPUT_MEDIA"], duplicated)},
            workspace,
        )
    first_stdout = artifacts["COMMAND_STDOUT"][0]
    with pytest.raises(ValueError, match="唯一绑定"):
        gate_module._verify_real_media(
            details,
            {
                **artifacts,
                "COMMAND_STDOUT": (
                    first_stdout,
                    *artifacts["COMMAND_STDOUT"],
                ),
            },
            workspace,
        )
    raw = RealMediaRawReport.model_validate_json(
        artifacts["AUDIT_REPORT"][0].snapshot.content or b""
    )
    first_command = raw.samples[0].commands[0]
    second_command = raw.samples[0].commands[1]
    reused_output = raw.model_copy(
        update={
            "samples": (
                raw.samples[0].model_copy(
                    update={
                        "commands": (
                            first_command,
                            second_command.model_copy(
                                update={
                                    "stdout_relative_path": first_command.stdout_relative_path,
                                    "stdout_sha256": first_command.stdout_sha256,
                                }
                            ),
                            *raw.samples[0].commands[2:],
                        )
                    }
                ),
                *raw.samples[1:],
            )
        }
    )
    with pytest.raises(ValueError, match="不得跨命令复用"):
        gate_module._verify_real_media_command_outputs(
            reused_output,
            details,
            artifacts["AUDIT_REPORT"][0],
            artifacts,
        )


def test_real_media_verifier_rejects_top_level_trace_exit_code_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, report_path = _write_real_media_pass_report(tmp_path, monkeypatch)
    report = load_machine_evidence(report_path, workspace_root=workspace)
    artifacts = verify_machine_artifacts(report, workspace_root=workspace)
    details = report.details
    assert isinstance(details, RealMediaDetails)

    with pytest.raises(ValueError, match="顶层 trace"):
        gate_module._verify_real_media(
            details.model_copy(
                update={"trace": details.trace.model_copy(update={"exit_code": 1})}
            ),
            artifacts,
            workspace,
        )


@pytest.mark.parametrize(
    ("format", "content"),
    (
        ("MP4", b"\x00\x00\x00\x00ftyp"),
        ("WAV", b"RIFF\xff\xff\xff\xffWAVE"),
        ("JPEG", b"\xff\xd8\xff\xd9"),
    ),
)
def test_real_media_verifier_rejects_structurally_malformed_magic(
    tmp_path: Path,
    format: str,
    content: bytes,
) -> None:
    path = tmp_path / f"malformed.{format.lower()}"
    path.write_bytes(content)
    media_file = RealMediaFile(
        role={"MP4": "SOURCE", "WAV": "AUDIO", "JPEG": "KEYFRAME"}[format],
        format=format,
        relative_path=f".codex/video-rag-demo/eval/generated/run-1/normal_audio/{path.name}",
        sha256="a" * 64,
        size_bytes=len(content),
    )

    with pytest.raises(ValueError, match="magic"):
        gate_module._verify_real_media_magic(
            gate_module._snapshot_real_media_file(path, max_bytes=1024), media_file
        )


def test_real_media_verifier_accepts_extended_size_iso_bmff(tmp_path: Path) -> None:
    path = tmp_path / "extended.mp4"
    path.write_bytes(b"\x00\x00\x00\x01ftyp\x00\x00\x00\x00\x00\x00\x00\x18isomiso2")
    media_file = RealMediaFile(
        role="SOURCE",
        format="MP4",
        relative_path=".codex/video-rag-demo/eval/generated/run-1/normal_audio/extended.mp4",
        sha256="a" * 64,
        size_bytes=path.stat().st_size,
    )

    gate_module._verify_real_media_magic(
        gate_module._snapshot_real_media_file(path, max_bytes=1024), media_file
    )


@pytest.mark.parametrize(
    "content",
    (
        _wave_bytes((b"fmt ", _PCM_FMT), (b"JUNK", b"skip"), (b"data", b"")),
        _wave_bytes((b"fmt ", _PCM_FMT), (b"LIST", b"odd"), (b"data", b"")),
        _wave_bytes((b"fmt ", _EXTENSIBLE_FMT), (b"fact", b"\x00\x00\x00\x00"), (b"data", b"")),
    ),
)
def test_real_media_verifier_accepts_legal_wave_chunks_between_fmt_and_data(
    tmp_path: Path,
    content: bytes,
) -> None:
    path = tmp_path / "valid.wav"
    path.write_bytes(content)
    media_file = RealMediaFile(
        role="AUDIO",
        format="WAV",
        relative_path=".codex/video-rag-demo/eval/generated/run-1/normal_audio/valid.wav",
        sha256="a" * 64,
        size_bytes=len(content),
    )

    gate_module._verify_real_media_magic(
        gate_module._snapshot_real_media_file(path, max_bytes=1024), media_file
    )


@pytest.mark.parametrize(
    "content",
    (
        _wave_bytes((b"fmt ", _PCM_FMT), (b"JUNK", b"skip")),
        _wave_bytes((b"data", b"")),
        b"RIFF\x14\x00\x00\x00WAVEfmt \x10\x00\x00\x00" + _PCM_FMT + b"data\xff\xff\xff\x7f",
    ),
)
def test_real_media_verifier_rejects_invalid_wave_chunk_layout(
    tmp_path: Path,
    content: bytes,
) -> None:
    path = tmp_path / "invalid.wav"
    path.write_bytes(content)
    media_file = RealMediaFile(
        role="AUDIO",
        format="WAV",
        relative_path=".codex/video-rag-demo/eval/generated/run-1/normal_audio/invalid.wav",
        sha256="a" * 64,
        size_bytes=len(content),
    )

    with pytest.raises(ValueError, match="magic"):
        gate_module._verify_real_media_magic(
            gate_module._snapshot_real_media_file(path, max_bytes=1024), media_file
        )


def test_real_media_verifier_rejects_extra_command_output_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, report_path = _write_real_media_pass_report(tmp_path, monkeypatch)
    report = load_machine_evidence(report_path, workspace_root=workspace)
    artifacts = verify_machine_artifacts(report, workspace_root=workspace)
    details = report.details
    assert isinstance(details, RealMediaDetails)
    extra_path = workspace / ".codex/video-rag-demo/eval/reports/run-1/extra.stdout.txt"
    extra_path.write_bytes(b"extra\n")
    extra = TraceArtifact(
        role="COMMAND_STDOUT",
        relative_path=extra_path.relative_to(workspace).as_posix(),
        sha256=_digest(b"extra\n"),
    )
    extra_snapshot = evidence_module._read_file_snapshot(
        extra_path, max_bytes=64 * 1024 * 1024, capture_content=True
    )
    extra_artifact = evidence_module.VerifiedArtifact(extra, extra_snapshot)

    with pytest.raises(ValueError, match="精确绑定"):
        gate_module._verify_real_media(
            details,
            {**artifacts, "COMMAND_STDOUT": (*artifacts["COMMAND_STDOUT"], extra_artifact)},
            workspace,
        )


def test_real_media_raw_rejects_command_output_from_another_run() -> None:
    sample = _real_media_sample("normal_audio")
    foreign = sample.commands[0].model_copy(
        update={
            "stdout_relative_path": ".codex/video-rag-demo/eval/reports/run-2/foreign.stdout.txt"
        }
    )
    sample = sample.model_copy(update={"commands": (foreign, *sample.commands[1:])})

    with pytest.raises(ValidationError, match="命令输出必须绑定"):
        _real_media_raw_report(
            samples=(
                sample,
                _real_media_sample("no_audio"),
                _real_media_sample("rotation"),
                _real_media_sample("vfr"),
            )
        )


def test_real_media_raw_rejects_failed_sample_with_incoherent_command_facts() -> None:
    failed = _real_media_sample(
        "normal_audio",
        execution_status="FAILED",
        failure_code=ErrorCode.VIDEO_PROCESS_FAILED,
    )
    all_zero = failed.model_copy(
        update={
            "commands": tuple(
                command.model_copy(update={"exit_code": 0})
                for command in failed.commands
            )
        }
    )
    not_started = tuple(
        _real_media_sample(case_id).model_copy(
            update={
                "execution_status": "NOT_STARTED",
                "files": (),
                "commands": (),
                "duration_ms": None,
                "has_audio": None,
                "rotation_degrees": None,
                "is_variable_frame_rate": None,
                "warnings": (),
                "opencv_decoded_frame_count": None,
                "scene_count": None,
                "selected_keyframe_count": None,
            }
        )
        for case_id in ("no_audio", "rotation", "vfr")
    )

    with pytest.raises(ValidationError, match="失败媒体样本命令"):
        _real_media_raw_report(
            status=GateStatus.FAIL,
            failure_code=ErrorCode.VIDEO_PROCESS_FAILED,
            samples=(all_zero, *not_started),
        )


def test_real_media_setup_failure_is_typed_and_keeps_all_cases_not_started() -> None:
    from video_demo.evaluation.evidence import SetupMediaCommand

    report = RealMediaRawReport(
        schema_version="1.0.0",
        status=GateStatus.FAIL,
        trace_exit_code=1,
        evaluation_run_id="run-1",
        ffmpeg_version=None,
        ffprobe_version=None,
        implementation_sha256="d" * 64,
        setup_commands=(
            SetupMediaCommand(
                phase="ffmpeg_version",
                executable="ffmpeg",
                exit_code=1,
                stdout_relative_path=".codex/video-rag-demo/eval/reports/run-1/ffmpeg.stdout.txt",
                stderr_relative_path=".codex/video-rag-demo/eval/reports/run-1/ffmpeg.stderr.txt",
                stdout_sha256="a" * 64,
                stderr_sha256="b" * 64,
            ),
        ),
        samples=tuple(
            RealMediaSample(case_id=case_id, execution_status="NOT_STARTED")
            for case_id in ("normal_audio", "no_audio", "rotation", "vfr")
        ),
        failure_code=ErrorCode.VIDEO_BINARY_PROBE_FAILED,
    )

    assert report.setup_commands[0].phase == "ffmpeg_version"


def test_real_media_snapshot_keeps_digest_and_magic_on_one_open_fd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "source.mp4"
    original = b"\x00\x00\x00\x18ftypisom\x00\x00\x00\x00isomiso2"
    replacement = tmp_path / "replacement.mp4"
    path.write_bytes(original)
    replacement.write_bytes(b"not-an-mp4")
    initial = evidence_module._read_file_snapshot(
        path, max_bytes=1024, capture_content=False
    )
    real_read = os.read
    replaced = False

    def replace_after_first_read(descriptor: int, size: int) -> bytes:
        nonlocal replaced
        chunk = real_read(descriptor, size)
        if not replaced:
            replaced = True
            os.replace(replacement, path)
            os.utime(path, ns=(initial.identity.mtime_ns, initial.identity.mtime_ns))
        return chunk

    monkeypatch.setattr(gate_module.os, "read", replace_after_first_read)

    with pytest.raises(ValueError, match="读取期间发生变化"):
        gate_module._snapshot_real_media_file(path, max_bytes=1024)


@pytest.mark.parametrize(
    ("format", "content"),
    (
        ("MP4", b"not-an-mp4"),
        ("WAV", b"not-a-wave"),
        ("JPEG", b"not-a-jpeg"),
    ),
)
def test_real_media_verifier_rejects_invalid_magic_without_reading_entire_media(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    format: str,
    content: bytes,
) -> None:
    path = tmp_path / f"invalid.{format.lower()}"
    path.write_bytes(content)
    media_file = RealMediaFile(
        role={"MP4": "SOURCE", "WAV": "AUDIO", "JPEG": "KEYFRAME"}[format],
        format=format,
        relative_path=f".codex/video-rag-demo/eval/generated/run-1/normal_audio/{path.name}",
        sha256="a" * 64,
        size_bytes=len(content),
    )

    monkeypatch.setattr(Path, "read_bytes", lambda _path: (_ for _ in ()).throw(AssertionError))
    with pytest.raises(ValueError, match="magic"):
        gate_module._verify_real_media_magic(
            gate_module._snapshot_real_media_file(path, max_bytes=1024), media_file
        )


def test_real_media_snapshot_ignores_atime_change_during_read(
    tmp_path: Path,
) -> None:
    path = tmp_path / "invalid.wav"
    content = b"not-a-wave"
    path.write_bytes(content)
    metadata = path.stat()
    os.utime(path, ns=(metadata.st_atime_ns - 1_000_000_000, metadata.st_mtime_ns))
    media_file = RealMediaFile(
        role="AUDIO",
        format="WAV",
        relative_path=".codex/video-rag-demo/eval/generated/run-1/normal_audio/invalid.wav",
        sha256=_digest(content),
        size_bytes=len(content),
    )

    with pytest.raises(ValueError, match="magic"):
        gate_module._verify_real_media_magic(
            gate_module._snapshot_real_media_file(path, max_bytes=1024), media_file
        )


def _static_report(
    store: EvidenceStore,
    runtime_root: Path,
    *,
    producer: str = "任意调用方",
    status: GateStatus = GateStatus.PASS,
    violation_count: int = 0,
) -> MachineEvidenceReport:
    stdout = _write(runtime_root, "eval/audit/stdout.txt", b"audit completed\n")
    stderr = _write(runtime_root, "eval/audit/stderr.txt", b"")
    raw = _write(runtime_root, "eval/audit/result.json", b'{"violations":[]}')
    return MachineEvidenceReport(
        schema_version="1.0.0",
        check_id="no_indexing",
        status=status,
        kind=EvidenceKind.STATIC_AUDIT,
        level=EvidenceLevel.STATIC,
        covered_items=("no_indexing",),
        summary="无索引审计完成",
        producer=producer,
        started_at=_NOW,
        finished_at=_LATER,
        artifacts=(
            store.bind_artifact(
                stdout.relative_to(runtime_root), "COMMAND_STDOUT"
            ),
            store.bind_artifact(
                stderr.relative_to(runtime_root), "COMMAND_STDERR"
            ),
            store.bind_artifact(raw.relative_to(runtime_root), "AUDIT_REPORT"),
        ),
        details=StaticAuditDetails(
            type="STATIC_AUDIT",
            trace=CommandTrace(
                command=("python", "-m", "video_demo.evaluation.no_indexing"),
                exit_code=0,
                stdout_sha256=sha256_file(stdout, max_bytes=64 * 1024 * 1024),
                stderr_sha256=sha256_file(stderr, max_bytes=64 * 1024 * 1024),
            ),
            audited_paths=("src/video_demo",),
            violation_count=violation_count,
        ),
    )


def test_report_run_writer_rejects_artifact_after_close(tmp_path: Path) -> None:
    _workspace, _runtime_root, store = _roots(tmp_path)
    writer = store.open_exclusive_report_run("run-1")
    writer.close()

    with pytest.raises(ValueError):
        writer.write_artifact("late.stdout.txt", "COMMAND_STDOUT", b"")


@pytest.mark.parametrize(
    "evaluation_run_id",
    (
        "../live-authority/run-replay",
        "run/escape",
        "..",
        "run.with.dot",
    ),
)
def test_report_run_open_rejects_non_stable_run_id(
    tmp_path: Path,
    evaluation_run_id: str,
) -> None:
    _workspace, runtime_root, store = _roots(tmp_path)
    (runtime_root / "eval/live-authority").mkdir(parents=True)

    with pytest.raises(ValueError):
        store.open_exclusive_report_run(evaluation_run_id)

    assert not (runtime_root / "eval/live-authority/run-replay").exists()


def test_report_run_writer_rejects_json_after_close(tmp_path: Path) -> None:
    _workspace, runtime_root, store = _roots(tmp_path)
    report = _static_report(store, runtime_root)
    writer = store.open_exclusive_report_run("run-1")
    writer.close()

    with pytest.raises(ValueError):
        writer.write_json(report)


def test_report_run_writer_rejects_later_writes_after_first_failure(
    tmp_path: Path,
) -> None:
    _workspace, runtime_root, store = _roots(tmp_path)
    writer = store.open_exclusive_report_run("run-1")
    try:
        with pytest.raises(ValueError):
            writer.write_artifact("invalid.json", "AUDIT_REPORT", b"")

        with pytest.raises(ValueError):
            writer.write_artifact("later.stdout.txt", "COMMAND_STDOUT", b"")

        assert not (runtime_root / "eval/reports/run-1/later.stdout.txt").exists()
    finally:
        writer.close()


@pytest.mark.parametrize("write_kind", ("artifact", "json"))
def test_closed_report_run_writer_cannot_write_through_reused_descriptor(
    tmp_path: Path,
    write_kind: str,
) -> None:
    _workspace, runtime_root, store = _roots(tmp_path)
    report = _static_report(store, runtime_root)
    stale_writer = store.open_exclusive_report_run("run-1")
    stale_descriptor = stale_writer._descriptor
    stale_writer.close()
    active_writer = store.open_exclusive_report_run("run-2")
    try:
        assert active_writer._descriptor == stale_descriptor
        run_2 = runtime_root / "eval/reports/run-2"
        before = {path.name for path in run_2.iterdir()}

        with pytest.raises(ValueError):
            if write_kind == "artifact":
                stale_writer.write_artifact(
                    "stale.stdout.txt", "COMMAND_STDOUT", b""
                )
            else:
                stale_writer.write_json(report)

        assert {path.name for path in run_2.iterdir()} == before
    finally:
        active_writer.close()


def test_report_run_open_fails_if_incomplete_marker_cannot_be_created(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _workspace, runtime_root, store = _roots(tmp_path)
    real_open = os.open

    def fail_marker_open(path: Any, *args: Any, **kwargs: Any) -> int:
        if path == ".real-media.incomplete":
            raise OSError("secret-marker-create")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(evidence_module.os, "open", fail_marker_open)

    with pytest.raises(ValueError) as captured:
        store.open_exclusive_report_run("run-1")

    assert "secret-marker-create" not in str(captured.value)
    assert captured.value.__cause__ is None
    run = runtime_root / "eval/reports/run-1"
    assert run.is_dir()
    assert not (run / ".real-media.incomplete").exists()


def test_report_run_marker_exists_until_authority_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _workspace, runtime_root, store = _roots(tmp_path)
    report = _static_report(store, runtime_root)
    real_open = os.open
    opened_leaves: list[str] = []

    def recording_open(path: Any, *args: Any, **kwargs: Any) -> int:
        if isinstance(path, str):
            opened_leaves.append(path)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(evidence_module.os, "open", recording_open)
    monkeypatch.setattr(
        evidence_module,
        "_build_verified_gate_check",
        lambda *_args, **_kwargs: object(),
    )
    writer = store.open_exclusive_report_run("run-1")
    marker = runtime_root / "eval/reports/run-1/.real-media.incomplete"
    try:
        assert ".real-media.incomplete" in opened_leaves
        assert marker.is_file()

        writer.write_json(report)

        assert not marker.exists()
    finally:
        writer.close()


def test_real_media_builder_validates_report_path_before_commit_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _workspace, _runtime_root, _store = _roots(tmp_path)
    outside_report = tmp_path.parent / "untrusted-report" / "real-media.json"
    monkeypatch.setattr(
        evidence_module,
        "_load_real_media_commit_snapshot",
        lambda *_args: pytest.fail("不得探测可信工作区外的 commit"),
    )

    with pytest.raises(ValueError, match="可信门禁"):
        build_verified_gate_check(
            "real_media_chain",
            outside_report,
            workspace_root=tmp_path,
        )


def test_atomic_write_interruption_leaves_no_authoritative_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, runtime_root, store = _roots(tmp_path)
    report = _static_report(store, runtime_root)
    target = runtime_root / "eval/reports/no-indexing.json"

    monkeypatch.setattr(
        evidence_module.os,
        "replace",
        lambda _source, _target: (_ for _ in ()).throw(OSError("secret-body")),
    )

    with pytest.raises(ValueError) as captured:
        store.write_json(Path("eval/reports/no-indexing.json"), report)

    assert not target.exists()
    assert "secret-body" not in str(captured.value)
    assert captured.value.__cause__ is None
    parts = tuple(target.parent.glob("*.part"))
    assert parts
    for part in parts:
        with pytest.raises(ValueError):
            load_machine_evidence(part, workspace_root=workspace)


def test_atomic_replacement_failure_removes_stale_authoritative_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, runtime_root, store = _roots(tmp_path)
    report = _static_report(store, runtime_root)
    target = runtime_root / "eval/reports/no-indexing.json"
    store.write_json(target.relative_to(runtime_root), report)
    assert load_machine_evidence(target, workspace_root=workspace) == report

    monkeypatch.setattr(
        evidence_module.os,
        "replace",
        lambda _source, _target: (_ for _ in ()).throw(OSError("interrupted")),
    )
    with pytest.raises(ValueError):
        store.write_json(target.relative_to(runtime_root), report)

    assert not target.exists()


def test_real_media_raw_report_requires_exact_case_set_and_consistent_status() -> None:
    valid = _real_media_raw_report()

    assert valid.status == GateStatus.PASS
    assert tuple(sample.case_id for sample in valid.samples) == (
        "normal_audio",
        "no_audio",
        "rotation",
        "vfr",
    )

    duplicate = tuple(
        _real_media_sample(case_id)
        for case_id in ("normal_audio", "normal_audio", "rotation", "vfr")
    )
    with pytest.raises(ValidationError):
        _real_media_raw_report(samples=duplicate)
    with pytest.raises(ValidationError):
        _real_media_raw_report(
            samples=(
                _real_media_sample("normal_audio", execution_status="FAILED"),
                _real_media_sample("no_audio"),
                _real_media_sample("rotation"),
                _real_media_sample("vfr"),
            )
        )
    with pytest.raises(ValidationError):
        _real_media_raw_report(
            status=GateStatus.FAIL,
            samples=tuple(
                _real_media_sample(case_id, execution_status="FAILED")
                for case_id in ("normal_audio", "no_audio", "rotation", "vfr")
            ),
        )


def test_real_media_raw_schema_rejects_unsafe_paths_duplicate_phase_and_failure_disguise() -> None:
    with pytest.raises(ValidationError):
        RealMediaFile(
            relative_path="/private/media.mp4", sha256="a" * 64, size_bytes=1
        )
    with pytest.raises(ValidationError):
        RealMediaFile(
            relative_path=".codex/video-rag-demo/../media.mp4",
            sha256="a" * 64,
            size_bytes=1,
        )
    with pytest.raises(ValidationError):
        RealMediaSample(
            case_id="normal_audio",
                execution_status="SUCCESS",
            files=(),
            commands=(
                RealMediaCommand(
                    phase="probe",
                    executable="ffprobe",
                    arguments=("-version",),
                    input_relative_paths=(),
                    output_relative_paths=(),
                    exit_code=0,
                    stdout_relative_path=".codex/video-rag-demo/eval/raw/stdout.txt",
                    stderr_relative_path=".codex/video-rag-demo/eval/raw/stderr.txt",
                    stdout_sha256="a" * 64,
                    stderr_sha256="b" * 64,
                ),
                RealMediaCommand(
                    phase="probe",
                    executable="ffprobe",
                    arguments=("-version",),
                    input_relative_paths=(),
                    output_relative_paths=(),
                    exit_code=0,
                    stdout_relative_path=".codex/video-rag-demo/eval/raw/stdout.txt",
                    stderr_relative_path=".codex/video-rag-demo/eval/raw/stderr.txt",
                    stdout_sha256="a" * 64,
                    stderr_sha256="b" * 64,
                ),
            ),
        )
    with pytest.raises(ValidationError):
        _real_media_raw_report(
            status=GateStatus.FAIL,
            failure_code=ErrorCode.VIDEO_PROCESS_FAILED,
        )


def test_real_media_fail_requires_stable_failure_code_and_non_successful_fact() -> None:
    not_started = tuple(
        _real_media_sample(case_id).model_copy(
            update={
                "execution_status": "NOT_STARTED",
                "files": (),
                "commands": (),
                "duration_ms": None,
                "has_audio": None,
                "rotation_degrees": None,
                "is_variable_frame_rate": None,
                "warnings": (),
                "opencv_decoded_frame_count": None,
                "scene_count": None,
                "selected_keyframe_count": None,
            }
        )
        for case_id in ("no_audio", "rotation", "vfr")
    )
    report = _real_media_raw_report(
        status=GateStatus.FAIL,
        failure_code=ErrorCode.VIDEO_PROCESS_FAILED,
        samples=(
            _real_media_sample(
                "normal_audio",
                execution_status="FAILED",
                failure_code=ErrorCode.VIDEO_PROCESS_FAILED,
            ),
            *not_started,
        ),
    )

    assert report.failure_code == ErrorCode.VIDEO_PROCESS_FAILED
    with pytest.raises(ValidationError):
        _real_media_raw_report(status=GateStatus.FAIL, samples=report.samples)


def test_real_media_preflight_requires_exact_ordered_issues_and_keeps_legacy_shape() -> None:
    raw = PreflightRawReport(
        schema_version="1.0.0",
        check_id="real_media_chain",
        reason_code="REAL_MEDIA_CHAIN_UNAVAILABLE",
        execution_started=False,
        issues=(
            PreflightIssue(code=ErrorCode.VIDEO_FFMPEG_UNAVAILABLE),
            PreflightIssue(code=ErrorCode.VIDEO_FFPROBE_UNAVAILABLE),
            PreflightIssue(code=ErrorCode.VISUAL_DEPENDENCY_UNAVAILABLE),
        ),
        implementation_sha256="a" * 64,
        evaluation_run_id="run-1",
    )
    live = PreflightRawReport(
        schema_version="1.0.0",
        check_id="qwen_live",
        reason_code="QWEN_CREDENTIALS_UNAVAILABLE",
        execution_started=False,
        issues=(
            PreflightIssue(code=ErrorCode.QWEN_ENDPOINT_UNAVAILABLE),
            PreflightIssue(code=ErrorCode.QWEN_API_KEY_UNAVAILABLE),
            PreflightIssue(code=ErrorCode.QWEN_MODEL_ID_UNAVAILABLE),
            PreflightIssue(code=ErrorCode.LIVE_AUTHORIZED_CLIP_UNAVAILABLE),
        ),
        implementation_sha256="b" * 64,
        evaluation_run_id="run-live",
    )

    assert raw.issues[0].code == ErrorCode.VIDEO_FFMPEG_UNAVAILABLE
    assert tuple(issue.code for issue in live.issues or ()) == (
        ErrorCode.QWEN_ENDPOINT_UNAVAILABLE,
        ErrorCode.QWEN_API_KEY_UNAVAILABLE,
        ErrorCode.QWEN_MODEL_ID_UNAVAILABLE,
        ErrorCode.LIVE_AUTHORIZED_CLIP_UNAVAILABLE,
    )
    for issues in (
        (),
        (
            PreflightIssue(code=ErrorCode.VIDEO_FFPROBE_UNAVAILABLE),
            PreflightIssue(code=ErrorCode.VIDEO_FFMPEG_UNAVAILABLE),
        ),
        (
            PreflightIssue(code=ErrorCode.VIDEO_FFMPEG_UNAVAILABLE),
            PreflightIssue(code=ErrorCode.VIDEO_FFMPEG_UNAVAILABLE),
        ),
    ):
        with pytest.raises(ValidationError):
            PreflightRawReport(
                schema_version="1.0.0",
                check_id="real_media_chain",
                reason_code="REAL_MEDIA_CHAIN_UNAVAILABLE",
                execution_started=False,
                issues=issues,
                implementation_sha256="a" * 64,
                evaluation_run_id="run-1",
            )


def _live_sample() -> LiveSample:
    return LiveSample(
        sample_id="sample-001",
        language="zh",
        duration_ms=1_000,
        source_media_relative_path=(
            ".codex/video-rag-demo/eval/media/sample-001.mp4"
        ),
        source_media_sha256="1" * 64,
        audio_relative_path=(
            ".codex/video-rag-demo/eval/live/run-1/sample-001/audio.wav"
        ),
        audio_sha256="2" * 64,
        keyframe_relative_path=(
            ".codex/video-rag-demo/eval/live/run-1/sample-001/keyframe.jpg"
        ),
        keyframe_sha256="3" * 64,
        clip_relative_path=(
            ".codex/video-rag-demo/eval/live/run-1/sample-001/clip.mp4"
        ),
        clip_sha256="4" * 64,
        annotation_sha256="5" * 64,
    )


def _live_inputs(sample: LiveSample) -> tuple[LiveInputArtifact, ...]:
    return (
        LiveInputArtifact(
            kind="SOURCE_MEDIA",
            sample_id=sample.sample_id,
            relative_path=sample.source_media_relative_path,
            sha256=sample.source_media_sha256,
            source_media_sha256=sample.source_media_sha256,
            size_bytes=256,
        ),
        LiveInputArtifact(
            kind="AUDIO",
            sample_id=sample.sample_id,
            relative_path=sample.audio_relative_path,
            sha256=sample.audio_sha256,
            source_media_sha256=sample.source_media_sha256,
            size_bytes=64,
        ),
        LiveInputArtifact(
            kind="KEYFRAME",
            sample_id=sample.sample_id,
            relative_path=sample.keyframe_relative_path,
            sha256=sample.keyframe_sha256,
            source_media_sha256=sample.source_media_sha256,
            size_bytes=32,
        ),
        LiveInputArtifact(
            kind="CLIP",
            sample_id=sample.sample_id,
            relative_path=sample.clip_relative_path,
            sha256=sample.clip_sha256,
            source_media_sha256=sample.source_media_sha256,
            size_bytes=128,
        ),
    )


def _language_live_sample(language: str) -> LiveSample:
    sample_id = f"sample-{language}"

    def digest(kind: str) -> str:
        return hashlib.sha256(f"{sample_id}:{kind}".encode()).hexdigest()

    return LiveSample(
        sample_id=sample_id,
        language=language,
        duration_ms=1_000,
        source_media_relative_path=(
            f".codex/video-rag-demo/eval/media/{sample_id}.mp4"
        ),
        source_media_sha256=digest("source"),
        audio_relative_path=(
            f".codex/video-rag-demo/eval/live/run-1/{sample_id}/audio.wav"
        ),
        audio_sha256=digest("audio"),
        keyframe_relative_path=(
            f".codex/video-rag-demo/eval/live/run-1/{sample_id}/keyframe.jpg"
        ),
        keyframe_sha256=digest("keyframe"),
        clip_relative_path=(
            f".codex/video-rag-demo/eval/live/run-1/{sample_id}/clip.mp4"
        ),
        clip_sha256=digest("clip"),
        annotation_sha256=digest("annotation"),
    )


def _model_fact(
    component: str,
    *,
    language: str | None = "zh",
    input_kind: str = "AUDIO",
) -> ModelExecutionFact:
    operation = {
        "baidu_ocr": "recognize",
        "qwen": "capability_probe",
        "pyannote": "diarize",
        "silero_vad": "vad",
        "faster_whisper": "transcribe",
        "whisperx": "align",
        "yamnet": "detect",
    }[component]
    return ModelExecutionFact(
        component=component,
        operation=operation,
        evaluation_run_id="run-live",
        model=ModelIdentity(
            component=component,
            provider="local" if component not in {"baidu_ocr", "qwen"} else component,
            model_id={
                "baidu_ocr": "accurate_basic",
                "qwen": "qwen3-vl-plus",
                "pyannote": "pyannote/speaker-diarization-community-1",
                "silero_vad": "silero-vad",
                "faster_whisper": "large-v3",
                "whisperx": f"whisperx-align-{language}",
                "yamnet": "yamnet",
            }[component],
            device="cpu" if component not in {"baidu_ocr", "qwen"} else None,
        ),
        sample_id="sample-001",
        language=language,
        input_kind=input_kind,
        input_sha256=("3" if input_kind == "KEYFRAME" else "4" if input_kind == "CLIP" else "2")
        * 64,
        output_sha256="6" * 64,
        request_id_sha256=(
            hashlib.sha256(
                ("123456789" if component == "baidu_ocr" else "chatcmpl_001").encode()
            ).hexdigest()
            if component in {"baidu_ocr", "qwen"}
            else None
        ),
        http_status=200 if component in {"baidu_ocr", "qwen"} else None,
        capabilities=("video_input", "strict_json_schema") if component == "qwen" else (),
    )


def _single_sample_live_payload(
    check_id: str,
    facts: tuple[ModelExecutionFact, ...],
    *,
    status: str = "PASS",
) -> dict[str, object]:
    sample = _live_sample()
    payload: dict[str, object] = {
        "schema_version": "1.0.0",
        "check_id": check_id,
        "status": status,
        "execution_started": True,
        "evaluation_run_id": "run-live",
        "sample": sample,
        "inputs": _live_inputs(sample),
        "dataset_sha256": "7" * 64,
        "authorization_sha256": "8" * 64,
        "settings_fingerprint": "9" * 64,
        "implementation_sha256": "a" * 64,
        "executions": facts,
    }
    if status == "FAIL":
        payload.update(
            failure_code=ErrorCode.DEPENDENCY_TEMPORARY_FAILURE,
            failure_component={
                "baidu_ocr_live": "baidu_ocr",
                "qwen_live": "qwen",
                "pyannote_live": "pyannote",
            }[check_id],
        )
    return payload


def test_live_input_and_sample_contracts_reject_unbound_or_unsafe_paths() -> None:
    sample = _live_sample()
    artifact = LiveInputArtifact(
        kind="AUDIO",
        sample_id=sample.sample_id,
        relative_path=sample.audio_relative_path,
        sha256=sample.audio_sha256,
        source_media_sha256=sample.source_media_sha256,
        size_bytes=64,
    )

    assert artifact.source_media_sha256 == sample.source_media_sha256
    for field, value in (
        ("audio_relative_path", "/private/tmp/audio.wav"),
        ("keyframe_relative_path", "../keyframe.jpg"),
        ("clip_relative_path", "data:video/mp4;base64,AAAA"),
    ):
        with pytest.raises(ValidationError):
            LiveSample.model_validate(
                {**sample.model_dump(mode="python"), field: value},
            )
    with pytest.raises(ValidationError):
        LiveInputArtifact.model_validate(
            {
                **artifact.model_dump(mode="python"),
                "sample_id": "sample-other",
                "unexpected": True,
            }
        )


def test_live_sample_rejects_reused_derived_path_across_kinds() -> None:
    sample = _live_sample().model_dump(mode="python")
    with pytest.raises(ValidationError):
        LiveSample.model_validate(
            {**sample, "clip_relative_path": sample["audio_relative_path"]}
        )


@pytest.mark.parametrize(
    ("report_type", "check_id", "facts"),
    (
        (BaiduLiveRawReport, "baidu_ocr_live", (_model_fact("baidu_ocr", input_kind="KEYFRAME"),)),
        (
            QwenLiveRawReport,
            "qwen_live",
            (
                _model_fact("qwen", input_kind="CLIP"),
                _model_fact("qwen", input_kind="CLIP").model_copy(
                    update={"operation": "understand_segment", "capabilities": ()}
                ),
            ),
        ),
        (PyannoteLiveRawReport, "pyannote_live", (_model_fact("pyannote"),)),
    ),
)
def test_live_raw_reports_require_started_execution_and_failure_code(
    report_type: type[object],
    check_id: str,
    facts: tuple[ModelExecutionFact, ...],
) -> None:
    sample = _live_sample()
    base = {
        "schema_version": "1.0.0",
        "check_id": check_id,
        "status": "PASS",
        "execution_started": True,
        "evaluation_run_id": "run-live",
        "sample": sample.model_dump(mode="python"),
        "inputs": [item.model_dump(mode="python") for item in _live_inputs(sample)],
        "dataset_sha256": "7" * 64,
        "authorization_sha256": "8" * 64,
        "settings_fingerprint": "9" * 64,
        "implementation_sha256": "a" * 64,
        "executions": [fact.model_dump(mode="python") for fact in facts],
    }
    report_type.model_validate(base)  # type: ignore[attr-defined]

    with pytest.raises(ValidationError):
        report_type.model_validate({**base, "execution_started": False})  # type: ignore[attr-defined]
    with pytest.raises(ValidationError):
        report_type.model_validate({**base, "status": "NOT_RUN"})  # type: ignore[attr-defined]
    with pytest.raises(ValidationError):
        report_type.model_validate(  # type: ignore[attr-defined]
            {**base, "status": "FAIL", "executions": []}
        )
    failed = report_type.model_validate(  # type: ignore[attr-defined]
        {
            **base,
            "status": "FAIL",
            "executions": [],
            "failure_code": ErrorCode.DEPENDENCY_TEMPORARY_FAILURE,
            "failure_component": facts[0].component,
        }
    )
    assert failed.status == GateStatus.FAIL


def test_qwen_raw_close_failure_requires_complete_execution_and_system_failure() -> None:
    probe = _model_fact("qwen", input_kind="CLIP")
    segment = probe.model_copy(
        update={"operation": "understand_segment", "capabilities": ()}
    )
    payload = {
        **_single_sample_live_payload(
            "qwen_live",
            (probe, segment),
            status="FAIL",
        ),
        "failure_code": ErrorCode.SYSTEM_FAILURE,
        "failure_component": "components_close",
    }

    report = QwenLiveRawReport.model_validate(payload)

    assert report.failure_component == "components_close"
    assert report.failure_code == ErrorCode.SYSTEM_FAILURE
    with pytest.raises(ValidationError):
        QwenLiveRawReport.model_validate(
            {**payload, "executions": [probe.model_dump(mode="python")]}
        )
    with pytest.raises(ValidationError):
        QwenLiveRawReport.model_validate(
            {**payload, "failure_code": ErrorCode.DEPENDENCY_TEMPORARY_FAILURE}
        )
    with pytest.raises(ValidationError):
        ModelExecutionFact.model_validate(
            {**probe.model_dump(mode="python"), "component": "components_close"}
        )
    summary_payload = probe.model_dump(mode="python")
    summary_payload.pop("output_sha256")
    with pytest.raises(ValidationError):
        evidence_module.LiveExecutionSummary.model_validate(
            {
                **summary_payload,
                "schema_version": "1.0.0",
                "component": "components_close",
                "output_item_count": 1,
            }
        )


@pytest.mark.parametrize(
    ("report_type", "check_id", "failure_component"),
    (
        (BaiduLiveRawReport, "baidu_ocr_live", "qwen"),
        (QwenLiveRawReport, "qwen_live", "pyannote"),
        (PyannoteLiveRawReport, "pyannote_live", "yamnet"),
    ),
)
def test_live_raw_failure_component_must_belong_to_check(
    report_type: type[object], check_id: str, failure_component: str
) -> None:
    sample = _live_sample()
    with pytest.raises(ValidationError):
        report_type.model_validate(  # type: ignore[attr-defined]
            {
                "schema_version": "1.0.0",
                "check_id": check_id,
                "status": "FAIL",
                "execution_started": True,
                "evaluation_run_id": "run-live",
                "sample": sample.model_dump(mode="python"),
                "inputs": [
                    item.model_dump(mode="python") for item in _live_inputs(sample)
                ],
                "dataset_sha256": "7" * 64,
                "authorization_sha256": "8" * 64,
                "settings_fingerprint": "9" * 64,
                "implementation_sha256": "a" * 64,
                "executions": [],
                "failure_code": ErrorCode.DEPENDENCY_TEMPORARY_FAILURE,
                "failure_component": failure_component,
            }
        )


def test_model_execution_fact_rejects_service_fields_on_local_component_and_body() -> None:
    local = _model_fact("faster_whisper")
    with pytest.raises(ValidationError):
        ModelExecutionFact.model_validate(
            {
                **local.model_dump(mode="python"),
                "request_id_sha256": "1" * 64,
                "http_status": 200,
            }
        )
    with pytest.raises(ValidationError):
        ModelExecutionFact.model_validate(
            {
                **local.model_dump(mode="python"),
                "capabilities": ["api_key=visible-secret"],
            }
        )
    with pytest.raises(ValidationError):
        ModelExecutionFact.model_validate(
            {
                **local.model_dump(mode="python"),
                "capabilities": ["video_input"],
            }
        )


def test_model_execution_fact_persists_only_request_id_digest() -> None:
    source_request_id = "chatcmpl_providerresponsebody"
    fact = _model_fact("qwen", input_kind="CLIP").model_dump(mode="python")
    fact["request_id_sha256"] = hashlib.sha256(
        source_request_id.encode("utf-8")
    ).hexdigest()

    validated = ModelExecutionFact.model_validate(fact)

    encoded = validated.model_dump_json()
    assert validated.request_id_sha256 == hashlib.sha256(
        source_request_id.encode("utf-8")
    ).hexdigest()
    assert source_request_id not in encoded
    with pytest.raises(ValidationError):
        ModelExecutionFact.model_validate(
            {**fact, "request_id": source_request_id}
        )


def test_model_execution_fact_requires_evaluation_run_id() -> None:
    fact = _model_fact("qwen", input_kind="CLIP").model_dump(mode="python")
    fact.pop("evaluation_run_id", None)

    with pytest.raises(ValidationError, match="evaluation_run_id"):
        ModelExecutionFact.model_validate(fact)


def test_live_execution_summary_requires_evaluation_run_id() -> None:
    fact = _model_fact("qwen", input_kind="CLIP")
    summary = {
        "schema_version": "1.0.0",
        "component": fact.component,
        "operation": fact.operation,
        "model": fact.model.model_dump(mode="python"),
        "sample_id": fact.sample_id,
        "language": fact.language,
        "input_kind": fact.input_kind,
        "input_sha256": fact.input_sha256,
        "request_id_sha256": fact.request_id_sha256,
        "http_status": fact.http_status,
        "capabilities": fact.capabilities,
        "output_item_count": 0,
    }

    with pytest.raises(ValidationError, match="evaluation_run_id"):
        evidence_module.LiveExecutionSummary.model_validate(summary)


def test_live_raw_rejects_execution_fact_from_other_run() -> None:
    fact = _model_fact("baidu_ocr", input_kind="KEYFRAME").model_copy(
        update={"evaluation_run_id": "run-foreign"}
    )

    with pytest.raises(ValidationError, match="当前评测 run"):
        BaiduLiveRawReport.model_validate(
            _single_sample_live_payload("baidu_ocr_live", (fact,))
        )


def test_live_execution_summary_repeats_only_safe_verifiable_facts() -> None:
    summary_type = evidence_module.LiveExecutionSummary
    fact = _model_fact("qwen", input_kind="CLIP")
    summary = summary_type(
        schema_version="1.0.0",
        component=fact.component,
        operation=fact.operation,
        evaluation_run_id=fact.evaluation_run_id,
        model=fact.model,
        sample_id=fact.sample_id,
        language=fact.language,
        input_kind=fact.input_kind,
        input_sha256=fact.input_sha256,
        request_id_sha256=fact.request_id_sha256,
        http_status=fact.http_status,
        capabilities=fact.capabilities,
        output_item_count=0,
    )

    assert summary.model_dump(mode="python")["request_id_sha256"] == (
        fact.request_id_sha256
    )
    assert "request_id" not in summary.model_dump(mode="python")
    with pytest.raises(ValidationError):
        summary_type.model_validate(
            {
                **summary.model_dump(mode="python"),
                "model": summary.model.model_copy(update={"model_id": "qwen-foreign"}),
            }
        )


def test_provider_response_artifact_accepts_live_execution_summary(
    tmp_path: Path,
) -> None:
    _workspace, _runtime_root, store = _roots(tmp_path)
    fact = _model_fact("baidu_ocr", input_kind="KEYFRAME")
    summary = evidence_module.LiveExecutionSummary(
        schema_version="1.0.0",
        component=fact.component,
        operation=fact.operation,
        evaluation_run_id=fact.evaluation_run_id,
        model=fact.model,
        sample_id=fact.sample_id,
        language=fact.language,
        input_kind=fact.input_kind,
        input_sha256=fact.input_sha256,
        request_id_sha256=fact.request_id_sha256,
        http_status=fact.http_status,
        capabilities=fact.capabilities,
        output_item_count=1,
    )

    artifact = store.write_artifact(
        Path("eval/live/run-live/baidu-response.json"),
        "PROVIDER_RESPONSE",
        summary.model_dump_json(exclude_none=True).encode("utf-8"),
    )

    assert artifact.sha256 == _digest(
        summary.model_dump_json(exclude_none=True).encode("utf-8")
    )


def test_legacy_live_evidence_persists_only_request_id_digest() -> None:
    source_request_id = "chatcmpl_providerresponsebody"
    request_id_sha256 = hashlib.sha256(source_request_id.encode("utf-8")).hexdigest()
    trace = CommandTrace(
        command=("pytest",),
        exit_code=0,
        stdout_sha256="1" * 64,
        stderr_sha256="2" * 64,
    )
    details = LiveServiceDetails(
        type="LIVE_SERVICE",
        trace=trace,
        service="QWEN",
        model_id="qwen3-vl-plus",
        request_id_sha256=request_id_sha256,
        input_sha256="3" * 64,
        output_sha256="4" * 64,
        http_status=200,
    )
    summary = ProviderResponseSummary(
        schema_version="1.0.0",
        service="QWEN",
        model_id="qwen3-vl-plus",
        request_id_sha256=request_id_sha256,
        http_status=200,
        input_sha256="3" * 64,
        output_sha256="4" * 64,
    )

    assert source_request_id not in details.model_dump_json()
    assert source_request_id not in summary.model_dump_json()
    with pytest.raises(ValidationError):
        LiveServiceDetails.model_validate(
            {
                **details.model_dump(mode="python"),
                "request_id": source_request_id,
            }
        )
    with pytest.raises(ValidationError):
        ProviderResponseSummary.model_validate(
            {
                **summary.model_dump(mode="python"),
                "request_id": source_request_id,
            }
        )


def test_model_execution_fact_validation_error_hides_plaintext_request_id() -> None:
    source_request_id = "chatcmpl_SECRET_REQUEST_ID"
    fact = _model_fact("qwen", input_kind="CLIP").model_dump(mode="python")

    with pytest.raises(ValidationError) as captured:
        ModelExecutionFact.model_validate(
            {**fact, "request_id": source_request_id}
        )

    assert source_request_id not in str(captured.value)


@pytest.mark.parametrize("revision", ("1" * 300 + ".1", []))
def test_model_execution_fact_revalidates_preconstructed_model_identity(
    revision: object,
) -> None:
    fact = _model_fact("faster_whisper")
    invalid_model = fact.model.model_copy(update={"revision": revision})

    with pytest.raises(ValidationError):
        ModelExecutionFact.model_validate(
            fact.model_copy(update={"model": invalid_model})
        )


@pytest.mark.parametrize(
    ("component", "field", "unsafe_value"),
    (
        ("baidu_ocr", "provider", "Baidu response says quota exceeded"),
        ("baidu_ocr", "model_id", "OCR result: 身份证号码 110101199001011234"),
        ("baidu_ocr", "request_id", "quota-exceeded-for-this-account"),
        ("qwen", "provider", "Qwen response contains generated answer"),
        ("qwen", "model_id", "The requested model cannot process this video"),
        ("qwen", "model_id", "quota-exceeded-for-this-account"),
        ("qwen", "revision", "provider-response-body-quota-exceeded"),
        ("qwen", "request_id", "This account has insufficient balance"),
        ("qwen", "request_id", "request_quota-exceeded-for-this-account"),
        ("qwen", "capabilities", ("video_input", "provider_response_body")),
    ),
)
def test_model_execution_fact_rejects_ordinary_provider_or_business_body(
    component: str,
    field: str,
    unsafe_value: object,
) -> None:
    fact = _model_fact(
        component,
        input_kind="KEYFRAME" if component == "baidu_ocr" else "CLIP",
    ).model_dump(mode="python")
    if field in {"provider", "model_id", "revision"}:
        fact["model"] = {**fact["model"], field: unsafe_value}
    else:
        fact[field] = unsafe_value

    with pytest.raises(ValidationError):
        ModelExecutionFact.model_validate(fact)


def _five_language_report_payload() -> dict[str, object]:
    languages = ("zh", "en", "ja", "ko", "es")
    samples = tuple(_language_live_sample(language) for language in languages)
    sample_by_language = {sample.language: sample for sample in samples}
    facts = (
        _model_fact("silero_vad"),
        *tuple(
            _model_fact("faster_whisper", language=language)
            for language in languages
        ),
        *tuple(_model_fact("whisperx", language=language) for language in languages),
        _model_fact("yamnet"),
    )
    facts = tuple(
        fact.model_copy(
            update={
                "sample_id": (
                    sample := sample_by_language[
                        fact.language
                        if fact.component in {"faster_whisper", "whisperx"}
                        else "zh"
                    ]
                ).sample_id,
                "input_sha256": sample.audio_sha256,
            }
        )
        for fact in facts
    )
    return {
        "schema_version": "1.0.0",
        "check_id": "five_language_models",
        "status": "PASS",
        "execution_started": True,
        "evaluation_run_id": "run-live",
        "samples": [sample.model_dump(mode="python") for sample in samples],
        "inputs": [
            item.model_dump(mode="python")
            for sample in samples
            for item in _live_inputs(sample)
        ],
        "dataset_sha256": "7" * 64,
        "authorization_sha256": "8" * 64,
        "settings_fingerprint": "9" * 64,
        "implementation_sha256": "a" * 64,
        "executions": [fact.model_dump(mode="python") for fact in facts],
    }


def test_five_language_raw_requires_exact_component_and_language_coverage() -> None:
    base = _five_language_report_payload()

    report = FiveLanguageModelsRawReport.model_validate(base)
    assert len(report.executions) == 12
    for mutated in (
        {**base, "executions": base["executions"][:-1]},
        {**base, "executions": [*base["executions"], base["executions"][1]]},
        {**base, "samples": base["samples"][:-1]},
    ):
        with pytest.raises(ValidationError):
            FiveLanguageModelsRawReport.model_validate(mutated)


@pytest.mark.parametrize(
    ("kind", "path_field", "digest_field"),
    (
        ("SOURCE_MEDIA", "source_media_relative_path", "source_media_sha256"),
        ("AUDIO", "audio_relative_path", "audio_sha256"),
        ("KEYFRAME", "keyframe_relative_path", "keyframe_sha256"),
        ("CLIP", "clip_relative_path", "clip_sha256"),
    ),
)
def test_five_language_raw_rejects_input_reused_across_samples(
    kind: str,
    path_field: str,
    digest_field: str,
) -> None:
    base = _five_language_report_payload()
    samples = [dict(item) for item in base["samples"]]  # type: ignore[union-attr]
    assert isinstance(samples[0], dict)
    assert isinstance(samples[1], dict)
    samples[1][path_field] = samples[0][path_field]
    samples[1][digest_field] = samples[0][digest_field]
    inputs = [dict(item) for item in base["inputs"]]  # type: ignore[union-attr]
    for item in inputs:
        if not isinstance(item, dict) or item["sample_id"] != samples[1]["sample_id"]:
            continue
        if kind == "SOURCE_MEDIA":
            item["source_media_sha256"] = samples[0][digest_field]
        if item["kind"] == kind:
            item["relative_path"] = samples[0][path_field]
            item["sha256"] = samples[0][digest_field]
    executions = [dict(item) for item in base["executions"]]  # type: ignore[union-attr]
    if kind == "AUDIO":
        for fact in executions:
            if fact["sample_id"] == samples[1]["sample_id"]:
                fact["input_sha256"] = samples[0][digest_field]

    with pytest.raises(ValidationError):
        FiveLanguageModelsRawReport.model_validate(
            {
                **base,
                "samples": samples,
                "inputs": inputs,
                "executions": executions,
            }
        )


@pytest.mark.parametrize(
    ("kind", "digest_field"),
    (
        ("SOURCE_MEDIA", "source_media_sha256"),
        ("AUDIO", "audio_sha256"),
        ("KEYFRAME", "keyframe_sha256"),
        ("CLIP", "clip_sha256"),
    ),
)
def test_five_language_raw_rejects_digest_reused_with_different_path(
    kind: str,
    digest_field: str,
) -> None:
    base = _five_language_report_payload()
    samples = [dict(item) for item in base["samples"]]  # type: ignore[union-attr]
    reused_digest = samples[0][digest_field]
    samples[1][digest_field] = reused_digest
    inputs = [dict(item) for item in base["inputs"]]  # type: ignore[union-attr]
    for item in inputs:
        if item["sample_id"] != samples[1]["sample_id"]:
            continue
        if kind == "SOURCE_MEDIA":
            item["source_media_sha256"] = reused_digest
        if item["kind"] == kind:
            item["sha256"] = reused_digest
    executions = [dict(item) for item in base["executions"]]  # type: ignore[union-attr]
    if kind == "AUDIO":
        for fact in executions:
            if fact["sample_id"] == samples[1]["sample_id"]:
                fact["input_sha256"] = reused_digest

    with pytest.raises(ValidationError):
        FiveLanguageModelsRawReport.model_validate(
            {
                **base,
                "samples": samples,
                "inputs": inputs,
                "executions": executions,
            }
        )


@pytest.mark.parametrize(
    ("report_type", "check_id", "foreign_fact"),
    (
        (
            BaiduLiveRawReport,
            "baidu_ocr_live",
            _model_fact("qwen", input_kind="CLIP"),
        ),
        (
            QwenLiveRawReport,
            "qwen_live",
            _model_fact("baidu_ocr", input_kind="KEYFRAME"),
        ),
        (PyannoteLiveRawReport, "pyannote_live", _model_fact("yamnet")),
    ),
)
def test_single_sample_live_fail_rejects_foreign_execution_fact(
    report_type: type[object],
    check_id: str,
    foreign_fact: ModelExecutionFact,
) -> None:
    sample = _live_sample()
    report = {
        "schema_version": "1.0.0",
        "check_id": check_id,
        "status": "FAIL",
        "execution_started": True,
        "evaluation_run_id": "run-live",
        "sample": sample.model_dump(mode="python"),
        "inputs": [item.model_dump(mode="python") for item in _live_inputs(sample)],
        "dataset_sha256": "7" * 64,
        "authorization_sha256": "8" * 64,
        "settings_fingerprint": "9" * 64,
        "implementation_sha256": "a" * 64,
        "executions": [foreign_fact.model_dump(mode="python")],
        "failure_code": ErrorCode.DEPENDENCY_TEMPORARY_FAILURE,
        "failure_component": {
            "baidu_ocr_live": "baidu_ocr",
            "qwen_live": "qwen",
            "pyannote_live": "pyannote",
        }[check_id],
    }

    with pytest.raises(ValidationError):
        report_type.model_validate(report)  # type: ignore[attr-defined]


def test_five_language_live_fail_rejects_foreign_sample_or_input_digest() -> None:
    base = _five_language_report_payload()
    failure = {
        **base,
        "status": "FAIL",
        "failure_code": ErrorCode.DEPENDENCY_TEMPORARY_FAILURE,
        "failure_component": "faster_whisper",
    }
    fact = dict(base["executions"][1])  # type: ignore[index]
    for mutated in (
        {**fact, "sample_id": "sample-foreign"},
        {**fact, "input_sha256": "f" * 64},
        {
            **_model_fact("pyannote").model_dump(mode="python"),
            "sample_id": "sample-zh",
            "input_sha256": base["samples"][0]["audio_sha256"],  # type: ignore[index]
        },
    ):
        with pytest.raises(ValidationError):
            FiveLanguageModelsRawReport.model_validate(
                {**failure, "executions": [mutated]}
            )


def test_live_raw_requires_exact_inputs_bound_to_sample() -> None:
    sample = _live_sample()
    base = {
        "schema_version": "1.0.0",
        "check_id": "baidu_ocr_live",
        "status": "PASS",
        "execution_started": True,
        "evaluation_run_id": "run-live",
        "sample": sample.model_dump(mode="python"),
        "inputs": [item.model_dump(mode="python") for item in _live_inputs(sample)],
        "dataset_sha256": "7" * 64,
        "authorization_sha256": "8" * 64,
        "settings_fingerprint": "9" * 64,
        "implementation_sha256": "a" * 64,
        "executions": [
            _model_fact("baidu_ocr", input_kind="KEYFRAME").model_dump(
                mode="python"
            )
        ],
    }
    for inputs in (
        base["inputs"][:-1],
        [
            *base["inputs"][:-1],
            {
                **base["inputs"][-1],
                "source_media_sha256": "f" * 64,
            },
        ],
    ):
        with pytest.raises(ValidationError):
            BaiduLiveRawReport.model_validate({**base, "inputs": inputs})


def test_live_raw_revalidates_preconstructed_sample_path() -> None:
    sample = _live_sample().model_copy(
        update={"audio_relative_path": ".codex/video-rag-demo/../escaped.wav"}
    )
    inputs = tuple(
        item.model_copy(update={"relative_path": sample.audio_relative_path})
        if item.kind == "AUDIO"
        else item
        for item in _live_inputs(_live_sample())
    )
    payload = _single_sample_live_payload(
        "baidu_ocr_live",
        (_model_fact("baidu_ocr", input_kind="KEYFRAME"),),
    )

    with pytest.raises(ValidationError):
        BaiduLiveRawReport.model_validate(
            {**payload, "sample": sample, "inputs": inputs}
        )


@pytest.mark.parametrize(
    ("target", "field", "value"),
    (
        ("input", "size_bytes", -1),
        ("input", "kind", "UNKNOWN_INPUT"),
        ("sample", "sample_id", "bad sample id"),
        ("execution", "component", "foreign component"),
    ),
)
def test_live_raw_revalidates_preconstructed_nested_field_constraints(
    target: str,
    field: str,
    value: object,
) -> None:
    sample = _live_sample()
    inputs = _live_inputs(sample)
    execution = _model_fact("baidu_ocr", input_kind="KEYFRAME")
    if target == "input":
        inputs = (inputs[0].model_copy(update={field: value}), *inputs[1:])
    elif target == "sample":
        sample = sample.model_copy(update={field: value})
        inputs = tuple(item.model_copy(update={field: value}) for item in inputs)
        execution = execution.model_copy(update={field: value})
    else:
        execution = execution.model_copy(update={field: value})
    payload = _single_sample_live_payload("baidu_ocr_live", (execution,))

    with pytest.raises(ValidationError):
        BaiduLiveRawReport.model_validate(
            {**payload, "sample": sample, "inputs": inputs}
        )


@pytest.mark.parametrize(
    ("report_type", "check_id", "failure_component"),
    (
        (BaiduLiveRawReport, "baidu_ocr_live", "baidu_ocr"),
        (QwenLiveRawReport, "qwen_live", "qwen"),
        (PyannoteLiveRawReport, "pyannote_live", "pyannote"),
    ),
)
def test_single_sample_live_fail_accepts_empty_partial_facts(
    report_type: type[object],
    check_id: str,
    failure_component: str,
) -> None:
    report = report_type.model_validate(  # type: ignore[attr-defined]
        _single_sample_live_payload(check_id, (), status="FAIL")
    )
    assert report.failure_component == failure_component


def test_five_language_live_fail_accepts_empty_partial_facts() -> None:
    base = _five_language_report_payload()
    report = FiveLanguageModelsRawReport.model_validate(
        {
            **base,
            "status": "FAIL",
            "executions": [],
            "failure_code": ErrorCode.DEPENDENCY_TEMPORARY_FAILURE,
            "failure_component": "silero_vad",
        }
    )
    assert report.executions == ()


def test_qwen_live_fail_accepts_only_valid_execution_prefixes() -> None:
    probe = _model_fact("qwen", input_kind="CLIP")
    segment = probe.model_copy(
        update={"operation": "understand_segment", "capabilities": ()}
    )
    for facts in ((), (probe,), (probe, segment)):
        QwenLiveRawReport.model_validate(
            _single_sample_live_payload("qwen_live", facts, status="FAIL")
        )
    with pytest.raises(ValidationError):
        QwenLiveRawReport.model_validate(
            _single_sample_live_payload(
                "qwen_live", (segment,), status="FAIL"
            )
        )


def test_qwen_live_pass_requires_same_model_identity_across_stages() -> None:
    probe = _model_fact("qwen", input_kind="CLIP")
    segment = probe.model_copy(
        update={
            "operation": "understand_segment",
            "capabilities": (),
            "model": probe.model.model_copy(update={"model_id": "qwen2.5-vl-max"}),
        }
    )

    with pytest.raises(ValidationError):
        QwenLiveRawReport.model_validate(
            _single_sample_live_payload("qwen_live", (probe, segment))
        )


def test_qwen_live_fail_rejects_segment_after_unsuccessful_capability_probe() -> None:
    probe = _model_fact("qwen", input_kind="CLIP").model_copy(
        update={"capabilities": ()}
    )
    segment = probe.model_copy(
        update={"operation": "understand_segment", "capabilities": ()}
    )

    with pytest.raises(ValidationError):
        QwenLiveRawReport.model_validate(
            _single_sample_live_payload(
                "qwen_live", (probe, segment), status="FAIL"
            )
        )


def test_qwen_live_fail_rejects_http_500_capability_probe() -> None:
    probe = _model_fact("qwen", input_kind="CLIP").model_copy(
        update={"http_status": 500}
    )

    with pytest.raises(ValidationError, match=r"能力探测|HTTP|2xx"):
        QwenLiveRawReport.model_validate(
            _single_sample_live_payload("qwen_live", (probe,), status="FAIL")
        )


def test_qwen_live_fail_rejects_capability_probe_without_required_capabilities() -> None:
    probe = _model_fact("qwen", input_kind="CLIP").model_copy(
        update={"capabilities": ()}
    )

    with pytest.raises(ValidationError, match="能力探测"):
        QwenLiveRawReport.model_validate(
            _single_sample_live_payload("qwen_live", (probe,), status="FAIL")
        )




@pytest.mark.parametrize(
    ("details_type", "details_literal"),
    (
        (BaiduLiveDetails, "BAIDU_LIVE"),
        (QwenLiveDetails, "QWEN_LIVE"),
        (PyannoteLiveDetails, "PYANNOTE_LIVE"),
        (FiveLanguageModelsDetails, "FIVE_LANGUAGE_MODELS"),
    ),
)
def test_check_specific_live_details_bind_raw_implementation_and_authorization(
    details_type: type[object], details_literal: str
) -> None:
    details = details_type.model_validate(  # type: ignore[attr-defined]
        {
            "type": details_literal,
            "trace": {
                "command": ["python", "-m", "video_demo.evaluation.live_runner"],
                "exit_code": 0,
                "stdout_sha256": "1" * 64,
                "stderr_sha256": "2" * 64,
            },
            "raw_report_sha256": "3" * 64,
            "implementation_sha256": "4" * 64,
            "settings_fingerprint": "5" * 64,
            "dataset_sha256": "6" * 64,
            "authorization_sha256": "7" * 64,
        }
    )
    assert details.raw_report_sha256 == "3" * 64


@pytest.mark.parametrize(
    ("check_id", "reason_code", "issues"),
    (
        (
            "baidu_ocr_live",
            "BAIDU_OCR_CREDENTIALS_UNAVAILABLE",
            (
                ErrorCode.BAIDU_API_KEY_UNAVAILABLE,
                ErrorCode.BAIDU_SECRET_KEY_UNAVAILABLE,
                ErrorCode.LIVE_AUTHORIZED_KEYFRAME_UNAVAILABLE,
            ),
        ),
        (
            "qwen_live",
            "QWEN_CREDENTIALS_UNAVAILABLE",
            (
                ErrorCode.QWEN_ENDPOINT_UNAVAILABLE,
                ErrorCode.QWEN_API_KEY_UNAVAILABLE,
                ErrorCode.QWEN_MODEL_ID_UNAVAILABLE,
                ErrorCode.LIVE_AUTHORIZED_CLIP_UNAVAILABLE,
            ),
        ),
        (
            "pyannote_live",
            "PYANNOTE_MODEL_UNAVAILABLE",
            (
                ErrorCode.PYANNOTE_TOKEN_UNAVAILABLE,
                ErrorCode.PYANNOTE_TERMS_UNAVAILABLE,
                ErrorCode.PYANNOTE_DEPENDENCY_UNAVAILABLE,
                ErrorCode.PYANNOTE_MODEL_UNAVAILABLE,
                ErrorCode.LIVE_AUTHORIZED_AUDIO_UNAVAILABLE,
            ),
        ),
        (
            "five_language_models",
            "FIVE_LANGUAGE_MODELS_UNAVAILABLE",
            (
                ErrorCode.SILERO_DEPENDENCY_UNAVAILABLE,
                ErrorCode.SILERO_MODEL_UNAVAILABLE,
                ErrorCode.FASTER_WHISPER_DEPENDENCY_UNAVAILABLE,
                ErrorCode.FASTER_WHISPER_MODEL_UNAVAILABLE,
                ErrorCode.WHISPERX_DEPENDENCY_UNAVAILABLE,
                ErrorCode.WHISPERX_MODEL_UNAVAILABLE,
                ErrorCode.YAMNET_DEPENDENCY_UNAVAILABLE,
                ErrorCode.YAMNET_MODEL_UNAVAILABLE,
                ErrorCode.LIVE_FIVE_LANGUAGE_AUDIO_UNAVAILABLE,
            ),
        ),
    ),
)
def test_live_preflight_requires_exact_nonempty_ordered_issues(
    check_id: str,
    reason_code: str,
    issues: tuple[ErrorCode, ...],
) -> None:
    payload = {
        "schema_version": "1.0.0",
        "check_id": check_id,
        "reason_code": reason_code,
        "execution_started": False,
        "issues": [{"code": code} for code in issues],
        "implementation_sha256": "a" * 64,
        "evaluation_run_id": "run-live",
    }
    report = PreflightRawReport.model_validate(payload)
    assert tuple(issue.code for issue in report.issues or ()) == issues
    with pytest.raises(ValidationError):
        PreflightRawReport.model_validate({**payload, "issues": []})
    with pytest.raises(ValidationError):
        PreflightRawReport.model_validate(
            {**payload, "issues": list(reversed(payload["issues"]))}
        )
    with pytest.raises(ValidationError):
        PreflightRawReport.model_validate(
            {**payload, "issues": [*payload["issues"], payload["issues"][0]]}
        )


def test_write_artifact_atomically_writes_validated_machine_payload(tmp_path: Path) -> None:
    workspace, runtime_root, store = _roots(tmp_path)
    payload = b'{"schema_version":"1.0.0","result":"safe"}'

    artifact = store.write_artifact(
        Path("eval/raw/media.json"),
        "AUDIT_REPORT",
        payload,
    )

    target = runtime_root / "eval/raw/media.json"
    assert target.read_bytes() == payload
    assert artifact.relative_path == ".codex/video-rag-demo/eval/raw/media.json"
    assert artifact.sha256 == _digest(payload)
    assert artifact.max_bytes == 64 * 1024 * 1024
    assert target.is_file()
    assert workspace in target.parents


@pytest.mark.parametrize(
    ("role", "payload"),
    (
        ("AUDIT_REPORT", b""),
        ("AUDIT_REPORT", b"\xff"),
        ("AUDIT_REPORT", b"{not-json}"),
        ("COMMAND_STDOUT", b'{"third_party":"body"}'),
    ),
)
def test_write_artifact_fails_closed_for_invalid_machine_content(
    tmp_path: Path,
    role: str,
    payload: bytes,
) -> None:
    _workspace, runtime_root, store = _roots(tmp_path)
    target = runtime_root / "eval/raw/invalid.bin"
    target.parent.mkdir(parents=True)
    target.write_bytes(b'{"stale":true}')

    with pytest.raises(ValueError, match="原子写入失败"):
        store.write_artifact(target.relative_to(runtime_root), role, payload)

    assert target.read_bytes() == b'{"stale":true}'
    assert not list(target.parent.glob("*.part"))


def test_write_artifact_rejects_media_and_cleans_partial_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _workspace, runtime_root, store = _roots(tmp_path)
    target = runtime_root / "eval/raw/interrupted.json"

    with pytest.raises(ValueError, match="原子写入失败"):
        store.write_artifact(Path("eval/raw/movie.mp4"), "INPUT_MEDIA", b"video")

    monkeypatch.setattr(
        evidence_module.os,
        "rename",
        lambda _source, _target, **_kwargs: (_ for _ in ()).throw(
            OSError("interrupted")
        ),
    )
    with pytest.raises(ValueError, match="原子写入失败"):
        store.write_artifact(
            target.relative_to(runtime_root), "AUDIT_REPORT", b'{"safe":true}'
        )
    assert not target.exists()
    assert not list(target.parent.glob("*.part"))


def test_write_artifact_parent_replacement_never_touches_external_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _workspace, runtime_root, store = _roots(tmp_path)
    parent = runtime_root / "eval/raw"
    parent.mkdir(parents=True)
    target = parent / "atomic.json"
    target.write_bytes(b'{"trusted":"old"}')
    moved = runtime_root / "eval/raw-held"
    external = tmp_path / "external-artifact-parent"
    external.mkdir()
    external_target = external / target.name
    external_target.write_bytes(b'{"external":"sentinel"}')
    real_rename = evidence_module.os.rename
    injected = False

    def replace_after_parent_swap(
        source: object,
        destination: object,
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
    ) -> None:
        nonlocal injected
        if not injected:
            injected = True
            parent.rename(moved)
            parent.symlink_to(external, target_is_directory=True)
        real_rename(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
        )

    monkeypatch.setattr(evidence_module.os, "rename", replace_after_parent_swap)

    with pytest.raises(ValueError, match="原子写入失败"):
        store.write_artifact(
            target.relative_to(runtime_root),
            "AUDIT_REPORT",
            b'{"trusted":"new"}',
        )

    assert injected
    assert external_target.read_bytes() == b'{"external":"sentinel"}'
    assert set(external.iterdir()) == {external_target}
    assert (moved / target.name).read_bytes() == b'{"trusted":"new"}'
    assert not list(moved.glob("*.part"))


def test_write_artifact_rejects_concurrent_replacement_after_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _workspace, runtime_root, store = _roots(tmp_path)
    target = runtime_root / "eval/raw/atomic.json"
    payload = b'{"writer":"trusted"}'
    replacement = b'{"writer":"concurrent"}'
    real_read = evidence_module._read_file_snapshot_at
    replaced = False

    def replace_before_final_read(
        parent_descriptor: int,
        path: Path,
        *,
        max_bytes: int,
        allow_empty: bool = False,
    ) -> Any:
        nonlocal replaced
        if not replaced:
            replaced = True
            replacement_name = ".atomic.json.concurrent.part"
            descriptor = os.open(
                replacement_name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=parent_descriptor,
            )
            try:
                assert os.write(descriptor, replacement) == len(replacement)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            os.rename(
                replacement_name,
                path.name,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
            )
        return real_read(
            parent_descriptor,
            path,
            max_bytes=max_bytes,
            allow_empty=allow_empty,
        )

    monkeypatch.setattr(
        evidence_module,
        "_read_file_snapshot_at",
        replace_before_final_read,
    )

    with pytest.raises(ValueError, match="原子写入失败"):
        store.write_artifact(
            target.relative_to(runtime_root),
            "AUDIT_REPORT",
            payload,
        )

    assert replaced
    assert target.read_bytes() == replacement
    assert not list(target.parent.glob("*.part"))


def test_write_artifact_cleanup_never_unlinks_replacement_after_stale_stat(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _workspace, runtime_root, store = _roots(tmp_path)
    target = runtime_root / "eval/raw/atomic.json"
    replacement = b'{"writer":"concurrent-cleanup"}'
    real_rename = evidence_module.os.rename
    real_unlink = evidence_module.os.unlink
    cleanup_started = False
    replacement_injected = False
    target.parent.mkdir(parents=True)
    replacement_name = ".atomic.json.cleanup-replacement.part"
    replacement_path = target.parent / replacement_name
    replacement_path.write_bytes(replacement)

    def fail_after_publish(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal cleanup_started
        cleanup_started = True
        raise ValueError("controlled post-publish failure")

    def replace_before_cleanup_unlink(
        path: object,
        *,
        dir_fd: int | None = None,
    ) -> None:
        nonlocal replacement_injected
        if cleanup_started and not replacement_injected and path == target.name:
            assert dir_fd is not None
            replacement_injected = True
            real_rename(
                replacement_name,
                target.name,
                src_dir_fd=dir_fd,
                dst_dir_fd=dir_fd,
            )
        real_unlink(path, dir_fd=dir_fd)

    monkeypatch.setattr(
        evidence_module,
        "_read_file_snapshot_at",
        fail_after_publish,
    )
    monkeypatch.setattr(evidence_module.os, "unlink", replace_before_cleanup_unlink)

    with pytest.raises(ValueError, match="原子写入失败"):
        store.write_artifact(
            target.relative_to(runtime_root),
            "AUDIT_REPORT",
            b'{"writer":"trusted"}',
        )

    if not replacement_injected:
        real_rename(replacement_path, target)
        replacement_injected = True
    assert replacement_injected
    assert target.read_bytes() == replacement
    assert not list(target.parent.glob("*.part"))


def test_write_artifact_fails_closed_without_required_fd_capability(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _workspace, runtime_root, store = _roots(tmp_path)
    target = runtime_root / "eval/raw/capability.json"
    monkeypatch.delattr(evidence_module.os, "O_NOFOLLOW")

    with pytest.raises(ValueError, match="原子写入失败"):
        store.write_artifact(
            target.relative_to(runtime_root),
            "AUDIT_REPORT",
            b'{"safe":true}',
        )

    assert not target.exists()


def test_real_media_raw_binds_every_file_to_run_and_case_with_semantic_facts() -> None:
    report = _real_media_raw_report()

    assert report.evaluation_run_id == "run-1"
    assert report.samples[0].files[0].role == "SOURCE"
    assert report.samples[0].files[0].format == "MP4"
    assert report.samples[1].warnings == ("NO_AUDIO_TRACK",)
    assert report.samples[2].rotation_degrees == 90
    assert report.samples[3].is_variable_frame_rate is True

    wrong_case_path = _real_media_sample("normal_audio").model_copy(
        update={
            "files": (
                RealMediaFile(
                    role="SOURCE",
                    format="MP4",
                    relative_path=(
                        ".codex/video-rag-demo/eval/generated/run-1/no_audio/source.mp4"
                    ),
                    sha256="a" * 64,
                    size_bytes=1,
                ),
            )
        }
    )
    with pytest.raises(ValidationError):
        _real_media_raw_report(
            samples=(
                wrong_case_path,
                _real_media_sample("no_audio"),
                _real_media_sample("rotation"),
                _real_media_sample("vfr"),
            )
        )
    duplicate = _real_media_sample("no_audio").model_copy(
        update={"files": _real_media_sample("normal_audio").files}
    )
    with pytest.raises(ValidationError):
        _real_media_raw_report(
            samples=(
                _real_media_sample("normal_audio"),
                duplicate,
                _real_media_sample("rotation"),
                _real_media_sample("vfr"),
            )
        )


def test_real_media_raw_strictly_rejects_bool_in_every_integer_fact() -> None:
    payload = _real_media_raw_report().model_dump(mode="json")
    for location in (
        ("trace_exit_code",),
        ("samples", 0, "duration_ms"),
        ("samples", 0, "opencv_decoded_frame_count"),
        ("samples", 0, "scene_count"),
        ("samples", 0, "selected_keyframe_count"),
        ("samples", 0, "commands", 0, "exit_code"),
        ("samples", 0, "files", 0, "size_bytes"),
    ):
        mutated = json.loads(json.dumps(payload))
        current: object = mutated
        for item in location[:-1]:
            current = current[item]  # type: ignore[index]
        current[location[-1]] = True  # type: ignore[index]
        with pytest.raises(ValidationError):
            RealMediaRawReport.model_validate(mutated)


@pytest.mark.parametrize(
    ("field", "invalid_values"),
    (
        ("has_audio", (0, 1, "true", "false")),
        ("is_variable_frame_rate", (0, 1, "true", "false")),
    ),
)
def test_real_media_raw_strictly_rejects_non_boolean_probe_facts(
    field: str,
    invalid_values: tuple[object, ...],
) -> None:
    payload = _real_media_raw_report().model_dump(mode="json")
    for value in invalid_values:
        mutated = json.loads(json.dumps(payload))
        mutated["samples"][0][field] = value
        with pytest.raises(ValidationError):
            RealMediaRawReport.model_validate(mutated)
        with pytest.raises(ValidationError):
            RealMediaRawReport.model_validate_json(json.dumps(mutated))


@pytest.mark.parametrize("value", (0, 1, "true", "false"))
def test_preflight_raw_strictly_rejects_non_boolean_execution_started(
    value: object,
) -> None:
    payload = {
        "schema_version": "1.0.0",
        "check_id": "qwen_live",
        "reason_code": "QWEN_CREDENTIALS_UNAVAILABLE",
        "execution_started": value,
    }

    with pytest.raises(ValidationError):
        PreflightRawReport.model_validate(payload)
    with pytest.raises(ValidationError):
        PreflightRawReport.model_validate_json(json.dumps(payload))


def test_real_media_command_requires_fixed_phase_executable_and_explicit_paths() -> None:
    report = _real_media_raw_report()
    first = report.samples[0]
    foreign_path = (
        ".codex/video-rag-demo/eval/generated/run-2/no_audio/source.mp4"
    )
    broken_variants = (
        first.model_copy(
            update={"commands": first.commands[1:]}
        ),
        first.model_copy(
            update={"commands": (*first.commands[1:], first.commands[0])}
        ),
        first.model_copy(
            update={
                "commands": (
                    first.commands[0].model_copy(update={"phase": "unknown"}),
                    *first.commands[1:],
                )
            }
        ),
        first.model_copy(
            update={
                "commands": (
                    first.commands[0].model_copy(update={"executable": "bash"}),
                    *first.commands[1:],
                )
            }
        ),
        first.model_copy(
            update={
                "commands": (
                    first.commands[0].model_copy(
                        update={
                            "arguments": (foreign_path,),
                            "output_relative_paths": (),
                        }
                    ),
                    *first.commands[1:],
                )
            }
        ),
        first.model_copy(
            update={
                "commands": (
                    first.commands[0].model_copy(
                        update={
                            "output_relative_paths": (foreign_path,),
                        }
                    ),
                    *first.commands[1:],
                )
            }
        ),
    )
    for sample in broken_variants:
        with pytest.raises(ValidationError):
            _real_media_raw_report(
                samples=(
                    sample,
                    _real_media_sample("no_audio"),
                    _real_media_sample("rotation"),
                    _real_media_sample("vfr"),
                )
            )


def test_real_media_command_rejects_embedded_foreign_runtime_path() -> None:
    report = _real_media_raw_report()
    first = report.samples[0]
    foreign_path = ".codex/video-rag-demo/eval/generated/run-2/no_audio/source.mp4"
    command = first.commands[0].model_copy(
        update={
            "arguments": (f"input={foreign_path}",),
            "output_relative_paths": (),
        }
    )
    sample = first.model_copy(update={"commands": (command, *first.commands[1:])})

    with pytest.raises(ValidationError):
        _real_media_raw_report(
            samples=(
                sample,
                _real_media_sample("no_audio"),
                _real_media_sample("rotation"),
                _real_media_sample("vfr"),
            )
        )


def test_real_media_command_rejects_output_not_bound_to_sample_file() -> None:
    report = _real_media_raw_report()
    first = report.samples[0]
    unbound_output = (
        ".codex/video-rag-demo/eval/generated/run-1/normal_audio/unbound-output.mp4"
    )
    command = first.commands[0].model_copy(
        update={"output_relative_paths": (unbound_output,)}
    )
    sample = first.model_copy(update={"commands": (command, *first.commands[1:])})

    with pytest.raises(ValidationError):
        _real_media_raw_report(
            samples=(
                sample,
                _real_media_sample("no_audio"),
                _real_media_sample("rotation"),
                _real_media_sample("vfr"),
            )
        )


def test_real_media_pass_requires_complete_execution_facts_and_allows_not_started_tail(
) -> None:
    empty = _real_media_sample("normal_audio").model_copy(
        update={"files": (), "commands": ()}
    )
    with pytest.raises(ValidationError):
        _real_media_raw_report(
            samples=(
                empty,
                _real_media_sample("no_audio"),
                _real_media_sample("rotation"),
                _real_media_sample("vfr"),
            )
        )
    failed = _real_media_sample(
        "normal_audio",
        execution_status="FAILED",
        failure_code=ErrorCode.VIDEO_PROCESS_FAILED,
    )
    not_started = tuple(
        _real_media_sample(case_id).model_copy(
            update={
                "execution_status": "NOT_STARTED",
                "files": (),
                "commands": (),
                "duration_ms": None,
                "has_audio": None,
                "rotation_degrees": None,
                "is_variable_frame_rate": None,
                "warnings": (),
                "opencv_decoded_frame_count": None,
                "scene_count": None,
                "selected_keyframe_count": None,
            }
        )
        for case_id in ("no_audio", "rotation", "vfr")
    )
    report = _real_media_raw_report(
        status=GateStatus.FAIL,
        failure_code=ErrorCode.VIDEO_PROCESS_FAILED,
        samples=(failed, *not_started),
    )

    assert report.samples[1].execution_status == "NOT_STARTED"


def test_real_media_success_requires_keyframe_file_count_to_match_selection() -> None:
    sample = _real_media_sample("normal_audio").model_copy(
        update={
            "files": (
                *_real_media_sample("normal_audio").files,
                RealMediaFile(
                    role="KEYFRAME",
                    format="JPEG",
                    relative_path=(
                        ".codex/video-rag-demo/eval/generated/run-1/normal_audio/"
                        "visual/keyframes/000002.jpg"
                    ),
                    sha256="e" * 64,
                    size_bytes=1,
                ),
            )
        }
    )

    with pytest.raises(ValidationError):
        _real_media_raw_report(
            samples=(
                sample,
                _real_media_sample("no_audio"),
                _real_media_sample("rotation"),
                _real_media_sample("vfr"),
            )
        )


@pytest.mark.parametrize("role", ("COMMAND_STDOUT", "COMMAND_STDERR"))
def test_write_artifact_allows_empty_command_output(tmp_path: Path, role: str) -> None:
    _workspace, runtime_root, store = _roots(tmp_path)

    artifact = store.write_artifact(Path(f"eval/raw/{role}.txt"), role, b"")

    assert artifact.sha256 == _digest(b"")
    assert (runtime_root / f"eval/raw/{role}.txt").read_bytes() == b""


def test_atomic_write_flushes_fsyncs_and_replaces_in_same_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, runtime_root, store = _roots(tmp_path)
    report = _static_report(store, runtime_root)
    target = runtime_root / "eval/reports/no-indexing.json"
    real_fsync = os.fsync
    real_replace = os.replace
    fsync_calls: list[int] = []
    replacements: list[tuple[Path, Path]] = []

    def recording_fsync(descriptor: int) -> None:
        fsync_calls.append(descriptor)
        real_fsync(descriptor)

    def recording_replace(source: str | Path, destination: str | Path) -> None:
        replacements.append((Path(source), Path(destination)))
        real_replace(source, destination)

    monkeypatch.setattr(evidence_module.os, "fsync", recording_fsync)
    monkeypatch.setattr(evidence_module.os, "replace", recording_replace)

    reference = store.write_json(Path("eval/reports/no-indexing.json"), report)

    assert len(fsync_calls) >= 2
    assert replacements == [(replacements[0][0], target)]
    assert replacements[0][0].parent == target.parent
    assert not replacements[0][0].exists()
    assert reference.sha256 == sha256_file(target, max_bytes=64 * 1024 * 1024)
    assert load_machine_evidence(target, workspace_root=workspace) == report


def _write_failed_static_report(
    runtime_root: Path,
    store: EvidenceStore,
) -> tuple[MachineEvidenceReport, Path]:
    report = _static_report(
        store,
        runtime_root,
        status=GateStatus.FAIL,
        violation_count=1,
    )
    report_path = runtime_root / "eval/reports/strict-authoritative.json"
    store.write_json(report_path.relative_to(runtime_root), report)
    return report, report_path


def _mutate_authoritative_report(encoded: bytes, mutation: str) -> bytes:
    text = encoded.decode("utf-8")
    if mutation == "duplicate_top_level":
        return ('{"summary":"\\u0000",' + text[1:]).encode("utf-8")
    if mutation == "duplicate_nested":
        return text.replace(
            '"violation_count":1',
            '"violation_count":"\\u0000","violation_count":1',
            1,
        ).encode("utf-8")
    if mutation == "non_standard_constant":
        return ('{"summary":NaN,' + text[1:]).encode("utf-8")
    if mutation == "depth_65":
        overwritten = "[" * 64 + '"ignored"' + "]" * 64
        return ('{"summary":' + overwritten + "," + text[1:]).encode("utf-8")
    raise AssertionError(f"未知测试变体: {mutation}")


@pytest.mark.parametrize("public_entry", ("load", "build"))
@pytest.mark.parametrize(
    "mutation",
    (
        "duplicate_top_level",
        "duplicate_nested",
        "non_standard_constant",
        "depth_65",
    ),
)
def test_authoritative_report_public_paths_use_strict_json_contract(
    tmp_path: Path,
    public_entry: str,
    mutation: str,
) -> None:
    workspace, runtime_root, store = _roots(tmp_path)
    _report, report_path = _write_failed_static_report(runtime_root, store)
    report_path.write_bytes(
        _mutate_authoritative_report(report_path.read_bytes(), mutation)
    )

    if public_entry == "load":
        operation = lambda: load_machine_evidence(  # noqa: E731
            report_path,
            workspace_root=workspace,
        )
        message = "机器证据报告非法或不可信"
    else:
        operation = lambda: build_verified_gate_check(  # noqa: E731
            "no_indexing",
            report_path,
            workspace_root=workspace,
        )
        message = "机器证据无法形成可信门禁检查"

    with pytest.raises(ValueError, match=message) as captured:
        operation()

    assert captured.value.__cause__ is None


def test_authoritative_report_normal_public_paths_remain_compatible(
    tmp_path: Path,
) -> None:
    workspace, runtime_root, store = _roots(tmp_path)
    report, report_path = _write_failed_static_report(runtime_root, store)

    assert load_machine_evidence(report_path, workspace_root=workspace) == report
    check = build_verified_gate_check(
        "no_indexing",
        report_path,
        workspace_root=workspace,
    )
    assert check.status == GateStatus.FAIL


def test_strict_json_accepts_exactly_64_levels_for_authoritative_decoder() -> None:
    nested = "[" * 63 + '"safe"' + "]" * 63
    payload = evidence_module._decode_strict_json('{"value":' + nested + "}")

    assert isinstance(payload, dict)


def test_atomic_writer_revalidates_encoded_report_with_strict_decoder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _workspace, runtime_root, store = _roots(tmp_path)
    report = _static_report(store, runtime_root)
    target = runtime_root / "eval/reports/strict-writer.json"
    real_decode = evidence_module._decode_strict_json
    decoded_values: list[str] = []

    def recording_decode(value: str) -> object:
        decoded_values.append(value)
        return real_decode(value)

    monkeypatch.setattr(evidence_module, "_decode_strict_json", recording_decode)

    store.write_json(target.relative_to(runtime_root), report)

    assert len(decoded_values) == 1
    assert json.loads(decoded_values[0])["check_id"] == "no_indexing"


def test_paths_symlinks_escape_and_secret_argv_fail_closed_without_leaks(
    tmp_path: Path,
) -> None:
    workspace, runtime_root, store = _roots(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (runtime_root / "eval").mkdir()
    (runtime_root / "eval" / "linked").symlink_to(outside, target_is_directory=True)
    secret = "must-not-leak-secret-value"

    operations: tuple[Callable[[], object], ...] = (
        lambda: store.bind_artifact(Path("../outside/value.json"), "AUDIT_REPORT"),
        lambda: store.bind_artifact(
            Path("eval/linked/value.json"), "AUDIT_REPORT"
        ),
        lambda: EvidenceStore(workspace, outside),
        lambda: load_machine_evidence(
            runtime_root / "eval/reports/unwritten.json.part",
            workspace_root=workspace,
        ),
    )
    for operation in operations:
        with pytest.raises(ValueError) as captured:
            operation()
        assert str(workspace) not in str(captured.value)
        assert captured.value.__cause__ is None

    payload = _static_report(store, runtime_root).model_dump(mode="json")
    payload["details"]["trace"]["command"] = (
        "python",
        "--api-key",
        secret,
    )
    report_path = _write(
        runtime_root,
        "eval/reports/secret.json",
        json.dumps(payload, ensure_ascii=False).encode("utf-8"),
    )
    with pytest.raises(ValueError) as captured:
        load_machine_evidence(report_path, workspace_root=workspace)
    assert secret not in str(captured.value)
    assert str(report_path) not in str(captured.value)
    assert captured.value.__cause__ is None


def test_report_rejects_duplicate_artifact_paths() -> None:
    artifact = {
        "role": "COMMAND_STDOUT",
        "relative_path": ".codex/video-rag-demo/eval/stdout.txt",
        "sha256": "a" * 64,
    }
    with pytest.raises(ValidationError, match="不得重复"):
        MachineEvidenceReport.model_validate(
            {
                "schema_version": "1.0.0",
                "check_id": "no_indexing",
                "status": "PASS",
                "kind": "STATIC_AUDIT",
                "level": "STATIC",
                "covered_items": ["no_indexing"],
                "summary": "审计完成",
                "producer": "caller",
                "started_at": _NOW,
                "finished_at": _LATER,
                "artifacts": [artifact, artifact],
                "details": {
                    "type": "STATIC_AUDIT",
                    "trace": {
                        "command": ["ruff", "check"],
                        "exit_code": 0,
                        "stdout_sha256": "a" * 64,
                        "stderr_sha256": "b" * 64,
                    },
                    "audited_paths": ["src"],
                    "violation_count": 0,
                },
            }
        )


def test_sha256_file_streams_large_media_in_one_mib_chunks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    media = tmp_path / "large.mp4"
    with media.open("wb") as stream:
        stream.seek(64 * 1024 * 1024)
        stream.write(b"media-tail")
    read_sizes: list[int] = []
    real_read = os.read

    def recording_read(descriptor: int, size: int) -> bytes:
        read_sizes.append(size)
        return real_read(descriptor, size)

    monkeypatch.setattr(
        Path,
        "read_bytes",
        lambda _path: (_ for _ in ()).throw(AssertionError("不得整体读取")),
    )
    monkeypatch.setattr(evidence_module.os, "read", recording_read)

    digest = sha256_file(media, max_bytes=65 * 1024 * 1024)

    assert len(digest) == 64
    assert read_sizes
    assert set(read_sizes) == {1024 * 1024}


def test_sha256_file_rejects_over_limit_before_opening(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    oversized = tmp_path / "oversized.bin"
    with oversized.open("wb") as stream:
        stream.seek(1024)
        stream.write(b"x")
    read_started = False

    def forbidden_read(_descriptor: int, _size: int) -> bytes:
        nonlocal read_started
        read_started = True
        raise AssertionError("超限文件不得开始读取")

    monkeypatch.setattr(evidence_module.os, "read", forbidden_read)

    with pytest.raises(ValueError):
        sha256_file(oversized, max_bytes=1024)
    assert read_started is False


def test_sha256_file_rejects_symlink_parent(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "artifact.bin").write_bytes(b"artifact")
    linked = tmp_path / "linked"
    linked.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError):
        sha256_file(linked / "artifact.bin", max_bytes=1024)


def test_sha256_file_uses_open_parent_descriptor_during_directory_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    malicious = tmp_path / "malicious"
    source.mkdir()
    malicious.mkdir()
    target = source / "artifact.bin"
    target.write_bytes(b"trusted")
    (malicious / target.name).write_bytes(b"replaced")
    original_open = os.open
    replaced = False

    def replacing_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal replaced
        if Path(path).name == target.name and not replaced:
            replaced = True
            source.rename(tmp_path / "original-source")
            source.symlink_to(malicious, target_is_directory=True)
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(evidence_module.os, "open", replacing_open)

    assert sha256_file(target, max_bytes=1024) == _digest(b"trusted")


@pytest.mark.skipif(
    not hasattr(os, "O_NONBLOCK") or not hasattr(os, "mkfifo"),
    reason="当前平台不支持以非阻塞方式验证 FIFO",
)
def test_special_fifo_is_opened_nonblocking_and_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _workspace, runtime_root, store = _roots(tmp_path)
    fifo = runtime_root / "eval/special/no-writer.fifo"
    fifo.parent.mkdir(parents=True)
    os.mkfifo(fifo)
    original_open = os.open
    final_component_opened = False

    def assert_nonblocking_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal final_component_opened
        if Path(path).name == fifo.name:
            final_component_opened = True
            assert flags & os.O_NONBLOCK
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(evidence_module.os, "open", assert_nonblocking_open)

    with pytest.raises(ValueError, match="文件摘要计算失败"):
        sha256_file(fifo, max_bytes=1024)
    assert final_component_opened

    with pytest.raises(ValueError, match="机器证据产物绑定失败"):
        store.bind_artifact(
            fifo.relative_to(runtime_root),
            "DATASET_MANIFEST",
        )


def _authorized_dataset_detail_payload(max_video_bytes: object) -> dict[str, object]:
    return {
        "type": "AUTHORIZED_DATASET",
        "trace": {
            "command": ["python", "validate-dataset"],
            "exit_code": 0,
            "stdout_sha256": "a" * 64,
            "stderr_sha256": "b" * 64,
        },
        "manifest_sha256": "c" * 64,
        "authorization_record_sha256": "d" * 64,
        "item_count": 30,
        "language_counts": {language: 6 for language in ("zh", "en", "ja", "ko", "es")},
        "max_video_bytes": max_video_bytes,
    }


@pytest.mark.parametrize("invalid_limit", (True, False))
def test_artifact_and_dataset_limits_reject_bool_before_integer_coercion(
    invalid_limit: bool,
) -> None:
    with pytest.raises(ValidationError):
        TraceArtifact.model_validate(
            {
                "role": "INPUT_MEDIA",
                "relative_path": ".codex/video-rag-demo/eval/media/sample.mp4",
                "sha256": "a" * 64,
                "max_bytes": invalid_limit,
            }
        )
    with pytest.raises(ValidationError):
        AuthorizedDatasetDetails.model_validate(
            _authorized_dataset_detail_payload(invalid_limit)
        )


def test_artifact_and_dataset_limits_accept_regular_integer() -> None:
    artifact = TraceArtifact.model_validate(
        {
            "role": "INPUT_MEDIA",
            "relative_path": ".codex/video-rag-demo/eval/media/sample.mp4",
            "sha256": "a" * 64,
            "max_bytes": 1024,
        }
    )
    details = AuthorizedDatasetDetails.model_validate(
        _authorized_dataset_detail_payload(1024)
    )

    assert artifact.max_bytes == 1024
    assert details.max_video_bytes == 1024


def test_authorized_dataset_detail_rejects_unbounded_media_limit() -> None:
    with pytest.raises(ValidationError):
        AuthorizedDatasetDetails(
            type="AUTHORIZED_DATASET",
            trace=CommandTrace(
                command=("python", "validate-dataset"),
                exit_code=0,
                stdout_sha256="a" * 64,
                stderr_sha256="b" * 64,
            ),
            manifest_sha256="c" * 64,
            authorization_record_sha256="d" * 64,
            item_count=30,
            language_counts={
                "zh": 6,
                "en": 6,
                "ja": 6,
                "ko": 6,
                "es": 6,
            },
            max_video_bytes=4 * 1024 * 1024 * 1024 + 1,
        )


def test_machine_evidence_allows_only_fixed_mypy_dev_null_argument() -> None:
    evidence_module._validate_persisted_string("--cache-dir=/dev/null")


@pytest.mark.parametrize(
    "unsafe",
    (
        "--cache-dir=/tmp/cache",
        "/dev/null",
        "/dev/null/child",
        "/dev/null?child",
        "/dev/null:child",
        "/etc/passwd",
    ),
)
def test_machine_evidence_still_rejects_other_absolute_paths(unsafe: str) -> None:
    with pytest.raises(ValueError, match="绝对路径"):
        evidence_module._validate_persisted_string(unsafe)


def test_media_uses_explicit_limit_but_machine_json_keeps_64_mib_limit(
    tmp_path: Path,
) -> None:
    workspace, runtime_root, store = _roots(tmp_path)
    media = _write(runtime_root, "eval/media/large.mp4", b"x" * (64 * 1024 * 1024 + 1))

    artifact = store.bind_artifact(
        media.relative_to(runtime_root),
        "INPUT_MEDIA",
        max_bytes=65 * 1024 * 1024,
    )
    assert artifact.sha256 == sha256_file(media, max_bytes=65 * 1024 * 1024)

    report = runtime_root / "eval/reports/oversized.json"
    report.parent.mkdir(parents=True)
    with report.open("wb") as stream:
        stream.seek(64 * 1024 * 1024)
        stream.write(b"x")
    with pytest.raises(ValueError):
        load_machine_evidence(report, workspace_root=workspace)


def test_media_limit_above_four_gib_is_rejected_before_path_lookup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _workspace, _runtime_root, store = _roots(tmp_path)
    monkeypatch.setattr(
        evidence_module,
        "_runtime_existing_file",
        lambda *_args: (_ for _ in ()).throw(AssertionError("不得访问文件")),
    )

    with pytest.raises(ValueError, match="产物绑定失败"):
        store.bind_artifact(
            Path("eval/media/oversized.mp4"),
            "INPUT_MEDIA",
            max_bytes=4 * 1024 * 1024 * 1024 + 1,
        )


def test_artifact_reference_persists_the_effective_media_limit(tmp_path: Path) -> None:
    _workspace, runtime_root, store = _roots(tmp_path)
    media = _write(runtime_root, "eval/media/limited.mp4", b"media")

    artifact = store.bind_artifact(
        media.relative_to(runtime_root),
        "INPUT_MEDIA",
        max_bytes=1024,
    )

    assert artifact.max_bytes == 1024
    with pytest.raises(ValidationError):
        TraceArtifact(
            role="INPUT_MEDIA",
            relative_path=artifact.relative_path,
            sha256=artifact.sha256,
            max_bytes=4 * 1024 * 1024 * 1024 + 1,
        )


def test_atomic_writer_rejects_copied_artifact_with_bool_limit(tmp_path: Path) -> None:
    _workspace, runtime_root, store = _roots(tmp_path)
    report = _static_report(store, runtime_root)
    forged = report.model_copy(
        update={
            "artifacts": (
                report.artifacts[0].model_copy(update={"max_bytes": True}),
                *report.artifacts[1:],
            )
        }
    )
    target = runtime_root / "eval/reports/bool-limit.json"

    with pytest.raises(ValueError, match="机器证据原子写入失败"):
        store.write_json(target.relative_to(runtime_root), forged)
    assert not target.exists()


def test_static_pass_capability_is_closed_for_every_producer(
    tmp_path: Path,
) -> None:
    workspace, runtime_root, store = _roots(tmp_path)
    report_path = runtime_root / "eval/reports/no-indexing.json"

    for producer in ("internal", "untrusted-caller"):
        report = _static_report(store, runtime_root, producer=producer)
        store.write_json(report_path.relative_to(runtime_root), report)
        with pytest.raises(ValueError, match="无法形成可信门禁检查"):
            build_verified_gate_check(
                "no_indexing", report_path, workspace_root=workspace
            )

    failed = _static_report(
        store,
        runtime_root,
        producer="untrusted-caller",
        status=GateStatus.FAIL,
        violation_count=1,
    )
    store.write_json(report_path.relative_to(runtime_root), failed)
    assert build_verified_gate_check(
        "no_indexing", report_path, workspace_root=workspace
    ).status == GateStatus.FAIL


def test_generic_command_and_media_pass_capabilities_are_closed(tmp_path: Path) -> None:
    workspace, runtime_root, store = _roots(tmp_path)
    stdout = _write(runtime_root, "eval/generic/stdout.txt", b"completed\n")
    stderr = _write(runtime_root, "eval/generic/stderr.txt", b"")
    audit = _write(runtime_root, "eval/generic/audit.json", b'{"result":"ok"}')
    input_media = _write(runtime_root, "eval/generic/input.mp4", b"not-real-media")
    output_media = _write(runtime_root, "eval/generic/output.mp4", b"not-real-output")
    trace = CommandTrace(
        command=("python", "runner.py"),
        exit_code=0,
        stdout_sha256=_digest(stdout.read_bytes()),
        stderr_sha256=_digest(stderr.read_bytes()),
    )
    common_artifacts = (
        store.bind_artifact(stdout.relative_to(runtime_root), "COMMAND_STDOUT"),
        store.bind_artifact(stderr.relative_to(runtime_root), "COMMAND_STDERR"),
    )
    reports = (
        MachineEvidenceReport(
            schema_version="1.0.0",
            check_id="alembic_roundtrip",
            status=GateStatus.PASS,
            kind=EvidenceKind.COMMAND_REPORT,
            level=EvidenceLevel.CONTRACT,
            covered_items=("alembic_roundtrip",),
            summary="命令自报通过",
            producer="任意调用方",
            started_at=_NOW,
            finished_at=_LATER,
            artifacts=(
                *common_artifacts,
                store.bind_artifact(
                    audit.relative_to(runtime_root),
                    "AUDIT_REPORT",
                ),
            ),
            details=CommandEvidenceDetails(type="COMMAND", trace=trace),
        ),
        MachineEvidenceReport(
            schema_version="1.0.0",
            check_id="real_media_chain",
            status=GateStatus.PASS,
            kind=EvidenceKind.COMMAND_REPORT,
            level=EvidenceLevel.REAL_MEDIA,
            covered_items=("real_media_chain",),
            summary="媒体链自报通过",
            producer="任意调用方",
            started_at=_NOW,
            finished_at=_LATER,
            artifacts=(
                *common_artifacts,
                store.bind_artifact(
                    input_media.relative_to(runtime_root),
                    "INPUT_MEDIA",
                    max_bytes=1024,
                ),
                store.bind_artifact(
                    output_media.relative_to(runtime_root),
                    "OUTPUT_MEDIA",
                    max_bytes=1024,
                ),
            ),
            details=RealMediaDetails(
                type="REAL_MEDIA",
                trace=trace,
                ffmpeg_version="自报版本",
                ffprobe_version="自报版本",
                raw_report_sha256="d" * 64,
                implementation_sha256="e" * 64,
            ),
        ),
    )

    for report in reports:
        report_path = runtime_root / f"eval/reports/{report.check_id}.json"
        store.write_json(report_path.relative_to(runtime_root), report)
        with pytest.raises(ValueError, match="无法形成可信门禁检查"):
            build_verified_gate_check(
                report.check_id,
                report_path,
                workspace_root=workspace,
            )


def test_verified_check_rejects_report_artifact_stdout_and_metadata_tampering(
    tmp_path: Path,
) -> None:
    workspace, runtime_root, store = _roots(tmp_path)
    report = _static_report(store, runtime_root)
    report_path = runtime_root / "eval/reports/no-indexing.json"
    store.write_json(report_path.relative_to(runtime_root), report)

    stdout_path = runtime_root / "eval/audit/stdout.txt"
    stdout_path.write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError):
        build_verified_gate_check("no_indexing", report_path, workspace_root=workspace)

    stdout_path.write_bytes(b"audit completed\n")
    payload = report.model_dump(mode="json")
    payload["summary"] = "伪造摘要"
    report_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError):
        build_verified_gate_check("ruff", report_path, workspace_root=workspace)


def test_verified_check_rejects_wrong_kind_or_level_without_final_report(
    tmp_path: Path,
) -> None:
    workspace, runtime_root, store = _roots(tmp_path)
    report = _static_report(store, runtime_root).model_copy(
        update={
            "kind": EvidenceKind.COMMAND_REPORT,
            "level": EvidenceLevel.CONTRACT,
        }
    )
    report_path = runtime_root / "eval/reports/no-indexing.json"
    store.write_json(report_path.relative_to(runtime_root), report)

    with pytest.raises(ValueError):
        build_verified_gate_check("no_indexing", report_path, workspace_root=workspace)


def test_not_run_with_evidence_still_requires_complete_detail_artifacts(
    tmp_path: Path,
) -> None:
    _workspace, runtime_root, store = _roots(tmp_path)
    report = _static_report(store, runtime_root).model_copy(
        update={"status": GateStatus.NOT_RUN, "not_run_reason": "自报未运行"}
    )
    report_path = runtime_root / "eval/reports/not-run.json"
    with pytest.raises(ValueError, match="机器证据原子写入失败"):
        store.write_json(report_path.relative_to(runtime_root), report)


def test_atomic_writer_revalidates_copied_model_before_persisting(
    tmp_path: Path,
) -> None:
    _workspace, runtime_root, store = _roots(tmp_path)
    copied = _static_report(store, runtime_root).model_copy(
        update={"summary": '"api_key":"visible-secret"'}
    )
    target = runtime_root / "eval/reports/copied.json"

    with pytest.raises(ValueError, match="原子写入失败"):
        store.write_json(target.relative_to(runtime_root), copied)
    assert not target.exists()


def _preflight_report(
    store: EvidenceStore,
    runtime_root: Path,
    *,
    check_id: str = "qwen_live",
    reason_code: str = "QWEN_CREDENTIALS_UNAVAILABLE",
    reported_reason: str = "缺少 Qwen 凭据或真实联调结果",
    execution_started: bool = False,
    exit_code: int = 0,
    implementation_sha256: str = "a" * 64,
) -> tuple[MachineEvidenceReport, Path]:
    issues = {
        "baidu_ocr_live": (
            "BAIDU_API_KEY_UNAVAILABLE",
            "BAIDU_SECRET_KEY_UNAVAILABLE",
            "LIVE_AUTHORIZED_KEYFRAME_UNAVAILABLE",
        ),
        "qwen_live": (
            "QWEN_ENDPOINT_UNAVAILABLE",
            "QWEN_API_KEY_UNAVAILABLE",
            "QWEN_MODEL_ID_UNAVAILABLE",
            "LIVE_AUTHORIZED_CLIP_UNAVAILABLE",
        ),
        "pyannote_live": (
            "PYANNOTE_TOKEN_UNAVAILABLE",
            "PYANNOTE_TERMS_UNAVAILABLE",
            "PYANNOTE_DEPENDENCY_UNAVAILABLE",
            "PYANNOTE_MODEL_UNAVAILABLE",
            "LIVE_AUTHORIZED_AUDIO_UNAVAILABLE",
        ),
        "five_language_models": (
            "SILERO_DEPENDENCY_UNAVAILABLE",
            "SILERO_MODEL_UNAVAILABLE",
            "FASTER_WHISPER_DEPENDENCY_UNAVAILABLE",
            "FASTER_WHISPER_MODEL_UNAVAILABLE",
            "WHISPERX_DEPENDENCY_UNAVAILABLE",
            "WHISPERX_MODEL_UNAVAILABLE",
            "YAMNET_DEPENDENCY_UNAVAILABLE",
            "YAMNET_MODEL_UNAVAILABLE",
            "LIVE_FIVE_LANGUAGE_AUDIO_UNAVAILABLE",
        ),
    }[check_id]
    raw = _write(
        runtime_root,
        f"eval/reports/run-live/{check_id}-preflight.json",
        json.dumps(
            {
                "schema_version": "1.0.0",
                "check_id": check_id,
                "reason_code": reason_code,
                "execution_started": execution_started,
                "issues": [{"code": code} for code in issues],
                "implementation_sha256": implementation_sha256,
                "evaluation_run_id": "run-live",
            },
            separators=(",", ":"),
        ).encode("utf-8"),
    )
    stdout = _write(
        runtime_root,
        f"eval/reports/run-live/{check_id}-stdout.txt",
        b"",
    )
    stderr = _write(
        runtime_root,
        f"eval/reports/run-live/{check_id}-stderr.txt",
        b"",
    )
    payload = {
        "schema_version": "1.0.0",
        "check_id": check_id,
        "status": "NOT_RUN",
        "kind": "LIVE_SERVICE_REPORT",
        "level": "REAL_SERVICE",
        "covered_items": [check_id],
        "summary": "前置条件检查未通过",
        "producer": "preflight-runner",
        "started_at": _NOW,
        "finished_at": _LATER,
        "not_run_reason": reported_reason,
        "artifacts": [
            store.bind_artifact(stdout.relative_to(runtime_root), "COMMAND_STDOUT").model_dump(),
            store.bind_artifact(stderr.relative_to(runtime_root), "COMMAND_STDERR").model_dump(),
            store.bind_artifact(raw.relative_to(runtime_root), "AUDIT_REPORT").model_dump(),
        ],
        "details": {
            "type": "PREFLIGHT",
            "trace": {
                "command": ["python", "-m", "video_demo.evaluation.preflight"],
                "exit_code": exit_code,
                "stdout_sha256": _digest(stdout.read_bytes()),
                "stderr_sha256": _digest(stderr.read_bytes()),
            },
            "preflight_report_sha256": _digest(raw.read_bytes()),
        },
    }
    return MachineEvidenceReport.model_validate(payload), runtime_root / (
        f"eval/reports/run-live/{check_id}.json"
    )


def test_only_structured_preflight_can_derive_machine_not_run(tmp_path: Path) -> None:
    workspace, runtime_root, store = _roots(tmp_path)
    project_root = Path(__file__).parents[2]
    for relative_path in gate_module._LIVE_IMPLEMENTATION_FILES:
        source = project_root / relative_path
        if not source.is_file():
            continue
        target = workspace / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    report, report_path = _preflight_report(
        store,
        runtime_root,
        implementation_sha256=gate_module._current_live_implementation_sha256(
            workspace
        ),
    )
    store.write_json(report_path.relative_to(runtime_root), report)

    check = build_verified_gate_check(
        "qwen_live",
        report_path,
        workspace_root=workspace,
    )

    assert check.status == GateStatus.NOT_RUN
    assert check.not_run_reason == "缺少 Qwen 凭据或真实联调结果"


def test_preflight_rejects_nonzero_trace_exit_code(tmp_path: Path) -> None:
    _workspace, runtime_root, store = _roots(tmp_path)

    with pytest.raises(ValidationError, match="preflight"):
        _preflight_report(store, runtime_root, exit_code=1)


@pytest.mark.parametrize(
    ("reason_code", "execution_started", "reported_reason"),
    (
        ("BAIDU_OCR_CREDENTIALS_UNAVAILABLE", False, "缺少 Qwen 凭据或真实联调结果"),
        ("QWEN_CREDENTIALS_UNAVAILABLE", True, "缺少 Qwen 凭据或真实联调结果"),
        ("QWEN_CREDENTIALS_UNAVAILABLE", False, "调用方自由原因"),
    ),
)
def test_preflight_rejects_wrong_check_started_execution_or_free_reason(
    tmp_path: Path,
    reason_code: str,
    execution_started: bool,
    reported_reason: str,
) -> None:
    workspace, runtime_root, store = _roots(tmp_path)
    report, report_path = _preflight_report(
        store,
        runtime_root,
        reason_code=reason_code,
        execution_started=execution_started,
        reported_reason=reported_reason,
    )
    store.write_json(report_path.relative_to(runtime_root), report)

    with pytest.raises(ValueError):
        build_verified_gate_check(
            "qwen_live",
            report_path,
            workspace_root=workspace,
        )


@pytest.mark.parametrize(
    ("field", "unsafe_value"),
    (
        ("summary", "Authorization: Bearer visible-secret"),
        ("producer", "api_key=visible-secret"),
        ("producer", "API key: visible-secret"),
        ("producer", '"api_key":"visible-secret"'),
        ("covered_items", ["data:text/plain;base64,ZXZpbA=="]),
        ("covered_items", ["/secret"]),
        ("covered_items", [r"\\server\share\secret.txt"]),
    ),
)
def test_machine_report_rejects_unsafe_persisted_strings(
    tmp_path: Path,
    field: str,
    unsafe_value: object,
) -> None:
    _workspace, runtime_root, store = _roots(tmp_path)
    payload = _static_report(store, runtime_root).model_dump(mode="json")
    payload[field] = unsafe_value

    with pytest.raises(ValidationError):
        MachineEvidenceReport.model_validate(payload)


@pytest.mark.parametrize(
    ("role", "content"),
    (
        ("COMMAND_STDOUT", b"Authorization: Bearer visible-secret"),
        ("COMMAND_STDOUT", b'{"transcript":"provider body"}'),
        ("COMMAND_STDOUT", b'INFO {"transcript":"provider body"}'),
        ("COMMAND_STDERR", b"data:text/plain;base64,ZXZpbA=="),
        ("COMMAND_STDERR", b"data:,opaque-provider-body"),
        ("AUDIT_REPORT", b'{"path":"/Users/alice/private.txt"}'),
        (
            "PROVIDER_RESPONSE",
            b'{"id":"chatcmpl_001","choices":[{"text":"provider body"}]}',
        ),
    ),
)
def test_small_machine_artifacts_reject_secrets_paths_data_urls_and_provider_body(
    tmp_path: Path,
    role: str,
    content: bytes,
) -> None:
    _workspace, runtime_root, store = _roots(tmp_path)
    artifact = _write(runtime_root, f"eval/unsafe/{role}.txt", content)

    with pytest.raises(ValueError, match="产物绑定失败"):
        store.bind_artifact(artifact.relative_to(runtime_root), role)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("role", "content"),
    (
        ("DATASET_MANIFEST", b'{"url":"url=data:,opaque"}'),
        ("AUTHORIZATION_RECORD", b'{"body":"body=data:text/plain,x"}'),
        ("ANNOTATION", b'{"cwd":"cwd:/Users/alice/x"}'),
        ("QUALITY_DETAIL", b'{"path":"path=/Users/alice/x"}'),
        ("PREDICTION_INDEX", b'{"cwd":"cwd=C:\\\\Users\\\\alice\\\\x"}'),
        (
            "SEMANTIC_JUDGMENT",
            b'{"share":"share=\\\\server\\share\\x"}',
        ),
        ("PERFORMANCE_REPORT", '{"control":"\u007f"}'.encode()),
        ("AUDIT_REPORT", '{"control":"\u0085"}'.encode()),
    ),
)
def test_all_non_media_artifacts_reject_unsafe_text(
    tmp_path: Path,
    role: str,
    content: bytes,
) -> None:
    _workspace, runtime_root, store = _roots(tmp_path)
    artifact = _write(runtime_root, f"eval/unsafe-all/{role}.json", content)

    with pytest.raises(ValueError, match="机器证据产物绑定失败"):
        store.bind_artifact(artifact.relative_to(runtime_root), role)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "role",
    (
        "DATASET_MANIFEST",
        "AUTHORIZATION_RECORD",
        "ANNOTATION",
        "QUALITY_DETAIL",
    ),
)
def test_structured_non_media_artifacts_remain_bindable(
    tmp_path: Path,
    role: str,
) -> None:
    _workspace, runtime_root, store = _roots(tmp_path)
    content = b'{"schema_version":"1.0.0","url":"https://example.com/path"}'
    artifact = _write(runtime_root, f"eval/safe/{role}.json", content)

    reference = store.bind_artifact(artifact.relative_to(runtime_root), role)  # type: ignore[arg-type]

    assert reference.role == role
    assert reference.sha256 == _digest(content)


@pytest.mark.parametrize(
    ("role", "content"),
    (
        ("ANNOTATION", b'{"nested":{"control":"\\u0000"}}'),
        ("QUALITY_DETAIL", b'{"\\u007f":"value"}'),
    ),
)
def test_single_json_artifacts_reject_escaped_control_in_keys_or_values(
    tmp_path: Path,
    role: str,
    content: bytes,
) -> None:
    _workspace, runtime_root, store = _roots(tmp_path)
    artifact = _write(runtime_root, f"eval/escaped/{role}.json", content)

    with pytest.raises(ValueError, match="机器证据产物绑定失败"):
        store.bind_artifact(artifact.relative_to(runtime_root), role)  # type: ignore[arg-type]


def test_dataset_manifest_jsonl_rejects_escaped_control_value(
    tmp_path: Path,
) -> None:
    _workspace, runtime_root, store = _roots(tmp_path)
    content = b'{"sample_id":"safe"}\n{"nested":{"control":"\\u0085"}}\n'
    manifest = _write(runtime_root, "eval/escaped/dataset.jsonl", content)

    with pytest.raises(ValueError, match="机器证据产物绑定失败"):
        store.bind_artifact(
            manifest.relative_to(runtime_root),
            "DATASET_MANIFEST",
        )


def test_command_output_keeps_non_json_escaped_text_compatibility(
    tmp_path: Path,
) -> None:
    _workspace, runtime_root, store = _roots(tmp_path)
    content = b"completed literal \\u0000 marker\n"
    stdout = _write(runtime_root, "eval/escaped/stdout.txt", content)

    artifact = store.bind_artifact(
        stdout.relative_to(runtime_root),
        "COMMAND_STDOUT",
    )

    assert artifact.sha256 == _digest(content)


def test_final_verifier_rejects_escaped_control_from_artifact_snapshot(
    tmp_path: Path,
) -> None:
    workspace, runtime_root, store = _roots(tmp_path)
    report = _static_report(
        store,
        runtime_root,
        status=GateStatus.FAIL,
        violation_count=1,
    )
    raw_path = runtime_root / "eval/audit/result.json"
    unsafe_content = b'{"violations":[{"message":"\\u0000"}]}'
    raw_path.write_bytes(unsafe_content)
    forged = report.model_copy(
        update={
            "artifacts": tuple(
                artifact.model_copy(update={"sha256": _digest(unsafe_content)})
                if artifact.role == "AUDIT_REPORT"
                else artifact
                for artifact in report.artifacts
            )
        }
    )
    report_path = runtime_root / "eval/reports/escaped-control.json"
    store.write_json(report_path.relative_to(runtime_root), forged)

    with pytest.raises(ValueError, match="无法形成可信门禁检查"):
        build_verified_gate_check(
            "no_indexing",
            report_path,
            workspace_root=workspace,
        )


@pytest.mark.parametrize(
    "content",
    (
        b'{"control":"\\u0000","control":"safe"}',
        b'{"nested":{"control":"\\u0000","control":"safe"}}',
    ),
)
def test_strict_json_rejects_duplicate_keys_at_every_object_depth(
    tmp_path: Path,
    content: bytes,
) -> None:
    _workspace, runtime_root, store = _roots(tmp_path)
    artifact = _write(runtime_root, "eval/strict-json/audit.json", content)

    with pytest.raises(ValueError, match="机器证据产物绑定失败") as captured:
        store.bind_artifact(artifact.relative_to(runtime_root), "AUDIT_REPORT")

    assert captured.value.__cause__ is None


def test_strict_jsonl_rejects_nested_duplicate_key(tmp_path: Path) -> None:
    _workspace, runtime_root, store = _roots(tmp_path)
    content = (
        b'{"sample_id":"safe","nested":{"value":"\\u0000","value":"safe"}}\n'
    )
    manifest = _write(runtime_root, "eval/strict-json/dataset.jsonl", content)

    with pytest.raises(ValueError, match="机器证据产物绑定失败") as captured:
        store.bind_artifact(
            manifest.relative_to(runtime_root),
            "DATASET_MANIFEST",
        )

    assert captured.value.__cause__ is None


@pytest.mark.parametrize(
    ("role", "content"),
    (
        ("AUDIT_REPORT", b'{"value":NaN}'),
        ("AUDIT_REPORT", b'{"value":Infinity}'),
        ("AUDIT_REPORT", b'{"value":-Infinity}'),
        ("DATASET_MANIFEST", b'{"value":NaN}\n'),
    ),
)
def test_strict_json_rejects_non_standard_constants(
    tmp_path: Path,
    role: str,
    content: bytes,
) -> None:
    _workspace, runtime_root, store = _roots(tmp_path)
    artifact = _write(runtime_root, f"eval/strict-json/{role}.json", content)

    with pytest.raises(ValueError, match="机器证据产物绑定失败") as captured:
        store.bind_artifact(artifact.relative_to(runtime_root), role)  # type: ignore[arg-type]

    assert captured.value.__cause__ is None


def test_strict_json_final_verifier_rejects_duplicate_key_with_synced_digest(
    tmp_path: Path,
) -> None:
    workspace, runtime_root, store = _roots(tmp_path)
    report = _static_report(
        store,
        runtime_root,
        status=GateStatus.FAIL,
        violation_count=1,
    )
    raw_path = runtime_root / "eval/audit/result.json"
    duplicate_content = b'{"violations":"\\u0000","violations":[1]}'
    raw_path.write_bytes(duplicate_content)
    forged = report.model_copy(
        update={
            "artifacts": tuple(
                artifact.model_copy(update={"sha256": _digest(duplicate_content)})
                if artifact.role == "AUDIT_REPORT"
                else artifact
                for artifact in report.artifacts
            )
        }
    )
    report_path = runtime_root / "eval/reports/duplicate-key.json"
    store.write_json(report_path.relative_to(runtime_root), forged)

    with pytest.raises(ValueError, match="无法形成可信门禁检查") as captured:
        build_verified_gate_check(
            "no_indexing",
            report_path,
            workspace_root=workspace,
        )

    assert captured.value.__cause__ is None


def _nested_json(depth: int) -> bytes:
    return b"[" * depth + b'"safe"' + b"]" * depth


def _nested_object_json(depth: int) -> bytes:
    return b'{"value":' * depth + b'"safe"' + b"}" * depth


@pytest.mark.parametrize(
    ("role", "content"),
    (
        ("AUDIT_REPORT", _nested_json(1200)),
        ("DATASET_MANIFEST", _nested_json(1200) + b"\n"),
        ("COMMAND_STDOUT", b"INFO " + _nested_json(1200)),
    ),
)
def test_json_depth_binder_fails_closed_without_recursion_error(
    tmp_path: Path,
    role: str,
    content: bytes,
) -> None:
    _workspace, runtime_root, store = _roots(tmp_path)
    artifact = _write(runtime_root, f"eval/json-depth/{role}.txt", content)

    with pytest.raises(ValueError, match="机器证据产物绑定失败") as captured:
        store.bind_artifact(artifact.relative_to(runtime_root), role)  # type: ignore[arg-type]

    assert captured.value.__cause__ is None


def test_json_depth_final_verifier_fails_closed_without_recursion_error(
    tmp_path: Path,
) -> None:
    workspace, runtime_root, store = _roots(tmp_path)
    report = _static_report(
        store,
        runtime_root,
        status=GateStatus.FAIL,
        violation_count=1,
    )
    raw_path = runtime_root / "eval/audit/result.json"
    deep_content = _nested_json(1200)
    raw_path.write_bytes(deep_content)
    forged = report.model_copy(
        update={
            "artifacts": tuple(
                artifact.model_copy(update={"sha256": _digest(deep_content)})
                if artifact.role == "AUDIT_REPORT"
                else artifact
                for artifact in report.artifacts
            )
        }
    )
    report_path = runtime_root / "eval/reports/deep-artifact.json"
    store.write_json(report_path.relative_to(runtime_root), forged)

    with pytest.raises(ValueError, match="无法形成可信门禁检查") as captured:
        build_verified_gate_check(
            "no_indexing",
            report_path,
            workspace_root=workspace,
        )

    assert captured.value.__cause__ is None


def test_json_depth_accepts_reasonably_nested_json(tmp_path: Path) -> None:
    _workspace, runtime_root, store = _roots(tmp_path)
    content = _nested_json(64)
    artifact = _write(runtime_root, "eval/json-depth/reasonable.json", content)

    reference = store.bind_artifact(
        artifact.relative_to(runtime_root),
        "AUDIT_REPORT",
    )

    assert reference.sha256 == _digest(content)


def test_json_depth_rejects_first_level_above_limit(tmp_path: Path) -> None:
    _workspace, runtime_root, store = _roots(tmp_path)
    content = _nested_json(65)
    artifact = _write(runtime_root, "eval/json-depth/above-limit.json", content)

    with pytest.raises(ValueError, match="机器证据产物绑定失败") as captured:
        store.bind_artifact(artifact.relative_to(runtime_root), "AUDIT_REPORT")

    assert captured.value.__cause__ is None


def test_json_depth_ignores_brackets_and_escaped_quotes_inside_strings(
    tmp_path: Path,
) -> None:
    _workspace, runtime_root, store = _roots(tmp_path)
    content = json.dumps(
        {"message": 'quoted " value ' + "[" * 100 + "}" * 100},
        separators=(",", ":"),
    ).encode("utf-8")
    artifact = _write(runtime_root, "eval/json-depth/string-content.json", content)

    reference = store.bind_artifact(
        artifact.relative_to(runtime_root),
        "AUDIT_REPORT",
    )

    assert reference.sha256 == _digest(content)


def test_json_depth_keeps_ordinary_command_text_compatible(tmp_path: Path) -> None:
    _workspace, runtime_root, store = _roots(tmp_path)
    content = b"progress {step 1 of 3}; completed without structured output\n"
    stdout = _write(runtime_root, "eval/json-depth/stdout.txt", content)

    reference = store.bind_artifact(
        stdout.relative_to(runtime_root),
        "COMMAND_STDOUT",
    )

    assert reference.sha256 == _digest(content)


@pytest.mark.parametrize("role", ("COMMAND_STDOUT", "COMMAND_STDERR"))
@pytest.mark.parametrize(
    "content",
    (
        b"progress " + b"[" * 65 + b" not-json\n",
        b"progress " + b"{" * 65 + b" not-json\n",
        b'observed "' + b"[" * 65 + b'" as ordinary text\n',
        b'observed "' + b"{" * 65 + b'" as ordinary text\n',
    ),
)
def test_command_text_accepts_65_non_json_opening_brackets(
    tmp_path: Path,
    role: str,
    content: bytes,
) -> None:
    _workspace, runtime_root, store = _roots(tmp_path)
    path = _write(runtime_root, f"eval/command-candidates/{role}.txt", content)

    reference = store.bind_artifact(
        path.relative_to(runtime_root),
        role,  # type: ignore[arg-type]
    )

    assert reference.sha256 == _digest(content)


@pytest.mark.parametrize("role", ("COMMAND_STDOUT", "COMMAND_STDERR"))
@pytest.mark.parametrize("depth", (64, 65, 1200))
def test_command_text_rejects_real_json_body_at_or_above_depth_limit(
    tmp_path: Path,
    role: str,
    depth: int,
) -> None:
    _workspace, runtime_root, store = _roots(tmp_path)
    content = b"INFO " + _nested_json(depth)
    path = _write(runtime_root, f"eval/command-json/{role}-{depth}.txt", content)

    with pytest.raises(ValueError, match="机器证据产物绑定失败") as captured:
        store.bind_artifact(
            path.relative_to(runtime_root),
            role,  # type: ignore[arg-type]
        )

    assert captured.value.__cause__ is None


def test_command_json_detection_does_not_depth_scan_invalid_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_validate = evidence_module._validate_json_nesting
    calls: list[tuple[int, int | None]] = []

    def recording_validate(
        value: str,
        *,
        start: int = 0,
        end: int | None = None,
    ) -> None:
        calls.append((start, end))
        if end is None:
            real_validate(value, start=start)
        else:
            real_validate(value, start=start, end=end)

    monkeypatch.setattr(
        evidence_module,
        "_validate_json_nesting",
        recording_validate,
    )

    assert evidence_module._contains_json_body("{x}" * 2000) is False
    assert calls == []


def test_command_json_detection_avoids_redundant_depth_rescan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_validate = evidence_module._validate_json_nesting
    calls: list[tuple[int, int | None]] = []

    def recording_validate(
        value: str,
        *,
        start: int = 0,
        end: int | None = None,
    ) -> None:
        calls.append((start, end))
        if end is None:
            real_validate(value, start=start)
        else:
            real_validate(value, start=start, end=end)

    monkeypatch.setattr(
        evidence_module,
        "_validate_json_nesting",
        recording_validate,
    )
    prefix = "{x}" * 2000 + " INFO "
    body = '{"ok":[1]}'
    value = prefix + body + " trailing ordinary text"

    assert evidence_module._contains_json_body(value) is True
    assert calls == []


def test_many_invalid_object_candidates_remain_bindable_command_text(
    tmp_path: Path,
) -> None:
    _workspace, runtime_root, store = _roots(tmp_path)
    content = ("{x}" * 5000 + " completed\n").encode("utf-8")
    stdout = _write(runtime_root, "eval/command-candidates/many.txt", content)

    reference = store.bind_artifact(
        stdout.relative_to(runtime_root),
        "COMMAND_STDOUT",
    )

    assert reference.sha256 == _digest(content)


@pytest.mark.parametrize("role", ("COMMAND_STDOUT", "COMMAND_STDERR"))
@pytest.mark.parametrize(
    "content",
    (
        b"progress " + b"[" * 1200 + b" not-json\n",
        b"progress " + b"{" * 1200 + b" not-json\n",
        (
            b'progress {"message":"escaped \\\" [oops] {not-json}",'
            b'"nested":[{"value":1,"items":[2,3 ordinary-tail\n'
        ),
    ),
)
def test_command_text_accepts_deep_unclosed_non_json_candidates(
    tmp_path: Path,
    role: str,
    content: bytes,
) -> None:
    _workspace, runtime_root, store = _roots(tmp_path)
    path = _write(runtime_root, f"eval/command-unclosed/{role}.txt", content)

    reference = store.bind_artifact(
        path.relative_to(runtime_root),
        role,  # type: ignore[arg-type]
    )

    assert reference.sha256 == _digest(content)


@pytest.mark.parametrize("role", ("COMMAND_STDOUT", "COMMAND_STDERR"))
@pytest.mark.parametrize(
    "content",
    (
        b'INFO {"result":"ok"}',
        b"INFO [1,2,3]",
        b'INFO {"value":1,"value":2}',
        b'INFO {"value":NaN}',
        b"INFO " + _nested_json(64),
        b"INFO " + _nested_object_json(64),
    ),
)
def test_command_text_still_rejects_balanced_json_candidates(
    tmp_path: Path,
    role: str,
    content: bytes,
) -> None:
    _workspace, runtime_root, store = _roots(tmp_path)
    path = _write(runtime_root, f"eval/command-balanced/{role}.txt", content)

    with pytest.raises(ValueError, match="机器证据产物绑定失败") as captured:
        store.bind_artifact(
            path.relative_to(runtime_root),
            role,  # type: ignore[arg-type]
        )

    assert captured.value.__cause__ is None


@pytest.mark.parametrize("role", ("COMMAND_STDOUT", "COMMAND_STDERR"))
@pytest.mark.parametrize(
    "content",
    (
        b"INFO " + b"[" * 1200 + b"not-json" + b"]" * 1200,
        b"INFO " + b'{"value":' * 1200 + b"not-json" + b"}" * 1200,
    ),
)
def test_command_text_accepts_balanced_but_invalid_deep_candidates(
    tmp_path: Path,
    role: str,
    content: bytes,
) -> None:
    _workspace, runtime_root, store = _roots(tmp_path)
    path = _write(runtime_root, f"eval/command-invalid-deep/{role}.txt", content)

    reference = store.bind_artifact(
        path.relative_to(runtime_root),
        role,  # type: ignore[arg-type]
    )

    assert reference.sha256 == _digest(content)


@pytest.mark.parametrize("role", ("COMMAND_STDOUT", "COMMAND_STDERR"))
@pytest.mark.parametrize(
    "content",
    (
        b'[oops {"ok":true}]',
        b'{oops {"ok":true}}',
        b"[oops [1,2]]",
        b"{oops [1,2]}",
        b'[oops {"ok":true}',
        b"{oops [1,2]",
        b'[{"ok":true}}',
        b'[{"ok":true}]',
    ),
)
def test_command_text_rejects_embedded_json_after_candidate_recovery(
    tmp_path: Path,
    role: str,
    content: bytes,
) -> None:
    _workspace, runtime_root, store = _roots(tmp_path)
    path = _write(runtime_root, f"eval/embedded-json/{role}.txt", content)

    with pytest.raises(ValueError, match="机器证据产物绑定失败") as captured:
        store.bind_artifact(
            path.relative_to(runtime_root),
            role,  # type: ignore[arg-type]
        )

    assert captured.value.__cause__ is None


@pytest.mark.parametrize("role", ("COMMAND_STDOUT", "COMMAND_STDERR"))
@pytest.mark.parametrize(
    "prefix",
    (
        b'WARN "unterminated\n',
        b'WARN "unterminated\\\n',
    ),
)
def test_command_text_rejects_embedded_json_after_unterminated_quote_newline(
    tmp_path: Path,
    role: str,
    prefix: bytes,
) -> None:
    _workspace, runtime_root, store = _roots(tmp_path)
    content = prefix + b'INFO {"ok":true}\n'
    path = _write(runtime_root, f"eval/unterminated-quote/{role}.txt", content)

    with pytest.raises(ValueError, match="机器证据产物绑定失败") as captured:
        store.bind_artifact(
            path.relative_to(runtime_root),
            role,  # type: ignore[arg-type]
        )

    assert captured.value.__cause__ is None


@pytest.mark.parametrize("role", ("COMMAND_STDOUT", "COMMAND_STDERR"))
@pytest.mark.parametrize(
    "content",
    (
        b'WARN "unterminated INFO {}\n',
        b'WARN "unterminated INFO [1,2]\n',
        b'WARN "unterminated INFO {}',
        b'WARN "unterminated INFO [1,2]',
        b'WARN "unterminated INFO {}\\\n',
        b'WARN "unterminated INFO [1,2]\\',
    ),
)
def test_command_text_rejects_json_after_same_line_unterminated_quote(
    tmp_path: Path,
    role: str,
    content: bytes,
) -> None:
    _workspace, runtime_root, store = _roots(tmp_path)
    path = _write(runtime_root, f"eval/unterminated-quote-line/{role}.txt", content)

    with pytest.raises(ValueError, match="机器证据产物绑定失败") as captured:
        store.bind_artifact(
            path.relative_to(runtime_root),
            role,  # type: ignore[arg-type]
        )

    assert captured.value.__cause__ is None


@pytest.mark.parametrize("role", ("COMMAND_STDOUT", "COMMAND_STDERR"))
@pytest.mark.parametrize(
    "json_body",
    (
        b'{"ok":true}',
        b'["value",2]',
        b'[{"ok":true},2]',
    ),
    ids=("nonempty-object", "string-array", "object-array"),
)
@pytest.mark.parametrize(
    "ending",
    (b"\n", b"\r\n", b"\\\n", b"\\\r\n", b""),
    ids=("lf", "crlf", "backslash-lf", "backslash-crlf", "eof"),
)
def test_command_text_rejects_json_with_strings_after_unterminated_quote(
    tmp_path: Path,
    role: str,
    json_body: bytes,
    ending: bytes,
) -> None:
    _workspace, runtime_root, store = _roots(tmp_path)
    content = b'WARN "unterminated INFO ' + json_body + ending
    path = _write(
        runtime_root,
        f"eval/unterminated-quote-json-string/{role}.txt",
        content,
    )

    with pytest.raises(ValueError, match="机器证据产物绑定失败") as captured:
        store.bind_artifact(
            path.relative_to(runtime_root),
            role,  # type: ignore[arg-type]
        )

    assert captured.value.__cause__ is None


@pytest.mark.parametrize("role", ("COMMAND_STDOUT", "COMMAND_STDERR"))
@pytest.mark.parametrize(
    "json_body",
    (
        b'{"ok":"value"}',
        b'{"first":1,"second":2}',
        b'{"outer":{"enabled":true},"message":"done"}',
    ),
    ids=("string-value", "multiple-keys", "nested-then-string"),
)
@pytest.mark.parametrize(
    "ending",
    (b"\n", b"\r\n", b"\\\n", b"\\\r\n", b""),
    ids=("lf", "crlf", "backslash-lf", "backslash-crlf", "eof"),
)
def test_command_text_rejects_later_json_strings_after_unterminated_quote(
    tmp_path: Path,
    role: str,
    json_body: bytes,
    ending: bytes,
) -> None:
    _workspace, runtime_root, store = _roots(tmp_path)
    content = b'WARN "unterminated INFO ' + json_body + ending
    path = _write(
        runtime_root,
        f"eval/unterminated-quote-later-json-string/{role}.txt",
        content,
    )

    with pytest.raises(ValueError, match="机器证据产物绑定失败") as captured:
        store.bind_artifact(
            path.relative_to(runtime_root),
            role,  # type: ignore[arg-type]
        )

    assert captured.value.__cause__ is None


@pytest.mark.parametrize("role", ("COMMAND_STDOUT", "COMMAND_STDERR"))
@pytest.mark.parametrize(
    "line_break",
    (b"\n", b"\r", b"\r\n"),
    ids=("lf", "cr", "crlf"),
)
@pytest.mark.parametrize("json_body", (b"{}", b"[]"), ids=("object", "array"))
def test_command_text_rejects_json_after_backslash_line_break_with_tail_quote(
    tmp_path: Path,
    role: str,
    line_break: bytes,
    json_body: bytes,
) -> None:
    _workspace, runtime_root, store = _roots(tmp_path)
    content = b'WARN "unterminated\\' + line_break + b"INFO " + json_body + b'"\n'
    path = _write(
        runtime_root,
        f"eval/backslash-line-break-tail-quote/{role}.txt",
        content,
    )

    with pytest.raises(ValueError, match="机器证据产物绑定失败") as captured:
        store.bind_artifact(
            path.relative_to(runtime_root),
            role,  # type: ignore[arg-type]
        )

    assert captured.value.__cause__ is None


@pytest.mark.parametrize("role", ("COMMAND_STDOUT", "COMMAND_STDERR"))
@pytest.mark.parametrize(
    "json_body",
    (b'{"ok":"value"}', b'["value",2]'),
    ids=("object", "array"),
)
def test_command_text_rejects_json_opener_immediately_after_backslash(
    tmp_path: Path,
    role: str,
    json_body: bytes,
) -> None:
    _workspace, runtime_root, store = _roots(tmp_path)
    content = b'WARN "unterminated INFO \\' + json_body + b'"\n'
    path = _write(
        runtime_root,
        f"eval/backslash-before-json-opener/{role}.txt",
        content,
    )

    with pytest.raises(ValueError, match="机器证据产物绑定失败") as captured:
        store.bind_artifact(
            path.relative_to(runtime_root),
            role,  # type: ignore[arg-type]
        )

    assert captured.value.__cause__ is None


@pytest.mark.parametrize("role", ("COMMAND_STDOUT", "COMMAND_STDERR"))
@pytest.mark.parametrize(
    "content",
    (
        b'WARN "quoted {} and [1,2]" completed\n',
        b'WARN "quoted {"first":1,"second":2}" completed\n',
        b'WARN "quoted {"ok":"value"}" completed\n',
        b'WARN "quoted [{"nested":[1,2]}]" completed\n',
    ),
)
def test_command_text_rejects_json_body_inside_paired_quote(
    tmp_path: Path,
    role: str,
    content: bytes,
) -> None:
    _workspace, runtime_root, store = _roots(tmp_path)
    path = _write(runtime_root, f"eval/paired-quote/{role}.txt", content)

    with pytest.raises(ValueError, match="机器证据产物绑定失败") as captured:
        store.bind_artifact(
            path.relative_to(runtime_root),
            role,  # type: ignore[arg-type]
        )

    assert captured.value.__cause__ is None


@pytest.mark.parametrize("role", ("COMMAND_STDOUT", "COMMAND_STDERR"))
@pytest.mark.parametrize(
    "content",
    (
        b'INFO {"message":"{oops]"}',
        b'INFO {"message":"[oops}"}',
        b'INFO {"message":"[oops"}',
        b'INFO ["{oops]"]',
        b'INFO {"message":"{","next":1}',
        b'INFO ["[",1]',
        (
            b'INFO {"message":"'
            + b"[" * 1200
            + b"oops"
            + b"]" * 1200
            + b'"}'
        ),
    ),
    ids=(
        "mismatched-object-opener",
        "mismatched-array-opener",
        "unclosed-array-opener",
        "array-string-with-mismatched-opener",
        "unclosed-object-opener-before-next-field",
        "unclosed-array-opener-before-next-value",
        "deep-invalid-array-candidate",
    ),
)
def test_command_text_rejects_outer_json_with_invalid_string_candidates(
    tmp_path: Path,
    role: str,
    content: bytes,
) -> None:
    _workspace, runtime_root, store = _roots(tmp_path)
    path = _write(runtime_root, f"eval/json-string-candidate/{role}.txt", content)

    with pytest.raises(ValueError, match="机器证据产物绑定失败") as captured:
        store.bind_artifact(
            path.relative_to(runtime_root),
            role,  # type: ignore[arg-type]
        )

    assert captured.value.__cause__ is None


@pytest.mark.parametrize("role", ("COMMAND_STDOUT", "COMMAND_STDERR"))
def test_command_text_accepts_invalid_brackets_without_lane_saturation(
    tmp_path: Path,
    role: str,
) -> None:
    _workspace, runtime_root, store = _roots(tmp_path)
    content = b"progress {x} [oops] [progress"
    path = _write(runtime_root, f"eval/json-lane-control/{role}.txt", content)

    artifact = store.bind_artifact(
        path.relative_to(runtime_root),
        role,  # type: ignore[arg-type]
    )

    assert artifact.sha256 == _digest(content)


def test_invalid_unicode_string_lane_is_reclaimed_without_false_saturation() -> None:
    assert evidence_module._contains_json_body('["\\u["{') is False


@pytest.mark.parametrize("role", ("COMMAND_STDOUT", "COMMAND_STDERR"))
def test_command_text_accepts_invalid_unicode_string_without_json(
    tmp_path: Path,
    role: str,
) -> None:
    _workspace, runtime_root, store = _roots(tmp_path)
    content = b'INFO ["\\u["{'
    path = _write(runtime_root, f"eval/dead-lane/{role}.txt", content)

    artifact = store.bind_artifact(
        path.relative_to(runtime_root),
        role,  # type: ignore[arg-type]
    )

    assert artifact.sha256 == _digest(content)


@pytest.mark.parametrize("role", ("COMMAND_STDOUT", "COMMAND_STDERR"))
@pytest.mark.parametrize("json_body", (b"{}", b"[]"))
def test_command_text_recovers_json_after_invalid_unicode_string_lane(
    tmp_path: Path,
    role: str,
    json_body: bytes,
) -> None:
    _workspace, runtime_root, store = _roots(tmp_path)
    content = b'INFO ["\\u[" tail\n' + json_body
    path = _write(runtime_root, f"eval/dead-lane-recovery/{role}.txt", content)

    with pytest.raises(ValueError, match="机器证据产物绑定失败") as captured:
        store.bind_artifact(
            path.relative_to(runtime_root),
            role,  # type: ignore[arg-type]
        )

    assert captured.value.__cause__ is None


@pytest.mark.parametrize(
    "value",
    (
        '["\\q["{',
        '["\\u["{',
        '["\\u12["{',
        '["\x01["{',
    ),
    ids=(
        "invalid-escape",
        "invalid-unicode-first-character",
        "incomplete-unicode-before-opener",
        "control-character",
    ),
)
def test_invalid_string_lane_matrix_reclaims_capacity(value: str) -> None:
    assert evidence_module._contains_json_body(value) is False
    assert evidence_module._contains_json_body(value + " {}") is True


def test_string_invalidating_opener_is_reprocessed_as_new_candidate() -> None:
    assert evidence_module._contains_json_body('["\\u[]') is True


@pytest.mark.parametrize("role", ("COMMAND_STDOUT", "COMMAND_STDERR"))
@pytest.mark.parametrize(
    "content",
    (
        b'WARN "escaped \\\"{\\\"ok\\\":true}\\\" and \\\\[text" completed\n',
        b'WARN "literal \\{not-json} and \\[oops]" completed\n',
    ),
)
def test_command_text_accepts_non_json_escaped_text_inside_paired_quote(
    tmp_path: Path,
    role: str,
    content: bytes,
) -> None:
    _workspace, runtime_root, store = _roots(tmp_path)
    path = _write(runtime_root, f"eval/paired-quote-escaped/{role}.txt", content)

    reference = store.bind_artifact(
        path.relative_to(runtime_root),
        role,  # type: ignore[arg-type]
    )

    assert reference.sha256 == _digest(content)


@pytest.mark.parametrize("role", ("COMMAND_STDOUT", "COMMAND_STDERR"))
@pytest.mark.parametrize(
    "content",
    (
        b'[0 " ," INFO {"ok":true}',
        b'{"invalid" " :" INFO [1,2]',
    ),
    ids=("array-shell-object-body", "object-shell-array-body"),
)
def test_command_text_rejects_json_after_paired_quote_and_prior_candidate(
    tmp_path: Path,
    role: str,
    content: bytes,
) -> None:
    _workspace, runtime_root, store = _roots(tmp_path)
    path = _write(runtime_root, f"eval/prior-candidate-paired-quote/{role}.txt", content)

    with pytest.raises(ValueError, match="机器证据产物绑定失败") as captured:
        store.bind_artifact(
            path.relative_to(runtime_root),
            role,  # type: ignore[arg-type]
        )

    assert captured.value.__cause__ is None


@pytest.mark.parametrize("role", ("COMMAND_STDOUT", "COMMAND_STDERR"))
@pytest.mark.parametrize("line_break", (b"\n", b"\r", b"\r\n"))
@pytest.mark.parametrize("json_body", (b"{}", b"[]"))
def test_command_text_rejects_next_line_json_after_prior_candidate_and_quote(
    tmp_path: Path,
    role: str,
    line_break: bytes,
    json_body: bytes,
) -> None:
    _workspace, runtime_root, store = _roots(tmp_path)
    content = b'[0 " ,"tail' + line_break + b"INFO " + json_body
    path = _write(runtime_root, f"eval/prior-candidate-newline/{role}.txt", content)

    with pytest.raises(ValueError, match="机器证据产物绑定失败") as captured:
        store.bind_artifact(
            path.relative_to(runtime_root),
            role,  # type: ignore[arg-type]
        )

    assert captured.value.__cause__ is None


@pytest.mark.parametrize("role", ("COMMAND_STDOUT", "COMMAND_STDERR"))
@pytest.mark.parametrize("depth", (64, 65, 1200))
def test_command_text_rejects_deep_json_inside_or_crossing_paired_quote(
    tmp_path: Path,
    role: str,
    depth: int,
) -> None:
    _workspace, runtime_root, store = _roots(tmp_path)
    nested = b"[" * depth + b"]" * (depth - 1) + b',"x"]'
    content = b'WARN "' + nested + b'" completed\n'
    path = _write(runtime_root, f"eval/quoted-deep-json/{role}-{depth}.txt", content)

    with pytest.raises(ValueError, match="机器证据产物绑定失败") as captured:
        store.bind_artifact(
            path.relative_to(runtime_root),
            role,  # type: ignore[arg-type]
        )

    assert captured.value.__cause__ is None


@pytest.mark.parametrize("role", ("COMMAND_STDOUT", "COMMAND_STDERR"))
@pytest.mark.parametrize(
    "content",
    (
        b'WARN "ordinary [" INFO {"ok":true}\n',
        b'WARN "ordinary {" INFO ["value",2]\n',
        b'WARN "ordinary [ text" INFO {"ok":true}\n',
        b'WARN "ordinary { text" INFO ["value",2]\n',
    ),
)
def test_command_text_rejects_json_after_paired_quote_with_invalid_candidate(
    tmp_path: Path,
    role: str,
    content: bytes,
) -> None:
    _workspace, runtime_root, store = _roots(tmp_path)
    path = _write(runtime_root, f"eval/paired-quote-then-json/{role}.txt", content)

    with pytest.raises(ValueError, match="机器证据产物绑定失败") as captured:
        store.bind_artifact(
            path.relative_to(runtime_root),
            role,  # type: ignore[arg-type]
        )

    assert captured.value.__cause__ is None


@pytest.mark.parametrize("role", ("COMMAND_STDOUT", "COMMAND_STDERR"))
@pytest.mark.parametrize(
    "content",
    (
        b'[oops {"value":1,"value":2}]',
        b'[oops {"value":NaN}]',
        b"[oops " + _nested_json(1200) + b"]",
    ),
)
def test_command_text_rejects_embedded_json_policy_violations(
    tmp_path: Path,
    role: str,
    content: bytes,
) -> None:
    _workspace, runtime_root, store = _roots(tmp_path)
    path = _write(runtime_root, f"eval/embedded-json-policy/{role}.txt", content)

    with pytest.raises(ValueError, match="机器证据产物绑定失败") as captured:
        store.bind_artifact(
            path.relative_to(runtime_root),
            role,  # type: ignore[arg-type]
        )

    assert captured.value.__cause__ is None


@pytest.mark.parametrize("role", ("COMMAND_STDOUT", "COMMAND_STDERR"))
@pytest.mark.parametrize(
    "content",
    (
        b"INFO [oops {not-json}] completed\n",
        b"INFO {oops [1,broken] completed\n",
        b"INFO [oops {still-text\n",
        b"INFO [oops {still-text] completed\n",
        b"{x}[oops]{not-json} completed\n",
    ),
)
def test_command_text_accepts_candidate_recovery_controls_without_json(
    tmp_path: Path,
    role: str,
    content: bytes,
) -> None:
    _workspace, runtime_root, store = _roots(tmp_path)
    path = _write(runtime_root, f"eval/candidate-recovery-controls/{role}.txt", content)

    reference = store.bind_artifact(
        path.relative_to(runtime_root),
        role,  # type: ignore[arg-type]
    )

    assert reference.sha256 == _digest(content)


class _CountingCommandText(str):
    scanned_characters: int

    def __new__(cls, value: str) -> _CountingCommandText:
        instance = super().__new__(cls, value)
        instance.scanned_characters = 0
        return instance

    def __getitem__(self, key: int | slice) -> str:
        if isinstance(key, int):
            self.scanned_characters += 1
        return super().__getitem__(key)


class _TrackingCompactStack(bytearray):
    instances: ClassVar[list[_TrackingCompactStack]] = []
    max_length: int

    def __init__(self, *args: object) -> None:
        super().__init__(*args)
        self.max_length = len(self)
        self.instances.append(self)

    def append(self, item: int) -> None:
        super().append(item)
        self.max_length = max(self.max_length, len(self))

    def extend(self, value: object) -> None:
        super().extend(value)  # type: ignore[arg-type]
        self.max_length = max(self.max_length, len(self))


class _TrackingFrameList(list[object]):
    instances: ClassVar[list[_TrackingFrameList]] = []
    max_length: int

    def __init__(self) -> None:
        super().__init__()
        self.max_length = 0
        self.instances.append(self)

    def append(self, item: object) -> None:
        if len(self) >= evidence_module._MAX_COMMAND_JSON_FRAMES:
            raise AssertionError("单条 lane 活跃 frame 超过 64")
        super().append(item)
        self.max_length = max(self.max_length, len(self))


def _install_candidate_storage_counters(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, int]:
    real_frame = evidence_module._JsonCandidateFrame
    real_lane = evidence_module._JsonCandidateLane
    counts = {"active": 0, "created": 0, "peak": 0, "lanes": 0}

    class CountingFrame:
        _tracked: bool

        def __init__(self, *args: object, **kwargs: object) -> None:
            self._tracked = False
            if counts["active"] >= 128:
                raise AssertionError("双 lane 活跃普通 JSON 候选 frame 超过 128")
            original = real_frame(*args, **kwargs)  # type: ignore[arg-type]
            for name, item in vars(original).items():
                setattr(self, name, item)
            counts["active"] += 1
            counts["created"] += 1
            counts["peak"] = max(counts["peak"], counts["active"])
            self._tracked = True

        def __del__(self) -> None:
            if getattr(self, "_tracked", False):
                counts["active"] -= 1

    class TrackingLane(real_lane):
        def __init__(self) -> None:
            counts["lanes"] += 1
            super().__init__(
                frames=_TrackingFrameList(),  # type: ignore[arg-type]
                expected_closers=_TrackingCompactStack(),
            )

    _TrackingCompactStack.instances = []
    _TrackingFrameList.instances = []
    monkeypatch.setattr(evidence_module, "_JsonCandidateFrame", CountingFrame)
    monkeypatch.setattr(evidence_module, "_JsonCandidateLane", TrackingLane)
    return counts


def test_command_scanner_creates_exactly_two_fixed_lanes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    counts = _install_candidate_storage_counters(monkeypatch)

    assert evidence_module._contains_json_body("ordinary text") is False
    assert counts["lanes"] == 2
    assert len(_TrackingFrameList.instances) == 2
    assert len(_TrackingCompactStack.instances) == 2


def test_command_scanner_fails_closed_when_both_lanes_are_in_strings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_lane = evidence_module._JsonCandidateLane
    real_frame = evidence_module._JsonCandidateFrame
    created_lanes = 0

    def saturated_lane() -> object:
        nonlocal created_lanes
        created_lanes += 1
        return real_lane(
            frames=[
                real_frame(
                    kind="array",
                    state="value_or_end",
                    start=0,
                    depth=1,
                    accepted_by_parent=False,
                    in_string=True,
                )
            ],
            expected_closers=bytearray(b"]"),
        )

    monkeypatch.setattr(evidence_module, "_JsonCandidateLane", saturated_lane)

    assert evidence_module._contains_json_body("[") is True
    assert created_lanes == 2


@pytest.mark.parametrize(
    ("opener", "count"),
    (("[", 100_000), ("{", 1_000_000)),
)
def test_incremental_scanner_bounds_candidate_frames_and_compact_closer_bytes(
    monkeypatch: pytest.MonkeyPatch,
    opener: str,
    count: int,
) -> None:
    counts = _install_candidate_storage_counters(monkeypatch)
    text = _CountingCommandText(
        opener + '"' + opener + '""' + opener * count + " ordinary-tail"
    )

    assert evidence_module._contains_json_body(text) is False
    assert evidence_module._MAX_COMMAND_JSON_FRAMES == 64
    assert counts["lanes"] == 2
    assert counts["created"] <= 128
    assert counts["peak"] <= 128
    assert len(_TrackingFrameList.instances) == 2
    assert all(
        frames.max_length <= 64 for frames in _TrackingFrameList.instances
    )
    assert sum(frames.max_length for frames in _TrackingFrameList.instances) <= 128
    assert len(_TrackingCompactStack.instances) == 2
    assert sum(
        stack.max_length for stack in _TrackingCompactStack.instances
    ) <= 2 * text.count(opener)
    assert 0 < text.scanned_characters <= 2 * len(text) + 2


def test_incremental_scanner_recovers_shallow_json_inside_deep_invalid_shell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_decoder = evidence_module._strict_json_decoder()
    raw_decode_calls = 0
    counts = _install_candidate_storage_counters(monkeypatch)

    class CountingDecoder:
        def raw_decode(self, value: str, index: int = 0) -> tuple[object, int]:
            nonlocal raw_decode_calls
            raw_decode_calls += 1
            return real_decoder.raw_decode(value, index)

    monkeypatch.setattr(
        evidence_module,
        "_strict_json_decoder",
        lambda: CountingDecoder(),
    )
    body = '{"ok":true}'
    text = _CountingCommandText("[oops " * 100_000 + body)

    assert evidence_module._contains_json_body(text) is True
    assert raw_decode_calls == 1
    assert counts["created"] <= 64
    assert counts["peak"] <= 64
    assert 0 < text.scanned_characters <= len(text) + 1


@pytest.mark.parametrize("opener", ("[", "{"))
def test_unclosed_overlapping_candidates_are_scanned_once_without_raw_decode(
    monkeypatch: pytest.MonkeyPatch,
    opener: str,
) -> None:
    real_decoder = evidence_module._strict_json_decoder()
    raw_decode_calls = 0

    class CountingDecoder:
        def raw_decode(self, value: str, index: int = 0) -> tuple[object, int]:
            nonlocal raw_decode_calls
            raw_decode_calls += 1
            return real_decoder.raw_decode(value, index)

    monkeypatch.setattr(
        evidence_module,
        "_strict_json_decoder",
        lambda: CountingDecoder(),
    )
    text = _CountingCommandText(opener * 1200 + " not-json")

    assert evidence_module._contains_json_body(text) is False
    assert raw_decode_calls == 0
    assert 0 < text.scanned_characters <= len(text) + 1


def test_balanced_candidate_uses_one_strict_decode_after_linear_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_decoder = evidence_module._strict_json_decoder()
    raw_decode_calls = 0

    class CountingDecoder:
        def raw_decode(self, value: str, index: int = 0) -> tuple[object, int]:
            nonlocal raw_decode_calls
            raw_decode_calls += 1
            return real_decoder.raw_decode(value, index)

    monkeypatch.setattr(
        evidence_module,
        "_strict_json_decoder",
        lambda: CountingDecoder(),
    )
    body = '{"message":"escaped \\\" [ ] }","values":[1,{"ok":true}]}'
    text = _CountingCommandText("INFO " + body + " trailing")

    assert evidence_module._contains_json_body(text) is True
    assert raw_decode_calls == 1
    assert 0 < text.scanned_characters <= len(text) + 1


def test_incremental_scanner_recovers_embedded_json_without_opener_backtracking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_decoder = evidence_module._strict_json_decoder()
    raw_decode_calls = 0

    class CountingDecoder:
        def raw_decode(self, value: str, index: int = 0) -> tuple[object, int]:
            nonlocal raw_decode_calls
            raw_decode_calls += 1
            return real_decoder.raw_decode(value, index)

    monkeypatch.setattr(
        evidence_module,
        "_strict_json_decoder",
        lambda: CountingDecoder(),
    )
    body = '{"ok":true}'
    text = _CountingCommandText("[oops " * 1200 + body + "]" * 1200)

    assert evidence_module._contains_json_body(text) is True
    assert raw_decode_calls == 1
    assert 0 < text.scanned_characters <= len(text) + 1


def test_incremental_scanner_does_not_rescan_nested_invalid_spans(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_decoder = evidence_module._strict_json_decoder()
    raw_decode_calls = 0

    class CountingDecoder:
        def raw_decode(self, value: str, index: int = 0) -> tuple[object, int]:
            nonlocal raw_decode_calls
            raw_decode_calls += 1
            return real_decoder.raw_decode(value, index)

    monkeypatch.setattr(
        evidence_module,
        "_strict_json_decoder",
        lambda: CountingDecoder(),
    )
    text = _CountingCommandText("[oops " * 1200 + "tail" + "]" * 1200)

    assert evidence_module._contains_json_body(text) is False
    assert raw_decode_calls == 0
    assert 0 < text.scanned_characters <= 2 * len(text) + 2


def test_incremental_scanner_decodes_at_most_twice_before_a_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_decoder = evidence_module._strict_json_decoder()
    raw_decode_calls = 0

    class CountingDecoder:
        def raw_decode(self, value: str, index: int = 0) -> tuple[object, int]:
            nonlocal raw_decode_calls
            raw_decode_calls += 1
            return real_decoder.raw_decode(value, index)

    monkeypatch.setattr(
        evidence_module,
        "_strict_json_decoder",
        lambda: CountingDecoder(),
    )
    text = _CountingCommandText('INFO ["[",1]')

    assert evidence_module._contains_json_body(text) is True
    assert raw_decode_calls <= 2
    assert 0 < text.scanned_characters <= 2 * len(text) + 2


def test_quote_characters_do_not_add_a_second_command_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_decoder = evidence_module._strict_json_decoder()
    raw_decode_calls = 0

    class CountingDecoder:
        def raw_decode(self, value: str, index: int = 0) -> tuple[object, int]:
            nonlocal raw_decode_calls
            raw_decode_calls += 1
            return real_decoder.raw_decode(value, index)

    monkeypatch.setattr(
        evidence_module,
        "_strict_json_decoder",
        lambda: CountingDecoder(),
    )
    body = '{"message":"value","items":[{"ok":true}]}'
    text = _CountingCommandText('WARN "' + "[oops " * 5000 + body + "\\\n")

    assert evidence_module._contains_json_body(text) is True
    assert raw_decode_calls == 1
    assert 0 < text.scanned_characters <= len(text) + 1


def test_command_scanner_uses_one_scan_with_two_bounded_lanes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame_fields = set(evidence_module._JsonCandidateFrame.__dataclass_fields__)
    counts = _install_candidate_storage_counters(monkeypatch)
    real_scan_branch = evidence_module._scan_command_candidate_branch
    branch_calls = 0

    def count_branch(*args: object, **kwargs: object) -> bool:
        nonlocal branch_calls
        branch_calls += 1
        return real_scan_branch(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        evidence_module,
        "_scan_command_candidate_branch",
        count_branch,
    )
    text = _CountingCommandText('WARN "ordinary [" INFO {"ok":true}\n')

    assert evidence_module._contains_json_body(text) is True
    assert branch_calls == 1
    assert counts["lanes"] == 2
    assert counts["created"] <= 128
    assert counts["peak"] <= 128
    assert all(
        frames.max_length <= 64 for frames in _TrackingFrameList.instances
    )
    assert sum(frames.max_length for frames in _TrackingFrameList.instances) <= 128
    assert sum(
        stack.max_length for stack in _TrackingCompactStack.instances
    ) <= 2 * text.count("[") + 2 * text.count("{")
    assert 0 < text.scanned_characters <= 2 * len(text) + 2
    obsolete_helpers = (
        "_command_" + "quote_boundaries",
        "_command_" + "quote_boundary_at",
        "_advance_command_" + "quote_boundary",
    )
    assert all(not hasattr(evidence_module, name) for name in obsolete_helpers)
    assert frame_fields == {
        "kind",
        "state",
        "start",
        "depth",
        "accepted_by_parent",
        "valid",
        "in_string",
        "string_escaped",
        "unicode_escape_remaining",
    }


def test_machine_report_replacement_during_parse_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, runtime_root, store = _roots(tmp_path)
    report = _static_report(store, runtime_root, status=GateStatus.FAIL, violation_count=1)
    report_path = runtime_root / "eval/reports/replaced.json"
    store.write_json(report_path.relative_to(runtime_root), report)
    original_validate = MachineEvidenceReport.model_validate

    def replace_while_validating(
        _cls: type[MachineEvidenceReport],
        payload: object,
        *args: object,
        **kwargs: object,
    ) -> MachineEvidenceReport:
        replacement = report_path.with_suffix(".replacement")
        replacement.write_bytes(report_path.read_bytes())
        os.replace(replacement, report_path)
        return original_validate(payload, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        MachineEvidenceReport,
        "model_validate",
        classmethod(replace_while_validating),
    )

    with pytest.raises(ValueError, match="报告非法或不可信"):
        load_machine_evidence(report_path, workspace_root=workspace)


def test_artifact_replacement_between_digest_and_status_derivation_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, runtime_root, store = _roots(tmp_path)
    report = _static_report(store, runtime_root, status=GateStatus.FAIL, violation_count=1)
    report_path = runtime_root / "eval/reports/artifact-replaced.json"
    store.write_json(report_path.relative_to(runtime_root), report)
    raw_path = runtime_root / "eval/audit/result.json"
    from video_demo.evaluation import gate as gate_module

    real_derive = gate_module._derive_machine_gate_status

    def replacing_derive(*args: object, **kwargs: object) -> tuple[GateStatus, str | None]:
        replacement = raw_path.with_suffix(".replacement")
        replacement.write_bytes(raw_path.read_bytes())
        os.replace(replacement, raw_path)
        return real_derive(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(gate_module, "_derive_machine_gate_status", replacing_derive)

    with pytest.raises(ValueError, match="无法形成可信门禁检查"):
        build_verified_gate_check("no_indexing", report_path, workspace_root=workspace)


def _annotation(sample_id: str, media_sha256: str, language: str) -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "sample_id": sample_id,
        "media_sha256": media_sha256,
        "duration_ms": 1000,
        "language": language,
        "reference_text": "测试",
        "words": [
            {
                "word_id": f"word_{sample_id}",
                "text": "测试",
                "start_ms": 0,
                "end_ms": 500,
            }
        ],
        "speaker_turns": [
            {
                "turn_id": f"turn_{sample_id}",
                "speaker_id": "speaker_1",
                "start_ms": 0,
                "end_ms": 900,
            }
        ],
        "ocr_frames": [
            {
                "frame_id": f"frame_{sample_id}",
                "timestamp_ms": 100,
                "text_lines": ["测试"],
            }
        ],
        "audio_events": [
            {
                "event_id": f"event_{sample_id}",
                "normalized_event": "speech",
                "start_ms": 0,
                "end_ms": 500,
            }
        ],
        "scene_boundaries_ms": [100],
        "semantic_boundaries_ms": [200],
        "supported_facts": [{"fact_id": f"fact_{sample_id}", "canonical_text": "事实"}],
        "key_fact_ids": [f"fact_{sample_id}"],
        "known_people": [],
    }


def _authorized_dataset_report(
    workspace: Path,
    runtime_root: Path,
    store: EvidenceStore,
    *,
    media_relative_path: str = "media/shared.mp4",
    dataset_root: str = "eval",
) -> tuple[MachineEvidenceReport, Path]:
    eval_root = runtime_root / dataset_root
    media = _write(runtime_root, f"{dataset_root}/{media_relative_path}", b"authorized-media")
    media_sha = _digest(media.read_bytes())
    manifest_lines: list[str] = []
    languages = ("zh", "en", "ja", "ko", "es")
    for index in range(30):
        sample_id = f"sample_{index:02d}"
        language = languages[index % len(languages)]
        annotation_path = _write(
            runtime_root,
            f"{dataset_root}/annotations/{sample_id}.json",
            json.dumps(
                _annotation(sample_id, media_sha, language),
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8"),
        )
        manifest_lines.append(
            json.dumps(
                {
                    "sample_id": sample_id,
                    "language": language,
                    "authorization_id": "auth_001",
                    "media_relative_path": media_relative_path,
                    "media_sha256": media_sha,
                    "annotations_relative_path": f"annotations/{sample_id}.json",
                    "annotations_sha256": _digest(annotation_path.read_bytes()),
                },
                ensure_ascii=False,
            )
        )
    manifest = _write(
        runtime_root,
        f"{dataset_root}/dataset.jsonl",
        "\n".join(manifest_lines).encode("utf-8"),
    )
    authorization = _write(
        runtime_root,
        f"{dataset_root}/authorization.json",
        json.dumps(
            {
                "schema_version": "1.0.0",
                "records": [
                    {
                        "schema_version": "1.0.0",
                        "authorization_id": "auth_001",
                        "source_category": "OWNED",
                        "allowed_purposes": ["VIDEO_QUALITY_EVALUATION"],
                        "confirmed_at": _NOW,
                        "media_sha256": [media_sha],
                    }
                ],
            }
        ).encode("utf-8"),
    )
    stdout = _write(runtime_root, f"{dataset_root}/dataset-stdout.txt", b"validated\n")
    stderr = _write(runtime_root, f"{dataset_root}/dataset-stderr.txt", b"")
    report = MachineEvidenceReport(
        schema_version="1.0.0",
        check_id="authorized_dataset",
        status=GateStatus.PASS,
        kind=EvidenceKind.COMMAND_REPORT,
        level=EvidenceLevel.REAL_MEDIA,
        covered_items=("authorized_dataset",),
        summary="授权五语数据集已重验",
        producer="dataset-runner",
        started_at=_NOW,
        finished_at=_LATER,
        artifacts=(
            store.bind_artifact(manifest.relative_to(runtime_root), "DATASET_MANIFEST"),
            store.bind_artifact(
                authorization.relative_to(runtime_root), "AUTHORIZATION_RECORD"
            ),
            store.bind_artifact(stdout.relative_to(runtime_root), "COMMAND_STDOUT"),
            store.bind_artifact(stderr.relative_to(runtime_root), "COMMAND_STDERR"),
        ),
        details=AuthorizedDatasetDetails(
            type="AUTHORIZED_DATASET",
            trace=CommandTrace(
                command=("python", "-m", "video_demo.evaluation.dataset"),
                exit_code=0,
                stdout_sha256=_digest(stdout.read_bytes()),
                stderr_sha256=_digest(stderr.read_bytes()),
            ),
            manifest_sha256=_digest(manifest.read_bytes()),
            authorization_record_sha256=_digest(authorization.read_bytes()),
            item_count=30,
            language_counts={language: 6 for language in languages},
            max_video_bytes=1024,
        ),
    )
    report_path = eval_root / "reports/authorized.json"
    return report, report_path


def test_authorized_dataset_pass_reloads_manifest_authorization_and_sources(
    tmp_path: Path,
) -> None:
    workspace, runtime_root, store = _roots(tmp_path)
    report, report_path = _authorized_dataset_report(workspace, runtime_root, store)
    store.write_json(report_path.relative_to(runtime_root), report)

    check = build_verified_gate_check(
        "authorized_dataset", report_path, workspace_root=workspace
    )
    assert check.status == GateStatus.PASS

    annotation = runtime_root / "eval/annotations/sample_00.json"
    annotation.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        build_verified_gate_check(
            "authorized_dataset", report_path, workspace_root=workspace
        )


def test_authorized_dataset_rejects_thirty_generated_media_manifest_entries(
    tmp_path: Path,
) -> None:
    workspace, runtime_root, store = _roots(tmp_path)
    report, report_path = _authorized_dataset_report(
        workspace,
        runtime_root,
        store,
        media_relative_path="generated/run-1/authorized.mp4",
    )
    store.write_json(report_path.relative_to(runtime_root), report)

    with pytest.raises(ValueError, match="机器证据无法形成可信门禁检查"):
        build_verified_gate_check(
            "authorized_dataset",
            report_path,
            workspace_root=workspace,
        )


def test_authorized_dataset_rejects_media_resolved_under_generated_manifest_root(
    tmp_path: Path,
) -> None:
    workspace, runtime_root, store = _roots(tmp_path)
    report, report_path = _authorized_dataset_report(
        workspace,
        runtime_root,
        store,
        media_relative_path="authorized.mp4",
        dataset_root="eval/generated/run-1",
    )
    store.write_json(report_path.relative_to(runtime_root), report)

    with pytest.raises(ValueError, match="机器证据无法形成可信门禁检查"):
        build_verified_gate_check(
            "authorized_dataset",
            report_path,
            workspace_root=workspace,
        )


def test_atomic_writer_rejects_copied_dataset_detail_with_bool_limit(
    tmp_path: Path,
) -> None:
    workspace, runtime_root, store = _roots(tmp_path)
    report, _report_path = _authorized_dataset_report(workspace, runtime_root, store)
    forged = report.model_copy(
        update={
            "details": report.details.model_copy(
                update={"max_video_bytes": True}
            )
        }
    )
    target = runtime_root / "eval/reports/bool-dataset-limit.json"

    with pytest.raises(ValueError, match="机器证据原子写入失败"):
        store.write_json(target.relative_to(runtime_root), forged)
    assert not target.exists()


def test_authorized_dataset_rejects_source_replacement_during_strict_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, runtime_root, store = _roots(tmp_path)
    report, report_path = _authorized_dataset_report(workspace, runtime_root, store)
    store.write_json(report_path.relative_to(runtime_root), report)
    annotation = runtime_root / "eval/annotations/sample_00.json"
    from video_demo.evaluation import gate as gate_module

    real_loader = gate_module.load_evaluation_package

    def replacing_loader(*args: object, **kwargs: object) -> object:
        package = real_loader(*args, **kwargs)  # type: ignore[arg-type]
        replacement = annotation.with_suffix(".replacement")
        replacement.write_bytes(annotation.read_bytes())
        os.replace(replacement, annotation)
        return package

    monkeypatch.setattr(gate_module, "load_evaluation_package", replacing_loader)

    with pytest.raises(ValueError, match="无法形成可信门禁检查"):
        build_verified_gate_check(
            "authorized_dataset",
            report_path,
            workspace_root=workspace,
        )


def _performance_report(
    runtime_root: Path,
    store: EvidenceStore,
) -> tuple[MachineEvidenceReport, Path]:
    media_a = _write(runtime_root, "eval/performance/a.mp4", b"sample-a")
    media_b = _write(runtime_root, "eval/performance/b.mp4", b"sample-b")
    samples = (
        PerformanceSampleDetails(
            sample_sha256=_digest(media_a.read_bytes()),
            duration_ms=1_800_000,
            width=1920,
            height=1080,
            elapsed_seconds=3600.0,
            rtf=2.0,
            oom_detected=False,
            peak_concurrency=1,
            outside_workspace_write_count=0,
            peak_rss_bytes=1024,
            peak_disk_bytes=2048,
            succeeded=True,
        ),
        PerformanceSampleDetails(
            sample_sha256=_digest(media_b.read_bytes()),
            duration_ms=1_800_000,
            width=1920,
            height=1080,
            elapsed_seconds=4500.0,
            rtf=2.5,
            oom_detected=False,
            peak_concurrency=1,
            outside_workspace_write_count=0,
            peak_rss_bytes=2048,
            peak_disk_bytes=4096,
            succeeded=True,
        ),
    )
    raw = _write(
        runtime_root,
        "eval/performance/raw.json",
        json.dumps(
            {
                "schema_version": "1.0.0",
                "samples": [sample.model_dump(mode="json") for sample in samples],
            },
            separators=(",", ":"),
        ).encode("utf-8"),
    )
    stdout = _write(runtime_root, "eval/performance/stdout.txt", b"completed\n")
    stderr = _write(runtime_root, "eval/performance/stderr.txt", b"")
    report = MachineEvidenceReport(
        schema_version="1.0.0",
        check_id="m1_durability",
        status=GateStatus.PASS,
        kind=EvidenceKind.PERFORMANCE_REPORT,
        level=EvidenceLevel.PERFORMANCE,
        covered_items=("m1_durability",),
        summary="M1 两段耐久完成",
        producer="durability-runner",
        started_at=_NOW,
        finished_at=_LATER,
        artifacts=(
            store.bind_artifact(
                media_a.relative_to(runtime_root), "INPUT_MEDIA", max_bytes=1024
            ),
            store.bind_artifact(
                media_b.relative_to(runtime_root), "INPUT_MEDIA", max_bytes=1024
            ),
            store.bind_artifact(raw.relative_to(runtime_root), "PERFORMANCE_REPORT"),
            store.bind_artifact(stdout.relative_to(runtime_root), "COMMAND_STDOUT"),
            store.bind_artifact(stderr.relative_to(runtime_root), "COMMAND_STDERR"),
        ),
        details=PerformanceDetails(
            type="PERFORMANCE",
            trace=CommandTrace(
                command=("python", "-m", "video_demo.evaluation.durability"),
                exit_code=0,
                stdout_sha256=_digest(stdout.read_bytes()),
                stderr_sha256=_digest(stderr.read_bytes()),
            ),
            performance_report_sha256=_digest(raw.read_bytes()),
            samples=samples,
        ),
    )
    return report, runtime_root / "eval/reports/durability.json"


def test_m1_durability_recalculates_each_raw_sample_instead_of_summary(
    tmp_path: Path,
) -> None:
    workspace, runtime_root, store = _roots(tmp_path)
    report, report_path = _performance_report(runtime_root, store)
    store.write_json(report_path.relative_to(runtime_root), report)

    with pytest.raises(ValueError, match="无法形成可信门禁检查"):
        build_verified_gate_check(
            "m1_durability", report_path, workspace_root=workspace
        )

    payload = json.loads(
        (runtime_root / "eval/performance/raw.json").read_text(encoding="utf-8")
    )
    payload["samples"][1]["elapsed_seconds"] = 7200.0
    payload["samples"][1]["rtf"] = 2.5
    (runtime_root / "eval/performance/raw.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    forged_report = report.model_copy(
        update={
            "details": report.details.model_copy(
                update={
                    "performance_report_sha256": sha256_file(
                        runtime_root / "eval/performance/raw.json",
                        max_bytes=64 * 1024 * 1024,
                    )
                }
            ),
            "artifacts": tuple(
                artifact.model_copy(
                    update={
                        "sha256": sha256_file(
                            runtime_root / "eval/performance/raw.json",
                            max_bytes=64 * 1024 * 1024,
                        )
                    }
                )
                if artifact.role == "PERFORMANCE_REPORT"
                else artifact
                for artifact in report.artifacts
            ),
        }
    )
    store.write_json(report_path.relative_to(runtime_root), forged_report)
    with pytest.raises(ValueError):
        build_verified_gate_check(
            "m1_durability", report_path, workspace_root=workspace
        )


def test_quality_source_roles_are_bounded_and_do_not_create_a_sixteenth_gate(
    tmp_path: Path,
) -> None:
    _workspace, runtime_root, store = _roots(tmp_path)
    for role in (
        "PREDICTION_INDEX",
        "ANNOTATION",
        "SEMANTIC_JUDGMENT",
        "QUALITY_DETAIL",
    ):
        path = _write(runtime_root, f"eval/quality/{role}.json", b"{}")
        artifact = store.bind_artifact(path.relative_to(runtime_root), role)
        assert artifact.role == role
