from __future__ import annotations

import errno
import hashlib
import importlib.util
import json
import os
import shutil
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest

from video_demo.config import Settings
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.evaluation.evidence import (
    EvidenceStore,
    RealMediaCommand,
    RealMediaFile,
    RealMediaRawReport,
    RealMediaSample,
    TraceArtifact,
    build_verified_gate_check,
    load_machine_evidence,
)
from video_demo.evaluation.gate import _REAL_MEDIA_IMPLEMENTATION_FILES
from video_demo.evaluation.gate import (
    _current_real_media_implementation_sha256 as _unpatched_implementation_digest,
)
from video_demo.evaluation.report import GateStatus
from video_demo.media.process import ProcessResult, SafeProcessRunner


@pytest.fixture(autouse=True)
def _stable_implementation_digest(monkeypatch: pytest.MonkeyPatch) -> None:
    import video_demo.evaluation.gate as gate_module
    import video_demo.evaluation.media_runner as runner_module

    monkeypatch.setattr(
        gate_module, "_current_real_media_implementation_sha256", lambda _root: "a" * 64
    )
    monkeypatch.setattr(
        runner_module, "_current_real_media_implementation_sha256", lambda _root: "a" * 64
    )


def _runner(tmp_path: Path):
    from video_demo.evaluation.media_runner import RealMediaRunner

    runtime = tmp_path / ".codex" / "video-rag-demo"
    runtime.mkdir(parents=True)
    settings = Settings(workspace_root=tmp_path)
    return RealMediaRunner(settings, EvidenceStore(tmp_path, runtime)), runtime


def _complete_dependencies(
    runner: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner, "_resolve_binary", lambda name: tmp_path / name)
    monkeypatch.setattr(importlib.util, "find_spec", lambda _name: object())


def _successful_versions(runner: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        runner,
        "_run_version",
        lambda name, _path: (0, f"{name} version test\n".encode(), b""),
    )


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _rewrite_json(path: Path, value: object) -> None:
    path.write_bytes(_json_bytes(value))


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _rehash_raw(runtime: Path, raw: dict[str, Any], report: dict[str, Any]) -> None:
    encoded = _json_bytes(raw)
    (runtime / "eval/reports/run-1/raw.json").write_bytes(encoded)
    digest = hashlib.sha256(encoded).hexdigest()
    audit = next(item for item in report["artifacts"] if item["role"] == "AUDIT_REPORT")
    audit["sha256"] = digest
    report["details"]["raw_report_sha256"] = digest


def _controlled_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Any, Path, Path]:
    runner, runtime = _runner(tmp_path)
    _complete_dependencies(runner, tmp_path, monkeypatch)
    _successful_versions(runner, monkeypatch)
    assert runner.run(evaluation_run_id="run-1").status == GateStatus.FAIL
    return runner, runtime, runtime / "eval/reports/run-1/real-media.json"


def _artifact_by_suffix(report: dict[str, Any], suffix: str) -> dict[str, Any]:
    return next(
        item for item in report["artifacts"] if item["relative_path"].endswith(suffix)
    )


def _media_bytes(format_name: str) -> bytes:
    if format_name == "MP4":
        return b"\x00\x00\x00\x18ftypisom\x00\x00\x00\x00isomiso2"
    if format_name == "WAV":
        return (
            b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00"
            + b"\x00" * 16
            + b"data\x00\x00\x00\x00"
        )
    return b"\xff\xd8\xff\x00\xff\xd9"


def _write_media_file(
    tmp_path: Path,
    runtime: Path,
    relative_path: str,
    role: str,
    format_name: str,
) -> tuple[RealMediaFile, TraceArtifact]:
    content = _media_bytes(format_name)
    path = tmp_path / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    media_file = RealMediaFile(
        role=role,
        format=format_name,
        relative_path=relative_path,
        sha256=hashlib.sha256(content).hexdigest(),
        size_bytes=len(content),
    )
    artifact = EvidenceStore(tmp_path, runtime).bind_artifact(
        path.relative_to(runtime),
        "INPUT_MEDIA" if role == "SOURCE" else "OUTPUT_MEDIA",
    )
    return media_file, artifact


def _write_command(
    writer: Any,
    case_id: str,
    phase: str,
    executable: str,
    *,
    exit_code: int,
    inputs: tuple[str, ...] = (),
    outputs: tuple[str, ...] = (),
) -> tuple[RealMediaCommand, tuple[TraceArtifact, TraceArtifact]]:
    stdout = writer.write_artifact(
        f"{case_id}-{phase}.stdout.txt", "COMMAND_STDOUT", b""
    )
    stderr = writer.write_artifact(
        f"{case_id}-{phase}.stderr.txt", "COMMAND_STDERR", b""
    )
    return (
        RealMediaCommand(
            phase=phase,
            executable=executable,
            arguments=inputs,
            input_relative_paths=inputs,
            output_relative_paths=outputs,
            exit_code=exit_code,
            stdout_relative_path=stdout.relative_path,
            stderr_relative_path=stderr.relative_path,
            stdout_sha256=stdout.sha256,
            stderr_sha256=stderr.sha256,
        ),
        (stdout, stderr),
    )


_EXECUTABLES = (
    ("generate", "ffmpeg"),
    ("probe", "ffprobe"),
    ("proxy", "FFmpegTranscoder"),
    ("audio", "FFmpegTranscoder"),
    ("opencv_decode", "OpenCvFrameExtractor"),
    ("scene_detect", "PySceneDetectAdapter"),
    ("keyframe_select", "KeyframeSelector"),
)


def _record_successful_case(
    tmp_path: Path,
    runtime: Path,
    journal: Any,
    case_id: str,
    *,
    finalize: bool = True,
) -> tuple[RealMediaSample, tuple[TraceArtifact, ...]]:
    root = f".codex/video-rag-demo/eval/generated/run-1/{case_id}"
    definitions = [
        ("SOURCE", "MP4", f"{root}/source.mp4"),
        ("PROXY", "MP4", f"{root}/media/proxy.mp4"),
        ("KEYFRAME", "JPEG", f"{root}/visual/keyframes/000001.jpg"),
    ]
    if case_id != "no_audio":
        definitions.append(("AUDIO", "WAV", f"{root}/media/audio.wav"))
    media_by_phase = {
        "generate": definitions[0],
        "proxy": definitions[1],
        "keyframe_select": definitions[2],
    }
    if case_id != "no_audio":
        media_by_phase["audio"] = definitions[3]
    files: list[RealMediaFile] = []
    artifacts: list[TraceArtifact] = []
    source = f"{root}/source.mp4"
    proxy = f"{root}/media/proxy.mp4"
    flows = (
        ((), (source,)),
        ((source,), ()),
        ((source,), (proxy,)),
        ((source,), (() if case_id == "no_audio" else (f"{root}/media/audio.wav",))),
        ((proxy,), ()),
        ((proxy,), ()),
        ((proxy,), (f"{root}/visual/keyframes/000001.jpg",)),
    )
    commands: list[RealMediaCommand] = []
    for (phase, executable), (inputs, outputs) in zip(_EXECUTABLES, flows, strict=True):
        journal.begin_phase(
            case_id=case_id,
            phase=phase,
            executable=executable,
            arguments=inputs,
            input_relative_paths=inputs,
            output_relative_paths=outputs,
        )
        definition = media_by_phase.get(phase)
        if definition is not None:
            role, format_name, relative_path = definition
            media_file, artifact = _write_media_file(
                tmp_path, runtime, relative_path, role, format_name
            )
            journal.record_media_file(case_id, media_file, artifact)
            files.append(media_file)
            artifacts.append(artifact)
        command, output_artifacts = _write_command(
            journal,
            case_id,
            phase,
            executable,
            exit_code=0,
            inputs=inputs,
            outputs=outputs,
        )
        journal.record_completed_command(command, output_artifacts)
        commands.append(command)
        artifacts.extend(output_artifacts)
    sample = RealMediaSample(
        case_id=case_id,
        execution_status="SUCCESS",
        duration_ms=1_000,
        has_audio=case_id != "no_audio",
        rotation_degrees=90 if case_id == "rotation" else 0,
        is_variable_frame_rate=case_id == "vfr",
        warnings=("NO_AUDIO_TRACK",) if case_id == "no_audio" else (),
        opencv_decoded_frame_count=1,
        scene_count=1,
        selected_keyframe_count=1,
        files=tuple(files),
        commands=tuple(commands),
    )
    if finalize:
        journal.record_completed_sample(sample, tuple(artifacts))
    return sample, tuple(artifacts)


def test_invalid_run_id_has_no_side_effects(tmp_path: Path) -> None:
    runner, runtime = _runner(tmp_path)

    with pytest.raises(VideoDemoError, match="evaluation_run_id"):
        runner.run(evaluation_run_id="../bad")

    assert not (runtime / "eval").exists()


@pytest.mark.parametrize(
    ("missing", "expected"),
    (
        ("ffmpeg", (ErrorCode.VIDEO_FFMPEG_UNAVAILABLE,)),
        ("ffprobe", (ErrorCode.VIDEO_FFPROBE_UNAVAILABLE,)),
        ("cv2", (ErrorCode.VISUAL_DEPENDENCY_UNAVAILABLE,)),
        ("scenedetect", (ErrorCode.VISUAL_DEPENDENCY_UNAVAILABLE,)),
    ),
)
def test_missing_preflight_dependency_writes_verified_not_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    missing: str,
    expected: tuple[ErrorCode, ...],
) -> None:
    runner, runtime = _runner(tmp_path)

    monkeypatch.setattr(
        runner, "_resolve_binary", lambda name: None if name == missing else tmp_path / name
    )
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name: None if name == missing else object(),
    )
    check = runner.run(evaluation_run_id="run-1")

    assert check.status == GateStatus.NOT_RUN
    report_path = runtime / "eval/reports/run-1/real-media.json"
    report = load_machine_evidence(report_path, workspace_root=tmp_path)
    raw = next(a for a in report.artifacts if a.role == "AUDIT_REPORT")
    raw_text = (tmp_path / raw.relative_path).read_text(encoding="utf-8")
    assert all(code.value in raw_text for code in expected)
    assert not (runtime / "eval/generated/run-1").exists()


def test_visual_dependencies_are_deduplicated_in_stable_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner, runtime = _runner(tmp_path)
    monkeypatch.setattr(runner, "_resolve_binary", lambda _name: None)
    monkeypatch.setattr(importlib.util, "find_spec", lambda _name: None)

    assert runner.run(evaluation_run_id="run-1").status == GateStatus.NOT_RUN
    raw = (runtime / "eval/reports/run-1/preflight.json").read_text(encoding="utf-8")
    assert raw.index("VIDEO_FFMPEG_UNAVAILABLE") < raw.index("VIDEO_FFPROBE_UNAVAILABLE")
    assert raw.count("VISUAL_DEPENDENCY_UNAVAILABLE") == 1


def test_version_failure_forms_setup_fail_without_starting_cases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner, runtime = _runner(tmp_path)
    monkeypatch.setattr(runner, "_resolve_binary", lambda name: tmp_path / name)
    monkeypatch.setattr(importlib.util, "find_spec", lambda _name: object())
    monkeypatch.setattr(
        runner, "_run_version", lambda _name, _path: (1, b"secret", b"/abs/third party body")
    )

    check = runner.run(evaluation_run_id="run-1")

    assert check.status == GateStatus.FAIL
    raw = (runtime / "eval/reports/run-1/raw.json").read_text(encoding="utf-8")
    assert '"phase":"ffmpeg_version"' in raw
    assert '"execution_status":"NOT_STARTED"' in raw
    assert "secret" not in raw and "/abs/" not in raw


def test_version_probe_exception_forms_fail_without_leaking_cause(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner, runtime = _runner(tmp_path)
    monkeypatch.setattr(runner, "_resolve_binary", lambda name: tmp_path / name)
    monkeypatch.setattr(importlib.util, "find_spec", lambda _name: object())

    def raise_probe_error(_name: str, _path: Path) -> tuple[int, bytes, bytes]:
        raise VideoDemoError(ErrorCode.VIDEO_PROCESS_FAILED, "Secret /abs/第三方正文")

    monkeypatch.setattr(runner, "_run_version", raise_probe_error)

    assert runner.run(evaluation_run_id="run-1").status == GateStatus.FAIL
    raw = (runtime / "eval/reports/run-1/raw.json").read_text(encoding="utf-8")
    assert "Secret" not in raw and "/abs/" not in raw and "第三方正文" not in raw


def test_complete_preflight_reaches_controlled_fail_not_fake_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner, runtime = _runner(tmp_path)
    monkeypatch.setattr(runner, "_resolve_binary", lambda name: tmp_path / name)
    monkeypatch.setattr(importlib.util, "find_spec", lambda _name: object())
    monkeypatch.setattr(
        runner,
        "_run_version",
        lambda name, _path: (0, f"{name} version test\n".encode(), b""),
    )

    check = runner.run(evaluation_run_id="run-1")

    assert check.status == GateStatus.FAIL
    report = load_machine_evidence(
        runtime / "eval/reports/run-1/real-media.json", workspace_root=tmp_path
    )
    raw = RealMediaRawReport.model_validate_json(
        (runtime / "eval/reports/run-1/raw.json").read_bytes()
    )
    assert raw.ffmpeg_version == "test" and raw.ffprobe_version == "test"
    assert raw.samples[0].case_id == "normal_audio"
    assert raw.samples[0].execution_status == "FAILED"
    assert tuple(command.phase for command in raw.samples[0].commands) == ("generate",)
    assert all(sample.execution_status == "NOT_STARTED" for sample in raw.samples[1:])
    outputs = [artifact for artifact in report.artifacts if artifact.role.startswith("COMMAND_")]
    assert len({artifact.relative_path for artifact in outputs}) == len(outputs)


def test_typed_late_phase_failure_keeps_success_prefix_and_stops_cases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner, runtime = _runner(tmp_path)
    monkeypatch.setattr(runner, "_resolve_binary", lambda name: tmp_path / name)
    monkeypatch.setattr(importlib.util, "find_spec", lambda _name: object())
    monkeypatch.setattr(
        runner,
        "_run_version",
        lambda name, _path: (0, f"{name} version test\n".encode(), b""),
    )
    from video_demo.evaluation.media_runner import _MediaExecutionFailure

    def execution(
        _run: str,
        _binaries: dict[str, Path],
        journal: Any,
    ) -> _MediaExecutionFailure:
        journal.begin_phase(
            case_id="normal_audio", phase="generate", executable="ffmpeg"
        )
        generate, artifacts = _write_command(
            journal, "normal_audio", "generate", "ffmpeg", exit_code=0
        )
        journal.record_completed_command(generate, artifacts)
        journal.begin_phase(
            case_id="normal_audio", phase="probe", executable="ffprobe"
        )
        return journal.recover_failure()

    monkeypatch.setattr(runner, "_execute_media_port", execution)

    assert runner.run(evaluation_run_id="run-1").status == GateStatus.FAIL
    raw = (runtime / "eval/reports/run-1/raw.json").read_text(encoding="utf-8")
    assert '"phase":"generate"' in raw and '"phase":"probe"' in raw
    assert '"exit_code":0' in raw and '"exit_code":1' in raw
    assert raw.count('"execution_status":"NOT_STARTED"') == 3


def test_report_run_creation_rejects_internal_reports_symlink(tmp_path: Path) -> None:
    runner, runtime = _runner(tmp_path)
    (runtime / "eval").mkdir()
    elsewhere = tmp_path / "inside-workspace-target"
    elsewhere.mkdir()
    (runtime / "eval/reports").symlink_to(elsewhere, target_is_directory=True)

    with pytest.raises(VideoDemoError, match="不完整"):
        runner.run(evaluation_run_id="run-1")

    assert not (elsewhere / "run-1").exists()


def test_existing_complete_report_is_reverified_without_ports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner, _runtime = _runner(tmp_path)
    monkeypatch.setattr(runner, "_resolve_binary", lambda _name: None)
    first = runner.run(evaluation_run_id="run-1")
    assert first.status == GateStatus.NOT_RUN
    monkeypatch.setattr(runner, "_resolve_binary", lambda _name: pytest.fail("不得探测"))

    assert runner.run(evaluation_run_id="run-1") == first


def test_real_media_run_publishes_strict_commit_bound_to_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, runtime = _runner(tmp_path)
    monkeypatch.setattr(runner, "_resolve_binary", lambda _name: None)

    check = runner.run(evaluation_run_id="run-1")

    run = runtime / "eval/reports/run-1"
    authority = run / "real-media.json"
    commit = _load_json(run / ".real-media.commit.json")
    assert commit == {
        "schema_version": "1.0.0",
        "evaluation_run_id": "run-1",
        "authority_sha256": hashlib.sha256(authority.read_bytes()).hexdigest(),
    }
    assert build_verified_gate_check(
        "real_media_chain", authority, workspace_root=tmp_path
    ) == check


def test_real_media_builder_rejects_authority_without_positive_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, runtime = _runner(tmp_path)
    monkeypatch.setattr(runner, "_resolve_binary", lambda _name: None)
    runner.run(evaluation_run_id="run-1")
    report_path = runtime / "eval/reports/run-1/real-media.json"
    (report_path.parent / ".real-media.commit.json").unlink()

    with pytest.raises(ValueError, match="可信门禁"):
        build_verified_gate_check(
            "real_media_chain", report_path, workspace_root=tmp_path
        )


@pytest.mark.parametrize("deletion_timing", ("before", "after"))
def test_staged_verification_requires_same_incomplete_marker_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    deletion_timing: str,
) -> None:
    runner, runtime = _runner(tmp_path)
    monkeypatch.setattr(runner, "_resolve_binary", lambda _name: None)
    import video_demo.evaluation.evidence as evidence_module

    real_build = evidence_module._build_verified_gate_check

    def delete_during_staged(*args: Any, **kwargs: Any) -> object:
        marker = runtime / "eval/reports/run-1/.real-media.incomplete"
        if deletion_timing == "before":
            marker.unlink()
        result = real_build(*args, **kwargs)
        if deletion_timing == "after":
            marker.unlink()
        return result

    monkeypatch.setattr(
        evidence_module,
        "_build_verified_gate_check",
        delete_during_staged,
    )

    with pytest.raises(ValueError, match="原子写入失败"):
        runner.run(evaluation_run_id="run-1")

    run = runtime / "eval/reports/run-1"
    assert not (run / ".real-media.commit.json").exists()
    monkeypatch.setattr(evidence_module, "_build_verified_gate_check", real_build)
    with pytest.raises(ValueError, match="可信门禁"):
        build_verified_gate_check(
            "real_media_chain", run / "real-media.json", workspace_root=tmp_path
        )


def test_incomplete_marker_deleted_after_staged_before_commit_stays_uncommitted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, runtime = _runner(tmp_path)
    monkeypatch.setattr(runner, "_resolve_binary", lambda _name: None)
    import video_demo.evaluation.evidence as evidence_module

    real_assert = evidence_module.ReportRunWriter._assert_incomplete_marker_current
    calls = 0

    def delete_before_second_marker_assert(writer: Any, descriptor: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            (runtime / "eval/reports/run-1/.real-media.incomplete").unlink()
        real_assert(writer, descriptor)

    monkeypatch.setattr(
        evidence_module.ReportRunWriter,
        "_assert_incomplete_marker_current",
        delete_before_second_marker_assert,
    )

    with pytest.raises(ValueError, match="原子写入失败"):
        runner.run(evaluation_run_id="run-1")

    run = runtime / "eval/reports/run-1"
    assert calls == 2
    assert not (run / ".real-media.commit.json").exists()
    with pytest.raises(ValueError, match="可信门禁"):
        build_verified_gate_check(
            "real_media_chain", run / "real-media.json", workspace_root=tmp_path
        )


@pytest.mark.parametrize(
    "mutation",
    ("wrong_run", "wrong_digest", "extra_field", "copied"),
)
def test_real_media_builder_rejects_invalid_or_copied_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    runner, runtime = _runner(tmp_path)
    monkeypatch.setattr(runner, "_resolve_binary", lambda _name: None)
    runner.run(evaluation_run_id="run-1")
    run = runtime / "eval/reports/run-1"
    report_path = run / "real-media.json"
    commit_path = run / ".real-media.commit.json"
    commit = _load_json(commit_path)
    if mutation == "wrong_run":
        commit["evaluation_run_id"] = "run-2"
    elif mutation == "wrong_digest":
        commit["authority_sha256"] = "b" * 64
    elif mutation == "extra_field":
        commit["unexpected"] = True
    else:
        copied_run = runtime / "eval/reports/run-2"
        copied_run.mkdir()
        copied_report = copied_run / "real-media.json"
        copied_report.write_bytes(report_path.read_bytes())
        copied_commit = copied_run / ".real-media.commit.json"
        copied_commit.write_bytes(commit_path.read_bytes())
        report_path = copied_report
        commit_path = copied_commit
    if mutation != "copied":
        _rewrite_json(commit_path, commit)

    with pytest.raises(ValueError, match="可信门禁"):
        build_verified_gate_check(
            "real_media_chain", report_path, workspace_root=tmp_path
        )


def test_real_media_builder_rejects_run_replacement_before_final_identity_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, runtime = _runner(tmp_path)
    monkeypatch.setattr(runner, "_resolve_binary", lambda _name: None)
    runner.run(evaluation_run_id="run-1")
    report_path = runtime / "eval/reports/run-1/real-media.json"
    replacement = tmp_path / "builder-replacement-target"
    replacement.mkdir()
    import video_demo.evaluation.evidence as evidence_module

    real_assert = evidence_module._assert_real_media_commit_current

    def replace_then_assert(*args: Any, **kwargs: Any) -> None:
        _replace_run_with_internal_symlink(runtime, replacement)
        real_assert(*args, **kwargs)

    monkeypatch.setattr(
        evidence_module,
        "_assert_real_media_commit_current",
        replace_then_assert,
    )

    with pytest.raises(ValueError, match="可信门禁"):
        build_verified_gate_check(
            "real_media_chain", report_path, workspace_root=tmp_path
        )


def test_real_media_builder_rejects_run_symlink_to_same_open_inode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, runtime = _runner(tmp_path)
    monkeypatch.setattr(runner, "_resolve_binary", lambda _name: None)
    runner.run(evaluation_run_id="run-1")
    report_path = runtime / "eval/reports/run-1/real-media.json"
    import video_demo.evaluation.evidence as evidence_module

    real_assert = evidence_module._assert_real_media_commit_current

    def replace_with_symlink_to_moved_run(*args: Any, **kwargs: Any) -> None:
        run = runtime / "eval/reports/run-1"
        moved = runtime / "eval/reports/run-1-held-by-fd"
        os.rename(run, moved)
        run.symlink_to(moved, target_is_directory=True)
        real_assert(*args, **kwargs)

    monkeypatch.setattr(
        evidence_module,
        "_assert_real_media_commit_current",
        replace_with_symlink_to_moved_run,
    )

    with pytest.raises(ValueError, match="可信门禁"):
        build_verified_gate_check(
            "real_media_chain", report_path, workspace_root=tmp_path
        )


def test_staged_verification_rejects_run_symlink_to_same_open_inode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, runtime = _runner(tmp_path)
    monkeypatch.setattr(runner, "_resolve_binary", lambda _name: None)
    import video_demo.evaluation.evidence as evidence_module

    real_assert = evidence_module._assert_staged_real_media_run
    staged_calls = 0

    def replace_before_second_staged_assert(*args: Any, **kwargs: Any) -> None:
        nonlocal staged_calls
        staged_calls += 1
        if staged_calls == 2:
            run = runtime / "eval/reports/run-1"
            moved = runtime / "eval/reports/run-1-held-by-fd"
            os.rename(run, moved)
            run.symlink_to(moved, target_is_directory=True)
        real_assert(*args, **kwargs)

    monkeypatch.setattr(
        evidence_module,
        "_assert_staged_real_media_run",
        replace_before_second_staged_assert,
    )

    with pytest.raises(ValueError, match="原子写入失败"):
        runner.run(evaluation_run_id="run-1")

    run = runtime / "eval/reports/run-1"
    assert staged_calls == 2
    assert not (run / ".real-media.commit.json").exists()
    with pytest.raises(ValueError, match="可信门禁"):
        build_verified_gate_check(
            "real_media_chain", run / "real-media.json", workspace_root=tmp_path
        )


def test_commit_publish_failure_before_rename_remains_uncommitted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, runtime = _runner(tmp_path)
    monkeypatch.setattr(runner, "_resolve_binary", lambda _name: None)
    import video_demo.evaluation.evidence as evidence_module

    real_rename = evidence_module.os.rename

    def fail_commit_rename(source: str, destination: str, **kwargs: Any) -> None:
        if destination == ".real-media.commit.json":
            raise OSError("secret-commit-rename")
        real_rename(source, destination, **kwargs)

    monkeypatch.setattr(evidence_module.os, "rename", fail_commit_rename)

    with pytest.raises(ValueError, match="原子写入失败"):
        runner.run(evaluation_run_id="run-1")

    run = runtime / "eval/reports/run-1"
    assert not (run / ".real-media.commit.json").exists()
    with pytest.raises(ValueError, match="可信门禁"):
        build_verified_gate_check(
            "real_media_chain", run / "real-media.json", workspace_root=tmp_path
        )


@pytest.mark.parametrize("post_commit_failure", ("fsync", "cleanup"))
def test_commit_rename_is_linearization_point_for_later_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    post_commit_failure: str,
) -> None:
    runner, runtime = _runner(tmp_path)
    monkeypatch.setattr(runner, "_resolve_binary", lambda _name: None)
    import video_demo.evaluation.evidence as evidence_module

    real_rename = evidence_module.os.rename
    real_fsync = evidence_module.os.fsync
    real_unlink = evidence_module.os.unlink
    committed = False

    def track_commit_rename(source: str, destination: str, **kwargs: Any) -> None:
        nonlocal committed
        real_rename(source, destination, **kwargs)
        if destination == ".real-media.commit.json":
            committed = True

    def fail_after_commit_fsync(descriptor: int) -> None:
        if committed and post_commit_failure == "fsync":
            raise OSError("secret-post-commit-fsync")
        real_fsync(descriptor)

    def fail_incomplete_cleanup(path: str, **kwargs: Any) -> None:
        if path == ".real-media.incomplete" and post_commit_failure == "cleanup":
            raise RuntimeError("secret-incomplete-cleanup")
        real_unlink(path, **kwargs)

    monkeypatch.setattr(evidence_module.os, "rename", track_commit_rename)
    monkeypatch.setattr(evidence_module.os, "fsync", fail_after_commit_fsync)
    monkeypatch.setattr(evidence_module.os, "unlink", fail_incomplete_cleanup)

    first = runner.run(evaluation_run_id="run-1")

    assert first.status == GateStatus.NOT_RUN
    report_path = runtime / "eval/reports/run-1/real-media.json"
    assert build_verified_gate_check(
        "real_media_chain", report_path, workspace_root=tmp_path
    ) == first
    assert runner.run(evaluation_run_id="run-1") == first


@pytest.mark.parametrize("post_commit_step", ("cleanup", "fsync"))
@pytest.mark.parametrize("termination", (KeyboardInterrupt, SystemExit))
def test_commit_post_processing_propagates_termination_without_rollback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    post_commit_step: str,
    termination: type[BaseException],
) -> None:
    runner, runtime = _runner(tmp_path)
    monkeypatch.setattr(runner, "_resolve_binary", lambda _name: None)
    import video_demo.evaluation.evidence as evidence_module

    real_rename = evidence_module.os.rename
    real_fsync = evidence_module.os.fsync
    real_unlink = evidence_module.os.unlink
    committed = False
    post_commit_fsync_calls = 0

    def track_commit_rename(source: str, destination: str, **kwargs: Any) -> None:
        nonlocal committed
        real_rename(source, destination, **kwargs)
        if destination == ".real-media.commit.json":
            committed = True

    def terminate_after_commit_fsync(descriptor: int) -> None:
        nonlocal post_commit_fsync_calls
        if committed:
            post_commit_fsync_calls += 1
            if post_commit_step == "fsync" and post_commit_fsync_calls == 2:
                raise termination("post-commit-fsync")
        real_fsync(descriptor)

    def terminate_incomplete_cleanup(path: str, **kwargs: Any) -> None:
        if path == ".real-media.incomplete" and post_commit_step == "cleanup":
            raise termination("post-commit-cleanup")
        real_unlink(path, **kwargs)

    monkeypatch.setattr(evidence_module.os, "rename", track_commit_rename)
    monkeypatch.setattr(evidence_module.os, "fsync", terminate_after_commit_fsync)
    monkeypatch.setattr(evidence_module.os, "unlink", terminate_incomplete_cleanup)

    with pytest.raises(termination):
        runner.run(evaluation_run_id="run-1")

    run = runtime / "eval/reports/run-1"
    report_path = run / "real-media.json"
    assert committed
    assert (run / ".real-media.commit.json").is_file()
    monkeypatch.setattr(evidence_module.os, "fsync", real_fsync)
    monkeypatch.setattr(evidence_module.os, "unlink", real_unlink)
    committed_check = build_verified_gate_check(
        "real_media_chain", report_path, workspace_root=tmp_path
    )
    assert committed_check.status == GateStatus.NOT_RUN
    assert runner.run(evaluation_run_id="run-1") == committed_check


def test_real_media_builder_accepts_commit_despite_diagnostic_incomplete_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, runtime = _runner(tmp_path)
    monkeypatch.setattr(runner, "_resolve_binary", lambda _name: None)
    runner.run(evaluation_run_id="run-1")
    report_path = runtime / "eval/reports/run-1/real-media.json"
    (report_path.parent / ".real-media.incomplete").write_bytes(b"")

    assert build_verified_gate_check(
        "real_media_chain", report_path, workspace_root=tmp_path
    ).status == GateStatus.NOT_RUN


def test_staged_authority_verification_failure_keeps_run_uncommitted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, runtime = _runner(tmp_path)
    monkeypatch.setattr(runner, "_resolve_binary", lambda _name: None)
    import video_demo.evaluation.evidence as evidence_module

    real_internal_build = evidence_module._build_verified_gate_check
    staged_calls = 0

    def fail_staged_verification(*_args: Any, **_kwargs: Any) -> object:
        nonlocal staged_calls
        staged_calls += 1
        raise OSError("secret-transient-staged-verification")

    monkeypatch.setattr(
        evidence_module,
        "_build_verified_gate_check",
        fail_staged_verification,
        raising=False,
    )
    with pytest.raises(ValueError) as captured:
        runner.run(evaluation_run_id="run-1")

    assert captured.value.__cause__ is None
    assert "secret-transient" not in str(captured.value)
    run = runtime / "eval/reports/run-1"
    assert (run / "real-media.json").is_file()
    assert (run / ".real-media.incomplete").is_file()
    assert staged_calls == 1

    monkeypatch.setattr(
        evidence_module,
        "_build_verified_gate_check",
        lambda check_id, report_path, *, workspace_root, **_kwargs: real_internal_build(
            check_id,
            report_path,
            workspace_root=workspace_root,
            allow_incomplete_real_media_run=False,
        ),
    )
    with pytest.raises(ValueError, match="可信门禁"):
        runner.run(evaluation_run_id="run-1")

    assert staged_calls == 1


@pytest.mark.parametrize("cleanup_error", (OSError, RuntimeError))
def test_authority_publish_and_cleanup_failure_keeps_incomplete_run_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cleanup_error: type[Exception],
) -> None:
    runner, runtime = _runner(tmp_path)
    monkeypatch.setattr(runner, "_resolve_binary", lambda _name: None)
    import video_demo.evaluation.evidence as evidence_module

    real_fsync = evidence_module.os.fsync
    real_rename = evidence_module.os.rename
    real_unlink = evidence_module.os.unlink
    authority_published = False

    def track_authority_rename(
        source: str,
        destination: str,
        **kwargs: Any,
    ) -> None:
        nonlocal authority_published
        real_rename(source, destination, **kwargs)
        if destination == "real-media.json":
            authority_published = True

    def fail_post_publish_fsync(descriptor: int) -> None:
        if authority_published:
            raise OSError("secret-publish-fsync")
        real_fsync(descriptor)

    def fail_authority_cleanup(path: str, **kwargs: Any) -> None:
        if path == "real-media.json":
            raise cleanup_error("secret-authority-unlink")
        real_unlink(path, **kwargs)

    monkeypatch.setattr(evidence_module.os, "rename", track_authority_rename)
    monkeypatch.setattr(evidence_module.os, "fsync", fail_post_publish_fsync)
    monkeypatch.setattr(evidence_module.os, "unlink", fail_authority_cleanup)

    with pytest.raises(ValueError) as captured:
        runner.run(evaluation_run_id="run-1")

    assert captured.value.__cause__ is None
    assert "secret-publish-fsync" not in str(captured.value)
    assert "secret-authority-unlink" not in str(captured.value)
    run = runtime / "eval/reports/run-1"
    assert (run / "real-media.json").is_file()
    assert (run / ".real-media.incomplete").is_file()

    with pytest.raises(ValueError, match="可信门禁"):
        runner.run(evaluation_run_id="run-1")


def test_preflight_implementation_drift_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner, _runtime = _runner(tmp_path)
    monkeypatch.setattr(runner, "_resolve_binary", lambda _name: None)
    assert runner.run(evaluation_run_id="run-1").status == GateStatus.NOT_RUN
    import video_demo.evaluation.gate as gate_module

    monkeypatch.setattr(
        gate_module, "_current_real_media_implementation_sha256", lambda _root: "b" * 64
    )
    with pytest.raises(ValueError, match="可信门禁"):
        runner.run(evaluation_run_id="run-1")


def test_residual_run_directory_fails_closed_without_overwrite(tmp_path: Path) -> None:
    runner, runtime = _runner(tmp_path)
    residual = runtime / "eval/reports/run-1"
    residual.mkdir(parents=True)
    marker = residual / "marker.txt"
    marker.write_text("保留", encoding="utf-8")

    with pytest.raises(VideoDemoError, match="不完整"):
        runner.run(evaluation_run_id="run-1")

    assert marker.read_text(encoding="utf-8") == "保留"


def test_run_directory_is_created_exclusively(tmp_path: Path) -> None:
    runner, runtime = _runner(tmp_path)
    reports = runtime / "eval/reports"
    reports.mkdir(parents=True)
    (reports / "run-1").mkdir()

    with pytest.raises(VideoDemoError, match="不完整"):
        runner.run(evaluation_run_id="run-1")


def test_unsafe_successful_version_line_fails_without_persisting_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner, runtime = _runner(tmp_path)
    monkeypatch.setattr(runner, "_resolve_binary", lambda name: tmp_path / name)
    monkeypatch.setattr(importlib.util, "find_spec", lambda _name: object())
    monkeypatch.setattr(
        runner,
        "_run_version",
        lambda _name, _path: (0, b"ffmpeg version /abs Secret\\n", b""),
    )

    assert runner.run(evaluation_run_id="run-1").status == GateStatus.FAIL
    raw = (runtime / "eval/reports/run-1/raw.json").read_text(encoding="utf-8")
    assert "/abs" not in raw and "Secret" not in raw


@pytest.mark.parametrize(
    "mutation",
    (
        "missing_ffmpeg_setup",
        "missing_ffprobe_setup",
        "reversed_setup",
        "nonzero_setup_with_media_case",
        "empty_arguments",
        "duplicate_version_argument",
        "raw_run_id_mismatch",
        "setup_output_cross_run",
        "shared_setup_output",
        "setup_reuses_trace_output",
        "setup_reuses_media_output",
    ),
)
def test_public_builder_rejects_invalid_setup_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    _runner_value, runtime, report_path = _controlled_report(tmp_path, monkeypatch)
    raw_path = runtime / "eval/reports/run-1/raw.json"
    raw = _load_json(raw_path)
    report = _load_json(report_path)
    setup = raw["setup_commands"]
    if mutation == "missing_ffmpeg_setup":
        raw["setup_commands"] = setup[1:]
    elif mutation == "missing_ffprobe_setup":
        raw["setup_commands"] = setup[:1]
    elif mutation == "reversed_setup":
        raw["setup_commands"] = list(reversed(setup))
    elif mutation == "nonzero_setup_with_media_case":
        setup[1]["exit_code"] = 1
    elif mutation == "empty_arguments":
        setup[0]["arguments"] = []
    elif mutation == "duplicate_version_argument":
        setup[0]["arguments"] = ["-version", "-version"]
    elif mutation == "raw_run_id_mismatch":
        raw["evaluation_run_id"] = "run-2"
        for command in setup:
            command["stdout_relative_path"] = command["stdout_relative_path"].replace(
                "/run-1/", "/run-2/"
            )
            command["stderr_relative_path"] = command["stderr_relative_path"].replace(
                "/run-1/", "/run-2/"
            )
    elif mutation == "setup_output_cross_run":
        setup[0]["stdout_relative_path"] = setup[0]["stdout_relative_path"].replace(
            "/run-1/", "/run-2/"
        )
    elif mutation == "shared_setup_output":
        setup[1]["stdout_relative_path"] = setup[0]["stdout_relative_path"]
        setup[1]["stdout_sha256"] = setup[0]["stdout_sha256"]
    elif mutation == "setup_reuses_trace_output":
        trace = _artifact_by_suffix(report, "trace.stdout.txt")
        setup[0]["stdout_relative_path"] = trace["relative_path"]
        setup[0]["stdout_sha256"] = trace["sha256"]
    else:
        media = _artifact_by_suffix(report, "normal_audio-generate.stdout.txt")
        setup[0]["stdout_relative_path"] = media["relative_path"]
        setup[0]["stdout_sha256"] = media["sha256"]
    _rehash_raw(runtime, raw, report)
    _rewrite_json(report_path, report)

    with pytest.raises(ValueError, match="可信门禁"):
        build_verified_gate_check("real_media_chain", report_path, workspace_root=tmp_path)


def test_real_newline_version_first_line_is_accepted_before_controlled_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner, runtime = _runner(tmp_path)
    _complete_dependencies(runner, tmp_path, monkeypatch)
    monkeypatch.setattr(
        runner,
        "_run_version",
        lambda name, _path: (
            0,
            f"{name} version 7.1.2\nCopyright third-party body\n".encode(),
            b"ignored stderr",
        ),
    )

    assert runner.run(evaluation_run_id="run-1").status == GateStatus.FAIL
    raw = RealMediaRawReport.model_validate_json(
        (runtime / "eval/reports/run-1/raw.json").read_bytes()
    )
    assert raw.ffmpeg_version == "7.1.2"
    assert raw.ffprobe_version == "7.1.2"
    persisted = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (runtime / "eval/reports/run-1").iterdir()
    )
    assert "Copyright third-party body" not in persisted
    assert "ignored stderr" not in persisted


@pytest.mark.parametrize(
    ("payload", "forbidden"),
    (
        (b"ffprobe version 7.1\n", ("ffprobe version 7.1",)),
        (b"provider quota exceeded; retry later\n", ("provider quota exceeded",)),
        (b"Authorization: Basic abc Bearer token\n", ("Authorization", "Bearer", "token")),
        (b"ffmpeg version /private/tmp/tool\n", ("/private/tmp/tool",)),
        (b"ffmpeg version C:\\Users\\name\\tool.exe\n", ("C:\\Users", "tool.exe")),
        (b"ffmpeg version \\\\server\\share\\tool.exe\n", ("server", "share", "tool.exe")),
        (b"ffmpeg version data:text/plain;base64,QUJD\n", ("data:text", "QUJD")),
        (b"ffmpeg version \xff\xfe\n", ()),
    ),
)
def test_invalid_version_output_forms_reverifiable_fail_without_body_leak(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    payload: bytes,
    forbidden: tuple[str, ...],
) -> None:
    runner, runtime = _runner(tmp_path)
    _complete_dependencies(runner, tmp_path, monkeypatch)
    monkeypatch.setattr(runner, "_run_version", lambda _name, _path: (0, payload, payload))

    first = runner.run(evaluation_run_id="run-1")
    assert first.status == GateStatus.FAIL
    raw = RealMediaRawReport.model_validate_json(
        (runtime / "eval/reports/run-1/raw.json").read_bytes()
    )
    assert raw.failure_code == ErrorCode.VIDEO_BINARY_PROBE_FAILED
    assert all(sample.execution_status == "NOT_STARTED" for sample in raw.samples)
    report_path = runtime / "eval/reports/run-1/real-media.json"
    assert build_verified_gate_check(
        "real_media_chain", report_path, workspace_root=tmp_path
    ) == first
    persisted = b"\n".join(
        path.read_bytes() for path in (runtime / "eval/reports/run-1").iterdir()
    )
    assert payload not in persisted
    decoded = persisted.decode("utf-8")
    assert all(fragment not in decoded for fragment in forbidden)
    monkeypatch.setattr(runner, "_run_version", lambda *_args: pytest.fail("不得再次探测"))
    assert runner.run(evaluation_run_id="run-1") == first


@pytest.mark.parametrize(
    "mutation",
    (
        "raw_tamper",
        "detail_tamper",
        "artifact_tamper",
        "extra_stdout",
        "extra_stderr",
        "cross_run_raw",
        "cross_run_trace",
        "same_digest_different_path",
        "trace_output_identity_reuse",
    ),
)
def test_public_builder_rejects_unclosed_real_media_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    runner, runtime = _runner(tmp_path)
    monkeypatch.setattr(runner, "_resolve_binary", lambda _name: None)
    assert runner.run(evaluation_run_id="run-1").status == GateStatus.NOT_RUN
    report_path = runtime / "eval/reports/run-1/real-media.json"
    raw_path = runtime / "eval/reports/run-1/preflight.json"
    raw = _load_json(raw_path)
    report = _load_json(report_path)
    if mutation == "raw_tamper":
        raw["reason_code"] = "REAL_MEDIA_CHAIN_UNAVAILABLE "
        _rewrite_json(raw_path, raw)
    elif mutation == "detail_tamper":
        report["details"]["preflight_report_sha256"] = "0" * 64
        _rewrite_json(report_path, report)
    elif mutation == "artifact_tamper":
        raw_path.write_bytes(raw_path.read_bytes() + b" ")
    elif mutation in {"extra_stdout", "extra_stderr"}:
        role = "COMMAND_STDOUT" if mutation == "extra_stdout" else "COMMAND_STDERR"
        suffix = "extra.stdout.txt" if mutation == "extra_stdout" else "extra.stderr.txt"
        artifact = EvidenceStore(tmp_path, runtime).write_artifact(
            Path("eval/reports/run-1") / suffix, role, b""
        )
        report["artifacts"].append(artifact.model_dump(mode="json"))
        _rewrite_json(report_path, report)
    elif mutation == "cross_run_raw":
        raw["evaluation_run_id"] = "run-2"
        _rewrite_json(raw_path, raw)
        digest = hashlib.sha256(raw_path.read_bytes()).hexdigest()
        report["details"]["preflight_report_sha256"] = digest
        _artifact_by_suffix(report, "preflight.json")["sha256"] = digest
        _rewrite_json(report_path, report)
    elif mutation == "cross_run_trace":
        trace = _artifact_by_suffix(report, "trace.stdout.txt")
        duplicate = runtime / "eval/reports/run-2/trace.stdout.txt"
        duplicate.parent.mkdir(parents=True)
        duplicate.write_bytes(b"")
        trace["relative_path"] = duplicate.relative_to(tmp_path).as_posix()
        _rewrite_json(report_path, report)
    elif mutation == "same_digest_different_path":
        original = _artifact_by_suffix(report, "trace.stdout.txt")
        duplicate = runtime / "eval/reports/run-1/trace-copy.stdout.txt"
        duplicate.write_bytes(b"")
        copied = dict(original)
        copied["relative_path"] = duplicate.relative_to(tmp_path).as_posix()
        report["artifacts"].append(copied)
        _rewrite_json(report_path, report)
    else:
        stdout = _artifact_by_suffix(report, "trace.stdout.txt")
        stderr = _artifact_by_suffix(report, "trace.stderr.txt")
        stderr["relative_path"] = stdout["relative_path"]
        stderr["sha256"] = stdout["sha256"]
        report["details"]["trace"]["stderr_sha256"] = stdout["sha256"]
        _rewrite_json(report_path, report)

    with pytest.raises(ValueError, match="可信门禁"):
        build_verified_gate_check("real_media_chain", report_path, workspace_root=tmp_path)


def test_task4_success_result_dispatches_to_shared_persistence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner, _runtime = _runner(tmp_path)
    _complete_dependencies(runner, tmp_path, monkeypatch)
    _successful_versions(runner, monkeypatch)
    from video_demo.evaluation.media_runner import _MediaExecutionSuccess

    success = _MediaExecutionSuccess(samples=(), artifacts=())
    expected = object()
    monkeypatch.setattr(runner, "_execute_media_port", lambda *_args: success)
    received: list[_MediaExecutionSuccess] = []

    def persist(*args: Any) -> object:
        received.append(args[-1])
        return expected

    monkeypatch.setattr(runner, "_persist_media_execution", persist)

    assert runner.run(evaluation_run_id="run-1") is expected
    assert received == [success]


def test_task4_no_audio_partial_failure_keeps_real_samples_and_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner, runtime = _runner(tmp_path)
    _complete_dependencies(runner, tmp_path, monkeypatch)
    _successful_versions(runner, monkeypatch)
    from video_demo.evaluation.media_runner import _MediaExecutionFailure

    def execute(
        _run: str,
        _binaries: dict[str, Path],
        journal: Any,
    ) -> _MediaExecutionFailure:
        _record_successful_case(tmp_path, runtime, journal, "normal_audio")
        journal.begin_phase(
            case_id="no_audio", phase="generate", executable="ffmpeg"
        )
        generate, generate_artifacts = _write_command(
            journal, "no_audio", "generate", "ffmpeg", exit_code=0
        )
        journal.record_completed_command(generate, generate_artifacts)
        journal.begin_phase(
            case_id="no_audio", phase="probe", executable="ffprobe"
        )
        return journal.recover_failure()

    monkeypatch.setattr(runner, "_execute_media_port", execute)

    check = runner.run(evaluation_run_id="run-1")

    assert check.status == GateStatus.FAIL
    raw = RealMediaRawReport.model_validate_json(
        (runtime / "eval/reports/run-1/raw.json").read_bytes()
    )
    assert tuple(sample.execution_status for sample in raw.samples) == (
        "SUCCESS",
        "FAILED",
        "NOT_STARTED",
        "NOT_STARTED",
    )
    assert tuple(command.phase for command in raw.samples[1].commands) == (
        "generate",
        "probe",
    )
    report_path = runtime / "eval/reports/run-1/real-media.json"
    assert build_verified_gate_check(
        "real_media_chain", report_path, workspace_root=tmp_path
    ) == check


@pytest.mark.parametrize("exception_type", (ValueError, RuntimeError))
def test_media_port_exception_is_reverifiable_without_body_or_cause_leak(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    exception_type: type[Exception],
) -> None:
    runner, runtime = _runner(tmp_path)
    _complete_dependencies(runner, tmp_path, monkeypatch)
    _successful_versions(runner, monkeypatch)

    def raise_sensitive(*_args: object) -> object:
        try:
            raise ValueError("Secret /private/cause data:text/plain,third-party")
        except ValueError as cause:
            raise exception_type("Bearer token C:\\Users\\third-party") from cause

    monkeypatch.setattr(runner, "_execute_media_port", raise_sensitive)

    first = runner.run(evaluation_run_id="run-1")

    assert first.status == GateStatus.FAIL
    report_path = runtime / "eval/reports/run-1/real-media.json"
    assert build_verified_gate_check(
        "real_media_chain", report_path, workspace_root=tmp_path
    ) == first
    persisted = b"\n".join(
        path.read_bytes() for path in (runtime / "eval/reports/run-1").iterdir()
    ).decode("utf-8")
    assert all(
        value not in persisted
        for value in ("Secret", "/private", "data:text", "Bearer", "C:\\Users", "third-party")
    )
    monkeypatch.setattr(runner, "_execute_media_port", lambda *_args: pytest.fail("不得重跑"))
    assert runner.run(evaluation_run_id="run-1") == first


@pytest.mark.parametrize("failed_phase", ("generate", "probe"))
def test_media_port_exception_recovers_registered_partial_journal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failed_phase: str,
) -> None:
    runner, runtime = _runner(tmp_path)
    _complete_dependencies(runner, tmp_path, monkeypatch)
    _successful_versions(runner, monkeypatch)

    def execute(
        _run: str,
        _binaries: dict[str, Path],
        journal: Any,
    ) -> object:
        _record_successful_case(tmp_path, runtime, journal, "normal_audio")
        if failed_phase == "probe":
            journal.begin_phase(
                case_id="no_audio",
                phase="generate",
                executable="ffmpeg",
            )
            generate, generate_artifacts = _write_command(
                journal,
                "no_audio",
                "generate",
                "ffmpeg",
                exit_code=0,
            )
            journal.record_completed_command(generate, generate_artifacts)
        journal.begin_phase(
            case_id="no_audio",
            phase=failed_phase,
            executable="ffmpeg" if failed_phase == "generate" else "ffprobe",
        )
        if failed_phase == "probe":
            journal.write_current_outputs()
        raise RuntimeError("Secret /private/later data:text/plain,third-party Bearer token")

    monkeypatch.setattr(runner, "_execute_media_port", execute)

    check = runner.run(evaluation_run_id="run-1")

    assert check.status == GateStatus.FAIL
    raw = RealMediaRawReport.model_validate_json(
        (runtime / "eval/reports/run-1/raw.json").read_bytes()
    )
    assert tuple(sample.execution_status for sample in raw.samples) == (
        "SUCCESS",
        "FAILED",
        "NOT_STARTED",
        "NOT_STARTED",
    )
    expected_phases = ("generate",) if failed_phase == "generate" else ("generate", "probe")
    assert tuple(command.phase for command in raw.samples[1].commands) == expected_phases
    assert raw.samples[1].commands[-1].exit_code != 0
    report_path = runtime / "eval/reports/run-1/real-media.json"
    report = load_machine_evidence(report_path, workspace_root=tmp_path)
    listed = {artifact.relative_path for artifact in report.artifacts}
    report_run_files = {
        path.relative_to(tmp_path).as_posix()
        for path in (runtime / "eval/reports/run-1").iterdir()
        if path.name not in {"real-media.json", ".real-media.commit.json"}
    }
    assert report_run_files == {
        relative_path
        for relative_path in listed
        if relative_path.startswith(
            ".codex/video-rag-demo/eval/reports/run-1/"
        )
    }
    assert all((tmp_path / relative_path).is_file() for relative_path in listed)
    assert build_verified_gate_check(
        "real_media_chain", report_path, workspace_root=tmp_path
    ) == check
    persisted = b"\n".join(
        path.read_bytes() for path in (runtime / "eval/reports/run-1").iterdir()
    ).decode("utf-8")
    assert all(
        text not in persisted
        for text in ("Secret", "/private", "data:text", "third-party", "Bearer", "token")
    )


def test_media_port_exception_between_phases_fails_the_next_expected_phase(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, runtime = _runner(tmp_path)
    _complete_dependencies(runner, tmp_path, monkeypatch)
    _successful_versions(runner, monkeypatch)

    def execute(
        _run: str,
        _binaries: dict[str, Path],
        journal: Any,
    ) -> object:
        _record_successful_case(tmp_path, runtime, journal, "normal_audio")
        journal.begin_phase(
            case_id="no_audio",
            phase="generate",
            executable="ffmpeg",
        )
        generate, generate_artifacts = _write_command(
            journal,
            "no_audio",
            "generate",
            "ffmpeg",
            exit_code=0,
        )
        journal.record_completed_command(generate, generate_artifacts)
        raise RuntimeError("Secret between generate and probe")

    monkeypatch.setattr(runner, "_execute_media_port", execute)

    assert runner.run(evaluation_run_id="run-1").status == GateStatus.FAIL
    raw = RealMediaRawReport.model_validate_json(
        (runtime / "eval/reports/run-1/raw.json").read_bytes()
    )
    failed = raw.samples[1]
    assert tuple(command.phase for command in failed.commands) == (
        "generate",
        "probe",
    )
    assert tuple(command.exit_code for command in failed.commands) == (0, 1)


def test_media_port_exception_during_sample_finalization_preserves_all_commands(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, runtime = _runner(tmp_path)
    _complete_dependencies(runner, tmp_path, monkeypatch)
    _successful_versions(runner, monkeypatch)

    def execute(
        _run: str,
        _binaries: dict[str, Path],
        journal: Any,
    ) -> object:
        for phase, executable in _EXECUTABLES:
            journal.begin_phase(
                case_id="normal_audio",
                phase=phase,
                executable=executable,
            )
            command, artifacts = _write_command(
                journal,
                "normal_audio",
                phase,
                executable,
                exit_code=0,
            )
            journal.record_completed_command(command, artifacts)
        raise RuntimeError("Secret during sample finalization")

    monkeypatch.setattr(runner, "_execute_media_port", execute)

    assert runner.run(evaluation_run_id="run-1").status == GateStatus.FAIL
    raw = RealMediaRawReport.model_validate_json(
        (runtime / "eval/reports/run-1/raw.json").read_bytes()
    )
    failed = raw.samples[0]
    assert failed.execution_status == "FAILED"
    assert tuple(command.phase for command in failed.commands) == tuple(
        phase for phase, _executable in _EXECUTABLES
    )
    assert all(command.exit_code == 0 for command in failed.commands)
    assert all(sample.execution_status == "NOT_STARTED" for sample in raw.samples[1:])


def test_media_port_exception_preserves_media_registered_during_active_phase(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, runtime = _runner(tmp_path)
    _complete_dependencies(runner, tmp_path, monkeypatch)
    _successful_versions(runner, monkeypatch)

    def execute(
        _run: str,
        _binaries: dict[str, Path],
        journal: Any,
    ) -> object:
        relative_path = (
            ".codex/video-rag-demo/eval/generated/run-1/normal_audio/source.mp4"
        )
        journal.begin_phase(
            case_id="normal_audio",
            phase="generate",
            executable="ffmpeg",
            output_relative_paths=(relative_path,),
        )
        media_file, artifact = _write_media_file(
            tmp_path, runtime, relative_path, "SOURCE", "MP4"
        )
        journal.record_media_file("normal_audio", media_file, artifact)
        raise RuntimeError("Secret after generated media")

    monkeypatch.setattr(runner, "_execute_media_port", execute)

    check = runner.run(evaluation_run_id="run-1")

    assert check.status == GateStatus.FAIL
    raw = RealMediaRawReport.model_validate_json(
        (runtime / "eval/reports/run-1/raw.json").read_bytes()
    )
    assert raw.samples[0].files == (
        RealMediaFile(
            role="SOURCE",
            format="MP4",
            relative_path=(
                ".codex/video-rag-demo/eval/generated/run-1/normal_audio/source.mp4"
            ),
            sha256=hashlib.sha256(_media_bytes("MP4")).hexdigest(),
            size_bytes=len(_media_bytes("MP4")),
        ),
    )
    report_path = runtime / "eval/reports/run-1/real-media.json"
    report = load_machine_evidence(report_path, workspace_root=tmp_path)
    declared_media = {
        artifact.relative_path
        for artifact in report.artifacts
        if artifact.role in {"INPUT_MEDIA", "OUTPUT_MEDIA"}
    }
    assert declared_media == {raw.samples[0].files[0].relative_path}
    assert build_verified_gate_check(
        "real_media_chain", report_path, workspace_root=tmp_path
    ) == check


def test_media_port_sample_finalization_preserves_registered_media_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, runtime = _runner(tmp_path)
    _complete_dependencies(runner, tmp_path, monkeypatch)
    _successful_versions(runner, monkeypatch)

    def execute(
        _run: str,
        _binaries: dict[str, Path],
        journal: Any,
    ) -> object:
        _record_successful_case(
            tmp_path,
            runtime,
            journal,
            "normal_audio",
            finalize=False,
        )
        raise RuntimeError("Secret while building completed sample")

    monkeypatch.setattr(runner, "_execute_media_port", execute)

    check = runner.run(evaluation_run_id="run-1")

    assert check.status == GateStatus.FAIL
    raw = RealMediaRawReport.model_validate_json(
        (runtime / "eval/reports/run-1/raw.json").read_bytes()
    )
    failed = raw.samples[0]
    assert failed.execution_status == "FAILED"
    assert tuple(media.role for media in failed.files) == (
        "SOURCE",
        "PROXY",
        "AUDIO",
        "KEYFRAME",
    )
    report_path = runtime / "eval/reports/run-1/real-media.json"
    report = load_machine_evidence(report_path, workspace_root=tmp_path)
    declared_media = {
        artifact.relative_path
        for artifact in report.artifacts
        if artifact.role in {"INPUT_MEDIA", "OUTPUT_MEDIA"}
    }
    assert declared_media == {media.relative_path for media in failed.files}
    assert build_verified_gate_check(
        "real_media_chain", report_path, workspace_root=tmp_path
    ) == check


def test_media_port_exception_after_all_samples_preserves_four_case_facts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, runtime = _runner(tmp_path)
    _complete_dependencies(runner, tmp_path, monkeypatch)
    _successful_versions(runner, monkeypatch)

    def execute(
        _run: str,
        _binaries: dict[str, Path],
        journal: Any,
    ) -> object:
        for case_id in ("normal_audio", "no_audio", "rotation", "vfr"):
            _record_successful_case(tmp_path, runtime, journal, case_id)
        raise RuntimeError("Secret before returning success result")

    monkeypatch.setattr(runner, "_execute_media_port", execute)

    check = runner.run(evaluation_run_id="run-1")

    assert check.status == GateStatus.FAIL
    raw = RealMediaRawReport.model_validate_json(
        (runtime / "eval/reports/run-1/raw.json").read_bytes()
    )
    assert tuple(sample.execution_status for sample in raw.samples) == (
        "SUCCESS",
        "SUCCESS",
        "SUCCESS",
        "FAILED",
    )
    failed = raw.samples[-1]
    assert failed.case_id == "vfr"
    assert len(failed.files) == 4
    assert len(failed.commands) == len(_EXECUTABLES)
    assert failed.duration_ms == 1_000
    assert failed.is_variable_frame_rate is True
    assert build_verified_gate_check(
        "real_media_chain",
        runtime / "eval/reports/run-1/real-media.json",
        workspace_root=tmp_path,
    ) == check


@pytest.mark.parametrize(
    ("case_id", "phase", "executable"),
    (
        ("no_audio", "generate", "ffmpeg"),
        ("normal_audio", "probe", "ffprobe"),
        ("normal_audio", "generate", "ffprobe"),
    ),
)
def test_media_journal_rejects_invalid_case_phase_order(
    tmp_path: Path,
    case_id: str,
    phase: str,
    executable: str,
) -> None:
    from video_demo.evaluation.media_runner import _MediaExecutionJournal

    _workspace, runtime, store = tmp_path, tmp_path / ".codex/video-rag-demo", None
    runtime.mkdir(parents=True)
    store = EvidenceStore(tmp_path, runtime)
    writer = store.open_exclusive_report_run("run-1")
    try:
        journal = _MediaExecutionJournal(writer)

        with pytest.raises(VideoDemoError, match="顺序"):
            journal.begin_phase(
                case_id=case_id,
                phase=phase,
                executable=executable,
            )
    finally:
        writer.close()


def test_media_journal_rejects_mismatched_or_duplicate_media_fact(
    tmp_path: Path,
) -> None:
    from video_demo.evaluation.media_runner import _MediaExecutionJournal

    runtime = tmp_path / ".codex/video-rag-demo"
    runtime.mkdir(parents=True)
    writer = EvidenceStore(tmp_path, runtime).open_exclusive_report_run("run-1")
    try:
        journal = _MediaExecutionJournal(writer)
        relative_path = (
            ".codex/video-rag-demo/eval/generated/run-1/normal_audio/source.mp4"
        )
        journal.begin_phase(
            case_id="normal_audio",
            phase="generate",
            executable="ffmpeg",
            output_relative_paths=(relative_path,),
        )
        media_file, artifact = _write_media_file(
            tmp_path, runtime, relative_path, "SOURCE", "MP4"
        )
        mismatched = artifact.model_copy(update={"sha256": "b" * 64})

        with pytest.raises(VideoDemoError, match="事实非法"):
            journal.record_media_file("normal_audio", media_file, mismatched)

        journal.record_media_file("normal_audio", media_file, artifact)
        with pytest.raises(VideoDemoError, match="事实非法"):
            journal.record_media_file("normal_audio", media_file, artifact)
    finally:
        writer.close()


def test_media_journal_rejects_completed_sample_that_omits_draft_files(
    tmp_path: Path,
) -> None:
    from video_demo.evaluation.media_runner import _MediaExecutionJournal

    runtime = tmp_path / ".codex/video-rag-demo"
    runtime.mkdir(parents=True)
    writer = EvidenceStore(tmp_path, runtime).open_exclusive_report_run("run-1")
    try:
        journal = _MediaExecutionJournal(writer)
        sample, artifacts = _record_successful_case(
            tmp_path,
            runtime,
            journal,
            "normal_audio",
            finalize=False,
        )
        incomplete = sample.model_copy(update={"files": sample.files[:-1]})

        with pytest.raises(VideoDemoError, match="当前媒体事实"):
            journal.record_completed_sample(incomplete, artifacts)
    finally:
        writer.close()


def test_media_port_nonzero_command_is_recovered_as_active_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, runtime = _runner(tmp_path)
    _complete_dependencies(runner, tmp_path, monkeypatch)
    _successful_versions(runner, monkeypatch)

    def execute(
        _run: str,
        _binaries: dict[str, Path],
        journal: Any,
    ) -> object:
        journal.begin_phase(
            case_id="normal_audio",
            phase="generate",
            executable="ffmpeg",
        )
        command, artifacts = _write_command(
            journal,
            "normal_audio",
            "generate",
            "ffmpeg",
            exit_code=1,
        )
        journal.record_completed_command(command, artifacts)
        raise AssertionError("非零命令不得被登记为完成")

    monkeypatch.setattr(runner, "_execute_media_port", execute)

    assert runner.run(evaluation_run_id="run-1").status == GateStatus.FAIL
    raw = RealMediaRawReport.model_validate_json(
        (runtime / "eval/reports/run-1/raw.json").read_bytes()
    )
    assert tuple(command.exit_code for command in raw.samples[0].commands) == (1,)


@pytest.mark.parametrize("termination", (KeyboardInterrupt, SystemExit))
def test_media_port_does_not_swallow_process_termination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    termination: type[BaseException],
) -> None:
    runner, _runtime = _runner(tmp_path)
    _complete_dependencies(runner, tmp_path, monkeypatch)
    _successful_versions(runner, monkeypatch)

    def terminate(*_args: object) -> object:
        raise termination()

    monkeypatch.setattr(runner, "_execute_media_port", terminate)

    with pytest.raises(termination):
        runner.run(evaluation_run_id="run-1")


@pytest.mark.parametrize("mode", ("setup_fail", "not_run"))
@pytest.mark.parametrize(
    ("role", "cross_run"),
    (("QUALITY_DETAIL", False), ("ANNOTATION", True)),
)
def test_setup_and_not_run_reject_unowned_artifact_roles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    role: str,
    cross_run: bool,
) -> None:
    runner, runtime = _runner(tmp_path)
    if mode == "setup_fail":
        _complete_dependencies(runner, tmp_path, monkeypatch)
        monkeypatch.setattr(runner, "_run_version", lambda *_args: (1, b"", b""))
    else:
        monkeypatch.setattr(runner, "_resolve_binary", lambda _name: None)
    runner.run(evaluation_run_id="run-1")
    report_path = runtime / "eval/reports/run-1/real-media.json"
    report = _load_json(report_path)
    target_run = "run-2" if cross_run else "run-1"
    artifact = EvidenceStore(tmp_path, runtime).write_artifact(
        Path(f"eval/reports/{target_run}/extra.json"), role, b"{}"
    )
    report["artifacts"].append(artifact.model_dump(mode="json"))
    _rewrite_json(report_path, report)

    with pytest.raises(ValueError, match="可信门禁"):
        build_verified_gate_check("real_media_chain", report_path, workspace_root=tmp_path)


@pytest.mark.parametrize("mode", ("setup_fail", "not_run"))
def test_setup_and_not_run_reject_copied_authoritative_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    runner, runtime = _runner(tmp_path)
    if mode == "setup_fail":
        _complete_dependencies(runner, tmp_path, monkeypatch)
        monkeypatch.setattr(runner, "_run_version", lambda *_args: (1, b"", b""))
    else:
        monkeypatch.setattr(runner, "_resolve_binary", lambda _name: None)
    runner.run(evaluation_run_id="run-1")
    original = runtime / "eval/reports/run-1/real-media.json"
    copied = runtime / "eval/reports/run-2/real-media.json"
    copied.parent.mkdir(parents=True)
    copied.write_bytes(original.read_bytes())

    with pytest.raises(ValueError, match="可信门禁"):
        build_verified_gate_check("real_media_chain", copied, workspace_root=tmp_path)


def _replace_run_with_internal_symlink(runtime: Path, target: Path) -> Path:
    run = runtime / "eval/reports/run-1"
    moved = runtime / "eval/reports/run-1-held-by-fd"
    os.rename(run, moved)
    run.symlink_to(target, target_is_directory=True)
    return moved


def test_run_writer_does_not_follow_replacement_before_first_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner, runtime = _runner(tmp_path)
    replacement = tmp_path / "artifact-replacement-target"
    replacement.mkdir()
    monkeypatch.setattr(runner, "_resolve_binary", lambda _name: None)
    import video_demo.evaluation.evidence as evidence_module

    real_write = evidence_module.ReportRunWriter.write_artifact
    moved: Path | None = None

    def replace_then_write(writer: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal moved
        if moved is None:
            moved = _replace_run_with_internal_symlink(runtime, replacement)
        return real_write(writer, *args, **kwargs)

    monkeypatch.setattr(evidence_module.ReportRunWriter, "write_artifact", replace_then_write)

    with pytest.raises(ValueError, match="原子写入失败"):
        runner.run(evaluation_run_id="run-1")

    assert not tuple(replacement.iterdir())
    assert moved is not None
    assert (moved / "preflight.json").is_file()
    assert (moved / "real-media.json").is_file()


def test_run_writer_does_not_follow_replacement_before_authoritative_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner, runtime = _runner(tmp_path)
    replacement = tmp_path / "json-replacement-target"
    replacement.mkdir()
    monkeypatch.setattr(runner, "_resolve_binary", lambda _name: None)
    import video_demo.evaluation.evidence as evidence_module

    real_write = evidence_module.ReportRunWriter.write_json
    moved: Path | None = None

    def replace_then_write(writer: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal moved
        moved = _replace_run_with_internal_symlink(runtime, replacement)
        return real_write(writer, *args, **kwargs)

    monkeypatch.setattr(evidence_module.ReportRunWriter, "write_json", replace_then_write)

    with pytest.raises(ValueError, match="原子写入失败"):
        runner.run(evaluation_run_id="run-1")

    assert not tuple(replacement.iterdir())
    assert moved is not None
    assert (moved / "preflight.json").is_file()
    assert (moved / "real-media.json").is_file()


def _ffprobe_payload(case_id: str) -> bytes:
    video: dict[str, object] = {
        "index": 0,
        "codec_type": "video",
        "codec_name": "h264",
        "width": 640,
        "height": 360,
        "avg_frame_rate": "20/1" if case_id == "vfr" else "25/1",
        "r_frame_rate": "30/1" if case_id == "vfr" else "25/1",
        "pix_fmt": "yuv420p",
    }
    if case_id == "rotation":
        video["side_data_list"] = [{"rotation": 90}]
    streams: list[dict[str, object]] = [video]
    if case_id != "no_audio":
        streams.append(
            {
                "index": 1,
                "codec_type": "audio",
                "codec_name": "aac",
                "sample_rate": "48000",
                "channels": 1,
            }
        )
    return _json_bytes(
        {
            "streams": streams,
            "format": {"duration": "2.000", "format_name": "mov,mp4"},
        }
    )


def _install_controlled_media_chain(
    tmp_path: Path,
    runtime: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    process_failure: str | None = None,
    source_bytes: bytes | None = None,
    source_race: str | None = None,
    ancestor_race_hook: Callable[
        [str, tuple[Path, ...], tuple[Path, ...]], None
    ]
    | None = None,
    opened_payloads: list[tuple[str, bytes]] | None = None,
    inherited_fds: list[int] | None = None,
) -> tuple[list[list[str]], list[str]]:
    import video_demo.evaluation.real_media_execution as execution_module
    from video_demo.visual.keyframes import FrameCandidate, KeyframeSelector

    argv_calls: list[list[str]] = []
    adapter_calls: list[str] = []
    mp4 = source_bytes or _media_bytes("MP4")

    class Array:
        def __init__(self, marker: int) -> None:
            self.marker = marker

        def __le__(self, _value: int) -> Array:
            return self

        def mean(self) -> float:
            return 0.0

        def var(self) -> float:
            return float(self.marker)

        def reshape(self, _size: int) -> Array:
            return self

        def __iter__(self):  # type: ignore[no-untyped-def]
            return iter([0] * 32 + [255] * 32)

    class Encoded:
        def __init__(self, marker: int) -> None:
            self.marker = marker

        def tobytes(self) -> bytes:
            return b"\xff\xd8\xff" + str(self.marker).encode() + b"\xff\xd9"

    class Capture:
        def __init__(self, path: str) -> None:
            adapter_calls.append("OpenCvFrameExtractor")
            proxy = Path(path)
            if ancestor_race_hook is not None:
                ancestor_race_hook("opencv_decode", (proxy,), ())
            if opened_payloads is not None:
                opened_payloads.append(("opencv_decode", proxy.read_bytes()))
            assert path.startswith("/dev/fd/") or path.endswith("/media/proxy.mp4")
            self._timestamp = 0

        def isOpened(self) -> bool:
            return True

        def set(self, _prop: int, value: float) -> bool:
            self._timestamp = round(value)
            adapter_calls.append(f"OpenCv.seek:{self._timestamp}")
            return True

        def read(self) -> tuple[bool, object]:
            return True, Array(self._timestamp)

        def get(self, _prop: int) -> float:
            return float(self._timestamp + 33)

        def release(self) -> None:
            pass

    class Cv2:
        CAP_PROP_POS_MSEC = 10
        COLOR_BGR2GRAY = 20
        CV_64F = 30
        INTER_AREA = 40
        IMWRITE_JPEG_QUALITY = 50
        VideoCapture = Capture
        cvtColor = staticmethod(lambda pixels, _mode: pixels)
        Laplacian = staticmethod(lambda gray, _kind: gray)
        resize = staticmethod(lambda gray, _size, **_kwargs: gray)
        imencode = staticmethod(
            lambda _extension, pixels, _params: (True, Encoded(pixels.marker))
        )

    class Timecode:
        def __init__(self, milliseconds: int) -> None:
            self._seconds = milliseconds / 1_000

        def get_seconds(self) -> float:
            return self._seconds

    class SceneManager:
        def add_detector(self, _detector: object) -> None:
            pass

        def detect_scenes(self, *, video: object, show_progress: bool) -> None:
            assert video == "controlled-video"
            assert show_progress is False

        def get_scene_list(self, *, start_in_scene: bool) -> object:
            assert start_in_scene is True
            return ((Timecode(0), Timecode(2_033)),)

    class SceneModule:
        @staticmethod
        def open_video(path: str) -> str:
            adapter_calls.append("PySceneDetectAdapter")
            proxy = Path(path)
            if ancestor_race_hook is not None:
                ancestor_race_hook("scene_detect", (proxy,), ())
            if opened_payloads is not None:
                opened_payloads.append(("scene_detect", proxy.read_bytes()))
            assert path.startswith("/dev/fd/") or path.endswith("/media/proxy.mp4")
            return "controlled-video"

    SceneModule.SceneManager = SceneManager

    class Detectors:
        ContentDetector = object

    monkeypatch.setattr(execution_module, "_load_cv2", lambda: Cv2, raising=False)
    monkeypatch.setattr(
        execution_module,
        "_load_scenedetect",
        lambda: (SceneModule, Detectors),
        raising=False,
    )

    def controlled_run(
        _runner: SafeProcessRunner,
        args: Sequence[str],
        *,
        timeout_seconds: int,
        pass_fds: tuple[int, ...] = (),
    ) -> ProcessResult:
        assert type(args) is list
        assert timeout_seconds > 0
        argv = list(args)
        argv_calls.append(argv)
        if inherited_fds is not None:
            inherited_fds.extend(pass_fds)
        if "-show_format" in argv:
            adapter_calls.append("FFprobeClient")
            source = Path(argv[-1])
            if ancestor_race_hook is not None:
                ancestor_race_hook("probe", (source,), ())
            if opened_payloads is not None:
                opened_payloads.append(("probe", source.read_bytes()))
            case_id = (
                _case_id_for_output(runtime, source, pass_fds)
                if pass_fds
                else source.parent.name
            )
            if process_failure == f"{case_id}:probe":
                return ProcessResult(2, b"provider body", b"Secret /private/body")
            return ProcessResult(0, _ffprobe_payload(case_id), b"")
        output = Path(argv[-1])
        input_paths = (
            (Path(argv[argv.index("-i") + 1]),) if "-i" in argv else ()
        )
        adapter_phase = (
            "audio"
            if len(pass_fds) == 2 and "pcm_s16le" in argv
            else "proxy"
            if len(pass_fds) == 2 and "libx264" in argv
            else None
        )
        if adapter_phase is not None and ancestor_race_hook is not None:
            ancestor_race_hook(adapter_phase, input_paths, (output,))
        if adapter_phase is not None and opened_payloads is not None:
            opened_payloads.extend(
                (adapter_phase, input_path.read_bytes()) for input_path in input_paths
            )
        if not pass_fds:
            output.parent.mkdir(parents=True, exist_ok=True)
        if adapter_phase == "audio" or output.suffix == ".wav":
            adapter_calls.append("FFmpegTranscoder.audio")
            output.write_bytes(_media_bytes("WAV"))
        elif adapter_phase == "proxy" or ".proxy." in output.name:
            adapter_calls.append("FFmpegTranscoder.proxy")
            output.write_bytes(_media_bytes("MP4"))
        else:
            case_id = _case_id_for_output(runtime, output, pass_fds)
            if process_failure == f"{case_id}:generate":
                return ProcessResult(3, b"third-party", b"Bearer token")
            if source_race == "temporary":
                race_target = tmp_path / "temporary-race-target"
                race_target.write_bytes(mp4)
                output.symlink_to(race_target)
            else:
                output.write_bytes(mp4)
            if source_race == "target":
                target = output.parent / "source.mp4"
                race_target = tmp_path / "target-race-target"
                race_target.write_bytes(mp4)
                target.symlink_to(race_target)
        return ProcessResult(0, b"untrusted stdout", b"untrusted stderr")

    monkeypatch.setattr(SafeProcessRunner, "run", controlled_run)
    real_select = KeyframeSelector.select

    def controlled_select(
        selector: KeyframeSelector, window: Any, candidates: Sequence[FrameCandidate]
    ) -> Any:
        assert isinstance(selector, KeyframeSelector)
        adapter_calls.append("KeyframeSelector")
        if ancestor_race_hook is not None:
            ancestor_race_hook(
                "keyframe_select",
                tuple(runtime / candidate.relative_path for candidate in candidates),
                (),
            )
        return real_select(selector, window, candidates)

    monkeypatch.setattr(KeyframeSelector, "select", controlled_select)
    return argv_calls, adapter_calls


def _case_id_for_output(
    runtime: Path,
    output: Path,
    pass_fds: tuple[int, ...],
) -> str:
    if not pass_fds:
        return output.parent.name
    identity = os.fstat(pass_fds[-1])
    generated = runtime / "eval/generated"
    for candidate in generated.rglob("*"):
        try:
            details = candidate.stat()
        except OSError:
            continue
        if (details.st_dev, details.st_ino) == (identity.st_dev, identity.st_ino):
            return candidate.parent.name
    raise AssertionError("受控 source fd 不属于当前 case")


def _run_controlled_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Any, Path, Any, list[list[str]], list[str]]:
    runner, runtime = _runner(tmp_path)
    _complete_dependencies(runner, tmp_path, monkeypatch)
    _successful_versions(runner, monkeypatch)
    argv, adapters = _install_controlled_media_chain(
        tmp_path, runtime, monkeypatch
    )
    check = runner.run(evaluation_run_id="run-1")
    return runner, runtime, check, argv, adapters


def test_task4_controlled_chain_produces_four_case_internal_execution_facts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from video_demo.evaluation.media_runner import _MediaExecutionJournal
    from video_demo.evaluation.real_media_execution import execute_real_media

    runtime = tmp_path / ".codex/video-rag-demo"
    runtime.mkdir(parents=True)
    settings = Settings(workspace_root=tmp_path)
    store = EvidenceStore(tmp_path, runtime)
    writer = store.open_exclusive_report_run("run-1")
    journal = _MediaExecutionJournal(writer)
    inherited_fds: list[int] = []
    argv, adapters = _install_controlled_media_chain(
        tmp_path,
        runtime,
        monkeypatch,
        inherited_fds=inherited_fds,
    )
    try:
        facts = execute_real_media(
            evaluation_run_id="run-1",
            binaries={"ffmpeg": tmp_path / "ffmpeg", "ffprobe": tmp_path / "ffprobe"},
            settings=settings,
            store=store,
            journal=journal,
        )
    finally:
        writer.close()

    assert tuple(sample.case_id for sample in facts.samples) == (
        "normal_audio",
        "no_audio",
        "rotation",
        "vfr",
    )
    assert all(
        tuple(command.phase for command in sample.commands)
        == tuple(phase for phase, _executable in _EXECUTABLES)
        for sample in facts.samples
    )
    assert all(type(arguments) is list for arguments in argv)
    assert sum("-show_format" in arguments for arguments in argv) == 4
    assert adapters.count("FFprobeClient") == 4
    assert adapters.count("FFmpegTranscoder.proxy") == 4
    assert adapters.count("FFmpegTranscoder.audio") == 3
    assert adapters.count("OpenCvFrameExtractor") == 4
    assert adapters.count("PySceneDetectAdapter") == 4
    assert adapters.count("KeyframeSelector") == 4
    assert tuple(sample.has_audio for sample in facts.samples) == (True, False, True, True)
    assert facts.samples[1].warnings == ("NO_AUDIO_TRACK",)
    assert facts.samples[2].rotation_degrees == 90
    assert facts.samples[3].is_variable_frame_rate is True
    assert all(sample.opencv_decoded_frame_count == 6 for sample in facts.samples)
    assert {
        int(call.rsplit(":", 1)[1])
        for call in adapters
        if call.startswith("OpenCv.seek:")
    } == {167, 500, 833, 1_167, 1_500, 1_833}
    assert all(sample.scene_count == 1 for sample in facts.samples)
    assert all(sample.selected_keyframe_count == 1 for sample in facts.samples)
    assert all(
        not Path(argument).is_absolute()
        for sample in facts.samples
        for command in sample.commands
        for argument in command.arguments
    )
    source_argv = [
        arguments
        for arguments in argv
        if arguments[-1].startswith("/dev/fd/")
        and ("testsrc2=" in " ".join(arguments) or "rotate=90" in arguments)
    ]
    assert len(source_argv) == 5
    assert all(arguments[-3:-1] == ["-f", "mp4"] for arguments in source_argv)
    assert all(
        "/dev/fd/" not in argument
        for sample in facts.samples
        for command in sample.commands
        for argument in command.arguments
    )
    adapter_fd_argv = [
        arguments
        for arguments in argv
        if arguments[-1].startswith("/dev/fd/") and arguments not in source_argv
    ]
    assert len(adapter_fd_argv) == 11
    declared = {media.relative_path for sample in facts.samples for media in sample.files}
    generated = {
        path.relative_to(tmp_path).as_posix()
        for path in (runtime / "eval/generated/run-1").rglob("*")
        if path.is_file() or path.is_symlink()
    }
    assert generated == declared
    assert not any(".part" in path for path in generated)
    persisted = b"\n".join(
        path.read_bytes()
        for path in (runtime / "eval/reports/run-1").iterdir()
        if path.is_file()
    ).decode("utf-8")
    assert all(
        forbidden not in persisted
        for forbidden in (
            "untrusted stdout",
            "untrusted stderr",
            "provider body",
            "Secret",
            "third-party",
            "Bearer",
        )
    )
    report_root = runtime / "eval/reports/run-1"
    assert not (report_root / "real-media.json").exists()
    assert not (report_root / ".real-media.commit.json").exists()
    assert len(facts.artifacts) == len({item.relative_path for item in facts.artifacts})
    assert inherited_fds
    assert all(_descriptor_is_closed(descriptor) for descriptor in set(inherited_fds))


@pytest.mark.parametrize("failure_mode", ("nonzero", "exception"))
def test_task4_adapter_descriptors_close_after_probe_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_mode: str,
) -> None:
    from video_demo.evaluation.media_runner import _MediaExecutionJournal
    from video_demo.evaluation.real_media_execution import execute_real_media

    runtime = tmp_path / ".codex/video-rag-demo"
    runtime.mkdir(parents=True)
    settings = Settings(workspace_root=tmp_path)
    store = EvidenceStore(tmp_path, runtime)
    writer = store.open_exclusive_report_run("run-1")
    journal = _MediaExecutionJournal(writer)
    inherited_fds: list[int] = []
    _install_controlled_media_chain(
        tmp_path,
        runtime,
        monkeypatch,
        process_failure="normal_audio:probe",
        inherited_fds=inherited_fds,
    )
    if failure_mode == "exception":
        controlled_run = SafeProcessRunner.run

        def raise_for_probe(
            runner: SafeProcessRunner,
            args: Sequence[str],
            *,
            timeout_seconds: int,
            pass_fds: tuple[int, ...] = (),
        ) -> ProcessResult:
            if "-show_format" in args:
                inherited_fds.extend(pass_fds)
                raise RuntimeError("controlled probe exception")
            return controlled_run(
                runner,
                args,
                timeout_seconds=timeout_seconds,
                pass_fds=pass_fds,
            )

        monkeypatch.setattr(SafeProcessRunner, "run", raise_for_probe)
    try:
        with pytest.raises((RuntimeError, VideoDemoError)):
            execute_real_media(
                evaluation_run_id="run-1",
                binaries={
                    "ffmpeg": tmp_path / "ffmpeg",
                    "ffprobe": tmp_path / "ffprobe",
                },
                settings=settings,
                store=store,
                journal=journal,
            )
    finally:
        writer.close()

    assert inherited_fds
    assert all(_descriptor_is_closed(descriptor) for descriptor in set(inherited_fds))


@pytest.mark.parametrize(
    ("failed_output", "expected_prior_stages"),
    (
        (Path("media/proxy.mp4"), (Path("media/proxy.mp4"),)),
        (
            Path("media/audio.wav"),
            (Path("media/proxy.mp4"), Path("media/audio.wav")),
        ),
    ),
)
def test_task4_staging_failure_closes_source_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failed_output: Path,
    expected_prior_stages: tuple[Path, ...],
) -> None:
    from video_demo.evaluation.media_runner import _MediaExecutionJournal
    from video_demo.evaluation.real_media_execution import execute_real_media
    from video_demo.evaluation.real_media_source import CaseExecutionSession

    runtime = tmp_path / ".codex/video-rag-demo"
    runtime.mkdir(parents=True)
    settings = Settings(workspace_root=tmp_path)
    store = EvidenceStore(tmp_path, runtime)
    writer = store.open_exclusive_report_run("run-1")
    journal = _MediaExecutionJournal(writer)
    opened_descriptors: list[int] = []
    staged_outputs: list[Path] = []
    real_open = CaseExecutionSession.open_registered_leaf
    real_stage = CaseExecutionSession.stage_output

    def capture_open(session: CaseExecutionSession, relative_path: Path) -> int:
        descriptor = real_open(session, relative_path)
        opened_descriptors.append(descriptor)
        return descriptor

    def fail_selected_stage(
        session: CaseExecutionSession,
        relative_path: Path,
    ) -> Any:
        staged_outputs.append(relative_path)
        if relative_path == failed_output:
            raise OSError("controlled staging failure")
        return real_stage(session, relative_path)

    monkeypatch.setattr(CaseExecutionSession, "open_registered_leaf", capture_open)
    monkeypatch.setattr(CaseExecutionSession, "stage_output", fail_selected_stage)
    _install_controlled_media_chain(tmp_path, runtime, monkeypatch)
    try:
        with pytest.raises(OSError, match="controlled staging failure"):
            execute_real_media(
                evaluation_run_id="run-1",
                binaries={
                    "ffmpeg": tmp_path / "ffmpeg",
                    "ffprobe": tmp_path / "ffprobe",
                },
                settings=settings,
                store=store,
                journal=journal,
            )
    finally:
        writer.close()

    assert tuple(staged_outputs) == expected_prior_stages
    assert opened_descriptors
    assert _descriptor_is_closed(opened_descriptors[-1])


def _descriptor_is_closed(descriptor: int) -> bool:
    try:
        os.fstat(descriptor)
    except OSError as error:
        return error.errno == errno.EBADF
    return False


@pytest.mark.parametrize(
    "phase",
    ("probe", "proxy", "audio", "opencv_decode", "scene_detect", "keyframe_select"),
)
@pytest.mark.parametrize("replaced_level", ("case", "run", "generated"))
def test_task4_production_adapters_keep_io_bound_to_held_case_when_ancestor_moves(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
    replaced_level: str,
) -> None:
    runner, runtime = _runner(tmp_path)
    _complete_dependencies(runner, tmp_path, monkeypatch)
    _successful_versions(runner, monkeypatch)
    case_root = runtime / "eval/generated/run-1/normal_audio"
    outside_root = tmp_path / f"outside-{phase}-{replaced_level}"
    marker = f"external-{phase}-{replaced_level}".encode()
    opened_payloads: list[tuple[str, bytes]] = []
    external_before: dict[str, bytes] = {}
    injected = False

    def inject_race(
        current_phase: str,
        _inputs: tuple[Path, ...],
        _outputs: tuple[Path, ...],
    ) -> None:
        nonlocal external_before, injected
        if injected or current_phase != phase:
            return
        injected = True
        replaced = {
            "case": case_root,
            "run": case_root.parent,
            "generated": case_root.parent.parent,
        }[replaced_level]
        moved = replaced.with_name(f"{replaced.name}-held-{phase}-{replaced_level}")
        replaced.rename(moved)
        external_target = outside_root / "replacement"
        external_case = external_target / {
            "case": Path(),
            "run": Path("normal_audio"),
            "generated": Path("run-1/normal_audio"),
        }[replaced_level]
        moved_case = moved / {
            "case": Path(),
            "run": Path("normal_audio"),
            "generated": Path("run-1/normal_audio"),
        }[replaced_level]
        shutil.copytree(moved_case, external_case)
        for external_file in external_case.rglob("*"):
            if external_file.is_file():
                external_file.write_bytes(external_file.read_bytes() + marker)
        sentinel = external_target / "sentinel.bin"
        sentinel.write_bytes(marker)
        external_before = {
            item.relative_to(external_target).as_posix(): item.read_bytes()
            for item in external_target.rglob("*")
            if item.is_file()
        }
        replaced.symlink_to(external_target, target_is_directory=True)

    _install_controlled_media_chain(
        tmp_path,
        runtime,
        monkeypatch,
        ancestor_race_hook=inject_race,
        opened_payloads=opened_payloads,
    )

    try:
        check = runner.run(evaluation_run_id="run-1")
    except ValueError as error:
        assert "原子写入失败" in str(error)
    else:
        assert check.status == GateStatus.FAIL
    assert injected
    assert all(marker not in payload for _name, payload in opened_payloads)
    external_target = outside_root / "replacement"
    assert {
        item.relative_to(external_target).as_posix(): item.read_bytes()
        for item in external_target.rglob("*")
        if item.is_file()
    } == external_before


def test_task4_middle_case_failure_keeps_success_prefix_and_stops_following_cases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner, runtime = _runner(tmp_path)
    _complete_dependencies(runner, tmp_path, monkeypatch)
    _successful_versions(runner, monkeypatch)
    _install_controlled_media_chain(
        tmp_path,
        runtime,
        monkeypatch,
        process_failure="no_audio:generate",
    )

    check = runner.run(evaluation_run_id="run-1")

    assert check.status == GateStatus.FAIL
    raw = RealMediaRawReport.model_validate_json(
        (runtime / "eval/reports/run-1/raw.json").read_bytes()
    )
    assert tuple(sample.execution_status for sample in raw.samples) == (
        "SUCCESS",
        "FAILED",
        "NOT_STARTED",
        "NOT_STARTED",
    )
    assert tuple(command.phase for command in raw.samples[1].commands) == ("generate",)
    assert raw.samples[1].commands[0].exit_code != 0
    no_audio_root = runtime / "eval/generated/run-1/no_audio"
    assert not any(
        path.is_file() or path.is_symlink() for path in no_audio_root.rglob("*")
    )
    assert build_verified_gate_check(
        "real_media_chain",
        runtime / "eval/reports/run-1/real-media.json",
        workspace_root=tmp_path,
    ) == check


@pytest.mark.parametrize(
    ("residual_kind", "relative_leaf"),
    (
        ("regular", "stale.bin"),
        ("jpeg", "visual/keyframes/stale.jpg"),
        ("part", ".source.stale.part.mp4"),
        ("directory_symlink", "linked"),
    ),
)
def test_task4_existing_case_root_is_rejected_without_touching_residual(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    residual_kind: str,
    relative_leaf: str,
) -> None:
    runner, runtime = _runner(tmp_path)
    _complete_dependencies(runner, tmp_path, monkeypatch)
    _successful_versions(runner, monkeypatch)
    argv, _adapters = _install_controlled_media_chain(
        tmp_path, runtime, monkeypatch
    )
    case_root = runtime / "eval/generated/run-1/normal_audio"
    leaf = case_root / relative_leaf
    leaf.parent.mkdir(parents=True, exist_ok=True)
    outside = tmp_path / "outside-residual"
    outside.mkdir()
    sentinel = outside / "sentinel.txt"
    sentinel.write_text("不得触碰", encoding="utf-8")
    if residual_kind == "directory_symlink":
        leaf.symlink_to(outside, target_is_directory=True)
    else:
        leaf.write_bytes(
            _media_bytes("JPEG") if residual_kind == "jpeg" else b"residual"
        )

    check = runner.run(evaluation_run_id="run-1")

    assert check.status == GateStatus.FAIL
    assert leaf.is_symlink() if residual_kind == "directory_symlink" else leaf.is_file()
    assert sentinel.read_text(encoding="utf-8") == "不得触碰"
    assert not argv
    assert build_verified_gate_check(
        "real_media_chain",
        runtime / "eval/reports/run-1/real-media.json",
        workspace_root=tmp_path,
    ) == check


def test_task4_case_creation_race_does_not_claim_or_clean_attacker_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import video_demo.evaluation.real_media_execution as execution_module
    import video_demo.evaluation.real_media_source as source_module

    runner, runtime = _runner(tmp_path)
    _complete_dependencies(runner, tmp_path, monkeypatch)
    _successful_versions(runner, monkeypatch)
    argv, _adapters = _install_controlled_media_chain(
        tmp_path, runtime, monkeypatch
    )
    original = source_module.open_case_execution_session
    case_root = runtime / "eval/generated/run-1/normal_audio"
    stale = case_root / "attacker-owned.bin"

    def race_before_exclusive_create(*args: Any, **kwargs: Any) -> Any:
        stale.parent.mkdir(parents=True)
        stale.write_bytes(b"attacker-owned")
        return original(*args, **kwargs)

    monkeypatch.setattr(
        execution_module, "open_case_execution_session", race_before_exclusive_create
    )

    check = runner.run(evaluation_run_id="run-1")

    assert check.status == GateStatus.FAIL
    assert stale.read_bytes() == b"attacker-owned"
    assert not argv


def test_task4_case_replacement_after_source_never_cleans_attacker_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import video_demo.evaluation.real_media_execution as execution_module

    runner, runtime = _runner(tmp_path)
    _complete_dependencies(runner, tmp_path, monkeypatch)
    _successful_versions(runner, monkeypatch)
    _install_controlled_media_chain(
        tmp_path,
        runtime,
        monkeypatch,
        process_failure="normal_audio:probe",
    )
    original = execution_module.generate_source
    case_root = runtime / "eval/generated/run-1/normal_audio"
    moved_root = case_root.with_name("normal_audio-owned")
    attacker_source = case_root / "source.mp4"
    sentinel = case_root / "sentinel.bin"

    def replace_after_source(*args: Any, **kwargs: Any) -> Any:
        generated = original(*args, **kwargs)
        case_root.rename(moved_root)
        case_root.mkdir()
        attacker_source.write_bytes((moved_root / "source.mp4").read_bytes())
        sentinel.write_bytes(b"attacker-owned")
        return generated

    monkeypatch.setattr(execution_module, "generate_source", replace_after_source)

    check = runner.run(evaluation_run_id="run-1")

    assert check.status == GateStatus.FAIL
    assert sentinel.read_bytes() == b"attacker-owned"
    assert attacker_source.is_file()
    assert not (moved_root / "source.mp4").exists()
    raw = RealMediaRawReport.model_validate_json(
        (runtime / "eval/reports/run-1/raw.json").read_bytes()
    )
    assert raw.samples[0].files == ()


def test_task4_source_bind_rejects_same_bytes_from_replaced_inode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import video_demo.evaluation.real_media_execution as execution_module

    runner, runtime = _runner(tmp_path)
    _complete_dependencies(runner, tmp_path, monkeypatch)
    _successful_versions(runner, monkeypatch)
    _install_controlled_media_chain(tmp_path, runtime, monkeypatch)
    original = execution_module.generate_source
    source = runtime / "eval/generated/run-1/normal_audio/source.mp4"
    original_source = source.with_name("source.original.mp4")

    def replace_published_leaf(*args: Any, **kwargs: Any) -> Any:
        generated = original(*args, **kwargs)
        original_identity = source.stat().st_ino
        contents = source.read_bytes()
        source.rename(original_source)
        source.write_bytes(contents)
        assert source.stat().st_ino != original_identity
        return generated

    monkeypatch.setattr(execution_module, "generate_source", replace_published_leaf)

    check = runner.run(evaluation_run_id="run-1")

    assert check.status == GateStatus.FAIL
    raw = RealMediaRawReport.model_validate_json(
        (runtime / "eval/reports/run-1/raw.json").read_bytes()
    )
    assert raw.samples[0].execution_status == "FAILED"
    assert raw.samples[0].files == ()
    assert not source.exists()
    assert not original_source.exists()


def test_task4_success_fails_closed_when_late_undeclared_leaf_appears(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import video_demo.evaluation.real_media_execution as execution_module

    runner, runtime = _runner(tmp_path)
    _complete_dependencies(runner, tmp_path, monkeypatch)
    _successful_versions(runner, monkeypatch)
    _install_controlled_media_chain(tmp_path, runtime, monkeypatch)
    original = execution_module._keyframe_phase

    def create_late_residual(*args: Any, **kwargs: Any) -> Any:
        selected = original(*args, **kwargs)
        residual = runtime / "eval/generated/run-1/normal_audio/late.part"
        residual.write_bytes(b"late residual")
        return selected

    monkeypatch.setattr(execution_module, "_keyframe_phase", create_late_residual)

    check = runner.run(evaluation_run_id="run-1")

    assert check.status == GateStatus.FAIL
    case_root = runtime / "eval/generated/run-1/normal_audio"
    raw = RealMediaRawReport.model_validate_json(
        (runtime / "eval/reports/run-1/raw.json").read_bytes()
    )
    declared = {media.relative_path for media in raw.samples[0].files}
    observed = {
        path.relative_to(tmp_path).as_posix()
        for path in case_root.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    assert observed == declared
    assert build_verified_gate_check(
        "real_media_chain",
        runtime / "eval/reports/run-1/real-media.json",
        workspace_root=tmp_path,
    ) == check


def test_task4_leaf_inserted_after_closure_scan_cannot_commit_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import video_demo.evaluation.real_media_execution as execution_module

    runner, runtime = _runner(tmp_path)
    _complete_dependencies(runner, tmp_path, monkeypatch)
    _successful_versions(runner, monkeypatch)
    _install_controlled_media_chain(tmp_path, runtime, monkeypatch)
    original = execution_module._assert_case_media_closed
    late = runtime / "eval/generated/run-1/normal_audio/late.part"

    def insert_after_real_scan(*args: Any, **kwargs: Any) -> None:
        original(*args, **kwargs)
        late.write_bytes(b"late residual")

    monkeypatch.setattr(
        execution_module,
        "_assert_case_media_closed",
        insert_after_real_scan,
    )

    check = runner.run(evaluation_run_id="run-1")

    assert check.status == GateStatus.FAIL
    assert not late.exists()
    raw = RealMediaRawReport.model_validate_json(
        (runtime / "eval/reports/run-1/raw.json").read_bytes()
    )
    assert raw.samples[0].execution_status == "FAILED"
    assert all(sample.execution_status != "SUCCESS" for sample in raw.samples)
    assert build_verified_gate_check(
        "real_media_chain",
        runtime / "eval/reports/run-1/real-media.json",
        workspace_root=tmp_path,
    ) == check


@pytest.mark.parametrize("failure_point", ("bind", "record"))
def test_task4_transfer_before_registration_rolls_back_unowned_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_point: str,
) -> None:
    runner, runtime = _runner(tmp_path)
    _complete_dependencies(runner, tmp_path, monkeypatch)
    _successful_versions(runner, monkeypatch)
    _install_controlled_media_chain(tmp_path, runtime, monkeypatch)
    if failure_point == "bind":
        from video_demo.evaluation.real_media_source import CaseExecutionSession

        original_snapshot = CaseExecutionSession.snapshot_leaf
        calls = 0

        def fail_second_snapshot(
            session: CaseExecutionSession,
            relative_path: Path,
            max_bytes: int,
        ) -> Any:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise ValueError("bind fault")
            return original_snapshot(session, relative_path, max_bytes)

        monkeypatch.setattr(
            CaseExecutionSession,
            "snapshot_leaf",
            fail_second_snapshot,
        )
    else:
        from video_demo.evaluation.media_runner import _MediaExecutionJournal

        monkeypatch.setattr(
            _MediaExecutionJournal,
            "record_media_file",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("record fault")),
        )

    check = runner.run(evaluation_run_id="run-1")

    assert check.status == GateStatus.FAIL
    raw = RealMediaRawReport.model_validate_json(
        (runtime / "eval/reports/run-1/raw.json").read_bytes()
    )
    assert raw.samples[0].execution_status == "FAILED"
    assert raw.samples[0].files == ()
    case_root = runtime / "eval/generated/run-1/normal_audio"
    assert not tuple(case_root.rglob("*"))
    assert build_verified_gate_check(
        "real_media_chain",
        runtime / "eval/reports/run-1/real-media.json",
        workspace_root=tmp_path,
    ) == check


def test_task4_failure_after_registration_keeps_journal_owned_source_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner, runtime = _runner(tmp_path)
    _complete_dependencies(runner, tmp_path, monkeypatch)
    _successful_versions(runner, monkeypatch)
    _install_controlled_media_chain(
        tmp_path,
        runtime,
        monkeypatch,
        process_failure="normal_audio:probe",
    )

    check = runner.run(evaluation_run_id="run-1")

    assert check.status == GateStatus.FAIL
    raw = RealMediaRawReport.model_validate_json(
        (runtime / "eval/reports/run-1/raw.json").read_bytes()
    )
    assert tuple(media.role for media in raw.samples[0].files) == ("SOURCE",)
    source = runtime / "eval/generated/run-1/normal_audio/source.mp4"
    assert source.is_file()
    assert {
        path for path in source.parent.rglob("*") if path.is_file() or path.is_symlink()
    } == {source}
    assert build_verified_gate_check(
        "real_media_chain",
        runtime / "eval/reports/run-1/real-media.json",
        workspace_root=tmp_path,
    ) == check


def test_task4_exception_immediately_after_record_keeps_journal_owned_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import video_demo.evaluation.real_media_execution as execution_module

    runner, runtime = _runner(tmp_path)
    _complete_dependencies(runner, tmp_path, monkeypatch)
    _successful_versions(runner, monkeypatch)
    _install_controlled_media_chain(tmp_path, runtime, monkeypatch)
    monkeypatch.setattr(
        execution_module,
        "_accept_transfer",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("post-record fault")),
    )

    check = runner.run(evaluation_run_id="run-1")

    assert check.status == GateStatus.FAIL
    raw = RealMediaRawReport.model_validate_json(
        (runtime / "eval/reports/run-1/raw.json").read_bytes()
    )
    assert tuple(media.role for media in raw.samples[0].files) == ("SOURCE",)
    source = runtime / "eval/generated/run-1/normal_audio/source.mp4"
    assert source.is_file()
    assert build_verified_gate_check(
        "real_media_chain",
        runtime / "eval/reports/run-1/real-media.json",
        workspace_root=tmp_path,
    ) == check


def test_task4_real_journal_commit_then_runtime_error_keeps_reverifiable_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from video_demo.evaluation.media_runner import _MediaExecutionJournal

    runner, runtime = _runner(tmp_path)
    _complete_dependencies(runner, tmp_path, monkeypatch)
    _successful_versions(runner, monkeypatch)
    _install_controlled_media_chain(tmp_path, runtime, monkeypatch)
    original = _MediaExecutionJournal.record_media_file

    def fail_after_real_commit(
        journal: _MediaExecutionJournal,
        case_id: str,
        media_file: RealMediaFile,
        artifact: TraceArtifact,
    ) -> None:
        original(journal, case_id, media_file, artifact)
        raise RuntimeError("post-commit fault")

    monkeypatch.setattr(
        _MediaExecutionJournal, "record_media_file", fail_after_real_commit
    )

    check = runner.run(evaluation_run_id="run-1")

    assert check.status == GateStatus.FAIL
    raw = RealMediaRawReport.model_validate_json(
        (runtime / "eval/reports/run-1/raw.json").read_bytes()
    )
    assert tuple(media.role for media in raw.samples[0].files) == ("SOURCE",)
    source = runtime / "eval/generated/run-1/normal_audio/source.mp4"
    assert source.is_file()
    assert build_verified_gate_check(
        "real_media_chain",
        runtime / "eval/reports/run-1/real-media.json",
        workspace_root=tmp_path,
    ) == check


@pytest.mark.parametrize("termination", (KeyboardInterrupt, SystemExit))
def test_task4_real_journal_commit_then_termination_keeps_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    termination: type[BaseException],
) -> None:
    from video_demo.evaluation.media_runner import _MediaExecutionJournal

    runner, runtime = _runner(tmp_path)
    _complete_dependencies(runner, tmp_path, monkeypatch)
    _successful_versions(runner, monkeypatch)
    _install_controlled_media_chain(tmp_path, runtime, monkeypatch)
    original = _MediaExecutionJournal.record_media_file

    def terminate_after_real_commit(
        journal: _MediaExecutionJournal,
        case_id: str,
        media_file: RealMediaFile,
        artifact: TraceArtifact,
    ) -> None:
        original(journal, case_id, media_file, artifact)
        raise termination("post-commit termination")

    monkeypatch.setattr(
        _MediaExecutionJournal, "record_media_file", terminate_after_real_commit
    )

    with pytest.raises(termination):
        runner.run(evaluation_run_id="run-1")

    source = runtime / "eval/generated/run-1/normal_audio/source.mp4"
    assert source.is_file()


def test_task4_opencv_failure_removes_unregistered_candidates_but_keeps_owned_media(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from video_demo.errors import ErrorCode
    from video_demo.visual.keyframes import OpenCvFrameExtractor

    runner, runtime = _runner(tmp_path)
    _complete_dependencies(runner, tmp_path, monkeypatch)
    _successful_versions(runner, monkeypatch)
    _install_controlled_media_chain(tmp_path, runtime, monkeypatch)

    def fail_after_candidate(
        _extractor: OpenCvFrameExtractor,
        _proxy: Path,
        run_relative_root: Path,
        _windows: Sequence[Any],
        **_kwargs: object,
    ) -> tuple[object, ...]:
        candidate = runtime / run_relative_root / "visual/keyframes/orphan.jpg"
        candidate.parent.mkdir(parents=True, exist_ok=True)
        candidate.write_bytes(_media_bytes("JPEG"))
        raise VideoDemoError(ErrorCode.VISUAL_MEDIA_INVALID, "controlled failure")

    monkeypatch.setattr(OpenCvFrameExtractor, "extract", fail_after_candidate)

    check = runner.run(evaluation_run_id="run-1")

    assert check.status == GateStatus.FAIL
    raw = RealMediaRawReport.model_validate_json(
        (runtime / "eval/reports/run-1/raw.json").read_bytes()
    )
    failed = raw.samples[0]
    assert tuple(media.role for media in failed.files) == ("SOURCE", "PROXY", "AUDIO")
    case_root = runtime / "eval/generated/run-1/normal_audio"
    kept = {
        path.relative_to(tmp_path).as_posix()
        for path in case_root.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    assert kept == {media.relative_path for media in failed.files}
    assert build_verified_gate_check(
        "real_media_chain",
        runtime / "eval/reports/run-1/real-media.json",
        workspace_root=tmp_path,
    ) == check


@pytest.mark.parametrize("source_race", ("target", "temporary"))
def test_task4_source_rejects_symlink_races_and_cleans_parts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_race: str,
) -> None:
    runner, runtime = _runner(tmp_path)
    _complete_dependencies(runner, tmp_path, monkeypatch)
    _successful_versions(runner, monkeypatch)
    _install_controlled_media_chain(
        tmp_path, runtime, monkeypatch, source_race=source_race
    )

    check = runner.run(evaluation_run_id="run-1")

    assert check.status == GateStatus.FAIL
    case_root = runtime / "eval/generated/run-1/normal_audio"
    assert not any(path.is_file() or path.is_symlink() for path in case_root.rglob("*"))
    assert build_verified_gate_check(
        "real_media_chain",
        runtime / "eval/reports/run-1/real-media.json",
        workspace_root=tmp_path,
    ) == check


def test_task4_source_rejects_parent_symlink_without_touching_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner, runtime = _runner(tmp_path)
    _complete_dependencies(runner, tmp_path, monkeypatch)
    _successful_versions(runner, monkeypatch)
    target = tmp_path / "generated-parent-target"
    target.mkdir()
    generated = runtime / "eval/generated"
    generated.parent.mkdir(parents=True)
    generated.symlink_to(target, target_is_directory=True)
    _install_controlled_media_chain(tmp_path, runtime, monkeypatch)

    check = runner.run(evaluation_run_id="run-1")

    assert check.status == GateStatus.FAIL
    assert not tuple(target.iterdir())
    assert build_verified_gate_check(
        "real_media_chain",
        runtime / "eval/reports/run-1/real-media.json",
        workspace_root=tmp_path,
    ) == check


def test_task4_source_enforces_configured_size_limit_and_cleans_part(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from video_demo.evaluation.media_runner import RealMediaRunner

    runtime = tmp_path / ".codex/video-rag-demo"
    runtime.mkdir(parents=True)
    settings = Settings(workspace_root=tmp_path, max_video_bytes=32)
    runner = RealMediaRunner(settings, EvidenceStore(tmp_path, runtime))
    _complete_dependencies(runner, tmp_path, monkeypatch)
    _successful_versions(runner, monkeypatch)
    oversized = _media_bytes("MP4") + b"x" * 33
    _install_controlled_media_chain(
        tmp_path, runtime, monkeypatch, source_bytes=oversized
    )

    check = runner.run(evaluation_run_id="run-1")

    assert check.status == GateStatus.FAIL
    case_root = runtime / "eval/generated/run-1/normal_audio"
    assert not any(path.is_file() or path.is_symlink() for path in case_root.rglob("*"))


@pytest.mark.parametrize(
    "changed_path",
    (
        Path("src/video_demo/config.py"),
        Path("src/video_demo/evaluation/real_media_execution.py"),
        Path("src/video_demo/evaluation/real_media_source.py"),
        Path("src/video_demo/media/process.py"),
    ),
)
def test_task4_direct_execution_dependencies_are_bound_into_implementation_digest(
    tmp_path: Path,
    changed_path: Path,
) -> None:
    assert changed_path in _REAL_MEDIA_IMPLEMENTATION_FILES
    project_root = Path(__file__).parents[2]
    for relative in _REAL_MEDIA_IMPLEMENTATION_FILES:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(project_root / relative, destination)
    before = _unpatched_implementation_digest(tmp_path)
    copied_source = tmp_path / changed_path
    copied_source.write_bytes(copied_source.read_bytes() + b"\n")
    assert _unpatched_implementation_digest(tmp_path) != before


def test_real_media_digest_does_not_claim_subtitle_or_speech_execution() -> None:
    # real_media_execution 只复用 pipeline 的四个无行为数据类；它不执行字幕选择。
    assert Path("src/video_demo/media/subtitles.py") not in _REAL_MEDIA_IMPLEMENTATION_FILES
    assert Path("src/video_demo/speech/isolated.py") not in _REAL_MEDIA_IMPLEMENTATION_FILES


def test_implementation_import_closure_ignores_type_checking_only_imports(
    tmp_path: Path,
) -> None:
    from video_demo.implementation import implementation_import_closure

    package = tmp_path / "src/video_demo"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "runtime_dependency.py").write_text("VALUE = 1\n", encoding="utf-8")
    (package / "runtime_else_dependency.py").write_text("VALUE = 3\n", encoding="utf-8")
    (package / "type_dependency.py").write_text("VALUE = 2\n", encoding="utf-8")
    entry = package / "entry.py"
    entry.write_text(
        "import typing\n"
        "from typing import TYPE_CHECKING\n"
        "from video_demo.runtime_dependency import VALUE\n"
        "if TYPE_CHECKING:\n"
        "    from video_demo.type_dependency import VALUE as TYPE_VALUE\n"
        "if typing.TYPE_CHECKING:\n"
        "    from video_demo.type_dependency import VALUE as ATTRIBUTE_TYPE_VALUE\n"
        "else:\n"
        "    from video_demo.runtime_else_dependency import VALUE as ELSE_VALUE\n",
        encoding="utf-8",
    )

    closure = implementation_import_closure(
        tmp_path,
        (Path("src/video_demo/entry.py"),),
        extra_files=(),
    )

    assert Path("src/video_demo/runtime_dependency.py") in closure
    assert Path("src/video_demo/runtime_else_dependency.py") in closure
    assert Path("src/video_demo/type_dependency.py") not in closure


def test_implementation_import_closure_does_not_trust_shadowed_type_checking(
    tmp_path: Path,
) -> None:
    from video_demo.implementation import implementation_import_closure

    package = tmp_path / "src/video_demo"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "runtime_dependency.py").write_text("VALUE = 1\n", encoding="utf-8")
    entry = package / "entry.py"
    entry.write_text(
        "from typing import TYPE_CHECKING\n"
        "TYPE_CHECKING = True\n"
        "if TYPE_CHECKING:\n"
        "    from video_demo.runtime_dependency import VALUE\n",
        encoding="utf-8",
    )

    closure = implementation_import_closure(
        tmp_path,
        (Path("src/video_demo/entry.py"),),
        extra_files=(),
    )

    assert Path("src/video_demo/runtime_dependency.py") in closure


@pytest.mark.parametrize(
    "unsafe_output",
    (
        "oversized",
        "target_symlink",
        "part_symlink",
        "parent_symlink",
        "internal_parent_symlink",
    ),
)
def test_task4_source_generator_rejects_unsafe_output_and_removes_part(
    tmp_path: Path,
    unsafe_output: str,
) -> None:
    from video_demo.evaluation.real_media_source import (
        generate_source,
        open_case_execution_session,
    )

    runtime = tmp_path / ".codex/video-rag-demo"
    runtime.mkdir(parents=True)
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(_media_bytes("MP4"))
    seen: list[list[str]] = []
    if unsafe_output in {"parent_symlink", "internal_parent_symlink"}:
        generated_target = (
            runtime / "internal-generated-target"
            if unsafe_output == "internal_parent_symlink"
            else tmp_path / "generated-target"
        )
        generated_target.mkdir()
        generated = runtime / "eval/generated"
        generated.parent.mkdir(parents=True)
        generated.symlink_to(generated_target, target_is_directory=True)

    class ControlledRunner:
        def run(
            self,
            args: Sequence[str],
            *,
            timeout_seconds: int,
            pass_fds: tuple[int, ...] = (),
        ) -> ProcessResult:
            assert type(args) is list and timeout_seconds == 7
            seen.append(list(args))
            if unsafe_output == "part_symlink":
                identity = os.fstat(pass_fds[-1])
                part = next(
                    path
                    for path in (runtime / "eval/generated/run-1/normal_audio").iterdir()
                    if path.stat().st_ino == identity.st_ino
                )
                part.unlink()
                part.symlink_to(outside)
            else:
                os.write(
                    pass_fds[-1],
                    _media_bytes("MP4")
                    + (b"x" * 100 if unsafe_output == "oversized" else b""),
                )
            if unsafe_output == "target_symlink":
                (runtime / "eval/generated/run-1/normal_audio/source.mp4").symlink_to(
                    outside
                )
            return ProcessResult(0, b"third-party", b"Secret")

    session = None
    with pytest.raises(VideoDemoError):
        session = open_case_execution_session(runtime, "run-1", "normal_audio")
        generate_source(
            session=session,
            case_id="normal_audio",
            executable=tmp_path / "ffmpeg",
            runner=ControlledRunner(),
            max_bytes=64,
            timeout_seconds=7,
        )
    if session is not None:
        session.cleanup_unregistered(set())
        session.close()

    case_root = runtime / "eval/generated/run-1/normal_audio"
    assert len(seen) == (
        0
        if unsafe_output in {"parent_symlink", "internal_parent_symlink"}
        else 1
    )
    assert not any(path.is_file() or path.is_symlink() for path in case_root.rglob("*"))
    assert outside.read_bytes() == _media_bytes("MP4")


def test_task4_source_generator_uses_random_part_then_atomic_source(
    tmp_path: Path,
) -> None:
    from video_demo.evaluation.real_media_source import (
        generate_source,
        open_case_execution_session,
    )

    runtime = tmp_path / ".codex/video-rag-demo"
    runtime.mkdir(parents=True)
    seen_commands: list[list[str]] = []

    class ControlledRunner:
        def run(
            self,
            args: Sequence[str],
            *,
            timeout_seconds: int,
            pass_fds: tuple[int, ...] = (),
        ) -> ProcessResult:
            assert type(args) is list and timeout_seconds == 7
            seen_commands.append(list(args))
            os.write(pass_fds[-1], _media_bytes("MP4"))
            return ProcessResult(0, b"third-party", b"Secret")

    session = open_case_execution_session(runtime, "run-1", "normal_audio")
    try:
        product = generate_source(
            session=session,
            case_id="normal_audio",
            executable=tmp_path / "ffmpeg",
            runner=ControlledRunner(),
            max_bytes=1_024,
            timeout_seconds=7,
        )
    finally:
        session.close()

    expected = runtime / "eval/generated/run-1/normal_audio/source.mp4"
    assert product.path == expected
    assert product.process_result.returncode == 0
    assert product.path.read_bytes() == _media_bytes("MP4")
    assert len(seen_commands) == 1
    command = seen_commands[0]
    assert command[-1].startswith("/dev/fd/")
    assert (
        command[command.index("-movflags") + 1]
        == "+frag_keyframe+empty_moov+delay_moov"
    )
    assert not any("part" in path.name for path in expected.parent.iterdir())


def test_task4_rotation_generator_uses_fragmented_mp4_for_both_fd_outputs(
    tmp_path: Path,
) -> None:
    from video_demo.evaluation.real_media_source import (
        generate_source,
        open_case_execution_session,
    )

    runtime = tmp_path / ".codex/video-rag-demo"
    runtime.mkdir(parents=True)
    seen_commands: list[list[str]] = []

    class ControlledRunner:
        def run(
            self,
            args: Sequence[str],
            *,
            timeout_seconds: int,
            pass_fds: tuple[int, ...] = (),
        ) -> ProcessResult:
            assert type(args) is list and timeout_seconds == 7
            seen_commands.append(list(args))
            os.write(pass_fds[-1], _media_bytes("MP4"))
            return ProcessResult(0, b"", b"")

    session = open_case_execution_session(runtime, "run-1", "rotation")
    try:
        product = generate_source(
            session=session,
            case_id="rotation",
            executable=tmp_path / "ffmpeg",
            runner=ControlledRunner(),
            max_bytes=1_024,
            timeout_seconds=7,
        )
    finally:
        session.close()

    assert product.process_result.returncode == 0
    assert len(seen_commands) == 2
    assert all(
        command[command.index("-movflags") + 1]
        == "+frag_keyframe+empty_moov+delay_moov"
        for command in seen_commands
    )


def test_task4_case_session_closes_all_owned_descriptors(tmp_path: Path) -> None:
    import video_demo.evaluation.real_media_source as source_module

    runtime = tmp_path / ".codex/video-rag-demo"
    runtime.mkdir(parents=True)

    class ControlledRunner:
        def run(
            self,
            _args: Sequence[str],
            *,
            timeout_seconds: int,
            pass_fds: tuple[int, ...] = (),
        ) -> ProcessResult:
            assert timeout_seconds == 7
            os.write(pass_fds[-1], _media_bytes("MP4"))
            return ProcessResult(0, b"", b"")

    session = source_module.open_case_execution_session(
        runtime,
        "run-1",
        "normal_audio",
    )
    source_module.generate_source(
        session=session,
        case_id="normal_audio",
        executable=tmp_path / "ffmpeg",
        runner=ControlledRunner(),
        max_bytes=1_024,
        timeout_seconds=7,
    )
    descriptors = [lease.descriptor for lease in session._leases]
    assert session._source_descriptor is not None
    descriptors.append(session._source_descriptor)

    session.close()
    session.close()

    for descriptor in descriptors:
        with pytest.raises(OSError) as raised:
            os.fstat(descriptor)
        assert raised.value.errno == errno.EBADF


def test_task4_case_open_failure_removes_just_created_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import video_demo.evaluation.real_media_source as source_module

    runtime = tmp_path / ".codex/video-rag-demo"
    runtime.mkdir(parents=True)

    def reject_final_identity(_session: object) -> None:
        raise VideoDemoError(
            ErrorCode.WORKSPACE_PATH_ESCAPE,
            "controlled final identity failure",
        )

    monkeypatch.setattr(
        source_module.CaseExecutionSession,
        "assert_current",
        reject_final_identity,
    )

    with pytest.raises(VideoDemoError):
        source_module.open_case_execution_session(
            runtime,
            "run-1",
            "normal_audio",
        )

    assert not (runtime / "eval/generated/run-1/normal_audio").exists()


def test_task4_source_session_fails_closed_without_required_posix_capability(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import video_demo.evaluation.real_media_source as source_module

    runtime = tmp_path / ".codex/video-rag-demo"
    runtime.mkdir(parents=True)
    monkeypatch.delattr(source_module.os, "O_NOFOLLOW")

    with pytest.raises(VideoDemoError) as raised:
        source_module.open_case_execution_session(
            runtime,
            "run-1",
            "normal_audio",
        )

    assert raised.value.code == ErrorCode.INVALID_CONFIGURATION
    assert not (runtime / "eval").exists()


@pytest.mark.parametrize("race", ("parent", "target", "temporary"))
def test_task4_source_publish_rejects_last_syscall_races(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    race: str,
) -> None:
    from video_demo.evaluation.real_media_source import (
        generate_source,
        open_case_execution_session,
    )

    runtime = tmp_path / ".codex/video-rag-demo"
    runtime.mkdir(parents=True)
    case_root = runtime / "eval/generated/run-1/normal_audio"
    moved_root = case_root.with_name("normal_audio-moved")
    outside = tmp_path / "outside-publish"
    outside.mkdir()
    outside_target = outside / "source.mp4"
    outside_target.write_bytes(b"outside-sentinel")
    alternate = _media_bytes("MP4") + b"raced"

    class ControlledRunner:
        def run(
            self,
            args: Sequence[str],
            *,
            timeout_seconds: int,
            pass_fds: tuple[int, ...] = (),
        ) -> ProcessResult:
            assert type(args) is list and timeout_seconds == 7
            os.write(pass_fds[-1], _media_bytes("MP4"))
            return ProcessResult(0, b"", b"")

    real_replace = os.replace
    real_rename = os.rename
    real_link = os.link
    injected = False

    def inject_before_publish(source: object, destination: object) -> None:
        nonlocal injected
        if injected:
            return
        injected = True
        if race == "parent":
            real_rename(case_root, moved_root)
            case_root.symlink_to(outside, target_is_directory=True)
            (outside / Path(str(source)).name).write_bytes(_media_bytes("MP4"))
        elif race == "target":
            target = case_root / Path(str(destination)).name
            target.symlink_to(outside_target)
        else:
            temporary = case_root / Path(str(source)).name
            backup = temporary.with_name(f"{temporary.name}.raced-backup")
            real_rename(temporary, backup)
            temporary.write_bytes(alternate)

    def racing_replace(source: object, destination: object) -> None:
        inject_before_publish(source, destination)
        real_replace(source, destination)

    def racing_rename(
        source: object,
        destination: object,
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
    ) -> None:
        inject_before_publish(source, destination)
        real_rename(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
        )

    def racing_link(
        source: object,
        destination: object,
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
        follow_symlinks: bool = True,
    ) -> None:
        inject_before_publish(source, destination)
        real_link(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
            follow_symlinks=follow_symlinks,
        )

    session = open_case_execution_session(runtime, "run-1", "normal_audio")
    monkeypatch.setattr(os, "replace", racing_replace)
    monkeypatch.setattr(os, "rename", racing_rename)
    monkeypatch.setattr(os, "link", racing_link)
    try:
        with pytest.raises(VideoDemoError):
            generate_source(
                session=session,
                case_id="normal_audio",
                executable=tmp_path / "ffmpeg",
                runner=ControlledRunner(),
                max_bytes=1_024,
                timeout_seconds=7,
            )
    finally:
        session.cleanup_unregistered(set())
        session.close()

    assert injected
    assert outside_target.read_bytes() == b"outside-sentinel"
    assert not outside_target.is_symlink()


@pytest.mark.parametrize("replaced_level", ("case", "run", "generated"))
def test_task4_source_runner_writes_only_preopened_leaf_when_tree_is_replaced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replaced_level: str,
) -> None:
    import video_demo.evaluation.real_media_source as source_module

    runtime = tmp_path / ".codex/video-rag-demo"
    runtime.mkdir(parents=True)
    session = source_module.open_case_execution_session(
        runtime,
        "run-1",
        "normal_audio",
    )
    case_root = runtime / "eval/generated/run-1/normal_audio"
    replaced = {
        "case": case_root,
        "run": case_root.parent,
        "generated": case_root.parent.parent,
    }[replaced_level]
    moved = replaced.with_name(f"{replaced.name}-owned")
    outside = tmp_path / "outside-runner-race"
    outside.mkdir()
    sentinel = outside / "sentinel.bin"
    sentinel.write_bytes(b"outside-sentinel")
    replacement_leaf = replaced / "source.mp4"
    actual_argv: list[str] = []

    class RacingRunner:
        def run(
            self,
            args: Sequence[str],
            *,
            timeout_seconds: int,
            pass_fds: tuple[int, ...] = (),
        ) -> ProcessResult:
            assert timeout_seconds == 7
            assert len(pass_fds) == 1
            actual_argv.extend(args)
            replaced.rename(moved)
            replaced.mkdir(parents=True)
            replacement_leaf.parent.mkdir(parents=True, exist_ok=True)
            replacement_leaf.write_bytes(b"replacement-tree")
            os.write(pass_fds[0], _media_bytes("MP4"))
            return ProcessResult(0, b"", b"")

    try:
        with pytest.raises(VideoDemoError):
            source_module.generate_source(
                session=session,
                case_id="normal_audio",
                executable=tmp_path / "ffmpeg",
                runner=RacingRunner(),
                max_bytes=1_024,
                timeout_seconds=7,
            )
    finally:
        session.cleanup_unregistered(set())
        session.close()

    assert actual_argv[-3:-1] == ["-f", "mp4"]
    assert actual_argv[-1].startswith("/dev/fd/")
    assert replacement_leaf.read_bytes() == b"replacement-tree"
    assert sentinel.read_bytes() == b"outside-sentinel"
    assert not (outside / "source.mp4").exists()


@pytest.mark.parametrize(
    ("case_id", "expected_calls"),
    (("normal_audio", 1), ("no_audio", 1), ("rotation", 2), ("vfr", 1)),
)
def test_task4_source_generator_preserves_each_case_command_semantics(
    tmp_path: Path,
    case_id: str,
    expected_calls: int,
) -> None:
    from video_demo.evaluation.real_media_source import (
        generate_source,
        open_case_execution_session,
    )

    runtime = tmp_path / ".codex/video-rag-demo"
    runtime.mkdir(parents=True)
    calls: list[list[str]] = []

    class ControlledRunner:
        def run(
            self,
            args: Sequence[str],
            *,
            timeout_seconds: int,
            pass_fds: tuple[int, ...] = (),
        ) -> ProcessResult:
            assert type(args) is list and timeout_seconds == 7
            calls.append(list(args))
            os.write(pass_fds[-1], _media_bytes("MP4"))
            return ProcessResult(0, b"", b"")

    session = open_case_execution_session(runtime, "run-1", case_id)
    try:
        generate_source(
            session=session,
            case_id=case_id,
            executable=tmp_path / "ffmpeg",
            runner=ControlledRunner(),
            max_bytes=1_024,
            timeout_seconds=7,
        )
    finally:
        session.close()

    flattened = tuple(argument for argv in calls for argument in argv)
    assert len(calls) == expected_calls
    assert any("testsrc2=" in argument for argument in flattened)
    assert any("sine=" in argument for argument in flattened) == (case_id != "no_audio")
    assert ("rotate=90" in flattened) == (case_id == "rotation")
    if case_id == "vfr":
        video_filter = flattened[flattened.index("-vf") + 1]
        assert "select=" in video_filter and "eq(" in video_filter
        assert "-fps_mode" in flattened and "vfr" in flattened


@pytest.mark.parametrize("first_outcome", ("nonzero", "exception", "success"))
def test_task4_rotation_closes_each_owned_output_descriptor_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    first_outcome: str,
) -> None:
    import video_demo.evaluation.real_media_source as source_module

    runtime = tmp_path / ".codex/video-rag-demo"
    runtime.mkdir(parents=True)
    tracked: set[int] = set()
    close_counts: dict[int, int] = {}
    real_close = source_module.os.close

    def reject_duplicate_close(descriptor: int) -> None:
        if descriptor in tracked:
            close_counts[descriptor] = close_counts.get(descriptor, 0) + 1
            if close_counts[descriptor] > 1:
                raise AssertionError(f"owned descriptor closed twice: {descriptor}")
        real_close(descriptor)

    class ControlledRunner:
        def __init__(self) -> None:
            self.calls = 0

        def run(
            self,
            _args: Sequence[str],
            *,
            timeout_seconds: int,
            pass_fds: tuple[int, ...] = (),
        ) -> ProcessResult:
            assert timeout_seconds == 7
            self.calls += 1
            tracked.update(pass_fds)
            if self.calls == 1 and first_outcome == "exception":
                raise RuntimeError("controlled first-stage failure")
            if self.calls == 1 and first_outcome == "nonzero":
                return ProcessResult(3, b"", b"")
            os.write(pass_fds[-1], _media_bytes("MP4"))
            return ProcessResult(0, b"", b"")

    runner = ControlledRunner()
    session = source_module.open_case_execution_session(
        runtime,
        "run-1",
        "rotation",
    )
    monkeypatch.setattr(source_module.os, "close", reject_duplicate_close)
    try:
        if first_outcome == "exception":
            with pytest.raises(RuntimeError, match="controlled first-stage failure"):
                source_module.generate_source(
                    session=session,
                    case_id="rotation",
                    executable=tmp_path / "ffmpeg",
                    runner=runner,
                    max_bytes=1_024,
                    timeout_seconds=7,
                )
        else:
            generated = source_module.generate_source(
                session=session,
                case_id="rotation",
                executable=tmp_path / "ffmpeg",
                runner=runner,
                max_bytes=1_024,
                timeout_seconds=7,
            )
            assert generated.process_result.returncode == (
                3 if first_outcome == "nonzero" else 0
            )
    finally:
        try:
            session.close()
        finally:
            monkeypatch.setattr(source_module.os, "close", real_close)

    assert close_counts == {descriptor: 1 for descriptor in tracked}
    assert len(tracked) == (2 if first_outcome == "success" else 1)
    assert runner.calls == (2 if first_outcome == "success" else 1)


@pytest.mark.parametrize("failure_point", ("bind", "record"))
def test_task4_media_transfer_rolls_back_until_journal_accepts_ownership(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_point: str,
) -> None:
    from video_demo.evaluation.real_media_execution import transfer_media_file

    runtime = tmp_path / ".codex/video-rag-demo"
    runtime.mkdir(parents=True)
    relative = Path("eval/generated/run-1/normal_audio/source.mp4")
    source = runtime / relative
    source.parent.mkdir(parents=True)
    source.write_bytes(_media_bytes("MP4"))
    store = EvidenceStore(tmp_path, runtime)

    class Journal:
        def __init__(self) -> None:
            self.registered: set[str] = set()

        def record_media_file(self, *_args: object) -> None:
            if failure_point == "record":
                raise RuntimeError("record fault")

        def is_media_file_registered(
            self, _case_id: str, relative_path: str
        ) -> bool:
            return relative_path in self.registered

    if failure_point == "bind":
        monkeypatch.setattr(
            store,
            "bind_artifact",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("bind fault")),
        )

    with pytest.raises((RuntimeError, ValueError)):
        transfer_media_file(
            case_id="normal_audio",
            path=source,
            role="SOURCE",
            format_name="MP4",
            store=store,
            journal=Journal(),
            max_bytes=1_024,
        )

    assert not source.exists()


def test_task4_media_transfer_failure_never_deletes_previously_registered_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from video_demo.evaluation.real_media_execution import transfer_media_file

    runtime = tmp_path / ".codex/video-rag-demo"
    runtime.mkdir(parents=True)
    first_relative = Path("eval/generated/run-1/normal_audio/source.mp4")
    second_relative = Path("eval/generated/run-1/normal_audio/media/proxy.mp4")
    first = runtime / first_relative
    second = runtime / second_relative
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.write_bytes(_media_bytes("MP4"))
    second.write_bytes(_media_bytes("MP4"))
    store = EvidenceStore(tmp_path, runtime)
    recorded = 0

    class Journal:
        def __init__(self) -> None:
            self.registered: set[str] = set()

        def record_media_file(self, *_args: object) -> None:
            nonlocal recorded
            recorded += 1
            if recorded == 2:
                raise RuntimeError("second record fault")
            media_file = _args[1]
            assert isinstance(media_file, RealMediaFile)
            self.registered.add(media_file.relative_path)

        def is_media_file_registered(
            self, _case_id: str, relative_path: str
        ) -> bool:
            return relative_path in self.registered

    journal = Journal()
    transfer_media_file(
        case_id="normal_audio",
        path=first,
        role="SOURCE",
        format_name="MP4",
        store=store,
        journal=journal,
        max_bytes=1_024,
    )
    with pytest.raises(RuntimeError):
        transfer_media_file(
            case_id="normal_audio",
            path=second,
            role="PROXY",
            format_name="MP4",
            store=store,
            journal=journal,
            max_bytes=1_024,
        )

    assert first.is_file()
    assert not second.exists()
