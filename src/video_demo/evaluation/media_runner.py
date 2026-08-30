from __future__ import annotations

import importlib.util
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from video_demo.capabilities import resolve_workspace_binary
from video_demo.config import Settings
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.evaluation.evidence import (
    ArtifactRole,
    CommandTrace,
    EvidenceKind,
    EvidenceLevel,
    EvidenceStore,
    MachineEvidenceReport,
    PreflightDetails,
    PreflightIssue,
    PreflightRawReport,
    RealMediaCommand,
    RealMediaDetails,
    RealMediaFile,
    RealMediaRawReport,
    RealMediaSample,
    ReportRunWriter,
    SetupMediaCommand,
    TraceArtifact,
    build_verified_gate_check,
)
from video_demo.evaluation.gate import GateCheck, _current_real_media_implementation_sha256
from video_demo.evaluation.report import GateStatus
from video_demo.media.process import SafeProcessRunner
from video_demo.storage.workspace import validate_path_component

_CHECK_ID = "real_media_chain"
_NOT_RUN_REASON = "缺少工作区 FFmpeg/ffprobe 与真实媒体运行结果"
_CASE_IDS = ("normal_audio", "no_audio", "rotation", "vfr")
_VERSION_TOKEN = re.compile(r"^[A-Za-z0-9._+-]{1,128}$")
_MediaPhase = Literal[
    "generate",
    "probe",
    "audio",
    "opencv_decode",
    "scene_detect",
    "keyframe_select",
]
_MediaExecutable = Literal[
    "ffmpeg",
    "ffprobe",
    "FFmpegTranscoder",
    "OpenCvFrameExtractor",
    "PySceneDetectAdapter",
    "KeyframeSelector",
]
_MEDIA_PHASE_SEQUENCE: tuple[_MediaPhase, ...] = (
    "generate",
    "probe",
    "audio",
    "opencv_decode",
    "scene_detect",
    "keyframe_select",
)
_MEDIA_PHASE_EXECUTABLES: dict[_MediaPhase, _MediaExecutable] = {
    "generate": "ffmpeg",
    "probe": "ffprobe",
    "audio": "FFmpegTranscoder",
    "opencv_decode": "OpenCvFrameExtractor",
    "scene_detect": "PySceneDetectAdapter",
    "keyframe_select": "KeyframeSelector",
}


@dataclass(frozen=True, slots=True)
class _MediaExecutionSuccess:
    """任务 4 端口交回的完整样本事实与已绑定产物。"""

    samples: tuple[RealMediaSample, ...]
    artifacts: tuple[TraceArtifact, ...]


@dataclass(frozen=True, slots=True)
class _MediaExecutionFailure:
    """任务 4 端口交回的完整 partial 事实与稳定失败码。"""

    samples: tuple[RealMediaSample, ...]
    artifacts: tuple[TraceArtifact, ...]
    failure_code: ErrorCode


_MediaExecutionResult = _MediaExecutionFailure | _MediaExecutionSuccess


class _MediaExecutionJournal:
    """任务 4 的事实登记上下文，异常时可忠实恢复 partial FAIL。"""

    def __init__(self, writer: ReportRunWriter) -> None:
        self._writer = writer
        self._samples: list[RealMediaSample] = []
        self._commands: list[RealMediaCommand] = []
        self._draft_artifacts: list[TraceArtifact] = []
        self._artifacts: list[TraceArtifact] = []
        self._media_registrations: list[
            tuple[str, RealMediaFile, TraceArtifact]
        ] = []
        self._current: tuple[
            str,
            _MediaPhase,
            _MediaExecutable,
            tuple[str, ...],
            tuple[str, ...],
            tuple[str, ...],
        ] | None = None
        self._current_outputs: tuple[TraceArtifact, TraceArtifact] | None = None
        self._next_phase: tuple[str, _MediaPhase, _MediaExecutable] | None = None
        self._sample_finalizing_case_id: str | None = None

    def write_artifact(
        self, filename: str, role: ArtifactRole, payload: bytes
    ) -> TraceArtifact:
        if self._current is None or role not in {
            "COMMAND_STDOUT",
            "COMMAND_STDERR",
        }:
            raise VideoDemoError(ErrorCode.SYSTEM_FAILURE, "命令输出与当前媒体阶段不匹配")
        case_id, phase, *_rest = self._current
        suffix = "stdout" if role == "COMMAND_STDOUT" else "stderr"
        if filename != f"{case_id}-{phase}.{suffix}.txt":
            raise VideoDemoError(ErrorCode.SYSTEM_FAILURE, "命令输出文件名与当前媒体阶段不匹配")
        artifact = self._writer.write_artifact(filename, role, payload)
        self._remember_artifacts((artifact,))
        return artifact

    def record_completed_sample(
        self,
        sample: RealMediaSample,
        artifacts: tuple[TraceArtifact, ...],
    ) -> None:
        expected_case = self._expected_case()
        draft_files = self._media_files_for_case(expected_case)
        draft_artifacts = (
            *self._draft_artifacts,
            *(
                artifact
                for registered_case, _media_file, artifact in self._media_registrations
                if registered_case == expected_case
            ),
        )
        if (
            expected_case is None
            or sample.case_id != expected_case
            or sample.execution_status != "SUCCESS"
            or self._current is not None
            or self._sample_finalizing_case_id != expected_case
            or sample.commands != tuple(self._commands)
            or sample.files != draft_files
            or len(artifacts) != len(self._artifact_identities(artifacts))
            or self._artifact_identities(artifacts)
            != self._artifact_identities(draft_artifacts)
        ):
            raise VideoDemoError(ErrorCode.SYSTEM_FAILURE, "完成样本与当前媒体事实不匹配")
        self._samples.append(sample)
        self._commands.clear()
        self._draft_artifacts.clear()
        self._current = None
        self._current_outputs = None
        self._next_phase = None
        self._sample_finalizing_case_id = None

    def finalize_sample(
        self,
        sample: RealMediaSample,
        artifacts: tuple[TraceArtifact, ...],
        verify_case_closed: Callable[[], None],
    ) -> None:
        """以同一 fd 闭包校验包围 success append，后验失败时恢复可恢复状态。"""

        commands = tuple(self._commands)
        draft_artifacts = tuple(self._draft_artifacts)
        verify_case_closed()
        self.record_completed_sample(sample, artifacts)
        try:
            verify_case_closed()
        except BaseException:
            if not self._samples or self._samples[-1] != sample:
                raise VideoDemoError(ErrorCode.SYSTEM_FAILURE, "完成样本回滚状态非法") from None
            self._samples.pop()
            self._commands.extend(commands)
            self._draft_artifacts.extend(draft_artifacts)
            self._sample_finalizing_case_id = sample.case_id
            raise

    def record_media_file(
        self,
        case_id: str,
        media_file: RealMediaFile,
        artifact: TraceArtifact,
    ) -> None:
        if self._current is None or case_id != self._current[0]:
            raise VideoDemoError(ErrorCode.SYSTEM_FAILURE, "媒体文件与当前 case 不匹配")
        expected_file: dict[_MediaPhase, tuple[str, str] | None] = {
            "generate": ("SOURCE", "MP4"),
            "probe": None,
            "audio": ("AUDIO", "WAV"),
            "opencv_decode": None,
            "scene_detect": None,
            "keyframe_select": ("KEYFRAME", "JPEG"),
        }
        artifact_role: ArtifactRole = (
            "INPUT_MEDIA" if media_file.role == "SOURCE" else "OUTPUT_MEDIA"
        )
        prefix = (
            f".codex/video-rag-demo/eval/generated/{self._writer.evaluation_run_id}/"
            f"{case_id}/"
        )
        if (
            expected_file[self._current[1]]
            != (media_file.role, media_file.format)
            or (case_id == "no_audio" and media_file.role == "AUDIO")
            or artifact.role != artifact_role
            or artifact.relative_path != media_file.relative_path
            or artifact.sha256 != media_file.sha256
            or not media_file.relative_path.startswith(prefix)
            or media_file.relative_path not in self._current[5]
            or self.is_media_file_registered(case_id, media_file.relative_path)
            or any(
                existing.relative_path == artifact.relative_path
                for existing in self._artifacts
            )
        ):
            raise VideoDemoError(ErrorCode.SYSTEM_FAILURE, "媒体文件登记事实非法")
        self._media_registrations.append((case_id, media_file, artifact))

    def is_media_file_registered(self, case_id: str, relative_path: str) -> bool:
        """返回媒体是否已越过 journal 的唯一所有权提交点。"""

        return any(
            registered_case == case_id
            and media_file.relative_path == relative_path
            for registered_case, media_file, _artifact in self._media_registrations
        )

    def record_completed_command(
        self,
        command: RealMediaCommand,
        artifacts: tuple[TraceArtifact, ...],
    ) -> None:
        if self._current is None or (
            command.phase,
            command.executable,
            command.arguments,
            command.input_relative_paths,
            command.output_relative_paths,
        ) != (
            self._current[1],
            self._current[2],
            self._current[3],
            self._current[4],
            self._current[5],
        ):
            raise VideoDemoError(ErrorCode.SYSTEM_FAILURE, "完成命令与当前媒体阶段不匹配")
        if not self._command_outputs_match(command, artifacts):
            raise VideoDemoError(ErrorCode.SYSTEM_FAILURE, "媒体命令输出登记事实非法")
        self._remember_artifacts(artifacts)
        self._draft_artifacts.extend(artifacts)
        if command.exit_code != 0:
            if (
                len(artifacts) == 2
                and artifacts[0].role == "COMMAND_STDOUT"
                and artifacts[1].role == "COMMAND_STDERR"
            ):
                self._current_outputs = (artifacts[0], artifacts[1])
            raise VideoDemoError(ErrorCode.VIDEO_PROCESS_FAILED, "媒体阶段执行失败")
        case_id = self._current[0]
        self._commands.append(command)
        phase_index = _MEDIA_PHASE_SEQUENCE.index(command.phase)
        self._next_phase = None
        self._sample_finalizing_case_id = None
        if phase_index + 1 < len(_MEDIA_PHASE_SEQUENCE):
            phase = _MEDIA_PHASE_SEQUENCE[phase_index + 1]
            self._next_phase = (case_id, phase, _MEDIA_PHASE_EXECUTABLES[phase])
        else:
            self._sample_finalizing_case_id = case_id
        self._current = None
        self._current_outputs = None

    def begin_phase(
        self,
        *,
        case_id: str,
        phase: _MediaPhase,
        executable: _MediaExecutable,
        arguments: tuple[str, ...] = (),
        input_relative_paths: tuple[str, ...] = (),
        output_relative_paths: tuple[str, ...] = (),
    ) -> None:
        expected_case = self._expected_case()
        expected_phase = (
            _MEDIA_PHASE_SEQUENCE[len(self._commands)]
            if len(self._commands) < len(_MEDIA_PHASE_SEQUENCE)
            else None
        )
        if (
            expected_case is None
            or case_id != expected_case
            or self._current is not None
            or self._sample_finalizing_case_id is not None
            or phase != expected_phase
            or executable != _MEDIA_PHASE_EXECUTABLES[phase]
            or (
                self._next_phase is not None
                and self._next_phase != (case_id, phase, executable)
            )
        ):
            raise VideoDemoError(ErrorCode.SYSTEM_FAILURE, "媒体阶段登记顺序非法")
        self._current = (
            case_id,
            phase,
            executable,
            arguments,
            input_relative_paths,
            output_relative_paths,
        )
        self._current_outputs = None
        self._next_phase = None
        self._sample_finalizing_case_id = None

    def write_current_outputs(self) -> tuple[TraceArtifact, TraceArtifact]:
        if self._current is None:
            raise VideoDemoError(ErrorCode.SYSTEM_FAILURE, "媒体阶段尚未登记")
        if self._current_outputs is None:
            case_id, phase, *_rest = self._current
            stdout_name = f"{case_id}-{phase}.stdout.txt"
            stderr_name = f"{case_id}-{phase}.stderr.txt"
            existing = {
                artifact.relative_path.rsplit("/", 1)[-1]: artifact
                for artifact in self._artifacts
            }
            stdout = existing.get(stdout_name)
            stderr = existing.get(stderr_name)
            if stdout is None:
                stdout = self.write_artifact(stdout_name, "COMMAND_STDOUT", b"")
            if stderr is None:
                stderr = self.write_artifact(stderr_name, "COMMAND_STDERR", b"")
            self._current_outputs = (stdout, stderr)
        return self._current_outputs

    def recover_failure(self) -> _MediaExecutionFailure:
        if self._current is None:
            if self._sample_finalizing_case_id is not None:
                return self._recover_sample_finalization_failure()
            if self._next_phase is not None:
                case_id, phase, executable = self._next_phase
                self.begin_phase(
                    case_id=case_id,
                    phase=phase,
                    executable=executable,
                )
            else:
                next_index = len(self._samples)
                if next_index == len(_CASE_IDS):
                    return self._recover_completed_run_finalization_failure()
                case_id = _CASE_IDS[next_index]
                self.begin_phase(
                    case_id=case_id, phase="generate", executable="ffmpeg"
                )
        assert self._current is not None
        case_id, phase, executable, arguments, inputs, outputs = self._current
        stdout, stderr = self.write_current_outputs()
        failed_command = RealMediaCommand(
            phase=phase,
            executable=executable,
            arguments=arguments,
            input_relative_paths=inputs,
            output_relative_paths=outputs,
            exit_code=1,
            stdout_relative_path=stdout.relative_path,
            stderr_relative_path=stderr.relative_path,
            stdout_sha256=stdout.sha256,
            stderr_sha256=stderr.sha256,
        )
        failed_index = _CASE_IDS.index(case_id)
        failed_sample = RealMediaSample(
            case_id=case_id,
            execution_status="FAILED",
            failure_code=ErrorCode.VIDEO_PROCESS_FAILED,
            files=self._media_files_for_case(case_id),
            commands=(*self._commands, failed_command),
        )
        samples = (
            *self._samples,
            failed_sample,
            *(
                RealMediaSample(case_id=case, execution_status="NOT_STARTED")
                for case in _CASE_IDS[failed_index + 1 :]
            ),
        )
        return _MediaExecutionFailure(
            samples=samples,
            artifacts=self._all_artifacts(),
            failure_code=ErrorCode.VIDEO_PROCESS_FAILED,
        )

    def _recover_sample_finalization_failure(self) -> _MediaExecutionFailure:
        assert self._sample_finalizing_case_id is not None
        case_id = self._sample_finalizing_case_id
        failed_index = _CASE_IDS.index(case_id)
        failed_sample = RealMediaSample(
            case_id=case_id,
            execution_status="FAILED",
            failure_code=ErrorCode.VIDEO_PROCESS_FAILED,
            files=self._media_files_for_case(case_id),
            commands=tuple(self._commands),
        )
        return _MediaExecutionFailure(
            samples=(
                *self._samples,
                failed_sample,
                *(
                    RealMediaSample(case_id=case, execution_status="NOT_STARTED")
                    for case in _CASE_IDS[failed_index + 1 :]
                ),
            ),
            artifacts=self._all_artifacts(),
            failure_code=ErrorCode.VIDEO_PROCESS_FAILED,
        )

    def _recover_completed_run_finalization_failure(self) -> _MediaExecutionFailure:
        if len(self._samples) != len(_CASE_IDS):
            raise VideoDemoError(ErrorCode.SYSTEM_FAILURE, "媒体执行恢复状态非法")
        completed = self._samples[-1]
        failed = RealMediaSample.model_validate(
            {
                **completed.model_dump(mode="python"),
                "execution_status": "FAILED",
                "failure_code": ErrorCode.VIDEO_PROCESS_FAILED,
            }
        )
        return _MediaExecutionFailure(
            samples=(*self._samples[:-1], failed),
            artifacts=self._all_artifacts(),
            failure_code=ErrorCode.VIDEO_PROCESS_FAILED,
        )

    def _expected_case(self) -> str | None:
        index = len(self._samples)
        return _CASE_IDS[index] if index < len(_CASE_IDS) else None

    def _media_files_for_case(self, case_id: str | None) -> tuple[RealMediaFile, ...]:
        return tuple(
            media_file
            for registered_case, media_file, _artifact in self._media_registrations
            if registered_case == case_id
        )

    def _all_artifacts(self) -> tuple[TraceArtifact, ...]:
        return (
            *self._artifacts,
            *(registration[2] for registration in self._media_registrations),
        )

    @staticmethod
    def _artifact_identities(
        artifacts: tuple[TraceArtifact, ...],
    ) -> set[tuple[ArtifactRole, str, str]]:
        return {
            (artifact.role, artifact.relative_path, artifact.sha256)
            for artifact in artifacts
        }

    @staticmethod
    def _command_outputs_match(
        command: RealMediaCommand,
        artifacts: tuple[TraceArtifact, ...],
    ) -> bool:
        return len(artifacts) == 2 and (
            artifacts[0].role,
            artifacts[0].relative_path,
            artifacts[0].sha256,
            artifacts[1].role,
            artifacts[1].relative_path,
            artifacts[1].sha256,
        ) == (
            "COMMAND_STDOUT",
            command.stdout_relative_path,
            command.stdout_sha256,
            "COMMAND_STDERR",
            command.stderr_relative_path,
            command.stderr_sha256,
        )

    def _remember_artifacts(self, artifacts: tuple[TraceArtifact, ...]) -> None:
        seen = {artifact.relative_path for artifact in self._artifacts}
        for artifact in artifacts:
            if artifact.relative_path not in seen:
                self._artifacts.append(artifact)
                seen.add(artifact.relative_path)


class RealMediaRunner:
    """真实媒体门禁的前置条件与失败闭环；不包含媒体生产编排。"""

    def __init__(self, settings: Settings, evidence_store: EvidenceStore) -> None:
        self._settings = settings
        self._store = evidence_store
        self._run_writer: ReportRunWriter | None = None

    def run(self, *, evaluation_run_id: str) -> GateCheck:
        validate_path_component(evaluation_run_id, "evaluation_run_id")
        report_relative = Path("eval/reports") / evaluation_run_id / "real-media.json"
        report_path = self._store.runtime_root / report_relative
        if report_path.exists():
            return build_verified_gate_check(
                _CHECK_ID, report_path, workspace_root=self._settings.workspace_root
            )
        report_root = report_path.parent
        if report_root.exists():
            raise VideoDemoError(ErrorCode.IDEMPOTENCY_CONFLICT, "运行目录已有不完整证据")
        try:
            writer = self._store.open_exclusive_report_run(evaluation_run_id)
        except ValueError:
            raise VideoDemoError(ErrorCode.IDEMPOTENCY_CONFLICT, "运行目录已有不完整证据") from None
        self._run_writer = writer
        try:
            binaries, issues = self._preflight()
            if issues:
                return self._write_not_run(evaluation_run_id, issues)
            return self._write_version_failure(evaluation_run_id, binaries)
        finally:
            self._run_writer = None
            writer.close()

    def _preflight(self) -> tuple[dict[str, Path], tuple[ErrorCode, ...]]:
        binaries: dict[str, Path] = {}
        issues: list[ErrorCode] = []
        for name, code in (
            ("ffmpeg", ErrorCode.VIDEO_FFMPEG_UNAVAILABLE),
            ("ffprobe", ErrorCode.VIDEO_FFPROBE_UNAVAILABLE),
        ):
            path = self._resolve_binary(name)
            if path is None:
                issues.append(code)
            else:
                binaries[name] = path
        if (
            importlib.util.find_spec("cv2") is None
            or importlib.util.find_spec("scenedetect") is None
        ):
            issues.append(ErrorCode.VISUAL_DEPENDENCY_UNAVAILABLE)
        return binaries, tuple(issues)

    def _resolve_binary(self, name: str) -> Path | None:
        configured = (
            self._settings.ffmpeg_path if name == "ffmpeg" else self._settings.ffprobe_path
        )
        assert self._settings.runtime_root is not None
        candidate = configured or self._settings.runtime_root / "tools" / name
        code = (
            ErrorCode.VIDEO_FFMPEG_UNAVAILABLE
            if name == "ffmpeg"
            else ErrorCode.VIDEO_FFPROBE_UNAVAILABLE
        )
        try:
            return resolve_workspace_binary(
                candidate,
                workspace_root=self._settings.workspace_root,
                unavailable_code=code,
            )
        except VideoDemoError:
            return None

    def _write_not_run(
        self, evaluation_run_id: str, issues: tuple[ErrorCode, ...]
    ) -> GateCheck:
        implementation = _current_real_media_implementation_sha256(
            self._settings.workspace_root
        )
        raw = PreflightRawReport(
            schema_version="1.0.0",
            check_id=_CHECK_ID,
            reason_code="REAL_MEDIA_CHAIN_UNAVAILABLE",
            execution_started=False,
            issues=tuple(PreflightIssue(code=code) for code in issues),
            implementation_sha256=implementation,
            evaluation_run_id=evaluation_run_id,
        )
        writer = self._writer()
        raw_artifact = writer.write_artifact(
            "preflight.json", "AUDIT_REPORT", raw.model_dump_json().encode("utf-8")
        )
        stdout = writer.write_artifact("trace.stdout.txt", "COMMAND_STDOUT", b"")
        stderr = writer.write_artifact("trace.stderr.txt", "COMMAND_STDERR", b"")
        trace = CommandTrace(
            command=("python", "-m", "video_demo.evaluation.media_runner"),
            exit_code=0,
            stdout_sha256=stdout.sha256,
            stderr_sha256=stderr.sha256,
        )
        report = MachineEvidenceReport(
            schema_version="1.0.0",
            check_id=_CHECK_ID,
            status=GateStatus.NOT_RUN,
            kind=EvidenceKind.COMMAND_REPORT,
            level=EvidenceLevel.REAL_MEDIA,
            covered_items=(_CHECK_ID,),
            summary="真实媒体整体前置条件不足",
            producer="RealMediaRunner",
            started_at=_now(),
            finished_at=_now(),
            not_run_reason=_NOT_RUN_REASON,
            artifacts=(raw_artifact, stdout, stderr),
            details=PreflightDetails(
                type="PREFLIGHT", trace=trace, preflight_report_sha256=raw_artifact.sha256
            ),
        )
        return writer.write_json(report)

    def _write_version_failure(
        self, evaluation_run_id: str, binaries: dict[str, Path]
    ) -> GateCheck:
        writer = self._writer()
        commands: list[SetupMediaCommand] = []
        command_artifacts: list[TraceArtifact] = []
        versions: dict[str, str | None] = {"ffmpeg": None, "ffprobe": None}
        for name in ("ffmpeg", "ffprobe"):
            try:
                exit_code, stdout_data, stderr_data = self._run_version(name, binaries[name])
            except (OSError, VideoDemoError):
                exit_code, stdout_data, stderr_data = 1, b"", b""
            version = _version_line(name, stdout_data)
            if exit_code == 0 and version is None:
                exit_code = 1
            stdout = writer.write_artifact(
                f"{name}.stdout.txt", "COMMAND_STDOUT", _safe_output(stdout_data)
            )
            stderr = writer.write_artifact(
                f"{name}.stderr.txt", "COMMAND_STDERR", _safe_output(stderr_data)
            )
            command_artifacts.extend((stdout, stderr))
            commands.append(
                SetupMediaCommand(
                    phase=f"{name}_version",
                    executable=name,
                    exit_code=exit_code,
                    stdout_relative_path=stdout.relative_path,
                    stderr_relative_path=stderr.relative_path,
                    stdout_sha256=stdout.sha256,
                    stderr_sha256=stderr.sha256,
                )
            )
            if exit_code != 0:
                return self._persist_real_failure(
                    evaluation_run_id, tuple(commands), versions, tuple(command_artifacts)
                )
            versions[name] = version
        journal = _MediaExecutionJournal(writer)
        try:
            execution = self._execute_media_port(evaluation_run_id, binaries, journal)
        except Exception:
            execution = journal.recover_failure()
        return self._persist_media_execution(
            evaluation_run_id,
            tuple(commands),
            versions,
            tuple(command_artifacts),
            execution,
        )

    def _execute_media_port(
        self,
        evaluation_run_id: str,
        binaries: dict[str, Path],
        journal: _MediaExecutionJournal,
    ) -> _MediaExecutionResult:
        """薄委托真实执行模块，并适配任务 3 已冻结的私有结果。"""

        from video_demo.evaluation.real_media_execution import execute_real_media

        facts = execute_real_media(
            evaluation_run_id=evaluation_run_id,
            binaries=binaries,
            settings=self._settings,
            store=self._store,
            journal=journal,
        )
        return _MediaExecutionSuccess(samples=facts.samples, artifacts=facts.artifacts)

    def _persist_media_execution(
        self,
        evaluation_run_id: str,
        setup_commands: tuple[SetupMediaCommand, ...],
        versions: dict[str, str | None],
        setup_artifacts: tuple[TraceArtifact, ...],
        execution: _MediaExecutionResult,
    ) -> GateCheck:
        """按端口原样持久化完整成功或 partial 失败事实。"""

        successful = isinstance(execution, _MediaExecutionSuccess)
        failure_code = (
            None
            if isinstance(execution, _MediaExecutionSuccess)
            else execution.failure_code
        )
        writer = self._writer()
        implementation = _current_real_media_implementation_sha256(
            self._settings.workspace_root
        )
        raw = RealMediaRawReport(
            schema_version="1.0.0",
            status=GateStatus.PASS if successful else GateStatus.FAIL,
            trace_exit_code=0 if successful else 1,
            evaluation_run_id=evaluation_run_id,
            ffmpeg_version=versions["ffmpeg"],
            ffprobe_version=versions["ffprobe"],
            implementation_sha256=implementation,
            setup_commands=setup_commands,
            samples=execution.samples,
            failure_code=failure_code,
        )
        raw_artifact = writer.write_artifact(
            "raw.json", "AUDIT_REPORT", raw.model_dump_json().encode("utf-8")
        )
        trace_stdout, trace_stderr = self._write_trace_artifacts(successful=successful)
        trace = CommandTrace(
            command=("python", "-m", "video_demo.evaluation.media_runner"),
            exit_code=0 if successful else 1,
            stdout_sha256=trace_stdout.sha256,
            stderr_sha256=trace_stderr.sha256,
        )
        report = MachineEvidenceReport(
            schema_version="1.0.0",
            check_id=_CHECK_ID,
            status=GateStatus.PASS if successful else GateStatus.FAIL,
            kind=EvidenceKind.COMMAND_REPORT,
            level=EvidenceLevel.REAL_MEDIA,
            covered_items=(_CHECK_ID,),
            summary=("真实媒体生产链执行完成" if successful else "真实媒体生产链执行失败"),
            producer="RealMediaRunner",
            started_at=_now(),
            finished_at=_now(),
            artifacts=(
                raw_artifact,
                *setup_artifacts,
                *execution.artifacts,
                trace_stdout,
                trace_stderr,
            ),
            details=RealMediaDetails(
                type="REAL_MEDIA",
                trace=trace,
                ffmpeg_version=versions["ffmpeg"],
                ffprobe_version=versions["ffprobe"],
                raw_report_sha256=raw_artifact.sha256,
                implementation_sha256=implementation,
            ),
        )
        return writer.write_json(report)

    def _run_version(self, _name: str, path: Path) -> tuple[int, bytes, bytes]:
        result = SafeProcessRunner(max_output_bytes=64 * 1024).run(
            [str(path), "-version"], timeout_seconds=self._settings.process_timeout_seconds
        )
        return result.returncode, result.stdout, result.stderr

    def _persist_real_failure(
        self,
        evaluation_run_id: str,
        commands: tuple[SetupMediaCommand, ...],
        versions: dict[str, str | None],
        command_artifacts: tuple[TraceArtifact, ...],
    ) -> GateCheck:
        writer = self._writer()
        implementation = _current_real_media_implementation_sha256(
            self._settings.workspace_root
        )
        raw = RealMediaRawReport(
            schema_version="1.0.0",
            status=GateStatus.FAIL,
            trace_exit_code=1,
            evaluation_run_id=evaluation_run_id,
            ffmpeg_version=versions["ffmpeg"],
            ffprobe_version=versions["ffprobe"],
            implementation_sha256=implementation,
            setup_commands=commands,
            samples=tuple(
                RealMediaSample(case_id=case, execution_status="NOT_STARTED")
                for case in _CASE_IDS
            ),
            failure_code=ErrorCode.VIDEO_BINARY_PROBE_FAILED,
        )
        raw_artifact = writer.write_artifact(
            "raw.json", "AUDIT_REPORT", raw.model_dump_json().encode("utf-8")
        )
        trace_stdout, trace_stderr = self._write_trace_artifacts()
        trace = CommandTrace(
            command=("python", "-m", "video_demo.evaluation.media_runner"),
            exit_code=1,
            stdout_sha256=trace_stdout.sha256,
            stderr_sha256=trace_stderr.sha256,
        )
        report = MachineEvidenceReport(
            schema_version="1.0.0",
            check_id=_CHECK_ID,
            status=GateStatus.FAIL,
            kind=EvidenceKind.COMMAND_REPORT,
            level=EvidenceLevel.REAL_MEDIA,
            covered_items=(_CHECK_ID,),
            summary="真实媒体版本探测或后续执行失败",
            producer="RealMediaRunner",
            started_at=_now(),
            finished_at=_now(),
            artifacts=(raw_artifact, *command_artifacts, trace_stdout, trace_stderr),
            details=RealMediaDetails(
                type="REAL_MEDIA",
                trace=trace,
                ffmpeg_version=versions["ffmpeg"],
                ffprobe_version=versions["ffprobe"],
                raw_report_sha256=raw_artifact.sha256,
                implementation_sha256=implementation,
            ),
        )
        return writer.write_json(report)

    def _write_trace_artifacts(
        self, *, successful: bool = False
    ) -> tuple[TraceArtifact, TraceArtifact]:
        writer = self._writer()
        return (
            writer.write_artifact(
                "trace.stdout.txt",
                "COMMAND_STDOUT",
                b"runner trace success\n" if successful else b"runner trace stdout\n",
            ),
            writer.write_artifact(
                "trace.stderr.txt", "COMMAND_STDERR", b"runner trace stderr\n"
            ),
        )

    def _writer(self) -> ReportRunWriter:
        if self._run_writer is None:
            raise VideoDemoError(ErrorCode.SYSTEM_FAILURE, "报告 run 写入会话未打开")
        return self._run_writer

def _safe_output(_value: bytes) -> bytes:
    return b""


def _version_line(name: str, value: bytes) -> str | None:
    try:
        lines = value.decode("utf-8", errors="strict").splitlines()
    except UnicodeDecodeError:
        return None
    prefix = f"{name} version "
    if not lines or not lines[0].startswith(prefix):
        return None
    token = lines[0][len(prefix) :].split(" ", 1)[0]
    return token if _VERSION_TOKEN.fullmatch(token) else None


def _now() -> datetime:
    return datetime.now(UTC)
