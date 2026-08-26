from __future__ import annotations

import hashlib
import importlib.util
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol, cast

from video_demo.application.legacy_composition import (
    ProductionDiagnosticComponents,
    ProductionModelIdentityReport,
    _normalized_qwen_model_id,
    build_production_diagnostic_components,
    build_production_model_identity_report,
)
from video_demo.config import Settings
from video_demo.domain.base import stable_identifier
from video_demo.domain.evidence import SceneBoundary
from video_demo.domain.run import ModelIdentity, TimeRange
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.evaluation.annotations import (
    ValidatedEvaluationPackage,
    load_evaluation_package,
    reverify_evaluation_package,
)
from video_demo.evaluation.dataset import ValidationLanguage
from video_demo.evaluation.evidence import (
    _LIVE_PREFLIGHT_CODES,
    ArtifactRole,
    BaiduLiveDetails,
    BaiduLiveRawReport,
    CommandTrace,
    EvidenceKind,
    EvidenceLevel,
    EvidenceStore,
    FileSnapshot,
    FiveLanguageModelsDetails,
    FiveLanguageModelsRawReport,
    LiveCheckId,
    LiveExecutionSummary,
    LiveFailureComponent,
    LiveInputArtifact,
    LiveReportRunConflict,
    LiveReportRunWriter,
    LiveSample,
    MachineEvidenceReport,
    ModelExecutionFact,
    PreflightDetails,
    PreflightIssue,
    PreflightRawReport,
    QwenLiveDetails,
    QwenLiveRawReport,
    TraceArtifact,
    _assert_snapshot_current,
    _read_file_snapshot,
    _reject_symlink_components,
    build_verified_gate_check,
    load_machine_evidence,
)
from video_demo.evaluation.gate import (
    _LIVE_COMPONENT_FAILURE_CODES,
    GateCheck,
    _current_live_implementation_sha256,
)
from video_demo.evaluation.report import GateStatus
from video_demo.fusion.timeline import build_timeline
from video_demo.integrations.video_port import SegmentUnderstandingRequest, VideoClipInput
from video_demo.storage.workspace import validate_path_component
from video_demo.visual.ocr import is_supported_ocr_image

_VALIDATION_LANGUAGES: tuple[ValidationLanguage, ...] = ("zh", "en", "ja", "ko", "es")

_CHECK_REASON: dict[str, tuple[str, str]] = {
    "baidu_ocr_live": (
        "BAIDU_OCR_CREDENTIALS_UNAVAILABLE",
        "缺少百度 OCR 凭据或真实联调结果",
    ),
    "qwen_live": (
        "QWEN_CREDENTIALS_UNAVAILABLE",
        "缺少 Qwen 凭据或真实联调结果",
    ),
    "five_language_models": (
        "FIVE_LANGUAGE_MODELS_UNAVAILABLE",
        "缺少五语授权素材、模型或真实预测",
    ),
}

_MODEL_IDENTITY_FAILURE_CODES: dict[str, ErrorCode] = {
    "baidu_ocr": ErrorCode.OCR_RESPONSE_INVALID,
    "qwen": ErrorCode.QWEN_RESPONSE_INVALID,
    "silero_vad": ErrorCode.SPEECH_MODEL_UNAVAILABLE,
    "cloud_whisper": ErrorCode.SPEECH_MODEL_UNAVAILABLE,
}


def collect_production_environment_issues(
    settings: Settings,
    store: EvidenceStore,
) -> tuple[ErrorCode, ...]:
    """复用 live 严格口径检查完整生产链的凭据、条款、依赖与模型缓存。"""

    runner = LiveValidationRunner(settings, store)
    issues: list[ErrorCode] = []
    runner._collect_baidu_environment_issues(issues)
    runner._collect_qwen_environment_issues(issues)
    runner._collect_local_stack_environment_issues(issues)
    return tuple(dict.fromkeys(issues))


class LiveExecutionPort(Protocol):
    def __call__(
        self,
        check_id: LiveCheckId,
        samples: tuple[LiveSample, ...],
        components: object,
        journal: _LiveExecutionJournal,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class _PreparedLiveRun:
    samples: tuple[LiveSample, ...]
    inputs: tuple[LiveInputArtifact, ...]
    snapshots: tuple[FileSnapshot, ...]


class _LiveExecutionJournal:
    """只接受无正文的成功阶段摘要，并据此恢复 FAIL partial facts。"""

    def __init__(
        self,
        *,
        check_id: LiveCheckId,
        evaluation_run_id: str,
        samples: tuple[LiveSample, ...],
        canonical_models: tuple[ModelIdentity, ...],
        writer: LiveReportRunWriter,
    ) -> None:
        self._evaluation_run_id = evaluation_run_id
        self._writer = writer
        self._stages = _execution_stages(check_id, samples)
        self._canonical_models = canonical_models
        self._facts: list[ModelExecutionFact] = []
        self._artifacts: list[TraceArtifact] = []

    @property
    def facts(self) -> tuple[ModelExecutionFact, ...]:
        return tuple(self._facts)

    @property
    def artifacts(self) -> tuple[TraceArtifact, ...]:
        return tuple(self._artifacts)

    @property
    def evaluation_run_id(self) -> str:
        return self._evaluation_run_id

    @property
    def is_complete(self) -> bool:
        return len(self._facts) == len(self._stages)

    @property
    def failure_component(self) -> str | None:
        if self.is_complete:
            return None
        return self._stages[len(self._facts)][0]

    def record_success(self, summary: LiveExecutionSummary) -> None:
        if self.is_complete:
            raise VideoDemoError(ErrorCode.SYSTEM_FAILURE, "live 执行阶段已经闭合")
        component, operation, sample = self._stages[len(self._facts)]
        input_kind, input_sha256 = _component_input(sample, component)
        if (
            summary.component != component
            or summary.operation != operation
            or summary.evaluation_run_id != self._evaluation_run_id
            or summary.sample_id != sample.sample_id
            or summary.language != sample.language
            or summary.input_kind != input_kind
            or summary.input_sha256 != input_sha256
        ):
            raise VideoDemoError(ErrorCode.SYSTEM_FAILURE, "live 成功阶段与预期顺序不匹配")
        if summary.model not in self._canonical_models:
            raise VideoDemoError(
                _MODEL_IDENTITY_FAILURE_CODES[component],
                "live 模型身份与当前生产组合不匹配",
            )
        artifact = self._writer.write_artifact(
            f"execution-{len(self._facts):03d}.json",
            "PROVIDER_RESPONSE",
            summary.model_dump_json(exclude_none=True).encode("utf-8"),
        )
        fact = ModelExecutionFact(
            component=summary.component,
            operation=summary.operation,
            evaluation_run_id=summary.evaluation_run_id,
            model=summary.model,
            sample_id=summary.sample_id,
            language=summary.language,
            input_kind=summary.input_kind,
            input_sha256=summary.input_sha256,
            output_sha256=artifact.sha256,
            request_id_sha256=summary.request_id_sha256,
            http_status=summary.http_status,
            capabilities=summary.capabilities,
        )
        self._artifacts.append(artifact)
        self._facts.append(fact)


class LiveValidationRunner:
    """live 门禁的授权加载、精确 preflight 与安全持久化状态机。"""

    def __init__(
        self,
        settings: Settings,
        evidence_store: EvidenceStore,
        *,
        components_factory: Callable[[Settings], ProductionDiagnosticComponents] | None = None,
        execution_port: LiveExecutionPort | None = None,
    ) -> None:
        self._settings = settings
        self._store = evidence_store
        self._components_factory = (
            components_factory or build_production_diagnostic_components
        )
        self._execution_port = execution_port or self._task5_execution_port
        self._allows_live_pass = components_factory is None and execution_port is None

    def run_baidu(
        self,
        evaluation_run_id: str,
        package: ValidatedEvaluationPackage,
    ) -> GateCheck:
        return self._run("baidu_ocr_live", evaluation_run_id, package)

    def run_qwen(
        self,
        evaluation_run_id: str,
        package: ValidatedEvaluationPackage,
    ) -> GateCheck:
        return self._run("qwen_live", evaluation_run_id, package)

    def run_local_model_stack(
        self,
        evaluation_run_id: str,
        package: ValidatedEvaluationPackage,
    ) -> GateCheck:
        return self._run("five_language_models", evaluation_run_id, package)

    def run_workspace_baidu(self, evaluation_run_id: str) -> GateCheck:
        return self._run(
            "baidu_ocr_live",
            evaluation_run_id,
            self._load_workspace_package(),
        )

    def run_workspace_qwen(self, evaluation_run_id: str) -> GateCheck:
        return self._run(
            "qwen_live",
            evaluation_run_id,
            self._load_workspace_package(),
        )

    def run_workspace_local_model_stack(self, evaluation_run_id: str) -> GateCheck:
        return self._run(
            "five_language_models",
            evaluation_run_id,
            self._load_workspace_package(),
        )

    def _load_workspace_package(self) -> ValidatedEvaluationPackage | None:
        dataset_path = self._store.runtime_root / "eval/dataset.jsonl"
        authorization_path = self._store.runtime_root / "eval/authorization.json"
        if not dataset_path.exists() and not authorization_path.exists():
            return None
        return load_evaluation_package(
            dataset_path,
            authorization_path,
            workspace_root=self._settings.workspace_root,
            runtime_root=self._store.runtime_root,
            max_video_bytes=self._settings.max_video_bytes,
        )

    def _run(
        self,
        check_id: str,
        evaluation_run_id: str,
        package: ValidatedEvaluationPackage | None,
    ) -> GateCheck:
        validate_path_component(evaluation_run_id, "evaluation_run_id")
        self._assert_runner_roots()
        report_path = self._report_path(evaluation_run_id, check_id)
        if report_path.is_file():
            return self._reverify_existing(check_id, report_path)
        if report_path.parent.exists():
            raise VideoDemoError(ErrorCode.IDEMPOTENCY_CONFLICT, "运行目录已有不完整证据")
        verified_package: ValidatedEvaluationPackage | None = None
        if package is not None:
            self._assert_package_roots(package)
            verified_package = reverify_evaluation_package(package)
        writer: LiveReportRunWriter | None = None
        claim_conflict = False
        try:
            writer = self._store.claim_exclusive_live_report_run(evaluation_run_id)
        except LiveReportRunConflict:
            claim_conflict = True
        if claim_conflict:
            raise VideoDemoError(
                ErrorCode.IDEMPOTENCY_CONFLICT,
                "运行目录已有不完整证据",
            ) from None
        assert writer is not None
        result: GateCheck | None = None
        write_conflict = False
        try:
            live_check_id = cast(LiveCheckId, check_id)
            issues = (
                self._preflight_without_package(check_id)
                if verified_package is None
                else self._preflight(check_id, evaluation_run_id, verified_package)
            )
            if issues:
                result = self._write_not_run(
                    writer,
                    check_id,
                    evaluation_run_id,
                    issues,
                )
            else:
                assert verified_package is not None
                prepared = self._prepare_live_run(
                    live_check_id,
                    evaluation_run_id,
                    verified_package,
                )
                result = self._execute_fail_closed(
                    writer,
                    live_check_id,
                    evaluation_run_id,
                    verified_package,
                    prepared,
                )
        except LiveReportRunConflict:
            write_conflict = True
        finally:
            writer.close()
        if write_conflict:
            raise VideoDemoError(
                ErrorCode.IDEMPOTENCY_CONFLICT,
                "运行目录已有冲突证据",
            ) from None
        assert result is not None
        return result

    def _assert_runner_roots(self) -> None:
        if (
            self._settings.workspace_root != self._store.workspace_root
            or self._settings.runtime_root != self._store.runtime_root
        ):
            raise VideoDemoError(
                ErrorCode.INVALID_CONFIGURATION,
                "live runner 的配置根与证据根不一致",
            ) from None

    def _assert_package_roots(self, package: ValidatedEvaluationPackage) -> None:
        if (
            not isinstance(package, ValidatedEvaluationPackage)
            or package.workspace_root != self._store.workspace_root
            or package.runtime_root != self._store.runtime_root
        ):
            raise VideoDemoError(
                ErrorCode.EVALUATION_DATASET_INVALID,
                "评测包来源不属于当前 live runner 可信根",
            ) from None

    def _reverify_existing(self, check_id: str, report_path: Path) -> GateCheck:
        try:
            report = load_machine_evidence(
                report_path,
                workspace_root=self._settings.workspace_root,
            )
            expected = {report_path.name}
            for artifact in report.artifacts:
                artifact_path = self._settings.workspace_root / artifact.relative_path
                if artifact_path.parent == report_path.parent:
                    expected.add(artifact_path.name)
            if _report_run_entries(report_path.parent) != expected:
                raise ValueError("live 报告 run 包含未声明产物")
            check = build_verified_gate_check(
                check_id,
                report_path,
                workspace_root=self._settings.workspace_root,
                settings=self._settings,
            )
            if _report_run_entries(report_path.parent) != expected:
                raise ValueError("live 报告 run 在重验期间发生变化")
            return check
        except (OSError, ValueError, VideoDemoError):
            raise VideoDemoError(
                ErrorCode.IDEMPOTENCY_CONFLICT,
                "运行目录已有冲突证据",
            ) from None

    def _execute_fail_closed(
        self,
        writer: LiveReportRunWriter,
        check_id: LiveCheckId,
        evaluation_run_id: str,
        package: ValidatedEvaluationPackage,
        prepared: _PreparedLiveRun,
    ) -> GateCheck:
        self._assert_inputs_current(package, prepared)
        identity = build_production_model_identity_report(self._settings)
        journal = _LiveExecutionJournal(
            check_id=check_id,
            evaluation_run_id=evaluation_run_id,
            samples=prepared.samples,
            canonical_models=identity.models,
            writer=writer,
        )
        execution_failure: Exception | None = None
        close_failure: Exception | None = None
        components: object | None = None
        try:
            components = self._components_factory(self._settings)
            self._execution_port(check_id, prepared.samples, components, journal)
        except Exception as error:
            execution_failure = error
        finally:
            closer = getattr(components, "close", None)
            if callable(closer):
                try:
                    closer()
                except Exception as error:
                    close_failure = error
        self._assert_inputs_current(package, prepared)
        failure_component: str
        if journal.is_complete:
            if execution_failure is not None:
                raise VideoDemoError(
                    ErrorCode.SYSTEM_FAILURE,
                    "live 完整执行后的端口状态非法",
                ) from None
            if close_failure is None:
                if not self._allows_live_pass:
                    raise VideoDemoError(
                        ErrorCode.SYSTEM_FAILURE,
                        "受控执行端口不得发布正式 PASS",
                    ) from None
                return self._write_success(
                    writer,
                    check_id,
                    evaluation_run_id,
                    package,
                    prepared,
                    journal,
                    identity,
                )
            failure_component = "components_close"
            failure_code = ErrorCode.SYSTEM_FAILURE
        else:
            pending_component = journal.failure_component
            assert pending_component is not None
            failure_component = pending_component
            failure_code = _safe_failure_code(
                execution_failure or close_failure,
                failure_component,
            )
        return self._write_failure(
            writer,
            check_id,
            evaluation_run_id,
            package,
            prepared,
            journal,
            failure_code,
            failure_component,
            identity,
        )

    def _task5_execution_port(
        self,
        check_id: LiveCheckId,
        samples: tuple[LiveSample, ...],
        components: object,
        journal: _LiveExecutionJournal,
    ) -> None:
        if not isinstance(components, ProductionDiagnosticComponents):
            raise VideoDemoError(ErrorCode.SYSTEM_FAILURE, "生产诊断组件类型非法")
        if check_id == "baidu_ocr_live":
            self._execute_baidu(samples[0], components, journal)
        elif check_id == "qwen_live":
            self._execute_qwen(samples[0], components, journal)
        else:
            self._execute_local_model_stack(samples, components, journal)

    def _execute_baidu(
        self,
        sample: LiveSample,
        components: ProductionDiagnosticComponents,
        journal: _LiveExecutionJournal,
    ) -> None:
        image = self._read_live_input_bytes(
            sample.keyframe_relative_path,
            sample.keyframe_sha256,
            max_bytes=self._settings.max_video_bytes,
        )
        if not is_supported_ocr_image(image):
            raise VideoDemoError(ErrorCode.VIDEO_INPUT_INVALID, "live OCR 关键帧格式非法")
        response = components.baidu_ocr_client.recognize(image, sample.language)
        journal.record_success(
            LiveExecutionSummary(
                schema_version="1.0.0",
                component="baidu_ocr",
                operation="recognize",
                evaluation_run_id=journal.evaluation_run_id,
                model=self._production_model(components, "baidu_ocr"),
                sample_id=sample.sample_id,
                language=sample.language,
                input_kind="KEYFRAME",
                input_sha256=sample.keyframe_sha256,
                request_id_sha256=_identifier_sha256(response.request_id),
                http_status=response.http_status,
                output_item_count=len(response.lines),
            )
        )

    def _execute_qwen(
        self,
        sample: LiveSample,
        components: ProductionDiagnosticComponents,
        journal: _LiveExecutionJournal,
    ) -> None:
        clip_path = self._verified_live_input_path(
            sample.clip_relative_path,
            sample.clip_sha256,
            max_bytes=self._settings.qwen_max_video_bytes,
        )
        duration_ms = min(
            sample.duration_ms,
            self._settings.qwen_max_video_duration_ms,
        )
        clip = VideoClipInput(
            clip_id=stable_identifier(
                "live_clip",
                {
                    "sample_id": sample.sample_id,
                    "clip_sha256": sample.clip_sha256,
                },
            ),
            start_ms=0,
            end_ms=duration_ms,
            path=clip_path,
            mime_type="video/mp4",
            sha256=sample.clip_sha256,
        )
        model = self._production_model(components, "qwen")
        capabilities, probe_receipt = (
            components.qwen_client.probe_capabilities_with_receipt(clip)
        )
        if (
            capabilities.model_id != model.model_id
            or capabilities.video_input != "data_url"
            or capabilities.json_schema is not True
        ):
            raise VideoDemoError(
                ErrorCode.QWEN_CAPABILITY_UNAVAILABLE,
                "Qwen 能力或模型身份与生产配置不一致",
            )
        journal.record_success(
            LiveExecutionSummary(
                schema_version="1.0.0",
                component="qwen",
                operation="capability_probe",
                evaluation_run_id=journal.evaluation_run_id,
                model=model,
                sample_id=sample.sample_id,
                language=sample.language,
                input_kind="CLIP",
                input_sha256=sample.clip_sha256,
                request_id_sha256=_identifier_sha256(probe_receipt.response_id),
                http_status=probe_receipt.http_status,
                capabilities=("video_input", "strict_json_schema"),
                output_item_count=1,
            )
        )
        scene = SceneBoundary(
            evidence_id=stable_identifier(
                "live_scene",
                {
                    "sample_id": sample.sample_id,
                    "clip_sha256": sample.clip_sha256,
                },
            ),
            start_ms=0,
            end_ms=duration_ms,
            transition="candidate",
            score=1.0,
        )
        request = SegmentUnderstandingRequest(
            clip=clip,
            window=TimeRange(start_ms=0, end_ms=duration_ms),
            timeline=build_timeline((scene,)),
            evidence=(scene,),
        )
        result, segment_receipt = (
            components.qwen_client.understand_segment_with_receipt(request)
        )
        journal.record_success(
            LiveExecutionSummary(
                schema_version="1.0.0",
                component="qwen",
                operation="understand_segment",
                evaluation_run_id=journal.evaluation_run_id,
                model=model,
                sample_id=sample.sample_id,
                language=sample.language,
                input_kind="CLIP",
                input_sha256=sample.clip_sha256,
                request_id_sha256=_identifier_sha256(segment_receipt.response_id),
                http_status=segment_receipt.http_status,
                output_item_count=len(result.evidence_refs),
            )
        )

    def _execute_local_model_stack(
        self,
        samples: tuple[LiveSample, ...],
        components: ProductionDiagnosticComponents,
        journal: _LiveExecutionJournal,
    ) -> None:
        by_language = {sample.language: sample for sample in samples}
        ordered = tuple(by_language[language] for language in _VALIDATION_LANGUAGES)
        first = ordered[0]
        first_audio = self._verified_live_input_path(
            first.audio_relative_path,
            first.audio_sha256,
            max_bytes=self._settings.max_video_bytes,
        )
        vad = components.speech_models.vad.detect(
            first_audio,
            duration_ms=first.duration_ms,
        )
        journal.record_success(
            LiveExecutionSummary(
                schema_version="1.0.0",
                component="silero_vad",
                operation="vad",
                evaluation_run_id=journal.evaluation_run_id,
                model=self._production_model(components, "silero_vad"),
                sample_id=first.sample_id,
                language=first.language,
                input_kind="AUDIO",
                input_sha256=first.audio_sha256,
                output_item_count=len(vad.speech),
            )
        )
        for sample in ordered:
            audio = self._verified_live_input_path(
                sample.audio_relative_path,
                sample.audio_sha256,
                max_bytes=self._settings.max_video_bytes,
            )
            result = components.speech_models.recognizer.transcribe_window(
                audio,
                language_hint=sample.language,
                prompt=None,
            )
            if not result.segments:
                raise VideoDemoError(
                    ErrorCode.SPEECH_MODEL_UNAVAILABLE,
                    "五语云端 Whisper 未返回转写片段",
                )
            if any(
                segment.start_ms < 0
                or segment.end_ms <= segment.start_ms
                or segment.end_ms > sample.duration_ms
                for segment in result.segments
            ):
                raise VideoDemoError(
                    ErrorCode.SPEECH_AUDIO_INVALID,
                    "五语云端 Whisper 返回了非法时间范围",
                )
            journal.record_success(
                LiveExecutionSummary(
                    schema_version="1.0.0",
                    component="cloud_whisper",
                    operation="transcribe",
                    evaluation_run_id=journal.evaluation_run_id,
                    model=self._production_model(components, "cloud_whisper"),
                    sample_id=sample.sample_id,
                    language=sample.language,
                    input_kind="AUDIO",
                    input_sha256=sample.audio_sha256,
                    output_item_count=len(result.segments),
                )
            )

    def _production_model(
        self,
        components: ProductionDiagnosticComponents,
        component: str,
    ) -> ModelIdentity:
        candidates = tuple(
            model
            for model in components.model_identity_report.models
            if model.component == component
        )
        if len(candidates) != 1:
            raise VideoDemoError(
                _MODEL_IDENTITY_FAILURE_CODES[component],
                "生产模型身份缺失或重复",
            )
        return candidates[0]

    def _verified_live_input_path(
        self,
        relative_path: str,
        expected_sha256: str,
        *,
        max_bytes: int,
    ) -> Path:
        try:
            snapshot = _read_file_snapshot(
                self._settings.workspace_root / relative_path,
                max_bytes=max_bytes,
                capture_content=False,
            )
        except (OSError, ValueError):
            raise VideoDemoError(
                ErrorCode.VIDEO_INPUT_INVALID,
                "live 输入文件不可用",
            ) from None
        if snapshot.sha256 != expected_sha256:
            raise VideoDemoError(
                ErrorCode.VIDEO_DIGEST_MISMATCH,
                "live 输入文件摘要不匹配",
            ) from None
        return snapshot.path

    def _read_live_input_bytes(
        self,
        relative_path: str,
        expected_sha256: str,
        *,
        max_bytes: int,
    ) -> bytes:
        try:
            snapshot = _read_file_snapshot(
                self._settings.workspace_root / relative_path,
                max_bytes=max_bytes,
                capture_content=True,
                require_utf8=False,
            )
        except (OSError, ValueError):
            raise VideoDemoError(
                ErrorCode.VIDEO_INPUT_INVALID,
                "live 输入文件不可用",
            ) from None
        if snapshot.sha256 != expected_sha256 or snapshot.content is None:
            raise VideoDemoError(
                ErrorCode.VIDEO_DIGEST_MISMATCH,
                "live 输入文件摘要不匹配",
            ) from None
        return snapshot.content

    def _prepare_live_run(
        self,
        check_id: LiveCheckId,
        evaluation_run_id: str,
        package: ValidatedEvaluationPackage,
    ) -> _PreparedLiveRun:
        if check_id == "five_language_models":
            samples = self._select_five_language_samples(
                package,
                evaluation_run_id,
            )
        else:
            filename = {
                "baidu_ocr_live": "keyframe.jpg",
                "qwen_live": "clip.mp4",
            }[check_id]
            samples = (
                self._select_single_sample(package, evaluation_run_id, filename),
            )
        inputs = tuple(
            input_artifact
            for sample, _snapshots in samples
            for input_artifact in _live_inputs(sample, _snapshots)
        )
        return _PreparedLiveRun(
            samples=tuple(sample for sample, _snapshots in samples),
            inputs=inputs,
            snapshots=tuple(
                snapshot
                for _sample, sample_snapshots in samples
                for snapshot in sample_snapshots
            ),
        )

    def _select_single_sample(
        self,
        package: ValidatedEvaluationPackage,
        evaluation_run_id: str,
        required_filename: str,
    ) -> tuple[LiveSample, tuple[FileSnapshot, ...]]:
        for index, dataset_sample in enumerate(package.dataset.samples):
            required_path = self._derived_path(
                evaluation_run_id,
                dataset_sample.sample_id,
                required_filename,
            )
            if not _snapshot_available(
                required_path,
                max_bytes=self._input_limit(required_filename),
            ):
                continue
            prepared = self._snapshot_sample(package, evaluation_run_id, index)
            if prepared is not None:
                return prepared
        raise VideoDemoError(
            ErrorCode.EVALUATION_ARTIFACT_INVALID,
            "授权 live 样本派生输入不完整",
        ) from None

    def _select_five_language_samples(
        self,
        package: ValidatedEvaluationPackage,
        evaluation_run_id: str,
    ) -> tuple[tuple[LiveSample, tuple[FileSnapshot, ...]], ...]:
        selected: list[tuple[LiveSample, tuple[FileSnapshot, ...]]] = []
        for language in _VALIDATION_LANGUAGES:
            prepared = next(
                (
                    value
                    for index, sample in enumerate(package.dataset.samples)
                    if sample.language == language
                    and _snapshot_available(
                        self._derived_path(
                            evaluation_run_id,
                            sample.sample_id,
                            "audio.wav",
                        ),
                        max_bytes=self._settings.max_video_bytes,
                    )
                    and (value := self._snapshot_sample(package, evaluation_run_id, index))
                    is not None
                ),
                None,
            )
            if prepared is None:
                raise VideoDemoError(
                    ErrorCode.EVALUATION_ARTIFACT_INVALID,
                    "五语 live 样本派生输入不完整",
                ) from None
            selected.append(prepared)
        return tuple(selected)

    def _snapshot_sample(
        self,
        package: ValidatedEvaluationPackage,
        evaluation_run_id: str,
        index: int,
    ) -> tuple[LiveSample, tuple[FileSnapshot, ...]] | None:
        dataset_sample = package.dataset.samples[index]
        annotation = package.annotations[index]
        source = package.dataset.eval_root / dataset_sample.media_relative_path
        paths = (
            source,
            self._derived_path(evaluation_run_id, dataset_sample.sample_id, "audio.wav"),
            self._derived_path(evaluation_run_id, dataset_sample.sample_id, "keyframe.jpg"),
            self._derived_path(evaluation_run_id, dataset_sample.sample_id, "clip.mp4"),
        )
        limits = (
            self._settings.max_video_bytes,
            self._settings.max_video_bytes,
            self._settings.max_video_bytes,
            self._settings.qwen_max_video_bytes,
        )
        try:
            snapshots = tuple(
                _read_file_snapshot(
                    path,
                    max_bytes=limit,
                    capture_content=False,
                )
                for path, limit in zip(paths, limits, strict=True)
            )
        except (OSError, ValueError):
            return None
        source_snapshot, audio_snapshot, keyframe_snapshot, clip_snapshot = snapshots
        if source_snapshot.sha256 != dataset_sample.media_sha256:
            return None
        workspace = self._settings.workspace_root
        return (
            LiveSample(
                sample_id=dataset_sample.sample_id,
                language=dataset_sample.language,
                duration_ms=annotation.annotation.duration_ms,
                source_media_relative_path=source.relative_to(workspace).as_posix(),
                source_media_sha256=source_snapshot.sha256,
                audio_relative_path=paths[1].relative_to(workspace).as_posix(),
                audio_sha256=audio_snapshot.sha256,
                keyframe_relative_path=paths[2].relative_to(workspace).as_posix(),
                keyframe_sha256=keyframe_snapshot.sha256,
                clip_relative_path=paths[3].relative_to(workspace).as_posix(),
                clip_sha256=clip_snapshot.sha256,
                annotation_sha256=annotation.sha256,
            ),
            snapshots,
        )

    def _input_limit(self, filename: str) -> int:
        if filename == "clip.mp4":
            return self._settings.qwen_max_video_bytes
        return self._settings.max_video_bytes

    def _assert_inputs_current(
        self,
        package: ValidatedEvaluationPackage,
        prepared: _PreparedLiveRun,
    ) -> None:
        try:
            reverify_evaluation_package(package)
            for snapshot in prepared.snapshots:
                _assert_snapshot_current(snapshot)
        except (OSError, ValueError, VideoDemoError):
            raise VideoDemoError(
                ErrorCode.EVALUATION_ARTIFACT_INVALID,
                "live 授权输入在执行边界发生变化",
            ) from None

    def _preflight(
        self,
        check_id: str,
        evaluation_run_id: str,
        package: ValidatedEvaluationPackage,
    ) -> tuple[ErrorCode, ...]:
        issues: list[ErrorCode] = []
        if check_id == "baidu_ocr_live":
            self._collect_baidu_issues(issues, evaluation_run_id, package)
        elif check_id == "qwen_live":
            self._collect_qwen_issues(issues, evaluation_run_id, package)
        else:
            self._collect_local_stack_issues(issues, evaluation_run_id, package)
        ordered = _LIVE_PREFLIGHT_CODES[check_id]
        return tuple(code for code in ordered if code in set(issues))

    def _preflight_without_package(self, check_id: str) -> tuple[ErrorCode, ...]:
        issues: list[ErrorCode] = []
        if check_id == "baidu_ocr_live":
            self._collect_baidu_environment_issues(issues)
            issues.append(ErrorCode.LIVE_AUTHORIZED_KEYFRAME_UNAVAILABLE)
        elif check_id == "qwen_live":
            self._collect_qwen_environment_issues(issues)
            issues.append(ErrorCode.LIVE_AUTHORIZED_CLIP_UNAVAILABLE)
        else:
            self._collect_local_stack_environment_issues(issues)
            issues.append(ErrorCode.LIVE_FIVE_LANGUAGE_AUDIO_UNAVAILABLE)
        ordered = _LIVE_PREFLIGHT_CODES[check_id]
        issue_set = set(issues)
        return tuple(code for code in ordered if code in issue_set)

    def _collect_baidu_issues(
        self,
        issues: list[ErrorCode],
        evaluation_run_id: str,
        package: ValidatedEvaluationPackage,
    ) -> None:
        self._collect_baidu_environment_issues(issues)
        if not self._has_single_live_input(package, evaluation_run_id, "keyframe.jpg"):
            issues.append(ErrorCode.LIVE_AUTHORIZED_KEYFRAME_UNAVAILABLE)

    def _collect_baidu_environment_issues(self, issues: list[ErrorCode]) -> None:
        if not _has_secret(self._settings.baidu_api_key):
            issues.append(ErrorCode.BAIDU_API_KEY_UNAVAILABLE)
        if not _has_secret(self._settings.baidu_secret_key):
            issues.append(ErrorCode.BAIDU_SECRET_KEY_UNAVAILABLE)

    def _collect_qwen_issues(
        self,
        issues: list[ErrorCode],
        evaluation_run_id: str,
        package: ValidatedEvaluationPackage,
    ) -> None:
        self._collect_qwen_environment_issues(issues)
        if not self._has_single_live_input(
            package,
            evaluation_run_id,
            "clip.mp4",
            max_bytes=self._settings.qwen_max_video_bytes,
        ):
            issues.append(ErrorCode.LIVE_AUTHORIZED_CLIP_UNAVAILABLE)

    def _collect_qwen_environment_issues(self, issues: list[ErrorCode]) -> None:
        if not _has_text(self._settings.qwen_base_url):
            issues.append(ErrorCode.QWEN_ENDPOINT_UNAVAILABLE)
        if not _has_secret(self._settings.qwen_api_key):
            issues.append(ErrorCode.QWEN_API_KEY_UNAVAILABLE)
        if not _has_valid_qwen_model_id(
            self._settings.qwen_model_id,
            allow_unrecognized=self._settings.demo_degraded_mode,
        ):
            issues.append(ErrorCode.QWEN_MODEL_ID_UNAVAILABLE)

    def _collect_local_stack_issues(
        self,
        issues: list[ErrorCode],
        evaluation_run_id: str,
        package: ValidatedEvaluationPackage,
    ) -> None:
        self._collect_local_stack_environment_issues(issues)
        if not self._has_five_language_audio(package, evaluation_run_id):
            issues.append(ErrorCode.LIVE_FIVE_LANGUAGE_AUDIO_UNAVAILABLE)

    def _collect_local_stack_environment_issues(
        self,
        issues: list[ErrorCode],
    ) -> None:
        model_root = self._store.runtime_root / "models"
        if not self._module_available("silero_vad"):
            issues.append(ErrorCode.SILERO_DEPENDENCY_UNAVAILABLE)
        if not _has_exact_file(model_root / "silero/model-id.txt", b"silero-vad\n"):
            issues.append(ErrorCode.SILERO_MODEL_UNAVAILABLE)
        try:
            self._settings.require_cloud_asr_configuration()
        except VideoDemoError:
            issues.append(ErrorCode.INVALID_CONFIGURATION)

    def _has_single_live_input(
        self,
        package: ValidatedEvaluationPackage,
        evaluation_run_id: str,
        filename: str,
        *,
        max_bytes: int | None = None,
    ) -> bool:
        for index, sample in enumerate(package.dataset.samples):
            required = self._derived_path(
                evaluation_run_id,
                sample.sample_id,
                filename,
            )
            if not _snapshot_available(
                required,
                max_bytes=max_bytes or self._settings.max_video_bytes,
            ):
                continue
            if self._snapshot_sample(package, evaluation_run_id, index) is not None:
                return True
        return False

    def _has_five_language_audio(
        self,
        package: ValidatedEvaluationPackage,
        evaluation_run_id: str,
    ) -> bool:
        available_languages = set()
        for index, sample in enumerate(package.dataset.samples):
            audio = self._derived_path(
                evaluation_run_id,
                sample.sample_id,
                "audio.wav",
            )
            if not _snapshot_available(
                audio,
                max_bytes=self._settings.max_video_bytes,
            ):
                continue
            if self._snapshot_sample(package, evaluation_run_id, index) is not None:
                available_languages.add(sample.language)
        return available_languages == {"zh", "en", "ja", "ko", "es"}

    def _derived_path(
        self,
        evaluation_run_id: str,
        sample_id: str,
        filename: str,
    ) -> Path:
        return (
            self._store.runtime_root
            / "eval/live"
            / evaluation_run_id
            / sample_id
            / filename
        )

    def _module_available(self, name: str) -> bool:
        try:
            if importlib.util.find_spec(name) is None:
                return False
            importlib.import_module(name)
            return True
        except (ImportError, ModuleNotFoundError, AttributeError, OSError, ValueError):
            return False

    def _write_not_run(
        self,
        writer: LiveReportRunWriter,
        check_id: str,
        evaluation_run_id: str,
        issues: tuple[ErrorCode, ...],
    ) -> GateCheck:
        reason_code, not_run_reason = _CHECK_REASON[check_id]
        implementation = _current_live_implementation_sha256(
            self._settings.workspace_root
        )
        raw = PreflightRawReport(
            schema_version="1.0.0",
            check_id=check_id,
            reason_code=reason_code,
            execution_started=False,
            issues=tuple(PreflightIssue(code=code) for code in issues),
            implementation_sha256=implementation,
            evaluation_run_id=evaluation_run_id,
        )
        raw_artifact = writer.write_artifact(
            "preflight.json",
            "AUDIT_REPORT",
            raw.model_dump_json(exclude_none=True).encode("utf-8"),
        )
        stdout = writer.write_artifact(
            "trace.stdout.txt",
            "COMMAND_STDOUT",
            b"",
        )
        stderr = writer.write_artifact(
            "trace.stderr.txt",
            "COMMAND_STDERR",
            b"",
        )
        timestamp = datetime.now(UTC)
        report = MachineEvidenceReport(
            schema_version="1.0.0",
            check_id=check_id,
            status=GateStatus.NOT_RUN,
            kind=EvidenceKind.LIVE_SERVICE_REPORT,
            level=EvidenceLevel.REAL_SERVICE,
            covered_items=(check_id,),
            summary="live 检查前置条件不足",
            producer="LiveValidationRunner",
            started_at=timestamp,
            finished_at=timestamp,
            not_run_reason=not_run_reason,
            artifacts=(raw_artifact, stdout, stderr),
            details=PreflightDetails(
                type="PREFLIGHT",
                trace=CommandTrace(
                    command=("python", "-m", "video_demo.evaluation.live_runner"),
                    exit_code=0,
                    stdout_sha256=stdout.sha256,
                    stderr_sha256=stderr.sha256,
                ),
                preflight_report_sha256=raw_artifact.sha256,
            ),
        )
        return writer.write_json(
            report,
            settings=self._settings,
        )

    def _write_success(
        self,
        writer: LiveReportRunWriter,
        check_id: LiveCheckId,
        evaluation_run_id: str,
        package: ValidatedEvaluationPackage,
        prepared: _PreparedLiveRun,
        journal: _LiveExecutionJournal,
        identity: ProductionModelIdentityReport,
    ) -> GateCheck:
        return self._write_execution_result(
            writer,
            check_id,
            evaluation_run_id,
            package,
            prepared,
            journal,
            identity,
            status=GateStatus.PASS,
            failure_code=None,
            failure_component=None,
        )

    def _write_failure(
        self,
        writer: LiveReportRunWriter,
        check_id: LiveCheckId,
        evaluation_run_id: str,
        package: ValidatedEvaluationPackage,
        prepared: _PreparedLiveRun,
        journal: _LiveExecutionJournal,
        failure_code: ErrorCode,
        failure_component: str,
        identity: ProductionModelIdentityReport,
    ) -> GateCheck:
        return self._write_execution_result(
            writer,
            check_id,
            evaluation_run_id,
            package,
            prepared,
            journal,
            identity,
            status=GateStatus.FAIL,
            failure_code=failure_code,
            failure_component=cast(LiveFailureComponent, failure_component),
        )

    def _write_execution_result(
        self,
        writer: LiveReportRunWriter,
        check_id: LiveCheckId,
        evaluation_run_id: str,
        package: ValidatedEvaluationPackage,
        prepared: _PreparedLiveRun,
        journal: _LiveExecutionJournal,
        identity: ProductionModelIdentityReport,
        *,
        status: Literal[GateStatus.PASS, GateStatus.FAIL],
        failure_code: ErrorCode | None,
        failure_component: LiveFailureComponent | None,
    ) -> GateCheck:
        implementation = _current_live_implementation_sha256(
            self._settings.workspace_root
        )
        raw_kwargs = {
            "schema_version": "1.0.0",
            "check_id": check_id,
            "status": status,
            "execution_started": True,
            "evaluation_run_id": evaluation_run_id,
            "dataset_sha256": package.dataset_sha256,
            "authorization_sha256": package.authorization_sha256,
            "settings_fingerprint": identity.settings_fingerprint,
            "implementation_sha256": implementation,
            "inputs": prepared.inputs,
            "executions": journal.facts,
            "failure_code": failure_code,
            "failure_component": failure_component,
        }
        raw: (
            BaiduLiveRawReport
            | QwenLiveRawReport
            | FiveLanguageModelsRawReport
        )
        details_type: (
            type[BaiduLiveDetails]
            | type[QwenLiveDetails]
            | type[FiveLanguageModelsDetails]
        )
        detail_name: Literal[
            "BAIDU_LIVE",
            "QWEN_LIVE",
            "FIVE_LANGUAGE_MODELS",
        ]
        if check_id == "baidu_ocr_live":
            raw = BaiduLiveRawReport(sample=prepared.samples[0], **raw_kwargs)
            details_type = BaiduLiveDetails
            detail_name = "BAIDU_LIVE"
        elif check_id == "qwen_live":
            raw = QwenLiveRawReport(sample=prepared.samples[0], **raw_kwargs)
            details_type = QwenLiveDetails
            detail_name = "QWEN_LIVE"
        else:
            raw = FiveLanguageModelsRawReport(samples=prepared.samples, **raw_kwargs)
            details_type = FiveLanguageModelsDetails
            detail_name = "FIVE_LANGUAGE_MODELS"
        raw_artifact = writer.write_artifact(
            "raw.json",
            "AUDIT_REPORT",
            raw.model_dump_json(exclude_none=True).encode("utf-8"),
        )
        stdout = writer.write_artifact(
            "trace.stdout.txt",
            "COMMAND_STDOUT",
            b"",
        )
        stderr = writer.write_artifact(
            "trace.stderr.txt",
            "COMMAND_STDERR",
            b"",
        )
        manifest = self._bind_package_artifact(package.dataset_path, "DATASET_MANIFEST")
        authorization = self._bind_package_artifact(
            package.authorization_path,
            "AUTHORIZATION_RECORD",
        )
        input_artifacts = tuple(
            self._store.bind_artifact(
                snapshot.path.relative_to(self._store.runtime_root),
                "INPUT_MEDIA",
                max_bytes=self._settings.max_video_bytes,
            )
            for snapshot in prepared.snapshots
        )
        trace = CommandTrace(
            command=("python", "-m", "video_demo.evaluation.live_runner"),
            exit_code=0 if status == GateStatus.PASS else 1,
            stdout_sha256=stdout.sha256,
            stderr_sha256=stderr.sha256,
        )
        details = details_type(
            type=detail_name,
            trace=trace,
            raw_report_sha256=raw_artifact.sha256,
            implementation_sha256=implementation,
            settings_fingerprint=identity.settings_fingerprint,
            dataset_sha256=package.dataset_sha256,
            authorization_sha256=package.authorization_sha256,
        )
        timestamp = datetime.now(UTC)
        report = MachineEvidenceReport(
            schema_version="1.0.0",
            check_id=check_id,
            status=status,
            kind=EvidenceKind.LIVE_SERVICE_REPORT,
            level=EvidenceLevel.REAL_SERVICE,
            covered_items=(check_id,),
            summary=(
                "live 检查执行成功"
                if status == GateStatus.PASS
                else "live 检查执行失败"
            ),
            producer="LiveValidationRunner",
            started_at=timestamp,
            finished_at=timestamp,
            artifacts=(
                raw_artifact,
                *journal.artifacts,
                manifest,
                authorization,
                *input_artifacts,
                stdout,
                stderr,
            ),
            details=details,
        )
        return writer.write_json(
            report,
            settings=self._settings,
        )

    def _bind_package_artifact(
        self,
        path: Path | None,
        role: ArtifactRole,
    ) -> TraceArtifact:
        if path is None:
            raise VideoDemoError(ErrorCode.EVALUATION_DATASET_INVALID, "评测包缺少来源")
        return self._store.bind_artifact(
            path.relative_to(self._store.runtime_root),
            role,
        )

    def _report_path(self, evaluation_run_id: str, check_id: str) -> Path:
        return (
            self._store.runtime_root
            / "eval"
            / "reports"
            / evaluation_run_id
            / f"{check_id}.json"
        )


def _is_nonempty_regular_file(path: Path) -> bool:
    return _snapshot_available(path, max_bytes=4 * 1024 * 1024 * 1024)


def _identifier_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _snapshot_available(path: Path, *, max_bytes: int) -> bool:
    try:
        _read_file_snapshot(path, max_bytes=max_bytes, capture_content=False)
    except (OSError, ValueError):
        return False
    return True


def _has_exact_file(path: Path, expected: bytes) -> bool:
    try:
        snapshot = _read_file_snapshot(
            path,
            max_bytes=max(len(expected), 1),
            capture_content=True,
        )
    except (OSError, UnicodeDecodeError, ValueError):
        return False
    return snapshot.content == expected


def _has_nonempty_model_tree(
    path: Path,
    *,
    ignored_relative_paths: frozenset[str] = frozenset(),
) -> bool:
    try:
        _reject_symlink_components(path)
        if not path.is_dir():
            return False
        has_model_file = False
        for candidate in path.rglob("*"):
            if candidate.is_symlink():
                return False
            if candidate.relative_to(path).as_posix() in ignored_relative_paths:
                continue
            if candidate.is_file() and _is_nonempty_regular_file(candidate):
                has_model_file = True
    except (OSError, ValueError):
        return False
    return has_model_file


def _has_text(value: str | None) -> bool:
    return value is not None and bool(value.strip())


def _has_valid_qwen_model_id(
    value: str | None,
    *,
    allow_unrecognized: bool = False,
) -> bool:
    try:
        return (
            _normalized_qwen_model_id(
                value,
                allow_unrecognized=allow_unrecognized,
            )
            is not None
        )
    except ValueError:
        return False


def _has_secret(value: object | None) -> bool:
    if value is None:
        return False
    getter = getattr(value, "get_secret_value", None)
    return callable(getter) and bool(cast(str, getter()).strip())


def _live_inputs(
    sample: LiveSample,
    snapshots: tuple[FileSnapshot, ...],
) -> tuple[LiveInputArtifact, ...]:
    source, audio, keyframe, clip = snapshots
    return (
        LiveInputArtifact(
            kind="SOURCE_MEDIA",
            sample_id=sample.sample_id,
            relative_path=sample.source_media_relative_path,
            sha256=sample.source_media_sha256,
            source_media_sha256=sample.source_media_sha256,
            size_bytes=source.identity.size,
        ),
        LiveInputArtifact(
            kind="AUDIO",
            sample_id=sample.sample_id,
            relative_path=sample.audio_relative_path,
            sha256=sample.audio_sha256,
            source_media_sha256=sample.source_media_sha256,
            size_bytes=audio.identity.size,
        ),
        LiveInputArtifact(
            kind="KEYFRAME",
            sample_id=sample.sample_id,
            relative_path=sample.keyframe_relative_path,
            sha256=sample.keyframe_sha256,
            source_media_sha256=sample.source_media_sha256,
            size_bytes=keyframe.identity.size,
        ),
        LiveInputArtifact(
            kind="CLIP",
            sample_id=sample.sample_id,
            relative_path=sample.clip_relative_path,
            sha256=sample.clip_sha256,
            source_media_sha256=sample.source_media_sha256,
            size_bytes=clip.identity.size,
        ),
    )


def _execution_stages(
    check_id: LiveCheckId,
    samples: tuple[LiveSample, ...],
) -> tuple[tuple[str, str, LiveSample], ...]:
    if check_id == "baidu_ocr_live":
        return (("baidu_ocr", "recognize", samples[0]),)
    if check_id == "qwen_live":
        return (
            ("qwen", "capability_probe", samples[0]),
            ("qwen", "understand_segment", samples[0]),
        )
    by_language = {sample.language: sample for sample in samples}
    ordered = tuple(by_language[language] for language in _VALIDATION_LANGUAGES)
    return (
        ("silero_vad", "vad", ordered[0]),
        *(("cloud_whisper", "transcribe", sample) for sample in ordered),
    )


def _component_input(sample: LiveSample, component: str) -> tuple[str, str]:
    if component == "baidu_ocr":
        return "KEYFRAME", sample.keyframe_sha256
    if component == "qwen":
        return "CLIP", sample.clip_sha256
    return "AUDIO", sample.audio_sha256


def _safe_failure_code(
    failure: Exception | None,
    failure_component: str,
) -> ErrorCode:
    if isinstance(failure, VideoDemoError):
        allowed = _LIVE_COMPONENT_FAILURE_CODES.get(failure_component, frozenset())
        if failure.code in allowed:
            return failure.code
    return ErrorCode.SYSTEM_FAILURE


def _report_run_entries(report_root: Path) -> set[str]:
    entries: set[str] = set()
    for path in report_root.iterdir():
        if path.is_symlink() or not path.is_file():
            raise ValueError("live 报告 run 只能包含普通文件")
        entries.add(path.name)
    return entries
