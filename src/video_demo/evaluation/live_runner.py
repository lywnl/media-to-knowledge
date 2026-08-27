from __future__ import annotations

import hashlib
import importlib.util
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol, cast

import httpx

from video_demo.application.composition import (
    ProductionModelIdentityReport,
    build_production_model_identity_report,
    production_tool_path,
)
from video_demo.application.legacy_composition import (
    ProductionModelIdentityReport as LegacyProductionModelIdentityReport,
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
from video_demo.evaluation.chapter_vlm_input import (
    prepare_chapter_vlm_input,
)
from video_demo.evaluation.chapter_vlm_live import (
    build_visual_text_score_fact,
    execute_chapter_vlm_live,
    has_selected_frame_selection,
    has_visual_text_projection,
)
from video_demo.evaluation.dataset import ValidationLanguage
from video_demo.evaluation.evidence import (
    _LIVE_PREFLIGHT_CODES,
    ArtifactRole,
    BaiduLiveDetails,
    BaiduLiveRawReport,
    ChapterVlmLiveDetails,
    ChapterVlmLiveRawReport,
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
from video_demo.integrations.qwen_vl import QwenVisionCallFailure, QwenVisionClient
from video_demo.integrations.video_port import SegmentUnderstandingRequest, VideoClipInput
from video_demo.speech.runtime import ProductionSpeechModels, build_diagnostic_speech_models
from video_demo.storage.workspace import validate_path_component
from video_demo.visual.ocr import is_supported_ocr_image

_VALIDATION_LANGUAGES: tuple[ValidationLanguage, ...] = ("zh", "en", "ja", "ko", "es")


def _has_eligible_chapter_frame_cluster(frames: object) -> bool:
    """仅把至少两个不同时间点且跨度在单章上限内的标注视为可执行输入。"""

    if not isinstance(frames, (tuple, list)):
        return False
    try:
        timestamps = sorted({int(frame.timestamp_ms) for frame in frames})
    except (TypeError, ValueError, AttributeError):
        return False
    return len(timestamps) >= 2 and timestamps[-1] - timestamps[0] <= 300_000


def _normalized_qwen_model_id(
    value: str | None,
    *,
    allow_unrecognized: bool = False,
) -> str | None:
    normalized = value.strip() if value is not None else ""
    if not normalized:
        return None
    if not allow_unrecognized and not normalized.startswith("qwen"):
        raise ValueError("Qwen 模型 ID 非法")
    return normalized


ProductionDiagnosticComponents = Any


def _package_media_path(
    package: ValidatedEvaluationPackage,
    sample_id: str,
) -> Path:
    """按评测包 eval_root 解析样本媒体，避免误接到工作区根。"""

    sample = next(
        (item for item in package.dataset.samples if item.sample_id == sample_id),
        None,
    )
    if sample is None:
        raise VideoDemoError(
            ErrorCode.LIVE_AUTHORIZED_CHAPTER_FRAMES_UNAVAILABLE,
            "章节 VLM 样本不存在",
        )
    return package.dataset.eval_root / sample.media_relative_path


class _LiveComponents:
    def __init__(
        self,
        *,
        chapter_vlm_client: QwenVisionClient,
        speech_models: ProductionSpeechModels,
        model_identity_report: ProductionModelIdentityReport,
        resources: tuple[httpx.Client, ...],
    ) -> None:
        self.chapter_vlm_client = chapter_vlm_client
        self.speech_models = speech_models
        self.model_identity_report = model_identity_report
        self._resources = resources

    def close(self) -> None:
        for resource in reversed(self._resources):
            resource.close()


def build_live_components(settings: Settings) -> _LiveComponents:
    """构造活动 live 所需的正式 Qwen3-VL 与五语语音组件。"""

    assert settings.runtime_root is not None
    vision = settings.require_vlm_configuration()
    speech_http = httpx.Client()
    vision_http = httpx.Client()
    try:
        return _LiveComponents(
            chapter_vlm_client=QwenVisionClient(
                vision_http,
                base_url=vision.base_url,
                api_key=vision.api_key.get_secret_value(),
                model_id=vision.model_id,
                runtime_root=settings.runtime_root,
                timeout_seconds=vision.timeout_seconds,
                max_attempts=vision.max_attempts,
                max_image_bytes=vision.max_image_bytes,
                max_request_image_bytes=vision.max_request_image_bytes,
                max_encoded_request_bytes=vision.max_encoded_request_bytes,
                max_response_bytes=settings.model_max_response_bytes,
            ),
            speech_models=build_diagnostic_speech_models(settings, speech_http),
            model_identity_report=build_production_model_identity_report(settings),
            resources=(speech_http, vision_http),
        )
    except Exception:
        speech_http.close()
        vision_http.close()
        raise


def _build_legacy_live_components(settings: Settings) -> object:
    """迁移期仅供旧报告/旧入口重验的组件构造器。"""
    return build_production_diagnostic_components(settings)


def build_production_diagnostic_components(settings: Settings) -> object:
    """兼容迁移期旧 live fixture 的惰性组合根入口。"""

    from video_demo.application.legacy_composition import (
        build_production_diagnostic_components as build_legacy_components,
    )

    return build_legacy_components(settings)


_CHECK_REASON: dict[str, tuple[str, str]] = {
    "chapter_vlm_live": (
        "CHAPTER_VLM_INPUT_UNAVAILABLE",
        "缺少章节多图 Qwen3-VL 凭据或真实联调结果",
    ),
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
    # 活动耐久链只依赖新章节视觉路径；旧百度/OCR 与全片 Qwen
    # 方法仍保留给迁移期历史报告读取，但不能污染当前 preflight。
    try:
        settings.require_text_llm_configuration()
    except VideoDemoError:
        issues.append(ErrorCode.INVALID_CONFIGURATION)
    runner._collect_chapter_vlm_environment_issues(issues)
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
        components_factory: Callable[[Settings], object] | None = None,
        execution_port: LiveExecutionPort | None = None,
    ) -> None:
        self._settings = settings
        self._store = evidence_store
        self._components_factory = components_factory or _build_legacy_live_components
        self._chapter_components_factory = components_factory or build_live_components
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

    def run_chapter_vlm(
        self,
        evaluation_run_id: str,
        package: ValidatedEvaluationPackage,
    ) -> GateCheck:
        """执行一次授权章节多图 Qwen3-VL live 检查。"""
        return self._run_chapter_vlm(evaluation_run_id, package)

    def run_workspace_chapter_vlm(self, evaluation_run_id: str) -> GateCheck:
        package = self._load_workspace_package()
        if package is None:
            return self._run_chapter_vlm(evaluation_run_id, None)
        return self._run_chapter_vlm(evaluation_run_id, package)

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

    def _run_chapter_vlm(
        self,
        evaluation_run_id: str,
        package: ValidatedEvaluationPackage | None,
    ) -> GateCheck:
        """章节 VLM 的专用执行路径，输入和调用回执使用 A1 适配器。"""
        validate_path_component(evaluation_run_id, "evaluation_run_id")
        self._assert_runner_roots()
        report_path = self._report_path(evaluation_run_id, "chapter_vlm_live")
        if report_path.is_file():
            return self._reverify_existing("chapter_vlm_live", report_path)
        if report_path.parent.exists():
            raise VideoDemoError(ErrorCode.IDEMPOTENCY_CONFLICT, "运行目录已有不完整证据")
        verified_package = None
        if package is not None:
            self._assert_package_roots(package)
            verified_package = reverify_evaluation_package(package)
        writer = self._store.claim_exclusive_live_report_run(evaluation_run_id)
        try:
            issues = self._preflight_chapter_vlm(evaluation_run_id, verified_package)
            if issues:
                return self._write_not_run(writer, "chapter_vlm_live", evaluation_run_id, issues)
            assert verified_package is not None
            return self._execute_chapter_vlm(writer, evaluation_run_id, verified_package)
        finally:
            writer.close()

    def _preflight_chapter_vlm(
        self,
        evaluation_run_id: str,
        package: ValidatedEvaluationPackage | None,
    ) -> tuple[ErrorCode, ...]:
        issues: list[ErrorCode] = []
        self._collect_chapter_vlm_environment_issues(issues)
        if package is None:
            issues.append(ErrorCode.LIVE_AUTHORIZED_CHAPTER_FRAMES_UNAVAILABLE)
        else:
            try:
                eligible = tuple(
                    item
                    for item in package.annotations
                    if _has_eligible_chapter_frame_cluster(item.annotation.visual_frames)
                )
                if not eligible:
                    issues.append(ErrorCode.LIVE_AUTHORIZED_CHAPTER_FRAMES_UNAVAILABLE)
                else:
                    sample_ids = {item.annotation.sample_id for item in eligible}
                    if not any(
                        self._authorized_source_available(package, sample_id)
                        for sample_id in sample_ids
                    ):
                        issues.append(ErrorCode.LIVE_AUTHORIZED_CHAPTER_FRAMES_UNAVAILABLE)
            except Exception:
                issues.append(ErrorCode.LIVE_AUTHORIZED_CHAPTER_FRAMES_UNAVAILABLE)
        ordered = _LIVE_PREFLIGHT_CODES["chapter_vlm_live"]
        return tuple(code for code in ordered if code in set(issues))

    def _authorized_source_available(
        self,
        package: ValidatedEvaluationPackage,
        sample_id: str,
    ) -> bool:
        """在执行 Run 创建前确认授权源媒体仍存在且摘要未变化。"""

        sample = next(
            (item for item in package.dataset.samples if item.sample_id == sample_id),
            None,
        )
        if sample is None:
            return False
        try:
            snapshot = _read_file_snapshot(
                package.dataset.eval_root / sample.media_relative_path,
                max_bytes=self._settings.max_video_bytes,
                capture_content=False,
            )
        except (OSError, ValueError):
            return False
        return snapshot.sha256 == sample.media_sha256

    def _collect_chapter_vlm_environment_issues(self, issues: list[ErrorCode]) -> None:
        try:
            self._settings.require_vlm_configuration()
        except VideoDemoError:
            issues.append(ErrorCode.INVALID_CONFIGURATION)
        if not self._module_available("cv2"):
            issues.append(ErrorCode.VISUAL_DEPENDENCY_UNAVAILABLE)
        assert self._settings.runtime_root is not None
        ffmpeg = self._settings.ffmpeg_path or self._settings.runtime_root / "tools" / "ffmpeg"
        ffprobe = self._settings.ffprobe_path or self._settings.runtime_root / "tools" / "ffprobe"
        if not ffmpeg.is_file():
            issues.append(ErrorCode.VIDEO_FFMPEG_UNAVAILABLE)
        if not ffprobe.is_file():
            issues.append(ErrorCode.VIDEO_FFPROBE_UNAVAILABLE)
        try:
            free_bytes = shutil.disk_usage(self._settings.runtime_root).free
            if free_bytes < self._settings.min_free_disk_reserve_bytes:
                issues.append(ErrorCode.VIDEO_DISK_SPACE_INSUFFICIENT)
        except OSError:
            issues.append(ErrorCode.VIDEO_DISK_SPACE_INSUFFICIENT)

    def _execute_chapter_vlm(
        self,
        writer: LiveReportRunWriter,
        evaluation_run_id: str,
        package: ValidatedEvaluationPackage,
    ) -> GateCheck:
        identity = build_production_model_identity_report(self._settings)
        vision = self._settings.require_vlm_configuration()
        assert self._settings.runtime_root is not None
        from video_demo.media.probe import FFprobeClient

        ffprobe = FFprobeClient.from_path(
            production_tool_path(self._settings, "ffprobe"),
            workspace_root=self._settings.workspace_root,
        )
        from video_demo.media.transcode import FFmpegTranscoder

        transcoder = FFmpegTranscoder.from_path(
            production_tool_path(self._settings, "ffmpeg"),
            self._settings.runtime_root,
            workspace_root=self._settings.workspace_root,
        )
        from video_demo.visual.keyframes import OpenCvFrameExtractor

        extractor = OpenCvFrameExtractor(
            self._settings.runtime_root,
            max_frame_bytes=vision.max_image_bytes,
            jpeg_quality=self._settings.keyframe_jpeg_quality,
        )
        parent_id = evaluation_run_id
        manifest = None
        try:
            preparation = prepare_chapter_vlm_input(
                package,
                parent_evaluation_run_id=parent_id,
                proxy_max_edge=self._settings.visual_proxy_max_edge,
                jpeg_quality=self._settings.keyframe_jpeg_quality,
                max_video_bytes=self._settings.max_video_bytes,
                vlm_max_image_bytes=vision.max_image_bytes,
                max_candidate_frame_bytes_per_run=self._settings.max_candidate_frame_bytes_per_run,
                max_candidate_frame_files_per_run=self._settings.max_candidate_frame_files_per_run,
                ffprobe=ffprobe,
                transcoder=transcoder,
                frame_extractor=extractor,
                runtime_root=self._settings.runtime_root,
            )
        except (OSError, TypeError, ValueError, VideoDemoError):
            code = (
                preparation.error_code
                if "preparation" in locals() and preparation.error_code is not None
                else ErrorCode.ARTIFACT_SCHEMA_INVALID
            )
            return self._write_chapter_vlm_result(
                writer,
                evaluation_run_id,
                None,
                package,
                None,
                None,
                identity,
                status=GateStatus.FAIL,
                failure_code=code,
            )
        if preparation.status != "READY" or preparation.manifest is None:
            return self._write_chapter_vlm_result(
                writer,
                evaluation_run_id,
                None,
                package,
                None,
                None,
                identity,
                status=GateStatus.FAIL,
                failure_code=preparation.error_code or ErrorCode.ARTIFACT_SCHEMA_INVALID,
            )
        manifest = preparation.manifest
        context = preparation.context
        if context is None:
            return self._write_chapter_vlm_result(
                writer,
                evaluation_run_id,
                manifest,
                package,
                None,
                None,
                identity,
                status=GateStatus.FAIL,
                failure_code=ErrorCode.ARTIFACT_SCHEMA_INVALID,
            )
        components: object | None = None
        result_status: Literal[GateStatus.PASS, GateStatus.FAIL] = GateStatus.FAIL
        result_receipt: object | None = None
        result_score: object | None = None
        result_failure_code: ErrorCode | None = None
        result_failure_receipt: object | None = None
        result_failure_component: Literal["chapter_vlm", "components_close"] | None = None
        try:
            components = self._chapter_components_factory(self._settings)
            if not isinstance(components, _LiveComponents):
                raise VideoDemoError(ErrorCode.SYSTEM_FAILURE, "章节 VLM 组件类型非法")
            client = components.chapter_vlm_client
            response, receipt = execute_chapter_vlm_live(
                manifest,
                context=context,
                expected_parent_evaluation_run_id=manifest.parent_evaluation_run_id,
                expected_evaluation_run_id=manifest.evaluation_run_id,
                vision_client=client,
            )
            annotation = next(
                item
                for item in package.annotations
                if item.annotation.sample_id == manifest.sample_id
            )
            result_receipt = receipt
            if not has_selected_frame_selection(response, manifest):
                result_failure_code = ErrorCode.VISUAL_RESULT_INVALID
            else:
                result_score = build_visual_text_score_fact(
                    manifest,
                    annotation,
                    response,
                    response_sha256=receipt.response_sha256,
                )
                result_status = (
                    GateStatus.PASS
                    if has_visual_text_projection(response, manifest)
                    else GateStatus.FAIL
                )
                if result_status == GateStatus.FAIL:
                    result_failure_code = ErrorCode.VISUAL_RESULT_INVALID
        except QwenVisionCallFailure as error:
            result_failure_code = error.code
            result_failure_receipt = error.provider
        except VideoDemoError as error:
            result_failure_code = error.code
        except (OSError, TypeError, ValueError, RuntimeError):
            result_failure_code = ErrorCode.SYSTEM_FAILURE
        finally:
            close = getattr(components, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    result_status = GateStatus.FAIL
                    result_receipt = None
                    result_score = None
                    result_failure_code = ErrorCode.SYSTEM_FAILURE
                    result_failure_receipt = None
                    result_failure_component = "components_close"
        if result_failure_code is not None and result_failure_component is None:
            result_failure_component = "chapter_vlm"
        return self._write_chapter_vlm_result(
            writer,
            evaluation_run_id,
            manifest,
            package,
            result_receipt,
            result_score,
            identity,
            status=result_status,
            failure_code=result_failure_code,
            failure_receipt=result_failure_receipt,
            failure_component=result_failure_component,
        )

    def _write_chapter_vlm_result(
        self,
        writer: LiveReportRunWriter,
        evaluation_run_id: str,
        manifest: object | None,
        package: ValidatedEvaluationPackage,
        receipt: object | None,
        score: object | None,
        identity: ProductionModelIdentityReport | LegacyProductionModelIdentityReport,
        *,
        status: Literal[GateStatus.PASS, GateStatus.FAIL],
        failure_code: ErrorCode | None = None,
        failure_receipt: object | None = None,
        failure_component: Literal["chapter_vlm", "components_close"] | None = None,
    ) -> GateCheck:
        from video_demo.evaluation.chapter_vlm_input import (
            ChapterVlmInputManifest,
            chapter_vlm_input_manifest_sha256,
        )
        from video_demo.evaluation.chapter_vlm_live import VisualTextScoreFact

        implementation = _current_live_implementation_sha256(self._settings.workspace_root)
        runtime = self._store.runtime_root
        if isinstance(manifest, ChapterVlmInputManifest):
            sample_id = manifest.sample_id
            source_media_sha256 = manifest.source_media_sha256
            annotation_sha256 = manifest.annotation_sha256
            run_id = manifest.evaluation_run_id
        else:
            candidate = package.dataset.samples[0]
            sample_id = candidate.sample_id
            source_media_sha256 = candidate.media_sha256
            annotation_sha256 = next(
                annotation.sha256
                for annotation in package.annotations
                if annotation.annotation.sample_id == sample_id
            )
            run_id = evaluation_run_id
        source_path = _package_media_path(package, sample_id)
        source_snapshot = _read_file_snapshot(
            source_path, max_bytes=self._settings.max_video_bytes, capture_content=False
        )
        if source_snapshot.sha256 != source_media_sha256:
            raise VideoDemoError(
                ErrorCode.VIDEO_DIGEST_MISMATCH,
                "章节 VLM 源媒体摘要与授权样本不一致",
            )
        source_input = LiveInputArtifact(
            kind="SOURCE_MEDIA",
            sample_id=sample_id,
            relative_path=source_path.relative_to(self._settings.workspace_root).as_posix(),
            sha256=source_snapshot.sha256,
            source_media_sha256=source_media_sha256,
            size_bytes=source_snapshot.identity.size,
        )
        manifest_input = None
        frame_inputs: tuple[LiveInputArtifact, ...] = ()
        run_root = runtime / "runs/evaluation" / run_id
        if isinstance(manifest, ChapterVlmInputManifest):
            manifest_path = run_root / "visual/chapter-vlm-input.json"
            manifest_input = LiveInputArtifact(
                kind="FRAME_MANIFEST",
                sample_id=manifest.sample_id,
                relative_path=manifest_path.relative_to(self._settings.workspace_root).as_posix(),
                sha256=chapter_vlm_input_manifest_sha256(manifest),
                source_media_sha256=manifest.source_media_sha256,
                size_bytes=manifest_path.stat().st_size,
            )
            frame_inputs = tuple(
            LiveInputArtifact(
                kind="CHAPTER_FRAME",
                sample_id=manifest.sample_id,
                relative_path=(run_root / frame.relative_path)
                .relative_to(self._settings.workspace_root)
                .as_posix(),
                sha256=frame.sha256,
                source_media_sha256=manifest.source_media_sha256,
                size_bytes=frame.size_bytes,
            )
            for frame in manifest.frames
            )
        raw = ChapterVlmLiveRawReport(
            schema_version="1.0.0",
            check_id="chapter_vlm_live",
            status=status,
            execution_started=True,
            parent_evaluation_run_id=(
                manifest.parent_evaluation_run_id if isinstance(manifest, ChapterVlmInputManifest)
                else evaluation_run_id
            ),
            evaluation_run_id=evaluation_run_id,
            sample_id=sample_id,
            annotation_sha256=annotation_sha256,
            source_media_input=source_input,
            frame_manifest_input=manifest_input,
            chapter_frames=frame_inputs,
            model=next(model for model in identity.models if model.component == "chapter_vlm"),
            operation="analyze_chapter",
            settings_fingerprint=identity.settings_fingerprint,
            implementation_sha256=implementation,
            call_receipt=receipt,
            response_sha256=getattr(receipt, "response_sha256", None),
            visual_text_score_fact=score,
            failure_receipt=failure_receipt,
            failure_code=failure_code,
            failure_component=failure_component
            if failure_component is not None
            else ("chapter_vlm" if failure_code is not None else None),
        )
        raw_artifact = writer.write_artifact(
            "raw.json", "AUDIT_REPORT", raw.model_dump_json(exclude_none=True).encode()
        )
        score_artifact = None
        if isinstance(score, VisualTextScoreFact):
            score_artifact = writer.write_artifact(
                "visual-text-score.json", "QUALITY_DETAIL", score.model_dump_json().encode()
            )
        stdout = writer.write_artifact("trace.stdout.txt", "COMMAND_STDOUT", b"")
        stderr = writer.write_artifact("trace.stderr.txt", "COMMAND_STDERR", b"")
        dataset_artifact = self._bind_package_artifact(
            runtime / "eval/dataset.jsonl", "DATASET_MANIFEST"
        )
        authorization_artifact = self._bind_package_artifact(
            runtime / "eval/authorization.json", "AUTHORIZATION_RECORD"
        )
        input_artifacts = tuple(
            self._store.bind_artifact(
                Path(item.relative_path).relative_to(".codex/video-rag-demo"),
                "INPUT_MEDIA",
                max_bytes=(
                    self._settings.model_max_response_bytes
                    if item.kind == "FRAME_MANIFEST"
                    else self._settings.vlm_max_image_bytes
                    if item.kind == "CHAPTER_FRAME"
                    else self._settings.max_video_bytes
                ),
            )
            for item in (
                source_input,
                *((manifest_input,) if manifest_input else ()),
                *frame_inputs,
            )
        )
        details = ChapterVlmLiveDetails(
            type="CHAPTER_VLM",
            trace=CommandTrace(
                command=("python", "-m", "video_demo.evaluation.live_runner"),
                exit_code=0 if status == GateStatus.PASS else 1,
                stdout_sha256=stdout.sha256,
                stderr_sha256=stderr.sha256,
            ),
            raw_report_sha256=raw_artifact.sha256,
            dataset_sha256=dataset_artifact.sha256,
            authorization_sha256=authorization_artifact.sha256,
            implementation_sha256=implementation,
            settings_fingerprint=identity.settings_fingerprint,
            status=status,
            parent_evaluation_run_id=(
                manifest.parent_evaluation_run_id
                if isinstance(manifest, ChapterVlmInputManifest)
                else evaluation_run_id
            ),
            evaluation_run_id=evaluation_run_id,
            sample_id=sample_id,
            manifest_sha256=manifest_input.sha256 if manifest_input else None,
            model=raw.model,
            operation="analyze_chapter",
            response_sha256=raw.response_sha256,
            visual_text_score_fact_sha256=score_artifact.sha256 if score_artifact else None,
            failure_code=failure_code,
            failure_receipt=failure_receipt,
        )
        report = MachineEvidenceReport(
            schema_version="1.0.0",
            check_id="chapter_vlm_live",
            status=status,
            kind=EvidenceKind.LIVE_SERVICE_REPORT,
            level=EvidenceLevel.REAL_SERVICE,
            covered_items=("chapter_vlm_live",),
            summary="章节 VLM live 检查完成"
            if status == GateStatus.PASS
            else "章节 VLM live 检查失败",
            producer="LiveValidationRunner",
            started_at=datetime.now(UTC),
            finished_at=datetime.now(UTC),
            artifacts=(
                raw_artifact,
                *((score_artifact,) if score_artifact else ()),
                dataset_artifact,
                authorization_artifact,
                *input_artifacts,
                stdout,
                stderr,
            ),
            details=details,
        )
        return writer.write_json(report, settings=self._settings)

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
        identity: ProductionModelIdentityReport | LegacyProductionModelIdentityReport
        if check_id in {"baidu_ocr_live", "qwen_live", "five_language_models"}:
            # 11B 前旧诊断报告仍需使用旧模型身份；允许缺失新文本/VLM
            # 配置，章节 VLM 仍使用 3.0 生产身份。
            from video_demo.application.legacy_composition import (
                build_production_model_identity_report as build_legacy_identity,
            )

            identity = build_legacy_identity(self._settings)
        else:
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
        # 旧 live 入口在 11B 前仍由 legacy_composition 提供组件；这里只按
        # 检查 ID 分派，避免新章节组件类型检查误伤历史诊断链。
        if check_id == "baidu_ocr_live":
            self._execute_baidu(samples[0], components, journal)
        elif check_id == "qwen_live":
            self._execute_qwen(samples[0], components, journal)
        elif check_id == "five_language_models":
            self._execute_local_model_stack(samples, components, journal)
        else:
            raise VideoDemoError(ErrorCode.SYSTEM_FAILURE, "章节 VLM 必须使用专用执行路径")

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
        capabilities, probe_receipt = components.qwen_client.probe_capabilities_with_receipt(clip)
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
        result, segment_receipt = components.qwen_client.understand_segment_with_receipt(request)
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
        components: object,
        journal: _LiveExecutionJournal,
    ) -> None:
        speech_models = getattr(components, "speech_models", None)
        if speech_models is None:
            raise VideoDemoError(ErrorCode.SYSTEM_FAILURE, "语音诊断组件缺少模型")
        by_language = {sample.language: sample for sample in samples}
        ordered = tuple(by_language[language] for language in _VALIDATION_LANGUAGES)
        first = ordered[0]
        first_audio = self._verified_live_input_path(
            first.audio_relative_path,
            first.audio_sha256,
            max_bytes=self._settings.max_video_bytes,
        )
        vad = speech_models.vad.detect(
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
            result = speech_models.recognizer.transcribe_window(
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
        components: Any,
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
        return cast(ModelIdentity, candidates[0])

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
            samples = (self._select_single_sample(package, evaluation_run_id, filename),)
        inputs = tuple(
            input_artifact
            for sample, _snapshots in samples
            for input_artifact in _live_inputs(sample, _snapshots)
        )
        return _PreparedLiveRun(
            samples=tuple(sample for sample, _snapshots in samples),
            inputs=inputs,
            snapshots=tuple(
                snapshot for _sample, sample_snapshots in samples for snapshot in sample_snapshots
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
        return self._store.runtime_root / "eval/live" / evaluation_run_id / sample_id / filename

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
        implementation = _current_live_implementation_sha256(self._settings.workspace_root)
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
        identity: ProductionModelIdentityReport | LegacyProductionModelIdentityReport,
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
        identity: ProductionModelIdentityReport | LegacyProductionModelIdentityReport,
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
        identity: ProductionModelIdentityReport | LegacyProductionModelIdentityReport,
        *,
        status: Literal[GateStatus.PASS, GateStatus.FAIL],
        failure_code: ErrorCode | None,
        failure_component: LiveFailureComponent | None,
    ) -> GateCheck:
        implementation = _current_live_implementation_sha256(self._settings.workspace_root)
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
            type[BaiduLiveDetails] | type[QwenLiveDetails] | type[FiveLanguageModelsDetails]
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
            summary=("live 检查执行成功" if status == GateStatus.PASS else "live 检查执行失败"),
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
            self._store.runtime_root / "eval" / "reports" / evaluation_run_id / f"{check_id}.json"
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
    ordered = tuple(
        by_language[language] for language in _VALIDATION_LANGUAGES if language in by_language
    )
    if len(ordered) != len(_VALIDATION_LANGUAGES):
        raise VideoDemoError(ErrorCode.EVALUATION_ARTIFACT_INVALID, "五语 live 样本不完整")
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
