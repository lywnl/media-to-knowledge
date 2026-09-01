from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path, PurePosixPath
from typing import Any, cast
from xml.etree import ElementTree

from pydantic import Field, ValidationError, ValidationInfo, model_validator

from video_demo.application.composition import (
    build_production_model_identity_report,
    production_tool_path,
)
from video_demo.config import Settings
from video_demo.domain.base import FrozenModel
from video_demo.domain.document_artifact import DocumentArtifactPayload
from video_demo.errors import ErrorCode
from video_demo.evaluation.annotations import (
    AuthorizationFile,
    ValidatedEvaluationPackage,
    load_evaluation_package,
)
from video_demo.evaluation.chapter_vlm_input import (
    ChapterVlmInputManifest,
    ValidatedChapterVlmInputContext,
    _source_display_size,
    chapter_vlm_input_manifest_sha256,
    validate_chapter_vlm_input_manifest,
)
from video_demo.evaluation.chapter_vlm_live import VisualTextScoreFact
from video_demo.evaluation.document_judgments import DocumentQualityReport
from video_demo.evaluation.evidence import (
    _CHUNK_BYTES,
    ArtifactRole,
    AuthorizedDatasetDetails,
    ChapterVlmLiveDetails,
    ChapterVlmLiveRawReport,
    CommandEvidenceDetails,
    EvidenceKind,
    EvidenceLevel,
    EvidenceReference,
    FileIdentity,
    FileSnapshot,
    FiveLanguageModelsDetails,
    FiveLanguageModelsRawReport,
    LiveExecutionSummary,
    LiveInputArtifact,
    LiveSample,
    LiveServiceDetails,
    MachineEvidenceReport,
    ModelExecutionFact,
    OfflineEvidenceDetails,
    OfflineRawReport,
    PerformanceDetails,
    PerformanceRawReport,
    PerformanceSampleRawReport,
    PreflightDetails,
    PreflightRawReport,
    RealMediaDetails,
    RealMediaFile,
    RealMediaRawReport,
    StaticAuditDetails,
    VerifiedArtifact,
    _assert_snapshot_current,
    _LiveCheckDetails,
    _LiveRawReport,
    _open_regular_file,
    _read_file_snapshot,
    build_verified_gate_check,
    offline_audited_paths,
    offline_command,
    offline_observation_sha256,
)
from video_demo.evaluation.evidence import CommandTrace as CommandTrace
from video_demo.evaluation.report import GateStatus, QualityReport
from video_demo.implementation import implementation_import_closure
from video_demo.media.probe import FFprobeClient, ProbeLimits, SupportedMime

REQUIRED_FAILURE_SCENARIOS: tuple[str, ...] = (
    "corrupted_media",
    "spoofed_mime",
    "vfr",
    "rotation",
    "no_audio",
    "no_speech",
    "black_frames",
    "malformed_json",
    "cancellation",
    "retry",
    "restart_resume",
    "disk_insufficient",
    "cross_tenant",
    "redaction",
    "prompt_injection",
)

_IMPLEMENTATION_REFERENCE_ROOT = Path(__file__).resolve().parents[3]
_GATE_VERIFIER_LEAF = frozenset({Path("src/video_demo/evaluation/gate.py")})
_REAL_MEDIA_IMPLEMENTATION_FILES: tuple[Path, ...] = implementation_import_closure(
    _IMPLEMENTATION_REFERENCE_ROOT,
    (
        Path("src/video_demo/evaluation/media_runner.py"),
        Path("src/video_demo/evaluation/real_media_execution.py"),
        Path("src/video_demo/evaluation/real_media_source.py"),
    ),
    leaf_files=_GATE_VERIFIER_LEAF | frozenset({Path("src/video_demo/application/pipeline.py")}),
)

_LIVE_ONLY_PRODUCTION_ISOLATION_FILES = frozenset(
    {
        Path("src/video_demo/speech/snapshots.py"),
        Path("src/video_demo/storage/artifacts.py"),
        Path("src/video_demo/storage/snapshots.py"),
    }
)
_LIVE_IMPLEMENTATION_FILES: tuple[Path, ...] = implementation_import_closure(
    _IMPLEMENTATION_REFERENCE_ROOT,
    (Path("src/video_demo/evaluation/live_runner.py"),),
    extra_files=(
        Path("pyproject.toml"),
        Path("uv.lock"),
    ),
    excluded_files=_LIVE_ONLY_PRODUCTION_ISOLATION_FILES,
    leaf_files=_GATE_VERIFIER_LEAF,
)

_DURABILITY_IMPLEMENTATION_FILES: tuple[Path, ...] = implementation_import_closure(
    _IMPLEMENTATION_REFERENCE_ROOT,
    (Path("src/video_demo/evaluation/durability.py"),),
    extra_files=(
        Path("pyproject.toml"),
        Path("uv.lock"),
    ),
    leaf_files=_GATE_VERIFIER_LEAF,
)

FAILURE_SCENARIO_TESTS: dict[str, tuple[str, ...]] = {
    "corrupted_media": (
        "tests/media/test_probe.py::test_ffprobe_client_rejects_corrupted_media_decode_failure",
    ),
    "spoofed_mime": (
        "tests/storage/test_object_store.py::test_ingest_rejects_extension_and_declared_mime_mismatch",
    ),
    "vfr": ("tests/media/test_probe.py::test_parse_rotation_vfr_and_no_audio_warning",),
    "rotation": ("tests/media/test_probe.py::test_parse_rotation_vfr_and_no_audio_warning",),
    "no_audio": (
        "tests/media/test_transcode.py::test_extract_audio_without_track_returns_explicit_no_audio",
        "tests/application/test_document_pipeline.py::test_pipeline_protocols_use_time_point_frame_search",
    ),
    "no_speech": (
        "tests/application/test_document_pipeline.py::test_pipeline_does_not_reference_scene_index_runtime",
    ),
    "black_frames": (
        "tests/application/test_chapter_frames.py::test_single_frame_failure_degrades_chapter_but_keeps_successful_frames",
    ),
    "malformed_json": (
        "tests/integrations/test_qwen_vl.py::test_qwen_request_rejection_is_not_response_content_invalid",
    ),
    "cancellation": (
        "tests/application/test_video_scheduler.py::test_scheduler_skips_cancelled_pending_run",
        "tests/application/test_image_pipeline_executor.py::test_executor_respects_cancellation_before_handler",
    ),
    "retry": (
        "tests/persistence/test_video_stage_repository.py::test_stage_repository_creates_claims_and_recovers_only_due_stages",
    ),
    "restart_resume": (
        "tests/application/test_video_scheduler.py::test_scheduler_recovers_checkpointed_run_into_llm_queue",
        "tests/persistence/test_video_stage_repository.py::test_stage_repository_creates_claims_and_recovers_only_due_stages",
    ),
    "disk_insufficient": (
        "tests/media/test_transcode.py::test_transcode_rejects_insufficient_disk_before_starting",
    ),
    "cross_tenant": (
        "tests/api/test_results.py::test_result_routes_are_scope_isolated",
        "tests/api/test_objects.py::test_object_lookup_is_hidden_from_other_tenant",
    ),
    "redaction": (
        "tests/api/test_runs.py::test_run_status_does_not_expose_internal_paths_or_secret_fields",
        "tests/integrations/test_qwen_vl.py::test_qwen_authentication_status_is_not_masked_by_oversized_body",
    ),
    "prompt_injection": (
        "tests/integrations/test_prompt_isolation.py::test_untrusted_asr_text_never_enters_trusted_system_instruction",
    ),
}

_ALLOWED_AUTOMATED_XFAILS: dict[str, str] = {
    (
        "tests/evaluation/test_live_runner.py::"
        "test_not_run_verification_rejects_directory_aba_during_entire_verifier"
    ): (
        "已知延期：路径型 live verifier 存在目录 ABA 窗口；单进程 Demo 主流程不受影响，"
        "后续改为基于 writer fd 的同源验证"
    ),
}

FINAL_GATE_CHECKS: tuple[str, ...] = (
    "failure_matrix",
    "no_indexing",
    "automated_tests",
    "ruff",
    "mypy",
    "alembic_roundtrip",
    "openapi_contract",
    "secret_scan",
    "authorized_dataset",
    "real_media_chain",
    "chapter_vlm_live",
    "five_language_models",
    "m1_durability",
)

_LIVE_GATE_CHECKS = frozenset(
    {
        "chapter_vlm_live",
        "five_language_models",
    }
)
# 11B 前保留旧报告的离线重验能力；这些 ID 不会出现在 FINAL_GATE_CHECKS、CLI
# 或新的活动汇总中。
_LIVE_AUTHORITY_CHECKS = _LIVE_GATE_CHECKS
_LIVE_IMPLEMENTATION_CHECKS = _LIVE_AUTHORITY_CHECKS

_LOCAL_MODEL_FAILURE_CODES = frozenset(
    {
        ErrorCode.SPEECH_DEPENDENCY_UNAVAILABLE,
        ErrorCode.SPEECH_MODEL_UNAVAILABLE,
        ErrorCode.SPEECH_AUDIO_INVALID,
        ErrorCode.VIDEO_DIGEST_MISMATCH,
        ErrorCode.WORKSPACE_PATH_ESCAPE,
        ErrorCode.SYSTEM_FAILURE,
    }
)
_LIVE_COMPONENT_FAILURE_CODES: dict[str, frozenset[ErrorCode]] = {
    "chapter_vlm": frozenset(
        {
            ErrorCode.QWEN_CAPABILITY_UNAVAILABLE,
            ErrorCode.QWEN_AUTHENTICATION_FAILED,
            ErrorCode.QWEN_REQUEST_REJECTED,
            ErrorCode.QWEN_RESPONSE_INVALID,
            ErrorCode.DEPENDENCY_TEMPORARY_FAILURE,
            ErrorCode.ARTIFACT_SCHEMA_INVALID,
            ErrorCode.INPUT_BUDGET_EXCEEDED,
            ErrorCode.VISUAL_RESULT_INVALID,
            ErrorCode.VISUAL_MEDIA_INVALID,
            ErrorCode.VIDEO_DIGEST_MISMATCH,
            ErrorCode.WORKSPACE_PATH_ESCAPE,
            ErrorCode.IDEMPOTENCY_CONFLICT,
            ErrorCode.JOB_CANCELLED,
            ErrorCode.SYSTEM_FAILURE,
        }
    ),
    "silero_vad": _LOCAL_MODEL_FAILURE_CODES,
    "cloud_whisper": _LOCAL_MODEL_FAILURE_CODES,
    "components_close": frozenset({ErrorCode.SYSTEM_FAILURE}),
}

_MISSING_CHECK_REASONS: dict[str, str] = {
    "failure_matrix": "未提供失败矩阵测试证据",
    "no_indexing": "未提供无索引源码审计证据",
    "automated_tests": "未提供全量自动化测试结果",
    "ruff": "未提供 ruff 检查结果",
    "mypy": "未提供 mypy 检查结果",
    "alembic_roundtrip": "未提供 Alembic 往返证据",
    "openapi_contract": "未提供 OpenAPI 契约审计证据",
    "secret_scan": "未提供敏感字段审计证据",
    "authorized_dataset": "缺少授权五语评测集",
    "real_media_chain": "缺少工作区 FFmpeg/ffprobe 与真实媒体运行结果",
    "chapter_vlm_live": "缺少章节多图 Qwen3-VL 凭据或真实联调结果",
    "five_language_models": "缺少五语授权素材、模型或真实预测",
    "m1_durability": "未提供 M1 耐久机器证据",
}

_DURABILITY_ISSUE_REASONS: dict[ErrorCode, str] = {
    ErrorCode.M1_SAMPLE_COUNT_INVALID: "耐久样本必须恰好两段",
    ErrorCode.M1_DURATION_TOO_SHORT: "耐久样本时长不足 30 分钟",
    ErrorCode.M1_DURATION_TOO_LONG: "耐久样本时长超过 2 小时上限",
    ErrorCode.M1_RESOLUTION_TOO_SMALL: "耐久样本分辨率低于 1920×1080",
    ErrorCode.M1_AUTHORIZATION_UNAVAILABLE: "耐久素材授权记录不可用",
    ErrorCode.M1_MEDIA_INVALID: "耐久媒体或 Manifest 不可用",
    ErrorCode.M1_PROBE_MISMATCH: "耐久媒体探测结果与 Manifest 不一致",
    ErrorCode.INVALID_CONFIGURATION: "CPU/int8/单并发配置不满足要求",
    ErrorCode.M1_PSUTIL_UNAVAILABLE: "psutil 不可用",
    ErrorCode.VIDEO_FFMPEG_UNAVAILABLE: "工作区 FFmpeg/ffprobe 不可用",
    ErrorCode.VIDEO_FFPROBE_UNAVAILABLE: "工作区 FFmpeg/ffprobe 不可用",
    ErrorCode.VIDEO_BINARY_PROBE_FAILED: "工作区 FFmpeg/ffprobe 不可用",
    ErrorCode.VISUAL_DEPENDENCY_UNAVAILABLE: "视觉依赖不可用",
    ErrorCode.SILERO_DEPENDENCY_UNAVAILABLE: "本地音频模型、依赖或授权不可用",
    ErrorCode.SILERO_MODEL_UNAVAILABLE: "本地音频模型、依赖或授权不可用",
    ErrorCode.VIDEO_DISK_SPACE_INSUFFICIENT: "可用磁盘空间不足",
}


def build_durability_not_run_reason(issues: Sequence[ErrorCode]) -> str:
    """只描述原始 preflight 实际报告的耐久缺项，并按稳定错误码顺序去重。"""

    order = tuple(_DURABILITY_ISSUE_REASONS)
    reasons = tuple(
        dict.fromkeys(
            _DURABILITY_ISSUE_REASONS[issue] for issue in sorted(set(issues), key=order.index)
        )
    )
    return f"M1 耐久前置条件不足: {'；'.join(reasons)}"


class GateCheck(FrozenModel):
    check_id: str = Field(min_length=3, max_length=128)
    status: GateStatus
    evidence: tuple[EvidenceReference, ...] = ()
    not_run_reason: str | None = Field(default=None, min_length=1, max_length=500)

    @model_validator(mode="after")
    def validate_check(self) -> GateCheck:
        if self.status == GateStatus.NOT_RUN:
            if self.not_run_reason is None:
                raise ValueError("NOT_RUN 检查必须提供未运行原因")
        elif not self.evidence or self.not_run_reason is not None:
            raise ValueError("已运行检查必须只提供证据")
        return self


_FINAL_GATE_BUILD_TOKEN = object()


class FinalGateReport(FrozenModel):
    status: GateStatus
    quality: QualityReport
    document_quality: DocumentQualityReport | None = None
    checks: tuple[GateCheck, ...] = Field(min_length=len(FINAL_GATE_CHECKS))

    @model_validator(mode="after")
    def validate_report(self, info: ValidationInfo) -> FinalGateReport:
        check_ids = tuple(check.check_id for check in self.checks)
        if len(check_ids) != len(set(check_ids)) or set(check_ids) != set(FINAL_GATE_CHECKS):
            raise ValueError("最终门禁必须且只能包含全部权威检查")
        if self.status != _aggregate_status(self.quality, self.checks, self.document_quality):
            raise ValueError("最终状态必须与质量和权威检查一致")
        workspace_root = info.context.get("workspace_root") if info.context else None
        if not isinstance(workspace_root, Path):
            raise ValueError("最终报告必须提供工作区证据验证上下文")
        if info.context is None or info.context.get("build_token") is not _FINAL_GATE_BUILD_TOKEN:
            raise ValueError("最终报告必须通过权威构建流程生成")
        settings = info.context.get("settings") if info.context else None
        if settings is not None and not isinstance(settings, Settings):
            raise ValueError("最终报告 settings 验证上下文非法")
        for check in self.checks:
            _verify_gate_check(check, workspace_root, settings=settings)
        return self


def build_failure_matrix_check(
    report_path: Path,
    *,
    workspace_root: Path,
) -> GateCheck:
    failure_message: str | None = None
    try:
        return _build_failure_matrix_check(report_path, workspace_root=workspace_root)
    except (OSError, UnicodeDecodeError):
        failure_message = "失败矩阵证据非法或不可信"
    except ValueError as error:
        failure_message = str(error)
    raise ValueError(failure_message)


def _build_failure_matrix_check(
    report_path: Path,
    *,
    workspace_root: Path,
) -> GateCheck:
    _, relative_path, encoded = _load_workspace_evidence(report_path, workspace_root)
    outcomes = _parse_failure_matrix_junit(encoded)
    failed = tuple(
        scenario for scenario in REQUIRED_FAILURE_SCENARIOS if outcomes[scenario] == "FAIL"
    )
    missing = tuple(
        scenario for scenario in REQUIRED_FAILURE_SCENARIOS if outcomes[scenario] == "NOT_RUN"
    )
    covered = tuple(
        scenario for scenario in REQUIRED_FAILURE_SCENARIOS if outcomes[scenario] != "NOT_RUN"
    )
    evidence = EvidenceReference(
        kind=EvidenceKind.PYTEST_JUNIT,
        level=EvidenceLevel.CONTRACT,
        relative_path=relative_path,
        sha256=hashlib.sha256(encoded).hexdigest(),
        covered_items=covered or ("failure_matrix_report",),
        summary=(
            f"pytest JUnit 离线契约报告：{len(covered)}/{len(REQUIRED_FAILURE_SCENARIOS)} "
            "个失败场景已完整执行"
        ),
    )
    if failed:
        return GateCheck(
            check_id="failure_matrix",
            status=GateStatus.FAIL,
            evidence=(evidence,),
        )
    if missing:
        return GateCheck(
            check_id="failure_matrix",
            status=GateStatus.NOT_RUN,
            evidence=(evidence,),
            not_run_reason=f"缺少失败场景证据: {', '.join(missing)}",
        )
    return GateCheck(
        check_id="failure_matrix",
        status=GateStatus.PASS,
        evidence=(evidence,),
    )


def build_automated_tests_check(
    report_path: Path,
    *,
    collection_path: Path,
    workspace_root: Path,
) -> GateCheck:
    failure_message: str | None = None
    try:
        return _build_automated_tests_check(
            report_path,
            collection_path=collection_path,
            workspace_root=workspace_root,
        )
    except (OSError, UnicodeDecodeError):
        failure_message = "全量自动化测试证据非法或不可信"
    except ValueError as error:
        failure_message = str(error)
    raise ValueError(failure_message)


def _build_automated_tests_check(
    report_path: Path,
    *,
    collection_path: Path,
    workspace_root: Path,
) -> GateCheck:
    _, report_relative_path, report_encoded = _load_workspace_evidence(
        report_path,
        workspace_root,
    )
    _, collection_relative_path, collection_encoded = _load_workspace_evidence(
        collection_path,
        workspace_root,
    )
    junit_root = _parse_junit_xml(report_encoded)
    observed = _junit_case_statuses(junit_root)
    acknowledged_xfails = _acknowledged_automated_xfail_identities(junit_root)
    collected = _parse_pytest_collection(collection_encoded)
    authoritative, collection_error = _collect_current_pytest_nodes(workspace_root)
    collected_identities = {_pytest_collection_identity(node_id): node_id for node_id in collected}
    executed_identities = set(observed)
    missing_identities = set(collected_identities) - executed_identities
    unexpected_identities = executed_identities - set(collected_identities)
    missing = tuple(sorted(collected_identities[identity] for identity in missing_identities))
    unexpected = tuple(sorted(f"{classname}::{name}" for classname, name in unexpected_identities))
    source_missing = tuple(sorted(authoritative - collected))
    source_unexpected = tuple(sorted(collected - authoritative))
    statuses = tuple(
        status
        for identity, values in observed.items()
        for status in values
        if status != "NOT_RUN" or identity not in acknowledged_xfails
    )
    evidence = (
        EvidenceReference(
            kind=EvidenceKind.PYTEST_JUNIT,
            level=EvidenceLevel.CONTRACT,
            relative_path=report_relative_path,
            sha256=hashlib.sha256(report_encoded).hexdigest(),
            covered_items=("automated_tests",),
            summary=f"pytest JUnit 报告包含 {len(executed_identities)} 个精确用例身份",
        ),
        EvidenceReference(
            kind=EvidenceKind.PYTEST_COLLECTION,
            level=EvidenceLevel.CONTRACT,
            relative_path=collection_relative_path,
            sha256=hashlib.sha256(collection_encoded).hexdigest(),
            covered_items=("automated_tests",),
            summary=f"pytest 收集清单包含 {len(collected)} 个精确 node ID",
        ),
    )
    if "FAIL" in statuses:
        return GateCheck(check_id="automated_tests", status=GateStatus.FAIL, evidence=evidence)
    incomplete = (*missing, *unexpected, *source_missing, *source_unexpected)
    if not collected or collection_error or incomplete or "NOT_RUN" in statuses:
        details = _pytest_incomplete_reason(missing, unexpected, statuses)
        if collection_error:
            details = f"{details}；当前源码收集失败: {collection_error}"
        if source_missing or source_unexpected:
            source_details = _source_collection_mismatch_reason(
                source_missing,
                source_unexpected,
            )
            details = f"{details}；{source_details}"
        return GateCheck(
            check_id="automated_tests",
            status=GateStatus.NOT_RUN,
            evidence=evidence,
            not_run_reason=f"全量 pytest 证据不完整: {details}",
        )
    return GateCheck(check_id="automated_tests", status=GateStatus.PASS, evidence=evidence)


def build_final_gate_report(
    *,
    quality: QualityReport,
    checks: Sequence[GateCheck],
    workspace_root: Path,
    settings: Settings | None = None,
    document_quality: DocumentQualityReport | None = None,
) -> FinalGateReport:
    failure_message: str | None = None
    try:
        return _build_final_gate_report(
            quality=quality,
            checks=checks,
            workspace_root=workspace_root,
            settings=settings,
            document_quality=document_quality,
        )
    except ValidationError:
        failure_message = "最终门禁报告非法或不可信"
    except OSError:
        failure_message = "最终门禁报告非法或不可信"
    except ValueError as error:
        failure_message = str(error)
    raise ValueError(failure_message)


def _build_final_gate_report(
    *,
    quality: QualityReport,
    checks: Sequence[GateCheck],
    workspace_root: Path,
    settings: Settings | None,
    document_quality: DocumentQualityReport | None,
) -> FinalGateReport:
    supplied = {check.check_id: check for check in checks}
    if len(supplied) != len(checks):
        raise ValueError("最终门禁检查 ID 不得重复")
    unknown = sorted(set(supplied) - set(FINAL_GATE_CHECKS))
    if unknown:
        raise ValueError(f"未知最终检查: {', '.join(unknown)}")
    if any(check.status == GateStatus.NOT_RUN and not check.evidence for check in checks):
        raise ValueError("无证据 NOT_RUN 不能作为权威输入")
    completed = tuple(
        supplied.get(
            check_id,
            GateCheck(
                check_id=check_id,
                status=GateStatus.NOT_RUN,
                not_run_reason=_MISSING_CHECK_REASONS[check_id],
            ),
        )
        for check_id in FINAL_GATE_CHECKS
    )
    for check in completed:
        _verify_gate_check(check, workspace_root, settings=settings)
    return FinalGateReport.model_validate(
        {
            "status": _aggregate_status(quality, completed, document_quality),
            "quality": quality,
            "document_quality": document_quality,
            "checks": completed,
        },
        context={
            "workspace_root": workspace_root,
            "settings": settings,
            "build_token": _FINAL_GATE_BUILD_TOKEN,
        },
    )


def _aggregate_status(
    quality: QualityReport,
    checks: Sequence[GateCheck],
    document_quality: DocumentQualityReport | None = None,
) -> GateStatus:
    statuses = [quality.status, *(check.status for check in checks)]
    if document_quality is not None:
        statuses.append(GateStatus(document_quality.status))
    if GateStatus.FAIL in statuses:
        return GateStatus.FAIL
    if GateStatus.NOT_RUN in statuses:
        return GateStatus.NOT_RUN
    return GateStatus.PASS


_MINIMUM_EVIDENCE_LEVEL: dict[str, EvidenceLevel] = {
    "failure_matrix": EvidenceLevel.CONTRACT,
    "no_indexing": EvidenceLevel.STATIC,
    "automated_tests": EvidenceLevel.CONTRACT,
    "ruff": EvidenceLevel.STATIC,
    "mypy": EvidenceLevel.STATIC,
    "alembic_roundtrip": EvidenceLevel.CONTRACT,
    "openapi_contract": EvidenceLevel.CONTRACT,
    "secret_scan": EvidenceLevel.STATIC,
    "authorized_dataset": EvidenceLevel.REAL_MEDIA,
    "real_media_chain": EvidenceLevel.REAL_MEDIA,
    "chapter_vlm_live": EvidenceLevel.REAL_SERVICE,
    "five_language_models": EvidenceLevel.REAL_SERVICE,
    "m1_durability": EvidenceLevel.PERFORMANCE,
}

_ALLOWED_EVIDENCE_KINDS: dict[str, frozenset[EvidenceKind]] = {
    "failure_matrix": frozenset({EvidenceKind.PYTEST_JUNIT}),
    "no_indexing": frozenset({EvidenceKind.STATIC_AUDIT}),
    "automated_tests": frozenset(
        {EvidenceKind.PYTEST_JUNIT, EvidenceKind.PYTEST_COLLECTION},
    ),
    "ruff": frozenset({EvidenceKind.STATIC_AUDIT}),
    "mypy": frozenset({EvidenceKind.STATIC_AUDIT}),
    "alembic_roundtrip": frozenset({EvidenceKind.COMMAND_REPORT}),
    "openapi_contract": frozenset({EvidenceKind.COMMAND_REPORT}),
    "secret_scan": frozenset({EvidenceKind.STATIC_AUDIT}),
    "authorized_dataset": frozenset({EvidenceKind.COMMAND_REPORT}),
    "real_media_chain": frozenset({EvidenceKind.COMMAND_REPORT}),
    "chapter_vlm_live": frozenset({EvidenceKind.LIVE_SERVICE_REPORT}),
    "five_language_models": frozenset({EvidenceKind.LIVE_SERVICE_REPORT}),
    "m1_durability": frozenset({EvidenceKind.PERFORMANCE_REPORT}),
}

_EVIDENCE_LEVEL_RANK = {
    EvidenceLevel.CONTRACT: 1,
    EvidenceLevel.STATIC: 1,
    EvidenceLevel.REAL_MEDIA: 2,
    EvidenceLevel.REAL_SERVICE: 2,
    EvidenceLevel.PERFORMANCE: 3,
}

_MAX_EVIDENCE_BYTES = 64 * 1024 * 1024
_MAX_LIVE_MEDIA_BYTES = 4 * 1024 * 1024 * 1024


def _verify_gate_check(
    check: GateCheck,
    workspace_root: Path,
    *,
    settings: Settings | None = None,
) -> None:
    if check.status == GateStatus.NOT_RUN and not check.evidence:
        return
    for evidence in check.evidence:
        _verify_evidence_reference(evidence, workspace_root)
        if evidence.kind not in _ALLOWED_EVIDENCE_KINDS[check.check_id]:
            raise ValueError(f"{check.check_id} 的证据类型不符合门禁要求")
        required_level = _MINIMUM_EVIDENCE_LEVEL[check.check_id]
        if _EVIDENCE_LEVEL_RANK[evidence.level] < _EVIDENCE_LEVEL_RANK[required_level]:
            raise ValueError(f"{check.check_id} 的证据级别不足")
        if evidence.kind not in (EvidenceKind.PYTEST_JUNIT, EvidenceKind.PYTEST_COLLECTION):
            verified = build_verified_gate_check(
                check.check_id,
                Path(evidence.relative_path),
                workspace_root=workspace_root,
                settings=settings,
            )
            if verified.status != check.status or verified.not_run_reason != check.not_run_reason:
                raise ValueError(f"{check.check_id} 的状态与机器明细不一致")
            if verified.evidence != (evidence,):
                raise ValueError(f"{check.check_id} 的机器证据引用元数据不一致")
    if check.check_id == "automated_tests":
        _verify_automated_tests_check(check, workspace_root)
        return
    if check.check_id != "failure_matrix":
        covered = {item for evidence in check.evidence for item in evidence.covered_items}
        if check.evidence and check.check_id not in covered:
            raise ValueError(f"{check.check_id} 的证据未声明覆盖该检查")
        return
    junit_evidence = tuple(
        evidence for evidence in check.evidence if evidence.kind == EvidenceKind.PYTEST_JUNIT
    )
    if len(junit_evidence) != 1:
        raise ValueError("失败矩阵必须且只能引用一份 pytest JUnit 报告")
    evidence = junit_evidence[0]
    parsed = build_failure_matrix_check(
        Path(evidence.relative_path),
        workspace_root=workspace_root,
    )
    if parsed.status != check.status or parsed.not_run_reason != check.not_run_reason:
        raise ValueError("失败矩阵状态与 JUnit 报告不一致")
    if parsed.evidence != check.evidence:
        raise ValueError("失败矩阵引用元数据与 JUnit 报告不一致")


def _verify_evidence_reference(evidence: EvidenceReference, workspace_root: Path) -> bytes:
    _, _, encoded = _load_workspace_evidence(Path(evidence.relative_path), workspace_root)
    if hashlib.sha256(encoded).hexdigest() != evidence.sha256:
        raise ValueError(f"证据文件摘要不匹配: {evidence.relative_path}")
    return encoded


def _derive_machine_gate_status(
    check_id: str,
    report: MachineEvidenceReport,
    artifacts: dict[ArtifactRole, tuple[VerifiedArtifact, ...]],
    workspace_root: Path,
    *,
    settings: Settings | None = None,
) -> tuple[GateStatus, str | None]:
    expected_type: dict[str, type[object] | tuple[type[object], ...]] = {
        "no_indexing": (StaticAuditDetails, OfflineEvidenceDetails),
        "ruff": (StaticAuditDetails, OfflineEvidenceDetails),
        "mypy": (StaticAuditDetails, OfflineEvidenceDetails),
        "secret_scan": (StaticAuditDetails, OfflineEvidenceDetails),
        "alembic_roundtrip": (CommandEvidenceDetails, OfflineEvidenceDetails),
        "openapi_contract": (CommandEvidenceDetails, OfflineEvidenceDetails),
        "authorized_dataset": AuthorizedDatasetDetails,
        "real_media_chain": RealMediaDetails,
        "chapter_vlm_live": ChapterVlmLiveDetails,
        "five_language_models": FiveLanguageModelsDetails,
        "m1_durability": PerformanceDetails,
    }
    if report.kind not in _ALLOWED_EVIDENCE_KINDS[check_id]:
        raise ValueError(f"{check_id} 的机器证据类型不符合门禁要求")
    if report.level != _MINIMUM_EVIDENCE_LEVEL[check_id]:
        raise ValueError(f"{check_id} 的机器证据级别不符合门禁要求")
    details = report.details
    if isinstance(details, PreflightDetails):
        return _derive_preflight_status(check_id, report, details, artifacts, workspace_root)
    required = expected_type.get(check_id)
    if required is None or not isinstance(details, required):
        raise ValueError(f"{check_id} 的机器证据明细类型不符合门禁要求")
    if isinstance(details, OfflineEvidenceDetails):
        passed = _verify_offline_evidence(
            check_id,
            details,
            report,
            artifacts,
            workspace_root,
        )
    elif isinstance(details, ChapterVlmLiveDetails):
        passed = _verify_chapter_vlm_live(
            details, report, artifacts, workspace_root, settings=settings
        )
    elif isinstance(details, FiveLanguageModelsDetails):
        passed = _verify_five_language_models(
            details,
            report,
            artifacts,
            workspace_root,
            settings=settings,
        )
    elif isinstance(details, LiveServiceDetails):
        expected_service = "FIVE_LANGUAGE_MODELS"
        if details.service != expected_service:
            raise ValueError(f"{check_id} 的真实服务身份不匹配")
        _require_artifact_digest(artifacts, "INPUT_MEDIA", details.input_sha256)
        _require_artifact_digest(artifacts, "PROVIDER_RESPONSE", details.output_sha256)
        passed = details.trace.exit_code == 0 and 200 <= details.http_status < 300
    elif isinstance(details, RealMediaDetails):
        passed = _verify_real_media(
            details,
            artifacts,
            workspace_root,
        )
    elif isinstance(details, AuthorizedDatasetDetails):
        _require_artifact_digest(artifacts, "DATASET_MANIFEST", details.manifest_sha256)
        _require_artifact_digest(
            artifacts,
            "AUTHORIZATION_RECORD",
            details.authorization_record_sha256,
        )
        _verify_authorized_dataset(details, artifacts, workspace_root)
        passed = details.trace.exit_code == 0
    elif isinstance(details, PerformanceDetails):
        passed = _verify_performance(
            details,
            artifacts,
            workspace_root,
            settings=settings,
        )
    elif isinstance(details, StaticAuditDetails):
        if len(artifacts.get("AUDIT_REPORT", ())) != 1:
            raise ValueError(f"{check_id} 必须绑定命令或审计原始报告")
        passed = details.trace.exit_code == 0 and details.violation_count == 0
    else:
        if len(artifacts.get("AUDIT_REPORT", ())) != 1:
            raise ValueError(f"{check_id} 必须绑定命令或审计原始报告")
        passed = details.trace.exit_code == 0
    strict_offline_pass = isinstance(details, OfflineEvidenceDetails) and check_id in {
        "no_indexing",
        "ruff",
        "mypy",
        "alembic_roundtrip",
        "openapi_contract",
        "secret_scan",
    }
    if (
        passed
        and not strict_offline_pass
        and check_id
        not in {
            "authorized_dataset",
            "real_media_chain",
            "chapter_vlm_live",
            "five_language_models",
            "m1_durability",
        }
    ):
        raise ValueError(f"{check_id} 尚未交付严格 raw verifier，不能形成 PASS")
    return (GateStatus.PASS if passed else GateStatus.FAIL), None


def _verify_offline_evidence(
    check_id: str,
    details: OfflineEvidenceDetails,
    report: MachineEvidenceReport,
    artifacts: dict[ArtifactRole, tuple[VerifiedArtifact, ...]],
    workspace_root: Path,
) -> bool:
    if check_id not in {
        "no_indexing",
        "ruff",
        "mypy",
        "alembic_roundtrip",
        "openapi_contract",
        "secret_scan",
    }:
        raise ValueError("检查不属于严格离线门禁")
    raw_artifact = _require_unique_artifact(
        artifacts,
        "AUDIT_REPORT",
        details.raw_report_sha256,
        "离线原始报告",
    )
    if raw_artifact.snapshot.content is None:
        raise ValueError("离线原始报告缺少正文")
    raw = OfflineRawReport.model_validate_json(raw_artifact.snapshot.content)
    if (
        raw.check_id != check_id
        or raw.status != report.status
        or raw.input_sha256 != details.input_sha256
        or raw.observation_sha256 != details.observation_sha256
        or raw.command != details.trace.command
        or raw.exit_code != details.trace.exit_code
        or raw.stdout_sha256 != details.trace.stdout_sha256
        or raw.stderr_sha256 != details.trace.stderr_sha256
        or raw.command != offline_command(raw.check_id)
        or raw.audited_paths != offline_audited_paths(raw.check_id)
    ):
        raise ValueError("离线 raw、detail 与机器报告绑定不一致")
    if _current_offline_input_sha256(workspace_root, raw.audited_paths) != raw.input_sha256:
        raise ValueError("离线检查输入不是当前工作区内容")
    observations = _recompute_offline_observations(raw.check_id, workspace_root)
    if (
        observations != raw.observations
        or offline_observation_sha256(observations) != raw.observation_sha256
    ):
        raise ValueError("离线检查观察结果无法由当前输入重算")
    passed = raw.exit_code == 0 and raw.violation_count == 0
    if passed != (not observations):
        raise ValueError("离线检查退出状态与观察结果不一致")
    return passed


def _current_offline_input_sha256(
    workspace_root: Path,
    audited_paths: tuple[str, ...],
) -> str:
    digest = hashlib.sha256()
    workspace = workspace_root.resolve(strict=True)
    for relative in audited_paths:
        candidate = workspace / relative
        paths = _offline_input_files(candidate)
        for path in paths:
            if path.is_symlink():
                raise ValueError("离线审计输入不得包含符号链接")
            resolved = path.resolve(strict=True)
            if not resolved.is_relative_to(workspace):
                raise ValueError("离线审计输入逃逸工作区")
            relative_path = resolved.relative_to(workspace).as_posix()
            digest.update(relative_path.encode("utf-8"))
            digest.update(b"\0")
            with resolved.open("rb") as stream:
                while chunk := stream.read(_CHUNK_BYTES):
                    digest.update(chunk)
            digest.update(b"\0")
    return digest.hexdigest()


def _offline_input_files(candidate: Path) -> tuple[Path, ...]:
    if candidate.is_dir():
        return tuple(
            sorted(
                path
                for path in candidate.rglob("*")
                if path.is_file()
                and not path.is_symlink()
                and "__pycache__" not in path.parts
                and path.suffix in {".py", ".json", ".toml", ".ini", ".yaml", ".yml"}
            )
        )
    return (candidate,) if candidate.is_file() else ()


def _recompute_offline_observations(
    check_id: str,
    workspace_root: Path,
) -> tuple[str, ...]:
    if check_id == "no_indexing":
        from video_demo.evaluation.no_indexing import audit_no_indexing_capability

        return tuple(
            f"{item.rule}:{item.relative_path}:{item.line}:{item.detail}"
            for item in audit_no_indexing_capability(workspace_root)
        )
    if check_id in {"ruff", "mypy"}:
        completed = subprocess.run(
            (sys.executable, *offline_command(cast(Any, check_id))[1:]),
            cwd=workspace_root,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=600,
        )
        return () if completed.returncode == 0 else (f"COMMAND_EXIT_{completed.returncode}",)
    if check_id == "alembic_roundtrip":
        return _alembic_roundtrip_observations(workspace_root)
    if check_id == "openapi_contract":
        return _openapi_contract_observations(workspace_root)
    if check_id == "secret_scan":
        return _secret_scan_observations(workspace_root)
    return ()


def _alembic_roundtrip_observations(workspace_root: Path) -> tuple[str, ...]:
    import contextlib
    import io
    import tempfile

    from alembic import command
    from alembic.config import Config
    from sqlalchemy import create_engine, inspect

    expected = {
        "job",
        "video_asset",
        "video_object",
        "video_segment",
        "video_summary",
        "video_understanding_run",
    }
    temporary_root = workspace_root / ".codex" / "tmp"
    temporary_root.mkdir(parents=True, exist_ok=True)
    observations: list[str] = []
    with tempfile.TemporaryDirectory(prefix="alembic-verify-", dir=temporary_root) as temporary:
        database = Path(temporary) / "migration.db"
        database_url = f"sqlite+pysqlite:///{database}"
        config = Config(
            str(workspace_root / "alembic.ini"),
            attributes={"configure_logging": False},
        )
        config.set_main_option("script_location", str(workspace_root / "migrations"))
        config.set_main_option("sqlalchemy.url", database_url)
        engine = None
        try:
            with (
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                command.upgrade(config, "head")
                engine = create_engine(database_url)
                if not expected <= set(inspect(engine).get_table_names()):
                    observations.append("UPGRADE_SCHEMA_INCOMPLETE")
                command.downgrade(config, "base")
                if expected & set(inspect(engine).get_table_names()):
                    observations.append("DOWNGRADE_SCHEMA_REMAINS")
                command.upgrade(config, "head")
                if not expected <= set(inspect(engine).get_table_names()):
                    observations.append("SECOND_UPGRADE_SCHEMA_INCOMPLETE")
        except Exception:
            observations.append("ALEMBIC_ROUNDTRIP_FAILED")
        finally:
            if engine is not None:
                engine.dispose()
    return tuple(observations)


def _openapi_contract_observations(workspace_root: Path) -> tuple[str, ...]:
    import tempfile

    from video_demo.api.app import create_app

    expected_paths = {
        "/api/kb/jobs/{job_id}",
        "/api/kb/jobs/{job_id}/cancel",
        "/api/kb/jobs/{job_id}/retry",
        "/api/kb/knowledge-bases/{kb_id}/video-objects",
        "/api/kb/knowledge-bases/{kb_id}/video-objects/{object_ref}",
        "/api/kb/knowledge-bases/{kb_id}/video-understanding-runs",
        "/api/kb/knowledge-bases/{kb_id}/video-understanding-runs/{run_id}",
        "/api/kb/knowledge-bases/{kb_id}/video-understanding-runs/{run_id}/evidence",
        "/api/kb/knowledge-bases/{kb_id}/video-understanding-runs/{run_id}/keyframes/"
        "{keyframe_id}/content",
        "/api/kb/knowledge-bases/{kb_id}/video-understanding-runs/{run_id}/result",
    }
    observations: list[str] = []
    temporary_root = workspace_root / ".codex" / "tmp"
    temporary_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="openapi-", dir=temporary_root) as temporary:
        settings = Settings(
            workspace_root=workspace_root,
            runtime_root=Path(temporary).relative_to(workspace_root),
        )
        schema = create_app(settings).openapi()
    paths = set(schema.get("paths", {}))
    if paths != expected_paths:
        observations.append("OPENAPI_PATH_SET_MISMATCH")
    components = schema.get("components", {}).get("schemas", {})
    for name in ("VideoUnderstandingResult", "EvidencePageResponse"):
        component = components.get(name)
        if not isinstance(component, dict) or component.get("additionalProperties") is not False:
            observations.append(f"OPENAPI_SCHEMA_NOT_CLOSED:{name}")
    encoded = json.dumps(schema, ensure_ascii=False, sort_keys=True).casefold()
    for forbidden in ("relative_path", "local_path", "api_key", "secret_key", "password"):
        if forbidden in encoded:
            observations.append(f"OPENAPI_FORBIDDEN_FIELD:{forbidden}")
    return tuple(observations)


def _secret_scan_observations(workspace_root: Path) -> tuple[str, ...]:
    import ast
    import re

    sensitive = re.compile(
        r"(?i)(?:^|_)(?:api_?key|secret_?key|token|password|authorization)(?:$|_)"
    )
    private_key = re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----")
    configured_secret = re.compile(
        r"(?i)^\s*([A-Za-z0-9_.-]*(?:api[_-]?key|secret[_-]?key|token|password|authorization)"
        r"[A-Za-z0-9_.-]*)\s*[:=]\s*([^#\s].*)$"
    )
    observations: list[str] = []
    files = {
        path
        for relative in offline_audited_paths("secret_scan")
        for path in _offline_input_files(workspace_root / relative)
    }
    for path in sorted(files):
        relative = path.relative_to(workspace_root).as_posix()
        text = path.read_text(encoding="utf-8")
        for line, value in enumerate(text.splitlines(), start=1):
            if private_key.search(value):
                observations.append(f"PRIVATE_KEY_LITERAL:{relative}:{line}")
            if path.suffix != ".py":
                match = configured_secret.fullmatch(value)
                if match is not None and match.group(2).strip().strip("\"'") not in {
                    "",
                    "None",
                    "null",
                }:
                    observations.append(f"SECRET_CONFIG_VALUE:{relative}:{line}")
        if path.suffix != ".py":
            continue
        tree = ast.parse(text, filename=relative)
        for node in ast.walk(tree):
            field_name, assigned_value = _sensitive_assignment(node)
            if (
                field_name is not None
                and sensitive.search(field_name)
                and _nonempty_secret_literal(field_name, assigned_value)
            ):
                line_number = getattr(node, "lineno", 0)
                observations.append(f"SECRET_LITERAL:{relative}:{line_number}")
    return tuple(sorted(set(observations)))


def _sensitive_assignment(node: object) -> tuple[str | None, object | None]:
    import ast

    if isinstance(node, ast.Assign) and len(node.targets) == 1:
        target = node.targets[0]
        return (target.id if isinstance(target, ast.Name) else None), node.value
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return node.target.id, node.value
    if isinstance(node, ast.keyword):
        return node.arg, node.value
    return None, None


def _nonempty_secret_literal(name: str, value: object | None) -> bool:
    import ast

    if not isinstance(value, ast.Constant):
        return False
    if not isinstance(value.value, str):
        return False
    literal = value.value.strip()
    return bool(literal) and literal != name and literal != name.upper()


def _verify_chapter_vlm_live(
    details: ChapterVlmLiveDetails,
    report: MachineEvidenceReport,
    artifacts: dict[ArtifactRole, tuple[VerifiedArtifact, ...]],
    workspace_root: Path,
    *,
    settings: Settings | None,
) -> bool:
    if settings is None:
        raise ValueError("章节 VLM live 必须提供可信 settings")
    settings_report = _require_live_settings(settings, workspace_root, check_id="chapter_vlm_live")
    _verify_live_report_shape(details, report, artifacts)
    raw_artifact = _require_unique_artifact(
        artifacts, "AUDIT_REPORT", details.raw_report_sha256, "章节 VLM 原始报告"
    )
    raw = ChapterVlmLiveRawReport.model_validate_json(_artifact_content(raw_artifact))
    if (
        raw.check_id != report.check_id
        or raw.status != report.status
        or raw.evaluation_run_id != raw_artifact.path.parent.name
        or raw.implementation_sha256 != details.implementation_sha256
        or raw.settings_fingerprint != details.settings_fingerprint
        or raw.parent_evaluation_run_id != details.parent_evaluation_run_id
        or raw.sample_id != details.sample_id
    ):
        raise ValueError("章节 VLM raw、detail 与 machine report 绑定不一致")
    if raw.implementation_sha256 != _current_live_implementation_sha256(workspace_root):
        raise ValueError("章节 VLM 实现摘要不是当前实现")
    _verify_chapter_vlm_inputs(raw, details, artifacts, workspace_root, settings=settings)
    if not isinstance(raw, ChapterVlmLiveRawReport):
        raise ValueError("章节 VLM raw 类型不匹配")
    if raw.model != details.model or raw.model.component != "chapter_vlm":
        raise ValueError("章节 VLM 模型身份与 detail 不一致")
    if raw.settings_fingerprint != settings_report.settings_fingerprint:
        raise ValueError("章节 VLM settings fingerprint 不是当前生产配置")
    canonical = {
        (model.component, model.provider, model.model_id, model.device, model.revision)
        for model in settings_report.models
    }
    identity = (
        raw.model.component,
        raw.model.provider,
        raw.model.model_id,
        raw.model.device,
        raw.model.revision,
    )
    if identity not in canonical:
        raise ValueError("章节 VLM 执行模型身份不是当前生产组合")
    if details.status != raw.status or details.operation != raw.operation:
        raise ValueError("章节 VLM detail 的状态或操作与 raw 不一致")
    if details.manifest_sha256 != (
        raw.frame_manifest_input.sha256 if raw.frame_manifest_input is not None else None
    ):
        raise ValueError("章节 VLM Manifest 摘要与 detail 不一致")
    if details.response_sha256 != raw.response_sha256:
        raise ValueError("章节 VLM 响应摘要与 detail 不一致")
    if raw.status == GateStatus.PASS:
        if len(raw.chapter_frames) not in {2, 3, 4} or raw.call_receipt is None:
            raise ValueError("章节 VLM PASS 必须绑定 2~4 张帧和调用回执")
        if _field(raw.call_receipt, "logical_analysis_count") != 1:
            raise ValueError("章节 VLM 必须恰好执行一次逻辑调用")
        if raw.response_sha256 is None or raw.visual_text_score_fact is None:
            raise ValueError("章节 VLM PASS 缺少响应或评分事实")
    score_artifacts = artifacts.get("QUALITY_DETAIL", ())
    if raw.visual_text_score_fact is None:
        if details.visual_text_score_fact_sha256 is not None or score_artifacts:
            raise ValueError("章节 VLM 无评分事实时不得绑定 QUALITY_DETAIL")
    else:
        if details.visual_text_score_fact_sha256 is None:
            raise ValueError("章节 VLM 评分事实缺少 detail 摘要")
        score_artifact = _require_unique_artifact(
            artifacts,
            "QUALITY_DETAIL",
            details.visual_text_score_fact_sha256,
            "章节 VLM 评分事实",
        )
        encoded = score_artifact.snapshot.content
        if encoded is None:
            raise ValueError("章节 VLM 评分事实缺少正文")
        score = VisualTextScoreFact.model_validate_json(encoded)
        if score != raw.visual_text_score_fact:
            raise ValueError("章节 VLM 评分事实与 raw 不一致")
        if raw.response_sha256 != score.response_sha256:
            raise ValueError("章节 VLM 评分事实响应摘要不一致")
    return raw.status == GateStatus.PASS


def _field(value: object, name: str) -> object | None:
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def _verify_chapter_vlm_inputs(
    raw: ChapterVlmLiveRawReport,
    details: ChapterVlmLiveDetails,
    artifacts: dict[ArtifactRole, tuple[VerifiedArtifact, ...]],
    workspace_root: Path,
    *,
    settings: Settings,
) -> None:
    """验证章节 VLM 的源媒体、Manifest 和 2~4 张 JPEG 闭包。"""

    package = _load_live_package(
        _require_unique_artifact(
            artifacts, "DATASET_MANIFEST", details.dataset_sha256, "章节 VLM 授权 Manifest"
        ),
        _require_unique_artifact(
            artifacts,
            "AUTHORIZATION_RECORD",
            details.authorization_sha256,
            "章节 VLM 授权记录",
        ),
        workspace_root,
    )
    if package.dataset_sha256 != details.dataset_sha256:
        raise ValueError("章节 VLM 数据集摘要与当前授权包不一致")
    if package.authorization_sha256 != details.authorization_sha256:
        raise ValueError("章节 VLM 授权摘要与当前授权包不一致")
    sample = next(
        (item for item in package.dataset.samples if item.sample_id == raw.sample_id),
        None,
    )
    verified_annotation = next(
        (item for item in package.annotations if item.annotation.sample_id == raw.sample_id),
        None,
    )
    if sample is None or verified_annotation is None:
        raise ValueError("章节 VLM 样本或标注不在当前授权包")
    if raw.annotation_sha256 != verified_annotation.sha256:
        raise ValueError("章节 VLM 标注摘要与当前授权标注不一致")
    if raw.source_media_input.source_media_sha256 != sample.media_sha256:
        raise ValueError("章节 VLM 源媒体摘要与当前授权样本不一致")
    source_path = package.dataset.eval_root / sample.media_relative_path
    source_relative = source_path.resolve(strict=True).relative_to(
        workspace_root.resolve(strict=True)
    )
    if raw.source_media_input.relative_path != source_relative.as_posix():
        raise ValueError("章节 VLM 源媒体路径与当前授权样本不一致")
    source_snapshot = _read_file_snapshot(
        source_path,
        max_bytes=settings.max_video_bytes,
        capture_content=False,
    )
    if (
        source_snapshot.sha256 != sample.media_sha256
        or source_snapshot.sha256 != raw.source_media_input.sha256
        or source_snapshot.identity.size != raw.source_media_input.size_bytes
    ):
        raise ValueError("章节 VLM 源媒体内容与授权声明不一致")
    _require_input_artifact(artifacts, raw.source_media_input)

    if raw.source_media_input.kind != "SOURCE_MEDIA":
        raise ValueError("章节 VLM 必须绑定 SOURCE_MEDIA")
    if raw.source_media_input.sample_id != raw.sample_id:
        raise ValueError("章节 VLM 源媒体样本不一致")
    if raw.frame_manifest_input is None or raw.frame_manifest_input.kind != "FRAME_MANIFEST":
        raise ValueError("章节 VLM 必须绑定 FRAME_MANIFEST")
    manifest_artifact = _require_input_artifact(artifacts, raw.frame_manifest_input)
    manifest = _load_chapter_vlm_manifest(manifest_artifact)
    manifest_sha256 = chapter_vlm_input_manifest_sha256(manifest)
    if manifest_sha256 != raw.frame_manifest_input.sha256:
        raise ValueError("章节 VLM Manifest 摘要与 raw 不一致")
    if (
        manifest.parent_evaluation_run_id != raw.parent_evaluation_run_id
        or manifest.evaluation_run_id != raw.evaluation_run_id
        or manifest.sample_id != raw.sample_id
        or manifest.source_media_sha256 != sample.media_sha256
        or manifest.annotation_sha256 != raw.annotation_sha256
    ):
        raise ValueError("章节 VLM Manifest 与运行、样本或授权标注不一致")
    run_root = manifest_artifact.path.parent.parent.resolve(strict=True)
    if run_root.name != raw.evaluation_run_id:
        raise ValueError("章节 VLM Manifest 不在当前评测 Run")
    proxy_path = run_root / manifest.proxy_relative_path
    proxy_snapshot = _read_file_snapshot(
        proxy_path,
        max_bytes=settings.max_video_bytes,
        capture_content=False,
    )
    if (
        proxy_snapshot.sha256 != manifest.proxy_sha256
        or proxy_snapshot.identity.size != manifest.proxy_size_bytes
    ):
        raise ValueError("章节 VLM 代理内容与 Manifest 不一致")
    ffprobe = FFprobeClient.from_path(
        production_tool_path(settings, "ffprobe"),
        workspace_root=workspace_root,
    )
    source_probe = ffprobe.probe(
        source_path,
        object_ref=sample.sample_id,
        source_sha256=source_snapshot.sha256,
        source_size_bytes=source_snapshot.identity.size,
        source_mime=_chapter_source_mime(source_path),
        limits=ProbeLimits(max_duration_ms=7_200_000),
    )
    proxy_probe = ffprobe.probe(
        proxy_path,
        object_ref=sample.sample_id,
        source_sha256=proxy_snapshot.sha256,
        source_size_bytes=proxy_snapshot.identity.size,
        source_mime="video/mp4",
        limits=ProbeLimits(max_duration_ms=7_200_000),
    )
    context = ValidatedChapterVlmInputContext(
        parent_evaluation_run_id=manifest.parent_evaluation_run_id,
        evaluation_run_id=manifest.evaluation_run_id,
        sample_id=manifest.sample_id,
        source_media_sha256=manifest.source_media_sha256,
        annotation_sha256=manifest.annotation_sha256,
        source_duration_ms=source_probe.manifest.duration_ms,
        source_display_width=_source_display_size(source_probe.manifest)[0],
        source_display_height=_source_display_size(source_probe.manifest)[1],
        allowed_run_root=run_root,
        proxy_relative_path=manifest.proxy_relative_path,
        proxy_sha256=proxy_snapshot.sha256,
        proxy_size_bytes=proxy_snapshot.identity.size,
        proxy_max_edge=manifest.proxy_max_edge,
        proxy_width=proxy_probe.manifest.video_stream.width,
        proxy_height=proxy_probe.manifest.video_stream.height,
        proxy_frame_rate=proxy_probe.manifest.video_stream.average_frame_rate,
        proxy_is_variable_frame_rate=proxy_probe.manifest.video_stream.is_variable_frame_rate,
        proxy_duration_ms=proxy_probe.manifest.duration_ms,
        duration_tolerance_ms=manifest.duration_tolerance_ms,
        jpeg_quality=manifest.jpeg_quality,
        vlm_max_image_bytes=settings.vlm_max_image_bytes,
        max_candidate_frame_bytes_per_run=settings.max_candidate_frame_bytes_per_run,
        max_candidate_frame_files_per_run=settings.max_candidate_frame_files_per_run,
    )
    validate_chapter_vlm_input_manifest(manifest, context=context)
    manifest_frame_inputs = tuple(
        LiveInputArtifact(
            kind="CHAPTER_FRAME",
            sample_id=manifest.sample_id,
            relative_path=(run_root / frame.relative_path)
            .relative_to(workspace_root)
            .as_posix(),
            sha256=frame.sha256,
            source_media_sha256=manifest.source_media_sha256,
            size_bytes=frame.size_bytes,
        )
        for frame in manifest.frames
    )
    if raw.chapter_frames != manifest_frame_inputs:
        raise ValueError("章节 VLM raw 帧闭包与 Manifest 不一致")
    if len(raw.chapter_frames) not in {2, 3, 4}:
        raise ValueError("章节 VLM 必须绑定 2~4 张 CHAPTER_FRAME")
    if any(
        frame.kind != "CHAPTER_FRAME" or frame.sample_id != raw.sample_id
        for frame in raw.chapter_frames
    ):
        raise ValueError("章节 VLM 帧类型或样本绑定非法")
    if len({frame.relative_path for frame in raw.chapter_frames}) != len(raw.chapter_frames):
        raise ValueError("章节 VLM 帧路径不得重复")
    if len({frame.sha256 for frame in raw.chapter_frames}) != len(raw.chapter_frames):
        raise ValueError("章节 VLM 帧摘要不得重复")
    expected = (raw.source_media_input, raw.frame_manifest_input, *raw.chapter_frames)
    for item in expected:
        _require_input_artifact(artifacts, item)
        if Path(item.relative_path).is_absolute() or ".." in Path(item.relative_path).parts:
            raise ValueError("章节 VLM 输入路径非法")
    if details.manifest_sha256 != raw.frame_manifest_input.sha256:
        raise ValueError("章节 VLM Manifest 摘要与 detail 不一致")
    if raw.call_receipt is not None:
        if raw.call_receipt.ordered_input_frame_ids != tuple(
            frame.frame_id for frame in manifest.frames
        ):
            raise ValueError("章节 VLM 回执输入帧顺序与 Manifest 不一致")
        if not 200 <= raw.call_receipt.provider.final_http_status < 300:
            raise ValueError("章节 VLM 回执最终 HTTP 状态必须是 2xx")


def _require_input_artifact(
    artifacts: dict[ArtifactRole, tuple[VerifiedArtifact, ...]],
    item: LiveInputArtifact,
) -> VerifiedArtifact:
    matches = tuple(
        artifact
        for artifact in artifacts.get("INPUT_MEDIA", ())
        if artifact.reference.relative_path == item.relative_path
        and artifact.reference.sha256 == item.sha256
    )
    if len(matches) != 1 or matches[0].snapshot.identity.size != item.size_bytes:
        raise ValueError("章节 VLM 输入产物未与 raw 精确绑定")
    return matches[0]


def _load_chapter_vlm_manifest(artifact: VerifiedArtifact) -> ChapterVlmInputManifest:
    encoded = artifact.snapshot.content
    if encoded is None:
        raise ValueError("章节 VLM Manifest 缺少正文")
    try:
        manifest = ChapterVlmInputManifest.model_validate_json(encoded)
    except (TypeError, ValueError) as error:
        raise ValueError("章节 VLM Manifest 非法") from error
    canonical_sha256 = chapter_vlm_input_manifest_sha256(manifest)
    if (
        hashlib.sha256(encoded).hexdigest() != artifact.reference.sha256
        or canonical_sha256 != artifact.reference.sha256
    ):
        raise ValueError("章节 VLM Manifest 摘要与规范正文不一致")
    return manifest


def _chapter_source_mime(path: Path) -> SupportedMime:
    mapping: dict[str, SupportedMime] = {
        ".mp4": "video/mp4",
        ".mov": "video/quicktime",
        ".mkv": "video/x-matroska",
        ".webm": "video/webm",
    }
    try:
        return mapping[path.suffix.casefold()]
    except KeyError:
        raise ValueError("章节 VLM 源媒体格式不受支持") from None


def _verify_five_language_models(
    details: FiveLanguageModelsDetails,
    report: MachineEvidenceReport,
    artifacts: dict[ArtifactRole, tuple[VerifiedArtifact, ...]],
    workspace_root: Path,
    *,
    settings: Settings | None,
) -> bool:
    settings_report = _require_live_settings(
        settings,
        workspace_root,
        check_id="five_language_models",
    )
    raw, _package = _verify_live_common(
        details,
        report,
        artifacts,
        workspace_root,
        raw_type=FiveLanguageModelsRawReport,
    )
    reports = [settings_report]
    matched = next(
        (report for report in reports if raw.settings_fingerprint == report.settings_fingerprint),
        None,
    )
    if matched is None:
        raise ValueError("live settings fingerprint 不是当前生产配置")
    _verify_canonical_live_models(raw, matched)
    return raw.status == GateStatus.PASS


def _require_live_settings(
    settings: Settings | None,
    workspace_root: Path,
    *,
    check_id: str | None = None,
) -> Any:
    if settings is None:
        raise ValueError("已执行 live 门禁必须提供可信 settings")
    workspace = workspace_root.resolve(strict=True)
    expected_runtime = workspace / ".codex" / "video-rag-demo"
    if settings.workspace_root != workspace:
        raise ValueError("settings.workspace_root 必须精确匹配 verifier 工作区")
    if settings.runtime_root != expected_runtime:
        raise ValueError("live verifier 只接受工作区固定 runtime root")
    assert settings is not None
    return build_production_model_identity_report(settings)


def _verify_settings_fingerprint(
    raw: _LiveRawReport,
    report: Any,
) -> None:
    if raw.settings_fingerprint != report.settings_fingerprint:
        raise ValueError("live settings fingerprint 不是当前生产配置")


def _verify_canonical_live_models(
    raw: _LiveRawReport,
    report: Any,
) -> None:
    canonical_components = {model.component for model in report.models}
    required_components: set[str]
    if isinstance(raw, ChapterVlmLiveRawReport):
        required_components = {"chapter_vlm"}
    elif isinstance(raw, FiveLanguageModelsRawReport):
        required_components = {"silero_vad", "cloud_whisper"}
    else:
        raise ValueError("未知 live raw 类型")
    if not required_components.issubset(canonical_components):
        raise ValueError("当前生产组合缺少 live 检查所需模型身份")
    canonical = {
        (model.component, model.provider, model.model_id, model.device, model.revision)
        for model in report.models
    }
    for fact in raw.executions:
        identity = (
            fact.model.component,
            fact.model.provider,
            fact.model.model_id,
            fact.model.device,
            fact.model.revision,
        )
        if identity not in canonical:
            raise ValueError("live 执行事实模型身份不是当前生产组合")


def _verify_live_common(
    details: _LiveCheckDetails,
    report: MachineEvidenceReport,
    artifacts: dict[ArtifactRole, tuple[VerifiedArtifact, ...]],
    workspace_root: Path,
    *,
    raw_type: type[_LiveRawReport],
) -> tuple[_LiveRawReport, ValidatedEvaluationPackage]:
    _verify_live_report_shape(details, report, artifacts)
    raw_artifact = _require_unique_artifact(
        artifacts,
        "AUDIT_REPORT",
        details.raw_report_sha256,
        "live 原始报告",
    )
    raw = raw_type.model_validate_json(_artifact_content(raw_artifact))
    _verify_live_raw_binding(
        raw,
        details,
        report,
        raw_artifact,
        artifacts,
        workspace_root,
    )
    if raw.dataset_sha256 is None or raw.authorization_sha256 is None:
        raise ValueError("旧 live raw 缺少授权摘要")
    manifest = _require_unique_artifact(
        artifacts,
        "DATASET_MANIFEST",
        raw.dataset_sha256,
        "授权 Manifest",
    )
    authorization = _require_unique_artifact(
        artifacts,
        "AUTHORIZATION_RECORD",
        raw.authorization_sha256,
        "授权记录",
    )
    source_snapshots = _snapshot_authorized_sources(
        manifest,
        authorization,
        workspace_root / ".codex" / "video-rag-demo",
        _MAX_LIVE_MEDIA_BYTES,
    )
    package = _load_live_package(manifest, authorization, workspace_root)
    samples = _live_samples(raw)
    _verify_live_samples(
        samples,
        raw.inputs,
        package,
        artifacts,
        raw.evaluation_run_id,
        workspace_root,
    )
    _verify_live_execution_summaries(raw.executions, artifacts, raw_artifact.path.parent)
    _verify_live_execution_sequence(raw)
    for snapshot in source_snapshots:
        _assert_snapshot_current(snapshot)
    return raw, package


def _verify_live_report_shape(
    details: _LiveCheckDetails,
    report: MachineEvidenceReport,
    artifacts: dict[ArtifactRole, tuple[VerifiedArtifact, ...]],
) -> None:
    allowed = {
        "AUDIT_REPORT",
        "COMMAND_STDOUT",
        "COMMAND_STDERR",
        "DATASET_MANIFEST",
        "AUTHORIZATION_RECORD",
        "INPUT_MEDIA",
        "PROVIDER_RESPONSE",
        "QUALITY_DETAIL",
    }
    if set(artifacts) - allowed:
        raise ValueError("live 报告包含未声明产物角色")
    exact_roles: tuple[ArtifactRole, ...] = (
        "AUDIT_REPORT",
        "COMMAND_STDOUT",
        "COMMAND_STDERR",
        "DATASET_MANIFEST",
        "AUTHORIZATION_RECORD",
    )
    if any(len(artifacts.get(role, ())) != 1 for role in exact_roles):
        raise ValueError("live 报告权威与命令产物必须唯一绑定")
    if report.status == GateStatus.NOT_RUN or report.not_run_reason is not None:
        raise ValueError("已执行 live 报告不得是 NOT_RUN")
    if (
        report.kind != EvidenceKind.LIVE_SERVICE_REPORT
        or report.level != EvidenceLevel.REAL_SERVICE
    ):
        raise ValueError("live 报告级别或类型不匹配")
    trace = details.trace
    trace_pairs: tuple[tuple[ArtifactRole, str], ...] = (
        ("COMMAND_STDOUT", trace.stdout_sha256),
        ("COMMAND_STDERR", trace.stderr_sha256),
    )
    for role, digest in trace_pairs:
        if (
            len(
                tuple(
                    artifact
                    for artifact in artifacts.get(role, ())
                    if artifact.reference.sha256 == digest
                )
            )
            != 1
        ):
            raise ValueError("live 顶层 trace 输出必须唯一绑定")


def _verify_live_raw_binding(
    raw: _LiveRawReport,
    details: _LiveCheckDetails,
    report: MachineEvidenceReport,
    raw_artifact: VerifiedArtifact,
    artifacts: dict[ArtifactRole, tuple[VerifiedArtifact, ...]],
    workspace_root: Path,
) -> None:
    if raw.check_id != report.check_id or raw.status != report.status:
        raise ValueError("live raw 与 machine report 检查或状态不一致")
    if raw.evaluation_run_id != raw_artifact.path.parent.name:
        raise ValueError("live raw 运行 ID 与报告目录不一致")
    if (
        raw.implementation_sha256 != details.implementation_sha256
        or raw.settings_fingerprint != details.settings_fingerprint
        or raw.dataset_sha256 != details.dataset_sha256
        or raw.authorization_sha256 != details.authorization_sha256
    ):
        raise ValueError("live raw、detail 摘要未完全绑定")
    if raw.implementation_sha256 != _current_live_implementation_sha256(workspace_root):
        raise ValueError("live 实现摘要不是当前实现")
    expected_exit = 0 if raw.status == GateStatus.PASS else None
    if expected_exit == 0 and details.trace.exit_code != 0:
        raise ValueError("live PASS 顶层命令必须成功")
    if raw.status == GateStatus.FAIL and details.trace.exit_code == 0:
        raise ValueError("live FAIL 顶层命令必须记录失败")
    trace_root = raw_artifact.path.parent
    trace_pairs: tuple[tuple[ArtifactRole, str], ...] = (
        ("COMMAND_STDOUT", details.trace.stdout_sha256),
        ("COMMAND_STDERR", details.trace.stderr_sha256),
    )
    for role, digest in trace_pairs:
        trace_artifact = _require_unique_artifact(
            artifacts,
            role,
            digest,
            "live trace 输出",
        )
        if trace_artifact.path.parent != trace_root:
            raise ValueError("live trace 输出必须位于当前运行目录")


def _load_live_package(
    manifest: VerifiedArtifact,
    authorization: VerifiedArtifact,
    workspace_root: Path,
) -> ValidatedEvaluationPackage:
    runtime_root = workspace_root / ".codex" / "video-rag-demo"
    return load_evaluation_package(
        manifest.path,
        authorization.path,
        workspace_root=workspace_root,
        runtime_root=runtime_root,
    )


def _live_samples(raw: _LiveRawReport) -> tuple[LiveSample, ...]:
    if isinstance(raw, ChapterVlmLiveRawReport):
        return ()
    if isinstance(raw, FiveLanguageModelsRawReport):
        return raw.samples
    raise ValueError("未知 live raw 类型")


def _verify_live_samples(
    samples: tuple[LiveSample, ...],
    raw_inputs: tuple[LiveInputArtifact, ...],
    package: ValidatedEvaluationPackage,
    artifacts: dict[ArtifactRole, tuple[VerifiedArtifact, ...]],
    evaluation_run_id: str,
    workspace_root: Path,
) -> None:
    package_by_id = {
        sample.sample_id: (sample, annotation)
        for sample, annotation in zip(
            package.dataset.samples,
            package.annotations,
            strict=True,
        )
    }
    if set(sample.sample_id for sample in samples) - set(package_by_id):
        raise ValueError("live 样本不属于授权评测包")
    input_artifacts = artifacts.get("INPUT_MEDIA", ())
    for sample in samples:
        dataset_sample, annotation = package_by_id[sample.sample_id]
        expected_source = (
            (package.dataset.eval_root / dataset_sample.media_relative_path)
            .relative_to(workspace_root)
            .as_posix()
        )
        if (
            sample.source_media_relative_path != expected_source
            or sample.source_media_sha256 != dataset_sample.media_sha256
            or sample.language != dataset_sample.language
            or sample.annotation_sha256 != annotation.sha256
        ):
            raise ValueError("live 源媒体或标注未绑定授权评测包")
        _verify_derived_paths(sample, evaluation_run_id)
    if len(input_artifacts) != len(raw_inputs):
        raise ValueError("live 输入产物数量不完整")
    for item in raw_inputs:
        matches = tuple(
            artifact
            for artifact in input_artifacts
            if artifact.reference.relative_path == item.relative_path
            and artifact.reference.sha256 == item.sha256
        )
        if len(matches) != 1:
            raise ValueError("live 输入产物未与 raw 精确绑定")
        if matches[0].snapshot.identity.size != item.size_bytes:
            raise ValueError("live 输入大小与 raw 不一致")
        if item.source_media_sha256 != next(
            sample.source_media_sha256 for sample in samples if sample.sample_id == item.sample_id
        ):
            raise ValueError("live 输入未绑定当前样本源媒体摘要")


def _verify_derived_paths(sample: LiveSample, evaluation_run_id: str) -> None:
    expected_parent = (
        PurePosixPath(".codex/video-rag-demo/eval/live") / evaluation_run_id / sample.sample_id
    )
    expected = {"AUDIO": (sample.audio_relative_path, "audio.mp3")}
    for relative_path, filename in expected.values():
        path = PurePosixPath(relative_path)
        if path.parent != expected_parent or path.name != filename:
            raise ValueError("live 派生产物未绑定当前 run 和样本")


def _verify_live_execution_summaries(
    facts: tuple[ModelExecutionFact, ...],
    artifacts: dict[ArtifactRole, tuple[VerifiedArtifact, ...]],
    run_root: Path,
) -> None:
    responses = artifacts.get("PROVIDER_RESPONSE", ())
    if len(responses) != len(facts):
        raise ValueError("live 执行事实与输出摘要数量不一致")
    used: set[str] = set()
    for fact in facts:
        artifact = _require_unique_artifact(
            artifacts,
            "PROVIDER_RESPONSE",
            fact.output_sha256,
            "live 执行输出摘要",
        )
        if artifact.path.parent != run_root or artifact.reference.sha256 in used:
            raise ValueError("live 输出摘要未绑定当前运行或被复用")
        summary = LiveExecutionSummary.model_validate_json(_artifact_content(artifact))
        if summary.evaluation_run_id != run_root.name:
            raise ValueError("live 输出摘要必须绑定当前 run")
        if not _summary_matches_fact(summary, fact):
            raise ValueError("live 输出摘要与执行事实不一致")
        used.add(artifact.reference.sha256)


def _summary_matches_fact(
    summary: LiveExecutionSummary,
    fact: ModelExecutionFact,
) -> bool:
    return (
        summary.component == fact.component
        and summary.operation == fact.operation
        and summary.evaluation_run_id == fact.evaluation_run_id
        and summary.model == fact.model
        and summary.sample_id == fact.sample_id
        and summary.language == fact.language
        and summary.input_kind == fact.input_kind
        and summary.input_sha256 == fact.input_sha256
        and summary.request_id_sha256 == fact.request_id_sha256
        and summary.http_status == fact.http_status
        and summary.capabilities == fact.capabilities
    )


def _verify_live_execution_sequence(raw: _LiveRawReport) -> None:
    stages = _live_execution_stages(raw)
    for index, fact in enumerate(raw.executions):
        if index >= len(stages) or not _fact_matches_stage(fact, stages[index]):
            raise ValueError("live raw 只允许合法执行前缀")
        if fact.component in {"chapter_vlm", "cloud_whisper"} and (
            fact.http_status is None or not 200 <= fact.http_status < 300
        ):
            raise ValueError("远程 live 阶段必须以 2xx 成功事实表示")
    if raw.status == GateStatus.PASS:
        if len(raw.executions) != len(stages):
            raise ValueError("live PASS 执行阶段不完整")
        return
    if raw.failure_component == "components_close":
        if len(raw.executions) != len(stages) or raw.failure_code != ErrorCode.SYSTEM_FAILURE:
            raise ValueError("live 组件关闭失败必须位于完整执行序列之后")
        return
    if len(raw.executions) >= len(stages):
        raise ValueError("live FAIL 只允许合法执行前缀")
    if raw.failure_component != stages[len(raw.executions)][0]:
        raise ValueError("live 失败组件与下一执行阶段不一致")
    if (
        raw.failure_component is None
        or raw.failure_code is None
        or raw.failure_code not in _LIVE_COMPONENT_FAILURE_CODES[raw.failure_component]
    ):
        raise ValueError("live 失败错误码与实际组件不一致")


def _live_execution_stages(
    raw: _LiveRawReport,
) -> tuple[tuple[str, str, str | None], ...]:
    if isinstance(raw, FiveLanguageModelsRawReport):
        languages = ("zh", "en", "ja", "ko", "es")
        return (
            ("silero_vad", "vad", None),
            *(("cloud_whisper", "transcribe", language) for language in languages),
        )
    raise ValueError("未知 live raw 类型")


def _fact_matches_stage(
    fact: ModelExecutionFact,
    stage: tuple[str, str, str | None],
) -> bool:
    component, operation, language = stage
    return (
        fact.component == component
        and fact.operation == operation
        and (language is None or fact.language == language)
    )


def _artifact_content(artifact: VerifiedArtifact) -> bytes:
    if artifact.snapshot.content is None:
        raise ValueError("live 结构摘要缺少内容")
    return artifact.snapshot.content


def _verify_real_media(
    details: RealMediaDetails,
    artifacts: dict[ArtifactRole, tuple[VerifiedArtifact, ...]],
    workspace_root: Path,
) -> bool:
    if len(artifacts.get("AUDIT_REPORT", ())) != 1:
        raise ValueError("真实媒体原始报告必须唯一绑定")
    raw_artifact = _require_unique_artifact(
        artifacts, "AUDIT_REPORT", details.raw_report_sha256, "真实媒体原始报告"
    )
    encoded = raw_artifact.snapshot.content
    if encoded is None:
        raise ValueError("真实媒体原始报告缺少快照")
    raw = RealMediaRawReport.model_validate_json(encoded)
    if raw_artifact.path.parent.name != raw.evaluation_run_id:
        raise ValueError("真实媒体运行 ID 与报告目录不一致")
    if (
        raw.ffmpeg_version != details.ffmpeg_version
        or raw.ffprobe_version != details.ffprobe_version
        or raw.implementation_sha256 != details.implementation_sha256
    ):
        raise ValueError("真实媒体 detail 与原始报告不一致")
    if raw.trace_exit_code != details.trace.exit_code:
        raise ValueError("真实媒体顶层 trace 与原始报告不一致")
    if raw.implementation_sha256 != _current_real_media_implementation_sha256(workspace_root):
        raise ValueError("真实媒体实现摘要不是当前实现")
    if raw.status == GateStatus.FAIL and all(
        sample.execution_status == "NOT_STARTED" for sample in raw.samples
    ):
        _verify_real_media_artifact_roles(artifacts, allow_media=False)
        _verify_real_media_setup_outputs(raw, details, raw_artifact, artifacts)
        return False
    _verify_real_media_artifact_roles(artifacts, allow_media=True)
    _verify_real_media_setup_outputs(raw, details, raw_artifact, artifacts)
    _verify_real_media_files(raw, artifacts, workspace_root)
    _verify_real_media_command_outputs(
        raw, details, raw_artifact, artifacts, allow_setup_outputs=True
    )
    return raw.status == GateStatus.PASS


def _verify_real_media_setup_outputs(
    raw: RealMediaRawReport,
    details: RealMediaDetails,
    raw_artifact: VerifiedArtifact,
    artifacts: dict[ArtifactRole, tuple[VerifiedArtifact, ...]],
) -> None:
    expected: list[tuple[ArtifactRole, str, str]] = []
    for command in raw.setup_commands:
        expected.extend(
            (
                ("COMMAND_STDOUT", command.stdout_relative_path, command.stdout_sha256),
                ("COMMAND_STDERR", command.stderr_relative_path, command.stderr_sha256),
            )
        )
    if len(expected) != len(set(expected)):
        raise ValueError("版本探测命令输出不得跨命令复用")
    command_roles: tuple[ArtifactRole, ArtifactRole] = (
        "COMMAND_STDOUT",
        "COMMAND_STDERR",
    )
    actual = {
        (role, artifact.reference.relative_path, artifact.reference.sha256)
        for role in command_roles
        for artifact in artifacts.get(role, ())
    }
    trace_expected: list[tuple[ArtifactRole, str, str]] = []
    trace_pairs: tuple[tuple[ArtifactRole, str], ...] = (
        ("COMMAND_STDOUT", details.trace.stdout_sha256),
        ("COMMAND_STDERR", details.trace.stderr_sha256),
    )
    for role, digest in trace_pairs:
        matches = tuple(
            artifact for artifact in artifacts.get(role, ()) if artifact.reference.sha256 == digest
        )
        if len(matches) != 1:
            raise ValueError("真实媒体顶层 trace 输出必须唯一绑定")
        trace_expected.append((role, matches[0].reference.relative_path, digest))
    if set(expected) & set(trace_expected):
        raise ValueError("版本探测命令输出不得与顶层 trace 复用")
    if (
        raw.status == GateStatus.FAIL
        and all(sample.execution_status == "NOT_STARTED" for sample in raw.samples)
        and actual != set(expected) | set(trace_expected)
    ):
        raise ValueError("版本探测命令输出产物必须精确绑定")
    if any(
        not artifact.path.is_relative_to(raw_artifact.path.parent)
        for role in command_roles
        for artifact in artifacts.get(role, ())
    ):
        raise ValueError("版本探测命令输出必须位于当前原始报告目录")


def _current_real_media_implementation_sha256(workspace_root: Path) -> str:
    entries: list[dict[str, str]] = []
    for relative_path in _REAL_MEDIA_IMPLEMENTATION_FILES:
        path = workspace_root / relative_path
        snapshot = _read_file_snapshot(
            path,
            max_bytes=64 * 1024 * 1024,
            capture_content=False,
        )
        entries.append(
            {
                "relative_path": relative_path.as_posix(),
                "sha256": snapshot.sha256,
            }
        )
    encoded = json.dumps(
        entries,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _current_live_implementation_sha256(workspace_root: Path) -> str:
    entries: list[dict[str, str]] = []
    for relative_path in _LIVE_IMPLEMENTATION_FILES:
        path = workspace_root / relative_path
        if not path.is_file():
            if relative_path == Path("src/video_demo/evaluation/live_runner.py"):
                sha256 = "MISSING"
            else:
                raise ValueError("live 当前实现文件缺失")
        else:
            sha256 = _read_file_snapshot(
                path,
                max_bytes=_MAX_EVIDENCE_BYTES,
                capture_content=False,
            ).sha256
        entries.append(
            {
                "relative_path": relative_path.as_posix(),
                "sha256": sha256,
            }
        )
    encoded = json.dumps(
        entries,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _current_durability_implementation_sha256(workspace_root: Path) -> str:
    entries: list[dict[str, str]] = []
    for relative_path in _DURABILITY_IMPLEMENTATION_FILES:
        snapshot = _read_file_snapshot(
            workspace_root / relative_path,
            max_bytes=_MAX_EVIDENCE_BYTES,
            capture_content=False,
        )
        entries.append({"relative_path": relative_path.as_posix(), "sha256": snapshot.sha256})
    encoded = json.dumps(
        entries,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _verify_real_media_files(
    raw: RealMediaRawReport,
    artifacts: dict[ArtifactRole, tuple[VerifiedArtifact, ...]],
    workspace_root: Path,
) -> None:
    expected: dict[tuple[ArtifactRole, str, str], RealMediaFile] = {}
    for sample in raw.samples:
        for media_file in sample.files:
            role: ArtifactRole = "INPUT_MEDIA" if media_file.role == "SOURCE" else "OUTPUT_MEDIA"
            expected_key = (role, media_file.relative_path, media_file.sha256)
            if expected_key in expected:
                raise ValueError("真实媒体原始报告重复声明文件")
            expected[expected_key] = media_file
    observed: dict[tuple[ArtifactRole, str, str], VerifiedArtifact] = {}
    for role in ("INPUT_MEDIA", "OUTPUT_MEDIA"):
        for artifact in artifacts.get(role, ()):
            observed_key = (
                role,
                artifact.reference.relative_path,
                artifact.reference.sha256,
            )
            if observed_key in observed:
                raise ValueError("真实媒体产物重复绑定")
            observed[observed_key] = artifact
    if set(observed) != set(expected):
        raise ValueError("真实媒体产物必须与原始报告精确绑定")
    for expected_key, media_file in expected.items():
        artifact = observed[expected_key]
        if artifact.path != workspace_root / media_file.relative_path:
            raise ValueError("真实媒体产物路径不一致")
        if artifact.snapshot.identity.size != media_file.size_bytes:
            raise ValueError("真实媒体产物大小与原始报告不一致")
        refreshed = _snapshot_real_media_file(
            artifact.path,
            max_bytes=artifact.reference.max_bytes,
        )
        if (
            refreshed.snapshot.sha256 != media_file.sha256
            or refreshed.snapshot.identity.size != media_file.size_bytes
            or refreshed.snapshot.identity != artifact.snapshot.identity
        ):
            raise ValueError("真实媒体产物重验快照不一致")
        _verify_real_media_magic(refreshed, media_file)


@dataclass(frozen=True)
class _RealMediaSnapshot:
    snapshot: FileSnapshot


def _snapshot_real_media_file(path: Path, *, max_bytes: int | None) -> _RealMediaSnapshot:
    descriptor = _open_regular_file(path)
    try:
        before = os.fstat(descriptor)
        if max_bytes is not None and before.st_size > max_bytes:
            raise ValueError("真实媒体产物超过大小上限")
        digest = hashlib.sha256()
        head = bytearray()
        tail = bytearray()
        total = 0
        while chunk := os.read(descriptor, _CHUNK_BYTES):
            total += len(chunk)
            if max_bytes is not None and total > max_bytes:
                raise ValueError("真实媒体产物超过大小上限")
            digest.update(chunk)
            if len(head) < 64:
                head.extend(chunk[: 64 - len(head)])
            tail.extend(chunk)
            if len(tail) > 2:
                del tail[:-2]
        after = os.fstat(descriptor)
        identity = FileIdentity(
            device=before.st_dev,
            inode=before.st_ino,
            size=before.st_size,
            mtime_ns=before.st_mtime_ns,
            ctime_ns=before.st_ctime_ns,
        )
        after_identity = FileIdentity(
            device=after.st_dev,
            inode=after.st_ino,
            size=after.st_size,
            mtime_ns=after.st_mtime_ns,
            ctime_ns=after.st_ctime_ns,
        )
        if after_identity != identity or total != before.st_size:
            raise ValueError("真实媒体产物读取期间发生变化")
        return _RealMediaSnapshot(
            snapshot=FileSnapshot(
                path=path,
                identity=identity,
                sha256=digest.hexdigest(),
                content=bytes(head) + bytes(tail),
            ),
        )
    finally:
        os.close(descriptor)


def _verify_real_media_magic(snapshot: _RealMediaSnapshot, media_file: RealMediaFile) -> None:
    head = snapshot.snapshot.content or b""
    if media_file.format == "MP4":
        valid = _has_iso_bmff_ftyp(head[:32], snapshot.snapshot.identity.size)
    elif media_file.format == "MP3":
        valid = _has_mp3_signature(head, snapshot.snapshot.identity.size)
    else:
        valid = (
            snapshot.snapshot.identity.size >= 5
            and len(head) >= 5
            and head[:3] == b"\xff\xd8\xff"
            and head[-2:] == b"\xff\xd9"
        )
    if not valid:
        raise ValueError("真实媒体文件 magic 不匹配")


def _has_mp3_signature(header: bytes, actual_size: int) -> bool:
    if actual_size <= 0:
        return False
    if header.startswith(b"ID3"):
        return _has_valid_id3_header(header, actual_size)
    return any(
        first == 0xFF and (second & 0xE0) == 0xE0
        for first, second in pairwise(header)
    )


def _has_valid_id3_header(header: bytes, actual_size: int) -> bool:
    if actual_size < 10 or len(header) < 10:
        return False
    major_version = header[3]
    revision = header[4]
    flags = header[5]
    if major_version not in (2, 3, 4) or revision == 0xFF:
        return False
    allowed_flags = 0xE0 if major_version in (2, 3) else 0xF0
    if flags & ~allowed_flags:
        return False
    size_bytes = header[6:10]
    if any(value & 0x80 for value in size_bytes):
        return False
    tag_size = sum(value << (7 * (3 - index)) for index, value in enumerate(size_bytes))
    return 10 + tag_size <= actual_size


def _has_iso_bmff_ftyp(header: bytes, actual_size: int) -> bool:
    if len(header) < 8 or header[4:8] != b"ftyp":
        return False
    size32 = int.from_bytes(header[:4], byteorder="big")
    if size32 == 0:
        return False
    if size32 == 1:
        if len(header) < 24:
            return False
        large_size = int.from_bytes(header[8:16], byteorder="big")
        return large_size >= 24 and (large_size - 24) % 4 == 0 and large_size <= actual_size
    return size32 >= 16 and (size32 - 16) % 4 == 0 and size32 <= actual_size


def _verify_real_media_command_outputs(
    raw: RealMediaRawReport,
    details: RealMediaDetails,
    raw_artifact: VerifiedArtifact,
    artifacts: dict[ArtifactRole, tuple[VerifiedArtifact, ...]],
    *,
    allow_setup_outputs: bool = False,
) -> None:
    required: list[tuple[ArtifactRole, str, str]] = []
    for sample in raw.samples:
        for command in sample.commands:
            required.extend(
                (
                    (
                        "COMMAND_STDOUT",
                        command.stdout_relative_path,
                        command.stdout_sha256,
                    ),
                    (
                        "COMMAND_STDERR",
                        command.stderr_relative_path,
                        command.stderr_sha256,
                    ),
                )
            )
    expected: set[tuple[ArtifactRole, str, str]] = set()
    for role, path, digest in required:
        matches = tuple(
            artifact
            for artifact in artifacts.get(role, ())
            if artifact.reference.relative_path == path and artifact.reference.sha256 == digest
        )
        if len(matches) != 1:
            raise ValueError("真实媒体命令输出必须唯一绑定")
        if not matches[0].path.is_relative_to(raw_artifact.path.parent):
            raise ValueError("真实媒体命令输出必须位于当前原始报告目录")
        identity = (role, path, digest)
        if identity in expected:
            raise ValueError("真实媒体命令输出不得跨命令复用")
        expected.add(identity)
    report_root = raw_artifact.path.parent
    trace_expected: set[tuple[ArtifactRole, str, str]] = set()
    trace_pairs: tuple[tuple[ArtifactRole, str], ...] = (
        ("COMMAND_STDOUT", details.trace.stdout_sha256),
        ("COMMAND_STDERR", details.trace.stderr_sha256),
    )
    for role, digest in trace_pairs:
        matches = tuple(
            artifact for artifact in artifacts.get(role, ()) if artifact.reference.sha256 == digest
        )
        if len(matches) != 1:
            raise ValueError("真实媒体顶层 trace 输出必须唯一绑定")
        artifact = matches[0]
        if not artifact.path.is_relative_to(report_root):
            raise ValueError("真实媒体命令输出必须位于当前原始报告目录")
        trace_expected.add((role, artifact.reference.relative_path, digest))
    command_roles: tuple[ArtifactRole, ArtifactRole] = (
        "COMMAND_STDOUT",
        "COMMAND_STDERR",
    )
    actual = {
        (role, artifact.reference.relative_path, artifact.reference.sha256)
        for role in command_roles
        for artifact in artifacts.get(role, ())
    }
    setup_expected = {
        (role, path, digest)
        for command in raw.setup_commands
        for role, path, digest in (
            ("COMMAND_STDOUT", command.stdout_relative_path, command.stdout_sha256),
            ("COMMAND_STDERR", command.stderr_relative_path, command.stderr_sha256),
        )
    }
    if len(setup_expected) != len(raw.setup_commands) * 2:
        raise ValueError("版本探测命令输出不得跨命令复用")
    if expected & trace_expected:
        raise ValueError("真实媒体命令输出不得与顶层 trace 复用")
    if allow_setup_outputs and (setup_expected & expected or setup_expected & trace_expected):
        raise ValueError("版本探测命令输出不得复用媒体或顶层 trace 输出")
    allowed = expected | trace_expected | (setup_expected if allow_setup_outputs else set())
    if actual != allowed:
        raise ValueError("真实媒体命令输出产物必须精确绑定")


def _require_unique_artifact(
    artifacts: dict[ArtifactRole, tuple[VerifiedArtifact, ...]],
    role: ArtifactRole,
    expected_sha256: str,
    label: str,
) -> VerifiedArtifact:
    matches = tuple(
        artifact
        for artifact in artifacts.get(role, ())
        if artifact.reference.sha256 == expected_sha256
    )
    if len(matches) != 1:
        raise ValueError(f"{label}必须唯一绑定")
    return matches[0]


def _derive_preflight_status(
    check_id: str,
    report: MachineEvidenceReport,
    details: PreflightDetails,
    artifacts: dict[ArtifactRole, tuple[VerifiedArtifact, ...]],
    workspace_root: Path,
) -> tuple[GateStatus, str]:
    raw_values = artifacts.get("AUDIT_REPORT", ())
    if len(raw_values) != 1:
        raise ValueError("preflight 必须绑定唯一原始报告")
    raw_artifact = raw_values[0]
    if raw_artifact.reference.sha256 != details.preflight_report_sha256:
        raise ValueError("preflight 原始报告摘要不匹配")
    encoded = raw_artifact.snapshot.content
    if encoded is None:
        raise ValueError("preflight 原始报告缺少快照")
    raw = PreflightRawReport.model_validate_json(encoded)
    if raw.check_id != check_id:
        raise ValueError("preflight 检查绑定不匹配")
    if check_id == "real_media_chain" and (
        raw.implementation_sha256 != _current_real_media_implementation_sha256(workspace_root)
    ):
        raise ValueError("真实媒体 preflight 实现摘要不是当前实现")
    if check_id == "real_media_chain":
        _verify_real_media_artifact_roles(artifacts, allow_media=False)
        _verify_real_media_preflight_artifacts(raw, report, artifacts, raw_artifact)
    if check_id in _LIVE_IMPLEMENTATION_CHECKS:
        if raw.implementation_sha256 != _current_live_implementation_sha256(workspace_root):
            raise ValueError("live preflight 实现摘要不是当前实现")
        _verify_live_preflight_artifacts(raw, report, artifacts, raw_artifact)
    if check_id == "m1_durability":
        if (
            raw.implementation_sha256 != _current_durability_implementation_sha256(workspace_root)
            or raw.evaluation_run_id is None
            or raw.evaluation_run_id != raw_artifact.path.parent.name
        ):
            raise ValueError("M1 preflight 实现或运行绑定不匹配")
        if set(artifacts) != {"AUDIT_REPORT", "COMMAND_STDOUT", "COMMAND_STDERR"}:
            raise ValueError("M1 preflight 产物角色必须精确闭合")
        expected_root = raw_artifact.path.parent
        if any(
            artifact.path.parent != expected_root
            for values in artifacts.values()
            for artifact in values
        ) or any(len(values) != 1 for values in artifacts.values()):
            raise ValueError("M1 preflight 产物必须位于当前报告目录")
    reason = (
        build_durability_not_run_reason(tuple(issue.code for issue in raw.issues or ()))
        if check_id == "m1_durability"
        else _MISSING_CHECK_REASONS[check_id]
    )
    if report.not_run_reason != reason:
        raise ValueError("preflight 自报原因与固定原因不一致")
    return GateStatus.NOT_RUN, reason


def _verify_real_media_artifact_roles(
    artifacts: dict[ArtifactRole, tuple[VerifiedArtifact, ...]],
    *,
    allow_media: bool,
) -> None:
    allowed: set[ArtifactRole] = {
        "AUDIT_REPORT",
        "COMMAND_STDOUT",
        "COMMAND_STDERR",
    }
    if allow_media:
        allowed.update(("INPUT_MEDIA", "OUTPUT_MEDIA"))
    if set(artifacts) - allowed:
        raise ValueError("真实媒体报告包含未声明产物角色")


def _verify_live_preflight_artifacts(
    raw: PreflightRawReport,
    report: MachineEvidenceReport,
    artifacts: dict[ArtifactRole, tuple[VerifiedArtifact, ...]],
    raw_artifact: VerifiedArtifact,
) -> None:
    if set(artifacts) != {"AUDIT_REPORT", "COMMAND_STDOUT", "COMMAND_STDERR"}:
        raise ValueError("live preflight 只能绑定 raw 与顶层命令输出")
    if raw.evaluation_run_id is None or raw.evaluation_run_id != raw_artifact.path.parent.name:
        raise ValueError("live preflight 运行 ID 与报告目录不一致")
    expected_root = raw_artifact.path.parent
    trace_pairs: tuple[tuple[ArtifactRole, str], ...] = (
        ("COMMAND_STDOUT", report.details.trace.stdout_sha256),
        ("COMMAND_STDERR", report.details.trace.stderr_sha256),
    )
    for role, digest in trace_pairs:
        artifact = _require_unique_artifact(
            artifacts,
            role,
            digest,
            "live preflight 顶层输出",
        )
        if artifact.path.parent != expected_root:
            raise ValueError("live preflight 顶层输出必须位于当前运行目录")
    if any(len(artifacts.get(role, ())) != 1 for role in artifacts):
        raise ValueError("live preflight 产物必须精确闭合")


def _verify_real_media_preflight_artifacts(
    raw: PreflightRawReport,
    report: MachineEvidenceReport,
    artifacts: dict[ArtifactRole, tuple[VerifiedArtifact, ...]],
    raw_artifact: VerifiedArtifact,
) -> None:
    if raw.evaluation_run_id is None or raw_artifact.path.parent.name != raw.evaluation_run_id:
        raise ValueError("真实媒体 preflight 运行 ID 与报告目录不一致")
    expected_root = raw_artifact.path.parent
    trace = report.details.trace
    expected: list[tuple[ArtifactRole, str]] = [
        ("AUDIT_REPORT", raw_artifact.reference.relative_path),
    ]
    trace_pairs: tuple[tuple[ArtifactRole, str], ...] = (
        ("COMMAND_STDOUT", trace.stdout_sha256),
        ("COMMAND_STDERR", trace.stderr_sha256),
    )
    for role, digest in trace_pairs:
        matches = tuple(
            artifact for artifact in artifacts.get(role, ()) if artifact.reference.sha256 == digest
        )
        if len(matches) != 1 or not matches[0].path.is_relative_to(expected_root):
            raise ValueError("真实媒体 preflight 顶层输出必须唯一且位于当前报告目录")
        expected.append((role, matches[0].reference.relative_path))
    roles: tuple[ArtifactRole, ArtifactRole, ArtifactRole] = (
        "AUDIT_REPORT",
        "COMMAND_STDOUT",
        "COMMAND_STDERR",
    )
    actual = [
        (role, artifact.reference.relative_path)
        for role in roles
        for artifact in artifacts.get(role, ())
    ]
    if len(actual) != 3 or len(set(actual)) != 3 or set(actual) != set(expected):
        raise ValueError("真实媒体 preflight 产物必须精确闭合")


def _require_artifact_digest(
    artifacts: dict[ArtifactRole, tuple[VerifiedArtifact, ...]],
    role: ArtifactRole,
    expected_sha256: str,
) -> None:
    if expected_sha256 not in {artifact.reference.sha256 for artifact in artifacts.get(role, ())}:
        raise ValueError(f"机器证据缺少匹配摘要的 {role} 产物")


def _verify_authorized_dataset(
    details: AuthorizedDatasetDetails,
    artifacts: dict[ArtifactRole, tuple[VerifiedArtifact, ...]],
    workspace_root: Path,
) -> None:
    manifests = artifacts.get("DATASET_MANIFEST", ())
    authorizations = artifacts.get("AUTHORIZATION_RECORD", ())
    if len(manifests) != 1 or len(authorizations) != 1:
        raise ValueError("授权数据集必须绑定唯一 Manifest 和授权记录")
    runtime_root = workspace_root / ".codex" / "video-rag-demo"
    source_snapshots = _snapshot_authorized_sources(
        manifests[0],
        authorizations[0],
        runtime_root,
        details.max_video_bytes,
    )
    package = load_evaluation_package(
        manifests[0].path,
        authorizations[0].path,
        workspace_root=workspace_root,
        runtime_root=runtime_root,
        max_video_bytes=details.max_video_bytes,
    )
    package.dataset.validate_final_gate(max_video_bytes=details.max_video_bytes)
    for snapshot in source_snapshots:
        _assert_snapshot_current(snapshot)
    counts = Counter(sample.language for sample in package.dataset.samples)
    if (
        len(package.dataset.samples) != details.item_count
        or dict(counts) != details.language_counts
    ):
        raise ValueError("授权数据集自报计数与严格重验不一致")
    if (
        package.dataset_sha256 != details.manifest_sha256
        or package.authorization_sha256 != details.authorization_record_sha256
    ):
        raise ValueError("授权数据集摘要与严格重验不一致")


def _snapshot_authorized_sources(
    manifest: VerifiedArtifact,
    authorization: VerifiedArtifact,
    runtime_root: Path,
    max_video_bytes: int,
) -> tuple[FileSnapshot, ...]:
    encoded = manifest.snapshot.content
    if encoded is None:
        raise ValueError("授权 Manifest 缺少受限快照")
    snapshots = [manifest.snapshot, authorization.snapshot]
    observed_paths = {manifest.path, authorization.path}
    eval_root = manifest.path.parent
    generated_root = (runtime_root / "eval" / "generated").resolve(strict=False)
    for line in encoded.decode("utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError("授权 Manifest 行必须是对象")
        for field, is_media in (
            ("media_relative_path", True),
            ("annotations_relative_path", False),
        ):
            relative = payload.get(field)
            if not isinstance(relative, str):
                raise ValueError("授权 Manifest 参与文件路径非法")
            candidate = eval_root / relative
            resolved = candidate.resolve(strict=True)
            if not resolved.is_relative_to(runtime_root.resolve(strict=True)):
                raise ValueError("授权 Manifest 参与文件逃逸运行根")
            if is_media and resolved.is_relative_to(generated_root):
                raise ValueError("授权 Manifest 不得引用生成媒体")
            if resolved in observed_paths:
                continue
            observed_paths.add(resolved)
            snapshots.append(
                _read_file_snapshot(
                    resolved,
                    max_bytes=max_video_bytes if is_media else _MAX_EVIDENCE_BYTES,
                    capture_content=False,
                )
            )
    return tuple(snapshots)


def _verify_performance(
    details: PerformanceDetails,
    artifacts: dict[ArtifactRole, tuple[VerifiedArtifact, ...]],
    workspace_root: Path,
    *,
    settings: Settings | None,
) -> bool:
    required_details = (
        details.evaluation_run_id,
        details.manifest_sha256,
        details.authorization_sha256,
        details.implementation_sha256,
        details.settings_fingerprint,
        details.sample_report_sha256s,
    )
    if any(value is None for value in required_details) or any(
        sample.sample_id is None
        or sample.media_relative_path is None
        or sample.authorization_id is None
        or sample.terminal_status is None
        or sample.probe_report_sha256 is None
        for sample in details.samples
    ):
        raise ValueError("旧版 M1 性能 Schema 缺少可信运行绑定")
    if settings is None:
        raise ValueError("M1 性能重验必须提供当前 Settings")
    raw_reports = artifacts.get("PERFORMANCE_REPORT", ())
    aggregate = tuple(
        artifact
        for artifact in raw_reports
        if artifact.reference.sha256 == details.performance_report_sha256
    )
    if len(raw_reports) != 3 or len(aggregate) != 1:
        raise ValueError("M1 性能证据必须绑定一份汇总和两份样本 raw")
    encoded = aggregate[0].snapshot.content
    if encoded is None:
        raise ValueError("M1 性能原始报告缺少快照")
    raw = PerformanceRawReport.model_validate_json(encoded)
    if (
        raw.evaluation_run_id != details.evaluation_run_id
        or raw.evaluation_run_id != aggregate[0].path.parent.name
        or raw.manifest_sha256 != details.manifest_sha256
        or raw.authorization_sha256 != details.authorization_sha256
        or raw.implementation_sha256 != details.implementation_sha256
        or raw.settings_fingerprint != details.settings_fingerprint
        or raw.sample_report_sha256s != details.sample_report_sha256s
        or raw.samples != details.samples
    ):
        raise ValueError("M1 detail、汇总 raw 与运行绑定不一致")
    if raw.implementation_sha256 != _current_durability_implementation_sha256(workspace_root):
        raise ValueError("M1 性能报告不是当前实现")
    current_identity = build_production_model_identity_report(settings)
    if raw.settings_fingerprint != current_identity.settings_fingerprint:
        raise ValueError("M1 性能报告不是当前运行设置")
    _verify_performance_sources(raw, artifacts, workspace_root)
    sample_raw = tuple(
        artifact
        for artifact in raw_reports
        if artifact.reference.sha256 in raw.sample_report_sha256s
    )
    if len(sample_raw) != 2 or {artifact.reference.sha256 for artifact in sample_raw} != set(
        raw.sample_report_sha256s
    ):
        raise ValueError("M1 两份样本 raw 未精确绑定")
    parsed_samples = tuple(
        PerformanceSampleRawReport.model_validate_json(artifact.snapshot.content).sample
        for artifact in sample_raw
        if artifact.snapshot.content is not None
    )
    if set(parsed_samples) != set(raw.samples):
        raise ValueError("M1 样本 raw 与汇总逐样本事实不一致")
    passed = details.trace.exit_code == 0
    for sample in raw.samples:
        expected_rtf = sample.elapsed_seconds / (sample.duration_ms / 1000)
        if abs(sample.rtf - expected_rtf) > 1e-12:
            raise ValueError("M1 性能逐样本 RTF 与原始时长不一致")
        passed = passed and (
            sample.duration_ms >= 1_800_000
            and sample.width >= 1920
            and sample.height >= 1080
            and sample.rtf <= 3.0
            and not sample.oom_detected
            and sample.peak_concurrency == 1
            and sample.outside_workspace_write_count == 0
            and sample.succeeded
        )
    if passed and any(
        sample.production_run_id is None
        or sample.job_id is None
        or sample.result_manifest_relative_path is None
        or sample.result_manifest_sha256 is None
        for sample in raw.samples
    ):
        raise ValueError("M1 PASS 的每个样本必须绑定完整生产 run/job/result Manifest")
    return passed


def _verify_performance_sources(
    raw: PerformanceRawReport,
    artifacts: dict[ArtifactRole, tuple[VerifiedArtifact, ...]],
    workspace_root: Path,
) -> None:
    manifests = artifacts.get("DATASET_MANIFEST", ())
    authorizations = artifacts.get("AUTHORIZATION_RECORD", ())
    inputs = artifacts.get("INPUT_MEDIA", ())
    probes = artifacts.get("AUDIT_REPORT", ())
    results = artifacts.get("PRODUCTION_RESULT", ())
    if len(manifests) != 1 or len(authorizations) != 1 or len(inputs) != 2 or len(probes) != 2:
        raise ValueError("M1 Manifest、授权、输入或 probe 绑定不完整")
    if (
        manifests[0].reference.sha256 != raw.manifest_sha256
        or authorizations[0].reference.sha256 != raw.authorization_sha256
    ):
        raise ValueError("M1 Manifest 或授权摘要不匹配")
    manifest_samples = _load_durability_manifest(manifests[0])
    if manifest_samples != tuple(
        (
            sample.sample_id,
            cast(str, sample.media_relative_path).removeprefix("eval/durability/"),
            sample.sample_sha256,
            sample.authorization_id,
            sample.duration_ms,
            sample.width,
            sample.height,
        )
        for sample in raw.samples
    ):
        raise ValueError("M1 Manifest 与逐样本事实不一致")
    authorization_encoded = authorizations[0].snapshot.content
    if authorization_encoded is None:
        raise ValueError("M1 授权记录缺少受限快照")
    authorization = AuthorizationFile.model_validate_json(authorization_encoded)
    records = {record.authorization_id: record for record in authorization.records}
    if any(
        (record := records.get(cast(str, sample.authorization_id))) is None
        or sample.sample_sha256 not in record.media_sha256
        for sample in raw.samples
    ):
        raise ValueError("M1 授权记录未覆盖逐样本媒体")
    if {artifact.reference.sha256 for artifact in inputs} != {
        sample.sample_sha256 for sample in raw.samples
    }:
        raise ValueError("M1 输入媒体与逐样本摘要不一致")
    for sample in raw.samples:
        matching_probe = tuple(
            artifact
            for artifact in probes
            if artifact.reference.sha256 == sample.probe_report_sha256
        )
        if len(matching_probe) != 1 or matching_probe[0].snapshot.content is None:
            raise ValueError("M1 probe 事实未唯一绑定")
        probe = json.loads(matching_probe[0].snapshot.content)
        if probe != {
            "schema_version": "1.0.0",
            "sample_id": sample.sample_id,
            "media_sha256": sample.sample_sha256,
            "duration_ms": sample.duration_ms,
            "width": sample.width,
            "height": sample.height,
        }:
            raise ValueError("M1 probe 事实与样本不一致")
    expected_results = {
        (sample.result_manifest_relative_path, sample.result_manifest_sha256)
        for sample in raw.samples
        if sample.result_manifest_relative_path is not None
    }
    actual_results = {
        (artifact.reference.relative_path, artifact.reference.sha256) for artifact in results
    }
    if actual_results != expected_results:
        raise ValueError("M1 生产结果 Manifest 必须精确闭合")
    result_by_path = {artifact.reference.relative_path: artifact for artifact in results}
    scope_key = hashlib.sha256(b"evaluation\x00video-demo\x00evaluation").hexdigest()[:24]
    for sample in raw.samples:
        if sample.result_manifest_relative_path is None:
            continue
        artifact = result_by_path[sample.result_manifest_relative_path]
        expected_parent = (
            workspace_root
            / ".codex/video-rag-demo/runs"
            / scope_key
            / cast(str, sample.production_run_id)
            / "result"
        )
        if artifact.path.parent != expected_parent or not artifact.path.name.startswith("bundle-"):
            raise ValueError("M1 生产结果 Manifest 路径非法")
        encoded = artifact.snapshot.content
        if encoded is None:
            raise ValueError("M1 生产结果 Manifest 缺少受限快照")
        envelope = json.loads(encoded)
        if (
            not isinstance(envelope, dict)
            or envelope.get("upstream_sha256") != sample.sample_sha256
        ):
            raise ValueError("M1 生产结果 Manifest 上游摘要不匹配")
        payload = DocumentArtifactPayload.model_validate(envelope.get("payload"))
        if (
            payload.result.run_id != sample.production_run_id
            or payload.result.asset_sha256 != sample.sample_sha256
            or payload.status != sample.terminal_status
        ):
            raise ValueError("M1 生产结果 Manifest 与逐样本执行事实不一致")


def _load_durability_manifest(
    artifact: VerifiedArtifact,
) -> tuple[tuple[str, str, str, str, int, int, int], ...]:
    encoded = artifact.snapshot.content
    if encoded is None:
        raise ValueError("M1 Manifest 缺少受限快照")
    values: list[tuple[str, str, str, str, int, int, int]] = []
    for line in encoded.decode("utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError("M1 Manifest 行必须是对象")
        values.append(
            (
                str(payload["sample_id"]),
                str(payload["media_relative_path"]),
                str(payload["media_sha256"]),
                str(payload["authorization_id"]),
                int(payload["duration_ms"]),
                int(payload["width"]),
                int(payload["height"]),
            )
        )
    return tuple(values)


def _load_workspace_evidence(
    candidate: Path,
    workspace_root: Path,
) -> tuple[Path, str, bytes]:
    root = workspace_root.expanduser().resolve(strict=True)
    unresolved = candidate.expanduser() if candidate.is_absolute() else root / candidate
    try:
        relative_unresolved = unresolved.relative_to(root)
    except ValueError:
        raise ValueError("证据文件必须位于工作区内") from None
    current = root
    for part in relative_unresolved.parts:
        current /= part
        if current.is_symlink():
            raise ValueError("证据文件及其父目录不得是符号链接")
    try:
        report = unresolved.resolve(strict=True)
    except FileNotFoundError:
        raise ValueError("证据文件不存在") from None
    if not report.is_relative_to(root):
        raise ValueError("证据文件必须位于工作区内")
    if not report.is_file():
        raise ValueError("证据路径必须指向普通文件")
    snapshot = _read_file_snapshot(
        report,
        max_bytes=_MAX_EVIDENCE_BYTES,
        capture_content=True,
    )
    if snapshot.content is None:
        raise ValueError("证据文件快照缺少正文")
    _assert_snapshot_current(snapshot)
    return report, report.relative_to(root).as_posix(), snapshot.content


def _parse_failure_matrix_junit(encoded: bytes) -> dict[str, str]:
    root = _parse_junit_xml(encoded)
    observed = _junit_node_statuses(root)

    outcomes: dict[str, str] = {}
    for scenario, node_ids in FAILURE_SCENARIO_TESTS.items():
        statuses = tuple(_node_status(observed.get(node_id, set())) for node_id in node_ids)
        if "FAIL" in statuses:
            outcomes[scenario] = "FAIL"
        elif "NOT_RUN" in statuses:
            outcomes[scenario] = "NOT_RUN"
        else:
            outcomes[scenario] = "PASS"
    return outcomes


def _verify_automated_tests_check(check: GateCheck, workspace_root: Path) -> None:
    junit = tuple(
        evidence for evidence in check.evidence if evidence.kind == EvidenceKind.PYTEST_JUNIT
    )
    collections = tuple(
        evidence for evidence in check.evidence if evidence.kind == EvidenceKind.PYTEST_COLLECTION
    )
    if len(junit) != 1 or len(collections) != 1 or len(check.evidence) != 2:
        raise ValueError("全量 pytest 门禁必须绑定一份 JUnit 和一份收集清单")
    parsed = build_automated_tests_check(
        Path(junit[0].relative_path),
        collection_path=Path(collections[0].relative_path),
        workspace_root=workspace_root,
    )
    if parsed.status != check.status or parsed.not_run_reason != check.not_run_reason:
        raise ValueError("全量 pytest 状态与 JUnit/收集清单不一致")
    if parsed.evidence != check.evidence:
        raise ValueError("全量 pytest 引用元数据与 JUnit/收集清单不一致")


def _parse_pytest_collection(encoded: bytes) -> set[str]:
    try:
        lines = encoded.decode("utf-8").splitlines()
    except UnicodeDecodeError:
        raise ValueError("pytest 收集清单必须是 UTF-8") from None
    collected = {
        line.strip() for line in lines if line.strip().startswith("tests/") and "::" in line
    }
    if not collected:
        raise ValueError("pytest 收集清单没有精确 node ID")
    return collected


def _collect_current_pytest_nodes(workspace_root: Path) -> tuple[set[str], str | None]:
    root = workspace_root.expanduser().resolve(strict=True)
    temporary_root = root / ".codex" / "tmp" / "pytest-gate"
    temporary_root.mkdir(parents=True, exist_ok=True)
    config_path = root / "pyproject.toml"
    if not config_path.is_file():
        config_path = temporary_root / "empty-pytest.ini"
        config_path.write_text("[pytest]\n", encoding="utf-8")
    command = (
        sys.executable,
        "-m",
        "pytest",
        "--collect-only",
        "-q",
        "-p",
        "no:cacheprovider",
        "--rootdir",
        str(root),
        "-c",
        str(config_path),
    )
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["TMPDIR"] = str(temporary_root)
    try:
        completed = subprocess.run(
            command,
            cwd=root,
            env=environment,
            capture_output=True,
            check=False,
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return set(), type(error).__name__
    if completed.returncode != 0:
        return set(), f"PYTEST_COLLECTION_FAILED(exit_code={completed.returncode})"
    try:
        return _parse_pytest_collection(completed.stdout), None
    except ValueError as error:
        return set(), str(error)


def _parse_junit_xml(encoded: bytes) -> ElementTree.Element:
    try:
        return ElementTree.fromstring(encoded)
    except (ElementTree.ParseError, UnicodeDecodeError):
        raise ValueError("证据文件不是合法的 JUnit XML") from None


def _junit_node_statuses(root: ElementTree.Element) -> dict[str, set[str]]:
    observed: dict[str, set[str]] = {}
    for case in root.iter("testcase"):
        node_id = _pytest_node_id(case)
        if node_id is None:
            continue
        if case.find("failure") is not None or case.find("error") is not None:
            status = "FAIL"
        elif case.find("skipped") is not None:
            status = "NOT_RUN"
        else:
            status = "PASS"
        observed.setdefault(node_id, set()).add(status)
    return observed


def _junit_case_statuses(root: ElementTree.Element) -> dict[tuple[str, str], set[str]]:
    observed: dict[tuple[str, str], set[str]] = {}
    for case in root.iter("testcase"):
        class_name = case.get("classname")
        test_name = case.get("name")
        if not class_name or not test_name:
            continue
        if case.find("failure") is not None or case.find("error") is not None:
            status = "FAIL"
        elif case.find("skipped") is not None:
            status = "NOT_RUN"
        else:
            status = "PASS"
        observed.setdefault((class_name, test_name), set()).add(status)
    return observed


def _acknowledged_automated_xfail_identities(
    root: ElementTree.Element,
) -> set[tuple[str, str]]:
    acknowledged: set[tuple[str, str]] = set()
    for case in root.iter("testcase"):
        class_name = case.get("classname")
        test_name = case.get("name")
        skipped = case.find("skipped")
        if not class_name or not test_name or skipped is None:
            continue
        node_id = f"{class_name.replace('.', '/')}.py::{test_name}"
        expected_reason = _ALLOWED_AUTOMATED_XFAILS.get(node_id)
        if (
            expected_reason is not None
            and skipped.get("type") == "pytest.xfail"
            and skipped.get("message") == expected_reason
        ):
            acknowledged.add((class_name, test_name))
    return acknowledged


def _pytest_node_id(case: ElementTree.Element) -> str | None:
    class_name = case.get("classname")
    test_name = case.get("name")
    if not class_name or not test_name:
        return None
    return f"{class_name.replace('.', '/')}.py::{test_name}"


def _node_status(statuses: set[str]) -> str:
    if "FAIL" in statuses:
        return "FAIL"
    if "NOT_RUN" in statuses or not statuses:
        return "NOT_RUN"
    return "PASS"


def _pytest_collection_identity(node_id: str) -> tuple[str, str]:
    parts = node_id.split("::")
    module = parts[0].removesuffix(".py").replace("/", ".")
    if len(parts) < 2:
        raise ValueError(f"pytest 收集项不是精确 node ID: {node_id}")
    test_name = parts[-1]
    class_name = ".".join((module, *parts[1:-1]))
    return class_name, test_name


def _pytest_incomplete_reason(
    missing: tuple[str, ...],
    unexpected: tuple[str, ...],
    statuses: tuple[str, ...],
) -> str:
    parts: list[str] = []
    if missing:
        parts.append(f"缺少 {len(missing)} 项，例：{', '.join(missing[:3])}")
    if unexpected:
        parts.append(f"多出 {len(unexpected)} 项，例：{', '.join(unexpected[:3])}")
    skipped_count = statuses.count("NOT_RUN")
    if skipped_count:
        parts.append(f"跳过 {skipped_count} 项")
    return "；".join(parts) or "收集清单为空"


def _source_collection_mismatch_reason(
    missing: tuple[str, ...],
    unexpected: tuple[str, ...],
) -> str:
    parts = ["提供的收集清单与当前源码不一致"]
    if missing:
        parts.append(f"当前源码另有 {len(missing)} 项，例：{', '.join(missing[:3])}")
    if unexpected:
        parts.append(f"清单多报 {len(unexpected)} 项，例：{', '.join(unexpected[:3])}")
    return "；".join(parts)
