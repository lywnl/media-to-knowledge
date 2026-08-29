from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import stat
import subprocess
import sys
import uuid
from collections import Counter
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import Field, field_validator, model_validator

from video_demo.capabilities import probe_runtime_capabilities
from video_demo.config import Settings
from video_demo.domain.base import FrozenModel, Sha256, StableId
from video_demo.evaluation.annotations import load_evaluation_package
from video_demo.evaluation.document_judgments import DocumentQualityReport
from video_demo.evaluation.durability import DurabilityRunReport, DurabilitySampleResult
from video_demo.evaluation.evidence import (
    AuthorizedDatasetDetails,
    CommandTrace,
    EvidenceKind,
    EvidenceLevel,
    EvidenceStore,
    MachineEvidenceReport,
    OfflineCheckId,
    OfflineEvidenceDetails,
    OfflineRawReport,
    PerformanceDetails,
    build_verified_gate_check,
    offline_audited_paths,
    offline_command,
    offline_observation_sha256,
)
from video_demo.evaluation.gate import (
    _FINAL_GATE_BUILD_TOKEN,
    FINAL_GATE_CHECKS,
    FinalGateReport,
    GateCheck,
    _current_offline_input_sha256,
    _recompute_offline_observations,
    build_automated_tests_check,
    build_failure_matrix_check,
    build_final_gate_report,
)
from video_demo.evaluation.metrics import RuntimeResourceMetrics
from video_demo.evaluation.report import (
    BoundQualityReport,
    GateStatus,
    MetricResult,
    QualityReport,
    build_quality_report,
)
from video_demo.evaluation.thresholds import QUALITY_THRESHOLDS
from video_demo.storage.workspace import validate_path_component

_SUCCESS_TERMINAL_STATUSES = frozenset({"SUCCEEDED", "PARTIAL_SUCCEEDED"})
_ACTIVE_MARKERS = frozenset({".evaluation-active", ".real-media.incomplete"})
_JOB_STATUS_BY_RUN_STATUS = {
    "SUCCEEDED": "SUCCEEDED",
    "PARTIAL_SUCCEEDED": "SUCCEEDED",
    "FAILED": "FAILED",
    "CANCELLED": "CANCELLED",
}


@dataclass(frozen=True, slots=True)
class RequirementSpec:
    requirement_id: int
    requirement: str
    check_ids: tuple[str, ...]


REQUIREMENT_SPECS: tuple[RequirementSpec, ...] = (
    RequirementSpec(1, "规格、实施计划与完成判定可追溯", ("automated_tests",)),
    RequirementSpec(2, "Python 3.11 项目骨架与依赖范围固定", ("ruff", "mypy")),
    RequirementSpec(3, "运行能力和真实工具版本可探测", ("real_media_chain",)),
    RequirementSpec(4, "工作区路径与环境 Secret 边界安全", ("secret_scan",)),
    RequirementSpec(5, "严格领域 Schema 拒绝非法与额外字段", ("automated_tests",)),
    RequirementSpec(6, "SQLite 迁移往返和三层作用域隔离", ("alembic_roundtrip",)),
    RequirementSpec(7, "上传对象隔离、格式校验和摘要绑定", ("failure_matrix",)),
    RequirementSpec(8, "输入扫描、路径逃逸和跨租户访问失败关闭", ("failure_matrix",)),
    RequirementSpec(9, "可靠任务支持租约、取消、重试和断点续跑", ("failure_matrix",)),
    RequirementSpec(10, "FastAPI 契约和作用域接口完整", ("openapi_contract",)),
    RequirementSpec(11, "ffprobe 对真实媒体执行安全预检", ("real_media_chain",)),
    RequirementSpec(12, "FFmpeg 标准化链真实执行并校验产物", ("real_media_chain",)),
    RequirementSpec(13, "Silero VAD 使用真实模型和授权音频", ("five_language_models",)),
    RequirementSpec(14, "五语识别覆盖中英日韩西", ("five_language_models",)),
    RequirementSpec(15, "云端 Whisper 严格串行执行五语 ASR", ("five_language_models",)),
    RequirementSpec(20, "真实 scene 检测生成候选边界", ("real_media_chain",)),
    RequirementSpec(21, "音画证据构造确定性混合窗口", ("real_media_chain",)),
    RequirementSpec(22, "关键帧选择过滤黑帧并去重", ("real_media_chain",)),
    RequirementSpec(23, "Qwen3-VL 对授权多图中的画面文字执行真实取证", ("chapter_vlm_live",)),
    RequirementSpec(24, "时间轴证据稳定排序且引用合法", ("automated_tests",)),
    RequirementSpec(25, "多模态理解端口只接收冻结证据", ("automated_tests",)),
    RequirementSpec(
        26,
        "章节 VLM 对 2~4 张本地 JPEG 完成一次有序多图逻辑调用",
        ("chapter_vlm_live",),
    ),
    RequirementSpec(27, "Qwen 输出严格校验且提示词与数据隔离", ("chapter_vlm_live",)),
    RequirementSpec(28, "边界合并仅吸附已有候选", ("automated_tests",)),
    RequirementSpec(29, "生产音画流水线发布结构化结果", ("five_language_models",)),
    RequirementSpec(30, "Markdown 作为唯一视频文本产物且不建设检索索引", ("no_indexing",)),
    RequirementSpec(31, "结果、证据和关键帧查询契约完整", ("openapi_contract",)),
    RequirementSpec(32, "失败恢复、重启续跑和取消语义正确", ("failure_matrix",)),
    RequirementSpec(33, "授权评测集不少于 30 条且五语各不少于 6 条", ("authorized_dataset",)),
    RequirementSpec(34, "质量指标由绑定预测和人工审阅重算", ("five_language_models",)),
    RequirementSpec(
        35,
        "十五项质量与资源阈值全部满足",
        ("authorized_dataset", "five_language_models", "m1_durability"),
    ),
    RequirementSpec(
        36,
        "失败矩阵、安全审计和全量自动化门禁通过",
        ("failure_matrix", "secret_scan", "automated_tests"),
    ),
    RequirementSpec(37, "M1 两段耐久和最终完成审计通过", ("m1_durability",)),
)


class RequirementEvidenceRow(FrozenModel):
    requirement_id: int = Field(ge=1, le=37)
    requirement: str = Field(min_length=1, max_length=300)
    check_ids: tuple[str, ...] = Field(min_length=1)
    evidence_paths: tuple[str, ...] = Field(min_length=1)
    status: GateStatus

    @field_validator("check_ids")
    @classmethod
    def validate_check_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)) or not set(value) <= set(FINAL_GATE_CHECKS):
            raise ValueError("要求证据只能引用固定最终检查")
        return value

    @field_validator("evidence_paths")
    @classmethod
    def validate_evidence_paths(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("要求证据路径不得重复")
        for item in value:
            path = PurePosixPath(item)
            if path.is_absolute() or not item or ".." in path.parts:
                raise ValueError("要求证据路径必须位于工作区内")
        return value


class RequirementEvidenceReport(FrozenModel):
    schema_version: Literal["1.0.0"]
    evaluation_run_id: StableId
    final_report_sha256: Sha256
    rows: tuple[RequirementEvidenceRow, ...] = Field(min_length=33, max_length=33)

    @model_validator(mode="after")
    def validate_complete_matrix(self) -> RequirementEvidenceReport:
        expected = {spec.requirement_id: spec for spec in REQUIREMENT_SPECS}
        supplied = {row.requirement_id: row for row in self.rows}
        if len(supplied) != len(expected) or set(supplied) != set(expected):
            raise ValueError("要求证据必须恰好覆盖当前保留的固定规格编号")
        for requirement_id, row in supplied.items():
            spec = expected[requirement_id]
            if row.requirement != spec.requirement or row.check_ids != spec.check_ids:
                raise ValueError("要求证据文本和检查映射必须来自固定规格")
        return self


class FinalValidationBundle(FrozenModel):
    schema_version: Literal["1.0.0"]
    evaluation_run_id: StableId
    final: FinalGateReport
    requirements: RequirementEvidenceReport

    @model_validator(mode="after")
    def validate_bundle_binding(self) -> FinalValidationBundle:
        if self.requirements.evaluation_run_id != self.evaluation_run_id:
            raise ValueError("最终报告与要求证据运行 ID 不一致")
        return self


class CleanupManifest(FrozenModel):
    schema_version: Literal["1.0.0"]
    evaluation_run_id: StableId
    created_at: datetime
    planned_paths: tuple[str, ...]
    inventory_sha256: Sha256


class CleanupResult(FrozenModel):
    evaluation_run_id: StableId
    deleted_paths: tuple[str, ...]
    manifest_path: str


class StageExecutionResult(FrozenModel):
    status: GateStatus
    report_path: str
    reason: str | None = Field(default=None, min_length=1, max_length=500)

    @model_validator(mode="after")
    def validate_reason(self) -> StageExecutionResult:
        if self.status != GateStatus.PASS and self.reason is None:
            raise ValueError("非 PASS 阶段结果必须提供稳定原因")
        if self.status == GateStatus.PASS and self.reason is not None:
            raise ValueError("PASS 阶段结果不得提供失败原因或未运行原因")
        path = PurePosixPath(self.report_path)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("阶段报告路径必须是工作区相对路径")
        return self


class PreflightSummary(FrozenModel):
    schema_version: Literal["1.0.0"]
    evaluation_run_id: StableId
    status: GateStatus
    stage_reasons: dict[str, str]
    created_at: datetime


class StageNotRunSummary(FrozenModel):
    schema_version: Literal["1.0.0"]
    evaluation_run_id: StableId
    stage: Literal["quality_predict", "quality_score"]
    status: Literal[GateStatus.NOT_RUN]
    reason: str = Field(min_length=1, max_length=500)
    created_at: datetime


class LiveCheckSummary(FrozenModel):
    check_id: str = Field(min_length=3, max_length=128)
    status: GateStatus
    report_path: str = Field(min_length=1, max_length=1024)
    reason: str | None = Field(default=None, min_length=1, max_length=500)

    @model_validator(mode="after")
    def validate_summary(self) -> LiveCheckSummary:
        if self.status == GateStatus.NOT_RUN and self.reason is None:
            raise ValueError("live NOT_RUN 检查必须提供稳定原因")
        if self.status != GateStatus.NOT_RUN and self.reason is not None:
            raise ValueError("已运行 live 检查不得包含未运行原因")
        path = PurePosixPath(self.report_path)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("live 检查报告必须是工作区相对路径")
        return self


class LiveValidationSummary(FrozenModel):
    schema_version: Literal["1.0.0"]
    evaluation_run_id: StableId
    status: GateStatus
    checks: tuple[LiveCheckSummary, LiveCheckSummary]
    created_at: datetime

    @model_validator(mode="after")
    def validate_checks(self) -> LiveValidationSummary:
        expected = ("chapter_vlm_live", "five_language_models")
        if tuple(item.check_id for item in self.checks) != expected:
            raise ValueError("live 汇总必须按固定顺序覆盖章节 VLM 与五语检查")
        if self.status != _aggregate_status(tuple(item.status for item in self.checks)):
            raise ValueError("live 汇总状态与检查状态不一致")
        return self


class FinalValidationRunner:
    """只编排现有 runner、重验权威证据并发布最终只读视图。"""

    def __init__(self, settings: Settings, evidence_store: EvidenceStore) -> None:
        self._settings = settings
        self._store = evidence_store

    def preflight(self, evaluation_run_id: str) -> StageExecutionResult:
        validate_path_component(evaluation_run_id, "evaluation_run_id")
        capabilities = probe_runtime_capabilities(self._settings)
        eval_root = self._store.runtime_root / "eval"
        stage_reasons: dict[str, str] = {}
        if capabilities.issues:
            stage_reasons["media"] = "工作区 FFmpeg/ffprobe 前置条件不足"
        if (
            not (eval_root / "dataset.jsonl").is_file()
            or not (eval_root / "authorization.json").is_file()
        ):
            stage_reasons["quality"] = "缺少授权五语评测集或授权记录"
            stage_reasons["live"] = "缺少授权五语评测集或授权记录"
        if (
            not (eval_root / "durability/dataset.jsonl").is_file()
            or not (eval_root / "durability/authorization.json").is_file()
        ):
            stage_reasons["durability"] = "缺少两段 30 分钟 1080p 耐久素材或授权记录"
        summary = PreflightSummary(
            schema_version="1.0.0",
            evaluation_run_id=evaluation_run_id,
            status=GateStatus.NOT_RUN if stage_reasons else GateStatus.PASS,
            stage_reasons=stage_reasons,
            created_at=datetime.now(UTC),
        )
        path = eval_root / "preflight" / f"{evaluation_run_id}.json"
        _atomic_write_json(
            path,
            summary.model_dump(mode="json"),
            workspace_root=self._settings.workspace_root,
        )
        return StageExecutionResult(
            status=summary.status,
            report_path=path.relative_to(self._settings.workspace_root).as_posix(),
            reason=("；".join(dict.fromkeys(stage_reasons.values())) if stage_reasons else None),
        )

    def final(self, evaluation_run_id: str) -> StageExecutionResult:
        validate_path_component(evaluation_run_id, "evaluation_run_id")
        checks = [*self._build_pytest_checks(evaluation_run_id)]
        checks.extend(self._build_offline_checks(evaluation_run_id))
        authorized_dataset = self.build_authorized_dataset_check(evaluation_run_id)
        if authorized_dataset is not None:
            checks.append(authorized_dataset)
        checks.extend(self._load_external_checks(evaluation_run_id))
        quality = self._load_quality(evaluation_run_id)
        document_quality = self._load_document_quality(evaluation_run_id)
        if document_quality is not None:
            self._verify_document_quality_binding(document_quality, quality)
        durability = next(
            (check for check in checks if check.check_id == "m1_durability"),
            None,
        )
        if (
            durability is not None
            and durability.status == GateStatus.PASS
            and isinstance(quality, BoundQualityReport)
        ):
            quality = self._bind_verified_durability(quality, durability)
            _atomic_write_json(
                self._store.runtime_root / "eval/reports" / evaluation_run_id / "quality.json",
                quality.model_dump(mode="json", exclude_none=True),
                workspace_root=self._settings.workspace_root,
            )
        final = build_final_gate_report(
            quality=quality,
            checks=checks,
            workspace_root=self._settings.workspace_root,
            settings=self._settings,
            document_quality=document_quality,
        )
        report_root = self._store.runtime_root / "eval/reports" / evaluation_run_id
        final_path = report_root / "final.json"
        _atomic_write_json(
            final_path,
            final.model_dump(mode="json"),
            workspace_root=self._settings.workspace_root,
        )
        requirements = build_requirement_evidence_report(
            evaluation_run_id=evaluation_run_id,
            final=final,
            final_path=final_path,
            workspace_root=self._settings.workspace_root,
        )
        requirement_path = report_root / "requirement-evidence.json"
        _atomic_write_json(
            requirement_path,
            requirements.model_dump(mode="json"),
            workspace_root=self._settings.workspace_root,
        )
        bundle = FinalValidationBundle.model_validate(
            {
                "schema_version": "1.0.0",
                "evaluation_run_id": evaluation_run_id,
                "final": final.model_dump(mode="json"),
                "requirements": requirements.model_dump(mode="json"),
            },
            context={
                "workspace_root": self._settings.workspace_root,
                "settings": self._settings,
                "build_token": _FINAL_GATE_BUILD_TOKEN,
            },
        )
        _atomic_write_json(
            report_root / "bundle.json",
            bundle.model_dump(mode="json"),
            workspace_root=self._settings.workspace_root,
        )
        write_report_schema(self._settings.workspace_root)
        return StageExecutionResult(
            status=final.status,
            report_path=final_path.relative_to(self._settings.workspace_root).as_posix(),
            reason=_final_reason(final),
        )

    def build_authorized_dataset_check(
        self,
        evaluation_run_id: str,
    ) -> GateCheck | None:
        """生成并立即重验授权数据集门禁；完全缺失输入时保留 NOT_RUN。"""

        validate_path_component(evaluation_run_id, "evaluation_run_id")
        eval_root = self._store.runtime_root / "eval"
        manifest = eval_root / "dataset.jsonl"
        authorization = eval_root / "authorization.json"
        report_relative = Path("eval/reports") / evaluation_run_id / "authorized_dataset.json"
        report_path = self._store.runtime_root / report_relative
        if report_path.is_file():
            return build_verified_gate_check(
                "authorized_dataset",
                report_path,
                workspace_root=self._settings.workspace_root,
            )
        if report_path.is_symlink():
            raise ValueError("授权数据集报告不得是符号链接")
        if not _any_dataset_input_present(manifest, authorization):
            return None
        started_at = datetime.now(UTC)
        try:
            package = load_evaluation_package(
                manifest,
                authorization,
                workspace_root=self._settings.workspace_root,
                runtime_root=self._store.runtime_root,
                max_video_bytes=self._settings.max_video_bytes,
            )
            package.dataset.validate_final_gate(max_video_bytes=self._settings.max_video_bytes)
        except Exception:
            raise ValueError("授权数据集非法或损坏") from None
        language_counts = Counter(sample.language for sample in package.dataset.samples)
        root = Path("eval/reports") / evaluation_run_id
        stdout = self._store.write_artifact(
            root / "authorized_dataset.stdout.txt",
            "COMMAND_STDOUT",
            b"authorized dataset validation completed\n",
        )
        stderr = self._store.write_artifact(
            root / "authorized_dataset.stderr.txt",
            "COMMAND_STDERR",
            b"",
        )
        manifest_artifact = self._store.bind_artifact(
            manifest.relative_to(self._store.runtime_root),
            "DATASET_MANIFEST",
        )
        authorization_artifact = self._store.bind_artifact(
            authorization.relative_to(self._store.runtime_root),
            "AUTHORIZATION_RECORD",
        )
        command = (
            "python",
            "-m",
            "video_demo.evaluation.cli",
            "final",
            "--evaluation-run-id",
            evaluation_run_id,
        )
        report = MachineEvidenceReport(
            schema_version="1.0.0",
            check_id="authorized_dataset",
            status=GateStatus.PASS,
            kind=EvidenceKind.COMMAND_REPORT,
            level=EvidenceLevel.REAL_MEDIA,
            covered_items=("authorized_dataset",),
            summary="授权五语数据集已完成最终门禁校验",
            producer="FinalValidationRunner",
            started_at=started_at,
            finished_at=datetime.now(UTC),
            artifacts=(
                manifest_artifact,
                authorization_artifact,
                stdout,
                stderr,
            ),
            details=AuthorizedDatasetDetails(
                type="AUTHORIZED_DATASET",
                trace=CommandTrace(
                    command=command,
                    exit_code=0,
                    stdout_sha256=stdout.sha256,
                    stderr_sha256=stderr.sha256,
                ),
                manifest_sha256=package.dataset_sha256,
                authorization_record_sha256=package.authorization_sha256,
                item_count=len(package.dataset.samples),
                language_counts={
                    "zh": language_counts["zh"],
                    "en": language_counts["en"],
                    "ja": language_counts["ja"],
                    "ko": language_counts["ko"],
                    "es": language_counts["es"],
                },
                max_video_bytes=self._settings.max_video_bytes,
            ),
        )
        reference = self._store.write_json(report_relative, report)
        return build_verified_gate_check(
            "authorized_dataset",
            self._settings.workspace_root / reference.relative_path,
            workspace_root=self._settings.workspace_root,
        )

    def _build_pytest_checks(self, evaluation_run_id: str) -> tuple[GateCheck, ...]:
        raw = self._store.runtime_root / "eval/reports" / evaluation_run_id / "raw"
        junit = raw / "pytest.xml"
        collection = raw / "pytest-collection.txt"
        checks: list[GateCheck] = []
        if junit.is_file():
            checks.append(
                build_failure_matrix_check(
                    junit,
                    workspace_root=self._settings.workspace_root,
                )
            )
        if junit.is_file() and collection.is_file():
            checks.append(
                build_automated_tests_check(
                    junit,
                    collection_path=collection,
                    workspace_root=self._settings.workspace_root,
                )
            )
        return tuple(checks)

    def _build_offline_checks(self, evaluation_run_id: str) -> tuple[GateCheck, ...]:
        runner = OfflineGateRunner(self._settings, self._store)
        check_ids: tuple[OfflineCheckId, ...] = (
            "no_indexing",
            "ruff",
            "mypy",
            "alembic_roundtrip",
            "openapi_contract",
            "secret_scan",
        )
        return tuple(
            runner.run(check_id, evaluation_run_id=evaluation_run_id) for check_id in check_ids
        )

    def _load_external_checks(self, evaluation_run_id: str) -> tuple[GateCheck, ...]:
        locations = {
            "real_media_chain": (
                stage_evaluation_run_id(evaluation_run_id, "media"),
                "real-media.json",
            ),
            "chapter_vlm_live": (
                stage_evaluation_run_id(evaluation_run_id, "chapter-vlm"),
                "chapter_vlm_live.json",
            ),
            "five_language_models": (
                stage_evaluation_run_id(evaluation_run_id, "models"),
                "five_language_models.json",
            ),
            "m1_durability": (
                stage_evaluation_run_id(evaluation_run_id, "durability"),
                "durability.json",
            ),
        }
        checks: list[GateCheck] = []
        for check_id, (run_id, filename) in locations.items():
            path = self._store.runtime_root / "eval/reports" / run_id / filename
            if path.is_file():
                checks.append(
                    build_verified_gate_check(
                        check_id,
                        path,
                        workspace_root=self._settings.workspace_root,
                        settings=self._settings,
                    )
                )
        return tuple(checks)

    def _load_quality(self, evaluation_run_id: str) -> QualityReport:
        report_root = self._store.runtime_root / "eval/reports" / evaluation_run_id
        prediction = report_root / "prediction.json"
        if not prediction.is_file():
            if (report_root / "quality.json").is_file():
                raise ValueError("质量报告缺少可重验预测来源")
            return build_quality_report({}, QUALITY_THRESHOLDS)
        try:
            from video_demo.evaluation.prediction_runner import score_prediction_run
            visual_quality = None
            dataset_path = self._store.runtime_root / "eval/dataset.jsonl"
            authorization_path = self._store.runtime_root / "eval/authorization.json"
            visual_path = report_root / "visual-quality.json"
            if visual_path.is_file() and dataset_path.is_file() and authorization_path.is_file():
                from video_demo.evaluation.visual_quality import (
                    VisualQualityReport,
                    build_visual_quality_set,
                    verify_visual_quality_report,
                )

                package = load_evaluation_package(
                    dataset_path,
                    authorization_path,
                    workspace_root=self._settings.workspace_root,
                    runtime_root=self._store.runtime_root,
                    max_video_bytes=self._settings.max_video_bytes,
                )
                visual_report = VisualQualityReport.model_validate_json(visual_path.read_bytes())
                quality_set = build_visual_quality_set(
                    package,
                    parent_evaluation_run_id=evaluation_run_id,
                    proxy_max_edge=1_920,
                    jpeg_quality=self._settings.keyframe_jpeg_quality,
                )
                visual_quality = verify_visual_quality_report(
                    visual_report, quality_set, package
                )

            report = score_prediction_run(
                evaluation_run_id,
                eval_root=self._store.runtime_root / "eval",
                visual_quality_report=visual_quality,
            )
        except Exception:
            raise ValueError("质量报告非法或损坏") from None
        if report.evaluation_run_id != evaluation_run_id:
            raise ValueError("质量报告运行 ID 不匹配")
        document_path = report_root / "document-quality.json"
        if document_path.is_file():
            from video_demo.evaluation.document_judgments import DocumentQualityReport

            document_quality = DocumentQualityReport.model_validate_json(document_path.read_bytes())
            if document_quality.evaluation_run_id != evaluation_run_id:
                raise ValueError("文档质量报告运行 ID 不匹配")
        return report

    def _load_document_quality(
        self, evaluation_run_id: str
    ) -> DocumentQualityReport | None:
        """读取与当前评测 Run 绑定的独立文档质量报告。"""

        path = (
            self._store.runtime_root
            / "eval/reports"
            / evaluation_run_id
            / "document-quality.json"
        )
        if not path.is_file():
            return None
        try:
            report = DocumentQualityReport.model_validate_json(path.read_bytes())
        except Exception:
            raise ValueError("文档质量报告非法或损坏") from None
        if report.evaluation_run_id != evaluation_run_id:
            raise ValueError("文档质量报告运行 ID 不匹配")
        return report

    @staticmethod
    def _verify_document_quality_binding(
        document_quality: DocumentQualityReport,
        quality: QualityReport,
    ) -> None:
        if not isinstance(quality, BoundQualityReport):
            raise ValueError("文档质量报告缺少绑定的预测质量报告")
        if (
            document_quality.evaluation_run_id != quality.evaluation_run_id
            or document_quality.dataset_sha256 != quality.dataset_sha256
            or document_quality.authorization_sha256 != quality.authorization_sha256
            or document_quality.prediction_index_sha256 != quality.prediction_index_sha256
        ):
            raise ValueError("文档质量报告与预测质量报告闭包不一致")

    def _bind_verified_durability(
        self,
        quality: BoundQualityReport,
        check: GateCheck,
    ) -> BoundQualityReport:
        if len(check.evidence) != 1:
            raise ValueError("耐久检查必须只有一份权威报告")
        report_path = self._settings.workspace_root / check.evidence[0].relative_path
        report = MachineEvidenceReport.model_validate_json(report_path.read_bytes())
        details = report.details
        if not isinstance(details, PerformanceDetails):
            raise ValueError("耐久报告明细类型非法")
        samples = tuple(
            DurabilitySampleResult(
                media_sha256=sample.sample_sha256,
                duration_ms=sample.duration_ms,
                width=sample.width,
                height=sample.height,
                elapsed_seconds=sample.elapsed_seconds,
                rtf=sample.rtf,
                peak_rss_bytes=sample.peak_rss_bytes,
                peak_disk_bytes=sample.peak_disk_bytes,
                oom=sample.oom_detected,
                peak_worker_concurrency=sample.peak_concurrency,
                outside_workspace_write_count=sample.outside_workspace_write_count,
                terminal_status=sample.terminal_status or "FAILED",
                failure_code=sample.failure_code,
            )
            for sample in details.samples
        )
        durability = DurabilityRunReport(
            schema_version="1.0.0",
            evaluation_run_id=details.evaluation_run_id or report_path.parent.name,
            status=check.status,
            samples=samples,
            started_at=report.started_at,
            finished_at=report.finished_at,
        )
        return bind_durability_to_quality(
            quality,
            durability,
            durability_report_sha256=check.evidence[0].sha256,
        )


class OfflineGateRunner:
    """执行固定离线检查，并交由 gate 从当前输入和 raw 事实重验。"""

    def __init__(self, settings: Settings, evidence_store: EvidenceStore) -> None:
        self._settings = settings
        self._store = evidence_store

    def run(self, check_id: OfflineCheckId, *, evaluation_run_id: str) -> GateCheck:
        validate_path_component(evaluation_run_id, "evaluation_run_id")
        report_relative = Path("eval/reports") / evaluation_run_id / f"{check_id}.json"
        report_path = self._store.runtime_root / report_relative
        if report_path.is_file():
            return build_verified_gate_check(
                check_id,
                report_path,
                workspace_root=self._settings.workspace_root,
            )
        if report_path.is_symlink():
            raise ValueError("离线门禁报告不得是符号链接")
        started_at = datetime.now(UTC)
        audited_paths = offline_audited_paths(check_id)
        input_sha256 = _current_offline_input_sha256(
            self._settings.workspace_root,
            audited_paths,
        )
        exit_code, stdout_bytes, stderr_bytes, observations = self._execute(check_id)
        status = GateStatus.PASS if exit_code == 0 and not observations else GateStatus.FAIL
        root = Path("eval/reports") / evaluation_run_id
        stdout = self._store.write_artifact(
            root / f"{check_id}.stdout.txt",
            "COMMAND_STDOUT",
            _safe_offline_output(stdout_bytes, self._settings.workspace_root),
        )
        stderr = self._store.write_artifact(
            root / f"{check_id}.stderr.txt",
            "COMMAND_STDERR",
            _safe_offline_output(stderr_bytes, self._settings.workspace_root),
        )
        observation_sha256 = offline_observation_sha256(observations)
        raw = OfflineRawReport(
            schema_version="1.0.0",
            check_id=check_id,
            evaluation_run_id=evaluation_run_id,
            status=status,
            input_sha256=input_sha256,
            command=offline_command(check_id),
            exit_code=exit_code,
            stdout_sha256=stdout.sha256,
            stderr_sha256=stderr.sha256,
            audited_paths=audited_paths,
            violation_count=len(observations),
            observations=observations,
            observation_sha256=observation_sha256,
        )
        raw_artifact = self._store.write_artifact(
            root / f"{check_id}.raw.json",
            "AUDIT_REPORT",
            raw.model_dump_json().encode("utf-8"),
        )
        report = MachineEvidenceReport(
            schema_version="1.0.0",
            check_id=check_id,
            status=status,
            kind=(
                EvidenceKind.STATIC_AUDIT
                if check_id in {"no_indexing", "ruff", "mypy", "secret_scan"}
                else EvidenceKind.COMMAND_REPORT
            ),
            level=(
                EvidenceLevel.STATIC
                if check_id in {"no_indexing", "ruff", "mypy", "secret_scan"}
                else EvidenceLevel.CONTRACT
            ),
            covered_items=(check_id,),
            summary=f"{check_id} 固定离线检查完成",
            producer="OfflineGateRunner",
            started_at=started_at,
            finished_at=datetime.now(UTC),
            artifacts=(raw_artifact, stdout, stderr),
            details=OfflineEvidenceDetails(
                type="OFFLINE",
                trace=CommandTrace(
                    command=offline_command(check_id),
                    exit_code=exit_code,
                    stdout_sha256=stdout.sha256,
                    stderr_sha256=stderr.sha256,
                ),
                raw_report_sha256=raw_artifact.sha256,
                input_sha256=input_sha256,
                observation_sha256=observation_sha256,
            ),
        )
        reference = self._store.write_json(report_relative, report)
        return build_verified_gate_check(
            check_id,
            self._settings.workspace_root / reference.relative_path,
            workspace_root=self._settings.workspace_root,
        )

    def _execute(
        self,
        check_id: OfflineCheckId,
    ) -> tuple[int, bytes, bytes, tuple[str, ...]]:
        if check_id == "ruff":
            return self._execute_tool("ruff")
        if check_id == "mypy":
            return self._execute_tool("mypy")
        if check_id == "alembic_roundtrip":
            return self._execute_alembic()
        observations = _recompute_offline_observations(
            check_id,
            self._settings.workspace_root,
        )
        output = f"{check_id}: {len(observations)} violation(s)\n".encode()
        return (0 if not observations else 1), output, b"", observations

    def _execute_tool(
        self,
        check_id: Literal["ruff", "mypy"],
    ) -> tuple[int, bytes, bytes, tuple[str, ...]]:
        arguments = (sys.executable, *offline_command(check_id)[1:])
        completed = subprocess.run(
            arguments,
            cwd=self._settings.workspace_root,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            check=False,
            timeout=600,
        )
        observations = (
            () if completed.returncode == 0 else (f"COMMAND_EXIT_{completed.returncode}",)
        )
        return completed.returncode, completed.stdout, completed.stderr, observations

    def _execute_alembic(self) -> tuple[int, bytes, bytes, tuple[str, ...]]:
        values = _recompute_offline_observations(
            "alembic_roundtrip",
            self._settings.workspace_root,
        )
        return (
            0 if not values else 1,
            b"alembic upgrade/downgrade/upgrade completed\n",
            b"",
            values,
        )


def bind_durability_to_quality(
    quality: BoundQualityReport,
    durability: DurabilityRunReport,
    *,
    durability_report_sha256: Sha256,
) -> BoundQualityReport:
    """使用已重验耐久报告的逐维最坏值回填运行时指标。"""

    if durability.status != GateStatus.PASS or not _durability_samples_pass(durability):
        raise ValueError("只有两段样本均通过的耐久报告才能回填质量")
    if quality.durability_report_sha256 is not None or quality.resources is not None:
        raise ValueError("质量报告已经绑定耐久资源，不得重复回填")
    resources = RuntimeResourceMetrics(
        rtf=max(sample.rtf for sample in durability.samples),
        peak_rss_bytes=max(sample.peak_rss_bytes for sample in durability.samples),
        peak_disk_bytes=max(sample.peak_disk_bytes for sample in durability.samples),
    )
    metrics = tuple(
        _bound_rtf(metric, resources.rtf) if metric.name == "rtf" else metric
        for metric in quality.metrics
    )
    if sum(metric.name == "rtf" for metric in quality.metrics) != 1:
        raise ValueError("质量报告必须恰好包含一个 RTF 指标")
    payload = quality.model_dump(mode="python")
    payload.update(
        {
            "status": _aggregate_metric_status(metrics, quality.failure_code),
            "metrics": metrics,
            "resources": resources,
            "resources_not_run_reason": None,
            "durability_report_sha256": durability_report_sha256,
        }
    )
    return BoundQualityReport.model_validate(payload)


def build_requirement_evidence_report(
    *,
    evaluation_run_id: str,
    final: FinalGateReport,
    final_path: Path,
    workspace_root: Path,
) -> RequirementEvidenceReport:
    validate_path_component(evaluation_run_id, "evaluation_run_id")
    workspace = workspace_root.resolve(strict=True)
    trusted_final = _trusted_regular_file(workspace, final_path)
    payload = json.loads(trusted_final.read_text(encoding="utf-8"))
    if payload != final.model_dump(mode="json"):
        raise ValueError("最终报告文件与已重验对象不一致")
    final_relative = trusted_final.relative_to(workspace).as_posix()
    checks = {check.check_id: check for check in final.checks}
    rows = tuple(_requirement_row(spec, checks, final_relative) for spec in REQUIREMENT_SPECS)
    return RequirementEvidenceReport(
        schema_version="1.0.0",
        evaluation_run_id=evaluation_run_id,
        final_report_sha256=hashlib.sha256(trusted_final.read_bytes()).hexdigest(),
        rows=rows,
    )


def write_live_validation_summary(
    *,
    evaluation_run_id: str,
    checks: tuple[GateCheck, GateCheck],
    workspace_root: Path,
) -> StageExecutionResult:
    validate_path_component(evaluation_run_id, "evaluation_run_id")
    workspace = workspace_root.resolve(strict=True)
    rows = tuple(_live_check_summary(check) for check in checks)
    summary = LiveValidationSummary(
        schema_version="1.0.0",
        evaluation_run_id=evaluation_run_id,
        status=_aggregate_status(tuple(check.status for check in checks)),
        checks=rows,
        created_at=datetime.now(UTC),
    )
    path = (
        workspace / ".codex/video-rag-demo/eval/reports" / evaluation_run_id / "live-summary.json"
    )
    _atomic_write_json(
        path,
        summary.model_dump(mode="json"),
        workspace_root=workspace,
    )
    return StageExecutionResult(
        status=summary.status,
        report_path=path.relative_to(workspace).as_posix(),
        reason=_checks_status_reason(checks),
    )


def write_stage_not_run_summary(
    *,
    evaluation_run_id: str,
    stage: Literal["quality_predict", "quality_score"],
    reason: str,
    workspace_root: Path,
) -> StageExecutionResult:
    validate_path_component(evaluation_run_id, "evaluation_run_id")
    workspace = workspace_root.resolve(strict=True)
    summary = StageNotRunSummary(
        schema_version="1.0.0",
        evaluation_run_id=evaluation_run_id,
        stage=stage,
        status=GateStatus.NOT_RUN,
        reason=reason,
        created_at=datetime.now(UTC),
    )
    filename = stage.replace("_", "-") + "-summary.json"
    path = workspace / ".codex/video-rag-demo/eval/reports" / evaluation_run_id / filename
    _atomic_write_json(
        path,
        summary.model_dump(mode="json"),
        workspace_root=workspace,
    )
    return StageExecutionResult(
        status=GateStatus.NOT_RUN,
        report_path=path.relative_to(workspace).as_posix(),
        reason=reason,
    )


def render_report_schema() -> bytes:
    payload = FinalValidationBundle.model_json_schema()
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def write_report_schema(workspace_root: Path) -> Path:
    workspace = workspace_root.resolve(strict=True)
    path = workspace / ".codex/video-rag-demo/eval/report.schema.json"
    _atomic_write_bytes(path, render_report_schema(), workspace_root=workspace)
    return path


def stage_evaluation_run_id(evaluation_run_id: str, stage: str) -> str:
    validate_path_component(evaluation_run_id, "evaluation_run_id")
    validate_path_component(stage, "stage")
    prefix = evaluation_run_id[:96]
    digest = hashlib.sha256(f"{evaluation_run_id}:{stage}".encode()).hexdigest()[:12]
    return f"{prefix}_{stage}_{digest}"


def cleanup_evaluation_run(
    workspace_root: Path,
    evaluation_run_id: str,
    *,
    settings: Settings | None = None,
) -> CleanupResult:
    try:
        validate_path_component(evaluation_run_id, "evaluation_run_id")
    except Exception:
        raise ValueError("评测运行 ID 非法") from None
    workspace = workspace_root.resolve(strict=True)
    runtime = workspace / ".codex/video-rag-demo"
    runtime.mkdir(parents=True, exist_ok=True)
    _reject_symlink_path(workspace, runtime)
    targets = _deduplicate_paths(
        (
            *_cleanup_targets(runtime, evaluation_run_id),
            *_bound_product_run_targets(
                workspace,
                runtime,
                evaluation_run_id,
                settings=settings,
            ),
        )
    )
    for target in targets:
        _assert_cleanup_target(runtime, target)
        if _contains_active_marker(target):
            raise ValueError("活跃评测运行不得清理")
    planned_paths = tuple(sorted(path.relative_to(workspace).as_posix() for path in targets))
    inventory = _cleanup_inventory(targets, workspace)
    manifest = CleanupManifest(
        schema_version="1.0.0",
        evaluation_run_id=evaluation_run_id,
        created_at=datetime.now(UTC),
        planned_paths=planned_paths,
        inventory_sha256=hashlib.sha256(
            json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    )
    manifest_relative = Path("eval/cleanup") / f"{evaluation_run_id}.json"
    manifest_path = runtime / manifest_relative
    _atomic_write_json(
        manifest_path,
        manifest.model_dump(mode="json"),
        workspace_root=workspace,
    )
    for target in targets:
        shutil.rmtree(target)
    return CleanupResult(
        evaluation_run_id=evaluation_run_id,
        deleted_paths=planned_paths,
        manifest_path=manifest_path.relative_to(workspace).as_posix(),
    )


def _durability_samples_pass(durability: DurabilityRunReport) -> bool:
    return all(
        sample.rtf <= 3.0
        and not sample.oom
        and sample.peak_worker_concurrency == 1
        and sample.outside_workspace_write_count == 0
        and sample.terminal_status in _SUCCESS_TERMINAL_STATUSES
        and sample.failure_code is None
        for sample in durability.samples
    )


def _bound_rtf(metric: MetricResult, rtf: float) -> MetricResult:
    status = GateStatus.PASS if rtf <= metric.threshold else GateStatus.FAIL
    return MetricResult(
        name=metric.name,
        value=rtf,
        threshold=metric.threshold,
        direction=metric.direction,
        status=status,
    )


def _aggregate_metric_status(
    metrics: tuple[MetricResult, ...],
    failure_code: str | None,
) -> GateStatus:
    if failure_code is not None or any(item.status == GateStatus.FAIL for item in metrics):
        return GateStatus.FAIL
    if any(item.status == GateStatus.NOT_RUN for item in metrics):
        return GateStatus.NOT_RUN
    return GateStatus.PASS


def _requirement_row(
    spec: RequirementSpec,
    checks: dict[str, GateCheck],
    final_relative: str,
) -> RequirementEvidenceRow:
    selected = tuple(checks[check_id] for check_id in spec.check_ids)
    evidence_paths = tuple(
        dict.fromkeys(
            (
                final_relative,
                *(evidence.relative_path for check in selected for evidence in check.evidence),
            )
        )
    )
    return RequirementEvidenceRow(
        requirement_id=spec.requirement_id,
        requirement=spec.requirement,
        check_ids=spec.check_ids,
        evidence_paths=evidence_paths,
        status=_aggregate_status(tuple(check.status for check in selected)),
    )


def _aggregate_status(statuses: Sequence[GateStatus]) -> GateStatus:
    if GateStatus.FAIL in statuses:
        return GateStatus.FAIL
    if GateStatus.NOT_RUN in statuses:
        return GateStatus.NOT_RUN
    return GateStatus.PASS


def _live_check_summary(check: GateCheck) -> LiveCheckSummary:
    if len(check.evidence) != 1:
        raise ValueError("live 检查必须恰好绑定一份权威报告")
    return LiveCheckSummary(
        check_id=check.check_id,
        status=check.status,
        report_path=check.evidence[0].relative_path,
        reason=check.not_run_reason,
    )


def _checks_status_reason(checks: Sequence[GateCheck]) -> str | None:
    failed = tuple(check.check_id for check in checks if check.status == GateStatus.FAIL)
    if failed:
        return f"失败门禁: {', '.join(failed)}"
    not_run = tuple(check.check_id for check in checks if check.status == GateStatus.NOT_RUN)
    if not_run:
        return f"未运行门禁: {', '.join(not_run)}"
    return None


def _trusted_regular_file(workspace: Path, path: Path) -> Path:
    candidate = path if path.is_absolute() else workspace / path
    _reject_symlink_path(workspace, candidate)
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(workspace) or not resolved.is_file():
        raise ValueError("证据文件必须位于工作区内")
    return resolved


def _any_dataset_input_present(manifest: Path, authorization: Path) -> bool:
    return any(path.exists() or path.is_symlink() for path in (manifest, authorization))


def _reject_symlink_path(root: Path, candidate: Path) -> None:
    try:
        relative = candidate.absolute().relative_to(root.absolute())
    except ValueError:
        raise ValueError("路径必须位于工作区内") from None
    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise ValueError("评测路径不得包含符号链接")


def _cleanup_targets(runtime: Path, evaluation_run_id: str) -> tuple[Path, ...]:
    stage_ids = {
        stage: stage_evaluation_run_id(evaluation_run_id, stage)
        for stage in ("media", "chapter-vlm", "models", "durability")
    }
    candidates = (
        runtime / "eval/reports" / evaluation_run_id,
        runtime / "eval/predictions" / evaluation_run_id,
        *(runtime / "eval/reports" / run_id for run_id in stage_ids.values()),
        *(
            runtime / "eval/live-authority" / stage_ids[stage]
            for stage in ("chapter-vlm", "models")
        ),
    )
    return tuple(path for path in candidates if path.exists() or path.is_symlink())


def _bound_product_run_targets(
    workspace: Path,
    runtime: Path,
    evaluation_run_id: str,
    *,
    settings: Settings | None,
) -> tuple[Path, ...]:
    run_ids = {
        *_verified_prediction_run_ids(workspace, runtime, evaluation_run_id),
        *_verified_durability_run_ids(
            workspace,
            runtime,
            evaluation_run_id,
            settings=settings,
        ),
    }
    scope_key = hashlib.sha256(b"evaluation\x00video-demo\x00evaluation").hexdigest()[:24]
    candidates = tuple(runtime / "runs" / scope_key / run_id for run_id in run_ids)
    return tuple(path for path in candidates if path.exists() or path.is_symlink())


def _verified_prediction_run_ids(
    workspace: Path,
    runtime: Path,
    evaluation_run_id: str,
) -> tuple[str, ...]:
    report_path = runtime / "eval/reports" / evaluation_run_id / "prediction.json"
    if not (report_path.exists() or report_path.is_symlink()):
        return ()
    try:
        from video_demo.evaluation.prediction_runner import (
            PredictionRunReport,
            _idempotency_key,
            _implementation_sha256,
            _prediction_index_digest,
            _prediction_index_path,
        )
        from video_demo.evaluation.predictions import load_verified_prediction

        trusted_report = _trusted_regular_file(workspace, report_path)
        report_bytes = trusted_report.read_bytes()
        report = PredictionRunReport.model_validate_json(report_bytes)
        if report_bytes != report.model_dump_json(exclude_none=True).encode("utf-8"):
            raise ValueError("预测报告不是规范机器序列化")
        if report.evaluation_run_id != evaluation_run_id:
            raise ValueError("预测报告运行 ID 不匹配")
        if report.status == GateStatus.NOT_RUN:
            return ()
        package = load_evaluation_package(
            runtime / "eval/dataset.jsonl",
            runtime / "eval/authorization.json",
            workspace_root=workspace,
            runtime_root=runtime,
        )
        if (
            report.dataset_sha256 != package.dataset_sha256
            or report.authorization_sha256 != package.authorization_sha256
            or report.implementation_sha256 != _implementation_sha256(workspace)
        ):
            raise ValueError("预测报告不是当前输入或实现")
        samples = {sample.sample_id: sample for sample in package.dataset.samples}
        if {item.sample_id for item in report.predictions} != set(samples):
            raise ValueError("预测报告未精确覆盖当前数据集")
        verified = tuple(
            load_verified_prediction(
                _prediction_index_path(
                    runtime / "eval",
                    evaluation_run_id,
                    item.sample_id,
                ),
                eval_root=runtime / "eval",
                workspace_root=workspace,
                runtime_root=runtime,
                sample=samples[item.sample_id],
            )
            for item in report.predictions
        )
        if (
            tuple(item.index for item in verified) != report.predictions
            or _prediction_index_digest(
                runtime / "eval",
                evaluation_run_id,
                report.predictions,
            )
            != report.prediction_index_sha256
        ):
            raise ValueError("预测报告与当前索引不一致")
        run_ids: list[str] = []
        for item in verified:
            bound_in_database = _database_run_is_owned(
                runtime,
                idempotency_key=_idempotency_key(
                    evaluation_run_id,
                    item.index.sample_id,
                ),
                run_id=item.index.run_id,
                job_id=item.index.job_id,
                media_sha256=item.index.media_sha256,
                terminal_status=item.index.terminal_status,
            )
            if bound_in_database:
                run_ids.append(item.index.run_id)
        return tuple(run_ids)
    except Exception:
        raise ValueError("预测清理证据非法或损坏") from None


def _verified_durability_run_ids(
    workspace: Path,
    runtime: Path,
    evaluation_run_id: str,
    *,
    settings: Settings | None,
) -> tuple[str, ...]:
    durability_run_id = stage_evaluation_run_id(evaluation_run_id, "durability")
    report_path = runtime / "eval/reports" / durability_run_id / "durability.json"
    if not (report_path.exists() or report_path.is_symlink()):
        return ()
    try:
        if settings is None:
            raise ValueError("耐久清理必须提供生成报告时的 Settings")
        check = build_verified_gate_check(
            "m1_durability",
            report_path,
            workspace_root=workspace,
            settings=settings,
        )
        if check.status == GateStatus.NOT_RUN:
            return ()
        report = MachineEvidenceReport.model_validate_json(report_path.read_bytes())
        details = report.details
        if not isinstance(details, PerformanceDetails):
            raise ValueError("耐久清理证据明细非法")
        run_ids: list[str] = []
        for sample in details.samples:
            if (
                sample.sample_id is None
                or sample.production_run_id is None
                or sample.job_id is None
            ):
                continue
            digest = hashlib.sha256(f"{durability_run_id}:{sample.sample_id}".encode()).hexdigest()
            bound_in_database = _database_run_is_owned(
                runtime,
                idempotency_key=f"durability-{digest[:40]}",
                run_id=sample.production_run_id,
                job_id=sample.job_id,
                media_sha256=sample.sample_sha256,
                terminal_status=sample.terminal_status or "FAILED",
            )
            if bound_in_database:
                run_ids.append(sample.production_run_id)
        return tuple(run_ids)
    except Exception:
        raise ValueError("耐久清理证据非法或损坏") from None


def _database_run_is_owned(
    runtime: Path,
    *,
    idempotency_key: str,
    run_id: str,
    job_id: str,
    media_sha256: str,
    terminal_status: str,
) -> bool:
    database = runtime / "video-demo.db"
    if not database.exists():
        return False
    _reject_symlink_path(runtime, database)
    if not database.is_file():
        raise ValueError("产品数据库不是普通文件")
    connection = sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            """
            SELECT run.run_id, asset.source_sha256, run.status, job.job_id, job.status
            FROM video_understanding_run AS run
            JOIN video_asset AS asset
              ON asset.tenant_id = run.tenant_id
             AND asset.application_id = run.application_id
             AND asset.knowledge_base_id = run.knowledge_base_id
             AND asset.asset_id = run.asset_id
            JOIN job
              ON job.tenant_id = run.tenant_id
             AND job.application_id = run.application_id
             AND job.knowledge_base_id = run.knowledge_base_id
             AND job.resource_id = run.run_id
            WHERE run.tenant_id = 'evaluation'
              AND run.application_id = 'video-demo'
              AND run.knowledge_base_id = 'evaluation'
              AND run.idempotency_key = ?
            """,
            (idempotency_key,),
        ).fetchall()
    finally:
        connection.close()
    if not rows:
        return False
    if len(rows) != 1:
        raise ValueError("产品运行幂等绑定不唯一")
    actual_run_id, actual_media, run_status, actual_job_id, job_status = rows[0]
    if (
        actual_run_id != run_id
        or actual_media != media_sha256
        or actual_job_id != job_id
        or run_status != terminal_status
    ):
        raise ValueError("产品运行与评测证据绑定不一致")
    if run_status in {"PENDING", "RUNNING"} or job_status in {
        "PENDING",
        "RUNNING",
        "RETRY_WAIT",
    }:
        raise ValueError("活跃产品运行不得清理")
    if _JOB_STATUS_BY_RUN_STATUS.get(run_status) != job_status:
        raise ValueError("产品运行与任务终态不一致")
    return True


def _deduplicate_paths(paths: Sequence[Path]) -> tuple[Path, ...]:
    return tuple(dict.fromkeys(paths))


def _assert_cleanup_target(runtime: Path, target: Path) -> None:
    _reject_symlink_path(runtime, target)
    metadata = target.lstat()
    if stat.S_ISLNK(metadata.st_mode):
        raise ValueError("评测清理目标不得是符号链接")
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError("评测清理目标必须是目录")
    for current, directories, files in os.walk(target, followlinks=False):
        for name in (*directories, *files):
            path = Path(current) / name
            if path.is_symlink():
                raise ValueError("评测清理子树不得包含符号链接")


def _contains_active_marker(target: Path) -> bool:
    return any(
        path.name in _ACTIVE_MARKERS or path.name.endswith(".incomplete")
        for path in target.rglob("*")
    )


def _cleanup_inventory(targets: tuple[Path, ...], workspace: Path) -> tuple[str, ...]:
    entries: list[str] = []
    for target in targets:
        entries.append(target.relative_to(workspace).as_posix())
        entries.extend(path.relative_to(workspace).as_posix() for path in sorted(target.rglob("*")))
    return tuple(entries)


def _atomic_write_json(
    path: Path,
    payload: object,
    *,
    workspace_root: Path,
) -> None:
    encoded = (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    _atomic_write_bytes(path, encoded, workspace_root=workspace_root)


def _atomic_write_bytes(
    path: Path,
    encoded: bytes,
    *,
    workspace_root: Path,
) -> None:
    workspace = workspace_root.resolve(strict=True)
    try:
        relative = path.absolute().relative_to(workspace)
    except ValueError:
        raise ValueError("汇总报告必须位于工作区内") from None
    if not relative.parts or relative.name in {"", ".", ".."}:
        raise ValueError("汇总报告目标路径非法")
    parent_descriptor = _open_workspace_parent(workspace, relative.parent)
    temporary = f".{relative.name}.{uuid.uuid4().hex}.part"
    descriptor: int | None = None
    published = False
    try:
        _validate_summary_leaf(parent_descriptor, relative.name)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(temporary, flags, 0o600, dir_fd=parent_descriptor)
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("汇总报告原子写入未推进")
            view = view[written:]
        os.fsync(descriptor)
        if os.fstat(descriptor).st_size != len(encoded):
            raise OSError("汇总报告写入大小不匹配")
        _assert_workspace_parent_current(
            workspace,
            relative.parent,
            parent_descriptor,
        )
        os.rename(
            temporary,
            relative.name,
            src_dir_fd=parent_descriptor,
            dst_dir_fd=parent_descriptor,
        )
        published = True
        os.fsync(parent_descriptor)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if not published:
            with suppress(FileNotFoundError):
                os.unlink(temporary, dir_fd=parent_descriptor)
        os.close(parent_descriptor)


def _open_workspace_parent(workspace: Path, relative_parent: Path) -> int:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(workspace, flags)
    try:
        for part in relative_parent.parts:
            if part in {"", ".", ".."}:
                raise ValueError("汇总报告父路径非法")
            with suppress(FileExistsError):
                os.mkdir(part, dir_fd=descriptor)
            child = os.open(part, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _validate_summary_leaf(parent_descriptor: int, filename: str) -> None:
    try:
        metadata = os.stat(filename, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError("汇总报告目标不得是符号链接或特殊文件")


def _assert_workspace_parent_current(
    workspace: Path,
    relative_parent: Path,
    held_descriptor: int,
) -> None:
    current = _open_workspace_parent(workspace, relative_parent)
    try:
        held = os.fstat(held_descriptor)
        observed = os.fstat(current)
        if (held.st_dev, held.st_ino) != (observed.st_dev, observed.st_ino):
            raise ValueError("汇总报告父目录身份已改变")
    finally:
        os.close(current)


def _safe_offline_output(payload: bytes, workspace_root: Path) -> bytes:
    text = payload.decode("utf-8", errors="replace")
    text = text.replace(str(workspace_root), ".")
    return text.encode("utf-8")[: 64 * 1024 * 1024]


def _final_reason(final: FinalGateReport) -> str | None:
    if final.status == GateStatus.PASS:
        return None
    failed = tuple(check.check_id for check in final.checks if check.status == GateStatus.FAIL)
    if failed or final.quality.status == GateStatus.FAIL:
        parts = (("quality",) if final.quality.status == GateStatus.FAIL else ()) + failed
        return f"失败门禁: {', '.join(parts)}"
    not_run = tuple(check.check_id for check in final.checks if check.status == GateStatus.NOT_RUN)
    parts = (("quality",) if final.quality.status == GateStatus.NOT_RUN else ()) + not_run
    return f"未运行门禁: {', '.join(parts)}"
