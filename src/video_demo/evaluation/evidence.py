from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import unicodedata
import uuid
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Annotated, Any, Literal, cast

from pydantic import (
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    ValidationError,
    field_validator,
    model_validator,
)

from video_demo.domain.base import FrozenModel, Sha256, StableId
from video_demo.domain.run import ModelIdentity
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.evaluation.dataset import ValidationLanguage
from video_demo.evaluation.report import GateStatus

if TYPE_CHECKING:
    from video_demo.config import Settings
    from video_demo.evaluation.gate import GateCheck

ArtifactRole = Literal[
    "INPUT_MEDIA",
    "OUTPUT_MEDIA",
    "PROVIDER_RESPONSE",
    "DATASET_MANIFEST",
    "AUTHORIZATION_RECORD",
    "PREDICTION_INDEX",
    "ANNOTATION",
    "SEMANTIC_JUDGMENT",
    "QUALITY_DETAIL",
    "PERFORMANCE_REPORT",
    "PRODUCTION_RESULT",
    "AUDIT_REPORT",
    "COMMAND_STDOUT",
    "COMMAND_STDERR",
]
LiveCheckId = Literal[
    "baidu_ocr_live",
    "qwen_live",
    "pyannote_live",
    "five_language_models",
]
OfflineCheckId = Literal[
    "no_indexing",
    "ruff",
    "mypy",
    "alembic_roundtrip",
    "openapi_contract",
    "secret_scan",
]

_MAX_MACHINE_BYTES = 64 * 1024 * 1024
_DEFAULT_MEDIA_BYTES = 4 * 1024 * 1024 * 1024
_CHUNK_BYTES = 1024 * 1024
_MAX_JSON_NESTING = 64
_MAX_COMMAND_JSON_FRAMES = _MAX_JSON_NESTING
_RUNTIME_RELATIVE_ROOT = PurePosixPath(".codex/video-rag-demo")
_REAL_MEDIA_INCOMPLETE_MARKER = ".real-media.incomplete"
_ARTIFACT_DIR_FD_FUNCTIONS = frozenset(
    (os.open, os.mkdir, os.stat, os.rename, os.unlink)
)
_REAL_MEDIA_COMMIT_RECORD = ".real-media.commit.json"
_LIVE_AUTHORITY_RELATIVE_ROOT = Path("eval/live-authority")
_LIVE_EXECUTED_CHECKS = frozenset(
    {
        "baidu_ocr_live",
        "qwen_live",
        "pyannote_live",
        "five_language_models",
    }
)
_MEDIA_ROLES = frozenset({"INPUT_MEDIA", "OUTPUT_MEDIA"})
_TEXT_MACHINE_ROLES = frozenset(
    {
        "PROVIDER_RESPONSE",
        "DATASET_MANIFEST",
        "AUTHORIZATION_RECORD",
        "PREDICTION_INDEX",
        "ANNOTATION",
        "SEMANTIC_JUDGMENT",
        "QUALITY_DETAIL",
        "PERFORMANCE_REPORT",
        "PRODUCTION_RESULT",
        "AUDIT_REPORT",
        "COMMAND_STDOUT",
        "COMMAND_STDERR",
    }
)
_JSON_MACHINE_ROLES = frozenset(
    {
        "PROVIDER_RESPONSE",
        "AUTHORIZATION_RECORD",
        "PREDICTION_INDEX",
        "ANNOTATION",
        "SEMANTIC_JUDGMENT",
        "QUALITY_DETAIL",
        "PERFORMANCE_REPORT",
        "PRODUCTION_RESULT",
        "AUDIT_REPORT",
    }
)
_JSONL_MACHINE_ROLES = frozenset({"DATASET_MANIFEST"})
_SECRET_PATTERN = re.compile(
    r"(?i)(?:authorization\s*[:=]\s*(?:bearer\s+)?\S+"
    r"|bearer\s+\S+"
    r"|[\"']?(?:api[ _-]?key|secret(?:[ _-]?key)?|token|password)"
    r"[\"']?\s*[:=]\s*[\"']?\S+)"
)
_DATA_URL_PATTERN = re.compile(r"(?i)(?<![a-z0-9_])data:[^,\s]*,")
_POSIX_ABSOLUTE_PATTERN = re.compile(
    r"(?:^|(?<=[\s\"'=:(]))/(?!/)[^\s\"']+"
)
_WINDOWS_ABSOLUTE_PATTERN = re.compile(
    r"(?i)(?:^|(?<=[\s\"'=:(]))[a-z]:[\\/]"
)
_UNC_ABSOLUTE_PATTERN = re.compile(
    r"(?:^|(?<=[\s\"'=:(]))\\{2,}[^\\\s]+\\+[^\\\s]+"
)
_FILE_URI_PATTERN = re.compile(r"(?i)file://")
_MYPY_DEV_NULL_ARGUMENT = "--cache-dir=/dev/null"
_JSON_NUMBER_PATTERN = re.compile(
    r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?"
)


class EvidenceKind(StrEnum):
    PYTEST_JUNIT = "PYTEST_JUNIT"
    PYTEST_COLLECTION = "PYTEST_COLLECTION"
    COMMAND_REPORT = "COMMAND_REPORT"
    STATIC_AUDIT = "STATIC_AUDIT"
    LIVE_SERVICE_REPORT = "LIVE_SERVICE_REPORT"
    PERFORMANCE_REPORT = "PERFORMANCE_REPORT"


class EvidenceLevel(StrEnum):
    CONTRACT = "CONTRACT"
    STATIC = "STATIC"
    REAL_MEDIA = "REAL_MEDIA"
    REAL_SERVICE = "REAL_SERVICE"
    PERFORMANCE = "PERFORMANCE"


class PreflightReasonCode(StrEnum):
    NO_INDEXING_INPUT_UNAVAILABLE = "NO_INDEXING_INPUT_UNAVAILABLE"
    RUFF_INPUT_UNAVAILABLE = "RUFF_INPUT_UNAVAILABLE"
    MYPY_INPUT_UNAVAILABLE = "MYPY_INPUT_UNAVAILABLE"
    ALEMBIC_INPUT_UNAVAILABLE = "ALEMBIC_INPUT_UNAVAILABLE"
    OPENAPI_INPUT_UNAVAILABLE = "OPENAPI_INPUT_UNAVAILABLE"
    SECRET_SCAN_INPUT_UNAVAILABLE = "SECRET_SCAN_INPUT_UNAVAILABLE"
    AUTHORIZED_DATASET_UNAVAILABLE = "AUTHORIZED_DATASET_UNAVAILABLE"
    REAL_MEDIA_CHAIN_UNAVAILABLE = "REAL_MEDIA_CHAIN_UNAVAILABLE"
    BAIDU_OCR_CREDENTIALS_UNAVAILABLE = "BAIDU_OCR_CREDENTIALS_UNAVAILABLE"
    QWEN_CREDENTIALS_UNAVAILABLE = "QWEN_CREDENTIALS_UNAVAILABLE"
    PYANNOTE_MODEL_UNAVAILABLE = "PYANNOTE_MODEL_UNAVAILABLE"
    FIVE_LANGUAGE_MODELS_UNAVAILABLE = "FIVE_LANGUAGE_MODELS_UNAVAILABLE"
    M1_DURABILITY_INPUT_UNAVAILABLE = "M1_DURABILITY_INPUT_UNAVAILABLE"


_PREFLIGHT_REASON_CHECK: dict[PreflightReasonCode, str] = {
    PreflightReasonCode.NO_INDEXING_INPUT_UNAVAILABLE: "no_indexing",
    PreflightReasonCode.RUFF_INPUT_UNAVAILABLE: "ruff",
    PreflightReasonCode.MYPY_INPUT_UNAVAILABLE: "mypy",
    PreflightReasonCode.ALEMBIC_INPUT_UNAVAILABLE: "alembic_roundtrip",
    PreflightReasonCode.OPENAPI_INPUT_UNAVAILABLE: "openapi_contract",
    PreflightReasonCode.SECRET_SCAN_INPUT_UNAVAILABLE: "secret_scan",
    PreflightReasonCode.AUTHORIZED_DATASET_UNAVAILABLE: "authorized_dataset",
    PreflightReasonCode.REAL_MEDIA_CHAIN_UNAVAILABLE: "real_media_chain",
    PreflightReasonCode.BAIDU_OCR_CREDENTIALS_UNAVAILABLE: "baidu_ocr_live",
    PreflightReasonCode.QWEN_CREDENTIALS_UNAVAILABLE: "qwen_live",
    PreflightReasonCode.PYANNOTE_MODEL_UNAVAILABLE: "pyannote_live",
    PreflightReasonCode.FIVE_LANGUAGE_MODELS_UNAVAILABLE: "five_language_models",
    PreflightReasonCode.M1_DURABILITY_INPUT_UNAVAILABLE: "m1_durability",
}


class EvidenceReference(FrozenModel):
    kind: EvidenceKind
    level: EvidenceLevel
    relative_path: str = Field(min_length=1, max_length=1024)
    sha256: Sha256
    covered_items: tuple[str, ...] = Field(min_length=1)
    summary: str = Field(min_length=1, max_length=500)

    @field_validator("relative_path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        _validate_workspace_relative(value)
        return value

    @model_validator(mode="after")
    def reject_unsafe_strings(self) -> EvidenceReference:
        _validate_persisted_value(self.model_dump(mode="python"))
        return self


class CommandTrace(FrozenModel):
    command: tuple[str, ...] = Field(min_length=1)
    exit_code: int
    stdout_sha256: Sha256
    stderr_sha256: Sha256

    @field_validator("command")
    @classmethod
    def reject_secret_arguments(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        serialized = " ".join(value).casefold()
        markers = (
            "api_key",
            "api-key",
            "secret_key",
            "secret-key",
            "authorization",
            "bearer ",
            "token=",
            "--token",
            "password",
        )
        if any(marker in serialized for marker in markers):
            raise ValueError("机器证据命令不得包含 Secret")
        return value


class TraceArtifact(FrozenModel):
    role: ArtifactRole
    relative_path: str = Field(min_length=1, max_length=1024)
    sha256: Sha256
    max_bytes: int | None = Field(default=None, gt=0, le=_DEFAULT_MEDIA_BYTES)

    @field_validator("relative_path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        _validate_workspace_relative(value)
        _require_runtime_relative(value)
        return value

    @field_validator("max_bytes", mode="before")
    @classmethod
    def validate_max_bytes(cls, value: Any) -> Any:
        if value is not None and type(value) is not int:
            raise ValueError("产物大小上限必须是正整数")
        return value

    @model_validator(mode="after")
    def enforce_role_limit(self) -> TraceArtifact:
        if self.max_bytes is not None:
            _artifact_limit(self.role, self.max_bytes)
        return self


class CommandEvidenceDetails(FrozenModel):
    type: Literal["COMMAND"]
    trace: CommandTrace


class StaticAuditDetails(FrozenModel):
    type: Literal["STATIC_AUDIT"]
    trace: CommandTrace
    audited_paths: tuple[str, ...] = Field(min_length=1)
    violation_count: int = Field(ge=0)


_OFFLINE_COMMANDS: dict[OfflineCheckId, tuple[str, ...]] = {
    "no_indexing": (
        "python",
        "-m",
        "video_demo.evaluation.final_runner",
        "no_indexing",
    ),
    "ruff": ("python", "-m", "ruff", "check", "--no-cache", "src", "tests"),
    "mypy": (
        "python",
        "-m",
        "mypy",
        "--no-incremental",
        "--cache-dir=/dev/null",
        "src",
    ),
    "alembic_roundtrip": (
        "python",
        "-m",
        "video_demo.evaluation.final_runner",
        "alembic_roundtrip",
    ),
    "openapi_contract": (
        "python",
        "-m",
        "video_demo.evaluation.final_runner",
        "openapi_contract",
    ),
    "secret_scan": (
        "python",
        "-m",
        "video_demo.evaluation.final_runner",
        "secret_scan",
    ),
}

_OFFLINE_AUDITED_PATHS: dict[OfflineCheckId, tuple[str, ...]] = {
    "no_indexing": ("src", "pyproject.toml", "uv.lock"),
    "ruff": ("src", "tests", "pyproject.toml"),
    "mypy": ("src", "pyproject.toml"),
    "alembic_roundtrip": (
        "src/video_demo/persistence",
        "migrations",
        "alembic.ini",
    ),
    "openapi_contract": ("src/video_demo", "pyproject.toml"),
    "secret_scan": (
        "src",
        "migrations",
        "pyproject.toml",
        "alembic.ini",
        ".env.example",
        "README.md",
    ),
}


def offline_command(check_id: OfflineCheckId) -> tuple[str, ...]:
    return _OFFLINE_COMMANDS[check_id]


def offline_audited_paths(check_id: OfflineCheckId) -> tuple[str, ...]:
    return _OFFLINE_AUDITED_PATHS[check_id]


def offline_observation_sha256(observations: tuple[str, ...]) -> str:
    encoded = json.dumps(
        observations,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class OfflineRawReport(FrozenModel):
    schema_version: Literal["1.0.0"]
    check_id: OfflineCheckId
    evaluation_run_id: StableId
    status: GateStatus
    input_sha256: Sha256
    command: tuple[str, ...]
    exit_code: int
    stdout_sha256: Sha256
    stderr_sha256: Sha256
    audited_paths: tuple[str, ...] = Field(min_length=1)
    violation_count: int = Field(ge=0)
    observations: tuple[str, ...]
    observation_sha256: Sha256

    @model_validator(mode="after")
    def validate_offline_binding(self) -> OfflineRawReport:
        if self.command != offline_command(self.check_id):
            raise ValueError("离线检查命令不是固定白名单命令")
        if self.audited_paths != offline_audited_paths(self.check_id):
            raise ValueError("离线检查范围不是固定审计范围")
        expected_status = (
            GateStatus.PASS
            if self.exit_code == 0 and self.violation_count == 0
            else GateStatus.FAIL
        )
        if self.status != expected_status:
            raise ValueError("离线检查状态与原始结果不一致")
        if self.observation_sha256 != offline_observation_sha256(self.observations):
            raise ValueError("离线检查观察摘要不匹配")
        return self


class OfflineEvidenceDetails(FrozenModel):
    type: Literal["OFFLINE"]
    trace: CommandTrace
    raw_report_sha256: Sha256
    input_sha256: Sha256
    observation_sha256: Sha256


class LiveServiceDetails(FrozenModel):
    type: Literal["LIVE_SERVICE"]
    trace: CommandTrace
    service: str = Field(min_length=2, max_length=64)
    model_id: str = Field(min_length=1, max_length=256)
    request_id_sha256: Sha256
    input_sha256: Sha256
    output_sha256: Sha256
    http_status: int = Field(ge=100, le=599)


class _LiveCheckDetails(FrozenModel):
    trace: CommandTrace
    raw_report_sha256: Sha256
    implementation_sha256: Sha256
    settings_fingerprint: Sha256
    dataset_sha256: Sha256
    authorization_sha256: Sha256


class BaiduLiveDetails(_LiveCheckDetails):
    type: Literal["BAIDU_LIVE"]


class QwenLiveDetails(_LiveCheckDetails):
    type: Literal["QWEN_LIVE"]


class PyannoteLiveDetails(_LiveCheckDetails):
    type: Literal["PYANNOTE_LIVE"]


class FiveLanguageModelsDetails(_LiveCheckDetails):
    type: Literal["FIVE_LANGUAGE_MODELS"]


class RealMediaDetails(FrozenModel):
    type: Literal["REAL_MEDIA"]
    trace: CommandTrace
    ffmpeg_version: str | None = Field(default=None, min_length=1, max_length=256)
    ffprobe_version: str | None = Field(default=None, min_length=1, max_length=256)
    raw_report_sha256: Sha256
    implementation_sha256: Sha256


class AuthorizedDatasetDetails(FrozenModel):
    type: Literal["AUTHORIZED_DATASET"]
    trace: CommandTrace
    manifest_sha256: Sha256
    authorization_record_sha256: Sha256
    item_count: int = Field(ge=30)
    language_counts: dict[Literal["zh", "en", "ja", "ko", "es"], int]
    max_video_bytes: int = Field(gt=0, le=_DEFAULT_MEDIA_BYTES)

    @field_validator("max_video_bytes", mode="before")
    @classmethod
    def reject_bool_limit(cls, value: Any) -> Any:
        if type(value) is not int:
            raise ValueError("媒体大小上限必须是正整数")
        return value

    @field_validator("language_counts")
    @classmethod
    def validate_language_counts(
        cls,
        value: dict[Literal["zh", "en", "ja", "ko", "es"], int],
    ) -> dict[Literal["zh", "en", "ja", "ko", "es"], int]:
        if set(value) != {"zh", "en", "ja", "ko", "es"} or any(
            count < 6 for count in value.values()
        ):
            raise ValueError("授权数据集必须覆盖五语且每语不少于 6 条")
        return value


class PerformanceSampleDetails(FrozenModel):
    sample_id: StableId | None = None
    media_relative_path: str | None = Field(default=None, min_length=1, max_length=1024)
    sample_sha256: Sha256
    authorization_id: StableId | None = None
    duration_ms: int = Field(gt=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    elapsed_seconds: float = Field(ge=0, allow_inf_nan=False)
    rtf: float = Field(ge=0, allow_inf_nan=False)
    oom_detected: bool
    peak_concurrency: int = Field(ge=0)
    outside_workspace_write_count: int = Field(ge=0)
    peak_rss_bytes: int = Field(ge=0)
    peak_disk_bytes: int = Field(ge=0)
    succeeded: bool
    terminal_status: str | None = Field(default=None, min_length=1, max_length=64)
    failure_code: str | None = Field(default=None, min_length=3, max_length=128)
    production_run_id: StableId | None = None
    job_id: StableId | None = None
    result_manifest_relative_path: str | None = Field(default=None, min_length=1, max_length=1024)
    result_manifest_sha256: Sha256 | None = None
    probe_report_sha256: Sha256 | None = None

    @model_validator(mode="after")
    def bind_success_artifacts(self) -> PerformanceSampleDetails:
        legacy_binding = (
            self.sample_id,
            self.media_relative_path,
            self.authorization_id,
            self.terminal_status,
            self.probe_report_sha256,
        )
        if any(value is None for value in legacy_binding):
            return self
        execution_identity = (
            self.production_run_id,
            self.job_id,
        )
        result_binding = (
            self.result_manifest_relative_path,
            self.result_manifest_sha256,
        )
        if sum(value is not None for value in execution_identity) == 1:
            raise ValueError("耐久样本的生产 run/job 必须同时绑定")
        if sum(value is not None for value in result_binding) == 1:
            raise ValueError("耐久样本的生产结果路径与摘要必须同时绑定")
        if any(value is not None for value in result_binding) and any(
            value is None for value in execution_identity
        ):
            raise ValueError("生产结果 Manifest 必须绑定对应 run/job")
        if not self.succeeded and any(value is not None for value in result_binding):
            raise ValueError("失败耐久样本不得伪造生产结果 Manifest")
        return self


class PerformanceDetails(FrozenModel):
    type: Literal["PERFORMANCE"]
    trace: CommandTrace
    performance_report_sha256: Sha256
    evaluation_run_id: StableId | None = None
    manifest_sha256: Sha256 | None = None
    authorization_sha256: Sha256 | None = None
    implementation_sha256: Sha256 | None = None
    settings_fingerprint: Sha256 | None = None
    sample_report_sha256s: tuple[Sha256, Sha256] | None = None
    samples: tuple[PerformanceSampleDetails, PerformanceSampleDetails]

    @field_validator("samples")
    @classmethod
    def reject_duplicate_samples(
        cls, value: tuple[PerformanceSampleDetails, ...]
    ) -> tuple[PerformanceSampleDetails, ...]:
        digests = tuple(sample.sample_sha256 for sample in value)
        if len(digests) != len(set(digests)):
            raise ValueError("耐久样本摘要不得重复")
        return value


class PreflightDetails(FrozenModel):
    type: Literal["PREFLIGHT"]
    trace: CommandTrace
    preflight_report_sha256: Sha256

    @model_validator(mode="after")
    def require_successful_preflight_trace(self) -> PreflightDetails:
        if self.trace.exit_code != 0:
            raise ValueError("preflight 命令必须成功完成前置条件检查")
        return self


MachineEvidenceDetails = Annotated[
    CommandEvidenceDetails
    | StaticAuditDetails
    | OfflineEvidenceDetails
    | LiveServiceDetails
    | BaiduLiveDetails
    | QwenLiveDetails
    | PyannoteLiveDetails
    | FiveLanguageModelsDetails
    | RealMediaDetails
    | AuthorizedDatasetDetails
    | PerformanceDetails
    | PreflightDetails,
    Field(discriminator="type"),
]


class MachineEvidenceReport(FrozenModel):
    schema_version: Literal["1.0.0"]
    check_id: str = Field(min_length=3, max_length=128)
    status: GateStatus
    kind: EvidenceKind
    level: EvidenceLevel
    covered_items: tuple[str, ...] = Field(min_length=1)
    summary: str = Field(min_length=1, max_length=500)
    producer: str = Field(min_length=1, max_length=128)
    started_at: datetime
    finished_at: datetime
    not_run_reason: str | None = Field(default=None, min_length=1, max_length=500)
    artifacts: tuple[TraceArtifact, ...] = ()
    details: MachineEvidenceDetails

    @model_validator(mode="after")
    def validate_report(self) -> MachineEvidenceReport:
        if self.kind in (EvidenceKind.PYTEST_JUNIT, EvidenceKind.PYTEST_COLLECTION):
            raise ValueError("pytest 证据必须直接解析原始文件")
        if self.finished_at < self.started_at:
            raise ValueError("机器证据结束时间不得早于开始时间")
        is_preflight = isinstance(self.details, PreflightDetails)
        if self.status == GateStatus.NOT_RUN and not is_preflight:
            raise ValueError("NOT_RUN 机器证据必须使用结构化 preflight")
        if is_preflight and self.status != GateStatus.NOT_RUN:
            raise ValueError("preflight 机器证据只能派生 NOT_RUN")
        if self.status == GateStatus.NOT_RUN and self.not_run_reason is None:
            raise ValueError("NOT_RUN 机器证据必须提供稳定原因")
        if self.status != GateStatus.NOT_RUN and self.not_run_reason is not None:
            raise ValueError("已运行机器证据不得包含 NOT_RUN 原因")
        paths = tuple(artifact.relative_path for artifact in self.artifacts)
        if len(paths) != len(set(paths)):
            raise ValueError("机器证据产物路径不得重复")
        if len(self.covered_items) != len(set(self.covered_items)):
            raise ValueError("机器证据覆盖项不得重复")
        _validate_persisted_value(self.model_dump(mode="python"))
        return self


class PerformanceRawReport(FrozenModel):
    schema_version: Literal["1.0.0"]
    evaluation_run_id: StableId
    manifest_sha256: Sha256
    authorization_sha256: Sha256
    implementation_sha256: Sha256
    settings_fingerprint: Sha256
    worker_concurrency: Literal[1]
    inference_device: Literal["cpu"]
    whisper_compute_type: Literal["int8"]
    sample_report_sha256s: tuple[Sha256, Sha256]
    samples: tuple[PerformanceSampleDetails, PerformanceSampleDetails]


class PerformanceSampleRawReport(FrozenModel):
    schema_version: Literal["1.0.0"]
    evaluation_run_id: StableId
    sample: PerformanceSampleDetails


_REAL_MEDIA_CASE_IDS: tuple[str, ...] = (
    "normal_audio",
    "no_audio",
    "rotation",
    "vfr",
)
_REAL_MEDIA_PREFLIGHT_CODES: tuple[ErrorCode, ...] = (
    ErrorCode.VIDEO_FFMPEG_UNAVAILABLE,
    ErrorCode.VIDEO_FFPROBE_UNAVAILABLE,
    ErrorCode.VISUAL_DEPENDENCY_UNAVAILABLE,
)
_LIVE_PREFLIGHT_CODES: dict[str, tuple[ErrorCode, ...]] = {
    "baidu_ocr_live": (
        ErrorCode.BAIDU_API_KEY_UNAVAILABLE,
        ErrorCode.BAIDU_SECRET_KEY_UNAVAILABLE,
        ErrorCode.LIVE_AUTHORIZED_KEYFRAME_UNAVAILABLE,
    ),
    "qwen_live": (
        ErrorCode.QWEN_ENDPOINT_UNAVAILABLE,
        ErrorCode.QWEN_API_KEY_UNAVAILABLE,
        ErrorCode.QWEN_MODEL_ID_UNAVAILABLE,
        ErrorCode.LIVE_AUTHORIZED_CLIP_UNAVAILABLE,
    ),
    "pyannote_live": (
        ErrorCode.PYANNOTE_TOKEN_UNAVAILABLE,
        ErrorCode.PYANNOTE_TERMS_UNAVAILABLE,
        ErrorCode.PYANNOTE_DEPENDENCY_UNAVAILABLE,
        ErrorCode.PYANNOTE_MODEL_UNAVAILABLE,
        ErrorCode.LIVE_AUTHORIZED_AUDIO_UNAVAILABLE,
    ),
    "five_language_models": (
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
}
_DURABILITY_PREFLIGHT_CODES: tuple[ErrorCode, ...] = (
    ErrorCode.M1_SAMPLE_COUNT_INVALID,
    ErrorCode.M1_DURATION_TOO_SHORT,
    ErrorCode.M1_RESOLUTION_TOO_SMALL,
    ErrorCode.M1_AUTHORIZATION_UNAVAILABLE,
    ErrorCode.M1_MEDIA_INVALID,
    ErrorCode.M1_PROBE_MISMATCH,
    ErrorCode.INVALID_CONFIGURATION,
    ErrorCode.M1_PSUTIL_UNAVAILABLE,
    ErrorCode.VIDEO_FFMPEG_UNAVAILABLE,
    ErrorCode.VIDEO_FFPROBE_UNAVAILABLE,
    ErrorCode.VIDEO_BINARY_PROBE_FAILED,
    ErrorCode.VISUAL_DEPENDENCY_UNAVAILABLE,
    ErrorCode.SILERO_DEPENDENCY_UNAVAILABLE,
    ErrorCode.SILERO_MODEL_UNAVAILABLE,
    ErrorCode.FASTER_WHISPER_DEPENDENCY_UNAVAILABLE,
    ErrorCode.FASTER_WHISPER_MODEL_UNAVAILABLE,
    ErrorCode.WHISPERX_DEPENDENCY_UNAVAILABLE,
    ErrorCode.WHISPERX_MODEL_UNAVAILABLE,
    ErrorCode.PYANNOTE_DEPENDENCY_UNAVAILABLE,
    ErrorCode.PYANNOTE_TERMS_UNAVAILABLE,
    ErrorCode.PYANNOTE_MODEL_UNAVAILABLE,
    ErrorCode.YAMNET_DEPENDENCY_UNAVAILABLE,
    ErrorCode.YAMNET_MODEL_UNAVAILABLE,
    ErrorCode.QWEN_ENDPOINT_UNAVAILABLE,
    ErrorCode.QWEN_API_KEY_UNAVAILABLE,
    ErrorCode.QWEN_MODEL_ID_UNAVAILABLE,
    ErrorCode.BAIDU_API_KEY_UNAVAILABLE,
    ErrorCode.BAIDU_SECRET_KEY_UNAVAILABLE,
    ErrorCode.PYANNOTE_TOKEN_UNAVAILABLE,
)
_REAL_MEDIA_PHASES: tuple[str, ...] = (
    "generate",
    "probe",
    "proxy",
    "audio",
    "opencv_decode",
    "scene_detect",
    "keyframe_select",
)
_REAL_MEDIA_PHASE_EXECUTABLES: dict[str, str] = {
    "generate": "ffmpeg",
    "probe": "ffprobe",
    "proxy": "FFmpegTranscoder",
    "audio": "FFmpegTranscoder",
    "opencv_decode": "OpenCvFrameExtractor",
    "scene_detect": "PySceneDetectAdapter",
    "keyframe_select": "KeyframeSelector",
}


class PreflightIssue(FrozenModel):
    code: ErrorCode


LiveInputKind = Literal["SOURCE_MEDIA", "AUDIO", "KEYFRAME", "CLIP"]
LiveComponent = Literal[
    "baidu_ocr",
    "qwen",
    "pyannote",
    "silero_vad",
    "faster_whisper",
    "whisperx",
    "yamnet",
]
LiveFailureComponent = Literal[
    "baidu_ocr",
    "qwen",
    "pyannote",
    "silero_vad",
    "faster_whisper",
    "whisperx",
    "yamnet",
    "components_close",
]
LiveOperation = Literal[
    "recognize",
    "capability_probe",
    "understand_segment",
    "diarize",
    "vad",
    "transcribe",
    "align",
    "detect",
]
LiveCapability = Literal["video_input", "strict_json_schema"]


class LiveInputArtifact(FrozenModel):
    model_config = ConfigDict(revalidate_instances="always")

    kind: LiveInputKind
    sample_id: StableId
    relative_path: str = Field(min_length=1, max_length=1024)
    sha256: Sha256
    source_media_sha256: Sha256
    size_bytes: StrictInt = Field(gt=0, le=_DEFAULT_MEDIA_BYTES)

    @field_validator("relative_path")
    @classmethod
    def validate_runtime_path(cls, value: str) -> str:
        _validate_workspace_relative(value)
        _require_runtime_relative(value)
        return value


class LiveSample(FrozenModel):
    model_config = ConfigDict(revalidate_instances="always")

    sample_id: StableId
    language: ValidationLanguage
    duration_ms: StrictInt = Field(gt=0, le=1_800_000)
    source_media_relative_path: str = Field(min_length=1, max_length=1024)
    source_media_sha256: Sha256
    audio_relative_path: str = Field(min_length=1, max_length=1024)
    audio_sha256: Sha256
    keyframe_relative_path: str = Field(min_length=1, max_length=1024)
    keyframe_sha256: Sha256
    clip_relative_path: str = Field(min_length=1, max_length=1024)
    clip_sha256: Sha256
    annotation_sha256: Sha256

    @field_validator(
        "source_media_relative_path",
        "audio_relative_path",
        "keyframe_relative_path",
        "clip_relative_path",
    )
    @classmethod
    def validate_runtime_paths(cls, value: str) -> str:
        _validate_workspace_relative(value)
        _require_runtime_relative(value)
        return value

    @model_validator(mode="after")
    def require_distinct_paths(self) -> LiveSample:
        paths = (
            self.source_media_relative_path,
            self.audio_relative_path,
            self.keyframe_relative_path,
            self.clip_relative_path,
        )
        if len(paths) != len(set(paths)):
            raise ValueError("live 样本源媒体和派生产物路径不得重复")
        return self


_COMPONENT_OPERATION: dict[str, tuple[str, ...]] = {
    "baidu_ocr": ("recognize",),
    "qwen": ("capability_probe", "understand_segment"),
    "pyannote": ("diarize",),
    "silero_vad": ("vad",),
    "faster_whisper": ("transcribe",),
    "whisperx": ("align",),
    "yamnet": ("detect",),
}
_COMPONENT_INPUT_KIND: dict[str, str] = {
    "baidu_ocr": "KEYFRAME",
    "qwen": "CLIP",
    "pyannote": "AUDIO",
    "silero_vad": "AUDIO",
    "faster_whisper": "AUDIO",
    "whisperx": "AUDIO",
    "yamnet": "AUDIO",
}
_COMPONENT_PROVIDER: dict[str, str] = {
    "baidu_ocr": "baidu_ocr",
    "qwen": "qwen",
    "pyannote": "local",
    "silero_vad": "local",
    "faster_whisper": "local",
    "whisperx": "local",
    "yamnet": "local",
}
_FIXED_MODEL_IDS: dict[str, str] = {
    "baidu_ocr": "accurate_basic",
    "pyannote": "pyannote/speaker-diarization-community-1",
    "silero_vad": "silero-vad",
    "faster_whisper": "large-v3",
    "yamnet": "yamnet",
}
_QWEN_MODEL_ID_PATTERN = re.compile(
    r"qwen(?:2(?:\.5)?|3)-vl-(?:plus|max|flash)"
    r"(?:-[0-9]{4}-[0-9]{2}-[0-9]{2})?\Z",
)
_MODEL_REVISION_PATTERN = re.compile(
    r"(?:[0-9a-f]{7,64}|v?[0-9]+(?:\.[0-9]+){1,3})\Z",
)
_WHISPERX_MODEL_IDS = frozenset(
    f"whisperx-align-{language}" for language in ("zh", "en", "ja", "ko", "es")
)


class ModelExecutionFact(FrozenModel):
    model_config = ConfigDict(revalidate_instances="always")

    component: LiveComponent
    operation: LiveOperation
    evaluation_run_id: StableId
    model: ModelIdentity
    sample_id: StableId
    language: ValidationLanguage | None = None
    input_kind: LiveInputKind
    input_sha256: Sha256
    output_sha256: Sha256
    request_id_sha256: Sha256 | None = None
    http_status: StrictInt | None = Field(default=None, ge=100, le=599)
    capabilities: tuple[LiveCapability, ...] = ()

    @model_validator(mode="after")
    def bind_component_facts(self) -> ModelExecutionFact:
        service_component = self.component in {"baidu_ocr", "qwen"}
        if (
            self.model.component != self.component
            or self.operation not in _COMPONENT_OPERATION[self.component]
            or self.input_kind != _COMPONENT_INPUT_KIND[self.component]
        ):
            raise ValueError("模型执行事实与组件身份不匹配")
        if service_component != (self.request_id_sha256 is not None):
            raise ValueError("远程服务执行事实必须绑定 request ID 摘要")
        if service_component != (self.http_status is not None):
            raise ValueError("远程服务执行事实必须绑定 HTTP 状态")
        if service_component and (
            self.http_status is None or not 200 <= self.http_status < 300
        ):
            raise ValueError("远程服务执行事实只能表示成功的 2xx 阶段")
        self._validate_model_identity()
        if self.component in {"faster_whisper", "whisperx"} and self.language is None:
            raise ValueError("五语模型执行事实必须声明语言")
        if self.component != "qwen" and self.capabilities:
            raise ValueError("仅 Qwen 能力探测可声明能力事实")
        if self.component == "qwen" and self.operation != "capability_probe" and self.capabilities:
            raise ValueError("仅 Qwen 能力探测可声明能力事实")
        if self.component == "qwen" and self.operation == "capability_probe" and set(
            self.capabilities
        ) != {"video_input", "strict_json_schema"}:
            raise ValueError("Qwen 能力探测事实必须完整证明视频输入与严格 Schema")
        if len(self.capabilities) != len(set(self.capabilities)):
            raise ValueError("模型能力事实不得重复")
        _validate_persisted_value(self.model_dump(mode="python"))
        return self

    def _validate_model_identity(self) -> None:
        if self.model.provider != _COMPONENT_PROVIDER[self.component]:
            raise ValueError("模型供应方与 live 组件不匹配")
        if self.model.revision is not None and not _MODEL_REVISION_PATTERN.fullmatch(
            self.model.revision
        ):
            raise ValueError("模型 revision 必须是受限 ASCII 标识")
        if self.component in _FIXED_MODEL_IDS:
            valid_model = self.model.model_id == _FIXED_MODEL_IDS[self.component]
        elif self.component == "whisperx":
            valid_model = (
                self.model.model_id in _WHISPERX_MODEL_IDS
                and self.language is not None
                and self.model.model_id == f"whisperx-align-{self.language}"
            )
        else:
            valid_model = bool(_QWEN_MODEL_ID_PATTERN.fullmatch(self.model.model_id))
        if not valid_model:
            raise ValueError("模型 ID 与 live 组件不匹配")
        if self.component in {"baidu_ocr", "qwen"}:
            if self.model.device is not None or self.model.revision is not None:
                raise ValueError("远程服务模型身份不得声明本地设备或推测 revision")
        elif self.model.device not in {"cpu", "mps"}:
            raise ValueError("本地模型身份必须声明受支持设备")


class LiveExecutionSummary(FrozenModel):
    """不含供应商正文、可由执行事实独立重验的结构摘要。"""

    model_config = ConfigDict(revalidate_instances="always")

    schema_version: Literal["1.0.0"]
    component: LiveComponent
    operation: LiveOperation
    evaluation_run_id: StableId
    model: ModelIdentity
    sample_id: StableId
    language: ValidationLanguage | None = None
    input_kind: LiveInputKind
    input_sha256: Sha256
    request_id_sha256: Sha256 | None = None
    http_status: StrictInt | None = Field(default=None, ge=100, le=599)
    capabilities: tuple[LiveCapability, ...] = ()
    output_item_count: StrictInt = Field(ge=0)

    @model_validator(mode="after")
    def bind_execution_facts(self) -> LiveExecutionSummary:
        ModelExecutionFact(
            component=self.component,
            operation=self.operation,
            evaluation_run_id=self.evaluation_run_id,
            model=self.model,
            sample_id=self.sample_id,
            language=self.language,
            input_kind=self.input_kind,
            input_sha256=self.input_sha256,
            output_sha256="0" * 64,
            request_id_sha256=self.request_id_sha256,
            http_status=self.http_status,
            capabilities=self.capabilities,
        )
        _validate_persisted_value(self.model_dump(mode="python"))
        return self


class _LiveRawReport(FrozenModel):
    model_config = ConfigDict(revalidate_instances="always")

    schema_version: Literal["1.0.0"]
    check_id: str = Field(min_length=3, max_length=128)
    status: Literal[GateStatus.PASS, GateStatus.FAIL]
    execution_started: StrictBool
    evaluation_run_id: StableId
    dataset_sha256: Sha256
    authorization_sha256: Sha256
    settings_fingerprint: Sha256
    implementation_sha256: Sha256
    inputs: tuple[LiveInputArtifact, ...] = Field(min_length=4)
    executions: tuple[ModelExecutionFact, ...]
    failure_code: ErrorCode | None = None
    failure_component: LiveFailureComponent | None = None

    @model_validator(mode="after")
    def validate_execution_status(self) -> _LiveRawReport:
        if not self.execution_started:
            raise ValueError("真实服务 raw 只能在执行开始后形成")
        if self.status == GateStatus.PASS:
            if (
                not self.executions
                or self.failure_code is not None
                or self.failure_component is not None
            ):
                raise ValueError("真实服务 PASS 必须有执行事实且不得有失败事实")
        elif self.failure_code is None or self.failure_component is None:
            raise ValueError("真实服务 FAIL 必须声明稳定失败码与组件")
        elif (
            self.failure_component == "components_close"
            and self.failure_code != ErrorCode.SYSTEM_FAILURE
        ):
            raise ValueError("组件关闭失败只能声明 SYSTEM_FAILURE")
        identities = tuple(
            (fact.component, fact.operation, fact.sample_id, fact.language)
            for fact in self.executions
        )
        if len(identities) != len(set(identities)):
            raise ValueError("真实服务执行事实不得重复")
        if any(
            fact.evaluation_run_id != self.evaluation_run_id
            for fact in self.executions
        ):
            raise ValueError("真实服务执行事实必须绑定当前评测 run")
        _validate_persisted_value(self.model_dump(mode="python"))
        return self

    def _validate_sample_inputs(self, samples: tuple[LiveSample, ...]) -> None:
        expected_count = len(samples) * 4
        if len(self.inputs) != expected_count:
            raise ValueError("live raw 必须为每个样本绑定四类精确输入")
        expected_by_identity = {
            (sample.sample_id, kind): expected
            for sample in samples
            for kind, expected in _expected_live_inputs(sample).items()
        }
        actual_by_identity = {
            (item.sample_id, item.kind): item for item in self.inputs
        }
        if (
            len(actual_by_identity) != expected_count
            or set(actual_by_identity) != set(expected_by_identity)
            or any(
                (
                    actual_by_identity[identity].relative_path,
                    actual_by_identity[identity].sha256,
                    actual_by_identity[identity].source_media_sha256,
                )
                != expected
                for identity, expected in expected_by_identity.items()
            )
        ):
            raise ValueError("live 输入未与样本路径、摘要和源媒体摘要精确绑定")
        input_paths = tuple(item.relative_path for item in self.inputs)
        kind_digests = tuple((item.kind, item.sha256) for item in self.inputs)
        if len(input_paths) != len(set(input_paths)) or len(kind_digests) != len(
            set(kind_digests)
        ):
            raise ValueError("同一 live 输入不得跨样本复用")


def _expected_live_inputs(sample: LiveSample) -> dict[LiveInputKind, tuple[str, Sha256, Sha256]]:
    values: dict[LiveInputKind, tuple[str, Sha256]] = {
        "SOURCE_MEDIA": (
            sample.source_media_relative_path,
            sample.source_media_sha256,
        ),
        "AUDIO": (sample.audio_relative_path, sample.audio_sha256),
        "KEYFRAME": (sample.keyframe_relative_path, sample.keyframe_sha256),
        "CLIP": (sample.clip_relative_path, sample.clip_sha256),
    }
    return {
        kind: (relative_path, sha256, sample.source_media_sha256)
        for kind, (relative_path, sha256) in values.items()
    }


class BaiduLiveRawReport(_LiveRawReport):
    check_id: Literal["baidu_ocr_live"]
    sample: LiveSample

    @model_validator(mode="after")
    def validate_baidu_execution(self) -> BaiduLiveRawReport:
        self._validate_sample_inputs((self.sample,))
        if self.status == GateStatus.FAIL:
            if self.failure_component == "components_close":
                if len(self.executions) != 1:
                    raise ValueError("百度组件关闭失败前执行事实不完整")
            elif self.failure_component != "baidu_ocr":
                raise ValueError("百度真实失败组件身份不匹配")
        if any(
            fact.component != "baidu_ocr"
            or fact.sample_id != self.sample.sample_id
            or fact.language != self.sample.language
            or fact.input_sha256 != self.sample.keyframe_sha256
            for fact in self.executions
        ):
            raise ValueError("百度执行事实与当前授权样本不匹配")
        if self.status == GateStatus.PASS and (
            len(self.executions) != 1
            or (fact := self.executions[0]).component != "baidu_ocr"
            or fact.operation != "recognize"
            or fact.sample_id != self.sample.sample_id
            or fact.language != self.sample.language
            or fact.input_sha256 != self.sample.keyframe_sha256
            or fact.http_status is None
            or not 200 <= fact.http_status < 300
        ):
            raise ValueError("百度真实执行事实不完整")
        return self


class QwenLiveRawReport(_LiveRawReport):
    check_id: Literal["qwen_live"]
    sample: LiveSample

    @model_validator(mode="after")
    def validate_qwen_execution(self) -> QwenLiveRawReport:
        self._validate_sample_inputs((self.sample,))
        operations = tuple(fact.operation for fact in self.executions)
        expected_operations = ("capability_probe", "understand_segment")
        if self.status == GateStatus.FAIL:
            if self.failure_component == "components_close":
                if operations != expected_operations:
                    raise ValueError("Qwen 组件关闭失败前执行事实不完整")
            elif self.failure_component != "qwen":
                raise ValueError("Qwen 真实失败组件身份不匹配")
        if (
            operations != expected_operations[: len(operations)]
            or any(
                fact.component != "qwen"
                or fact.sample_id != self.sample.sample_id
                or fact.language != self.sample.language
                or fact.input_sha256 != self.sample.clip_sha256
                for fact in self.executions
            )
        ):
            raise ValueError("Qwen 执行事实与当前授权 clip 或阶段顺序不匹配")
        if any(
            fact.http_status is None or not 200 <= fact.http_status < 300
            for fact in self.executions
        ):
            raise ValueError("Qwen 阶段事实必须绑定 2xx 成功 HTTP 状态")
        if self.executions and (
            set(self.executions[0].capabilities)
            != {"video_input", "strict_json_schema"}
        ):
            raise ValueError("Qwen 能力探测事实必须完整证明视频输入与严格 Schema")
        if self.status == GateStatus.PASS:
            if operations != expected_operations:
                raise ValueError("Qwen 必须先能力探测再执行 segment Schema")
            if any(
                fact.component != "qwen"
                or fact.sample_id != self.sample.sample_id
                or fact.language != self.sample.language
                or fact.input_sha256 != self.sample.clip_sha256
                or fact.http_status is None
                or not 200 <= fact.http_status < 300
                for fact in self.executions
            ):
                raise ValueError("Qwen 真实执行事实与授权 clip 不匹配")
            probe, segment = self.executions
            if segment.capabilities or probe.model != segment.model:
                raise ValueError("Qwen 能力探测与 segment 事实不匹配")
        elif len(self.executions) == 2:
            probe, segment = self.executions
            if (
                segment.capabilities or probe.model != segment.model
            ):
                raise ValueError("Qwen FAIL 的 segment 前必须已有完整能力探测事实")
        return self


class PyannoteLiveRawReport(_LiveRawReport):
    check_id: Literal["pyannote_live"]
    sample: LiveSample

    @model_validator(mode="after")
    def validate_pyannote_execution(self) -> PyannoteLiveRawReport:
        self._validate_sample_inputs((self.sample,))
        if self.status == GateStatus.FAIL:
            if self.failure_component == "components_close":
                if len(self.executions) != 1:
                    raise ValueError("pyannote 组件关闭失败前执行事实不完整")
            elif self.failure_component != "pyannote":
                raise ValueError("pyannote 真实失败组件身份不匹配")
        if any(
            fact.component != "pyannote"
            or fact.sample_id != self.sample.sample_id
            or fact.language != self.sample.language
            or fact.input_sha256 != self.sample.audio_sha256
            for fact in self.executions
        ):
            raise ValueError("pyannote 执行事实与当前授权音频不匹配")
        if self.status == GateStatus.PASS and (
            len(self.executions) != 1
            or (fact := self.executions[0]).component != "pyannote"
            or fact.operation != "diarize"
            or fact.sample_id != self.sample.sample_id
            or fact.input_sha256 != self.sample.audio_sha256
            or fact.model.model_id != "pyannote/speaker-diarization-community-1"
        ):
            raise ValueError("pyannote 真实执行事实不完整")
        return self


class FiveLanguageModelsRawReport(_LiveRawReport):
    check_id: Literal["five_language_models"]
    samples: tuple[LiveSample, ...] = Field(min_length=5, max_length=5)

    @model_validator(mode="after")
    def validate_model_coverage(self) -> FiveLanguageModelsRawReport:
        self._validate_sample_inputs(self.samples)
        local_components = {
            "silero_vad",
            "faster_whisper",
            "whisperx",
            "yamnet",
        }
        sample_by_id = {sample.sample_id: sample for sample in self.samples}
        if (
            len(sample_by_id) != 5
            or {sample.language for sample in self.samples} != {"zh", "en", "ja", "ko", "es"}
            or any(
                fact.sample_id not in sample_by_id
                or fact.component not in local_components
                or fact.input_sha256 != sample_by_id[fact.sample_id].audio_sha256
                or fact.language != sample_by_id[fact.sample_id].language
                for fact in self.executions
            )
        ):
            raise ValueError("本地模型执行事实未绑定完整五语授权音频")
        component_languages = {
            component: {
                fact.language
                for fact in self.executions
                if fact.component == component
            }
            for component in ("faster_whisper", "whisperx")
        }
        complete_execution = (
            len(self.executions) == 12
            and component_languages["faster_whisper"]
            == {"zh", "en", "ja", "ko", "es"}
            and component_languages["whisperx"]
            == {"zh", "en", "ja", "ko", "es"}
            and sum(fact.component == "silero_vad" for fact in self.executions) == 1
            and sum(fact.component == "yamnet" for fact in self.executions) == 1
            and not any(
                fact.language != sample_by_id[fact.sample_id].language
                for fact in self.executions
                if fact.component in {"faster_whisper", "whisperx"}
            )
        )
        if self.status == GateStatus.PASS:
            if not complete_execution:
                raise ValueError("本地模型栈必须完整执行 Silero、五语 ASR/对齐与 YAMNet")
        else:
            if self.failure_component == "components_close":
                if not complete_execution:
                    raise ValueError("本地模型栈组件关闭失败前执行事实不完整")
            elif self.failure_component not in local_components:
                raise ValueError("本地模型栈失败组件身份不匹配")
        return self


class RealMediaFile(FrozenModel):
    role: Literal["SOURCE", "PROXY", "AUDIO", "KEYFRAME"]
    format: Literal["MP4", "WAV", "JPEG"]
    relative_path: str = Field(min_length=1, max_length=1024)
    sha256: Sha256
    size_bytes: StrictInt = Field(gt=0, le=_DEFAULT_MEDIA_BYTES)

    @field_validator("relative_path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        _validate_workspace_relative(value)
        _require_runtime_relative(value)
        return value


class RealMediaCommand(FrozenModel):
    phase: Literal[
        "generate",
        "probe",
        "proxy",
        "audio",
        "opencv_decode",
        "scene_detect",
        "keyframe_select",
    ]
    executable: Literal[
        "ffmpeg",
        "ffprobe",
        "FFmpegTranscoder",
        "OpenCvFrameExtractor",
        "PySceneDetectAdapter",
        "KeyframeSelector",
    ]
    arguments: tuple[str, ...] = ()
    input_relative_paths: tuple[str, ...] = ()
    output_relative_paths: tuple[str, ...] = ()
    exit_code: StrictInt
    stdout_relative_path: str = Field(min_length=1, max_length=1024)
    stderr_relative_path: str = Field(min_length=1, max_length=1024)
    stdout_sha256: Sha256
    stderr_sha256: Sha256

    @field_validator("arguments")
    @classmethod
    def validate_arguments(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for argument in value:
            normalized = PurePosixPath(argument)
            if (
                not argument
                or normalized.is_absolute()
                or ".." in normalized.parts
                or argument != normalized.as_posix()
                or normalized.as_posix() == "."
            ):
                raise ValueError("真实媒体命令不得包含绝对或父级路径")
        _validate_persisted_value(value)
        return value

    @field_validator("input_relative_paths", "output_relative_paths")
    @classmethod
    def validate_explicit_runtime_paths(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("真实媒体命令显式路径不得重复")
        for path in value:
            _validate_workspace_relative(path)
            _require_runtime_relative(path)
        return value

    @field_validator("stdout_relative_path", "stderr_relative_path")
    @classmethod
    def validate_output_relative_path(cls, value: str) -> str:
        _validate_workspace_relative(value)
        _require_runtime_relative(value)
        return value

    @model_validator(mode="after")
    def validate_phase_executable_and_arguments(self) -> RealMediaCommand:
        if self.executable != _REAL_MEDIA_PHASE_EXECUTABLES[self.phase]:
            raise ValueError("真实媒体阶段与执行器不匹配")
        explicit_paths = set(self.input_relative_paths) | set(self.output_relative_paths)
        for argument in self.arguments:
            if argument.startswith(f"{_RUNTIME_RELATIVE_ROOT}/") and argument not in explicit_paths:
                raise ValueError("真实媒体路径参数必须显式声明输入或输出归属")
        return self


class SetupMediaCommand(FrozenModel):
    """样本执行前的工具版本探测事实。"""

    phase: Literal["ffmpeg_version", "ffprobe_version"]
    executable: Literal["ffmpeg", "ffprobe"]
    arguments: tuple[Literal["-version"]] = ("-version",)
    exit_code: StrictInt
    stdout_relative_path: str = Field(min_length=1, max_length=1024)
    stderr_relative_path: str = Field(min_length=1, max_length=1024)
    stdout_sha256: Sha256
    stderr_sha256: Sha256

    @field_validator("stdout_relative_path", "stderr_relative_path")
    @classmethod
    def validate_output_relative_path(cls, value: str) -> str:
        _validate_workspace_relative(value)
        _require_runtime_relative(value)
        return value

    @model_validator(mode="after")
    def validate_phase_and_executable(self) -> SetupMediaCommand:
        if self.executable != self.phase.removesuffix("_version"):
            raise ValueError("版本探测阶段与执行器不匹配")
        return self


class RealMediaSample(FrozenModel):
    case_id: Literal["normal_audio", "no_audio", "rotation", "vfr"]
    execution_status: Literal["SUCCESS", "FAILED", "NOT_STARTED"]
    failure_code: ErrorCode | None = None
    duration_ms: StrictInt | None = Field(default=None, gt=0)
    has_audio: StrictBool | None = None
    rotation_degrees: StrictInt | None = Field(default=None, ge=0, le=359)
    is_variable_frame_rate: StrictBool | None = None
    warnings: tuple[str, ...] = ()
    opencv_decoded_frame_count: StrictInt | None = Field(default=None, ge=0)
    scene_count: StrictInt | None = Field(default=None, ge=0)
    selected_keyframe_count: StrictInt | None = Field(default=None, ge=0)
    files: tuple[RealMediaFile, ...] = ()
    commands: tuple[RealMediaCommand, ...] = ()

    @field_validator("commands")
    @classmethod
    def reject_duplicate_command_phases(
        cls, value: tuple[RealMediaCommand, ...]
    ) -> tuple[RealMediaCommand, ...]:
        phases = tuple(command.phase for command in value)
        if len(phases) != len(set(phases)):
            raise ValueError("真实媒体阶段命令不得重复")
        return value

    @model_validator(mode="after")
    def validate_execution_facts(self) -> RealMediaSample:
        if self.execution_status == "SUCCESS" and self.failure_code is not None:
            raise ValueError("成功媒体样本不得携带失败码")
        if self.execution_status == "FAILED" and self.failure_code is None:
            raise ValueError("失败媒体样本必须携带稳定失败码")
        if self.execution_status == "NOT_STARTED":
            if self.failure_code is not None or self.files or self.commands:
                raise ValueError("未启动媒体样本不得携带执行产物或失败码")
            if any(
                value is not None
                for value in (
                    self.duration_ms,
                    self.has_audio,
                    self.rotation_degrees,
                    self.is_variable_frame_rate,
                    self.opencv_decoded_frame_count,
                    self.scene_count,
                    self.selected_keyframe_count,
                )
            ) or self.warnings:
                raise ValueError("未启动媒体样本不得携带 probe 或视觉事实")
        return self


class RealMediaRawReport(FrozenModel):
    schema_version: Literal["1.0.0"]
    status: Literal[GateStatus.PASS, GateStatus.FAIL]
    trace_exit_code: StrictInt
    evaluation_run_id: StableId
    ffmpeg_version: str | None = Field(default=None, min_length=1, max_length=256)
    ffprobe_version: str | None = Field(default=None, min_length=1, max_length=256)
    implementation_sha256: Sha256
    setup_commands: tuple[SetupMediaCommand, ...] = ()
    samples: tuple[RealMediaSample, ...] = Field(min_length=1)
    failure_code: ErrorCode | None = None

    @model_validator(mode="after")
    def validate_structured_media_facts(self) -> RealMediaRawReport:
        case_ids = tuple(sample.case_id for sample in self.samples)
        if case_ids != _REAL_MEDIA_CASE_IDS:
            raise ValueError("真实媒体报告必须按固定顺序覆盖四个 case")
        if self.status == GateStatus.PASS:
            if (
                self.trace_exit_code != 0
                or self.failure_code is not None
                or any(sample.execution_status != "SUCCESS" for sample in self.samples)
            ):
                raise ValueError("真实媒体 PASS 不得包含失败事实")
            self._validate_completed_setup()
        else:
            self._validate_failed_execution()
        seen_paths: set[str] = set()
        for sample in self.samples:
            self._validate_sample_files(sample, seen_paths)
            if sample.execution_status == "SUCCESS":
                self._validate_successful_sample(sample)
        return self

    def _validate_failed_execution(self) -> None:
        if self.trace_exit_code == 0 or self.failure_code is None:
            raise ValueError("真实媒体 FAIL 必须包含非零 trace 与稳定失败事实")
        statuses = tuple(sample.execution_status for sample in self.samples)
        if all(status == "NOT_STARTED" for status in statuses):
            self._validate_failed_setup()
            return
        try:
            failed_index = statuses.index("FAILED")
        except ValueError:
            raise ValueError("真实媒体 FAIL 必须包含失败样本") from None
        if (
            any(status != "SUCCESS" for status in statuses[:failed_index])
            or any(status != "NOT_STARTED" for status in statuses[failed_index + 1 :])
            or self.samples[failed_index].failure_code != self.failure_code
        ):
            raise ValueError("真实媒体 FAIL 必须在失败样本后停止后续 case")
        commands = self.samples[failed_index].commands
        phases = tuple(command.phase for command in commands)
        phase_failure = bool(
            commands
            and phases == _REAL_MEDIA_PHASES[: len(commands)]
            and all(command.exit_code == 0 for command in commands[:-1])
            and commands[-1].exit_code != 0
        )
        sample_finalization_failure = bool(
            phases == _REAL_MEDIA_PHASES
            and all(command.exit_code == 0 for command in commands)
        )
        if not phase_failure and not sample_finalization_failure:
            raise ValueError("失败媒体样本命令必须表达阶段失败或完整样本汇总失败")
        self._validate_completed_setup()

    def _validate_completed_setup(self) -> None:
        if (
            self.ffmpeg_version is None
            or self.ffprobe_version is None
            or tuple(command.phase for command in self.setup_commands)
            != ("ffmpeg_version", "ffprobe_version")
            or any(command.exit_code != 0 for command in self.setup_commands)
        ):
            raise ValueError("媒体样本执行必须拥有两条成功版本探测")
        self._validate_setup_output_paths()

    def _validate_failed_setup(self) -> None:
        phases = tuple(command.phase for command in self.setup_commands)
        allowed_prefixes = (
            ("ffmpeg_version",),
            ("ffmpeg_version", "ffprobe_version"),
        )
        if (
            not self.setup_commands
            or phases not in allowed_prefixes
            or any(command.exit_code != 0 for command in self.setup_commands[:-1])
            or self.setup_commands[-1].exit_code == 0
        ):
            raise ValueError("版本探测失败必须以固定 setup 前缀结束于非零退出")
        self._validate_setup_output_paths()
        successful_phases = phases[:-1]
        if (
            ("ffmpeg_version" in successful_phases) != (self.ffmpeg_version is not None)
            or ("ffprobe_version" in successful_phases) != (self.ffprobe_version is not None)
        ):
            raise ValueError("版本探测失败的版本字段必须精确表达已成功阶段")

    def _validate_setup_output_paths(self) -> None:
        prefix = f".codex/video-rag-demo/eval/reports/{self.evaluation_run_id}/"
        paths = tuple(
            path
            for command in self.setup_commands
            for path in (command.stdout_relative_path, command.stderr_relative_path)
        )
        if any(not path.startswith(prefix) for path in paths):
            raise ValueError("版本探测命令输出必须绑定当前评测运行报告目录")
        if len(paths) != len(set(paths)):
            raise ValueError("版本探测命令输出不得跨命令复用")

    def _validate_sample_files(
        self, sample: RealMediaSample, seen_paths: set[str]
    ) -> None:
        prefix = (
            f".codex/video-rag-demo/eval/generated/{self.evaluation_run_id}/"
            f"{sample.case_id}/"
        )
        for media_file in sample.files:
            if not media_file.relative_path.startswith(prefix):
                raise ValueError("真实媒体文件必须绑定当前评测运行与 case 目录")
            if media_file.relative_path in seen_paths:
                raise ValueError("真实媒体文件路径不得跨样本重复")
            seen_paths.add(media_file.relative_path)
        for command in sample.commands:
            if _REAL_MEDIA_PHASE_EXECUTABLES.get(command.phase) != command.executable:
                raise ValueError("真实媒体阶段与执行器不匹配")
            explicit_paths = set(command.input_relative_paths) | set(
                command.output_relative_paths
            )
            if any(
                f"{_RUNTIME_RELATIVE_ROOT}/" in argument
                and argument not in explicit_paths
                for argument in command.arguments
            ):
                raise ValueError("真实媒体路径参数必须显式声明输入或输出归属")
            for path in (
                *command.input_relative_paths,
                *command.output_relative_paths,
            ):
                if not path.startswith(prefix):
                    raise ValueError("真实媒体命令路径必须绑定当前评测运行与 case 目录")
                if sample.execution_status == "SUCCESS" and path not in {
                    media_file.relative_path for media_file in sample.files
                }:
                    raise ValueError("成功媒体命令路径必须精确绑定样本文件")
            report_prefix = (
                f".codex/video-rag-demo/eval/reports/{self.evaluation_run_id}/"
            )
            if not (
                command.stdout_relative_path.startswith(report_prefix)
                and command.stderr_relative_path.startswith(report_prefix)
            ):
                raise ValueError("真实媒体命令输出必须绑定当前评测运行报告目录")

    @staticmethod
    def _validate_successful_sample(sample: RealMediaSample) -> None:
        if (
            not sample.files
            or tuple(command.phase for command in sample.commands)
            != _REAL_MEDIA_PHASES
            or any(command.exit_code != 0 for command in sample.commands)
        ):
            raise ValueError("成功媒体样本必须包含成功文件与命令事实")
        if any(
            value is None
            for value in (
                sample.duration_ms,
                sample.has_audio,
                sample.rotation_degrees,
                sample.is_variable_frame_rate,
                sample.opencv_decoded_frame_count,
                sample.scene_count,
                sample.selected_keyframe_count,
            )
        ) or (
            sample.opencv_decoded_frame_count == 0
            or sample.scene_count == 0
            or sample.selected_keyframe_count == 0
        ):
            raise ValueError("成功媒体样本必须包含完整非空 probe 与视觉事实")
        files_by_role: dict[str, list[RealMediaFile]] = {}
        for media_file in sample.files:
            files_by_role.setdefault(media_file.role, []).append(media_file)
        expected_roles = {"SOURCE", "PROXY", "KEYFRAME"}
        if sample.case_id != "no_audio":
            expected_roles.add("AUDIO")
        if set(files_by_role) != expected_roles or any(
            len(files_by_role[role]) != 1 for role in expected_roles if role != "KEYFRAME"
        ) or len(files_by_role["KEYFRAME"]) != sample.selected_keyframe_count:
            raise ValueError("成功媒体样本文件角色与数量不符合 case 语义")
        expected_formats = {"SOURCE": "MP4", "PROXY": "MP4", "KEYFRAME": "JPEG"}
        if "AUDIO" in expected_roles:
            expected_formats["AUDIO"] = "WAV"
        if any(
            any(media_file.format != expected_formats[role] for media_file in files)
            for role, files in files_by_role.items()
        ):
            raise ValueError("成功媒体样本文件格式不符合 case 语义")
        paths_by_role = {
            role: tuple(media_file.relative_path for media_file in files)
            for role, files in files_by_role.items()
        }
        expected_flows = (
            ("generate", (), paths_by_role["SOURCE"]),
            ("probe", paths_by_role["SOURCE"], ()),
            ("proxy", paths_by_role["SOURCE"], paths_by_role["PROXY"]),
            ("audio", paths_by_role["SOURCE"], paths_by_role.get("AUDIO", ())),
            ("opencv_decode", paths_by_role["PROXY"], ()),
            ("scene_detect", paths_by_role["PROXY"], ()),
            ("keyframe_select", paths_by_role["PROXY"], paths_by_role["KEYFRAME"]),
        )
        if any(
            command.phase != phase
            or command.input_relative_paths != expected_inputs
            or command.output_relative_paths != expected_outputs
            for command, (phase, expected_inputs, expected_outputs) in zip(
                sample.commands, expected_flows, strict=True
            )
        ):
            raise ValueError("成功媒体命令数据流必须精确匹配文件事实")
        if (
            (sample.case_id == "no_audio") != (sample.has_audio is False)
            or (sample.case_id == "no_audio") != (sample.warnings == ("NO_AUDIO_TRACK",))
            or (sample.case_id == "rotation") != (sample.rotation_degrees == 90)
            or (sample.case_id != "rotation" and sample.rotation_degrees != 0)
            or (sample.case_id == "vfr") != sample.is_variable_frame_rate
        ):
            raise ValueError("成功媒体样本 probe 事实不符合 case 语义")


class PreflightRawReport(FrozenModel):
    schema_version: Literal["1.0.0"]
    check_id: str = Field(min_length=3, max_length=128)
    reason_code: PreflightReasonCode
    execution_started: StrictBool
    issues: tuple[PreflightIssue, ...] | None = None
    implementation_sha256: Sha256 | None = None
    evaluation_run_id: StableId | None = None

    @model_validator(mode="after")
    def bind_reason_to_check(self) -> PreflightRawReport:
        if self.execution_started:
            raise ValueError("preflight 不得在执行开始后形成 NOT_RUN")
        if _PREFLIGHT_REASON_CHECK[self.reason_code] != self.check_id:
            raise ValueError("preflight 原因与检查不匹配")
        strict_codes = (
            _REAL_MEDIA_PREFLIGHT_CODES
            if self.check_id == "real_media_chain"
            else (
                _DURABILITY_PREFLIGHT_CODES
                if self.check_id == "m1_durability"
                else _LIVE_PREFLIGHT_CODES.get(self.check_id)
            )
        )
        if strict_codes is None:
            if (
                self.issues is not None
                or self.implementation_sha256 is not None
                or self.evaluation_run_id is not None
            ):
                raise ValueError("该 preflight 不支持声明逐项缺失 issue")
            return self
        if (
            self.issues is None
            or not self.issues
            or self.implementation_sha256 is None
            or self.evaluation_run_id is None
        ):
            raise ValueError("真实运行 preflight 必须声明缺失 issue 与实现摘要")
        codes = tuple(issue.code for issue in self.issues)
        if (
            any(code not in strict_codes for code in codes)
            or len(codes) != len(set(codes))
            or codes != tuple(sorted(codes, key=strict_codes.index))
        ):
            raise ValueError("真实运行 preflight 缺失 issue 必须去重且按确定顺序排列")
        return self


class ProviderResponseSummary(FrozenModel):
    schema_version: Literal["1.0.0"]
    service: str = Field(min_length=2, max_length=64)
    model_id: str = Field(min_length=1, max_length=256)
    request_id_sha256: Sha256
    http_status: int = Field(ge=100, le=599)
    input_sha256: Sha256
    output_sha256: Sha256

    @model_validator(mode="after")
    def reject_unsafe_strings(self) -> ProviderResponseSummary:
        _validate_persisted_value(self.model_dump(mode="python"))
        return self


class _RealMediaCommitRecord(FrozenModel):
    schema_version: Literal["1.0.0"]
    evaluation_run_id: StableId
    authority_sha256: Sha256


class LiveAuthorityRecord(FrozenModel):
    """run 外独占发布的 live 机器报告权威绑定。"""

    schema_version: Literal["1.0.0"]
    check_id: LiveCheckId
    evaluation_run_id: StableId
    machine_report_path: str = Field(min_length=1, max_length=1024)
    machine_report_sha256: Sha256
    raw_report_sha256: Sha256
    settings_fingerprint: Sha256
    implementation_sha256: Sha256
    machine_report_identity_sha256: Sha256
    report_run_directory_identity_sha256: Sha256
    artifact_manifest_sha256: Sha256

    @field_validator("machine_report_path")
    @classmethod
    def validate_machine_report_path(cls, value: str) -> str:
        _validate_workspace_relative(value)
        _require_runtime_relative(value)
        return value

    @model_validator(mode="after")
    def reject_unsafe_values(self) -> LiveAuthorityRecord:
        _validate_persisted_value(self.model_dump(mode="python"))
        return self


@dataclass(frozen=True)
class FileIdentity:
    device: int
    inode: int
    size: int
    mtime_ns: int
    ctime_ns: int


@dataclass(frozen=True)
class FileSnapshot:
    path: Path
    identity: FileIdentity
    sha256: str
    content: bytes | None


@dataclass(frozen=True)
class _PublishedArtifact:
    descriptor: int
    identity: FileIdentity
    sha256: str


@dataclass(frozen=True)
class VerifiedArtifact:
    reference: TraceArtifact
    snapshot: FileSnapshot

    @property
    def path(self) -> Path:
        return self.snapshot.path


@dataclass(frozen=True)
class _StagedRealMediaRun:
    descriptor: int
    directory_identity: FileIdentity
    marker_identity: FileIdentity


@dataclass(frozen=True)
class _LiveAuthorityState:
    journal_snapshot: FileSnapshot
    run_descriptor: int
    run_identity: FileIdentity


class _ReportRunState(StrEnum):
    OPEN = "OPEN"
    FAILED = "FAILED"
    CLOSED = "CLOSED"


class LiveReportRunConflict(ValueError):
    """live 报告独占写入会话检测到目录或叶文件冲突。"""


class ReportRunWriter:
    """持有已验证 run 目录 fd 的原子写入会话。"""

    def __init__(
        self,
        workspace_root: Path,
        evaluation_run_id: str,
        descriptor: int,
        marker_identity: FileIdentity,
    ) -> None:
        self.workspace_root = workspace_root
        self.evaluation_run_id = evaluation_run_id
        self._descriptor: int | None = descriptor
        self._directory_identity = _identity(os.fstat(descriptor))
        self._marker_identity = marker_identity
        self._state = _ReportRunState.OPEN

    def __enter__(self) -> ReportRunWriter:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def close(self) -> None:
        if self._state == _ReportRunState.CLOSED:
            return
        descriptor = self._descriptor
        self._descriptor = None
        self._state = _ReportRunState.CLOSED
        if descriptor is not None:
            with suppress(OSError):
                os.close(descriptor)

    def write_artifact(
        self,
        filename: str,
        role: ArtifactRole,
        payload: bytes,
        *,
        max_bytes: int | None = None,
    ) -> TraceArtifact:
        descriptor = self._require_open()
        try:
            _validate_run_leaf(filename)
            if role in _MEDIA_ROLES:
                raise ValueError("媒体产物不得通过机器证据写入器写入")
            if not isinstance(payload, bytes):
                raise ValueError("机器证据产物必须是 bytes")
            effective_limit = _artifact_limit(role, max_bytes)
            if (
                (not payload and role not in {"COMMAND_STDOUT", "COMMAND_STDERR"})
                or len(payload) > effective_limit
            ):
                raise ValueError("机器证据产物超过大小上限或为空")
            _validate_artifact_content(role, payload)
            _write_run_payload(descriptor, filename, payload)
            return TraceArtifact(
                role=role,
                relative_path=self._workspace_relative(filename),
                sha256=hashlib.sha256(payload).hexdigest(),
                max_bytes=effective_limit,
            )
        except (OSError, RecursionError, ValueError):
            self._state = _ReportRunState.FAILED
            raise ValueError("机器证据产物原子写入失败") from None
        except BaseException:
            self._state = _ReportRunState.FAILED
            raise

    def write_json(
        self,
        model: FrozenModel,
        *,
        filename: str = "real-media.json",
        settings: Settings | None = None,
    ) -> GateCheck:
        descriptor = self._require_open()
        try:
            if not isinstance(model, MachineEvidenceReport):
                raise ValueError("仅机器报告可生成权威证据引用")
            payload = model.model_dump(
                mode="python", exclude_none=True, exclude_computed_fields=True
            )
            validated = MachineEvidenceReport.model_validate(payload)
            encoded = validated.model_dump_json(
                exclude_none=True, exclude_computed_fields=True
            ).encode("utf-8")
            if not encoded or len(encoded) > _MAX_MACHINE_BYTES:
                raise ValueError("机器报告超过大小上限")
            decoded = _decode_strict_json(encoded.decode("utf-8"))
            if not isinstance(decoded, dict):
                raise ValueError("机器报告顶层必须是对象")
            validated = MachineEvidenceReport.model_validate(decoded)
            reference = EvidenceReference(
                kind=validated.kind,
                level=validated.level,
                relative_path=self._workspace_relative(filename),
                sha256=hashlib.sha256(encoded).hexdigest(),
                covered_items=validated.covered_items,
                summary=validated.summary,
            )
            _validate_run_leaf(filename)
            _write_run_payload(descriptor, filename, encoded)
            self._assert_incomplete_marker_current(descriptor)
            is_real_media = validated.check_id == "real_media_chain"
            check = _build_verified_gate_check(
                validated.check_id,
                self.workspace_root / reference.relative_path,
                workspace_root=self.workspace_root,
                allow_incomplete_real_media_run=is_real_media,
                staged_real_media_run=(
                    _StagedRealMediaRun(
                        descriptor=descriptor,
                        directory_identity=self._directory_identity,
                        marker_identity=self._marker_identity,
                    )
                    if is_real_media
                    else None
                ),
                settings=settings,
            )
            self._assert_incomplete_marker_current(descriptor)
            if is_real_media:
                _publish_real_media_commit(
                    descriptor,
                    _RealMediaCommitRecord(
                        schema_version="1.0.0",
                        evaluation_run_id=self.evaluation_run_id,
                        authority_sha256=reference.sha256,
                    ),
                )
            with suppress(Exception):
                os.unlink(_REAL_MEDIA_INCOMPLETE_MARKER, dir_fd=descriptor)
            with suppress(Exception):
                os.fsync(descriptor)
        except (OSError, ValueError):
            self._state = _ReportRunState.FAILED
            raise ValueError("机器证据原子写入失败") from None
        except BaseException:
            self._state = _ReportRunState.FAILED
            raise
        self.close()
        return check

    def _assert_incomplete_marker_current(self, descriptor: int) -> None:
        if (
            _identity(os.stat(_REAL_MEDIA_INCOMPLETE_MARKER, dir_fd=descriptor))
            != self._marker_identity
        ):
            raise ValueError("真实媒体进行中标记身份发生变化")

    def _require_open(self) -> int:
        if self._state != _ReportRunState.OPEN or self._descriptor is None:
            raise ValueError("run 写入会话不可写")
        return self._descriptor

    def _workspace_relative(self, filename: str) -> str:
        return (
            Path(".codex/video-rag-demo/eval/reports")
            .joinpath(self.evaluation_run_id, filename)
            .as_posix()
        )


class LiveReportRunWriter:
    """持有 live run 目录 fd，并对目录身份与完整条目集负责。"""

    def __init__(
        self,
        workspace_root: Path,
        runtime_root: Path,
        evaluation_run_id: str,
        descriptor: int,
    ) -> None:
        self.workspace_root = workspace_root
        self.runtime_root = runtime_root
        self.evaluation_run_id = evaluation_run_id
        self._descriptor: int | None = descriptor
        self._directory_identity = _identity(os.fstat(descriptor))
        self._published: dict[str, FileSnapshot] = {}
        self._state = _ReportRunState.OPEN

    def __enter__(self) -> LiveReportRunWriter:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def close(self) -> None:
        if self._state == _ReportRunState.CLOSED:
            return
        descriptor = self._descriptor
        self._descriptor = None
        self._state = _ReportRunState.CLOSED
        if descriptor is not None:
            with suppress(OSError):
                os.close(descriptor)

    def write_artifact(
        self,
        filename: str,
        role: ArtifactRole,
        payload: bytes,
        *,
        max_bytes: int | None = None,
    ) -> TraceArtifact:
        descriptor = self._require_open()
        try:
            self._assert_current(descriptor)
            _validate_run_leaf(filename)
            if role in _MEDIA_ROLES:
                raise ValueError("媒体产物不得通过机器证据写入器写入")
            if not isinstance(payload, bytes):
                raise ValueError("机器证据产物必须是 bytes")
            effective_limit = _artifact_limit(role, max_bytes)
            if (
                (not payload and role not in {"COMMAND_STDOUT", "COMMAND_STDERR"})
                or len(payload) > effective_limit
            ):
                raise ValueError("机器证据产物超过大小上限或为空")
            _validate_artifact_content(role, payload)
            _write_exclusive_payload(descriptor, filename, payload)
            snapshot = _read_file_snapshot_at(
                descriptor,
                self._run_path / filename,
                max_bytes=effective_limit,
                allow_empty=role in {"COMMAND_STDOUT", "COMMAND_STDERR"},
            )
            if snapshot.sha256 != hashlib.sha256(payload).hexdigest():
                raise ValueError("live 产物发布内容不一致")
            _validate_artifact_content(role, snapshot.content)
            self._published[filename] = snapshot
            self._assert_current(descriptor)
            return TraceArtifact(
                role=role,
                relative_path=self._workspace_relative(filename),
                sha256=snapshot.sha256,
                max_bytes=effective_limit,
            )
        except (OSError, RecursionError, ValueError):
            self._state = _ReportRunState.FAILED
            raise LiveReportRunConflict("live 报告产物独占写入失败") from None

    def write_json(
        self,
        model: FrozenModel,
        *,
        settings: Settings | None = None,
    ) -> GateCheck:
        descriptor = self._require_open()
        try:
            self._assert_current(descriptor)
            validated, encoded = _encode_machine_report(model)
            filename = f"{validated.check_id}.json"
            report_path = self._run_path / filename
            self._assert_report_artifacts_declared(validated, report_path)
            raw: _LiveRawReport | None = None
            if _is_executed_live_machine_report(validated):
                raw = _validate_live_publication_context_at(
                    validated,
                    report_path,
                    self.workspace_root,
                    descriptor,
                )
            _write_exclusive_payload(descriptor, filename, encoded)
            report_snapshot = _read_file_snapshot_at(
                descriptor,
                report_path,
                max_bytes=_MAX_MACHINE_BYTES,
            )
            if report_snapshot.sha256 != hashlib.sha256(encoded).hexdigest():
                raise ValueError("live 机器报告发布内容不一致")
            self._published[filename] = report_snapshot
            self._assert_current(descriptor)
            if raw is not None:
                self._publish_authority(validated, raw, report_snapshot)
                self._assert_current(descriptor)
            check = build_verified_gate_check(
                validated.check_id,
                report_path,
                workspace_root=self.workspace_root,
                settings=settings,
            )
            self._assert_current(descriptor)
        except (OSError, RecursionError, ValueError, VideoDemoError):
            self._state = _ReportRunState.FAILED
            raise LiveReportRunConflict("live 机器报告独占发布失败") from None
        self.close()
        return check

    def assert_current(self) -> None:
        descriptor = self._require_open()
        try:
            self._assert_current(descriptor)
        except (OSError, ValueError):
            self._state = _ReportRunState.FAILED
            raise LiveReportRunConflict("live 运行目录身份冲突") from None

    @property
    def _run_path(self) -> Path:
        return self.runtime_root / "eval" / "reports" / self.evaluation_run_id

    def _require_open(self) -> int:
        if self._state != _ReportRunState.OPEN or self._descriptor is None:
            raise LiveReportRunConflict("live run 写入会话不可写")
        return self._descriptor

    def _workspace_relative(self, filename: str) -> str:
        return (self._run_path / filename).relative_to(self.workspace_root).as_posix()

    def _assert_current(self, descriptor: int) -> None:
        held_identity = _identity(os.fstat(descriptor))
        if not _same_directory(held_identity, self._directory_identity):
            raise ValueError("live 持有目录身份发生变化")
        current_descriptor = _open_directory_descriptor(self._run_path)
        try:
            current_identity = _identity(os.fstat(current_descriptor))
        finally:
            os.close(current_descriptor)
        if not _same_directory(current_identity, self._directory_identity):
            raise ValueError("live run 路径已指向其他目录")
        if set(os.listdir(descriptor)) != set(self._published):
            raise ValueError("live run 包含未声明或缺失条目")
        for filename, snapshot in self._published.items():
            metadata = os.stat(filename, dir_fd=descriptor, follow_symlinks=False)
            if not stat.S_ISREG(metadata.st_mode) or _identity(metadata) != snapshot.identity:
                raise ValueError("live run 已声明条目身份发生变化")

    def _assert_report_artifacts_declared(
        self,
        report: MachineEvidenceReport,
        report_path: Path,
    ) -> None:
        report_parent = report_path.parent.relative_to(self.workspace_root)
        declared = {
            Path(artifact.relative_path).name
            for artifact in report.artifacts
            if Path(artifact.relative_path).parent == report_parent
        }
        if declared != set(self._published):
            raise ValueError("live 机器报告未精确声明当前 run 产物")

    def _publish_authority(
        self,
        report: MachineEvidenceReport,
        raw: _LiveRawReport,
        report_snapshot: FileSnapshot,
    ) -> None:
        details = report.details
        if not isinstance(details, _LiveCheckDetails):
            raise ValueError("live authority 缺少已执行 detail")
        record = LiveAuthorityRecord(
            schema_version="1.0.0",
            check_id=_require_live_check_id(report.check_id),
            evaluation_run_id=self.evaluation_run_id,
            machine_report_path=report_snapshot.path.relative_to(
                self.workspace_root
            ).as_posix(),
            machine_report_sha256=report_snapshot.sha256,
            raw_report_sha256=details.raw_report_sha256,
            settings_fingerprint=details.settings_fingerprint,
            implementation_sha256=details.implementation_sha256,
            machine_report_identity_sha256=_identity_sha256(
                report_snapshot.identity
            ),
            report_run_directory_identity_sha256=_identity_sha256(
                _identity(os.fstat(self._require_open()))
            ),
            artifact_manifest_sha256=_artifact_manifest_sha256(report.artifacts),
        )
        if raw.evaluation_run_id != record.evaluation_run_id:
            raise ValueError("live authority 与 raw run 不匹配")
        _publish_live_authority_record(self.runtime_root, record)


class EvidenceStore:
    def __init__(self, workspace_root: Path, runtime_root: Path) -> None:
        try:
            self.workspace_root, self.runtime_root = _trusted_roots(
                workspace_root, runtime_root
            )
        except (OSError, ValueError):
            raise ValueError("机器证据可信根非法") from None

    def write_json(
        self, relative_path: Path, model: FrozenModel
    ) -> EvidenceReference:
        try:
            if not isinstance(model, MachineEvidenceReport):
                raise ValueError("仅机器报告可生成权威证据引用")
            target = _prepare_runtime_target(self.runtime_root, relative_path)
            payload = model.model_dump(
                mode="python",
                exclude_none=True,
                exclude_computed_fields=True,
            )
            validated = MachineEvidenceReport.model_validate(payload)
            encoded = validated.model_dump_json(
                exclude_none=True,
                exclude_computed_fields=True,
            ).encode("utf-8")
            if not encoded or len(encoded) > _MAX_MACHINE_BYTES:
                raise ValueError("机器报告超过大小上限")
            decoded = _decode_strict_json(encoded.decode("utf-8"))
            if not isinstance(decoded, dict):
                raise ValueError("机器报告顶层必须是对象")
            validated = MachineEvidenceReport.model_validate(decoded)
            if _is_executed_live_machine_report(validated):
                return self._write_live_machine_report(
                    target,
                    validated,
                    encoded,
                )
            temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.part")
            try:
                with temporary.open("xb") as stream:
                    stream.write(encoded)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, target)
                directory_descriptor = os.open(target.parent, os.O_RDONLY)
                try:
                    os.fsync(directory_descriptor)
                finally:
                    os.close(directory_descriptor)
            except OSError:
                target.unlink(missing_ok=True)
                raise
            relative = target.relative_to(self.workspace_root).as_posix()
            return EvidenceReference(
                kind=validated.kind,
                level=validated.level,
                relative_path=relative,
                sha256=hashlib.sha256(encoded).hexdigest(),
                covered_items=validated.covered_items,
                summary=validated.summary,
            )
        except (OSError, ValueError):
            raise ValueError("机器证据原子写入失败") from None

    def _write_live_machine_report(
        self,
        target: Path,
        report: MachineEvidenceReport,
        encoded: bytes,
    ) -> EvidenceReference:
        details = report.details
        if not isinstance(details, _LiveCheckDetails):
            raise ValueError("仅已执行 live 报告可发布 live authority")
        raw, run_id = _validate_live_publication_context(
            report,
            target,
            self.workspace_root,
            self.runtime_root,
        )
        report_snapshot, run_identity = _publish_exclusive_machine_report(
            target,
            encoded,
        )
        record = LiveAuthorityRecord(
            schema_version="1.0.0",
            check_id=_require_live_check_id(report.check_id),
            evaluation_run_id=run_id,
            machine_report_path=target.relative_to(self.workspace_root).as_posix(),
            machine_report_sha256=report_snapshot.sha256,
            raw_report_sha256=details.raw_report_sha256,
            settings_fingerprint=details.settings_fingerprint,
            implementation_sha256=details.implementation_sha256,
            machine_report_identity_sha256=_identity_sha256(
                report_snapshot.identity
            ),
            report_run_directory_identity_sha256=_identity_sha256(run_identity),
            artifact_manifest_sha256=_artifact_manifest_sha256(report.artifacts),
        )
        if raw.evaluation_run_id != record.evaluation_run_id:
            raise ValueError("live authority 与 raw run 不匹配")
        _publish_live_authority_record(self.runtime_root, record)
        return EvidenceReference(
            kind=report.kind,
            level=report.level,
            relative_path=record.machine_report_path,
            sha256=report_snapshot.sha256,
            covered_items=report.covered_items,
            summary=report.summary,
        )

    def open_exclusive_report_run(self, evaluation_run_id: str) -> ReportRunWriter:
        """安全创建 run，并把叶目录 fd 的所有权交给写入会话。"""

        _validate_stable_id(evaluation_run_id, "评测运行 ID")
        parts = ("eval", "reports", evaluation_run_id)
        descriptor = _open_directory_descriptor(self.runtime_root)
        try:
            for index, part in enumerate(parts):
                leaf = index == len(parts) - 1
                try:
                    os.mkdir(part, dir_fd=descriptor)
                except FileExistsError:
                    if leaf:
                        raise ValueError("运行目录已存在") from None
                child = _open_child_directory(descriptor, part)
                os.close(descriptor)
                descriptor = child
            marker_identity = _create_run_incomplete_marker(descriptor)
            return ReportRunWriter(
                self.workspace_root,
                evaluation_run_id,
                descriptor,
                marker_identity,
            )
        except OSError:
            with suppress(OSError):
                os.close(descriptor)
            raise ValueError("运行目录安全创建失败") from None
        except BaseException:
            with suppress(OSError):
                os.close(descriptor)
            raise

    def claim_exclusive_live_report_run(
        self,
        evaluation_run_id: str,
    ) -> LiveReportRunWriter:
        """独占声明 live run，并把叶目录 fd 的所有权交给写入会话。"""

        parent_descriptor: int | None = None
        child_descriptor: int | None = None
        try:
            _validate_stable_id(evaluation_run_id, "评测运行 ID")
            parent_descriptor = _open_or_create_runtime_parent(
                self.runtime_root,
                Path("eval/reports"),
            )
            os.mkdir(evaluation_run_id, dir_fd=parent_descriptor)
            child_descriptor = _open_child_directory(
                parent_descriptor,
                evaluation_run_id,
            )
            os.fsync(parent_descriptor)
            writer = LiveReportRunWriter(
                self.workspace_root,
                self.runtime_root,
                evaluation_run_id,
                child_descriptor,
            )
            writer.assert_current()
            child_descriptor = None
            return writer
        except (OSError, ValueError):
            raise LiveReportRunConflict("live 运行目录安全声明失败") from None
        finally:
            if child_descriptor is not None:
                with suppress(OSError):
                    os.close(child_descriptor)
            if parent_descriptor is not None:
                with suppress(OSError):
                    os.close(parent_descriptor)

    def bind_artifact(
        self,
        relative_path: Path,
        role: ArtifactRole,
        *,
        max_bytes: int | None = None,
    ) -> TraceArtifact:
        try:
            effective_limit = _artifact_limit(role, max_bytes)
            path = _runtime_existing_file(self.runtime_root, relative_path)
            snapshot = _read_file_snapshot(
                path,
                max_bytes=effective_limit,
                capture_content=role not in _MEDIA_ROLES,
                allow_empty=role in {"COMMAND_STDOUT", "COMMAND_STDERR"},
            )
            _validate_artifact_content(role, snapshot.content)
            return TraceArtifact(
                role=role,
                relative_path=path.relative_to(self.workspace_root).as_posix(),
                sha256=snapshot.sha256,
                max_bytes=effective_limit,
            )
        except (OSError, RecursionError, ValueError):
            raise ValueError("机器证据产物绑定失败") from None

    def write_artifact(
        self,
        relative_path: Path,
        role: ArtifactRole,
        payload: bytes,
        *,
        max_bytes: int | None = None,
    ) -> TraceArtifact:
        parent_descriptor: int | None = None
        published: _PublishedArtifact | None = None
        try:
            normalized = _normalize_runtime_relative(relative_path)
            _reject_public_live_authority_write(normalized)
            parent_descriptor = _open_or_create_runtime_parent(
                self.runtime_root,
                normalized.parent,
            )
            target_name = normalized.name
            _validate_writable_artifact_leaf(parent_descriptor, target_name)
            if role in _MEDIA_ROLES:
                raise ValueError("媒体产物不得通过机器证据写入器写入")
            if not isinstance(payload, bytes):
                raise ValueError("机器证据产物必须是 bytes")
            effective_limit = _artifact_limit(role, max_bytes)
            if (
                (not payload and role not in {"COMMAND_STDOUT", "COMMAND_STDERR"})
                or len(payload) > effective_limit
            ):
                raise ValueError("机器证据产物超过大小上限或为空")
            _validate_artifact_content(role, payload)
            published = _write_bound_artifact_payload(
                parent_descriptor,
                target_name,
                payload,
                self.runtime_root / normalized,
                role,
                effective_limit,
                lambda: _assert_runtime_parent_current(
                    self.runtime_root,
                    normalized.parent,
                    parent_descriptor,
                ),
            )
            _assert_runtime_parent_current(
                self.runtime_root,
                normalized.parent,
                parent_descriptor,
            )
            snapshot = _read_file_snapshot_at(
                parent_descriptor,
                self.runtime_root / normalized,
                max_bytes=effective_limit,
                allow_empty=role in {"COMMAND_STDOUT", "COMMAND_STDERR"},
            )
            if (
                not _same_inode(snapshot.identity, published.identity)
                or snapshot.sha256 != published.sha256
            ):
                raise ValueError("artifact 发布目标已被替换")
            _validate_artifact_content(role, snapshot.content)
            _assert_published_artifact_current(
                parent_descriptor,
                target_name,
                published.identity,
            )
            return TraceArtifact(
                role=role,
                relative_path=(self.runtime_root / normalized)
                .relative_to(self.workspace_root)
                .as_posix(),
                sha256=published.sha256,
                max_bytes=effective_limit,
            )
        except (OSError, RecursionError, ValueError):
            raise ValueError("机器证据产物原子写入失败") from None
        finally:
            if published is not None:
                with suppress(OSError):
                    os.close(published.descriptor)
            if parent_descriptor is not None:
                with suppress(OSError):
                    os.close(parent_descriptor)


def _is_executed_live_machine_report(report: MachineEvidenceReport) -> bool:
    return (
        report.check_id in _LIVE_EXECUTED_CHECKS
        and isinstance(report.details, _LiveCheckDetails)
    )


def _encode_machine_report(
    model: FrozenModel,
) -> tuple[MachineEvidenceReport, bytes]:
    if not isinstance(model, MachineEvidenceReport):
        raise ValueError("仅机器报告可生成权威证据引用")
    payload = model.model_dump(
        mode="python",
        exclude_none=True,
        exclude_computed_fields=True,
    )
    validated = MachineEvidenceReport.model_validate(payload)
    encoded = validated.model_dump_json(
        exclude_none=True,
        exclude_computed_fields=True,
    ).encode("utf-8")
    if not encoded or len(encoded) > _MAX_MACHINE_BYTES:
        raise ValueError("机器报告超过大小上限")
    decoded = _decode_strict_json(encoded.decode("utf-8"))
    if not isinstance(decoded, dict):
        raise ValueError("机器报告顶层必须是对象")
    return MachineEvidenceReport.model_validate(decoded), encoded


def _require_live_check_id(check_id: str) -> LiveCheckId:
    if check_id not in _LIVE_EXECUTED_CHECKS:
        raise ValueError("未知 live 检查")
    return cast(LiveCheckId, check_id)


def _live_raw_type(check_id: str) -> type[_LiveRawReport]:
    raw_types: dict[str, type[_LiveRawReport]] = {
        "baidu_ocr_live": BaiduLiveRawReport,
        "qwen_live": QwenLiveRawReport,
        "pyannote_live": PyannoteLiveRawReport,
        "five_language_models": FiveLanguageModelsRawReport,
    }
    try:
        return raw_types[check_id]
    except KeyError:
        raise ValueError("未知 live 检查") from None


def _validate_live_publication_context(
    report: MachineEvidenceReport,
    target: Path,
    workspace_root: Path,
    runtime_root: Path,
) -> tuple[_LiveRawReport, str]:
    details = report.details
    if not isinstance(details, _LiveCheckDetails):
        raise ValueError("live authority 缺少已执行 detail")
    relative = target.relative_to(runtime_root)
    if (
        len(relative.parts) != 4
        or relative.parts[:2] != ("eval", "reports")
        or relative.name != f"{report.check_id}.json"
    ):
        raise ValueError("live machine report 必须位于固定 run 报告路径")
    run_id = relative.parts[2]
    raw_references = tuple(
        artifact
        for artifact in report.artifacts
        if artifact.role == "AUDIT_REPORT"
        and artifact.sha256 == details.raw_report_sha256
    )
    if len(raw_references) != 1:
        raise ValueError("live authority 必须唯一绑定 raw report")
    raw_path = _workspace_runtime_file(
        Path(raw_references[0].relative_path),
        workspace_root,
        runtime_root,
    )
    if raw_path.parent != target.parent:
        raise ValueError("live raw report 必须位于当前报告 run")
    raw_snapshot = _read_file_snapshot(
        raw_path,
        max_bytes=_MAX_MACHINE_BYTES,
        capture_content=True,
    )
    if raw_snapshot.sha256 != details.raw_report_sha256 or raw_snapshot.content is None:
        raise ValueError("live raw report 摘要不匹配")
    raw = _live_raw_type(report.check_id).model_validate_json(raw_snapshot.content)
    if (
        raw.evaluation_run_id != run_id
        or raw.check_id != report.check_id
        or raw.status != report.status
    ):
        raise ValueError("live report、raw 与 run 绑定不一致")
    response_paths = tuple(
        _workspace_runtime_file(
            Path(artifact.relative_path),
            workspace_root,
            runtime_root,
        )
        for artifact in report.artifacts
        if artifact.role == "PROVIDER_RESPONSE"
    )
    if any(path.parent != target.parent for path in response_paths):
        raise ValueError("live provider 摘要必须位于当前报告 run")
    return raw, run_id


def _validate_live_publication_context_at(
    report: MachineEvidenceReport,
    target: Path,
    workspace_root: Path,
    run_descriptor: int,
) -> _LiveRawReport:
    details = report.details
    if not isinstance(details, _LiveCheckDetails):
        raise ValueError("live authority 缺少已执行 detail")
    report_relative = target.relative_to(workspace_root)
    if (
        len(report_relative.parts) != 6
        or report_relative.parts[:4]
        != (".codex", "video-rag-demo", "eval", "reports")
        or report_relative.name != f"{report.check_id}.json"
    ):
        raise ValueError("live machine report 必须位于固定 run 报告路径")
    run_id = report_relative.parts[4]
    raw_references = tuple(
        artifact
        for artifact in report.artifacts
        if artifact.role == "AUDIT_REPORT"
        and artifact.sha256 == details.raw_report_sha256
    )
    if len(raw_references) != 1:
        raise ValueError("live authority 必须唯一绑定 raw report")
    raw_relative = Path(raw_references[0].relative_path)
    if raw_relative.parent != report_relative.parent:
        raise ValueError("live raw report 必须位于当前报告 run")
    raw_snapshot = _read_file_snapshot_at(
        run_descriptor,
        target.parent / raw_relative.name,
        max_bytes=_MAX_MACHINE_BYTES,
    )
    if raw_snapshot.sha256 != details.raw_report_sha256 or raw_snapshot.content is None:
        raise ValueError("live raw report 摘要不匹配")
    raw = _live_raw_type(report.check_id).model_validate_json(raw_snapshot.content)
    if (
        raw.evaluation_run_id != run_id
        or raw.check_id != report.check_id
        or raw.status != report.status
    ):
        raise ValueError("live report、raw 与 run 绑定不一致")
    response_paths = tuple(
        Path(artifact.relative_path)
        for artifact in report.artifacts
        if artifact.role == "PROVIDER_RESPONSE"
    )
    if any(path.parent != report_relative.parent for path in response_paths):
        raise ValueError("live provider 摘要必须位于当前报告 run")
    return raw


def _publish_exclusive_machine_report(
    target: Path,
    payload: bytes,
) -> tuple[FileSnapshot, FileIdentity]:
    descriptor = _open_directory_descriptor(target.parent)
    try:
        _write_exclusive_payload(descriptor, target.name, payload)
        snapshot = _read_file_snapshot_at(
            descriptor,
            target,
            max_bytes=_MAX_MACHINE_BYTES,
        )
        return snapshot, _identity(os.fstat(descriptor))
    finally:
        os.close(descriptor)


def _publish_live_authority_record(
    runtime_root: Path,
    record: LiveAuthorityRecord,
) -> None:
    relative_parent = Path("eval/live-authority") / record.evaluation_run_id
    descriptor = _open_or_create_runtime_parent(runtime_root, relative_parent)
    try:
        filename = f"{record.check_id}.json"
        payload = record.model_dump_json().encode("utf-8")
        _write_exclusive_payload(descriptor, filename, payload)
        snapshot = _read_file_snapshot_at(
            descriptor,
            runtime_root / relative_parent / filename,
            max_bytes=16 * 1024,
        )
        if snapshot.content is None:
            raise ValueError("live authority journal 缺少正文")
        if LiveAuthorityRecord.model_validate_json(snapshot.content) != record:
            raise ValueError("live authority journal 发布内容不一致")
    finally:
        os.close(descriptor)


def _write_exclusive_payload(
    descriptor: int,
    filename: str,
    payload: bytes,
) -> None:
    _validate_run_leaf(filename)
    temporary = f".{filename}.{uuid.uuid4().hex}.part"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    stream_descriptor: int | None = None
    published = False
    try:
        stream_descriptor = os.open(temporary, flags, 0o600, dir_fd=descriptor)
        view = memoryview(payload)
        while view:
            written = os.write(stream_descriptor, view)
            if written <= 0:
                raise OSError("独占发布写入未推进")
            view = view[written:]
        os.fsync(stream_descriptor)
        if _identity(os.fstat(stream_descriptor)).size != len(payload):
            raise ValueError("独占发布大小不匹配")
        os.link(
            temporary,
            filename,
            src_dir_fd=descriptor,
            dst_dir_fd=descriptor,
            follow_symlinks=False,
        )
        published = True
    finally:
        if stream_descriptor is not None:
            with suppress(OSError):
                os.close(stream_descriptor)
        if published:
            with suppress(OSError):
                os.unlink(temporary, dir_fd=descriptor)
            with suppress(OSError):
                os.fsync(descriptor)
        else:
            with suppress(OSError):
                os.unlink(temporary, dir_fd=descriptor)


def _identity_sha256(identity: FileIdentity) -> str:
    payload = json.dumps(
        {
            "device": identity.device,
            "inode": identity.inode,
            "size": identity.size,
            "mtime_ns": identity.mtime_ns,
            "ctime_ns": identity.ctime_ns,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _artifact_manifest_sha256(artifacts: tuple[TraceArtifact, ...]) -> str:
    payload = json.dumps(
        [
            artifact.model_dump(
                mode="json",
                exclude_none=True,
                exclude_computed_fields=True,
            )
            for artifact in artifacts
        ],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path, *, max_bytes: int | None = None) -> str:
    try:
        if max_bytes is not None and (type(max_bytes) is not int or max_bytes <= 0):
            raise ValueError("大小上限必须是正整数")
        return _read_file_snapshot(
            path,
            max_bytes=max_bytes,
            capture_content=False,
            allow_empty=True,
        ).sha256
    except (OSError, ValueError):
        raise ValueError("文件摘要计算失败") from None


def load_machine_evidence(
    path: Path,
    *,
    workspace_root: Path,
) -> MachineEvidenceReport:
    try:
        report, snapshot = _load_machine_evidence_snapshot(
            path,
            workspace_root=workspace_root,
        )
        _assert_snapshot_current(snapshot)
        return report
    except (OSError, UnicodeDecodeError, ValueError, ValidationError):
        raise ValueError("机器证据报告非法或不可信") from None


def build_verified_gate_check(
    check_id: str,
    report_path: Path,
    *,
    workspace_root: Path,
    settings: Settings | None = None,
) -> GateCheck:
    return _build_verified_gate_check(
        check_id,
        report_path,
        workspace_root=workspace_root,
        settings=settings,
        allow_incomplete_real_media_run=False,
    )


def _build_verified_gate_check(
    check_id: str,
    report_path: Path,
    *,
    workspace_root: Path,
    settings: Settings | None = None,
    allow_incomplete_real_media_run: bool,
    staged_real_media_run: _StagedRealMediaRun | None = None,
) -> GateCheck:
    run_descriptor: int | None = None
    commit_snapshot: FileSnapshot | None = None
    live_authority: _LiveAuthorityState | None = None
    try:
        trusted_report_path = report_path
        if check_id == "real_media_chain":
            workspace, runtime = _trusted_roots(
                workspace_root,
                workspace_root / _RUNTIME_RELATIVE_ROOT,
            )
            trusted_report_path = _workspace_runtime_file(
                report_path,
                workspace,
                runtime,
            )
            if allow_incomplete_real_media_run:
                if staged_real_media_run is None:
                    raise ValueError("真实媒体 staged 验收缺少 writer 身份")
                _assert_staged_real_media_run(
                    staged_real_media_run,
                    trusted_report_path,
                )
            else:
                run_descriptor = _open_directory_descriptor(trusted_report_path.parent)
                commit_snapshot = _load_real_media_commit_snapshot(
                    run_descriptor,
                    trusted_report_path.parent,
                )
        report, report_snapshot = _load_machine_evidence_snapshot(
            trusted_report_path,
            workspace_root=workspace_root,
        )
        if report.check_id != check_id or check_id not in report.covered_items:
            raise ValueError("机器证据检查绑定不匹配")
        artifacts = verify_machine_artifacts(report, workspace_root=workspace_root)
        if check_id == "real_media_chain":
            _verify_real_media_report_boundary(report_snapshot.path, artifacts, workspace_root)
        if check_id in {
            "baidu_ocr_live",
            "qwen_live",
            "pyannote_live",
            "five_language_models",
        }:
            _verify_live_report_boundary(report_snapshot.path, artifacts)
            if isinstance(report.details, _LiveCheckDetails):
                live_authority = _load_live_authority_state(
                    report,
                    report_snapshot,
                    workspace_root,
                )
        from video_demo.evaluation.gate import GateCheck, _derive_machine_gate_status

        status, not_run_reason = _derive_machine_gate_status(
            check_id,
            report,
            artifacts,
            workspace_root,
            settings=settings,
        )
        if status != report.status or not_run_reason != report.not_run_reason:
            raise ValueError("机器证据自报状态与原始明细不一致")
        _assert_snapshot_current(report_snapshot)
        for values in artifacts.values():
            for artifact in values:
                _assert_snapshot_current(artifact.snapshot)
        if check_id == "real_media_chain":
            if allow_incomplete_real_media_run:
                assert staged_real_media_run is not None
                _assert_staged_real_media_run(
                    staged_real_media_run,
                    report_snapshot.path,
                )
            else:
                assert run_descriptor is not None and commit_snapshot is not None
                _assert_real_media_commit_current(
                    run_descriptor,
                    commit_snapshot,
                    report_snapshot,
                    report_snapshot.path.parent,
                )
        if live_authority is not None:
            _assert_live_authority_current(
                live_authority,
                report_snapshot,
                report_snapshot.path.parent,
            )
        workspace = workspace_root.resolve(strict=True)
        reference = EvidenceReference(
            kind=report.kind,
            level=report.level,
            relative_path=report_snapshot.path.relative_to(workspace).as_posix(),
            sha256=report_snapshot.sha256,
            covered_items=report.covered_items,
            summary=report.summary,
        )
        return GateCheck(
            check_id=check_id,
            status=status,
            evidence=(reference,),
            not_run_reason=not_run_reason,
        )
    except (OSError, ValueError, ValidationError, VideoDemoError):
        raise ValueError("机器证据无法形成可信门禁检查") from None
    finally:
        if run_descriptor is not None:
            with suppress(OSError):
                os.close(run_descriptor)
        if live_authority is not None:
            with suppress(OSError):
                os.close(live_authority.run_descriptor)


def _verify_real_media_report_boundary(
    report_path: Path,
    artifacts: dict[ArtifactRole, tuple[VerifiedArtifact, ...]],
    workspace_root: Path,
) -> None:
    allowed_roles: set[ArtifactRole] = {
        "AUDIT_REPORT",
        "COMMAND_STDOUT",
        "COMMAND_STDERR",
        "INPUT_MEDIA",
        "OUTPUT_MEDIA",
    }
    if set(artifacts) - allowed_roles:
        raise ValueError("真实媒体报告包含未声明产物角色")
    runtime = workspace_root.resolve(strict=True) / _RUNTIME_RELATIVE_ROOT
    relative = report_path.relative_to(runtime)
    if (
        len(relative.parts) != 4
        or relative.parts[:2] != ("eval", "reports")
        or relative.name != "real-media.json"
    ):
        raise ValueError("真实媒体权威报告路径非法")
    run_id = relative.parts[2]
    expected_root = runtime / "eval" / "reports" / run_id
    for role in ("AUDIT_REPORT", "COMMAND_STDOUT", "COMMAND_STDERR"):
        values = artifacts.get(role, ())
        if any(artifact.path.parent != expected_root for artifact in values):
            raise ValueError("真实媒体报告类产物必须与权威报告同 run")


def _verify_live_report_boundary(
    report_path: Path,
    artifacts: dict[ArtifactRole, tuple[VerifiedArtifact, ...]],
) -> None:
    raw_values = artifacts.get("AUDIT_REPORT", ())
    if len(raw_values) != 1 or raw_values[0].path.parent != report_path.parent:
        raise ValueError("live 权威报告与原始报告必须位于同一 run")


def _load_live_authority_state(
    report: MachineEvidenceReport,
    report_snapshot: FileSnapshot,
    workspace_root: Path,
) -> _LiveAuthorityState:
    details = report.details
    if not isinstance(details, _LiveCheckDetails):
        raise ValueError("已执行 live 报告缺少 detail")
    workspace, runtime = _trusted_roots(
        workspace_root,
        workspace_root / _RUNTIME_RELATIVE_ROOT,
    )
    relative = report_snapshot.path.relative_to(runtime)
    if (
        len(relative.parts) != 4
        or relative.parts[:2] != ("eval", "reports")
        or relative.name != f"{report.check_id}.json"
    ):
        raise ValueError("live machine report 权威路径非法")
    run_id = relative.parts[2]
    run_descriptor = _open_directory_descriptor(report_snapshot.path.parent)
    try:
        run_identity = _identity(os.fstat(run_descriptor))
        journal_path = (
            runtime
            / "eval"
            / "live-authority"
            / run_id
            / f"{report.check_id}.json"
        )
        trusted_journal = _workspace_runtime_file(
            journal_path,
            workspace,
            runtime,
        )
        journal_snapshot = _read_file_snapshot(
            trusted_journal,
            max_bytes=16 * 1024,
            capture_content=True,
        )
        if journal_snapshot.content is None:
            raise ValueError("live authority journal 缺少正文")
        record = LiveAuthorityRecord.model_validate_json(journal_snapshot.content)
        expected_report_path = report_snapshot.path.relative_to(workspace).as_posix()
        if (
            record.check_id != report.check_id
            or record.evaluation_run_id != run_id
            or record.machine_report_path != expected_report_path
            or record.machine_report_sha256 != report_snapshot.sha256
            or record.raw_report_sha256 != details.raw_report_sha256
            or record.settings_fingerprint != details.settings_fingerprint
            or record.implementation_sha256 != details.implementation_sha256
            or record.machine_report_identity_sha256
            != _identity_sha256(report_snapshot.identity)
            or record.report_run_directory_identity_sha256
            != _identity_sha256(run_identity)
            or record.artifact_manifest_sha256
            != _artifact_manifest_sha256(report.artifacts)
        ):
            raise ValueError("live authority journal 与报告发布事实不匹配")
        return _LiveAuthorityState(
            journal_snapshot=journal_snapshot,
            run_descriptor=run_descriptor,
            run_identity=run_identity,
        )
    except BaseException:
        os.close(run_descriptor)
        raise


def _assert_live_authority_current(
    authority: _LiveAuthorityState,
    report_snapshot: FileSnapshot,
    run_path: Path,
) -> None:
    _assert_snapshot_current(authority.journal_snapshot)
    _assert_snapshot_current(report_snapshot)
    held_identity = _identity(os.fstat(authority.run_descriptor))
    current_descriptor = _open_directory_descriptor(run_path)
    try:
        current_identity = _identity(os.fstat(current_descriptor))
    finally:
        os.close(current_descriptor)
    if (
        held_identity != authority.run_identity
        or current_identity != authority.run_identity
    ):
        raise ValueError("live authority 验收期间报告 run 身份发生变化")


def _assert_staged_real_media_run(
    staged: _StagedRealMediaRun,
    report_path: Path,
) -> None:
    if (
        not _same_directory(staged.directory_identity, _identity(os.fstat(staged.descriptor)))
        or _identity(
            os.stat(_REAL_MEDIA_INCOMPLETE_MARKER, dir_fd=staged.descriptor)
        )
        != staged.marker_identity
    ):
        raise ValueError("真实媒体 staged run 身份发生变化")
    path_descriptor = _open_directory_descriptor(report_path.parent)
    try:
        path_identity = _identity(os.fstat(path_descriptor))
    finally:
        os.close(path_descriptor)
    if not _same_directory(staged.directory_identity, path_identity):
        raise ValueError("真实媒体 staged run 身份发生变化")


def _load_real_media_commit_snapshot(
    descriptor: int,
    run_path: Path,
) -> FileSnapshot:
    snapshot = _read_file_snapshot_at(
        descriptor,
        run_path / _REAL_MEDIA_COMMIT_RECORD,
        max_bytes=4 * 1024,
    )
    if snapshot.content is None:
        raise ValueError("真实媒体提交记录缺少正文")
    decoded = _decode_strict_json(snapshot.content.decode("utf-8"))
    if not isinstance(decoded, dict):
        raise ValueError("真实媒体提交记录顶层必须是对象")
    _RealMediaCommitRecord.model_validate(decoded)
    return snapshot


def _assert_real_media_commit_current(
    descriptor: int,
    commit_snapshot: FileSnapshot,
    report_snapshot: FileSnapshot,
    run_path: Path,
) -> None:
    current = _read_file_snapshot_at(
        descriptor,
        run_path / _REAL_MEDIA_COMMIT_RECORD,
        max_bytes=4 * 1024,
    )
    if current != commit_snapshot:
        raise ValueError("真实媒体提交记录发生变化")
    if current.content is None:
        raise ValueError("真实媒体提交记录缺少正文")
    record = _RealMediaCommitRecord.model_validate_json(current.content)
    resolved_descriptor = _open_directory_descriptor(run_path)
    try:
        path_identity = _identity(os.fstat(resolved_descriptor))
    finally:
        os.close(resolved_descriptor)
    if (
        record.evaluation_run_id != run_path.name
        or record.authority_sha256 != report_snapshot.sha256
        or not _same_directory(
            _identity(os.fstat(descriptor)),
            path_identity,
        )
    ):
        raise ValueError("真实媒体提交记录与 authority 或 run 不匹配")


def _same_directory(left: FileIdentity, right: FileIdentity) -> bool:
    return (left.device, left.inode) == (right.device, right.inode)


def verify_machine_artifacts(
    report: MachineEvidenceReport,
    *,
    workspace_root: Path,
) -> dict[ArtifactRole, tuple[VerifiedArtifact, ...]]:
    try:
        workspace, runtime = _trusted_roots(
            workspace_root, workspace_root / _RUNTIME_RELATIVE_ROOT
        )
        grouped: dict[ArtifactRole, list[VerifiedArtifact]] = {}
        media_limit = (
            report.details.max_video_bytes
            if isinstance(report.details, AuthorizedDatasetDetails)
            else _DEFAULT_MEDIA_BYTES
        )
        for artifact in report.artifacts:
            path = _workspace_runtime_file(
                Path(artifact.relative_path), workspace, runtime
            )
            default_limit = (
                media_limit if artifact.role in _MEDIA_ROLES else _MAX_MACHINE_BYTES
            )
            limit = artifact.max_bytes if artifact.max_bytes is not None else default_limit
            _artifact_limit(artifact.role, limit)
            snapshot = _read_file_snapshot(
                path,
                max_bytes=limit,
                capture_content=artifact.role not in _MEDIA_ROLES,
                allow_empty=artifact.role in {"COMMAND_STDOUT", "COMMAND_STDERR"},
            )
            if snapshot.sha256 != artifact.sha256:
                raise ValueError("机器证据产物摘要不匹配")
            _validate_artifact_content(artifact.role, snapshot.content)
            grouped.setdefault(artifact.role, []).append(
                VerifiedArtifact(reference=artifact, snapshot=snapshot)
            )
        result = {role: tuple(values) for role, values in grouped.items()}
        _require_trace_outputs(report, result)
        return result
    except (OSError, RecursionError, ValueError):
        raise ValueError("机器证据产物重验失败") from None


def _require_trace_outputs(
    report: MachineEvidenceReport,
    artifacts: dict[ArtifactRole, tuple[VerifiedArtifact, ...]],
) -> None:
    required: tuple[tuple[ArtifactRole, str], ...] = (
        ("COMMAND_STDOUT", report.details.trace.stdout_sha256),
        ("COMMAND_STDERR", report.details.trace.stderr_sha256),
    )
    for role, expected in required:
        matches = tuple(
            artifact
            for artifact in artifacts.get(role, ())
            if artifact.reference.sha256 == expected
        )
        if len(matches) != 1:
            raise ValueError("机器证据命令输出绑定不完整")


def _artifact_limit(role: ArtifactRole, supplied: int | None) -> int:
    if supplied is not None and (type(supplied) is not int or supplied <= 0):
        raise ValueError("产物大小上限必须是正整数")
    if role in _MEDIA_ROLES:
        if supplied is not None and supplied > _DEFAULT_MEDIA_BYTES:
            raise ValueError("媒体产物不得放宽 4 GiB 硬上限")
        return supplied if supplied is not None else _DEFAULT_MEDIA_BYTES
    if supplied is not None and supplied > _MAX_MACHINE_BYTES:
        raise ValueError("机器 JSON 和命令输出不得放宽 64 MiB 上限")
    return supplied if supplied is not None else _MAX_MACHINE_BYTES


def _trusted_roots(workspace_root: Path, runtime_root: Path) -> tuple[Path, Path]:
    _reject_symlink_components(workspace_root)
    _reject_symlink_components(runtime_root)
    if not workspace_root.is_dir() or not runtime_root.is_dir():
        raise ValueError("可信根必须是目录")
    workspace = workspace_root.resolve(strict=True)
    expected = workspace_root / _RUNTIME_RELATIVE_ROOT
    if runtime_root.absolute() != expected.absolute():
        raise ValueError("运行根不是工作区固定目录")
    runtime = runtime_root.resolve(strict=True)
    if runtime != (workspace / _RUNTIME_RELATIVE_ROOT).resolve(strict=True):
        raise ValueError("运行根解析后不匹配")
    return workspace, runtime


def _prepare_runtime_target(runtime: Path, relative_path: Path) -> Path:
    normalized = _normalize_runtime_relative(relative_path)
    _reject_public_live_authority_write(normalized)
    current = runtime
    for part in normalized.parts[:-1]:
        current /= part
        if current.is_symlink():
            raise ValueError("目标父目录不得是符号链接")
        current.mkdir(exist_ok=True)
        if not current.is_dir():
            raise ValueError("目标父路径不是目录")
    target = current / normalized.name
    if target.is_symlink() or (target.exists() and not target.is_file()):
        raise ValueError("目标路径非法")
    return target


def _reject_public_live_authority_write(relative_path: Path) -> None:
    relative_parts = tuple(part.casefold() for part in relative_path.parts)
    reserved_parts = tuple(
        part.casefold() for part in _LIVE_AUTHORITY_RELATIVE_ROOT.parts
    )
    if relative_parts[: len(reserved_parts)] == reserved_parts:
        raise ValueError("live authority 只能通过独占发布器写入")


def _open_or_create_runtime_parent(runtime: Path, relative_parent: Path) -> int:
    _require_artifact_fd_capabilities()
    descriptor = _open_directory_descriptor(runtime)
    try:
        for part in relative_parent.parts:
            if part in {"", "."}:
                continue
            with suppress(FileExistsError):
                os.mkdir(part, dir_fd=descriptor)
            child = _open_child_directory(descriptor, part)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _validate_writable_artifact_leaf(descriptor: int, filename: str) -> None:
    try:
        details = os.stat(filename, dir_fd=descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return
    if not stat.S_ISREG(details.st_mode):
        raise ValueError("目标路径非法")


def _assert_runtime_parent_current(
    runtime: Path,
    relative_parent: Path,
    held_descriptor: int,
) -> None:
    current = _open_directory_descriptor(runtime / relative_parent)
    try:
        held = os.fstat(held_descriptor)
        observed = os.fstat(current)
        if (held.st_dev, held.st_ino) != (observed.st_dev, observed.st_ino):
            raise ValueError("artifact 父目录身份已改变")
    finally:
        os.close(current)


def _require_artifact_fd_capabilities() -> None:
    if (
        os.name != "posix"
        or not all(hasattr(os, name) for name in ("O_DIRECTORY", "O_NOFOLLOW", "O_CLOEXEC"))
        or not _ARTIFACT_DIR_FD_FUNCTIONS.issubset(os.supports_dir_fd)
        or os.stat not in os.supports_follow_symlinks
    ):
        raise ValueError("当前平台缺少 artifact fd 安全能力")


def _runtime_existing_file(runtime: Path, relative_path: Path) -> Path:
    normalized = _normalize_runtime_relative(relative_path)
    current = runtime
    for part in normalized.parts:
        current /= part
        if current.is_symlink():
            raise ValueError("产物路径不得包含符号链接")
    if not current.is_file():
        raise ValueError("产物不存在或不是普通文件")
    resolved = current.resolve(strict=True)
    if not resolved.is_relative_to(runtime):
        raise ValueError("产物逃逸运行根")
    return resolved


def _workspace_runtime_file(
    candidate: Path,
    workspace: Path,
    runtime: Path,
) -> Path:
    unresolved = candidate if candidate.is_absolute() else workspace / candidate
    try:
        relative = unresolved.absolute().relative_to(workspace.absolute())
    except ValueError:
        raise ValueError("路径不在工作区") from None
    current = workspace
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise ValueError("路径不得包含符号链接")
    if not current.is_file():
        raise ValueError("路径不是普通文件")
    resolved = current.resolve(strict=True)
    if not resolved.is_relative_to(runtime):
        raise ValueError("机器证据不在运行根")
    return resolved


def _normalize_runtime_relative(relative_path: Path) -> Path:
    if relative_path.is_absolute() or not relative_path.parts:
        raise ValueError("路径必须是运行根相对路径")
    if ".." in relative_path.parts or "." in relative_path.parts:
        raise ValueError("路径不是规范相对路径")
    if relative_path.as_posix() in ("", "."):
        raise ValueError("路径必须指向文件")
    return relative_path


def _validate_workspace_relative(value: str) -> None:
    normalized = PurePosixPath(value)
    if (
        normalized.is_absolute()
        or ".." in normalized.parts
        or value != normalized.as_posix()
        or normalized.as_posix() in ("", ".")
    ):
        raise ValueError("证据路径必须是规范的工作区相对路径")


def _require_runtime_relative(value: str) -> None:
    path = PurePosixPath(value)
    if path == _RUNTIME_RELATIVE_ROOT or not path.is_relative_to(
        _RUNTIME_RELATIVE_ROOT
    ):
        raise ValueError("机器证据必须位于固定运行根")


def _reject_symlink_components(path: Path) -> None:
    absolute = path.absolute()
    for component in (absolute, *absolute.parents):
        if component.is_symlink():
            raise ValueError("可信路径不得包含符号链接")


def _load_machine_evidence_snapshot(
    path: Path,
    *,
    workspace_root: Path,
) -> tuple[MachineEvidenceReport, FileSnapshot]:
    workspace, runtime = _trusted_roots(
        workspace_root,
        workspace_root / _RUNTIME_RELATIVE_ROOT,
    )
    report = _workspace_runtime_file(path, workspace, runtime)
    if report.name.endswith(".part"):
        raise ValueError("临时文件不能作为机器报告")
    snapshot = _read_file_snapshot(
        report,
        max_bytes=_MAX_MACHINE_BYTES,
        capture_content=True,
    )
    if snapshot.content is None:
        raise ValueError("机器报告快照缺少正文")
    decoded = _decode_strict_json(snapshot.content.decode("utf-8"))
    if not isinstance(decoded, dict):
        raise ValueError("机器报告顶层必须是对象")
    parsed = MachineEvidenceReport.model_validate(decoded)
    _assert_snapshot_current(snapshot)
    return parsed, snapshot


def _read_file_snapshot(
    path: Path,
    *,
    max_bytes: int | None,
    capture_content: bool,
    allow_empty: bool = False,
    require_utf8: bool = True,
) -> FileSnapshot:
    descriptor = _open_regular_file(path)
    try:
        before = _identity(os.fstat(descriptor))
        if max_bytes is not None and before.size > max_bytes:
            raise ValueError("文件超过大小上限")
        digest = hashlib.sha256()
        content = bytearray() if capture_content else None
        total = 0
        while chunk := os.read(descriptor, _CHUNK_BYTES):
            total += len(chunk)
            if max_bytes is not None and total > max_bytes:
                raise ValueError("文件超过大小上限")
            digest.update(chunk)
            if content is not None:
                content.extend(chunk)
        after = _identity(os.fstat(descriptor))
        if before != after or total != before.size:
            raise ValueError("文件读取期间发生变化")
        if total == 0 and not allow_empty:
            raise ValueError("机器文件不能为空")
        encoded = bytes(content) if content is not None else None
        if encoded is not None and require_utf8:
            encoded.decode("utf-8")
        return FileSnapshot(
            path=path,
            identity=before,
            sha256=digest.hexdigest(),
            content=encoded,
        )
    finally:
        os.close(descriptor)


def _read_file_snapshot_at(
    parent_descriptor: int,
    path: Path,
    *,
    max_bytes: int,
    allow_empty: bool = False,
) -> FileSnapshot:
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path.name, flags, dir_fd=parent_descriptor)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("提交记录不是普通文件")
        before = _identity(metadata)
        if (before.size == 0 and not allow_empty) or before.size > max_bytes:
            raise ValueError("提交记录大小非法")
        content = bytearray()
        while chunk := os.read(descriptor, min(_CHUNK_BYTES, max_bytes + 1)):
            content.extend(chunk)
            if len(content) > max_bytes:
                raise ValueError("提交记录超过大小上限")
        after = _identity(os.fstat(descriptor))
        if before != after or len(content) != before.size:
            raise ValueError("提交记录读取期间发生变化")
        encoded = bytes(content)
        encoded.decode("utf-8")
        return FileSnapshot(
            path=path,
            identity=before,
            sha256=hashlib.sha256(encoded).hexdigest(),
            content=encoded,
        )
    finally:
        os.close(descriptor)


def _read_open_file_snapshot(
    descriptor: int,
    path: Path,
    *,
    max_bytes: int,
    allow_empty: bool = False,
) -> FileSnapshot:
    before = _identity(os.fstat(descriptor))
    if (before.size == 0 and not allow_empty) or before.size > max_bytes:
        raise ValueError("提交记录大小非法")
    os.lseek(descriptor, 0, os.SEEK_SET)
    content = bytearray()
    while chunk := os.read(descriptor, min(_CHUNK_BYTES, max_bytes + 1)):
        content.extend(chunk)
        if len(content) > max_bytes:
            raise ValueError("提交记录超过大小上限")
    after = _identity(os.fstat(descriptor))
    if before != after or len(content) != before.size:
        raise ValueError("提交记录读取期间发生变化")
    encoded = bytes(content)
    encoded.decode("utf-8")
    return FileSnapshot(
        path=path,
        identity=before,
        sha256=hashlib.sha256(encoded).hexdigest(),
        content=encoded,
    )


def _open_regular_file(path: Path) -> int:
    parent_descriptor = _open_directory_descriptor(path.parent)
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    if hasattr(os, "O_NONBLOCK"):
        flags |= os.O_NONBLOCK
    descriptor: int | None = None
    try:
        descriptor = os.open(path.name, flags, dir_fd=parent_descriptor)
    finally:
        os.close(parent_descriptor)
    try:
        if stat.S_ISREG(os.fstat(descriptor).st_mode):
            return descriptor
        raise ValueError("摘要目标不是普通文件")
    except BaseException:
        os.close(descriptor)
        raise


def _open_directory_descriptor(path: Path) -> int:
    absolute = path.absolute()
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(absolute.anchor, flags)
    try:
        for component in absolute.parts[1:]:
            child = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _open_child_directory(parent_descriptor: int, name: str) -> int:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    return os.open(name, flags, dir_fd=parent_descriptor)


def _create_run_incomplete_marker(descriptor: int) -> FileIdentity:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    marker_descriptor: int | None = None
    try:
        marker_descriptor = os.open(
            _REAL_MEDIA_INCOMPLETE_MARKER,
            flags,
            0o600,
            dir_fd=descriptor,
        )
        marker_identity = _identity(os.fstat(marker_descriptor))
        os.fsync(marker_descriptor)
        os.close(marker_descriptor)
        marker_descriptor = None
        os.fsync(descriptor)
        return marker_identity
    except BaseException:
        if marker_descriptor is not None:
            with suppress(OSError):
                os.close(marker_descriptor)
        raise


def _publish_real_media_commit(
    descriptor: int,
    record: _RealMediaCommitRecord,
) -> None:
    payload = record.model_dump_json().encode("utf-8")
    _write_run_payload(
        descriptor,
        _REAL_MEDIA_COMMIT_RECORD,
        payload,
        preserve_published_on_fsync_failure=True,
    )


def _validate_run_leaf(filename: str) -> None:
    path = PurePosixPath(filename)
    if not filename or path.name != filename or filename in {".", ".."}:
        raise ValueError("run 写入器只接受叶文件名")


def _validate_stable_id(value: str, name: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9_-]{3,128}", value) is None:
        raise ValueError(f"{name} 必须是稳定标识")


def _write_bound_artifact_payload(
    descriptor: int,
    filename: str,
    payload: bytes,
    path: Path,
    role: ArtifactRole,
    max_bytes: int,
    assert_parent_current: Callable[[], None],
) -> _PublishedArtifact:
    _validate_run_leaf(filename)
    temporary = f".{filename}.{uuid.uuid4().hex}.part"
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    stream_descriptor: int | None = None
    published = False
    try:
        stream_descriptor = os.open(temporary, flags, 0o600, dir_fd=descriptor)
        view = memoryview(payload)
        while view:
            written = os.write(stream_descriptor, view)
            if written <= 0:
                raise OSError("artifact 原子写入未推进")
            view = view[written:]
        os.fsync(stream_descriptor)
        published_identity = _identity(os.fstat(stream_descriptor))
        if published_identity.size != len(payload):
            raise ValueError("artifact 发布大小不匹配")
        expected_sha256 = hashlib.sha256(payload).hexdigest()
        snapshot = _read_open_file_snapshot(
            stream_descriptor,
            path,
            max_bytes=max_bytes,
            allow_empty=role in {"COMMAND_STDOUT", "COMMAND_STDERR"},
        )
        if snapshot.identity != published_identity or snapshot.sha256 != expected_sha256:
            raise ValueError("本次 artifact 发布内容已改变")
        _validate_artifact_content(role, snapshot.content)
        assert_parent_current()
        os.rename(temporary, filename, src_dir_fd=descriptor, dst_dir_fd=descriptor)
        published = True
        os.fsync(descriptor)
        result = _PublishedArtifact(
            descriptor=stream_descriptor,
            identity=published_identity,
            sha256=expected_sha256,
        )
        stream_descriptor = None
        return result
    except BaseException:
        if not published:
            with suppress(BaseException):
                os.unlink(temporary, dir_fd=descriptor)
        raise
    finally:
        if stream_descriptor is not None:
            with suppress(BaseException):
                os.close(stream_descriptor)


def _assert_published_artifact_current(
    descriptor: int,
    filename: str,
    expected: FileIdentity,
) -> None:
    details = os.stat(filename, dir_fd=descriptor, follow_symlinks=False)
    if not stat.S_ISREG(details.st_mode) or not _same_inode(_identity(details), expected):
        raise ValueError("artifact 发布目标已被替换")


def _same_inode(left: FileIdentity, right: FileIdentity) -> bool:
    return (left.device, left.inode) == (right.device, right.inode)


def _write_run_payload(
    descriptor: int,
    filename: str,
    payload: bytes,
    *,
    preserve_published_on_fsync_failure: bool = False,
) -> None:
    _validate_run_leaf(filename)
    temporary = f".{filename}.{uuid.uuid4().hex}.part"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    stream_descriptor: int | None = None
    published = False
    try:
        stream_descriptor = os.open(temporary, flags, 0o600, dir_fd=descriptor)
        view = memoryview(payload)
        while view:
            written = os.write(stream_descriptor, view)
            if written <= 0:
                raise OSError("run 原子写入未推进")
            view = view[written:]
        os.fsync(stream_descriptor)
        os.close(stream_descriptor)
        stream_descriptor = None
        os.rename(temporary, filename, src_dir_fd=descriptor, dst_dir_fd=descriptor)
        published = True
        if preserve_published_on_fsync_failure:
            try:
                os.fsync(descriptor)
            except Exception:
                return
        else:
            os.fsync(descriptor)
    except BaseException:
        if stream_descriptor is not None:
            with suppress(BaseException):
                os.close(stream_descriptor)
        if published and not preserve_published_on_fsync_failure:
            with suppress(BaseException):
                os.unlink(filename, dir_fd=descriptor)
            with suppress(BaseException):
                os.fsync(descriptor)
        elif not published:
            with suppress(BaseException):
                os.unlink(temporary, dir_fd=descriptor)
        raise


def _identity(metadata: os.stat_result) -> FileIdentity:
    return FileIdentity(
        device=metadata.st_dev,
        inode=metadata.st_ino,
        size=metadata.st_size,
        mtime_ns=metadata.st_mtime_ns,
        ctime_ns=metadata.st_ctime_ns,
    )


def _assert_snapshot_current(snapshot: FileSnapshot) -> None:
    descriptor = _open_regular_file(snapshot.path)
    try:
        if _identity(os.fstat(descriptor)) != snapshot.identity:
            raise ValueError("证据文件身份发生变化")
    finally:
        os.close(descriptor)


def _validate_artifact_content(role: ArtifactRole, content: bytes | None) -> None:
    if role not in _TEXT_MACHINE_ROLES:
        return
    if content is None:
        raise ValueError("机器产物缺少受限文本快照")
    text = content.decode("utf-8")
    if role in _JSONL_MACHINE_ROLES:
        _validate_jsonl_values(text)
    elif role in _JSON_MACHINE_ROLES:
        payload = _decode_strict_json(text)
        _validate_persisted_value(payload)
        if role == "PROVIDER_RESPONSE":
            validators = (
                LiveExecutionSummary.model_validate,
                ProviderResponseSummary.model_validate,
            )
            if not any(_accepts_payload(validate, payload) for validate in validators):
                raise ValueError("供应商响应必须是无正文摘要") from None
    else:
        _validate_persisted_string(text)
    if role in {"COMMAND_STDOUT", "COMMAND_STDERR"} and _contains_json_body(text):
        raise ValueError("命令输出不得持久化未结构化 JSON 正文")


def _validate_jsonl_values(value: str) -> None:
    for line in value.splitlines():
        if line.strip():
            payload = _decode_strict_json(line)
            _validate_persisted_value(payload)


def _accepts_payload(
    validate: Callable[[Any], Any],
    payload: Any,
) -> bool:
    try:
        validate(payload)
    except (ValidationError, ValueError):
        return False
    return True


@dataclass
class _JsonCandidateFrame:
    kind: Literal["array", "object"]
    state: str
    start: int
    depth: int
    accepted_by_parent: bool
    valid: bool = True
    in_string: bool = False
    string_escaped: bool = False
    unicode_escape_remaining: int = 0


@dataclass
class _JsonCandidateLane:
    frames: list[_JsonCandidateFrame] = field(default_factory=list)
    expected_closers: bytearray = field(default_factory=bytearray)

    @property
    def active(self) -> bool:
        return bool(self.expected_closers)


def _contains_json_body(value: str) -> bool:
    lanes = (_JsonCandidateLane(), _JsonCandidateLane())
    return _scan_command_candidate_branch(
        value,
        decoder=_strict_json_decoder(),
        lanes=lanes,
    )


def _scan_command_candidate_branch(
    value: str,
    *,
    decoder: json.JSONDecoder,
    lanes: tuple[_JsonCandidateLane, _JsonCandidateLane],
) -> bool:
    scalar_ends = [0, 0]
    index = 0
    end = len(value)
    while index < end:
        character = value[index]
        opener_tracked = False
        for lane_index, lane in enumerate(lanes):
            if not lane.active:
                continue
            matched, scalar_ends[lane_index], opened = (
                _advance_candidate_lane_character(
                    value,
                    lane,
                    decoder=decoder,
                    character=character,
                    index=index,
                    end=end,
                    scalar_end=scalar_ends[lane_index],
                )
            )
            if matched:
                return True
            opener_tracked = opener_tracked or opened
            if not lane.active:
                lane.frames.clear()
        if character in "[{" and not opener_tracked:
            idle_lane = next((lane for lane in lanes if not lane.active), None)
            if idle_lane is None:
                return True
            _open_candidate_frame(
                idle_lane,
                character=character,
                start=index,
            )
        index += 1
    return False


def _advance_candidate_lane_character(
    value: str,
    lane: _JsonCandidateLane,
    *,
    decoder: json.JSONDecoder,
    character: str,
    index: int,
    end: int,
    scalar_end: int,
) -> tuple[bool, int, bool]:
    frame = _current_candidate_frame(
        lane.frames,
        depth=len(lane.expected_closers),
    )
    if frame is not None and frame.in_string:
        string_was_valid = frame.valid
        _advance_candidate_string_character(frame, character)
        if string_was_valid and not frame.valid:
            lane.frames.clear()
            lane.expected_closers.clear()
            return False, 0, False
        return False, scalar_end, False
    if index < scalar_end:
        return False, scalar_end, False
    if character in "[{":
        _open_candidate_frame(lane, character=character, start=index)
        return False, scalar_end, True
    if character == '"':
        if frame is not None and _candidate_expects_string(frame):
            frame.in_string = True
            frame.string_escaped = False
            frame.unicode_escape_remaining = 0
        elif frame is not None and frame.valid:
            frame.valid = False
        return False, scalar_end, False
    if character in "]}":
        matched = _close_candidate_frame(
            value,
            lane,
            decoder=decoder,
            character=character,
            end=index + 1,
        )
        return matched, scalar_end, False
    if frame is None or not frame.valid or character in " \t\r\n":
        return False, scalar_end, False
    if character in ",:":
        _consume_candidate_separator(frame, character)
        return False, scalar_end, False
    if _candidate_expects_value(frame):
        candidate_end = _scan_command_json_scalar(value, start=index, end=end)
        if candidate_end >= 0:
            _finish_candidate_value(frame)
            return False, candidate_end, False
    frame.valid = False
    return False, scalar_end, False


def _open_candidate_frame(
    lane: _JsonCandidateLane,
    *,
    character: str,
    start: int,
) -> None:
    parent = _current_candidate_frame(
        lane.frames,
        depth=len(lane.expected_closers),
    )
    accepted_by_parent = bool(
        parent and _consume_candidate_container(parent)
    )
    lane.expected_closers.append(ord("]" if character == "[" else "}"))
    kind: Literal["array", "object"] = (
        "array" if character == "[" else "object"
    )
    state = "value_or_end" if kind == "array" else "key_or_end"
    depth = len(lane.expected_closers)
    slot_index = (depth - 1) % _MAX_COMMAND_JSON_FRAMES
    if depth > len(lane.frames) and len(lane.frames) < _MAX_COMMAND_JSON_FRAMES:
        lane.frames.append(
            _JsonCandidateFrame(
                kind=kind,
                state=state,
                start=start,
                depth=depth,
                accepted_by_parent=accepted_by_parent,
            )
        )
        return
    # 按绝对深度复用固定槽，避免窗口满后搬移整段 frame 引用。
    frame = lane.frames[slot_index]
    frame.kind = kind
    frame.state = state
    frame.start = start
    frame.depth = depth
    frame.accepted_by_parent = accepted_by_parent
    frame.valid = True
    frame.in_string = False
    frame.string_escaped = False
    frame.unicode_escape_remaining = 0


def _close_candidate_frame(
    value: str,
    lane: _JsonCandidateLane,
    *,
    decoder: json.JSONDecoder,
    character: str,
    end: int,
) -> bool:
    while lane.expected_closers:
        depth = len(lane.expected_closers)
        frame = _current_candidate_frame(lane.frames, depth=depth)
        expected_closer = lane.expected_closers.pop()
        if ord(character) != expected_closer:
            _invalidate_candidate_parent(lane.frames, frame, depth=depth)
            continue
        if (
            frame is not None
            and frame.valid
            and _candidate_can_close(frame)
            and _decode_closed_candidate(
                value,
                decoder=decoder,
                start=frame.start,
                end=end,
            )
        ):
            return True
        _invalidate_candidate_parent(lane.frames, frame, depth=depth)
        return False
    lane.frames.clear()
    return False


def _invalidate_candidate_parent(
    frames: list[_JsonCandidateFrame],
    frame: _JsonCandidateFrame | None,
    *,
    depth: int,
) -> None:
    if frame is None or not frame.accepted_by_parent:
        return
    parent = _current_candidate_frame(frames, depth=depth - 1)
    if parent is not None:
        parent.valid = False


def _current_candidate_frame(
    frames: list[_JsonCandidateFrame],
    *,
    depth: int,
) -> _JsonCandidateFrame | None:
    if not frames or depth <= 0:
        return None
    frame = frames[(depth - 1) % _MAX_COMMAND_JSON_FRAMES]
    if frame.depth == depth:
        return frame
    return None


def _decode_closed_candidate(
    value: str,
    *,
    decoder: json.JSONDecoder,
    start: int,
    end: int,
) -> bool:
    try:
        decoded, decoded_end = decoder.raw_decode(value, start)
    except json.JSONDecodeError:
        return False
    except (RecursionError, ValueError):
        raise ValueError("机器证据 JSON 非法或超出解析边界") from None
    return decoded_end == end and isinstance(decoded, (dict, list))


def _advance_candidate_string_character(
    frame: _JsonCandidateFrame,
    character: str,
) -> None:
    if not frame.in_string:
        return
    if frame.unicode_escape_remaining:
        if character not in "0123456789abcdefABCDEF":
            frame.valid = False
        frame.unicode_escape_remaining -= 1
        return
    if frame.string_escaped:
        frame.string_escaped = False
        if character == "u":
            frame.unicode_escape_remaining = 4
        elif character not in '"\\/bfnrt':
            frame.valid = False
        return
    if character == '"':
        frame.in_string = False
        _consume_candidate_string(frame)
        return
    if character == "\\":
        frame.string_escaped = True
    elif ord(character) < 0x20:
        frame.valid = False


def _consume_candidate_container(frame: _JsonCandidateFrame) -> bool:
    if not frame.valid or not _candidate_expects_value(frame):
        frame.valid = False
        return False
    _finish_candidate_value(frame)
    return True


def _consume_candidate_string(frame: _JsonCandidateFrame) -> None:
    if frame.kind == "object" and frame.state in {"key_or_end", "key"}:
        frame.state = "colon"
        return
    if _candidate_expects_value(frame):
        _finish_candidate_value(frame)
        return
    frame.valid = False


def _consume_candidate_separator(
    frame: _JsonCandidateFrame,
    separator: str,
) -> None:
    if separator == ":":
        if frame.kind == "object" and frame.state == "colon":
            frame.state = "value"
        else:
            frame.valid = False
        return
    if frame.state != "comma_or_end":
        frame.valid = False
        return
    frame.state = "value" if frame.kind == "array" else "key"


def _candidate_expects_value(frame: _JsonCandidateFrame) -> bool:
    if frame.kind == "array":
        return frame.state in {"value_or_end", "value"}
    return frame.state == "value"


def _candidate_expects_string(frame: _JsonCandidateFrame) -> bool:
    if not frame.valid:
        return False
    if frame.kind == "object" and frame.state in {"key_or_end", "key"}:
        return True
    return _candidate_expects_value(frame)


def _finish_candidate_value(frame: _JsonCandidateFrame) -> None:
    frame.state = "comma_or_end"


def _candidate_can_close(frame: _JsonCandidateFrame) -> bool:
    if frame.kind == "array":
        return frame.state in {"value_or_end", "comma_or_end"}
    return frame.state in {"key_or_end", "comma_or_end"}


def _scan_command_json_scalar(value: str, *, start: int, end: int) -> int:
    for literal in ("true", "false", "null", "NaN", "Infinity", "-Infinity"):
        if value.startswith(literal, start, end):
            return start + len(literal)
    match = _JSON_NUMBER_PATTERN.match(value, start, end)
    return match.end() if match is not None else -1


def _decode_strict_json(value: str) -> Any:
    _validate_json_nesting(value)
    try:
        return _strict_json_decoder().decode(value)
    except (RecursionError, ValueError):
        raise ValueError("机器证据 JSON 非法") from None


def _strict_json_decoder() -> json.JSONDecoder:
    return json.JSONDecoder(
        object_pairs_hook=_reject_duplicate_json_keys,
        parse_constant=_reject_non_standard_json_constant,
    )


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, item in pairs:
        if key in result:
            raise ValueError("机器证据 JSON 对象键不得重复")
        result[key] = item
    return result


def _reject_non_standard_json_constant(_constant: str) -> Any:
    raise ValueError("机器证据 JSON 不得包含非标准常量")


def _validate_json_nesting(
    value: str,
    *,
    start: int = 0,
    end: int | None = None,
) -> None:
    expected_closers: list[str] = []
    in_string = False
    escaped = False
    stop = len(value) if end is None else end
    for index in range(start, stop):
        character = value[index]
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
            continue
        if character in "[{":
            expected_closers.append("]" if character == "[" else "}")
            if len(expected_closers) > _MAX_JSON_NESTING:
                raise ValueError("机器证据 JSON 嵌套超过上限")
            continue
        if character not in "]}":
            continue
        if not expected_closers or character != expected_closers[-1]:
            return
        expected_closers.pop()
        if not expected_closers:
            return


def _validate_persisted_value(value: Any) -> None:
    if isinstance(value, str):
        _validate_persisted_string(value)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _validate_persisted_string(str(key))
            _validate_persisted_value(item)
        return
    if isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            _validate_persisted_value(item)


def _validate_persisted_string(value: str) -> None:
    if any(
        unicodedata.category(character) == "Cc" and character not in "\t\n\r"
        for character in value
    ):
        raise ValueError("机器证据字符串包含非法控制字符")
    if _SECRET_PATTERN.search(value):
        raise ValueError("机器证据字符串包含明显 Secret")
    if _DATA_URL_PATTERN.search(value):
        raise ValueError("机器证据字符串不得包含 Data URL")
    path_checked = "DEV_NULL" if value == _MYPY_DEV_NULL_ARGUMENT else value
    if (
        _POSIX_ABSOLUTE_PATTERN.search(path_checked)
        or _WINDOWS_ABSOLUTE_PATTERN.search(value)
        or _UNC_ABSOLUTE_PATTERN.search(value)
        or _FILE_URI_PATTERN.search(value)
    ):
        raise ValueError("机器证据字符串不得包含绝对路径")
