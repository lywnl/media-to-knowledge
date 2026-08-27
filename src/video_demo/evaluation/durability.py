"""M1 两段耐久验收：输入先验签，执行后逐样本形成可重验证据。"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import os
import shutil
import sqlite3
import sys
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, cast

from fastapi.testclient import TestClient
from pydantic import Field, TypeAdapter, ValidationError, model_validator

from video_demo.api.app import create_app
from video_demo.api.schemas import PublicEvidence, PublicKeyframeEvidence
from video_demo.application.composition import (
    build_production_model_identity_report,
    build_worker,
)
from video_demo.capabilities import probe_runtime_capabilities
from video_demo.config import Settings
from video_demo.domain.base import FrozenModel, Sha256, StableId
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.evaluation.annotations import AuthorizationFile
from video_demo.evaluation.dataset import _read_json_file, _safe_relative_file, _sha256_media
from video_demo.evaluation.evidence import (
    CommandTrace,
    EvidenceKind,
    EvidenceLevel,
    EvidenceStore,
    MachineEvidenceReport,
    PerformanceDetails,
    PerformanceRawReport,
    PerformanceSampleDetails,
    PerformanceSampleRawReport,
    PreflightDetails,
    PreflightIssue,
    PreflightRawReport,
    ReportRunWriter,
    build_verified_gate_check,
)
from video_demo.evaluation.gate import (
    GateCheck,
    _current_durability_implementation_sha256,
    build_durability_not_run_reason,
)
from video_demo.evaluation.prediction_runner import (
    _evidence_matches_api,
    _mime_for_path,
    _normalize_public_evidence,
    _parse_api_result,
    _read_published_manifest,
    _require_status,
)
from video_demo.evaluation.report import GateStatus
from video_demo.media.probe import FFprobeClient, ProbeLimits, SupportedMime
from video_demo.persistence.repositories import Scope
from video_demo.storage.workspace import validate_path_component

try:
    psutil: Any = importlib.import_module("psutil")
except ModuleNotFoundError:  # pragma: no cover - 由 preflight 测试覆盖
    psutil = None

_CHECK_ID = "m1_durability"
_MIN_DURATION_MS = 1_800_000
MAX_DURABILITY_DURATION_MS = 7_200_000
_MIN_WIDTH = 1920
_MIN_HEIGHT = 1080
_MAX_RTF = 3.0
_SUCCESS_STATUSES = frozenset({"SUCCEEDED", "PARTIAL_SUCCEEDED"})
_PUBLIC_EVIDENCE_ADAPTER: TypeAdapter[tuple[PublicEvidence, ...]] = TypeAdapter(
    tuple[PublicEvidence, ...]
)
_DEFAULT_SCOPE_HEADERS = {
    "X-Tenant-Id": "evaluation",
    "X-Application-Id": "video-demo",
}


def _collect_active_production_environment_issues(
    settings: Settings,
    store: EvidenceStore,
) -> tuple[ErrorCode, ...]:
    """检查耐久活动链依赖；保持 durability 不导入历史 live 组合根。"""

    issues: list[ErrorCode] = []
    try:
        settings.require_text_llm_configuration()
    except VideoDemoError:
        issues.append(ErrorCode.INVALID_CONFIGURATION)
    try:
        settings.require_vlm_configuration()
    except VideoDemoError:
        issues.append(ErrorCode.INVALID_CONFIGURATION)
    if not _module_available("cv2"):
        issues.append(ErrorCode.VISUAL_DEPENDENCY_UNAVAILABLE)
    assert settings.runtime_root is not None
    ffmpeg = settings.ffmpeg_path or settings.runtime_root / "tools" / "ffmpeg"
    ffprobe = settings.ffprobe_path or settings.runtime_root / "tools" / "ffprobe"
    if not ffmpeg.is_file():
        issues.append(ErrorCode.VIDEO_FFMPEG_UNAVAILABLE)
    if not ffprobe.is_file():
        issues.append(ErrorCode.VIDEO_FFPROBE_UNAVAILABLE)
    try:
        if shutil.disk_usage(store.runtime_root).free < settings.min_free_disk_reserve_bytes:
            issues.append(ErrorCode.VIDEO_DISK_SPACE_INSUFFICIENT)
    except OSError:
        issues.append(ErrorCode.VIDEO_DISK_SPACE_INSUFFICIENT)
    model_root = store.runtime_root / "models"
    if not _module_available("silero_vad"):
        issues.append(ErrorCode.SILERO_DEPENDENCY_UNAVAILABLE)
    if not (model_root / "silero/model-id.txt").is_file():
        issues.append(ErrorCode.SILERO_MODEL_UNAVAILABLE)
    try:
        settings.require_cloud_asr_configuration()
    except VideoDemoError:
        issues.append(ErrorCode.INVALID_CONFIGURATION)
    return tuple(dict.fromkeys(issues))


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False
_PREFLIGHT_ORDER = (
    ErrorCode.M1_SAMPLE_COUNT_INVALID,
    ErrorCode.M1_DURATION_TOO_SHORT,
    ErrorCode.M1_DURATION_TOO_LONG,
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
    ErrorCode.VIDEO_DISK_SPACE_INSUFFICIENT,
)

_AUDIT_WRITE_EVENTS: dict[str, tuple[int, ...]] = {
    "os.remove": (0,),
    "os.rename": (0, 1),
    "os.rmdir": (0,),
    "os.mkdir": (0,),
    "os.symlink": (1,),
    "os.link": (1,),
    "os.chmod": (0,),
    "os.truncate": (0,),
}
_ACTIVE_WRITE_AUDITS: set[_WorkspaceWriteAudit] = set()
_AUDIT_LOCK = threading.RLock()
_AUDIT_HOOK_INSTALLED = False


class _WorkspaceWriteAudit:
    """只记录写目标路径；不会打开或遍历工作区外内容。"""

    def __init__(self, workspace_root: Path) -> None:
        self._workspace_root = Path(os.path.abspath(workspace_root))
        self._outside_paths: set[str] = set()

    @property
    def outside_write_count(self) -> int:
        with _AUDIT_LOCK:
            return len(self._outside_paths)

    def activate(self) -> None:
        _install_audit_hook()
        with _AUDIT_LOCK:
            _ACTIVE_WRITE_AUDITS.add(self)

    def deactivate(self) -> None:
        with _AUDIT_LOCK:
            _ACTIVE_WRITE_AUDITS.discard(self)

    def observe(self, event: str, args: tuple[object, ...]) -> None:
        for value in _audit_write_targets(event, args):
            if not isinstance(value, (str, bytes, os.PathLike)):
                continue
            try:
                raw = os.fspath(value)
            except TypeError:
                continue
            if isinstance(raw, bytes):
                raw = os.fsdecode(raw)
            target = Path(os.path.abspath(raw))
            if not target.is_relative_to(self._workspace_root):
                self._outside_paths.add(os.path.normcase(str(target)))


def _install_audit_hook() -> None:
    global _AUDIT_HOOK_INSTALLED
    with _AUDIT_LOCK:
        if _AUDIT_HOOK_INSTALLED:
            return
        sys.addaudithook(_durability_audit_hook)
        _AUDIT_HOOK_INSTALLED = True


def _durability_audit_hook(event: str, args: tuple[object, ...]) -> None:
    with _AUDIT_LOCK:
        trackers = tuple(_ACTIVE_WRITE_AUDITS)
    for tracker in trackers:
        tracker.observe(event, args)


def _audit_write_targets(
    event: str,
    args: tuple[object, ...],
) -> tuple[object, ...]:
    if event == "open" and len(args) >= 3:
        mode = args[1]
        flags = args[2]
        writes_by_mode = isinstance(mode, str) and any(
            marker in mode for marker in ("w", "a", "x", "+")
        )
        writes_by_flags = isinstance(flags, int) and bool(
            flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)
        )
        return (args[0],) if writes_by_mode or writes_by_flags else ()
    indexes = _AUDIT_WRITE_EVENTS.get(event, ())
    return tuple(args[index] for index in indexes if index < len(args))


class DurabilityManifestSample(FrozenModel):
    """一条独立耐久输入；自报属性必须与授权和 ffprobe 事实一致。"""

    sample_id: StableId
    media_relative_path: str = Field(min_length=1, max_length=1024)
    media_sha256: Sha256
    duration_ms: int = Field(gt=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    authorization_id: StableId

    @model_validator(mode="after")
    def validate_media_path(self) -> DurabilityManifestSample:
        path = Path(self.media_relative_path)
        if path.is_absolute() or ".." in path.parts or not path.parts:
            raise ValueError("耐久媒体路径必须是工作区内无穿越相对路径")
        return self


class DurabilityProbe(FrozenModel):
    duration_ms: int = Field(gt=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class DurabilityProbeReport(FrozenModel):
    """将探测事实唯一绑定到当前耐久样本和输入媒体。"""

    schema_version: str = Field(pattern=r"^1\.0\.0$")
    sample_id: StableId
    media_sha256: Sha256
    duration_ms: int = Field(gt=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class DurabilitySample(FrozenModel):
    """一次产品执行与资源采样得到的原始事实。"""

    elapsed_seconds: float = Field(ge=0, allow_inf_nan=False)
    peak_rss_bytes: int = Field(ge=0)
    peak_disk_bytes: int = Field(ge=0)
    oom_detected: bool
    peak_concurrency: int = Field(ge=0)
    outside_workspace_write_count: int = Field(ge=0)
    terminal_status: str = Field(min_length=1, max_length=64)
    failure_code: str | None = Field(default=None, min_length=3, max_length=128)
    production_run_id: StableId | None = None
    job_id: StableId | None = None
    result_manifest_relative_path: str | None = Field(default=None, max_length=1024)
    result_manifest_sha256: Sha256 | None = None


class DurabilitySampleResult(FrozenModel):
    media_sha256: Sha256
    duration_ms: int = Field(gt=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    elapsed_seconds: float = Field(ge=0, allow_inf_nan=False)
    rtf: float = Field(ge=0, allow_inf_nan=False)
    peak_rss_bytes: int = Field(ge=0)
    peak_disk_bytes: int = Field(ge=0)
    oom: bool
    peak_worker_concurrency: int = Field(ge=0)
    outside_workspace_write_count: int = Field(ge=0)
    terminal_status: str = Field(min_length=1, max_length=64)
    failure_code: str | None = Field(default=None, min_length=3, max_length=128)


class DurabilityRunReport(FrozenModel):
    schema_version: str = Field(pattern=r"^1\.0\.0$")
    evaluation_run_id: StableId
    status: GateStatus
    samples: tuple[DurabilitySampleResult, DurabilitySampleResult]
    started_at: datetime
    finished_at: datetime


class DurabilitySampler(Protocol):
    def sample(self, run_root: Path, pid: int) -> tuple[int, int, int]: ...


class ProcessTreeSampler:
    """采集当前进程树 RSS 与本次 run 目录大小，不扫描工作区外内容。"""

    def __init__(self, runtime_root: Path) -> None:
        self._runtime_root = runtime_root

    def sample(self, run_root: Path, pid: int) -> tuple[int, int, int]:
        if psutil is None:
            raise RuntimeError("psutil 不可用")
        process = psutil.Process(pid)
        rss = 0
        for child in (process, *process.children(recursive=True)):
            try:
                rss += child.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        disk = _regular_tree_size(run_root)
        return rss, disk, self._running_worker_count()

    def _running_worker_count(self) -> int:
        database = self._runtime_root / "video-demo.db"
        if not database.is_file() or database.is_symlink():
            return 0
        connection: sqlite3.Connection | None = None
        try:
            uri = f"file:{database.as_posix()}?mode=ro"
            connection = sqlite3.connect(uri, uri=True, timeout=0.1)
            row = connection.execute(
                "SELECT COUNT(*) FROM job WHERE status = ?",
                ("RUNNING",),
            ).fetchone()
            return int(row[0]) if row is not None else 0
        except (OSError, sqlite3.Error, TypeError, ValueError):
            return 0
        finally:
            if connection is not None:
                connection.close()


def _regular_tree_size(root: Path) -> int:
    """只沿普通目录项扫描，不跟随任何 symlink。"""

    try:
        if root.is_symlink() or not root.is_dir():
            return 0
        total = 0
        pending = [root]
        while pending:
            current = pending.pop()
            with os.scandir(current) as entries:
                for entry in entries:
                    if entry.is_symlink():
                        continue
                    if entry.is_dir(follow_symlinks=False):
                        pending.append(Path(entry.path))
                    elif entry.is_file(follow_symlinks=False):
                        total += entry.stat(follow_symlinks=False).st_size
        return total
    except OSError:
        return 0


ExecuteSample = Callable[[DurabilityManifestSample], DurabilitySample]
ProbeMedia = Callable[[Path, DurabilityManifestSample], DurabilityProbe]


class DurabilityRunner:
    """驱动唯一产品 API/Worker 链并发布 M1 性能证据。"""

    def __init__(
        self,
        settings: Settings,
        evidence_store: EvidenceStore,
        *,
        sampler: DurabilitySampler | None = None,
        execute_sample: ExecuteSample | None = None,
        probe_media: ProbeMedia | None = None,
    ) -> None:
        self._settings = settings
        self._store = evidence_store
        self._sampler = sampler or ProcessTreeSampler(evidence_store.runtime_root)
        self._execute_sample = execute_sample
        self._probe_media = probe_media
        self._allows_performance_pass = (
            sampler is None and execute_sample is None and probe_media is None
        )

    def run(self, manifest_path: Path, *, evaluation_run_id: str) -> GateCheck:
        validate_path_component(evaluation_run_id, "evaluation_run_id")
        report_path = self._report_path(evaluation_run_id)
        if report_path.is_file():
            return build_verified_gate_check(
                _CHECK_ID,
                report_path,
                workspace_root=self._settings.workspace_root,
                settings=self._settings,
            )
        started = datetime.now(UTC)
        samples, media_paths, probes, manifest, authorization, issues = self._preflight(
            manifest_path
        )
        writer = self._writer(evaluation_run_id)
        try:
            if issues:
                return self._write_not_run(writer, evaluation_run_id, issues)
            facts = tuple(self._run_sample(sample, evaluation_run_id) for sample in samples)
            if self._facts_all_pass(samples, facts) and not self._allows_performance_pass:
                raise VideoDemoError(
                    ErrorCode.SYSTEM_FAILURE,
                    "受控耐久执行端口不得发布正式 PASS",
                )
            performance_samples = cast(
                tuple[PerformanceSampleDetails, PerformanceSampleDetails],
                tuple(
                    self._to_performance_sample(sample, fact)
                    for sample, fact in zip(samples, facts, strict=True)
                ),
            )
            return self._write_executed(
                writer,
                evaluation_run_id,
                performance_samples,
                media_paths,
                probes,
                manifest,
                authorization,
                started,
            )
        finally:
            writer.close()

    def _preflight(
        self, manifest_path: Path
    ) -> tuple[
        tuple[DurabilityManifestSample, ...],
        tuple[Path, ...],
        tuple[DurabilityProbe, ...],
        Path,
        Path,
        tuple[ErrorCode, ...],
    ]:
        root = self._store.runtime_root
        fallback = root / "eval/durability/dataset.jsonl"
        authorization_path = root / "eval/durability/authorization.json"
        try:
            manifest = self._safe_contract(manifest_path)
            values = tuple(
                DurabilityManifestSample.model_validate(json.loads(line))
                for line in _read_json_file(manifest).decode("utf-8").splitlines()
                if line.strip()
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValidationError, ValueError):
            return (), (), (), fallback, authorization_path, (ErrorCode.M1_MEDIA_INVALID,)
        issues: list[ErrorCode] = []
        if len(values) != 2:
            issues.append(ErrorCode.M1_SAMPLE_COUNT_INVALID)
        if len({sample.media_sha256 for sample in values}) != len(values):
            issues.append(ErrorCode.M1_MEDIA_INVALID)
        if any(sample.duration_ms < _MIN_DURATION_MS for sample in values):
            issues.append(ErrorCode.M1_DURATION_TOO_SHORT)
        if any(sample.duration_ms > MAX_DURABILITY_DURATION_MS for sample in values):
            issues.append(ErrorCode.M1_DURATION_TOO_LONG)
        if any(sample.width < _MIN_WIDTH or sample.height < _MIN_HEIGHT for sample in values):
            issues.append(ErrorCode.M1_RESOLUTION_TOO_SMALL)
        media_paths = self._media_paths(manifest, values, issues)
        self._verify_authorization(authorization_path, values, issues)
        # 基础契约 (样本数量、授权、媒体摘要) 已经失败时，不再探测环境或
        # 启动任何运行时依赖；否则 NOT_RUN 原因会被无关机器缺项污染。
        if issues:
            return (
                values,
                media_paths,
                (),
                manifest,
                authorization_path,
                tuple(dict.fromkeys(issues)),
            )
        self._append_runtime_issues(issues)
        probes = () if issues else self._verify_probes(media_paths, values, issues)
        return (
            values,
            media_paths,
            probes,
            manifest,
            authorization_path,
            tuple(dict.fromkeys(issues)),
        )

    def _safe_contract(self, path: Path) -> Path:
        root = self._store.runtime_root.resolve(strict=True)
        unresolved = path if path.is_absolute() else root / path
        relative = unresolved.absolute().relative_to(root)
        current = root
        for part in relative.parts:
            current /= part
            if current.is_symlink():
                raise ValueError("耐久契约路径不能包含符号链接")
        resolved = unresolved.resolve(strict=True)
        if not resolved.is_file():
            raise ValueError("耐久契约不是普通文件")
        return resolved

    def _media_paths(
        self,
        manifest: Path,
        samples: tuple[DurabilityManifestSample, ...],
        issues: list[ErrorCode],
    ) -> tuple[Path, ...]:
        paths: list[Path] = []
        for sample in samples:
            try:
                media = _safe_relative_file(
                    manifest.parent,
                    sample.media_relative_path,
                    self._store.runtime_root,
                )
                if _sha256_media(media, self._settings.max_video_bytes) != sample.media_sha256:
                    raise ValueError("媒体摘要不匹配")
                paths.append(media)
            except (OSError, ValueError):
                issues.append(ErrorCode.M1_MEDIA_INVALID)
        return tuple(paths)

    def _verify_authorization(
        self,
        path: Path,
        samples: tuple[DurabilityManifestSample, ...],
        issues: list[ErrorCode],
    ) -> None:
        try:
            authorization = AuthorizationFile.model_validate_json(
                _read_json_file(self._safe_contract(path))
            )
            records = {record.authorization_id: record for record in authorization.records}
            if any(
                (record := records.get(sample.authorization_id)) is None
                or sample.media_sha256 not in record.media_sha256
                for sample in samples
            ):
                raise ValueError("授权未覆盖耐久媒体")
        except (OSError, ValidationError, ValueError):
            issues.append(ErrorCode.M1_AUTHORIZATION_UNAVAILABLE)

    def _verify_probes(
        self,
        paths: tuple[Path, ...],
        samples: tuple[DurabilityManifestSample, ...],
        issues: list[ErrorCode],
    ) -> tuple[DurabilityProbe, ...]:
        probes: list[DurabilityProbe] = []
        if len(paths) != len(samples):
            return ()
        for path, sample in zip(paths, samples, strict=True):
            try:
                probe = (
                    self._probe_media(path, sample)
                    if self._probe_media is not None
                    else self._probe_with_ffprobe(path, sample)
                )
                if (
                    probe.duration_ms != sample.duration_ms
                    or probe.width != sample.width
                    or probe.height != sample.height
                ):
                    raise ValueError("ffprobe 与耐久 Manifest 不一致")
                probes.append(probe)
            except (OSError, ValueError, VideoDemoError):
                issues.append(ErrorCode.M1_PROBE_MISMATCH)
        return tuple(probes)

    def _probe_with_ffprobe(self, path: Path, sample: DurabilityManifestSample) -> DurabilityProbe:
        root = self._settings.runtime_root
        assert root is not None
        executable = self._settings.ffprobe_path or root / "tools/ffprobe"
        client = FFprobeClient.from_path(executable, workspace_root=self._settings.workspace_root)
        result = client.probe(
            path,
            object_ref=f"durability-{sample.sample_id}",
            source_sha256=sample.media_sha256,
            source_size_bytes=path.stat().st_size,
            source_mime=cast(SupportedMime, _mime_for_path(path)),
            limits=ProbeLimits(
                max_duration_ms=max(sample.duration_ms, _MIN_DURATION_MS),
                max_width=max(sample.width, _MIN_WIDTH),
                max_height=max(sample.height, _MIN_HEIGHT),
            ),
        )
        stream = result.manifest.video_stream
        width = stream.height if stream.rotation_degrees in (90, 270) else stream.width
        height = stream.width if stream.rotation_degrees in (90, 270) else stream.height
        return DurabilityProbe(
            duration_ms=result.manifest.duration_ms,
            width=width,
            height=height,
        )

    def _append_runtime_issues(self, issues: list[ErrorCode]) -> None:
        if self._settings.worker_concurrency != 1:
            issues.append(ErrorCode.INVALID_CONFIGURATION)
        if psutil is None:
            issues.append(ErrorCode.M1_PSUTIL_UNAVAILABLE)
        try:
            free_bytes = shutil.disk_usage(self._store.runtime_root).free
        except OSError:
            issues.append(ErrorCode.VIDEO_DISK_SPACE_INSUFFICIENT)
        else:
            if free_bytes < self._settings.min_free_disk_reserve_bytes:
                issues.append(ErrorCode.VIDEO_DISK_SPACE_INSUFFICIENT)
        if self._probe_media is None:
            capabilities = probe_runtime_capabilities(self._settings)
            issues.extend(issue.code for issue in capabilities.issues)
            for module in ("cv2", "scenedetect"):
                try:
                    available = importlib.util.find_spec(module) is not None
                except (ImportError, ModuleNotFoundError, ValueError):
                    available = False
                if not available:
                    issues.append(ErrorCode.VISUAL_DEPENDENCY_UNAVAILABLE)
            issues.extend(
                _collect_active_production_environment_issues(self._settings, self._store)
            )

    def _run_sample(
        self,
        sample: DurabilityManifestSample,
        evaluation_run_id: str,
    ) -> DurabilitySample:
        active_run_root = [self._store.runtime_root / "eval/reports" / evaluation_run_id]

        def set_run_root(run_id: str) -> None:
            active_run_root[0] = self._production_run_root(run_id)

        return self._sample_execution(
            lambda: active_run_root[0],
            lambda: self._execute_one_sample(
                sample,
                evaluation_run_id,
                set_run_root,
            ),
        )

    def _execute_one_sample(
        self,
        sample: DurabilityManifestSample,
        evaluation_run_id: str,
        set_run_root: Callable[[str], None],
    ) -> DurabilitySample:
        try:
            return (
                self._execute_sample(sample)
                if self._execute_sample is not None
                else self._execute_production_sample(
                    sample,
                    evaluation_run_id,
                    set_run_root,
                )
            )
        except VideoDemoError as error:
            return DurabilitySample(
                elapsed_seconds=0.0,
                peak_rss_bytes=0,
                peak_disk_bytes=0,
                oom_detected=error.code == ErrorCode.OUT_OF_MEMORY,
                peak_concurrency=1,
                outside_workspace_write_count=0,
                terminal_status="FAILED",
                failure_code=error.code.value,
            )

    def _sample_execution(
        self,
        run_root: Callable[[], Path],
        execute: Callable[[], DurabilitySample],
    ) -> DurabilitySample:
        """每 50ms 采样进程树与指定 run，线程异常在主线程重新抛出。"""

        observations = [self._sampler.sample(run_root(), os.getpid())]
        errors: list[BaseException] = []
        stop = threading.Event()

        def collect() -> None:
            while not stop.wait(0.05):
                try:
                    observations.append(self._sampler.sample(run_root(), os.getpid()))
                except BaseException as error:
                    errors.append(error)
                    stop.set()
                    return

        thread = threading.Thread(
            target=collect,
            name="durability-resource-sampler",
            daemon=True,
        )
        write_audit = _WorkspaceWriteAudit(self._settings.workspace_root)
        thread.start()
        write_audit.activate()
        try:
            fact = execute()
            observations.append(self._sampler.sample(run_root(), os.getpid()))
        finally:
            write_audit.deactivate()
            stop.set()
            thread.join()
        if errors:
            raise errors[0]
        return fact.model_copy(
            update={
                "peak_rss_bytes": max(
                    fact.peak_rss_bytes, *(observation[0] for observation in observations)
                ),
                "peak_disk_bytes": max(
                    fact.peak_disk_bytes, *(observation[1] for observation in observations)
                ),
                "peak_concurrency": max(
                    fact.peak_concurrency,
                    *(observation[2] for observation in observations),
                ),
                "outside_workspace_write_count": max(
                    fact.outside_workspace_write_count,
                    write_audit.outside_write_count,
                ),
            }
        )

    def _execute_production_sample(
        self,
        sample: DurabilityManifestSample,
        evaluation_run_id: str,
        set_run_root: Callable[[str], None],
    ) -> DurabilitySample:
        media = _safe_relative_file(
            self._store.runtime_root / "eval/durability",
            sample.media_relative_path,
            self._store.runtime_root,
        )
        started = time.monotonic()
        app = create_app(self._settings)
        with TestClient(app) as client:
            with media.open("rb") as stream:
                upload = client.post(
                    "/api/kb/knowledge-bases/evaluation/video-objects",
                    files={"file": (media.name, stream, _mime_for_path(media))},
                    headers=_DEFAULT_SCOPE_HEADERS,
                )
            _require_status(upload, 201)
            created = client.post(
                "/api/kb/knowledge-bases/evaluation/video-understanding-runs",
                json={
                    "object_ref": str(upload.json()["object_ref"]),
                    "idempotency_key": self._idempotency_key(evaluation_run_id, sample.sample_id),
                },
                headers=_DEFAULT_SCOPE_HEADERS,
            )
            _require_status(created, 202)
            run_id = str(created.json()["run_id"])
            job_id = str(created.json()["job_id"])
            set_run_root(run_id)
            worker = build_worker(self._settings, worker_id=f"durability-{evaluation_run_id}")
            try:
                if not worker.run_once():
                    raise VideoDemoError(ErrorCode.JOB_NOT_FOUND, "耐久 Worker 未领取任务")
            finally:
                worker.close()
            run_response = client.get(
                f"/api/kb/knowledge-bases/evaluation/video-understanding-runs/{run_id}",
                headers=_DEFAULT_SCOPE_HEADERS,
            )
            _require_status(run_response, 200)
            payload = run_response.json()
            terminal = str(payload.get("status", "FAILED"))
            failure_code = payload.get("error_code")
            manifest_relative: str | None = None
            manifest_sha256: str | None = None
            manifest_bytes = _verify_production_queries(
                client,
                run_id=run_id,
                job_id=job_id,
                terminal=terminal,
            )
            if manifest_bytes is not None:
                manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
                manifest_relative = self._result_manifest_path(run_id).as_posix()
        return DurabilitySample(
            elapsed_seconds=time.monotonic() - started,
            peak_rss_bytes=0,
            peak_disk_bytes=0,
            oom_detected=failure_code == ErrorCode.OUT_OF_MEMORY.value,
            peak_concurrency=0,
            outside_workspace_write_count=0,
            terminal_status=terminal,
            failure_code=str(failure_code) if failure_code else None,
            production_run_id=run_id,
            job_id=job_id,
            result_manifest_relative_path=manifest_relative,
            result_manifest_sha256=manifest_sha256,
        )

    @staticmethod
    def _to_performance_sample(
        sample: DurabilityManifestSample,
        fact: DurabilitySample,
    ) -> PerformanceSampleDetails:
        succeeded = fact.terminal_status in _SUCCESS_STATUSES and fact.failure_code is None
        probe = DurabilityProbe(
            duration_ms=sample.duration_ms,
            width=sample.width,
            height=sample.height,
        )
        probe_sha256 = hashlib.sha256(probe.model_dump_json().encode("utf-8")).hexdigest()
        return PerformanceSampleDetails(
            sample_id=sample.sample_id,
            media_relative_path=(Path("eval/durability") / sample.media_relative_path).as_posix(),
            sample_sha256=sample.media_sha256,
            authorization_id=sample.authorization_id,
            duration_ms=sample.duration_ms,
            width=sample.width,
            height=sample.height,
            elapsed_seconds=fact.elapsed_seconds,
            rtf=fact.elapsed_seconds / (sample.duration_ms / 1000),
            oom_detected=fact.oom_detected,
            peak_concurrency=fact.peak_concurrency,
            outside_workspace_write_count=fact.outside_workspace_write_count,
            peak_rss_bytes=fact.peak_rss_bytes,
            peak_disk_bytes=fact.peak_disk_bytes,
            succeeded=succeeded,
            terminal_status=fact.terminal_status,
            failure_code=fact.failure_code,
            production_run_id=fact.production_run_id,
            job_id=fact.job_id,
            result_manifest_relative_path=fact.result_manifest_relative_path,
            result_manifest_sha256=fact.result_manifest_sha256,
            probe_report_sha256=probe_sha256,
        )

    def _write_not_run(
        self,
        writer: ReportRunWriter,
        evaluation_run_id: str,
        issues: tuple[ErrorCode, ...],
    ) -> GateCheck:
        raw = PreflightRawReport(
            schema_version="1.0.0",
            check_id=_CHECK_ID,
            reason_code="M1_DURABILITY_INPUT_UNAVAILABLE",
            execution_started=False,
            issues=tuple(
                PreflightIssue(code=code)
                for code in sorted(set(issues), key=_PREFLIGHT_ORDER.index)
            ),
            implementation_sha256=_current_durability_implementation_sha256(
                self._settings.workspace_root
            ),
            evaluation_run_id=evaluation_run_id,
        )
        raw_artifact = writer.write_artifact(
            "preflight.json", "AUDIT_REPORT", raw.model_dump_json().encode("utf-8")
        )
        stdout = writer.write_artifact("trace.stdout.txt", "COMMAND_STDOUT", b"")
        stderr = writer.write_artifact("trace.stderr.txt", "COMMAND_STDERR", b"")
        report = MachineEvidenceReport(
            schema_version="1.0.0",
            check_id=_CHECK_ID,
            status=GateStatus.NOT_RUN,
            kind=EvidenceKind.PERFORMANCE_REPORT,
            level=EvidenceLevel.PERFORMANCE,
            covered_items=(_CHECK_ID,),
            summary="M1 耐久前置条件不足",
            producer="DurabilityRunner",
            started_at=datetime.now(UTC),
            finished_at=datetime.now(UTC),
            not_run_reason=build_durability_not_run_reason(issues),
            artifacts=(raw_artifact, stdout, stderr),
            details=PreflightDetails(
                type="PREFLIGHT",
                trace=CommandTrace(
                    command=("python", "-m", "video_demo.evaluation.durability"),
                    exit_code=0,
                    stdout_sha256=stdout.sha256,
                    stderr_sha256=stderr.sha256,
                ),
                preflight_report_sha256=raw_artifact.sha256,
            ),
        )
        return writer.write_json(report, filename="durability.json")

    def _write_executed(
        self,
        writer: ReportRunWriter,
        evaluation_run_id: str,
        samples: tuple[PerformanceSampleDetails, PerformanceSampleDetails],
        media_paths: tuple[Path, ...],
        probes: tuple[DurabilityProbe, ...],
        manifest_path: Path,
        authorization_path: Path,
        started: datetime,
    ) -> GateCheck:
        manifest_artifact = self._store.bind_artifact(
            manifest_path.relative_to(self._store.runtime_root), "DATASET_MANIFEST"
        )
        authorization_artifact = self._store.bind_artifact(
            authorization_path.relative_to(self._store.runtime_root), "AUTHORIZATION_RECORD"
        )
        input_artifacts = tuple(
            self._store.bind_artifact(
                path.relative_to(self._store.runtime_root),
                "INPUT_MEDIA",
                max_bytes=self._settings.max_video_bytes,
            )
            for path in media_paths
        )
        probe_artifacts = tuple(
            writer.write_artifact(
                f"probe-{index}.json",
                "AUDIT_REPORT",
                DurabilityProbeReport(
                    schema_version="1.0.0",
                    sample_id=sample.sample_id,
                    media_sha256=sample.sample_sha256,
                    duration_ms=probe.duration_ms,
                    width=probe.width,
                    height=probe.height,
                )
                .model_dump_json()
                .encode("utf-8"),
            )
            for index, (sample, probe) in enumerate(zip(samples, probes, strict=True))
        )
        samples = cast(
            tuple[PerformanceSampleDetails, PerformanceSampleDetails],
            tuple(
                sample.model_copy(update={"probe_report_sha256": probe.sha256})
                for sample, probe in zip(samples, probe_artifacts, strict=True)
            ),
        )
        sample_raw_artifacts = tuple(
            writer.write_artifact(
                f"sample-bound-{index}.json",
                "PERFORMANCE_REPORT",
                PerformanceSampleRawReport(
                    schema_version="1.0.0",
                    evaluation_run_id=evaluation_run_id,
                    sample=sample,
                )
                .model_dump_json()
                .encode("utf-8"),
            )
            for index, sample in enumerate(samples)
        )
        result_artifacts = tuple(
            self._store.bind_artifact(
                Path(sample.result_manifest_relative_path).relative_to(".codex/video-rag-demo"),
                "PRODUCTION_RESULT",
            )
            for sample in samples
            if sample.result_manifest_relative_path is not None
        )
        implementation = _current_durability_implementation_sha256(self._settings.workspace_root)
        settings_fingerprint = build_production_model_identity_report(
            self._settings
        ).settings_fingerprint
        sample_report_sha256s = cast(
            tuple[str, str],
            tuple(artifact.sha256 for artifact in sample_raw_artifacts),
        )
        raw = PerformanceRawReport(
            schema_version="1.0.0",
            evaluation_run_id=evaluation_run_id,
            manifest_sha256=manifest_artifact.sha256,
            authorization_sha256=authorization_artifact.sha256,
            implementation_sha256=implementation,
            settings_fingerprint=settings_fingerprint,
            worker_concurrency=1,
            sample_report_sha256s=sample_report_sha256s,
            samples=samples,
        )
        raw_artifact = writer.write_artifact(
            "raw.json", "PERFORMANCE_REPORT", raw.model_dump_json().encode("utf-8")
        )
        stdout = writer.write_artifact(
            "trace.stdout.txt", "COMMAND_STDOUT", b"durability completed\n"
        )
        stderr = writer.write_artifact("trace.stderr.txt", "COMMAND_STDERR", b"")
        passed = self._all_pass(samples)
        report = MachineEvidenceReport(
            schema_version="1.0.0",
            check_id=_CHECK_ID,
            status=GateStatus.PASS if passed else GateStatus.FAIL,
            kind=EvidenceKind.PERFORMANCE_REPORT,
            level=EvidenceLevel.PERFORMANCE,
            covered_items=(_CHECK_ID,),
            summary="M1 耐久执行完成",
            producer="DurabilityRunner",
            started_at=started,
            finished_at=datetime.now(UTC),
            artifacts=(
                *input_artifacts,
                manifest_artifact,
                authorization_artifact,
                *sample_raw_artifacts,
                *probe_artifacts,
                *result_artifacts,
                raw_artifact,
                stdout,
                stderr,
            ),
            details=PerformanceDetails(
                type="PERFORMANCE",
                trace=CommandTrace(
                    command=("python", "-m", "video_demo.evaluation.durability"),
                    exit_code=0 if passed else 1,
                    stdout_sha256=stdout.sha256,
                    stderr_sha256=stderr.sha256,
                ),
                performance_report_sha256=raw_artifact.sha256,
                evaluation_run_id=evaluation_run_id,
                manifest_sha256=manifest_artifact.sha256,
                authorization_sha256=authorization_artifact.sha256,
                implementation_sha256=implementation,
                settings_fingerprint=settings_fingerprint,
                sample_report_sha256s=sample_report_sha256s,
                samples=samples,
            ),
        )
        return writer.write_json(
            report,
            filename="durability.json",
            settings=self._settings,
        )

    @staticmethod
    def _all_pass(samples: tuple[PerformanceSampleDetails, ...]) -> bool:
        return all(
            sample.rtf <= _MAX_RTF
            and not sample.oom_detected
            and sample.peak_concurrency == 1
            and sample.outside_workspace_write_count == 0
            and sample.succeeded
            for sample in samples
        )

    @staticmethod
    def _facts_all_pass(
        samples: tuple[DurabilityManifestSample, ...],
        facts: tuple[DurabilitySample, ...],
    ) -> bool:
        return all(
            fact.elapsed_seconds / (sample.duration_ms / 1000) <= _MAX_RTF
            and not fact.oom_detected
            and fact.peak_concurrency == 1
            and fact.outside_workspace_write_count == 0
            and fact.terminal_status in _SUCCESS_STATUSES
            and fact.failure_code is None
            for sample, fact in zip(samples, facts, strict=True)
        )

    def _writer(self, evaluation_run_id: str) -> ReportRunWriter:
        try:
            return self._store.open_exclusive_report_run(evaluation_run_id)
        except ValueError:
            raise VideoDemoError(
                ErrorCode.IDEMPOTENCY_CONFLICT,
                "运行目录已有不完整证据",
            ) from None

    def _report_path(self, evaluation_run_id: str) -> Path:
        return self._store.runtime_root / "eval/reports" / evaluation_run_id / "durability.json"

    @staticmethod
    def _idempotency_key(evaluation_run_id: str, sample_id: str) -> str:
        digest = hashlib.sha256(f"{evaluation_run_id}:{sample_id}".encode()).hexdigest()
        return f"durability-{digest[:40]}"

    def _result_manifest_path(self, run_id: str) -> Path:
        run_root = self._production_run_root(run_id) / "result"
        manifests = tuple(run_root.glob("bundle-*.json"))
        if len(manifests) != 1:
            raise VideoDemoError(ErrorCode.EVALUATION_ARTIFACT_INVALID, "生产结果 Manifest 不唯一")
        return manifests[0].relative_to(self._settings.workspace_root)

    def _production_run_root(self, run_id: str) -> Path:
        scope = Scope("evaluation", "video-demo", "evaluation")
        encoded = "\x00".join((scope.tenant_id, scope.application_id, scope.knowledge_base_id))
        scope_key = hashlib.sha256(encoded.encode()).hexdigest()[:24]
        return self._store.runtime_root / "runs" / scope_key / run_id


def _verify_production_queries(
    client: Any,
    *,
    run_id: str,
    job_id: str,
    terminal: str,
) -> bytes | None:
    """重用 17E-6 查询契约，闭合 job、结果、分页证据与关键帧内容。"""

    job_response = client.get(
        f"/api/kb/jobs/{job_id}",
        headers=_DEFAULT_SCOPE_HEADERS,
        params={"knowledge_base_id": "evaluation"},
    )
    _require_status(job_response, 200)
    job = job_response.json()
    expected_job_status = "SUCCEEDED" if terminal in _SUCCESS_STATUSES else terminal
    if (
        not isinstance(job, dict)
        or job.get("job_id") != job_id
        or job.get("resource_id") != run_id
        or job.get("status") != expected_job_status
    ):
        raise ValueError("耐久 job 查询与 run 终态不一致")
    if terminal not in _SUCCESS_STATUSES:
        return None

    result_response = client.get(
        f"/api/kb/knowledge-bases/evaluation/video-understanding-runs/{run_id}/result",
        headers=_DEFAULT_SCOPE_HEADERS,
    )
    _require_status(result_response, 200)
    result = _parse_api_result(result_response.json())
    raw_evidence: list[dict[str, object]] = []
    cursor: str | None = None
    seen: set[str] = set()
    while True:
        params: dict[str, str | int] = {"limit": 100}
        if cursor is not None:
            if cursor in seen:
                raise ValueError("耐久证据分页游标重复")
            seen.add(cursor)
            params["cursor"] = cursor
        response = client.get(
            f"/api/kb/knowledge-bases/evaluation/video-understanding-runs/{run_id}/evidence",
            headers=_DEFAULT_SCOPE_HEADERS,
            params=params,
        )
        _require_status(response, 200)
        page = response.json()
        items = page.get("items") if isinstance(page, dict) else None
        if not isinstance(items, list) or any(not isinstance(item, dict) for item in items):
            raise ValueError("耐久证据分页响应非法")
        raw_evidence.extend(cast(list[dict[str, object]], items))
        cursor = page.get("next_cursor")
        if cursor is None:
            break
        if not isinstance(cursor, str):
            raise ValueError("耐久证据分页游标非法")
    normalized = _normalize_public_evidence(raw_evidence)
    public_evidence = _PUBLIC_EVIDENCE_ADAPTER.validate_python(normalized)
    artifact, document_bytes = _read_published_manifest(
        client,
        run_id=run_id,
        scope=Scope("evaluation", "video-demo", "evaluation"),
    )
    if (
        artifact.result != result
        or artifact.status != terminal
        or not _evidence_matches_api(artifact.evidence, normalized)
    ):
        raise ValueError("耐久查询结果与生产 Manifest 不一致")
    for public_item in public_evidence:
        if not isinstance(public_item, PublicKeyframeEvidence):
            continue
        content = client.get(
            f"/api/kb/knowledge-bases/evaluation/video-understanding-runs/{run_id}"
            f"/keyframes/{public_item.keyframe_id}/content",
            headers=_DEFAULT_SCOPE_HEADERS,
        )
        _require_status(content, 200)
        mime = content.headers.get("content-type", "").split(";", 1)[0]
        if (
            mime != public_item.mime_type
            or hashlib.sha256(content.content).hexdigest() != public_item.sha256
        ):
            raise ValueError("耐久关键帧内容与证据不一致")
    return document_bytes
