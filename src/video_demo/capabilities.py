from __future__ import annotations

import os
import platform
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from video_demo.config import Settings
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.media.process import SafeProcessRunner


class CapabilityStatus(StrEnum):
    AVAILABLE = "AVAILABLE"
    UNAVAILABLE = "UNAVAILABLE"


class CapabilityIssue(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    code: ErrorCode
    message: str


class BinaryCapability(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    path: Path
    version: str


class CapabilityReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: CapabilityStatus
    platform: str
    cloud_asr_provider: str
    cloud_asr_model: str | None
    cloud_asr_base_url: str | None
    cloud_asr_configured: bool
    binaries: tuple[BinaryCapability, ...]
    issues: tuple[CapabilityIssue, ...]


def _read_binary_version(path: Path) -> str:
    result = SafeProcessRunner(max_output_bytes=64 * 1024).run(
        [str(path), "-version"],
        timeout_seconds=10,
    )
    output = result.stdout or result.stderr
    first_line = output.decode("utf-8", errors="replace").splitlines()
    if result.returncode != 0 or not first_line:
        raise VideoDemoError(ErrorCode.VIDEO_BINARY_PROBE_FAILED, "媒体工具版本探测失败")
    return first_line[0]


def _find_binary(settings: Settings, name: str, configured: Path | None) -> Path | None:
    assert settings.runtime_root is not None
    candidate = configured or settings.runtime_root / "tools" / name
    unavailable_code = (
        ErrorCode.VIDEO_FFMPEG_UNAVAILABLE
        if name == "ffmpeg"
        else ErrorCode.VIDEO_FFPROBE_UNAVAILABLE
    )
    try:
        return resolve_workspace_binary(
            candidate,
            workspace_root=settings.workspace_root,
            unavailable_code=unavailable_code,
        )
    except VideoDemoError:
        return None


def resolve_workspace_binary(
    candidate: Path,
    *,
    workspace_root: Path,
    unavailable_code: ErrorCode,
) -> Path:
    """返回工作区内无符号链接且可执行的普通文件。"""

    workspace = workspace_root.expanduser().resolve(strict=False)
    unresolved = candidate.expanduser()
    if not unresolved.is_absolute():
        unresolved = workspace / unresolved
    try:
        relative_path = unresolved.relative_to(workspace)
    except ValueError as error:
        raise VideoDemoError(unavailable_code, "媒体工具必须位于项目工作区内") from error

    current = workspace
    for component in relative_path.parts:
        current /= component
        if current.is_symlink():
            raise VideoDemoError(unavailable_code, "媒体工具路径不能包含符号链接")

    try:
        resolved = unresolved.resolve(strict=True)
    except OSError as error:
        raise VideoDemoError(unavailable_code, "未找到媒体工具可执行文件") from error
    if (
        not resolved.is_relative_to(workspace)
        or not resolved.is_file()
        or not os.access(resolved, os.X_OK)
    ):
        raise VideoDemoError(unavailable_code, "媒体工具不是工作区内可执行普通文件")
    return resolved


def probe_runtime_capabilities(settings: Settings) -> CapabilityReport:
    """执行无副作用能力探测; 缺少依赖时返回明确问题而非自动安装。"""

    binaries: list[BinaryCapability] = []
    issues: list[CapabilityIssue] = []
    cloud_asr = None
    try:
        cloud_asr = settings.require_cloud_asr_configuration()
    except VideoDemoError as error:
        issues.append(CapabilityIssue(code=error.code, message="云端语音识别配置不完整"))
    specifications = (
        ("ffmpeg", settings.ffmpeg_path, ErrorCode.VIDEO_FFMPEG_UNAVAILABLE),
        ("ffprobe", settings.ffprobe_path, ErrorCode.VIDEO_FFPROBE_UNAVAILABLE),
    )
    for name, configured, error_code in specifications:
        path = _find_binary(settings, name, configured)
        if path is None:
            issues.append(CapabilityIssue(code=error_code, message=f"未找到 {name} 可执行文件"))
            continue
        try:
            version = _read_binary_version(path)
        except (OSError, TimeoutError, VideoDemoError):
            issues.append(
                CapabilityIssue(
                    code=ErrorCode.VIDEO_BINARY_PROBE_FAILED,
                    message=f"{name} 版本探测失败",
                ),
            )
            continue
        binaries.append(BinaryCapability(name=name, path=path, version=version))

    system = "macOS" if platform.system() == "Darwin" else platform.system()
    return CapabilityReport(
        status=CapabilityStatus.UNAVAILABLE if issues else CapabilityStatus.AVAILABLE,
        platform=f"{system}-{platform.machine()}",
        cloud_asr_provider="openai_compatible",
        cloud_asr_model=cloud_asr.model if cloud_asr is not None else None,
        cloud_asr_base_url=cloud_asr.base_url if cloud_asr is not None else None,
        cloud_asr_configured=cloud_asr is not None,
        binaries=tuple(binaries),
        issues=tuple(issues),
    )
