from __future__ import annotations

import os
import selectors
import signal
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, cast

from video_demo.errors import ErrorCode, VideoDemoError


@dataclass(frozen=True, slots=True)
class ProcessResult:
    returncode: int
    stdout: bytes
    stderr: bytes


@dataclass(frozen=True, slots=True)
class ProcessErrorCodes:
    """子进程执行器的错误码策略；默认值保持既有视频兼容契约。"""

    invalid: ErrorCode = ErrorCode.VIDEO_PROCESS_FAILED
    cancelled: ErrorCode = ErrorCode.VIDEO_PROCESS_CANCELLED
    timeout: ErrorCode = ErrorCode.VIDEO_PROCESS_TIMEOUT
    output_too_large: ErrorCode = ErrorCode.VIDEO_PROCESS_OUTPUT_TOO_LARGE


class SafeProcessRunner:
    """不启用 shell 的受限子进程执行器。"""

    def __init__(
        self,
        *,
        max_output_bytes: int = 16 * 1024 * 1024,
        is_cancel_requested: Callable[[], bool] = lambda: False,
        workspace_root: Path | None = None,
        error_codes: ProcessErrorCodes | None = None,
    ) -> None:
        if max_output_bytes < 1:
            raise ValueError("max_output_bytes 必须大于 0")
        self._max_output_bytes = max_output_bytes
        self._is_cancel_requested = is_cancel_requested
        self._workspace_root = (
            workspace_root.resolve(strict=True) if workspace_root is not None else None
        )
        self._error_codes = error_codes or ProcessErrorCodes()

    def run(
        self,
        args: Sequence[str],
        *,
        timeout_seconds: int,
        pass_fds: tuple[int, ...] = (),
        output_paths: tuple[Path, ...] = (),
        env: Mapping[str, str] | None = None,
    ) -> ProcessResult:
        if (
            not args
            or timeout_seconds <= 0
            or any(not isinstance(argument, str) or not argument for argument in args)
            or not isinstance(pass_fds, tuple)
            or any(type(descriptor) is not int or descriptor < 0 for descriptor in pass_fds)
            or len(pass_fds) != len(set(pass_fds))
            or not isinstance(output_paths, tuple)
            or any(not isinstance(path, Path) for path in output_paths)
            or (
                env is not None
                and any(
                    not isinstance(key, str)
                    or not key
                    or "=" in key
                    or "\x00" in key
                    or not isinstance(value, str)
                    or "\x00" in value
                    for key, value in env.items()
                )
            )
        ):
            raise VideoDemoError(self._error_codes.invalid, "子进程参数必须是非空字符串数组")
        self._verify_output_paths(output_paths)
        if pass_fds:
            if os.name != "posix":
                raise VideoDemoError(
                    self._error_codes.invalid,
                    "当前平台不支持安全继承文件描述符",
                )
            try:
                for descriptor in pass_fds:
                    os.fstat(descriptor)
            except OSError:
                raise VideoDemoError(
                    self._error_codes.invalid,
                    "继承文件描述符非法",
                ) from None
        if self._is_cancel_requested():
            raise VideoDemoError(self._error_codes.cancelled, "媒体处理已取消")
        try:
            if pass_fds:
                process = subprocess.Popen(
                    list(args),
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    shell=False,
                    bufsize=0,
                    start_new_session=True,
                    pass_fds=pass_fds,
                    env=dict(env) if env is not None else None,
                )
            else:
                process = subprocess.Popen(
                    list(args),
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    shell=False,
                    bufsize=0,
                    start_new_session=os.name == "posix",
                    env=dict(env) if env is not None else None,
                )
        except OSError as error:
            raise VideoDemoError(self._error_codes.invalid, "媒体子进程无法启动") from error
        try:
            stdout, stderr = self._collect_output(process, timeout_seconds)
            return ProcessResult(process.wait(), stdout, stderr)
        except BaseException:
            _stop_process(process)
            raise
        finally:
            if process.stdout is not None:
                process.stdout.close()
            if process.stderr is not None:
                process.stderr.close()

    def _verify_output_paths(self, output_paths: tuple[Path, ...]) -> None:
        if not output_paths:
            return
        if self._workspace_root is None:
            raise VideoDemoError(
                ErrorCode.WORKSPACE_PATH_ESCAPE,
                "子进程输出声明缺少工作区边界",
            )
        for output in output_paths:
            target = Path(os.path.abspath(output))
            if not target.is_relative_to(self._workspace_root):
                raise VideoDemoError(
                    ErrorCode.WORKSPACE_PATH_ESCAPE,
                    "子进程输出必须位于工作区内",
                )
            relative = target.relative_to(self._workspace_root)
            current = self._workspace_root
            for component in relative.parts:
                current /= component
                if current.is_symlink():
                    raise VideoDemoError(
                        ErrorCode.WORKSPACE_PATH_ESCAPE,
                        "子进程输出路径不能包含符号链接",
                    )

    def _collect_output(
        self,
        process: subprocess.Popen[bytes],
        timeout_seconds: int,
    ) -> tuple[bytes, bytes]:
        assert process.stdout is not None
        assert process.stderr is not None
        selector = selectors.DefaultSelector()
        outputs = {"stdout": bytearray(), "stderr": bytearray()}
        for name, stream in (("stdout", process.stdout), ("stderr", process.stderr)):
            os.set_blocking(stream.fileno(), False)
            selector.register(stream, selectors.EVENT_READ, data=name)
        deadline = time.monotonic() + timeout_seconds
        try:
            while process.poll() is None or selector.get_map():
                self._check_process_state(deadline)
                for key, _mask in selector.select(timeout=0.05):
                    stream = cast(BinaryIO, key.fileobj)
                    chunk = os.read(stream.fileno(), 64 * 1024)
                    if not chunk:
                        selector.unregister(stream)
                        continue
                    outputs[cast(str, key.data)].extend(chunk)
                    if sum(len(value) for value in outputs.values()) > self._max_output_bytes:
                        raise VideoDemoError(
                            self._error_codes.output_too_large,
                            "媒体子进程输出超过安全上限",
                        )
        finally:
            selector.close()
        return bytes(outputs["stdout"]), bytes(outputs["stderr"])

    def _check_process_state(self, deadline: float) -> None:
        if self._is_cancel_requested():
            raise VideoDemoError(self._error_codes.cancelled, "媒体处理已取消")
        if time.monotonic() >= deadline:
            raise VideoDemoError(self._error_codes.timeout, "媒体子进程执行超时")


def _stop_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        _kill_process_group(process)
        return
    _terminate_process_group(process)
    with suppress(subprocess.TimeoutExpired):
        process.wait(timeout=1)
    _kill_process_group(process)
    with suppress(OSError, subprocess.SubprocessError):
        process.wait()


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    if os.name == "posix":
        _send_group_signal(process.pid, signal.SIGTERM)
        return
    with suppress(OSError):
        process.terminate()


def _kill_process_group(process: subprocess.Popen[bytes]) -> None:
    if os.name == "posix":
        _send_group_signal(process.pid, signal.SIGKILL)
        return
    with suppress(OSError):
        process.kill()


def _send_group_signal(process_group_id: int, signal_number: int) -> None:
    with suppress(OSError):
        os.killpg(process_group_id, signal_number)
