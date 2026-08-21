from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.media.process import SafeProcessRunner


@pytest.mark.skipif(os.name != "posix", reason="fd 继承契约仅适用于 POSIX")
def test_safe_process_runner_passes_preopened_file_descriptor(tmp_path: Path) -> None:
    target = tmp_path / "inherited-output.bin"
    descriptor = os.open(target, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        result = SafeProcessRunner(max_output_bytes=1_024).run(
            [
                sys.executable,
                "-c",
                "import os,sys; os.write(int(sys.argv[1]), b'fd-bound')",
                str(descriptor),
            ],
            timeout_seconds=5,
            pass_fds=(descriptor,),
        )
        os.fsync(descriptor)
    finally:
        os.close(descriptor)

    assert result.returncode == 0
    assert result.stdout == b""
    assert result.stderr == b""
    assert target.read_bytes() == b"fd-bound"


def test_safe_process_runner_rejects_invalid_pass_fd() -> None:
    with pytest.raises(VideoDemoError) as raised:
        SafeProcessRunner().run(
            [sys.executable, "-c", "pass"],
            timeout_seconds=5,
            pass_fds=(-1,),
        )

    assert raised.value.code == ErrorCode.VIDEO_PROCESS_FAILED


def test_safe_process_runner_fails_closed_when_pass_fds_are_unsupported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import video_demo.media.process as process_module

    target = tmp_path / "unsupported-fd.bin"
    descriptor = os.open(target, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
    monkeypatch.setattr(process_module.os, "name", "nt")
    try:
        with pytest.raises(VideoDemoError) as raised:
            SafeProcessRunner().run(
                [sys.executable, "-c", "pass"],
                timeout_seconds=5,
                pass_fds=(descriptor,),
            )
    finally:
        os.close(descriptor)

    assert raised.value.code == ErrorCode.VIDEO_PROCESS_FAILED


def test_safe_process_runner_rejects_declared_output_outside_workspace(
    tmp_path: Path,
) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside.bin"

    with pytest.raises(VideoDemoError, match="工作区"):
        SafeProcessRunner(workspace_root=tmp_path).run(
            [sys.executable, "-c", "raise SystemExit('不应启动')"],
            timeout_seconds=5,
            output_paths=(outside,),
        )

    assert not outside.exists()


def test_safe_process_runner_rejects_output_through_workspace_symlink(
    tmp_path: Path,
) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    link = tmp_path / "escaped"
    link.symlink_to(outside, target_is_directory=True)

    with pytest.raises(VideoDemoError, match="符号链接"):
        SafeProcessRunner(workspace_root=tmp_path).run(
            [sys.executable, "-c", "raise SystemExit('不应启动')"],
            timeout_seconds=5,
            output_paths=(link / "output.bin",),
        )

    assert not (outside / "output.bin").exists()
