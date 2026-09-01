from pathlib import Path

import pytest

from video_demo.application.audio_speech import discard_audio_slice
from video_demo.errors import ErrorCode, VideoDemoError


def test_discard_audio_slice_rejects_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target.wav"
    target.write_bytes(b"audio")
    link = tmp_path / "slice.wav"
    link.symlink_to(target)

    with pytest.raises(VideoDemoError) as raised:
        discard_audio_slice(link)

    assert raised.value.code == ErrorCode.WORKSPACE_PATH_ESCAPE
    assert target.exists()


def test_discard_audio_slice_reports_unlink_failure() -> None:
    class FailingPath:
        def is_symlink(self) -> bool:
            return False

        def unlink(self, *, missing_ok: bool = False) -> None:
            del missing_ok
            raise OSError("模拟清理失败")

    path = FailingPath()

    with pytest.raises(VideoDemoError) as raised:
        discard_audio_slice(path)  # type: ignore[arg-type]

    assert raised.value.code == ErrorCode.AUDIO_PROCESS_FAILED
