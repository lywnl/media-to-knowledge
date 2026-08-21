from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.storage.workspace import verified_mp4_file


def _normal_ftyp(size: int = 16, suffix: bytes = b"") -> bytes:
    return size.to_bytes(4, "big") + b"ftypisom\x00\x00\x00\x00" + suffix


def _extended_ftyp(large_size: int, suffix: bytes = b"") -> bytes:
    return (
        b"\x00\x00\x00\x01ftyp"
        + large_size.to_bytes(8, "big")
        + b"isom\x00\x00\x00\x00"
        + suffix
    )


@pytest.mark.parametrize(
    ("payload", "accepted"),
    [
        (_normal_ftyp(), True),
        (_normal_ftyp(20, b"iso2"), True),
        (_normal_ftyp(12), False),
        (_normal_ftyp(17, b"x"), False),
        (_normal_ftyp(20), False),
        (_normal_ftyp(0), False),
        (_extended_ftyp(24), True),
        (_extended_ftyp(28, b"iso2"), True),
        (_extended_ftyp(20), False),
        (_extended_ftyp(25, b"x"), False),
        (_extended_ftyp(28), False),
        (b"\x00\x00\x00\x10free" + b"x" * 8 + _normal_ftyp(), False),
        (b"junk" + _normal_ftyp(), False),
    ],
)
def test_verified_mp4_parses_only_a_structurally_valid_first_ftyp_box(
    tmp_path: Path,
    payload: bytes,
    accepted: bool,
) -> None:
    run_root = Path("runs/scope/run_001")
    path = tmp_path / run_root / "media/proxy.mp4"
    path.parent.mkdir(parents=True)
    path.write_bytes(payload)

    if accepted:
        assert verified_mp4_file(
            tmp_path,
            run_root,
            path,
            expected_sha256=hashlib.sha256(payload).hexdigest(),
            expected_size_bytes=len(payload),
            max_size_bytes=1024,
            message="测试 MP4",
        ) == path
        return

    with pytest.raises(VideoDemoError) as raised:
        verified_mp4_file(
            tmp_path,
            run_root,
            path,
            expected_sha256=hashlib.sha256(payload).hexdigest(),
            expected_size_bytes=len(payload),
            max_size_bytes=1024,
            message="测试 MP4",
        )

    assert raised.value.code == ErrorCode.VIDEO_INPUT_INVALID
