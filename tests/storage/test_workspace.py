from __future__ import annotations

import hashlib
from pathlib import Path

from video_demo.storage.workspace import verified_visual_file


def test_verified_visual_file_accepts_non_mp4_source_with_integrity_checks(
    tmp_path: Path,
) -> None:
    run_root = Path("runs/scope/run_001")
    path = tmp_path / run_root / "input/source.mkv"
    path.parent.mkdir(parents=True)
    payload = b"source video bytes"
    path.write_bytes(payload)

    assert verified_visual_file(
        tmp_path,
        run_root,
        path,
        expected_sha256=hashlib.sha256(payload).hexdigest(),
        expected_size_bytes=len(payload),
        max_size_bytes=1024,
        message="测试视觉输入",
    ) == path
