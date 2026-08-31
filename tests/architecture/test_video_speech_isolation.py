from __future__ import annotations

from pathlib import Path


def test_video_asr_subprocess_does_not_import_vad_runtime() -> None:
    roots = (
        Path("src/video_demo/speech/subprocess_main.py"),
        Path("src/video_demo/application/production_speech.py"),
        Path("src/video_demo/speech/video_asr.py"),
        Path("src/video_demo/speech/snapshots.py"),
    )

    for path in roots:
        source = path.read_text(encoding="utf-8")
        assert "video_demo.speech.vad" not in source
        assert "SileroVadAdapter" not in source
        assert "build_cloud_asr_windows" not in source
