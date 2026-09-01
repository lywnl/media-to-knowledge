from __future__ import annotations

from pathlib import Path


def test_video_asr_direct_path_does_not_import_vad_or_python_subprocess() -> None:
    roots = (
        Path("src/video_demo/application/production_speech.py"),
        Path("src/video_demo/application/composition.py"),
        Path("src/video_demo/speech/video_asr.py"),
        Path("src/video_demo/speech/snapshots.py"),
    )

    for path in roots:
        source = path.read_text(encoding="utf-8")
        assert "video_demo.speech.vad" not in source
        assert "SileroVadAdapter" not in source
        assert "build_cloud_asr_windows" not in source
        assert "subprocess_main" not in source
        assert "IsolatedSpeechAnalyzer" not in source
