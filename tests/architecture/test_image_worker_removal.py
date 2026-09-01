from __future__ import annotations

from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]


def test_image_external_worker_entrypoint_is_removed() -> None:
    assert not (_ROOT / "src/video_demo/image_worker_main.py").exists()
    pyproject = (_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "video-demo-image-worker" not in pyproject


def test_image_production_uses_stage_scheduler_instead_of_reliable_worker() -> None:
    source_files = (
        _ROOT / "src/video_demo/application/image_composition.py",
        _ROOT / "src/video_demo/application/image_pipeline_executor.py",
        _ROOT / "src/video_demo/application/image_scheduler.py",
        _ROOT / "src/video_demo/application/media_runs.py",
    )
    source = "\n".join(path.read_text(encoding="utf-8") for path in source_files)
    assert "build_image_worker" not in source
    assert "ReliableWorker" not in source
    assert "ImageTaskScheduler" in source
