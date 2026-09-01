from __future__ import annotations

from pathlib import Path


def test_external_worker_runtime_and_entrypoints_are_removed() -> None:
    assert not Path("src/video_demo/worker/runtime.py").exists()
    assert not Path("src/video_demo/worker/stages.py").exists()
    assert not Path("src/video_demo/worker/__init__.py").exists()
    assert not Path("src/video_demo/audio_worker_main.py").exists()
    assert not Path("src/video_demo/image_worker_main.py").exists()


def test_image_business_handler_has_no_external_entrypoint() -> None:
    module = __import__(
        "video_demo.application.image_pipeline_handler",
        fromlist=["ImageJobHandler"],
    )
    handler = module.ImageJobHandler
    # 类对象本身继承 type.__call__；这里只检查业务类没有显式的外部入口。
    assert "__call__" not in handler.__dict__
    assert "_mark_unsuccessful" not in handler.__dict__


def test_production_sources_do_not_reference_legacy_worker_modules() -> None:
    source_root = Path("src/video_demo")
    forbidden = (
        "video_demo.worker",
        "application.media_workers",
        "build_image_worker",
        "build_audio_worker",
    )
    for path in source_root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert all(marker not in text for marker in forbidden), path
