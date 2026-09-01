from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from video_demo.api.app import create_app
from video_demo.config import Settings


class _Scheduler:
    def __init__(self) -> None:
        self.started = False
        self.shutdown_called = False
        self.submitted: list[tuple[object, str]] = []

    def recover(self, items: tuple[object, ...] = ()) -> int:
        self.recovered = items
        return len(items)

    def start(self) -> None:
        self.started = True

    def shutdown(self, *, wait: bool = False, timeout: float | None = None) -> None:
        assert wait is True
        assert timeout == 10
        self.shutdown_called = True

    def submit(self, scope: object, run_id: str) -> str:
        self.submitted.append((scope, run_id))
        return "accepted"


def test_fastapi_lifespan_starts_and_stops_audio_scheduler(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import shutil

    import video_demo.api.app as app_module

    shutil.copytree(Path.cwd() / "migrations", tmp_path / "migrations")

    video_scheduler = _Scheduler()
    audio_scheduler = _Scheduler()
    monkeypatch.setattr(
        app_module,
        "build_video_scheduler",
        lambda *_args, **_kwargs: video_scheduler,
    )
    monkeypatch.setattr(
        app_module,
        "build_audio_scheduler",
        lambda *_args, **_kwargs: audio_scheduler,
    )
    settings = Settings(workspace_root=tmp_path, _env_file=None)

    with TestClient(create_app(settings)) as client:
        assert client.app.state.audio_scheduler is audio_scheduler
        assert client.app.state.container.audio_scheduler is audio_scheduler
        assert audio_scheduler.started is True

    assert audio_scheduler.shutdown_called is True

def test_fastapi_lifespan_starts_and_stops_video_scheduler(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import shutil

    import video_demo.api.app as app_module

    shutil.copytree(Path.cwd() / "migrations", tmp_path / "migrations")

    scheduler = _Scheduler()
    monkeypatch.setattr(app_module, "build_video_scheduler", lambda *_args, **_kwargs: scheduler)
    settings = Settings(workspace_root=tmp_path, _env_file=None)

    with TestClient(create_app(settings)) as client:
        assert client.app.state.video_scheduler is scheduler
        assert client.app.state.container.video_scheduler is scheduler
        assert scheduler.started is True

    assert scheduler.shutdown_called is True
