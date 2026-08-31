from __future__ import annotations

from typing import Any

import video_demo.main as main_module


def test_main_starts_api_on_project_default_port(monkeypatch: Any) -> None:
    calls: list[tuple[object, str, int]] = []

    monkeypatch.setattr(
        "uvicorn.run",
        lambda app, *, host, port: calls.append((app, host, port)),
    )

    main_module.main()

    assert calls == [(main_module.app, "127.0.0.1", 7999)]
