"""视频生产装配入口。

视频实现仍由 ``composition`` 承载，以兼容现有评测和调用方；独立入口让
视频 Worker 不再依赖音频装配模块。
"""

from __future__ import annotations

from video_demo.application.composition import build_worker

__all__ = ["build_worker"]
