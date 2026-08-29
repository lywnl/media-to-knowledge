from __future__ import annotations

from enum import StrEnum


class MediaKind(StrEnum):
    """上传资源的业务类型；视频、音频和图片分别进入独立流水线。"""

    VIDEO = "VIDEO"
    AUDIO = "AUDIO"
    IMAGE = "IMAGE"


class MediaRunStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    PARTIAL_SUCCEEDED = "PARTIAL_SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
