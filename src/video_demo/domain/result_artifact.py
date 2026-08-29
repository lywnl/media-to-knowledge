"""历史结果 bundle 外观定义；当前正式文档契约见 document_artifact。"""

from video_demo.domain.document import TranscriptSource
from video_demo.domain.document_artifact import (
    MODEL_METRIC_NAMES,
    RESULT_STAGE_NAMES,
    DocumentArtifactPayload,
)

__all__ = [
    "MODEL_METRIC_NAMES",
    "RESULT_STAGE_NAMES",
    "DocumentArtifactPayload",
    "TranscriptSource",
]
