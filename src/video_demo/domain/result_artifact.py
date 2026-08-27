"""唯一正式 3.0 结果 bundle 外观。"""

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
