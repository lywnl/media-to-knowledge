"""视频理解领域契约。"""

from video_demo.domain.evidence import (
    KeyframeEvidence,
    OcrEvidence,
    SceneBoundary,
    SpeechSegment,
    TimelineEvidence,
)
from video_demo.domain.manifest import VideoAssetManifest
from video_demo.domain.result import VideoSegment, VideoSummary, VideoUnderstandingResult
from video_demo.domain.run import TimeRange

__all__ = [
    "KeyframeEvidence",
    "OcrEvidence",
    "SceneBoundary",
    "SpeechSegment",
    "TimeRange",
    "TimelineEvidence",
    "VideoAssetManifest",
    "VideoSegment",
    "VideoSummary",
    "VideoUnderstandingResult",
]
