"""视频理解领域契约。"""

from video_demo.domain.evidence import (
    AlignedWord,
    AudioEvent,
    KeyframeEvidence,
    OcrEvidence,
    SceneBoundary,
    SpeakerTurn,
    SpeechSegment,
    TimelineEvidence,
)
from video_demo.domain.manifest import VideoAssetManifest
from video_demo.domain.result import VideoSegment, VideoSummary, VideoUnderstandingResult
from video_demo.domain.run import TimeRange

__all__ = [
    "AlignedWord",
    "AudioEvent",
    "KeyframeEvidence",
    "OcrEvidence",
    "SceneBoundary",
    "SpeakerTurn",
    "SpeechSegment",
    "TimeRange",
    "TimelineEvidence",
    "VideoAssetManifest",
    "VideoSegment",
    "VideoSummary",
    "VideoUnderstandingResult",
]
