"""视频理解领域契约。"""

from video_demo.domain.document import VideoUnderstandingResult
from video_demo.domain.evidence import (
    DocumentEvidenceItem,
    KeyframeEvidence,
    SpeechSegment,
    SubtitleCue,
    VisualObservationEvidence,
)
from video_demo.domain.manifest import VideoAssetManifest
from video_demo.domain.run import TimeRange

__all__ = [
    "DocumentEvidenceItem",
    "KeyframeEvidence",
    "SpeechSegment",
    "SubtitleCue",
    "TimeRange",
    "VideoAssetManifest",
    "VideoUnderstandingResult",
    "VisualObservationEvidence",
]
