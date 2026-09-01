"""媒体理解领域包；具体契约从所属子模块显式导入。

保留历史的包级属性访问，但采用惰性加载，避免音频阶段导入包时把视频领域
实现带入进程。
"""

from importlib import import_module

_EXPORTS = {
    "DocumentEvidenceItem": ("video_demo.domain.evidence", "DocumentEvidenceItem"),
    "KeyframeEvidence": ("video_demo.domain.evidence", "KeyframeEvidence"),
    "SpeechSegment": ("video_demo.domain.evidence", "SpeechSegment"),
    "SubtitleCue": ("video_demo.domain.evidence", "SubtitleCue"),
    "TimeRange": ("video_demo.domain.run", "TimeRange"),
    "VideoAssetManifest": ("video_demo.domain.manifest", "VideoAssetManifest"),
    "VideoUnderstandingResult": ("video_demo.domain.document", "VideoUnderstandingResult"),
    "VisualObservationEvidence": (
        "video_demo.domain.evidence",
        "VisualObservationEvidence",
    ),
}


def __getattr__(name: str) -> object:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module = import_module(target[0])
    value = getattr(module, target[1])
    globals()[name] = value
    return value


__all__ = list(_EXPORTS)
