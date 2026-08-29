from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from video_demo.domain.base import stable_identifier
from video_demo.domain.evidence import SceneBoundary
from video_demo.errors import ErrorCode, VideoDemoError


@dataclass(frozen=True, slots=True)
class RawScene:
    start_ms: int
    end_ms: int
    score: float
    transition: str


class SceneDetector(Protocol):
    def detect(
        self,
        proxy: Path,
        *,
        duration_ms: int,
        source_sha256: str,
        frame_tolerance_ms: int,
    ) -> tuple[SceneBoundary, ...]: ...


SceneModuleLoader = Callable[[], tuple[Any, Any]]

_MAX_AUDIO_LED_CONTAINER_TAIL_MS = 100


class PySceneDetectAdapter:
    """通过 PySceneDetect 0.6 公共 API 懒加载执行视觉输入检测。"""

    def __init__(self, *, module_loader: SceneModuleLoader | None = None) -> None:
        self._module_loader = module_loader or _load_scenedetect

    def detect(
        self,
        proxy: Path,
        *,
        duration_ms: int,
        source_sha256: str,
        frame_tolerance_ms: int,
        input_fd: int | None = None,
    ) -> tuple[SceneBoundary, ...]:
        if duration_ms <= 0 or not 0 <= frame_tolerance_ms <= 100:
            raise VideoDemoError(ErrorCode.VISUAL_MEDIA_INVALID, "视频时长非法")
        try:
            module, detectors = self._module_loader()
            video = module.open_video(
                f"/dev/fd/{input_fd}" if input_fd is not None else str(proxy)
            )
            manager = module.SceneManager()
            manager.add_detector(detectors.ContentDetector())
            manager.detect_scenes(video=video, show_progress=False)
            raw_list = manager.get_scene_list(start_in_scene=True)
            scenes = _convert_scene_list(
                raw_list,
                duration_ms,
                frame_tolerance_ms=frame_tolerance_ms,
            )
            return build_scene_evidence(scenes, source_sha256=source_sha256)
        except VideoDemoError:
            raise
        except (ModuleNotFoundError, ImportError):
            raise VideoDemoError(
                ErrorCode.VISUAL_DEPENDENCY_UNAVAILABLE,
                "视觉分析可选依赖不可用",
            ) from None
        except Exception:
            raise VideoDemoError(
                ErrorCode.VISUAL_MEDIA_INVALID,
                "PySceneDetect 无法分析视觉输入",
            ) from None


def build_scene_evidence(
    raw_scenes: Sequence[RawScene],
    *,
    source_sha256: str,
) -> tuple[SceneBoundary, ...]:
    ordered = sorted(raw_scenes, key=lambda item: (item.start_ms, item.end_ms))
    scenes: list[SceneBoundary] = []
    for raw in ordered:
        if scenes and raw.start_ms < scenes[-1].end_ms:
            raise ValueError("镜头区间不得重叠")
        if raw.transition not in ("hard_cut", "gradual", "candidate"):
            raise ValueError("不支持的镜头转场类型")
        scenes.append(
            SceneBoundary(
                evidence_id=stable_identifier(
                    "scene",
                    {
                        "start_ms": raw.start_ms,
                        "end_ms": raw.end_ms,
                        "transition": raw.transition,
                        "source_sha256": source_sha256,
                    },
                ),
                start_ms=raw.start_ms,
                end_ms=raw.end_ms,
                transition=raw.transition,
                score=raw.score,
            ),
        )
    return tuple(scenes)


def _load_scenedetect() -> tuple[Any, Any]:
    import scenedetect
    from scenedetect import detectors

    return scenedetect, detectors


def _convert_scene_list(
    raw_list: object,
    duration_ms: int,
    *,
    frame_tolerance_ms: int,
) -> tuple[RawScene, ...]:
    if not isinstance(raw_list, (list, tuple)):
        raise ValueError("镜头列表类型非法")
    if not raw_list:
        return (RawScene(0, duration_ms, 1.0, "candidate"),)
    converted: list[tuple[int, int]] = []
    for raw in raw_list:
        if not isinstance(raw, (list, tuple)) or len(raw) != 2:
            raise ValueError("镜头时间结构非法")
        converted.append((_milliseconds(raw[0]), _milliseconds(raw[1])))
    if abs(converted[0][0]) > frame_tolerance_ms:
        raise ValueError("镜头起点超出视觉时基容差")
    detected_end_ms = converted[-1][1]
    allowed_tail_ms = (
        max(frame_tolerance_ms, _MAX_AUDIO_LED_CONTAINER_TAIL_MS)
        if detected_end_ms < duration_ms
        else frame_tolerance_ms
    )
    if abs(detected_end_ms - duration_ms) > allowed_tail_ms:
        raise ValueError("镜头终点超出视觉时基容差")
    converted[0] = (0, converted[0][1])
    converted[-1] = (converted[-1][0], duration_ms)

    result: list[RawScene] = []
    for index, (start, end) in enumerate(converted):
        if start < 0 or end <= start or end > duration_ms:
            raise ValueError("镜头时间越界")
        if index == 0 and start != 0:
            raise ValueError("镜头未覆盖视频起点")
        if result and start != result[-1].end_ms:
            raise ValueError("镜头区间必须连续")
        result.append(
            RawScene(
                start_ms=start,
                end_ms=end,
                score=1.0,
                transition="candidate" if index == 0 else "hard_cut",
            ),
        )
    if result[-1].end_ms != duration_ms:
        raise ValueError("镜头未覆盖视频终点")
    return tuple(result)


def _milliseconds(timecode: object) -> int:
    getter = getattr(timecode, "get_seconds", None)
    if not callable(getter):
        raise ValueError("镜头时间码非法")
    seconds = float(getter())
    if not math.isfinite(seconds):
        raise ValueError("镜头时间码必须有限")
    return round(seconds * 1000)
