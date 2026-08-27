from __future__ import annotations

import heapq
from collections.abc import Callable, Sequence
from fractions import Fraction
from pathlib import Path

from video_demo.application.pipeline_contracts import (
    EvidencePreparationLimits,
    PreparedMedia,
    SceneIndex,
    scene_index_sha256,
)
from video_demo.domain.base import stable_identifier
from video_demo.domain.evidence import SceneBoundary
from video_demo.domain.manifest import Rational
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.storage.workspace import verified_mp4_file
from video_demo.visual.scenes import SceneDetector

_MIN_NORMALIZED_SCENE_MS = 1_200


class ProductionSceneIndexProvider:
    """只基于已验证代理视频和场景检测器构造 3.0 场景索引。"""

    def __init__(
        self,
        runtime_root: Path,
        scene_detector: SceneDetector,
        *,
        max_video_bytes: int = 4 * 1024 * 1024 * 1024,
    ) -> None:
        if max_video_bytes < 1:
            raise ValueError("视频字节上限必须大于 0")
        self._runtime_root = runtime_root.expanduser().resolve(strict=False)
        self._scene_detector = scene_detector
        self._max_video_bytes = max_video_bytes

    def prepare_scene_index(
        self,
        media: PreparedMedia,
        *,
        limits: EvidencePreparationLimits,
        is_cancel_requested: Callable[[], bool] = lambda: False,
    ) -> SceneIndex:
        proxy = verified_mp4_file(
            self._runtime_root,
            media.source.asset.run_relative_root,
            media.proxy_path,
            expected_sha256=media.proxy_sha256,
            expected_size_bytes=media.proxy_size_bytes,
            max_size_bytes=self._max_video_bytes,
            message="代理视频必须位于当前运行目录内",
        )
        _check_cancelled(is_cancel_requested)
        duration_ms = media.source.duration_ms
        frame_tolerance_ms = frame_tolerance_ms_for_rate(
            media.source.manifest.video_stream.average_frame_rate,
            is_variable_frame_rate=(
                media.source.manifest.video_stream.is_variable_frame_rate
            ),
        )
        raw_scenes = _validate_scenes(
            self._scene_detector.detect(
                proxy,
                duration_ms=duration_ms,
                source_sha256=media.proxy_sha256,
                frame_tolerance_ms=frame_tolerance_ms,
            ),
            duration_ms,
        )
        scenes = _normalize_scenes(
            raw_scenes,
            source_sha256=media.proxy_sha256,
            maximum=limits.max_scene_boundaries,
        )
        _check_cancelled(is_cancel_requested)
        return SceneIndex(
            proxy_sha256=media.proxy_sha256,
            duration_ms=duration_ms,
            frame_tolerance_ms=frame_tolerance_ms,
            scenes=scenes,
            index_sha256=scene_index_sha256(
                proxy_sha256=media.proxy_sha256,
                duration_ms=duration_ms,
                frame_tolerance_ms=frame_tolerance_ms,
                scenes=scenes,
            ),
        )


def frame_tolerance_ms_for_rate(
    frame_rate: Rational | None,
    *,
    is_variable_frame_rate: bool = False,
) -> int:
    """CFR 按单帧计算容差；VFR 的平均帧率无法约束最大间隔。"""

    if is_variable_frame_rate:
        return 100
    if (
        frame_rate is None
        or frame_rate.numerator <= 0
        or frame_rate.denominator <= 0
    ):
        return 100
    duration_ms = round(Fraction(1_000 * frame_rate.denominator, frame_rate.numerator))
    return min(100, max(1, duration_ms))


def _check_cancelled(is_cancel_requested: Callable[[], bool]) -> None:
    if is_cancel_requested():
        raise VideoDemoError(ErrorCode.JOB_CANCELLED, "任务已请求取消")


def _validate_scenes(
    scenes: Sequence[SceneBoundary],
    duration_ms: int,
) -> tuple[SceneBoundary, ...]:
    ordered = tuple(sorted(scenes, key=lambda item: (item.start_ms, item.end_ms, item.evidence_id)))
    if not ordered or ordered[0].start_ms != 0 or ordered[-1].end_ms != duration_ms:
        raise VideoDemoError(ErrorCode.VISUAL_RESULT_INVALID, "镜头结果未覆盖完整视频")
    for index, scene in enumerate(ordered):
        if scene.end_ms > duration_ms or (index and ordered[index - 1].end_ms != scene.start_ms):
            raise VideoDemoError(ErrorCode.VISUAL_RESULT_INVALID, "镜头结果越界或不连续")
        if index == 0 and scene.transition == "hard_cut":
            raise VideoDemoError(ErrorCode.VISUAL_RESULT_INVALID, "首个镜头不能伪造前置硬切")
    return ordered


def _normalize_scenes(
    scenes: Sequence[SceneBoundary],
    *,
    source_sha256: str,
    maximum: int,
) -> tuple[SceneBoundary, ...]:
    ranges = _merge_transient_scene_runs(scenes)
    ranges = _compress_scene_ranges(ranges, maximum)
    return tuple(
        SceneBoundary(
            evidence_id=stable_identifier(
                "scene",
                {
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "transition": transition,
                    "source_sha256": source_sha256,
                },
            ),
            start_ms=start_ms,
            end_ms=end_ms,
            transition=transition,
            score=1.0,
        )
        for start_ms, end_ms, transition in ranges
    )


def _compress_scene_ranges(
    ranges: list[tuple[int, int, str]],
    maximum: int,
) -> list[tuple[int, int, str]]:
    """按最短相邻组合压缩场景，并以原始左边界稳定解决同分。"""

    if maximum < 1:
        raise ValueError("规范场景数量上限必须大于 0")
    if len(ranges) <= maximum:
        return ranges

    starts = [item[0] for item in ranges]
    ends = [item[1] for item in ranges]
    transitions = [item[2] for item in ranges]
    previous: list[int | None] = [None, *range(len(ranges) - 1)]
    following: list[int | None] = [*range(1, len(ranges)), None]
    versions = [0] * len(ranges)
    active = [True] * len(ranges)
    heap: list[tuple[int, int, int, int, int]] = []

    def push_pair(left: int | None) -> None:
        if left is None or not active[left]:
            return
        right = following[left]
        if right is None or not active[right]:
            return
        heapq.heappush(
            heap,
            (
                ends[left] - starts[left] + ends[right] - starts[right],
                left,
                versions[left],
                right,
                versions[right],
            ),
        )

    for left in range(len(ranges) - 1):
        push_pair(left)

    remaining = len(ranges)
    while remaining > maximum:
        _duration, left, left_version, right, right_version = heapq.heappop(heap)
        if (
            not active[left]
            or not active[right]
            or following[left] != right
            or previous[right] != left
            or versions[left] != left_version
            or versions[right] != right_version
        ):
            continue
        right_neighbor = following[right]
        ends[left] = ends[right]
        following[left] = right_neighbor
        versions[left] += 1
        active[right] = False
        versions[right] += 1
        if right_neighbor is not None:
            previous[right_neighbor] = left
        remaining -= 1
        push_pair(previous[left])
        push_pair(left)

    compressed: list[tuple[int, int, str]] = []
    current: int | None = 0
    while current is not None:
        if active[current]:
            compressed.append((starts[current], ends[current], transitions[current]))
        current = following[current]
    return compressed


def _merge_transient_scene_runs(
    scenes: Sequence[SceneBoundary],
) -> list[tuple[int, int, str]]:
    normalized: list[tuple[int, int, str]] = []
    index = 0
    while index < len(scenes):
        scene = scenes[index]
        if scene.duration_ms >= _MIN_NORMALIZED_SCENE_MS:
            normalized.append((scene.start_ms, scene.end_ms, scene.transition))
            index += 1
            continue
        run_start = scene.start_ms
        run_end = scene.end_ms
        run_transition = scene.transition
        index += 1
        while index < len(scenes) and run_end - run_start < _MIN_NORMALIZED_SCENE_MS:
            run_end = scenes[index].end_ms
            index += 1
        if run_end - run_start >= _MIN_NORMALIZED_SCENE_MS:
            normalized.append((run_start, run_end, run_transition))
        elif normalized:
            previous_start, _previous_end, previous_transition = normalized[-1]
            normalized[-1] = (previous_start, run_end, previous_transition)
        else:
            normalized.append((run_start, run_end, run_transition))
    if len(normalized) > 1 and normalized[-1][1] - normalized[-1][0] < _MIN_NORMALIZED_SCENE_MS:
        previous_start, _previous_end, previous_transition = normalized[-2]
        normalized[-2:] = [(previous_start, normalized[-1][1], previous_transition)]
    return normalized
