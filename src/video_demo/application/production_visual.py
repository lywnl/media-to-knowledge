from __future__ import annotations

import hashlib
import heapq
import json
import logging
import re
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from fractions import Fraction
from itertools import pairwise
from pathlib import Path
from typing import Protocol

import httpx

from video_demo.application.adaptive_ocr import AdaptiveOcrRunner, ocr_language
from video_demo.application.pipeline_contracts import (
    EvidencePreparationLimits,
    PreparedMedia,
    SceneIndex,
    SpeechAnalysis,
    StageMetric,
    scene_index_sha256,
)
from video_demo.domain.base import stable_identifier
from video_demo.domain.evidence import (
    KeyframeEvidence,
    LegacyEvidenceItem,
    OcrEvidence,
    SceneBoundary,
)
from video_demo.domain.manifest import Rational
from video_demo.domain.run import TimeRange
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.fusion.merge import BoundaryPoint
from video_demo.integrations.baidu_ocr import BaiduOcrClient, BaiduOcrCredentials
from video_demo.integrations.video_port import VideoClipInput
from video_demo.media.transcode import ClipArtifact
from video_demo.storage.workspace import verified_mp4_file, verified_run_file
from video_demo.visual.keyframes import (
    FrameExtractor,
    KeyframeSelector,
    WindowFrameCandidates,
)
from video_demo.visual.ocr import (
    OcrClient,
    OcrProviderResponse,
)
from video_demo.visual.scenes import SceneDetector
from video_demo.visual.windows import (
    BoundaryCandidate,
    build_provisional_windows,
    merge_adjacent_windows,
    refine_windows_with_ocr,
)

_SPACE_AND_PUNCTUATION = re.compile(r"[^\w\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af]+")
_EMPTY_SPEECH_ANALYSIS = SpeechAnalysis(transcript_source="NONE")
_LOGGER = logging.getLogger(__name__)
_MIN_NORMALIZED_SCENE_MS = 1_200
_LEGACY_EVIDENCE_LIMITS = EvidencePreparationLimits(
    max_transcript_evidence_items=20_000,
    max_transcript_chars=2_000_000,
    max_scene_boundaries=20_000,
    max_base_segments=20_000,
)


class ClipClient(Protocol):
    def create_clip(
        self,
        source: Path,
        run_relative_root: Path,
        clip_id: str,
        time_range: TimeRange,
    ) -> ClipArtifact: ...


@dataclass(frozen=True, slots=True)
class VisualComponents:
    scene_detector: SceneDetector
    frame_extractor: FrameExtractor
    keyframe_selector: KeyframeSelector
    ocr_client: OcrClient
    clip_client: ClipClient


VisualComponentFactory = Callable[
    [PreparedMedia, Callable[[], bool]],
    VisualComponents,
]


@dataclass(frozen=True, slots=True)
class VisualPreparation:
    """Task 11 删除旧视觉链前保留的视觉准备 DTO。"""

    proxy_sha256: str
    proxy_size_bytes: int
    run_relative_root: Path
    duration_ms: int
    frame_tolerance_ms: int
    scenes: tuple[SceneBoundary, ...]
    preparation_sha256: str
    observation_windows: tuple[TimeRange, ...] = ()
    keyframes: tuple[KeyframeEvidence, ...] = ()
    ocr: tuple[OcrEvidence, ...] = ()
    warnings: tuple[str, ...] = ()
    stage_metrics: tuple[StageMetric, ...] = ()


@dataclass(frozen=True, slots=True)
class VisualAnalysis:
    """Task 11 删除旧视觉链前保留的视觉分析 DTO。"""

    evidence: tuple[LegacyEvidenceItem, ...]
    boundaries: tuple[BoundaryPoint, ...]
    clips: tuple[VideoClipInput, ...] = ()
    windows: tuple[TimeRange, ...] = ()
    warnings: tuple[str, ...] = ()
    stage_metrics: tuple[StageMetric, ...] = ()


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


class ProductionVisualAnalyzer:
    """编排真实 scene、帧、OCR 与最终 clip，并在每个外部边界前验证产物。"""

    def __init__(
        self,
        runtime_root: Path,
        component_factory: VisualComponentFactory,
        *,
        max_video_bytes: int = 4 * 1024 * 1024 * 1024,
        ocr_budget_timeout_seconds: float = 30.0,
        evidence_limits: EvidencePreparationLimits | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if max_video_bytes < 1:
            raise ValueError("视频字节上限必须大于 0")
        if ocr_budget_timeout_seconds <= 0:
            raise ValueError("OCR 预算超时必须大于 0")
        self._runtime_root = runtime_root.expanduser().resolve(strict=False)
        self._component_factory = component_factory
        self._max_video_bytes = max_video_bytes
        self._evidence_limits = evidence_limits or _LEGACY_EVIDENCE_LIMITS
        self._adaptive_ocr = (
            AdaptiveOcrRunner(
                self._runtime_root,
                timeout_seconds=ocr_budget_timeout_seconds,
            )
            if clock is None
            else AdaptiveOcrRunner(
                self._runtime_root,
                timeout_seconds=ocr_budget_timeout_seconds,
                clock=clock,
            )
        )

    def analyze(
        self,
        media: PreparedMedia,
        *,
        speech: SpeechAnalysis = _EMPTY_SPEECH_ANALYSIS,
        is_cancel_requested: Callable[[], bool] = lambda: False,
    ) -> VisualAnalysis:
        preparation = self.prepare(
            media,
            is_cancel_requested=is_cancel_requested,
        )
        return self.finalize(
            media,
            preparation,
            speech=speech,
            is_cancel_requested=is_cancel_requested,
        )

    def prepare(
        self,
        media: PreparedMedia,
        *,
        is_cancel_requested: Callable[[], bool] = lambda: False,
    ) -> VisualPreparation:
        scene_index = self.prepare_scene_index(
            media,
            limits=self._evidence_limits,
            is_cancel_requested=is_cancel_requested,
        )
        run_relative_root = media.source.asset.run_relative_root
        return VisualPreparation(
            proxy_sha256=scene_index.proxy_sha256,
            proxy_size_bytes=media.proxy_size_bytes,
            run_relative_root=run_relative_root,
            duration_ms=scene_index.duration_ms,
            frame_tolerance_ms=scene_index.frame_tolerance_ms,
            scenes=scene_index.scenes,
            preparation_sha256=_preparation_sha256(
                proxy_sha256=scene_index.proxy_sha256,
                proxy_size_bytes=media.proxy_size_bytes,
                run_relative_root=run_relative_root,
                duration_ms=scene_index.duration_ms,
                frame_tolerance_ms=scene_index.frame_tolerance_ms,
                scenes=scene_index.scenes,
            ),
        )

    def prepare_scene_index(
        self,
        media: PreparedMedia,
        *,
        limits: EvidencePreparationLimits,
        is_cancel_requested: Callable[[], bool] = lambda: False,
    ) -> SceneIndex:
        run_relative_root = media.source.asset.run_relative_root
        proxy = verified_mp4_file(
            self._runtime_root,
            run_relative_root,
            media.proxy_path,
            expected_sha256=media.proxy_sha256,
            expected_size_bytes=media.proxy_size_bytes,
            max_size_bytes=self._max_video_bytes,
            message="代理视频必须位于当前运行目录内",
        )
        self._check_cancelled(is_cancel_requested)
        components = self._component_factory(media, is_cancel_requested)
        duration_ms = media.source.duration_ms
        frame_tolerance_ms = frame_tolerance_ms_for_rate(
            media.source.manifest.video_stream.average_frame_rate,
            is_variable_frame_rate=media.source.manifest.video_stream.is_variable_frame_rate,
        )
        raw_scenes = _validate_scenes(
            components.scene_detector.detect(
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
        self._check_cancelled(is_cancel_requested)
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

    def finalize(
        self,
        media: PreparedMedia,
        preparation: VisualPreparation,
        *,
        speech: SpeechAnalysis,
        is_cancel_requested: Callable[[], bool] = lambda: False,
    ) -> VisualAnalysis:
        run_relative_root = media.source.asset.run_relative_root
        proxy = verified_mp4_file(
            self._runtime_root,
            run_relative_root,
            media.proxy_path,
            expected_sha256=media.proxy_sha256,
            expected_size_bytes=media.proxy_size_bytes,
            max_size_bytes=self._max_video_bytes,
            message="代理视频必须位于当前运行目录内",
        )
        duration_ms = media.source.duration_ms
        frame_tolerance_ms = frame_tolerance_ms_for_rate(
            media.source.manifest.video_stream.average_frame_rate,
            is_variable_frame_rate=media.source.manifest.video_stream.is_variable_frame_rate,
        )
        scenes = _validate_scenes(preparation.scenes, duration_ms)
        expected_preparation_sha256 = _preparation_sha256(
            proxy_sha256=media.proxy_sha256,
            proxy_size_bytes=media.proxy_size_bytes,
            run_relative_root=run_relative_root,
            duration_ms=duration_ms,
            frame_tolerance_ms=frame_tolerance_ms,
            scenes=scenes,
        )
        if (
            preparation.proxy_sha256 != media.proxy_sha256
            or preparation.proxy_size_bytes != media.proxy_size_bytes
            or preparation.run_relative_root != run_relative_root
            or preparation.duration_ms != duration_ms
            or preparation.frame_tolerance_ms != frame_tolerance_ms
            or preparation.preparation_sha256 != expected_preparation_sha256
        ):
            raise VideoDemoError(
                ErrorCode.VISUAL_RESULT_INVALID,
                "视觉准备结果与当前媒体不一致",
            )
        self._check_cancelled(is_cancel_requested)
        components = self._component_factory(media, is_cancel_requested)

        candidates = [
            BoundaryCandidate(item.start_ms, "scene", item.score)
            for item in scenes[1:]
        ]
        candidates.extend(
            BoundaryCandidate(item.timestamp_ms, item.source, item.score)
            for item in speech.boundary_candidates
        )
        provisional = build_provisional_windows(
            duration_ms=duration_ms,
            candidates=candidates,
        )
        groups = components.frame_extractor.extract(
            proxy,
            run_relative_root,
            provisional,
            is_cancel_requested=is_cancel_requested,
            frame_tolerance_ms=frame_tolerance_ms,
        )
        selected, warnings = self._select_keyframes(
            groups,
            provisional,
            run_relative_root,
            media.proxy_sha256,
            components.keyframe_selector,
            is_cancel_requested,
        )
        candidate_count = len(selected)
        ocr_result = self._adaptive_ocr.run(
            selected,
            speech=speech,
            media=media,
            run_relative_root=run_relative_root,
            client=components.ocr_client,
            is_cancel_requested=is_cancel_requested,
        )
        selected = ocr_result.keyframes
        ocr = ocr_result.evidence
        warnings.extend(ocr_result.warnings)
        self._adaptive_ocr.log_result(
            _LOGGER,
            duration_ms=duration_ms,
            candidate_count=candidate_count,
            result=ocr_result,
        )
        changes = _ocr_changes(ocr)
        final_windows = refine_windows_with_ocr(
            provisional,
            ocr_changes_ms=changes,
            duration_ms=duration_ms,
        )
        rebound_keyframes = tuple(_rebind(item, final_windows) for item in selected)
        rebound_ocr = tuple(_rebind(item, final_windows) for item in ocr)
        understanding_windows = merge_adjacent_windows(final_windows)
        evidence: tuple[LegacyEvidenceItem, ...] = (*scenes, *rebound_keyframes, *rebound_ocr)
        return VisualAnalysis(
            evidence=_sort_visual_evidence(evidence),
            boundaries=_build_boundaries(
                duration_ms,
                final_windows,
                speech,
                scenes,
                changes,
            ),
            clips=(),
            windows=understanding_windows,
            warnings=tuple(dict.fromkeys(warnings)),
        )

    def _select_keyframes(
        self,
        groups: Sequence[WindowFrameCandidates],
        windows: Sequence[TimeRange],
        run_relative_root: Path,
        proxy_sha256: str,
        selector: KeyframeSelector,
        is_cancel_requested: Callable[[], bool],
    ) -> tuple[tuple[KeyframeEvidence, ...], list[str]]:
        if tuple(item.window for item in groups) != tuple(windows):
            raise VideoDemoError(ErrorCode.VISUAL_RESULT_INVALID, "帧解码窗口与请求不一致")
        keyframes: list[KeyframeEvidence] = []
        warnings: list[str] = []
        expected_root = run_relative_root / "visual" / "keyframes"
        for group in groups:
            self._check_cancelled(is_cancel_requested)
            selection = selector.select(group.window, group.candidates)
            if not selection.frames:
                warnings.append(f"NO_KEYFRAME:{group.window.start_ms}:{group.window.end_ms}")
                continue
            for frame in selection.frames:
                self._check_cancelled(is_cancel_requested)
                if frame.relative_path.is_absolute():
                    raise VideoDemoError(
                        ErrorCode.WORKSPACE_PATH_ESCAPE,
                        "关键帧路径必须是运行时相对路径",
                    )
                path = verified_run_file(
                    self._runtime_root,
                    expected_root,
                    self._runtime_root / frame.relative_path,
                    message="关键帧必须位于当前运行的 visual/keyframes 目录内",
                )
                payload = path.read_bytes()
                if (
                    not payload
                    or len(payload) > 20 * 1024 * 1024
                    or not payload.startswith(b"\xff\xd8\xff")
                    or not payload.endswith(b"\xff\xd9")
                ):
                    raise VideoDemoError(ErrorCode.VISUAL_RESULT_INVALID, "关键帧 JPEG 格式非法")
                sha256 = hashlib.sha256(payload).hexdigest()
                keyframe_id = stable_identifier(
                    "keyframe",
                    {
                        "proxy_sha256": proxy_sha256,
                        "timestamp_ms": frame.timestamp_ms,
                        "sha256": sha256,
                    },
                )
                keyframes.append(
                    KeyframeEvidence(
                        evidence_id=stable_identifier(
                            "keyframe_evidence",
                            {"keyframe_id": keyframe_id, "timestamp_ms": frame.timestamp_ms},
                        ),
                        start_ms=group.window.start_ms,
                        end_ms=group.window.end_ms,
                        keyframe_id=keyframe_id,
                        timestamp_ms=frame.timestamp_ms,
                        relative_path=frame.relative_path.as_posix(),
                        mime_type="image/jpeg",
                        sha256=sha256,
                        perceptual_hash=frame.perceptual_hash,
                        size_bytes=len(payload),
                    ),
                )
        return tuple(keyframes), warnings

    def _create_clips(
        self,
        proxy: Path,
        run_relative_root: Path,
        proxy_sha256: str,
        windows: Sequence[TimeRange],
        client: ClipClient,
        is_cancel_requested: Callable[[], bool],
    ) -> tuple[VideoClipInput, ...]:
        clips: list[VideoClipInput] = []
        for window in windows:
            self._check_cancelled(is_cancel_requested)
            clip_id = stable_identifier(
                "clip",
                {
                    "proxy_sha256": proxy_sha256,
                    "start_ms": window.start_ms,
                    "end_ms": window.end_ms,
                },
            )
            artifact = client.create_clip(proxy, run_relative_root, clip_id, window)
            if (
                artifact.clip_id != clip_id
                or artifact.start_ms != window.start_ms
                or artifact.end_ms != window.end_ms
            ):
                raise VideoDemoError(ErrorCode.VISUAL_RESULT_INVALID, "clip 返回时间或标识非法")
            if Path(artifact.relative_path).is_absolute() or artifact.size_bytes < 1:
                raise VideoDemoError(ErrorCode.VISUAL_RESULT_INVALID, "clip 返回路径或大小非法")
            path = verified_mp4_file(
                self._runtime_root,
                run_relative_root / "visual" / "clips",
                Path(artifact.relative_path),
                expected_sha256=artifact.sha256,
                expected_size_bytes=artifact.size_bytes,
                max_size_bytes=self._max_video_bytes,
                message="视频 clip 必须位于当前运行目录内",
            )
            clips.append(
                VideoClipInput(
                    clip_id=clip_id,
                    start_ms=window.start_ms,
                    end_ms=window.end_ms,
                    path=path,
                    mime_type="video/mp4",
                    sha256=artifact.sha256,
                ),
            )
        return tuple(clips)

    @staticmethod
    def _check_cancelled(is_cancel_requested: Callable[[], bool]) -> None:
        if is_cancel_requested():
            raise VideoDemoError(ErrorCode.JOB_CANCELLED, "任务已请求取消")


def _limit_keyframes(
    keyframes: Sequence[KeyframeEvidence],
    maximum: int,
) -> tuple[KeyframeEvidence, ...]:
    if maximum < 1:
        raise ValueError("关键帧上限必须大于 0")
    ordered = tuple(
        sorted(keyframes, key=lambda item: (item.timestamp_ms, item.keyframe_id)),
    )
    if len(ordered) <= maximum:
        return ordered
    if maximum == 1:
        return (ordered[len(ordered) // 2],)
    denominator = maximum - 1
    first_timestamp_ms = ordered[0].timestamp_ms
    time_span_ms = ordered[-1].timestamp_ms - first_timestamp_ms
    remaining = list(ordered)
    limited: list[KeyframeEvidence] = []
    for index in range(maximum):
        target_timestamp_ms = (
            first_timestamp_ms
            + (index * time_span_ms + denominator // 2) // denominator
        )
        nearest = min(
            remaining,
            key=lambda item: (
                abs(item.timestamp_ms - target_timestamp_ms),
                item.timestamp_ms,
                item.keyframe_id,
            ),
        )
        limited.append(nearest)
        remaining.remove(nearest)
    return tuple(
        sorted(limited, key=lambda item: (item.timestamp_ms, item.keyframe_id)),
    )


class LazyBaiduOcrClient:
    """仅在首次真实 OCR 时读取凭据，并跨任务复用 HTTP client 与 Token cache。"""

    def __init__(
        self,
        http_client: httpx.Client,
        credentials_provider: Callable[[], tuple[str | None, str | None]],
        *,
        endpoint: str,
    ) -> None:
        self._http_client = http_client
        self._credentials_provider = credentials_provider
        self._endpoint = endpoint
        self._client: BaiduOcrClient | None = None
        self._lock = threading.Lock()

    def recognize(
        self,
        image: bytes,
        language: str,
        *,
        deadline: float | None = None,
    ) -> OcrProviderResponse:
        return self._get_client().recognize(image, language, deadline=deadline)

    def _get_client(self) -> BaiduOcrClient:
        with self._lock:
            if self._client is not None:
                return self._client
            api_key, secret_key = self._credentials_provider()
            if not api_key or not secret_key:
                raise VideoDemoError(
                    ErrorCode.OCR_AUTHENTICATION_FAILED,
                    "百度 OCR 凭据未配置",
                )
            self._client = BaiduOcrClient(
                self._http_client,
                BaiduOcrCredentials(api_key=api_key, secret_key=secret_key),
                endpoint=self._endpoint,
            )
            return self._client


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
        while (
            index < len(scenes)
            and run_end - run_start < _MIN_NORMALIZED_SCENE_MS
        ):
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


def _preparation_sha256(
    *,
    proxy_sha256: str,
    proxy_size_bytes: int,
    run_relative_root: Path,
    duration_ms: int,
    frame_tolerance_ms: int,
    scenes: Sequence[SceneBoundary],
) -> str:
    encoded = json.dumps(
        {
            "proxy_sha256": proxy_sha256,
            "proxy_size_bytes": proxy_size_bytes,
            "run_relative_root": run_relative_root.as_posix(),
            "duration_ms": duration_ms,
            "frame_tolerance_ms": frame_tolerance_ms,
            "scenes": [item.model_dump(mode="json") for item in scenes],
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _ocr_language(
    timestamp_ms: int,
    speech: SpeechAnalysis,
    media: PreparedMedia,
) -> tuple[str | None, str | None]:
    return ocr_language(timestamp_ms, speech, media)


def _ocr_changes(evidence: Sequence[OcrEvidence]) -> tuple[int, ...]:
    ordered = sorted(evidence, key=lambda item: (item.timestamp_ms, item.evidence_id))
    changes: list[int] = []
    for previous, current in pairwise(ordered):
        left = _normalized_ocr_text(previous)
        right = _normalized_ocr_text(current)
        if _is_obvious_text_change(left, right):
            changes.append(current.timestamp_ms)
    return tuple(sorted(set(changes)))


def _normalized_ocr_text(evidence: OcrEvidence) -> str:
    text = " ".join(line.text for line in evidence.lines).casefold()
    return _SPACE_AND_PUNCTUATION.sub("", text)


def _is_obvious_text_change(left: str, right: str) -> bool:
    if not left and not right:
        return False
    if not left or not right:
        return True
    if left == right:
        return False
    return SequenceMatcher(a=left, b=right, autojunk=False).ratio() < 0.65


def _rebind(
    item: KeyframeEvidence | OcrEvidence,
    windows: Sequence[TimeRange],
) -> KeyframeEvidence | OcrEvidence:
    window = next(
        (
            candidate
            for candidate in windows
            if candidate.start_ms <= item.timestamp_ms < candidate.end_ms
        ),
        None,
    )
    if window is None:
        raise VideoDemoError(ErrorCode.VISUAL_RESULT_INVALID, "视觉证据时间不属于最终窗口")
    return item.model_copy(update={"start_ms": window.start_ms, "end_ms": window.end_ms})


def _build_boundaries(
    duration_ms: int,
    windows: Sequence[TimeRange],
    speech: SpeechAnalysis,
    scenes: Sequence[SceneBoundary],
    ocr_changes: Sequence[int],
) -> tuple[BoundaryPoint, ...]:
    grouped: dict[int, set[str]] = {0: {"video_start"}, duration_ms: {"video_end"}}
    for window in windows:
        grouped.setdefault(window.start_ms, set()).add("clip_edge")
        grouped.setdefault(window.end_ms, set()).add("clip_edge")
    for candidate in speech.boundary_candidates:
        grouped.setdefault(candidate.timestamp_ms, set()).add(candidate.source)
    for scene in scenes[1:]:
        source = "scene_hard" if scene.transition == "hard_cut" else "scene"
        grouped.setdefault(scene.start_ms, set()).add(source)
    for timestamp_ms in ocr_changes:
        grouped.setdefault(timestamp_ms, set()).add("ocr_change")
    return tuple(
        BoundaryPoint(timestamp_ms=timestamp, sources=tuple(sorted(sources)))
        for timestamp, sources in sorted(grouped.items())
    )


def _sort_visual_evidence(
    items: Sequence[LegacyEvidenceItem],
) -> tuple[LegacyEvidenceItem, ...]:
    rank = {SceneBoundary: 0, KeyframeEvidence: 1, OcrEvidence: 2}
    allowed = (SceneBoundary, KeyframeEvidence, OcrEvidence)
    if any(not isinstance(item, allowed) for item in items):
        raise VideoDemoError(ErrorCode.VISUAL_RESULT_INVALID, "视觉阶段返回了非法证据类型")
    return tuple(
        sorted(
            items,
            key=lambda item: (
                item.start_ms,
                rank[type(item)],
                getattr(item, "timestamp_ms", item.start_ms),
                item.end_ms,
                item.evidence_id,
            ),
        ),
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "LazyBaiduOcrClient",
    "ProductionVisualAnalyzer",
    "VisualComponents",
    "WindowFrameCandidates",
    "frame_tolerance_ms_for_rate",
]
