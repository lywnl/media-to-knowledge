from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path

import pytest

from video_demo.application.chapter_frames import ChapterFrameSearcher
from video_demo.application.pipeline_contracts import (
    PipelineRunConfig,
    PreparedMedia,
    ProbedAsset,
    RegisteredAsset,
    SceneIndex,
    scene_index_sha256,
)
from video_demo.domain.base import stable_identifier
from video_demo.domain.document import DocumentGenerationConfig
from video_demo.domain.document_plan import ChapterPlan, VisualSearchTarget
from video_demo.domain.evidence import SceneBoundary, SpeechSegment
from video_demo.domain.manifest import Rational, VideoAssetManifest, VideoStream
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.media.probe import ProbeLimits
from video_demo.visual.candidate_artifacts import CandidateArtifactSession
from video_demo.visual.keyframes import (
    ExactFrameSampleResult,
    FrameCandidate,
    FrameSample,
)

_ASSET_SHA256 = "a" * 64
_JPEG = b"\xff\xd8\xffchapter-frame\xff\xd9"


class _FakeExtractor:
    def __init__(
        self,
        results: dict[int, tuple[str, bytes | None]],
        *,
        result_for_sample: Callable[
            [FrameSample], tuple[str, bytes | None, float, str]
        ]
        | None = None,
    ) -> None:
        self._results = results
        self._result_for_sample = result_for_sample
        self.calls: list[tuple[tuple[FrameSample, ...], int]] = []

    def extract_samples(
        self,
        _proxy: Path,
        run_relative_root: Path,
        samples: tuple[FrameSample, ...],
        *,
        is_cancel_requested: object,
        frame_tolerance_ms: int,
        artifact_session: CandidateArtifactSession | None = None,
    ) -> tuple[ExactFrameSampleResult, ...]:
        del is_cancel_requested
        self.calls.append((samples, frame_tolerance_ms))
        output: list[ExactFrameSampleResult] = []
        for sample in samples:
            if self._result_for_sample is None:
                status, payload = self._results.get(
                    sample.timestamp_ms,
                    ("DECODE_FAILED", None),
                )
                sharpness = float(sample.timestamp_ms)
                perceptual_hash = "0123456789abcdef"
            else:
                status, payload, sharpness, perceptual_hash = self._result_for_sample(sample)
            candidate = None
            artifact_status = None
            if payload is not None:
                digest = hashlib.sha256(payload).hexdigest()
                relative_path = run_relative_root / "visual" / "candidates" / f"{digest}.jpg"
                publication = None
                if artifact_session is not None:
                    artifact_session.prepare_run(run_relative_root)
                    publication = artifact_session.publish_jpeg(relative_path, payload, digest)
                if publication is not None and publication.status == "BUDGET_REJECTED":
                    output.append(
                        ExactFrameSampleResult(
                            sample_id=sample.sample_id,
                            requested_timestamp_ms=sample.timestamp_ms,
                            status="SUCCEEDED",
                            artifact_status="BUDGET_REJECTED",
                        ),
                    )
                    continue
                absolute_path = self.runtime_root / relative_path
                if publication is None:
                    absolute_path.parent.mkdir(parents=True, exist_ok=True)
                    absolute_path.write_bytes(payload)
                created = publication.created_by_call if publication is not None else True
                candidate = FrameCandidate(
                    timestamp_ms=sample.timestamp_ms,
                    sharpness=sharpness,
                    black_ratio=0.0,
                    perceptual_hash=perceptual_hash,
                    relative_path=relative_path,
                    created_by_call=created,
                )
                artifact_status = "PUBLISHED"
            output.append(
                ExactFrameSampleResult(
                    sample_id=sample.sample_id,
                    requested_timestamp_ms=sample.timestamp_ms,
                    status=status,
                    candidate=candidate,
                    artifact_status=artifact_status,
                ),
            )
        return tuple(output)


def test_single_chapter_keeps_base_candidate_when_semantic_target_has_no_candidate(
    tmp_path: Path,
) -> None:
    media, scene_index = _fixture(tmp_path)
    chapter = _chapter(
        semantic_targets=(_semantic_target("target_semantic", "asr_001"),),
    )
    extractor = _FakeExtractor(
        {
            2_000: ("QUALITY_REJECTED", None),
            3_500: ("QUALITY_REJECTED", None),
            5_000: ("SUCCEEDED", _JPEG),
        },
    )
    extractor.runtime_root = tmp_path

    batch = ChapterFrameSearcher(tmp_path, extractor).search(
        media,
        (chapter,),
        {"asr_001": _speech()},
        scene_index,
        DocumentGenerationConfig(),
        is_cancel_requested=lambda: False,
    )

    assert batch.chapter_status == ((chapter.chapter_id, "SUCCEEDED"),)
    assert tuple(candidate.target_ids for candidate in batch.frame_sets[0].candidates) == (
        ("target_base",),
    )


def test_search_uses_scene_index_tolerance_and_merges_same_content_targets(
    tmp_path: Path,
) -> None:
    media, scene_index = _fixture(tmp_path)
    extractor = _FakeExtractor({2_000: ("SUCCEEDED", _JPEG), 5_000: ("SUCCEEDED", _JPEG)})
    extractor.runtime_root = tmp_path
    searcher = ChapterFrameSearcher(tmp_path, extractor, max_candidate_bytes=1024)
    chapter = _chapter(
        semantic_targets=(_semantic_target("target_semantic", "asr_001"),),
    )

    batch = searcher.search(
        media,
        (chapter,),
        {"asr_001": _speech()},
        scene_index,
        DocumentGenerationConfig(),
        is_cancel_requested=lambda: False,
    )

    assert extractor.calls[0][1] == scene_index.frame_tolerance_ms == 40
    assert batch.frame_tolerance_ms == 40
    assert batch.chapter_status == ((chapter.chapter_id, "SUCCEEDED"),)
    assert len(batch.frame_sets[0].candidates) == 1
    assert set(batch.frame_sets[0].candidates[0].target_ids) == {
        "target_semantic",
        "target_base",
    }


def test_same_sha_preserves_each_target_window_when_selecting_window_best(
    tmp_path: Path,
) -> None:
    media, original_scene_index = _fixture(tmp_path)
    scenes = tuple(
        SceneBoundary(
            evidence_id=f"scene_{index:03d}",
            start_ms=index * 2_500,
            end_ms=(index + 1) * 2_500,
            transition="candidate",
            score=0.9,
        )
        for index in range(4)
    )
    scene_index = SceneIndex(
        proxy_sha256=media.proxy_sha256,
        duration_ms=original_scene_index.duration_ms,
        frame_tolerance_ms=original_scene_index.frame_tolerance_ms,
        scenes=scenes,
        index_sha256=scene_index_sha256(
            proxy_sha256=media.proxy_sha256,
            duration_ms=original_scene_index.duration_ms,
            frame_tolerance_ms=original_scene_index.frame_tolerance_ms,
            scenes=scenes,
        ),
    )
    chapter = ChapterPlan(
        chapter_id="chapter_001",
        start_ms=0,
        end_ms=10_000,
        segment_refs=("segment_001",),
        title_hint="章节",
        visual_mode="COMPARISON",
        semantic_targets=(
            _semantic_target("target_semantic_1", "asr_001"),
            _semantic_target("target_semantic_2", "asr_002"),
        ),
        base_coverage_targets=(),
    )

    def result_for_sample(sample: FrameSample) -> tuple[str, bytes | None, float, str]:
        if sample.timestamp_ms in {3_500, 6_500}:
            return "SUCCEEDED", _JPEG, 1.0, "0000000000000000"
        payload = b"\xff\xd8\xff" + sample.sample_id.encode("ascii") + b"\xff\xd9"
        return "SUCCEEDED", payload, 10.0, "ffffffffffffffff"

    extractor = _FakeExtractor({}, result_for_sample=result_for_sample)
    extractor.runtime_root = tmp_path

    batch = ChapterFrameSearcher(
        tmp_path,
        extractor,
        max_hash_distance_for_duplicate=0,
    ).search(
        media,
        (chapter,),
        {"asr_001": _speech(), "asr_002": _speech_2()},
        scene_index,
        DocumentGenerationConfig(),
        is_cancel_requested=lambda: False,
    )

    assert len(batch.frame_sets[0].candidates) == 2
    assert all(
        candidate.sha256 != hashlib.sha256(_JPEG).hexdigest()
        for candidate in batch.frame_sets[0].candidates
    )


def test_search_distinguishes_normal_no_candidate_from_decode_degradation(
    tmp_path: Path,
) -> None:
    media, scene_index = _fixture(tmp_path)
    chapter = _chapter()
    normal = _FakeExtractor(
        {
            2_500: ("SUCCEEDED", b"\xff\xd8\xffblack-1\xff\xd9"),
            5_000: ("SUCCEEDED", b"\xff\xd8\xffblack\xff\xd9"),
            7_500: ("SUCCEEDED", b"\xff\xd8\xffblack-2\xff\xd9"),
        },
    )
    normal.runtime_root = tmp_path
    degraded = _FakeExtractor({})
    degraded.runtime_root = tmp_path

    normal_batch = ChapterFrameSearcher(
        tmp_path,
        normal,
        max_candidate_bytes=1024,
        maximum_black_ratio=-1.0,
    ).search(
        media,
        (chapter,),
        {},
        scene_index,
        DocumentGenerationConfig(),
        is_cancel_requested=lambda: False,
    )
    degraded_batch = ChapterFrameSearcher(
        tmp_path,
        degraded,
        max_candidate_bytes=1024,
    ).search(
        media,
        (chapter,),
        {},
        scene_index,
        DocumentGenerationConfig(),
        is_cancel_requested=lambda: False,
    )

    assert normal_batch.chapter_status == ((chapter.chapter_id, "NO_CANDIDATE"),)
    assert normal_batch.status == "SUCCEEDED"
    assert degraded_batch.chapter_status == ((chapter.chapter_id, "DEGRADED"),)
    assert degraded_batch.status == "PARTIAL_SUCCEEDED"


def test_visual_disabled_short_circuits_without_opening_extractor(tmp_path: Path) -> None:
    media, scene_index = _fixture(tmp_path)
    extractor = _FakeExtractor({})
    extractor.runtime_root = tmp_path
    chapter = _chapter()

    batch = ChapterFrameSearcher(tmp_path, extractor).search(
        media,
        (chapter,),
        {},
        scene_index,
        DocumentGenerationConfig(max_visuals_per_chapter=0),
        is_cancel_requested=lambda: False,
    )

    assert extractor.calls == []
    assert batch.chapter_status == ((chapter.chapter_id, "DISABLED"),)
    assert batch.metrics == {"visual_disabled_chapters": 1}


def test_candidate_budget_removes_current_call_files_and_degrades_chapter(
    tmp_path: Path,
) -> None:
    media, scene_index = _fixture(tmp_path)
    payloads = {
        2_500: b"\xff\xd8\xffcandidate-one\xff\xd9",
        5_000: b"\xff\xd8\xffcandidate-two\xff\xd9",
        7_500: b"\xff\xd8\xffcandidate-three\xff\xd9",
    }
    extractor = _FakeExtractor(
        {timestamp_ms: ("SUCCEEDED", payload) for timestamp_ms, payload in payloads.items()},
    )
    extractor.runtime_root = tmp_path

    batch = ChapterFrameSearcher(
        tmp_path,
        extractor,
        max_candidate_bytes=1,
        max_hash_distance_for_duplicate=0,
    ).search(
        media,
        (_chapter(),),
        {},
        scene_index,
        DocumentGenerationConfig(),
        is_cancel_requested=lambda: False,
    )

    assert batch.frame_sets[0].candidates == ()
    assert batch.chapter_status == (("chapter_001", "DEGRADED"),)
    assert batch.warnings == ("VISUAL_CANDIDATE_BUDGET_DEGRADED:chapter_001",)
    assert batch.metrics == {"visual_candidate_budget_degraded_chapters": 1}
    candidate_root = tmp_path / "runs/scope/run_001/visual/candidates"
    assert candidate_root.is_dir()
    assert not tuple(candidate_root.iterdir())


def test_phash_deduplication_preserves_best_candidate_for_each_target(tmp_path: Path) -> None:
    media, scene_index = _fixture(tmp_path)
    chapter = _chapter(
        semantic_targets=(_semantic_target("target_semantic", "asr_001"),),
    )
    timestamps = (2_000, 2_500, 3_500, 5_000, 7_500)
    extractor = _FakeExtractor(
        {
            timestamp_ms: (
                "SUCCEEDED",
                b"\xff\xd8\xff" + str(timestamp_ms).encode("ascii") + b"\xff\xd9",
            )
            for timestamp_ms in timestamps
        },
    )
    extractor.runtime_root = tmp_path

    batch = ChapterFrameSearcher(
        tmp_path,
        extractor,
        max_candidate_bytes=1024,
    ).search(
        media,
        (chapter,),
        {"asr_001": _speech()},
        scene_index,
        DocumentGenerationConfig(),
        is_cancel_requested=lambda: False,
    )

    retained_targets = {
        target_id
        for candidate in batch.frame_sets[0].candidates
        for target_id in candidate.target_ids
    }
    assert retained_targets == {"target_semantic", "target_base"}


def test_final_ranking_prefers_base_coverage_after_equal_semantic_hits_and_sharpness(
    tmp_path: Path,
) -> None:
    media, scene_index = _fixture(tmp_path)
    chapter = ChapterPlan(
        chapter_id="chapter_001",
        start_ms=0,
        end_ms=10_000,
        segment_refs=("segment_001",),
        title_hint="章节",
        visual_mode="COMPARISON",
        semantic_targets=(
            _semantic_target("target_semantic_1", "asr_001"),
            _semantic_target("target_semantic_2", "asr_002"),
        ),
        base_coverage_targets=(
            VisualSearchTarget(
                target_id="target_base",
                purpose="BASE_COVERAGE",
                query_zh="代表画面",
                sample_timestamps_ms=(6_500,),
            ),
        ),
    )
    semantic_only = b"\xff\xd8\xffsemantic-only\xff\xd9"
    semantic_with_base = b"\xff\xd8\xffsemantic-with-base\xff\xd9"

    def result_for_sample(sample: FrameSample) -> tuple[str, bytes | None, float, str]:
        status, payload = {
            3_500: ("SUCCEEDED", semantic_only),
            6_500: ("SUCCEEDED", semantic_with_base),
        }.get(sample.timestamp_ms, ("SUCCEEDED", semantic_only))
        return status, payload, 1.0, "0123456789abcdef"

    extractor = _FakeExtractor(
        {
            3_500: ("SUCCEEDED", semantic_only),
            6_500: ("SUCCEEDED", semantic_with_base),
        },
        result_for_sample=result_for_sample,
    )
    extractor.runtime_root = tmp_path

    batch = ChapterFrameSearcher(
        tmp_path,
        extractor,
        max_hash_distance_for_duplicate=0,
    ).search(
        media,
        (chapter,),
        {"asr_001": _speech(), "asr_002": _speech_2()},
        scene_index,
        DocumentGenerationConfig(),
        is_cancel_requested=lambda: False,
    )

    assert "target_base" in batch.frame_sets[0].candidates[0].target_ids


def test_final_ranking_uses_frame_id_after_timestamp_tie(tmp_path: Path) -> None:
    media, scene_index = _fixture(tmp_path)
    first_target = _semantic_target("target_semantic_1", "asr_001")
    second_target = _semantic_target("target_semantic_2", "asr_002")
    chapter = ChapterPlan(
        chapter_id="chapter_001",
        start_ms=0,
        end_ms=10_000,
        segment_refs=("segment_001",),
        title_hint="章节",
        visual_mode="COMPARISON",
        semantic_targets=(first_target, second_target),
        base_coverage_targets=(),
    )
    shared_timestamp_ms = 3_500
    first_sample_id = stable_identifier(
        "sample",
        {
            "chapter_id": chapter.chapter_id,
            "target_id": first_target.target_id,
            "timestamp_ms": shared_timestamp_ms,
        },
    )
    payload_by_sample_id = {
        first_sample_id: b"\xff\xd8\xffsort-0\xff\xd9",
    }

    def result_for_sample(sample: FrameSample) -> tuple[str, bytes | None, float, str]:
        if sample.timestamp_ms != shared_timestamp_ms:
            return "QUALITY_REJECTED", None, 1.0, "0123456789abcdef"
        return (
            "SUCCEEDED",
            payload_by_sample_id.get(sample.sample_id, b"\xff\xd8\xffsort-8\xff\xd9"),
            1.0,
            "0123456789abcdef",
        )

    extractor = _FakeExtractor({}, result_for_sample=result_for_sample)
    extractor.runtime_root = tmp_path

    batch = ChapterFrameSearcher(
        tmp_path,
        extractor,
        max_hash_distance_for_duplicate=0,
    ).search(
        media,
        (chapter,),
        {
            "asr_001": _speech(),
            "asr_002": _speech_at("asr_002", 3_000, 4_000),
        },
        scene_index,
        DocumentGenerationConfig(),
        is_cancel_requested=lambda: False,
    )

    frame_ids = tuple(candidate.frame_id for candidate in batch.frame_sets[0].candidates)
    assert frame_ids == tuple(sorted(frame_ids))


def test_search_rejects_scene_index_for_different_proxy(tmp_path: Path) -> None:
    media, scene_index = _fixture(tmp_path)
    mismatched = SceneIndex(
        proxy_sha256="b" * 64,
        duration_ms=scene_index.duration_ms,
        frame_tolerance_ms=scene_index.frame_tolerance_ms,
        scenes=scene_index.scenes,
        index_sha256=scene_index_sha256(
            proxy_sha256="b" * 64,
            duration_ms=scene_index.duration_ms,
            frame_tolerance_ms=scene_index.frame_tolerance_ms,
            scenes=scene_index.scenes,
        ),
    )
    extractor = _FakeExtractor({})
    extractor.runtime_root = tmp_path

    with pytest.raises(VideoDemoError) as raised:
        ChapterFrameSearcher(tmp_path, extractor).search(
            media,
            (_chapter(),),
            {},
            mismatched,
            DocumentGenerationConfig(),
            is_cancel_requested=lambda: False,
        )

    assert raised.value.code == ErrorCode.VIDEO_DIGEST_MISMATCH


def test_single_scene_sampling_plan_caps_unique_decode_points_at_three(tmp_path: Path) -> None:
    media, scene_index = _fixture(tmp_path)
    extractor = _FakeExtractor({})
    extractor.runtime_root = tmp_path
    chapter = _chapter(
        semantic_targets=(
            _semantic_target("target_semantic_1", "asr_001"),
            _semantic_target("target_semantic_2", "asr_002"),
        ),
    )

    ChapterFrameSearcher(tmp_path, extractor).search(
        media,
        (chapter,),
        {"asr_001": _speech(), "asr_002": _speech_2()},
        scene_index,
        DocumentGenerationConfig(),
        is_cancel_requested=lambda: False,
    )

    samples = extractor.calls[0][0]
    assert len({sample.timestamp_ms for sample in samples}) == 3
    assert {sample.sample_id for sample in samples}
    tiers_by_target = {
        sample.admission_tier
        for sample in samples
        if sample.admission_tier.endswith("PRIMARY")
    }
    assert tiers_by_target == {"SEMANTIC_PRIMARY", "BASE_PRIMARY"}


def test_base_coverage_plan_includes_remaining_scene_fallbacks(tmp_path: Path) -> None:
    media, original = _fixture(tmp_path)
    scenes = tuple(
        SceneBoundary(
            evidence_id=f"scene_{index:03d}",
            start_ms=index * 2_500,
            end_ms=(index + 1) * 2_500,
            transition="candidate",
            score=0.9,
        )
        for index in range(4)
    )
    scene_index = SceneIndex(
        proxy_sha256=media.proxy_sha256,
        duration_ms=original.duration_ms,
        frame_tolerance_ms=original.frame_tolerance_ms,
        scenes=scenes,
        index_sha256=scene_index_sha256(
            proxy_sha256=media.proxy_sha256,
            duration_ms=original.duration_ms,
            frame_tolerance_ms=original.frame_tolerance_ms,
            scenes=scenes,
        ),
    )
    chapter = ChapterPlan(
        chapter_id="chapter_001",
        start_ms=0,
        end_ms=10_000,
        segment_refs=("segment_001",),
        title_hint="章节",
        visual_mode="SINGLE",
        semantic_targets=(),
        base_coverage_targets=(
            VisualSearchTarget(
                target_id="target_base",
                purpose="BASE_COVERAGE",
                query_zh="代表画面",
                scene_refs=("scene_000", "scene_003"),
            ),
        ),
    )
    fallback_timestamp_ms = 3_750
    extractor = _FakeExtractor(
        {fallback_timestamp_ms: ("SUCCEEDED", b"\xff\xd8\xfffallback\xff\xd9")},
    )
    extractor.runtime_root = tmp_path

    batch = ChapterFrameSearcher(tmp_path, extractor).search(
        media,
        (chapter,),
        {},
        scene_index,
        DocumentGenerationConfig(),
        is_cancel_requested=lambda: False,
    )

    requested = {sample.timestamp_ms for sample in extractor.calls[0][0]}
    assert fallback_timestamp_ms in requested
    assert batch.chapter_status == ((chapter.chapter_id, "SUCCEEDED"),)
    assert batch.frame_sets[0].candidates[0].timestamp_ms == fallback_timestamp_ms


def test_candidate_budget_never_returns_partial_target_coverage(tmp_path: Path) -> None:
    media, scene_index = _fixture(tmp_path)
    chapter = ChapterPlan(
        chapter_id="chapter_001",
        start_ms=0,
        end_ms=10_000,
        segment_refs=("segment_001",),
        title_hint="章节",
        visual_mode="COMPARISON",
        semantic_targets=(
            _semantic_target("target_semantic_1", "asr_001"),
            _semantic_target("target_semantic_2", "asr_002"),
        ),
        base_coverage_targets=_chapter().base_coverage_targets,
    )
    payloads = {
        3_500: b"\xff\xd8\xffsemantic-one\xff\xd9",
        6_500: b"\xff\xd8\xffsemantic-two\xff\xd9",
    }
    extractor = _FakeExtractor(
        {timestamp_ms: ("SUCCEEDED", payload) for timestamp_ms, payload in payloads.items()},
    )
    extractor.runtime_root = tmp_path
    single_frame_budget = max(len(payload) for payload in payloads.values())

    batch = ChapterFrameSearcher(
        tmp_path,
        extractor,
        max_candidate_bytes=single_frame_budget,
        max_hash_distance_for_duplicate=0,
    ).search(
        media,
        (chapter,),
        {"asr_001": _speech(), "asr_002": _speech_2()},
        scene_index,
        DocumentGenerationConfig(),
        is_cancel_requested=lambda: False,
    )

    assert batch.frame_sets[0].candidates == ()
    assert batch.chapter_status == ((chapter.chapter_id, "DEGRADED"),)


def test_rejected_supplement_does_not_degrade_when_target_still_has_candidate(
    tmp_path: Path,
) -> None:
    media, scene_index = _fixture(tmp_path)
    chapter = _chapter()
    extractor = _FakeExtractor(
        {
            5_000: ("SUCCEEDED", _JPEG),
            2_500: ("SUCCEEDED", b"\xff\xd8\xffsupplement\xff\xd9"),
        },
    )
    extractor.runtime_root = tmp_path

    batch = ChapterFrameSearcher(
        tmp_path,
        extractor,
        max_candidate_bytes=len(_JPEG),
        max_hash_distance_for_duplicate=0,
    ).search(
        media,
        (chapter,),
        {},
        scene_index,
        DocumentGenerationConfig(),
        is_cancel_requested=lambda: False,
    )

    assert batch.chapter_status == ((chapter.chapter_id, "SUCCEEDED"),)
    assert batch.warnings == ()
    assert len(batch.frame_sets[0].candidates) == 1


def test_frame_id_uses_keyframe_prefix(tmp_path: Path) -> None:
    media, scene_index = _fixture(tmp_path)
    extractor = _FakeExtractor({5_000: ("SUCCEEDED", _JPEG)})
    extractor.runtime_root = tmp_path

    batch = ChapterFrameSearcher(tmp_path, extractor).search(
        media,
        (_chapter(),),
        {},
        scene_index,
        DocumentGenerationConfig(),
        is_cancel_requested=lambda: False,
    )

    assert batch.frame_sets[0].candidates[0].frame_id.startswith("keyframe_")


def test_complex_chapter_never_returns_partial_semantic_target_coverage(
    tmp_path: Path,
) -> None:
    media, scene_index = _fixture(tmp_path)
    chapter = ChapterPlan(
        chapter_id="chapter_001",
        start_ms=0,
        end_ms=10_000,
        segment_refs=("segment_001",),
        title_hint="章节",
        visual_mode="COMPARISON",
        semantic_targets=(
            _semantic_target("target_semantic_1", "asr_001"),
            _semantic_target("target_semantic_2", "asr_002"),
        ),
        base_coverage_targets=_chapter().base_coverage_targets,
    )
    extractor = _FakeExtractor(
        {3_500: ("SUCCEEDED", b"\xff\xd8\xffonly-first-target\xff\xd9")},
    )
    extractor.runtime_root = tmp_path

    batch = ChapterFrameSearcher(tmp_path, extractor).search(
        media,
        (chapter,),
        {"asr_001": _speech(), "asr_002": _speech_2()},
        scene_index,
        DocumentGenerationConfig(),
        is_cancel_requested=lambda: False,
    )

    assert batch.frame_sets[0].candidates == ()
    assert batch.chapter_status == ((chapter.chapter_id, "DEGRADED"),)
    assert "visual_collapsed_same_frame_chapters" not in batch.metrics


def test_search_rejects_constructed_scene_index_with_invalid_digest(tmp_path: Path) -> None:
    media, scene_index = _fixture(tmp_path)
    invalid = SceneIndex.model_construct(
        proxy_sha256=scene_index.proxy_sha256,
        duration_ms=scene_index.duration_ms,
        frame_tolerance_ms=scene_index.frame_tolerance_ms,
        scenes=scene_index.scenes,
        index_sha256="0" * 64,
    )
    extractor = _FakeExtractor({})
    extractor.runtime_root = tmp_path

    with pytest.raises(VideoDemoError) as raised:
        ChapterFrameSearcher(tmp_path, extractor).search(
            media,
            (_chapter(),),
            {},
            invalid,
            DocumentGenerationConfig(max_visuals_per_chapter=0),
            is_cancel_requested=lambda: False,
        )

    assert raised.value.code == ErrorCode.VISUAL_RESULT_INVALID


def test_search_rejects_success_candidate_outside_frame_tolerance(tmp_path: Path) -> None:
    media, scene_index = _fixture(tmp_path)

    class InvalidExtractor(_FakeExtractor):
        def extract_samples(
            self,
            _proxy: Path,
            run_relative_root: Path,
            samples: tuple[FrameSample, ...],
            *,
            is_cancel_requested: object,
            frame_tolerance_ms: int,
            artifact_session: CandidateArtifactSession | None = None,
        ) -> tuple[ExactFrameSampleResult, ...]:
            del is_cancel_requested, frame_tolerance_ms, artifact_session
            sample = samples[0]
            digest = hashlib.sha256(_JPEG).hexdigest()
            relative_path = run_relative_root / "visual/candidates" / f"{digest}.jpg"
            destination = tmp_path / relative_path
            destination.parent.mkdir(parents=True)
            destination.write_bytes(_JPEG)
            return (
                ExactFrameSampleResult(
                    sample_id=sample.sample_id,
                    requested_timestamp_ms=sample.timestamp_ms,
                    status="SUCCEEDED",
                    artifact_status="PUBLISHED",
                    candidate=FrameCandidate(
                        timestamp_ms=sample.timestamp_ms + 1_000,
                        sharpness=1.0,
                        black_ratio=0.0,
                        perceptual_hash="0123456789abcdef",
                        relative_path=relative_path,
                    ),
                ),
                *(
                    ExactFrameSampleResult(
                        sample_id=item.sample_id,
                        requested_timestamp_ms=item.timestamp_ms,
                        status="DECODE_FAILED",
                    )
                    for item in samples[1:]
                ),
            )

    extractor = InvalidExtractor({})
    extractor.runtime_root = tmp_path

    with pytest.raises(VideoDemoError) as raised:
        ChapterFrameSearcher(tmp_path, extractor).search(
            media,
            (_chapter(),),
            {},
            scene_index,
            DocumentGenerationConfig(),
            is_cancel_requested=lambda: False,
        )

    assert raised.value.code == ErrorCode.VISUAL_RESULT_INVALID


@pytest.mark.parametrize(
    ("status", "has_candidate", "artifact_status"),
    (
        ("DECODE_FAILED", False, "BUDGET_REJECTED"),
        ("SUCCEEDED", True, None),
    ),
)
def test_search_rejects_invalid_sample_result_state_combination(
    tmp_path: Path,
    status: str,
    has_candidate: bool,
    artifact_status: str | None,
) -> None:
    media, scene_index = _fixture(tmp_path)

    class InvalidStateExtractor(_FakeExtractor):
        def extract_samples(
            self,
            _proxy: Path,
            _run_relative_root: Path,
            samples: tuple[FrameSample, ...],
            *,
            is_cancel_requested: object,
            frame_tolerance_ms: int,
            artifact_session: CandidateArtifactSession | None = None,
        ) -> tuple[ExactFrameSampleResult, ...]:
            del is_cancel_requested, frame_tolerance_ms, artifact_session
            results: list[ExactFrameSampleResult] = []
            for index, sample in enumerate(samples):
                candidate = (
                    FrameCandidate(
                        timestamp_ms=sample.timestamp_ms,
                        sharpness=1.0,
                        black_ratio=0.0,
                        perceptual_hash="0123456789abcdef",
                        relative_path=Path("unused.jpg"),
                    )
                    if has_candidate and index == 0
                    else None
                )
                result = object.__new__(ExactFrameSampleResult)
                object.__setattr__(result, "sample_id", sample.sample_id)
                object.__setattr__(result, "requested_timestamp_ms", sample.timestamp_ms)
                object.__setattr__(result, "status", status if index == 0 else "QUALITY_REJECTED")
                object.__setattr__(result, "candidate", candidate)
                object.__setattr__(
                    result,
                    "artifact_status",
                    artifact_status if index == 0 else None,
                )
                results.append(result)
            return tuple(results)

    extractor = InvalidStateExtractor({})
    extractor.runtime_root = tmp_path

    with pytest.raises(VideoDemoError) as raised:
        ChapterFrameSearcher(tmp_path, extractor).search(
            media,
            (_chapter(),),
            {},
            scene_index,
            DocumentGenerationConfig(),
            is_cancel_requested=lambda: False,
        )

    assert raised.value.code == ErrorCode.VISUAL_RESULT_INVALID
    assert raised.value.message == "精确抽帧状态、候选和制品状态组合非法"


@pytest.mark.parametrize("perceptual_hash", ("1", "0123456789ABCDEf", "g" * 16))
def test_search_rejects_noncanonical_perceptual_hash(
    tmp_path: Path,
    perceptual_hash: str,
) -> None:
    media, scene_index = _fixture(tmp_path)

    class InvalidHashExtractor(_FakeExtractor):
        def extract_samples(
            self,
            _proxy: Path,
            run_relative_root: Path,
            samples: tuple[FrameSample, ...],
            *,
            is_cancel_requested: object,
            frame_tolerance_ms: int,
            artifact_session: CandidateArtifactSession | None = None,
        ) -> tuple[ExactFrameSampleResult, ...]:
            del is_cancel_requested, frame_tolerance_ms
            assert artifact_session is not None
            sample = samples[0]
            digest = hashlib.sha256(_JPEG).hexdigest()
            relative_path = run_relative_root / "visual/candidates" / f"{digest}.jpg"
            artifact_session.prepare_run(run_relative_root)
            artifact_session.publish_jpeg(relative_path, _JPEG, digest)
            return (
                ExactFrameSampleResult(
                    sample_id=sample.sample_id,
                    requested_timestamp_ms=sample.timestamp_ms,
                    status="SUCCEEDED",
                    artifact_status="PUBLISHED",
                    candidate=FrameCandidate(
                        timestamp_ms=sample.timestamp_ms,
                        sharpness=1.0,
                        black_ratio=0.0,
                        perceptual_hash=perceptual_hash,
                        relative_path=relative_path,
                    ),
                ),
                *(
                    ExactFrameSampleResult(
                        sample_id=item.sample_id,
                        requested_timestamp_ms=item.timestamp_ms,
                        status="QUALITY_REJECTED",
                    )
                    for item in samples[1:]
                ),
            )

    extractor = InvalidHashExtractor({})
    extractor.runtime_root = tmp_path

    with pytest.raises(VideoDemoError) as raised:
        ChapterFrameSearcher(tmp_path, extractor).search(
            media,
            (_chapter(),),
            {},
            scene_index,
            DocumentGenerationConfig(),
            is_cancel_requested=lambda: False,
        )

    assert raised.value.code == ErrorCode.VISUAL_RESULT_INVALID


def _fixture(tmp_path: Path) -> tuple[PreparedMedia, SceneIndex]:
    run_root = Path("runs/scope/run_001")
    proxy = tmp_path / run_root / "media/proxy.mp4"
    proxy.parent.mkdir(parents=True)
    proxy.write_bytes(b"proxy")
    registered = RegisteredAsset(
        source_path=tmp_path / run_root / "input/source.mp4",
        source_sha256=_ASSET_SHA256,
        object_ref="obj_001",
        source_size_bytes=5,
        source_mime="video/mp4",
        run_relative_root=run_root,
        config=PipelineRunConfig(),
    )
    manifest = VideoAssetManifest(
        object_ref="obj_001",
        source_sha256=_ASSET_SHA256,
        source_size_bytes=5,
        source_mime="video/mp4",
        duration_ms=10_000,
        video_stream=VideoStream(
            index=0,
            codec_name="h264",
            width=640,
            height=360,
            average_frame_rate=Rational(numerator=25, denominator=1),
        ),
        format_name="mp4",
        ffprobe_version="test",
    )
    media = PreparedMedia(
        source=ProbedAsset(registered, manifest, ProbeLimits()),
        proxy_path=proxy,
        proxy_sha256=hashlib.sha256(b"proxy").hexdigest(),
        proxy_size_bytes=5,
        audio_path=None,
        audio_sha256=None,
    )
    scenes = (
        SceneBoundary(
            evidence_id="scene_001",
            start_ms=0,
            end_ms=10_000,
            transition="candidate",
            score=0.9,
        ),
    )
    digest = scene_index_sha256(
        proxy_sha256=media.proxy_sha256,
        duration_ms=10_000,
        frame_tolerance_ms=40,
        scenes=scenes,
    )
    return media, SceneIndex(
        proxy_sha256=media.proxy_sha256,
        duration_ms=10_000,
        frame_tolerance_ms=40,
        scenes=scenes,
        index_sha256=digest,
    )


def _chapter(
    *,
    semantic_targets: tuple[VisualSearchTarget, ...] = (),
) -> ChapterPlan:
    return ChapterPlan(
        chapter_id="chapter_001",
        start_ms=0,
        end_ms=10_000,
        segment_refs=("segment_001",),
        title_hint="章节",
        visual_mode="SINGLE",
        semantic_targets=semantic_targets,
        base_coverage_targets=(
            VisualSearchTarget(
                target_id="target_base",
                purpose="BASE_COVERAGE",
                query_zh="代表画面",
                scene_refs=("scene_001",),
            ),
        ),
    )


def _semantic_target(target_id: str, evidence_id: str) -> VisualSearchTarget:
    return VisualSearchTarget(
        target_id=target_id,
        purpose="SEMANTIC",
        query_zh="展示的参数",
        anchor_evidence_refs=(evidence_id,),
    )


def _speech() -> SpeechSegment:
    return SpeechSegment(
        evidence_id="asr_001",
        start_ms=3_000,
        end_ms=4_000,
        text="展示参数",
        language="zh",
        confidence=0.9,
        is_fully_evaluated_language=True,
    )


def _speech_2() -> SpeechSegment:
    return _speech_at("asr_002", 6_000, 7_000)


def _speech_at(evidence_id: str, start_ms: int, end_ms: int) -> SpeechSegment:
    return SpeechSegment(
        evidence_id=evidence_id,
        start_ms=start_ms,
        end_ms=end_ms,
        text="展示第二个参数",
        language="zh",
        confidence=0.9,
        is_fully_evaluated_language=True,
    )
