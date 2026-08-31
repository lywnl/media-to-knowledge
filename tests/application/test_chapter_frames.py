from __future__ import annotations

import hashlib
from pathlib import Path

from video_demo.application.chapter_frames import ChapterFrameSearcher
from video_demo.application.pipeline_contracts import (
    PipelineRunConfig,
    PreparedMedia,
    ProbedAsset,
    RegisteredAsset,
)
from video_demo.domain.document import DocumentGenerationConfig
from video_demo.domain.document_plan import ChapterPlan, VisualSearchTarget
from video_demo.domain.evidence import SpeechSegment
from video_demo.domain.manifest import Rational, VideoAssetManifest, VideoStream
from video_demo.media.probe import ProbeLimits
from video_demo.visual.candidate_artifacts import CandidateArtifactSession
from video_demo.visual.ffmpeg_frames import ExactFrameSampleResult, FrameCandidate, FrameSample

_ASSET_SHA256 = "a" * 64
_JPEG = b"\xff\xd8\xffframe\xff\xd9"


class _Extractor:
    def __init__(self, failures: set[int] | None = None) -> None:
        self.failures = failures or set()
        self.samples: tuple[FrameSample, ...] = ()

    def extract_samples(
        self,
        source: Path,
        run_relative_root: Path,
        samples: tuple[FrameSample, ...],
        *,
        is_cancel_requested: object,
        artifact_session: CandidateArtifactSession,
    ):
        del source, is_cancel_requested
        self.samples = samples
        artifact_session.prepare_run(run_relative_root)
        results: list[ExactFrameSampleResult] = []
        for sample in samples:
            if sample.timestamp_ms in self.failures:
                results.append(
                    ExactFrameSampleResult(sample.sample_id, sample.timestamp_ms, "DECODE_FAILED")
                )
                continue
            payload = b"\xff\xd8\xff" + str(sample.timestamp_ms).encode() + b"\xff\xd9"
            digest = hashlib.sha256(payload).hexdigest()
            relative = run_relative_root / "visual" / "candidates" / f"{digest}.jpg"
            publication = artifact_session.publish_jpeg(relative, payload, digest)
            candidate = FrameCandidate(
                timestamp_ms=sample.timestamp_ms,
                relative_path=Path("visual/candidates") / f"{digest}.jpg",
                sha256=digest,
                size_bytes=len(payload),
                created_by_call=publication.created_by_call,
            )
            results.append(
                ExactFrameSampleResult(
                    sample.sample_id,
                    sample.timestamp_ms,
                    "SUCCEEDED",
                    candidate=candidate,
                    artifact_status=publication.status,
                )
            )
        return tuple(results)


def _media(tmp_path: Path) -> PreparedMedia:
    run_root = Path("runs/scope/run_001")
    source = tmp_path / run_root / "input/source.mp4"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"video")
    registered = RegisteredAsset(
        source, _ASSET_SHA256, "obj_001", 5, "video/mp4", run_root, PipelineRunConfig()
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
    return PreparedMedia(
        ProbedAsset(registered, manifest, ProbeLimits()), source, _ASSET_SHA256, 5, None, None
    )


def _chapter(*, semantic_targets: tuple[VisualSearchTarget, ...] = ()) -> ChapterPlan:
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
                sample_timestamps_ms=(5_000,),
            ),
        ),
    )


def test_base_primary_uses_chapter_midpoint(tmp_path: Path) -> None:
    extractor = _Extractor()
    batch = ChapterFrameSearcher(tmp_path, extractor).search(
        _media(tmp_path),
        (_chapter(),),
        {},
        DocumentGenerationConfig(),
        is_cancel_requested=lambda: False,
    )
    assert [sample.timestamp_ms for sample in extractor.samples] == [5_000]
    assert batch.chapter_status == (("chapter_001", "SUCCEEDED"),)
    assert batch.frame_sets[0].candidates[0].target_ids == ("target_base",)


def test_single_frame_failure_degrades_chapter_but_keeps_successful_frames(tmp_path: Path) -> None:
    chapter = _chapter(
        semantic_targets=(
            VisualSearchTarget(
                target_id="semantic",
                purpose="SEMANTIC",
                query_zh="语义画面",
                anchor_evidence_refs=("asr_001",),
            ),
        )
    )
    extractor = _Extractor({5_000})
    speech = SpeechSegment(
        evidence_id="asr_001",
        start_ms=2_000,
        end_ms=3_000,
        text="语音",
        language="zh",
        confidence=0.9,
        is_fully_evaluated_language=True,
    )
    batch = ChapterFrameSearcher(tmp_path, extractor).search(
        _media(tmp_path),
        (chapter,),
        {"asr_001": speech},
        DocumentGenerationConfig(),
        is_cancel_requested=lambda: False,
    )
    assert batch.chapter_status == (("chapter_001", "DEGRADED"),)
    assert batch.status == "PARTIAL_SUCCEEDED"
    assert batch.frame_sets[0].candidates


def test_disabled_visual_search_does_not_call_extractor(tmp_path: Path) -> None:
    extractor = _Extractor()
    batch = ChapterFrameSearcher(tmp_path, extractor).search(
        _media(tmp_path),
        (_chapter(),),
        {},
        DocumentGenerationConfig(max_visuals_per_chapter=0),
        is_cancel_requested=lambda: False,
    )
    assert extractor.samples == ()
    assert batch.chapter_status == (("chapter_001", "DISABLED"),)
