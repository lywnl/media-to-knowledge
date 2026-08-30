from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from video_demo.domain.base import stable_identifier
from video_demo.domain.manifest import Rational, VideoAssetManifest, VideoStream
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.evaluation.annotations import (
    AuthorizationFile,
    EvaluationAnnotation,
    VerifiedAnnotation,
)
from video_demo.evaluation.chapter_vlm_input import (
    ChapterVlmInputFrame,
    ChapterVlmInputManifest,
    ChapterVlmInputPreparation,
    base_coverage_target_id,
    chapter_vlm_chapter_id,
    chapter_vlm_input_manifest_sha256,
    evaluation_run_id_for_input,
    prepare_chapter_vlm_input,
)
from video_demo.evaluation.dataset import EvaluationDataset, EvaluationSample
from video_demo.media.probe import ProbeResult
from video_demo.visual.keyframes import ExactFrameSampleResult, FrameCandidate


def _sha(seed: str) -> str:
    return hashlib.sha256(seed.encode()).hexdigest()


def _manifest() -> ChapterVlmInputManifest:
    source = _sha("source")
    image_a = _sha("image-a")
    image_b = _sha("image-b")
    evaluation_run_id = evaluation_run_id_for_input(
        "parent_run",
        "sample_001",
        source,
        _sha("annotation"),
        1280,
        90,
        ("reference_a", "reference_b"),
    )
    frame_a = ChapterVlmInputFrame(
        reference_frame_id="reference_a",
        frame_id=stable_identifier(
            "keyframe", {"asset_sha256": source, "timestamp_ms": 1000, "sha256": image_a}
        ),
        requested_timestamp_ms=1000,
        actual_timestamp_ms=1000,
        relative_path=f"visual/candidates/{image_a}.jpg",
        sha256=image_a,
        size_bytes=100,
        perceptual_hash="0123456789abcdef",
        target_ids=("target_001",),
    )
    frame_b = frame_a.model_copy(
        update={
            "reference_frame_id": "reference_b",
            "frame_id": stable_identifier(
                "keyframe", {"asset_sha256": source, "timestamp_ms": 2000, "sha256": image_b}
            ),
            "requested_timestamp_ms": 2000,
            "actual_timestamp_ms": 2000,
            "relative_path": f"visual/candidates/{image_b}.jpg",
            "sha256": image_b,
        }
    )
    target_id = stable_identifier(
        "visual_target",
        {
            "asset_sha256": source,
            "chapter_id": stable_identifier(
                "chapter",
                {
                    "asset_sha256": source,
                    "evaluation_run_id": evaluation_run_id,
                    "sample_id": "sample_001",
                    "requested_reference_frame_ids": ("reference_a", "reference_b"),
                },
            ),
            "purpose": "BASE_COVERAGE",
            "ordinal": 0,
            "target": {
                "query_zh": "识别并结构化提取这些画面中实际可见的文字、代码、表格、公式和界面状态"
            },
        },
    )
    frame_a = frame_a.model_copy(update={"target_ids": (target_id,)})
    frame_b = frame_b.model_copy(update={"target_ids": (target_id,)})
    return ChapterVlmInputManifest(
        schema_version="1.0.0",
        parent_evaluation_run_id="parent_run",
        evaluation_run_id=evaluation_run_id,
        sample_id="sample_001",
        source_media_sha256=source,
        source_duration_ms=10_000,
        annotation_sha256=_sha("annotation"),
        proxy_max_edge=1280,
        proxy_width=1280,
        proxy_height=720,
        proxy_frame_rate=Rational(numerator=30, denominator=1),
        proxy_is_variable_frame_rate=False,
        proxy_duration_ms=10_000,
        proxy_relative_path="media/source.mp4",
        duration_tolerance_ms=100,
        jpeg_quality=90,
        proxy_sha256=_sha("proxy"),
        proxy_size_bytes=1000,
        frame_tolerance_ms=34,
        requested_reference_frame_ids=("reference_a", "reference_b"),
        requested_image_sha256s=(image_a, image_b),
        retained_reference_frame_ids=("reference_a", "reference_b"),
        duplicate_frame_count=0,
        frames=(frame_a, frame_b),
    )


def test_manifest_binds_parent_run_and_recomputable_ids() -> None:
    manifest = _manifest()

    assert manifest.parent_evaluation_run_id == "parent_run"
    assert chapter_vlm_chapter_id(manifest).startswith("chapter_")
    assert base_coverage_target_id(manifest).startswith("visual_target_")
    assert chapter_vlm_input_manifest_sha256(manifest) == chapter_vlm_input_manifest_sha256(
        manifest.model_copy(deep=True)
    )


def test_manifest_rejects_non_program_target_binding() -> None:
    manifest = _manifest()
    with pytest.raises(ValueError, match="BASE_COVERAGE"):
        ChapterVlmInputManifest.model_validate(
            manifest.model_dump(
                mode="python",
                exclude_computed_fields=True,
            )
            | {
                "frames": (
                    manifest.frames[0]
                    .model_copy(update={"target_ids": ("manual_target",)})
                    .model_dump(mode="python"),
                    manifest.frames[1].model_dump(mode="python"),
                )
            }
        )


def test_preparation_status_requires_execution_shape() -> None:
    with pytest.raises(ValueError, match="NOT_RUN"):
        ChapterVlmInputPreparation(
            status="NOT_RUN",
            execution_started=True,
            error_code="LIVE_AUTHORIZED_CHAPTER_FRAMES_UNAVAILABLE",
        )


def _evaluation_package(tmp_path: Path, frame_count: int = 5) -> tuple[object, Path]:
    runtime = tmp_path / ".codex" / "video-rag-demo"
    eval_root = runtime / "eval"
    media_path = eval_root / "media" / "sample.mp4"
    media_path.parent.mkdir(parents=True)
    media_path.write_bytes(b"source-media")
    media_sha = hashlib.sha256(media_path.read_bytes()).hexdigest()
    visual_frames = tuple(
        {
            "frame_id": f"reference_{index:02d}",
            "timestamp_ms": index * 1_000,
            "text_lines": [f"文本 {index}"],
        }
        for index in range(frame_count)
    )
    annotation = EvaluationAnnotation.model_validate(
        {
            "schema_version": "2.0.0",
            "sample_id": "sample_001",
            "media_sha256": media_sha,
            "duration_ms": 10_000,
            "language": "zh",
            "reference_text": "参考文本",
            "visual_frames": visual_frames,
            "scene_boundaries_ms": [5_000],
            "semantic_boundaries_ms": [5_000],
            "supported_facts": [{"fact_id": "fact_001", "canonical_text": "事实"}],
            "key_fact_ids": ["fact_001"],
        }
    )
    annotation_sha = hashlib.sha256(b"annotation").hexdigest()
    sample = EvaluationSample(
        sample_id="sample_001",
        language="zh",
        authorization_id="auth_001",
        media_relative_path="media/sample.mp4",
        media_sha256=media_sha,
        annotations_relative_path="annotations/sample.json",
        annotations_sha256=annotation_sha,
    )
    package = EvaluationDataset(
        samples=(sample,),
        eval_root=eval_root,
        runtime_root=runtime,
        workspace_root=tmp_path,
    )
    authorization = AuthorizationFile.model_validate(
        {
            "schema_version": "1.0.0",
            "records": [
                {
                    "schema_version": "1.0.0",
                    "authorization_id": "auth_001",
                    "source_category": "OWNED",
                    "allowed_purposes": ["VIDEO_QUALITY_EVALUATION"],
                    "confirmed_at": "2026-08-18T00:00:00Z",
                    "media_sha256": [media_sha],
                }
            ],
        }
    )
    from video_demo.evaluation.annotations import ValidatedEvaluationPackage

    return (
        ValidatedEvaluationPackage(
            dataset=package,
            authorization=authorization,
            annotations=(VerifiedAnnotation(annotation=annotation, sha256=annotation_sha),),
            dataset_sha256="b" * 64,
            authorization_sha256="c" * 64,
        ),
        media_path,
    )


def _video_manifest(
    *,
    sha256: str,
    duration_ms: int = 10_000,
    width: int = 640,
    height: int = 360,
    frame_rate: Rational | None = None,
) -> VideoAssetManifest:
    actual_frame_rate = frame_rate or Rational(numerator=25, denominator=1)
    return VideoAssetManifest(
        object_ref="sample_001",
        source_sha256=sha256,
        source_size_bytes=1,
        source_mime="video/mp4",
        duration_ms=duration_ms,
        video_stream=VideoStream(
            index=0,
            codec_name="h264",
            width=width,
            height=height,
            average_frame_rate=actual_frame_rate,
        ),
        format_name="mov,mp4",
        ffprobe_version="ffprobe-test",
    )


class _PreparationProbe:
    def __init__(self, source_manifest: VideoAssetManifest, proxy_manifest: VideoAssetManifest):
        self.source_manifest = source_manifest
        self.proxy_manifest = proxy_manifest
        self.calls: list[Path] = []

    def probe(self, path: Path, **_kwargs: object) -> ProbeResult:
        self.calls.append(path)
        return ProbeResult(
            manifest=self.proxy_manifest if path.name == "source.mp4" else self.source_manifest,
            warnings=(),
        )


class _PreparationExtractor:
    def __init__(self, runtime: Path, payloads: tuple[bytes, ...]):
        self.runtime = runtime
        self.payloads = payloads
        self.calls: list[tuple[object, ...]] = []

    def extract_samples(
        self,
        _proxy: Path,
        run_relative_root: Path,
        samples: tuple[object, ...],
        *,
        is_cancel_requested: object,
        frame_tolerance_ms: int,
        artifact_session: object,
    ) -> tuple[ExactFrameSampleResult, ...]:
        del is_cancel_requested, frame_tolerance_ms
        self.calls.append(tuple(samples))
        results: list[ExactFrameSampleResult] = []
        for index, sample in enumerate(samples):
            payload = self.payloads[index]
            digest = hashlib.sha256(payload).hexdigest()
            relative = run_relative_root / "visual" / "candidates" / f"{digest}.jpg"
            publication = artifact_session.publish_jpeg(relative, payload, digest)
            if publication.status == "BUDGET_REJECTED":
                results.append(
                    ExactFrameSampleResult(
                        sample_id=sample.sample_id,
                        requested_timestamp_ms=sample.timestamp_ms,
                        status="SUCCEEDED",
                        artifact_status="BUDGET_REJECTED",
                    )
                )
                continue
            candidate = FrameCandidate(
                timestamp_ms=sample.timestamp_ms,
                sharpness=1.0,
                black_ratio=0.0,
                perceptual_hash="0123456789abcdef",
                relative_path=relative,
                created_by_call=publication.created_by_call,
            )
            results.append(
                ExactFrameSampleResult(
                    sample_id=sample.sample_id,
                    requested_timestamp_ms=sample.timestamp_ms,
                    status="SUCCEEDED",
                    candidate=candidate,
                    artifact_status=publication.status,
                )
            )
        return tuple(results)


def _prepare(
    tmp_path: Path,
    *,
    frame_count: int,
    payloads: tuple[bytes, ...] | None = None,
    max_candidate_frame_bytes_per_run: int = 1024 * 1024,
) -> tuple[object, object, _PreparationProbe, object, _PreparationExtractor]:
    package, _media_path = _evaluation_package(tmp_path, frame_count)
    runtime = tmp_path / ".codex" / "video-rag-demo"
    source = package.dataset.samples[0]
    media_sha = source.media_sha256
    source_manifest = _video_manifest(sha256=media_sha)
    probe = _PreparationProbe(source_manifest, source_manifest)
    transcoder = object()
    extracted_payloads = payloads or tuple(
        b"\xff\xd8\xff" + bytes([index]) + b"\xff\xd9" for index in range(4)
    )
    extractor = _PreparationExtractor(runtime, extracted_payloads)
    return package, runtime, probe, transcoder, extractor


def test_preparation_selects_four_evenly_from_five_reference_frames(
    tmp_path: Path,
) -> None:
    package, runtime, probe, _transcoder, extractor = _prepare(tmp_path, frame_count=5)

    result = prepare_chapter_vlm_input(
        package,
        parent_evaluation_run_id="parent_001",
        proxy_max_edge=1280,
        jpeg_quality=90,
        max_video_bytes=1024 * 1024,
        vlm_max_image_bytes=1024,
        max_candidate_frame_bytes_per_run=1024 * 1024,
        max_candidate_frame_files_per_run=10,
        ffprobe=probe,
        frame_extractor=extractor,
        runtime_root=runtime,
    )

    assert result.status == "READY"
    assert result.manifest is not None
    assert result.manifest.requested_reference_frame_ids == (
        "reference_00",
        "reference_01",
        "reference_03",
        "reference_04",
    )
    assert len(extractor.calls) == 1
    assert len(probe.calls) == 2


def test_preparation_rejects_budget_without_leaving_proxy_or_candidates(
    tmp_path: Path,
) -> None:
    package, runtime, probe, _transcoder, extractor = _prepare(
        tmp_path,
        frame_count=4,
        payloads=(b"\xff\xd8\xfflarge\xff\xd9",) * 4,
        max_candidate_frame_bytes_per_run=5,
    )

    result = prepare_chapter_vlm_input(
        package,
        parent_evaluation_run_id="parent_001",
        proxy_max_edge=1280,
        jpeg_quality=90,
        max_video_bytes=1024 * 1024,
        vlm_max_image_bytes=1024,
        max_candidate_frame_bytes_per_run=5,
        max_candidate_frame_files_per_run=10,
        ffprobe=probe,
        frame_extractor=extractor,
        runtime_root=runtime,
    )

    assert result.status == "FAIL"
    assert result.error_code == "INPUT_BUDGET_EXCEEDED"
    run_root = runtime / "runs" / "evaluation"
    assert not list(run_root.rglob("source.mp4"))
    assert not list(run_root.rglob("*.jpg"))


def test_preparation_maps_unlisted_dependency_error_to_stable_failure_code(
    tmp_path: Path,
) -> None:
    package, runtime, _probe, _transcoder, extractor = _prepare(tmp_path, frame_count=2)

    class FailedProbe:
        def probe(self, *_args: object, **_kwargs: object) -> object:
            raise VideoDemoError(ErrorCode.VIDEO_PROCESS_FAILED, "ffprobe failed")

    result = prepare_chapter_vlm_input(
        package,
        parent_evaluation_run_id="parent_001",
        proxy_max_edge=1280,
        jpeg_quality=90,
        max_video_bytes=1024 * 1024,
        vlm_max_image_bytes=1024,
        max_candidate_frame_bytes_per_run=1024 * 1024,
        max_candidate_frame_files_per_run=10,
        ffprobe=FailedProbe(),  # type: ignore[arg-type]
        frame_extractor=extractor,
        runtime_root=runtime,
    )

    assert result.status == "FAIL"
    assert result.execution_started is True
    assert result.error_code == ErrorCode.ARTIFACT_SCHEMA_INVALID
