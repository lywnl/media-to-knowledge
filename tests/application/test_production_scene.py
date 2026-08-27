from __future__ import annotations

import hashlib
from pathlib import Path

from video_demo.application.pipeline_contracts import (
    EvidencePreparationLimits,
    PipelineRunConfig,
    PreparedMedia,
    ProbedAsset,
    RegisteredAsset,
)
from video_demo.application.production_scene import ProductionSceneIndexProvider
from video_demo.domain.evidence import SceneBoundary
from video_demo.domain.manifest import Rational, VideoAssetManifest, VideoStream
from video_demo.implementation import implementation_import_closure
from video_demo.media.probe import ProbeLimits

_MP4 = b"\x00\x00\x00\x18ftypisom\x00\x02\x00\x00isomiso2"


def test_scene_index_provider_only_detects_and_normalizes_scenes(tmp_path: Path) -> None:
    media = _media(tmp_path, duration_ms=4_000)
    calls: list[tuple[Path, int, str, int]] = []

    class Detector:
        def detect(
            self,
            proxy: Path,
            *,
            duration_ms: int,
            source_sha256: str,
            frame_tolerance_ms: int,
        ) -> tuple[SceneBoundary, ...]:
            calls.append((proxy, duration_ms, source_sha256, frame_tolerance_ms))
            return (
                _scene(0, 500, transition="candidate"),
                _scene(500, 1_300, transition="hard_cut"),
                _scene(1_300, 3_300, transition="hard_cut"),
                _scene(3_300, 4_000, transition="hard_cut"),
            )

    provider = ProductionSceneIndexProvider(tmp_path, Detector())
    index = provider.prepare_scene_index(
        media,
        limits=_evidence_limits(scene_boundaries=2),
    )

    assert calls == [(media.proxy_path, 4_000, media.proxy_sha256, 40)]
    assert [(item.start_ms, item.end_ms) for item in index.scenes] == [
        (0, 1_300),
        (1_300, 4_000),
    ]
    assert not list((tmp_path / media.source.asset.run_relative_root).rglob("*.jpg"))


def test_scene_index_provider_honors_cancellation_before_detector(tmp_path: Path) -> None:
    media = _media(tmp_path, duration_ms=1_000)

    class Detector:
        def detect(self, *_args: object, **_kwargs: object) -> tuple[SceneBoundary, ...]:
            raise AssertionError("取消后不得调用场景检测")

    provider = ProductionSceneIndexProvider(tmp_path, Detector())

    try:
        provider.prepare_scene_index(
            media,
            limits=_evidence_limits(),
            is_cancel_requested=lambda: True,
        )
    except Exception as error:
        assert getattr(error, "code", None) == "JOB_CANCELLED"
    else:
        raise AssertionError("取消必须失败关闭")


def test_production_scene_import_closure_excludes_legacy_visual_chain() -> None:
    workspace_root = Path(__file__).parents[2]
    forbidden = {
        Path("src/video_demo/application/production_visual.py"),
        Path("src/video_demo/domain/legacy_result.py"),
        Path("src/video_demo/fusion/merge.py"),
        Path("src/video_demo/fusion/result_builder.py"),
        Path("src/video_demo/fusion/retrieval_text.py"),
    }

    closure = set(
        implementation_import_closure(
            workspace_root,
            (Path("src/video_demo/application/production_scene.py"),),
            extra_files=(),
        )
    )

    assert closure.isdisjoint(forbidden), sorted(
        path.as_posix() for path in closure & forbidden
    )


def _scene(start_ms: int, end_ms: int, *, transition: str) -> SceneBoundary:
    return SceneBoundary(
        evidence_id=f"scene_{start_ms}_{end_ms}",
        start_ms=start_ms,
        end_ms=end_ms,
        transition=transition,  # type: ignore[arg-type]
        score=0.9,
    )


def _evidence_limits(*, scene_boundaries: int = 20_000) -> EvidencePreparationLimits:
    return EvidencePreparationLimits(
        max_transcript_evidence_items=20_000,
        max_transcript_chars=2_000_000,
        max_scene_boundaries=scene_boundaries,
        max_base_segments=20_000,
    )


def _media(tmp_path: Path, *, duration_ms: int) -> PreparedMedia:
    run_root = Path("runs/scope/run_001")
    source_path = tmp_path / run_root / "input/source.mp4"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(b"source")
    registered = RegisteredAsset(
        source_path=source_path,
        source_sha256=hashlib.sha256(b"source").hexdigest(),
        object_ref="obj_001",
        source_size_bytes=6,
        source_mime="video/mp4",
        run_relative_root=run_root,
        config=PipelineRunConfig(),
    )
    manifest = VideoAssetManifest(
        object_ref="obj_001",
        source_sha256=registered.source_sha256,
        source_size_bytes=6,
        source_mime="video/mp4",
        duration_ms=duration_ms,
        video_stream=VideoStream(
            index=0,
            codec_name="h264",
            width=640,
            height=360,
            average_frame_rate=Rational(numerator=25, denominator=1),
        ),
        format_name="mov,mp4",
        ffprobe_version="test",
    )
    proxy = tmp_path / run_root / "media/proxy.mp4"
    proxy.parent.mkdir(parents=True, exist_ok=True)
    proxy.write_bytes(_MP4)
    return PreparedMedia(
        source=ProbedAsset(registered, manifest, ProbeLimits()),
        proxy_path=proxy,
        proxy_sha256=hashlib.sha256(_MP4).hexdigest(),
        proxy_size_bytes=len(_MP4),
        audio_path=None,
        audio_sha256=None,
    )
