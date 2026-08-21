from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path

import pytest

from video_demo.application.pipeline import PipelineContext, RegisteredAsset
from video_demo.application.production_media import (
    ProductionAssetProbe,
    ProductionAssetRegistrar,
    ProductionMediaTranscoder,
)
from video_demo.application.runs import RunService
from video_demo.application.uploads import UploadService
from video_demo.domain.manifest import AudioStream, Rational, VideoAssetManifest, VideoStream
from video_demo.errors import VideoDemoError
from video_demo.media.probe import ProbeResult
from video_demo.media.transcode import AudioArtifact, NoAudioArtifact, ProxyVideoArtifact
from video_demo.persistence.database import Database
from video_demo.persistence.repositories import Scope
from video_demo.storage.object_store import LocalVideoObjectStore

_MP4 = b"\x00\x00\x00\x18ftypisom\x00\x00\x02\x00isomiso2"


def test_production_registrar_materializes_scoped_input_and_carries_run_config(
    tmp_path: Path,
) -> None:
    runtime_root = tmp_path / ".codex" / "video-rag-demo"
    runtime_root.mkdir(parents=True)
    database = Database(f"sqlite+pysqlite:///{runtime_root / 'video-demo.db'}")
    database.create_schema()
    store = LocalVideoObjectStore(runtime_root, max_video_bytes=1024)
    scope = Scope("tenant-a", "app-a", "kb-a")
    content = b"\x00\x00\x00\x18ftypisomvideo"
    uploaded = UploadService(database, store).upload(
        BytesIO(content),
        "lesson.mp4",
        "video/mp4",
        scope,
    )
    run = RunService(database).create(
        scope=scope,
        object_ref=uploaded.object_ref,
        idempotency_key="production-media-0001",
        language_hints=("zh", "en"),
        min_speakers=1,
        max_speakers=3,
    )

    registered = ProductionAssetRegistrar(database, store).register(
        PipelineContext(run_id=run.run_id, scope=scope),
    )

    expected_root = Path("runs") / store.scope_key(scope) / run.run_id
    assert registered.source_path == runtime_root / expected_root / "input/source.mp4"
    assert registered.source_path.read_bytes() == content
    assert registered.source_sha256 == hashlib.sha256(content).hexdigest()
    assert registered.object_ref == uploaded.object_ref
    assert registered.source_size_bytes == len(content)
    assert registered.source_mime == "video/mp4"
    assert registered.run_relative_root == expected_root
    assert registered.config.language_hints == ("zh", "en")
    assert registered.config.min_speakers == 1
    assert registered.config.max_speakers == 3


def test_production_probe_uses_registered_metadata_and_preserves_manifest_warnings(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    registered = _registered(source)
    manifest = _manifest(has_audio=False)
    calls: list[dict[str, object]] = []

    class ProbeClient:
        def probe(self, path: Path, **kwargs: object) -> ProbeResult:
            calls.append({"path": path, **kwargs})
            return ProbeResult(manifest=manifest, warnings=("NO_AUDIO_TRACK",))

    probed = ProductionAssetProbe(lambda: ProbeClient()).probe(registered)

    assert calls == [
        {
            "path": source,
            "object_ref": "obj_001",
            "source_sha256": "a" * 64,
            "source_size_bytes": 6,
            "source_mime": "video/mp4",
            "limits": probed.limits,
        },
    ]
    assert probed.asset is registered
    assert probed.manifest == manifest
    assert probed.warnings == ("NO_AUDIO_TRACK",)
    assert probed.duration_ms == 1_000


def test_production_transcoder_generates_scoped_proxy_and_explicit_no_audio_warning(
    tmp_path: Path,
) -> None:
    runtime_root = tmp_path / ".codex" / "video-rag-demo"
    runtime_root.mkdir(parents=True)
    source = runtime_root / "runs/scope/run_001/input/source.mp4"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"source")
    registered = _registered(
        source,
        run_relative_root=Path("runs/scope/run_001"),
    )
    probed = ProductionAssetProbe(
        lambda: _StaticProbeClient(_manifest(has_audio=False)),
    ).probe(registered)

    class Transcoder:
        def create_proxy(self, path: Path, run_relative_root: Path) -> ProxyVideoArtifact:
            assert path == source
            assert run_relative_root == Path("runs/scope/run_001")
            relative_path = run_relative_root / "media/proxy.mp4"
            destination = runtime_root / relative_path
            destination.parent.mkdir(parents=True)
            destination.write_bytes(_MP4)
            return ProxyVideoArtifact(
                relative_path=relative_path.as_posix(),
                sha256=hashlib.sha256(_MP4).hexdigest(),
                size_bytes=len(_MP4),
                max_edge=1280,
                normalized_start_ms=0,
            )

        def extract_audio(
            self,
            path: Path,
            run_relative_root: Path,
            *,
            has_audio: bool,
        ) -> AudioArtifact | NoAudioArtifact:
            assert path == source
            assert run_relative_root == Path("runs/scope/run_001")
            assert has_audio is False
            return NoAudioArtifact(warning_code="NO_AUDIO_TRACK")

    prepared = ProductionMediaTranscoder(
        runtime_root,
        lambda _is_cancel_requested: Transcoder(),
    ).transcode(probed)

    assert prepared.proxy_path == runtime_root / "runs/scope/run_001/media/proxy.mp4"
    assert prepared.proxy_sha256 == hashlib.sha256(_MP4).hexdigest()
    assert prepared.proxy_size_bytes == len(_MP4)
    assert prepared.audio_path is None
    assert prepared.audio_sha256 is None
    assert prepared.warnings == ("NO_AUDIO_TRACK",)


@pytest.mark.parametrize("mutation", ["empty", "fake_mp4", "wrong_size", "too_large"])
def test_production_transcoder_rejects_invalid_proxy_before_prepared_media(
    tmp_path: Path,
    mutation: str,
) -> None:
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    source = runtime_root / "runs/scope/run_001/input/source.mp4"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"source")
    registered = _registered(source)
    probed = ProductionAssetProbe(
        lambda: _StaticProbeClient(_manifest(has_audio=False)),
    ).probe(registered)
    payload = b"" if mutation == "empty" else (b"not-mp4" if mutation == "fake_mp4" else _MP4)

    class Transcoder:
        def create_proxy(self, _path: Path, root: Path) -> ProxyVideoArtifact:
            relative = root / "media/proxy.mp4"
            output = runtime_root / relative
            output.parent.mkdir(parents=True)
            output.write_bytes(payload)
            declared_size = len(payload) + 1 if mutation == "wrong_size" else len(payload)
            if mutation == "too_large":
                declared_size = 2_000
            return ProxyVideoArtifact(
                relative_path=relative.as_posix(),
                sha256=hashlib.sha256(payload).hexdigest(),
                size_bytes=declared_size,
                max_edge=1280,
                normalized_start_ms=0,
            )

        def extract_audio(self, *_args: object, **_kwargs: object) -> NoAudioArtifact:
            raise AssertionError("非法 proxy 不得继续处理音频")

    transcoder = ProductionMediaTranscoder(
        runtime_root,
        lambda _cancel: Transcoder(),
        max_proxy_bytes=1_000,
    )
    with pytest.raises(VideoDemoError):
        transcoder.transcode(probed)


class _StaticProbeClient:
    def __init__(self, manifest: VideoAssetManifest) -> None:
        self._manifest = manifest

    def probe(self, _path: Path, **_kwargs: object) -> ProbeResult:
        warnings = () if self._manifest.audio_streams else ("NO_AUDIO_TRACK",)
        return ProbeResult(manifest=self._manifest, warnings=warnings)


def _registered(
    source: Path,
    *,
    run_relative_root: Path = Path("runs/scope/run_001"),
) -> RegisteredAsset:
    from video_demo.application.pipeline import PipelineRunConfig

    return RegisteredAsset(
        source_path=source,
        source_sha256="a" * 64,
        object_ref="obj_001",
        source_size_bytes=6,
        source_mime="video/mp4",
        run_relative_root=run_relative_root,
        config=PipelineRunConfig(language_hints=("en",)),
    )


def _manifest(*, has_audio: bool) -> VideoAssetManifest:
    return VideoAssetManifest(
        object_ref="obj_001",
        source_sha256="a" * 64,
        source_size_bytes=6,
        source_mime="video/mp4",
        duration_ms=1_000,
        video_stream=VideoStream(
            index=0,
            codec_name="h264",
            width=640,
            height=360,
            average_frame_rate=Rational(numerator=25, denominator=1),
        ),
        audio_streams=(
            AudioStream(index=1, codec_name="aac", sample_rate_hz=48_000, channels=2),
        )
        if has_audio
        else (),
        format_name="mov,mp4",
        ffprobe_version="ffprobe test",
    )
