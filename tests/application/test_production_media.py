from __future__ import annotations

import hashlib
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from typing import Literal

import pytest

from video_demo.application.pipeline import PipelineContext, RegisteredAsset
from video_demo.application.production_media import (
    ProductionAssetProbe,
    ProductionAssetRegistrar,
    ProductionMediaTranscoder,
)
from video_demo.application.runs import RunService
from video_demo.application.uploads import UploadService
from video_demo.domain.manifest import (
    AudioStream,
    Rational,
    SubtitleStream,
    VideoAssetManifest,
    VideoStream,
)
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.media.probe import ProbeResult
from video_demo.media.transcode import (
    AudioArtifact,
    NoAudioArtifact,
    ProxyVideoArtifact,
    SubtitleArtifact,
)
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
        speech_enrichment_mode="full",
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
    assert registered.config.speech_enrichment_mode == "full"


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


def test_production_probe_propagates_video_timeline_without_changing_container_duration(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    registered = _registered(source)
    manifest = _manifest(has_audio=False)

    class ProbeClient:
        def probe(self, _path: Path, **_kwargs: object) -> ProbeResult:
            return ProbeResult(
                manifest=manifest,
                warnings=("NO_AUDIO_TRACK",),
                timeline_duration_ms=900,
            )

    probed = ProductionAssetProbe(lambda: ProbeClient()).probe(registered)

    assert probed.duration_ms == 900
    assert probed.manifest.duration_ms == 1_000


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
            duration_ms: int,
        ) -> AudioArtifact | NoAudioArtifact:
            assert path == source
            assert run_relative_root == Path("runs/scope/run_001")
            assert has_audio is False
            assert duration_ms == 1_000
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
    assert prepared.subtitle is None
    assert prepared.warnings == ("NO_AUDIO_TRACK",)


def test_complete_text_subtitle_is_selected_before_audio_extraction(tmp_path: Path) -> None:
    runtime_root, probed = _probed_media(
        tmp_path,
        has_audio=True,
        subtitle_streams=(SubtitleStream(index=2, codec_name="mov_text", language="zh"),),
        duration_ms=30_000,
        language_hints=("zh",),
        speech_enrichment_mode="full",
    )
    client = _RecordingTranscoder(runtime_root, subtitle_payloads={2: _complete_vtt()})

    prepared = ProductionMediaTranscoder(
        runtime_root,
        lambda _cancel: client,
    ).transcode(probed)

    assert client.extract_subtitle_calls == [2]
    assert client.extract_audio_calls == []
    assert prepared.subtitle is not None
    assert prepared.subtitle.artifact.stream_index == 2
    assert prepared.audio_path is None
    assert prepared.audio_sha256 is None


def test_incomplete_text_subtitle_falls_back_to_audio(tmp_path: Path) -> None:
    runtime_root, probed = _probed_media(
        tmp_path,
        has_audio=True,
        subtitle_streams=(SubtitleStream(index=2, codec_name="mov_text", language="zh"),),
        duration_ms=30_000,
        language_hints=("zh",),
    )
    client = _RecordingTranscoder(
        runtime_root,
        subtitle_payloads={
            2: "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\n太短\n",
        },
    )

    prepared = ProductionMediaTranscoder(
        runtime_root,
        lambda _cancel: client,
    ).transcode(probed)

    assert client.extract_subtitle_calls == [2]
    assert client.extract_audio_calls == [(True, 30_000)]
    assert prepared.subtitle is None
    assert prepared.audio_path is not None
    assert prepared.warnings == ("SUBTITLE_TRACK_REJECTED:2:INCOMPLETE",)


def test_asr_audio_uses_video_timeline_instead_of_longer_container_duration(
    tmp_path: Path,
) -> None:
    runtime_root, probed = _probed_media(
        tmp_path,
        has_audio=True,
        subtitle_streams=(),
        duration_ms=302_366,
    )
    probed = replace(probed, timeline_duration_ms=302_101)
    client = _RecordingTranscoder(runtime_root)

    ProductionMediaTranscoder(
        runtime_root,
        lambda _cancel: client,
    ).transcode(probed)

    assert probed.manifest.duration_ms == 302_366
    assert probed.duration_ms == 302_101
    assert client.extract_audio_calls == [(True, 302_101)]


def test_rejected_first_subtitle_continues_to_second_complete_track(tmp_path: Path) -> None:
    runtime_root, probed = _probed_media(
        tmp_path,
        has_audio=True,
        subtitle_streams=(
            SubtitleStream(index=2, codec_name="mov_text", language="zh", is_default=True),
            SubtitleStream(index=3, codec_name="ass", language="en"),
        ),
        duration_ms=30_000,
        language_hints=("zh", "en"),
    )
    client = _RecordingTranscoder(
        runtime_root,
        subtitle_payloads={
            2: "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\n太短\n",
            3: _complete_vtt(),
        },
    )

    prepared = ProductionMediaTranscoder(
        runtime_root,
        lambda _cancel: client,
    ).transcode(probed)

    assert client.extract_subtitle_calls == [2, 3]
    assert client.extract_audio_calls == []
    assert prepared.subtitle is not None
    assert prepared.subtitle.artifact.stream_index == 3
    assert prepared.warnings == ("SUBTITLE_TRACK_REJECTED:2:INCOMPLETE",)


def test_subtitle_decode_failure_continues_to_next_track(tmp_path: Path) -> None:
    runtime_root, probed = _probed_media(
        tmp_path,
        has_audio=True,
        subtitle_streams=(
            SubtitleStream(index=2, codec_name="mov_text", language="zh"),
            SubtitleStream(index=3, codec_name="ass", language="en"),
        ),
        duration_ms=30_000,
    )
    client = _RecordingTranscoder(
        runtime_root,
        subtitle_payloads={3: _complete_vtt()},
        subtitle_errors={2: ErrorCode.VIDEO_PROCESS_FAILED},
    )

    prepared = ProductionMediaTranscoder(
        runtime_root,
        lambda _cancel: client,
    ).transcode(probed)

    assert client.extract_subtitle_calls == [2, 3]
    assert client.extract_audio_calls == []
    assert prepared.subtitle is not None
    assert prepared.warnings == ("SUBTITLE_TRACK_REJECTED:2:DECODE_FAILED",)


@pytest.mark.parametrize(
    "error_code",
    [
        ErrorCode.JOB_CANCELLED,
        ErrorCode.VIDEO_PROCESS_TIMEOUT,
        ErrorCode.WORKSPACE_PATH_ESCAPE,
    ],
)
def test_non_degradable_subtitle_failure_aborts_before_audio(
    tmp_path: Path,
    error_code: ErrorCode,
) -> None:
    runtime_root, probed = _probed_media(
        tmp_path,
        has_audio=True,
        subtitle_streams=(SubtitleStream(index=2, codec_name="mov_text", language="zh"),),
        duration_ms=30_000,
    )
    client = _RecordingTranscoder(
        runtime_root,
        subtitle_errors={2: error_code},
    )

    with pytest.raises(VideoDemoError) as raised:
        ProductionMediaTranscoder(
            runtime_root,
            lambda _cancel: client,
        ).transcode(probed)

    assert raised.value.code == error_code
    assert client.extract_audio_calls == []


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


def _manifest(
    *,
    has_audio: bool,
    subtitle_streams: tuple[SubtitleStream, ...] = (),
    duration_ms: int = 1_000,
) -> VideoAssetManifest:
    return VideoAssetManifest(
        object_ref="obj_001",
        source_sha256="a" * 64,
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
        audio_streams=(
            AudioStream(index=1, codec_name="aac", sample_rate_hz=48_000, channels=2),
        )
        if has_audio
        else (),
        subtitle_streams=subtitle_streams,
        format_name="mov,mp4",
        ffprobe_version="ffprobe test",
    )


def _probed_media(
    tmp_path: Path,
    *,
    has_audio: bool,
    subtitle_streams: tuple[SubtitleStream, ...],
    duration_ms: int,
    language_hints: tuple[str, ...] = (),
    speech_enrichment_mode: Literal["text", "full"] = "text",
) -> tuple[Path, object]:
    from video_demo.application.pipeline import PipelineRunConfig

    runtime_root = tmp_path / "runtime"
    source = runtime_root / "runs/scope/run_001/input/source.mp4"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"source")
    registered = _registered(source)
    registered = replace(
        registered,
        config=PipelineRunConfig(
            language_hints=language_hints,
            speech_enrichment_mode=speech_enrichment_mode,
        ),
    )
    probed = ProductionAssetProbe(
        lambda: _StaticProbeClient(
            _manifest(
                has_audio=has_audio,
                subtitle_streams=subtitle_streams,
                duration_ms=duration_ms,
            ),
        ),
    ).probe(registered)
    return runtime_root, probed


class _RecordingTranscoder:
    def __init__(
        self,
        runtime_root: Path,
        *,
        subtitle_payloads: dict[int, str] | None = None,
        subtitle_errors: dict[int, ErrorCode] | None = None,
    ) -> None:
        self._runtime_root = runtime_root
        self._subtitle_payloads = subtitle_payloads or {}
        self._subtitle_errors = subtitle_errors or {}
        self.extract_subtitle_calls: list[int] = []
        self.extract_audio_calls: list[tuple[bool, int]] = []

    def create_proxy(self, _path: Path, root: Path) -> ProxyVideoArtifact:
        relative = root / "media/proxy.mp4"
        output = self._runtime_root / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(_MP4)
        return ProxyVideoArtifact(
            relative_path=relative.as_posix(),
            sha256=hashlib.sha256(_MP4).hexdigest(),
            size_bytes=len(_MP4),
            max_edge=1280,
            normalized_start_ms=0,
        )

    def extract_subtitle(
        self,
        _path: Path,
        root: Path,
        stream: SubtitleStream,
    ) -> SubtitleArtifact:
        self.extract_subtitle_calls.append(stream.index)
        if stream.index in self._subtitle_errors:
            raise VideoDemoError(self._subtitle_errors[stream.index], "稳定测试错误")
        payload = self._subtitle_payloads[stream.index].encode("utf-8")
        relative = root / "media/subtitles" / f"{stream.index}.vtt"
        output = self._runtime_root / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(payload)
        return SubtitleArtifact(
            relative_path=relative.as_posix(),
            sha256=hashlib.sha256(payload).hexdigest(),
            size_bytes=len(payload),
            stream_index=stream.index,
            language=stream.language,
            codec_name=stream.codec_name,
        )

    def extract_audio(
        self,
        _path: Path,
        root: Path,
        *,
        has_audio: bool,
        duration_ms: int,
    ) -> AudioArtifact | NoAudioArtifact:
        self.extract_audio_calls.append((has_audio, duration_ms))
        if not has_audio:
            return NoAudioArtifact(warning_code="NO_AUDIO_TRACK")
        payload = b"wav"
        relative = root / "media/audio.wav"
        output = self._runtime_root / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(payload)
        return AudioArtifact(
            relative_path=relative.as_posix(),
            sha256=hashlib.sha256(payload).hexdigest(),
            size_bytes=len(payload),
            sample_rate_hz=16_000,
            channels=1,
            codec="pcm_s16le",
        )


def _complete_vtt() -> str:
    return (
        "WEBVTT\n\n"
        "00:00:00.000 --> 00:00:30.000\n"
        "这是一条覆盖完整视频并且字符数量足够的有效字幕文本内容\n"
    )
