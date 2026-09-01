from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

from video_demo.application.pipeline import PreparedMedia
from video_demo.evaluation import real_media_execution as execution
from video_demo.evaluation.evidence import RealMediaFile
from video_demo.media.audio_format import AUDIO_FORMAT_VERSION
from video_demo.media.audio_transcode import AudioArtifact


class _Session:
    def assert_registered_leaves(self, _paths: object) -> None:
        return None

    def open_registered_leaf(self, _path: Path) -> int:
        return os.open(os.devnull, os.O_RDONLY)

    def stage_output(self, _path: Path) -> SimpleNamespace:
        return SimpleNamespace(descriptor=os.open(os.devnull, os.O_RDONLY))

    def publish_output(self, _staged: object, _max_bytes: int) -> None:
        return None

    def discard_output(self, staged: SimpleNamespace) -> None:
        os.close(staged.descriptor)


def test_audio_phase_marks_mp3_format_version(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_root = Path("eval/generated/run-1/normal_audio")
    probed = SimpleNamespace(
        asset=SimpleNamespace(
            run_relative_root=run_root,
            source_path=tmp_path / "source.mp4",
        ),
        manifest=SimpleNamespace(audio_streams=(object(),)),
        duration_ms=10_000,
        warnings=(),
    )
    artifact = AudioArtifact(
        relative_path=(run_root / "media/audio.mp3").as_posix(),
        sha256="a" * 64,
        size_bytes=123,
        sample_rate_hz=16_000,
        channels=1,
        codec="mp3",
    )
    monkeypatch.setattr(
        execution,
        "_transfer_audio",
        lambda *_args, **_kwargs: SimpleNamespace(
            media_file=SimpleNamespace(sha256="b" * 64),
        ),
    )
    monkeypatch.setattr(execution, "_complete_phase", lambda *_args, **_kwargs: None)

    facts = SimpleNamespace(
        files=[
            RealMediaFile(
                role="SOURCE",
                format="MP4",
                relative_path=(
                    ".codex/video-rag-demo/eval/generated/run-1/normal_audio/source.mp4"
                ),
                sha256="c" * 64,
                size_bytes=123,
            ),
        ],
        commands=[],
        artifacts=[],
        registered_paths=set(),
    )
    prepared = execution._audio_phase(
        "normal_audio",
        probed,
        SimpleNamespace(extract_audio=lambda *_args, **_kwargs: artifact),
        SimpleNamespace(runtime_root=tmp_path, workspace_root=tmp_path),
        SimpleNamespace(begin_phase=lambda **_kwargs: None),
        facts,
        1024 * 1024,
        _Session(),
    )

    assert isinstance(prepared, PreparedMedia)
    assert prepared.audio_format_version == AUDIO_FORMAT_VERSION
