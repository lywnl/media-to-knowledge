"""音频切片产物的归属和摘要校验。"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Protocol

from video_demo.application.audio_transcode import AudioSliceArtifact
from video_demo.domain.run import TimeRange
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.storage.workspace import safe_runtime_path


class AudioSliceClient(Protocol):
    def create_audio_slice(
        self,
        source: Path,
        run_relative_root: Path,
        slice_id: str,
        time_range: TimeRange,
        *,
        source_duration_ms: int,
    ) -> AudioSliceArtifact: ...


class VerifiedAudioSlicer:
    """确保 ASR 切片始终属于当前音频 Run 且摘要未被篡改。"""

    def __init__(self, runtime_root: Path, client: AudioSliceClient, duration_ms: int) -> None:
        self._runtime_root = runtime_root.expanduser().resolve(strict=False)
        self._client = client
        self._duration_ms = duration_ms

    def create(
        self,
        audio: Path,
        run_relative_root: Path,
        slice_id: str,
        time_range: TimeRange,
    ) -> Path:
        artifact = self._client.create_audio_slice(
            audio,
            run_relative_root,
            slice_id,
            time_range,
            source_duration_ms=self._duration_ms,
        )
        run_root = safe_runtime_path(self._runtime_root, run_relative_root)
        output = safe_runtime_path(self._runtime_root, Path(artifact.relative_path))
        if not output.is_relative_to(run_root) or output.is_symlink() or not output.is_file():
            raise VideoDemoError(
                ErrorCode.WORKSPACE_PATH_ESCAPE,
                "音频切片必须位于当前运行目录内",
            )
        if _sha256_file(output) != artifact.sha256:
            raise VideoDemoError(ErrorCode.AUDIO_DIGEST_MISMATCH, "音频切片摘要校验失败")
        return output


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
