"""音频 ASR 窗口结果的不可变快照存储。"""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import ValidationError

from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.speech.audio_snapshots import AudioAsrWindowSnapshotPayload
from video_demo.storage.artifact_inspection import inspect_artifact
from video_demo.storage.artifacts import ArtifactReceipt, AtomicArtifactStore
from video_demo.storage.workspace import reject_symlink_components

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_RUN_COMPONENT_PATTERN = re.compile(r"^[A-Za-z0-9_-]{3,128}$")
_CACHE_MISS_ERROR_CODES = frozenset(
    {
        "ARTIFACT_NOT_FOUND",
        "ARTIFACT_DIGEST_MISMATCH",
        "ARTIFACT_SCHEMA_INVALID",
        "ARTIFACT_UPSTREAM_MISMATCH",
    }
)


class AudioAsrWindowSnapshotStore:
    """按音频窗口指纹保存可恢复的 ASR 结果。"""

    def __init__(self, artifact_store: AtomicArtifactStore) -> None:
        self._artifact_store = artifact_store

    def load(
        self,
        run_relative_root: Path,
        fingerprint: str,
    ) -> tuple[AudioAsrWindowSnapshotPayload, ArtifactReceipt] | None:
        path = self._window_path(run_relative_root, fingerprint)
        self._verified_path(path)
        try:
            receipt, payload = inspect_artifact(
                self._artifact_store,
                path,
                schema_version="1.0.0",
                upstream_sha256=fingerprint,
                max_bytes=16 * 1024 * 1024,
            )
            return AudioAsrWindowSnapshotPayload.model_validate(payload), receipt
        except VideoDemoError as error:
            if error.code.value not in _CACHE_MISS_ERROR_CODES:
                raise
            return None
        except (FileNotFoundError, OSError, ValueError, ValidationError):
            return None

    def publish(
        self,
        run_relative_root: Path,
        fingerprint: str,
        payload: AudioAsrWindowSnapshotPayload,
    ) -> ArtifactReceipt:
        path = self._window_path(run_relative_root, fingerprint)
        self._verified_path(path)
        try:
            return self._artifact_store.write_json(
                path,
                payload.model_dump(mode="json", exclude_computed_fields=True),
                schema_version="1.0.0",
                upstream_sha256=fingerprint,
                exclusive=True,
            )
        except FileExistsError:
            existing = self.load(run_relative_root, fingerprint)
            if existing is None or existing[0] != payload:
                raise VideoDemoError(
                    ErrorCode.ARTIFACT_DIGEST_MISMATCH,
                    "已有音频 ASR 窗口快照与待发布内容不一致",
                ) from None
            return existing[1]

    @staticmethod
    def _window_path(run_relative_root: Path, fingerprint: str) -> Path:
        if (
            run_relative_root.is_absolute()
            or len(run_relative_root.parts) != 3
            or run_relative_root.parts[0] != "runs"
            or not all(
                _RUN_COMPONENT_PATTERN.fullmatch(component)
                for component in run_relative_root.parts[1:]
            )
            or not _SHA256_PATTERN.fullmatch(fingerprint)
        ):
            raise VideoDemoError(
                ErrorCode.WORKSPACE_PATH_ESCAPE,
                "音频 ASR 窗口快照路径参数非法",
            )
        return (
            run_relative_root
            / "speech"
            / "snapshots"
            / "asr-windows"
            / f"window-{fingerprint}.json"
        )

    def _verified_path(self, relative_path: Path) -> Path:
        return reject_symlink_components(
            self._artifact_store.runtime_root,
            self._artifact_store.runtime_root / relative_path,
            message="音频 ASR 窗口快照路径必须位于运行目录内且不能包含符号链接",
        )
