from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Literal, TypeVar

from pydantic import ValidationError

from video_demo.domain.base import FrozenModel, Sha256
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.speech.snapshots import AsrWindowSnapshotPayload
from video_demo.storage.artifact_inspection import inspect_artifact
from video_demo.storage.artifacts import (
    ArtifactReceipt,
    AtomicArtifactStore,
    canonical_artifact_envelope_bytes,
)
from video_demo.storage.workspace import reject_symlink_components

SnapshotKind = Literal["asr"]
PayloadT = TypeVar("PayloadT", bound=FrozenModel)
_MAX_POINTER_BYTES = 64 * 1024
_MAX_SNAPSHOT_PAYLOAD_BYTES = 64 * 1024 * 1024
_MAX_ASR_WINDOW_PAYLOAD_BYTES = 16 * 1024 * 1024
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

__all__ = [
    "AsrWindowSnapshotStore",
    "SnapshotPointer",
    "SnapshotStore",
    "inspect_artifact",
]


class SnapshotPointer(FrozenModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    kind: SnapshotKind
    fingerprint: Sha256
    payload_receipt: ArtifactReceipt


class AsrWindowSnapshotStore:
    """按窗口指纹保存不可覆盖的云端 ASR 付费结果。"""

    def __init__(self, artifact_store: AtomicArtifactStore) -> None:
        self._artifact_store = artifact_store

    def load(
        self,
        run_relative_root: Path,
        fingerprint: str,
    ) -> tuple[AsrWindowSnapshotPayload, ArtifactReceipt] | None:
        path = self._window_path(run_relative_root, fingerprint)
        self._verified_path(path)
        try:
            receipt, payload = inspect_artifact(
                self._artifact_store,
                path,
                schema_version="1.1.0",
                upstream_sha256=fingerprint,
                max_bytes=_MAX_ASR_WINDOW_PAYLOAD_BYTES,
            )
            return AsrWindowSnapshotPayload.model_validate(payload), receipt
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
        payload: AsrWindowSnapshotPayload,
    ) -> ArtifactReceipt:
        path = self._window_path(run_relative_root, fingerprint)
        self._verified_path(path)
        payload_data = payload.model_dump(mode="json", exclude_computed_fields=True)
        try:
            return self._artifact_store.write_json(
                path,
                payload_data,
                schema_version=payload.schema_version,
                upstream_sha256=fingerprint,
                exclusive=True,
            )
        except FileExistsError:
            existing = self.load(run_relative_root, fingerprint)
            if existing is None or existing[0] != payload:
                raise VideoDemoError(
                    ErrorCode.ARTIFACT_DIGEST_MISMATCH,
                    "已有云端 ASR 窗口缓存与待发布内容不一致",
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
                "云端 ASR 窗口快照路径参数非法",
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
            message="云端 ASR 窗口快照路径必须位于运行目录内且不能包含符号链接",
        )


class SnapshotStore:
    """管理当前 Run 内不可变快照 payload 与原子 current pointer。"""

    def __init__(self, artifact_store: AtomicArtifactStore) -> None:
        self._artifact_store = artifact_store

    def load(
        self,
        run_relative_root: Path,
        kind: SnapshotKind,
        fingerprint: str,
        payload_type: type[PayloadT],
    ) -> tuple[PayloadT, ArtifactReceipt] | None:
        try:
            pointer_path = self._pointer_path(run_relative_root, kind)
            _pointer_receipt, pointer_payload = inspect_artifact(
                self._artifact_store,
                pointer_path,
                schema_version="1.0.0",
                upstream_sha256=fingerprint,
                max_bytes=_MAX_POINTER_BYTES,
            )
            pointer = SnapshotPointer.model_validate(pointer_payload)
            if pointer.kind != kind or pointer.fingerprint != fingerprint:
                return None
            receipt = pointer.payload_receipt
            expected_parent = self._payload_parent(run_relative_root, kind)
            payload_path = Path(receipt.relative_path)
            if payload_path.is_absolute() or ".." in payload_path.parts:
                self._raise_security_error()
            if not payload_path.is_relative_to(run_relative_root):
                self._raise_security_error()
            if (
                payload_path.parent != expected_parent
                or payload_path.name != f"payload-{receipt.sha256}.json"
                or receipt.upstream_sha256 != fingerprint
            ):
                if payload_path.parent != expected_parent:
                    self._raise_security_error()
                return None
            self._verified_lexical_path(payload_path)
            payload = self._artifact_store.read_verified_json_limited(
                receipt,
                max_bytes=_MAX_SNAPSHOT_PAYLOAD_BYTES,
            )
            return payload_type.model_validate(payload), receipt
        except VideoDemoError as error:
            if error.code.value not in _CACHE_MISS_ERROR_CODES:
                raise
            return None
        except (OSError, ValueError, ValidationError):
            return None

    def publish(
        self,
        run_relative_root: Path,
        kind: SnapshotKind,
        fingerprint: str,
        payload: FrozenModel,
    ) -> ArtifactReceipt:
        payload_data = payload.model_dump(mode="json", exclude_computed_fields=True)
        schema_version = _payload_schema_version(payload)
        encoded = canonical_artifact_envelope_bytes(
            payload_data,
            schema_version,
            fingerprint,
        )
        payload_sha256 = hashlib.sha256(encoded).hexdigest()
        payload_path = (
            self._payload_parent(run_relative_root, kind)
            / f"payload-{payload_sha256}.json"
        )
        self._verified_lexical_path(payload_path)
        receipt = self._artifact_store.write_json(
            payload_path,
            payload_data,
            schema_version=schema_version,
            upstream_sha256=fingerprint,
        )
        if receipt.sha256 != payload_sha256:
            raise RuntimeError("快照 payload 文件名摘要与写入回执不一致")
        self._artifact_store.read_verified_json(receipt)

        pointer = SnapshotPointer(
            kind=kind,
            fingerprint=fingerprint,
            payload_receipt=receipt,
        )
        pointer_path = self._pointer_path(run_relative_root, kind)
        self._verified_lexical_path(pointer_path)
        self._artifact_store.write_json(
            pointer_path,
            pointer.model_dump(mode="json"),
            schema_version=pointer.schema_version,
            upstream_sha256=fingerprint,
        )
        return receipt

    @staticmethod
    def _payload_parent(run_relative_root: Path, kind: SnapshotKind) -> Path:
        return run_relative_root / "speech" / "snapshots" / kind

    @staticmethod
    def _pointer_path(run_relative_root: Path, kind: SnapshotKind) -> Path:
        return run_relative_root / "speech" / "snapshots" / f"{kind}-current.json"

    def _verified_lexical_path(self, relative_path: Path) -> Path:
        return reject_symlink_components(
            self._artifact_store.runtime_root,
            self._artifact_store.runtime_root / relative_path,
            message="语音快照路径必须位于运行目录内且不能包含符号链接",
        )

    @staticmethod
    def _raise_security_error() -> None:
        from video_demo.errors import ErrorCode

        raise VideoDemoError(
            ErrorCode.WORKSPACE_PATH_ESCAPE,
            "语音快照必须位于当前运行目录内",
        )


def _payload_schema_version(payload: FrozenModel) -> str:
    value = getattr(payload, "schema_version", None)
    if not isinstance(value, str) or not value:
        raise ValueError("快照 payload 缺少 schema_version")
    return value
