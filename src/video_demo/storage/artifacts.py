from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any

from pydantic import Field

from video_demo.domain.base import FrozenModel, Sha256
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.storage.workspace import atomic_replace, safe_runtime_path


class ArtifactReceipt(FrozenModel):
    relative_path: str = Field(min_length=1, max_length=1024)
    schema_version: str = Field(min_length=1, max_length=32)
    sha256: Sha256
    upstream_sha256: Sha256


class AtomicArtifactStore:
    """将阶段 JSON 以规范 UTF-8 和原子替换方式写入运行目录。"""

    def __init__(self, runtime_root: Path) -> None:
        self.runtime_root = runtime_root.expanduser().resolve(strict=False)

    def write_json(
        self,
        relative_path: Path,
        payload: dict[str, Any] | list[Any],
        *,
        schema_version: str,
        upstream_sha256: str,
    ) -> ArtifactReceipt:
        destination = safe_runtime_path(self.runtime_root, relative_path)
        envelope = {
            "schema_version": schema_version,
            "upstream_sha256": upstream_sha256,
            "payload": payload,
        }
        encoded = json.dumps(
            envelope,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        digest = hashlib.sha256(encoded).hexdigest()
        temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.part")
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            with temporary.open("xb") as output:
                output.write(encoded)
                output.flush()
                os.fsync(output.fileno())
            atomic_replace(temporary, destination)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        return ArtifactReceipt(
            relative_path=relative_path.as_posix(),
            schema_version=schema_version,
            sha256=digest,
            upstream_sha256=upstream_sha256,
        )

    def read_verified_json(self, receipt: ArtifactReceipt) -> dict[str, Any] | list[Any]:
        artifact = safe_runtime_path(self.runtime_root, Path(receipt.relative_path))
        if artifact.is_symlink() or not artifact.is_file():
            raise VideoDemoError(ErrorCode.ARTIFACT_NOT_FOUND, "阶段产物不存在")
        encoded = artifact.read_bytes()
        if hashlib.sha256(encoded).hexdigest() != receipt.sha256:
            raise VideoDemoError(ErrorCode.ARTIFACT_DIGEST_MISMATCH, "阶段产物摘要不匹配")
        try:
            envelope: object = json.loads(encoded)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise VideoDemoError(
                ErrorCode.ARTIFACT_SCHEMA_INVALID,
                "阶段产物不是合法 JSON",
            ) from error
        if not isinstance(envelope, dict):
            raise VideoDemoError(ErrorCode.ARTIFACT_SCHEMA_INVALID, "阶段产物外层必须是对象")
        if envelope.get("schema_version") != receipt.schema_version:
            raise VideoDemoError(ErrorCode.ARTIFACT_SCHEMA_INVALID, "阶段产物 Schema 版本不匹配")
        if envelope.get("upstream_sha256") != receipt.upstream_sha256:
            raise VideoDemoError(ErrorCode.ARTIFACT_UPSTREAM_MISMATCH, "阶段产物上游摘要不匹配")
        payload = envelope.get("payload")
        if not isinstance(payload, (dict, list)):
            raise VideoDemoError(ErrorCode.ARTIFACT_SCHEMA_INVALID, "阶段产物 payload 类型非法")
        return payload

    def discard(self, receipt: ArtifactReceipt) -> bool:
        """仅删除摘要仍与本次写入一致的未发布产物。"""

        artifact = safe_runtime_path(self.runtime_root, Path(receipt.relative_path))
        if artifact.is_symlink() or not artifact.is_file():
            return False
        if hashlib.sha256(artifact.read_bytes()).hexdigest() != receipt.sha256:
            return False
        artifact.unlink()
        return True
