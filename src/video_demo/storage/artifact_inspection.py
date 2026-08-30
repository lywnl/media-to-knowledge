"""阶段制品的规范 envelope 读取工具。

该模块只依赖制品存储和校验基础设施，供不同媒体链路独立使用。
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from video_demo.storage.artifacts import (
    ArtifactReceipt,
    AtomicArtifactStore,
    canonical_artifact_envelope_bytes,
)
from video_demo.storage.workspace import reject_symlink_components


def inspect_artifact(
    store: AtomicArtifactStore,
    relative_path: Path,
    *,
    schema_version: str,
    upstream_sha256: str,
    max_bytes: int,
) -> tuple[ArtifactReceipt, dict[str, Any] | list[Any]]:
    """按稳定路径读取规范 envelope，并重建当前文件回执。"""

    candidate = reject_symlink_components(
        store.runtime_root,
        store.runtime_root / relative_path,
        message="阶段制品路径非法",
    )
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    if max_bytes < 1:
        raise ValueError("阶段制品读取上限必须大于 0")
    with candidate.open("rb") as stream:
        encoded = stream.read(max_bytes + 1)
    if len(encoded) > max_bytes:
        raise ValueError("阶段制品超过读取上限")
    receipt = ArtifactReceipt(
        relative_path=relative_path.as_posix(),
        schema_version=schema_version,
        sha256=hashlib.sha256(encoded).hexdigest(),
        upstream_sha256=upstream_sha256,
    )
    payload = store.read_verified_json_limited(receipt, max_bytes=max_bytes)
    if encoded != canonical_artifact_envelope_bytes(
        payload,
        schema_version,
        upstream_sha256,
    ):
        raise ValueError("阶段制品 envelope 不是规范编码")
    return receipt, payload
