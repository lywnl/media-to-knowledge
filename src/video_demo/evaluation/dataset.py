from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import Field, ValidationError, model_validator

from video_demo.domain.base import FrozenModel, Sha256, StableId
from video_demo.errors import ErrorCode, VideoDemoError

ValidationLanguage = Literal["zh", "en", "ja", "ko", "es"]


class EvaluationSample(FrozenModel):
    sample_id: StableId
    language: ValidationLanguage
    authorization_id: StableId
    media_relative_path: str = Field(min_length=1, max_length=1024)
    media_sha256: Sha256
    annotations_relative_path: str = Field(min_length=1, max_length=1024)
    annotations_sha256: Sha256

    @model_validator(mode="after")
    def validate_relative_paths(self) -> EvaluationSample:
        for value in (self.media_relative_path, self.annotations_relative_path):
            path = Path(value)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("评测路径必须是无穿越的相对路径")
        return self


class EvaluationDataset(FrozenModel):
    """通过 Manifest 路径加载的评测集；可信根目录故意不进入 JSON。"""

    samples: tuple[EvaluationSample, ...] = Field(min_length=1)
    eval_root: Path = Field(exclude=True)
    runtime_root: Path = Field(exclude=True)
    workspace_root: Path = Field(exclude=True)
    source_path: Path | None = Field(default=None, exclude=True)

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        workspace_root: Path,
        runtime_root: Path,
    ) -> EvaluationDataset:
        samples: list[EvaluationSample] = []
        try:
            safe_runtime_root = _safe_runtime_root(workspace_root, runtime_root)
            _require_within_root(path.parent, safe_runtime_root)
            encoded = _read_json_file(path)
            for line in encoded.decode("utf-8").splitlines():
                if not line.strip():
                    continue
                payload = json.loads(line)
                samples.append(EvaluationSample.model_validate(payload))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValidationError, ValueError):
            raise VideoDemoError(
                ErrorCode.EVALUATION_DATASET_INVALID,
                "评测集 Manifest 非法",
            ) from None
        try:
            dataset = cls(
                samples=tuple(samples),
                eval_root=path.parent.resolve(strict=True),
                runtime_root=safe_runtime_root,
                workspace_root=workspace_root.resolve(strict=True),
                source_path=path.resolve(strict=True),
            )
        except ValidationError:
            raise VideoDemoError(
                ErrorCode.EVALUATION_DATASET_INVALID,
                "评测集 Manifest 不能为空",
            ) from None
        if len({sample.sample_id for sample in dataset.samples}) != len(dataset.samples):
            raise VideoDemoError(ErrorCode.EVALUATION_DATASET_INVALID, "评测样本 ID 重复")
        return dataset

    def validate_final_gate(self, *, max_video_bytes: int) -> None:
        _validate_max_video_bytes(max_video_bytes)
        language_counts = Counter(sample.language for sample in self.samples)
        required_languages: tuple[ValidationLanguage, ...] = ("zh", "en", "ja", "ko", "es")
        if len(self.samples) < 30 or any(
            language_counts[language] < 6 for language in required_languages
        ):
            raise VideoDemoError(
                ErrorCode.EVALUATION_DATASET_INVALID,
                "最终评测至少需要 30 条且每种验收语言至少 6 条",
                {"sample_count": len(self.samples)},
            )
        for sample in self.samples:
            self._validate_media(sample.media_relative_path, sample.media_sha256, max_video_bytes)
            self._validate_artifact(
                sample.annotations_relative_path,
                sample.annotations_sha256,
            )

    def _validate_artifact(self, relative_path: str, expected_sha256: str) -> None:
        try:
            artifact = _safe_relative_file(self.eval_root, relative_path, self.runtime_root)
            artifact_bytes = _read_json_file(artifact)
            if hashlib.sha256(artifact_bytes).hexdigest() != expected_sha256:
                raise ValueError("评测文件类型或摘要不匹配")
        except (OSError, ValueError):
            raise VideoDemoError(
                ErrorCode.EVALUATION_DATASET_INVALID,
                "评测文件缺失、越界或摘要不匹配",
            ) from None

    def _validate_media(
        self,
        relative_path: str,
        expected_sha256: str,
        max_video_bytes: int,
    ) -> None:
        try:
            artifact = _safe_relative_file(self.eval_root, relative_path, self.runtime_root)
            if _sha256_media(artifact, max_video_bytes) != expected_sha256:
                raise ValueError("媒体摘要不匹配")
        except (OSError, ValueError):
            raise VideoDemoError(
                ErrorCode.EVALUATION_DATASET_INVALID,
                "评测媒体缺失、越界或摘要不匹配",
            ) from None


_MAX_CONTRACT_BYTES = 64 * 1024 * 1024


def _read_json_file(path: Path) -> bytes:
    """读取受限 JSON/JSONL 契约，拒绝空文件、链接和超大文件。"""

    _reject_symlink_components(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError("契约文件不存在、不是普通文件或为空")
    if path.stat().st_size > _MAX_CONTRACT_BYTES:
        raise ValueError("契约文件超过大小上限")
    content = path.read_bytes()
    if not content.strip():
        raise ValueError("契约文件不能为空")
    content.decode("utf-8")
    return content


def _reject_symlink_components(path: Path) -> None:
    absolute = path.absolute()
    components = (absolute, *absolute.parents)
    if any(component.is_symlink() for component in components):
        raise ValueError("契约路径不能包含符号链接")


def _safe_relative_file(root: Path, relative_path: str, runtime_root: Path) -> Path:
    """逐组件拒绝符号链接，避免 resolve 后才发现的目录穿越。"""

    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError("路径必须是无穿越的相对路径")
    root = _safe_root(root, runtime_root)
    current = root
    for component in relative.parts:
        current = current / component
        if current.is_symlink():
            raise ValueError("评测路径不能包含符号链接")
    if not current.is_file():
        raise ValueError("评测路径不是普通文件")
    return current


def _safe_root(root: Path, runtime_root: Path) -> Path:
    """在解析前验证根及其父目录无 symlink，且同时位于受信 runtime 内。"""

    _require_within_root(root, runtime_root)
    if not root.is_dir():
        raise ValueError("评测根目录必须是目录")
    return root.resolve(strict=True)


def _safe_runtime_root(workspace_root: Path, runtime_root: Path) -> Path:
    """验证调用方传入的工作区和 runtime 根，避免外部目录自封为受信根。"""

    _reject_symlink_components(workspace_root)
    _reject_symlink_components(runtime_root)
    if not workspace_root.is_dir() or not runtime_root.is_dir():
        raise ValueError("工作区和运行根必须是目录")
    workspace = workspace_root.resolve(strict=True)
    expected_runtime = workspace_root / ".codex" / "video-rag-demo"
    if runtime_root.absolute() != expected_runtime.absolute():
        raise ValueError("运行根必须是工作区固定评测目录")
    runtime = runtime_root.resolve(strict=True)
    if runtime != (workspace / ".codex" / "video-rag-demo").resolve(strict=True):
        raise ValueError("运行根必须是工作区固定评测目录")
    return runtime


def _require_within_root(path: Path, root: Path) -> None:
    """同时检查词法和解析后的路径归属，且在读取前拒绝任意链接组件。"""

    _reject_symlink_components(path)
    _reject_symlink_components(root)
    try:
        path.absolute().relative_to(root.absolute())
        path.resolve(strict=False).relative_to(root.resolve(strict=True))
    except ValueError as error:
        raise ValueError("路径不在受信运行根内") from error


def _validate_max_video_bytes(max_video_bytes: int) -> None:
    if type(max_video_bytes) is not int or max_video_bytes <= 0:
        raise ValueError("媒体大小上限必须是正整数")


def _sha256_media(path: Path, max_video_bytes: int) -> str:
    _validate_max_video_bytes(max_video_bytes)
    if not path.is_file() or path.stat().st_size == 0 or path.stat().st_size > max_video_bytes:
        raise ValueError("媒体文件为空、类型非法或超过大小上限")
    digest = hashlib.sha256()
    total = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            total += len(chunk)
            if total > max_video_bytes:
                raise ValueError("媒体文件超过大小上限")
            digest.update(chunk)
    if total == 0:
        raise ValueError("媒体文件不能为空")
    return digest.hexdigest()
