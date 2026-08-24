from __future__ import annotations

import hashlib
import json
import os
import stat
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Generic, Literal, TypeVar

from pydantic import Field, ValidationError, field_validator

try:
    import fcntl
except ImportError:  # pragma: no cover - 生产平台门禁，CI 使用 POSIX。
    fcntl = None  # type: ignore[assignment]

from video_demo.domain.base import FrozenModel, Sha256
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.storage.workspace import reject_symlink_components, validate_path_component

ResponseModel = TypeVar("ResponseModel", bound=FrozenModel)
SuccessfulPath = Literal["MAIN", "REPAIR"]
_READ_CHUNK_BYTES = 1024 * 1024
_OPEN_SUPPORTS_DIR_FD = os.open in os.supports_dir_fd


class ModelInvocationIdentity(FrozenModel):
    """提前可计算且覆盖供应商、模型、Schema 与 Prompt 的逻辑调用身份。"""

    logical_operation: str = Field(min_length=3, max_length=128)
    provider_config_fingerprint: Sha256
    model_id: str = Field(min_length=1, max_length=256)
    generation_config: tuple[tuple[str, str], ...] = Field(max_length=64)
    main_response_schema_name: str = Field(min_length=1, max_length=128)
    main_prompt_version: str = Field(min_length=1, max_length=128)
    repair_response_schema_name: str = Field(min_length=1, max_length=128)
    repair_prompt_version: str = Field(min_length=1, max_length=128)

    @field_validator("generation_config")
    @classmethod
    def require_canonical_generation_config(
        cls,
        value: tuple[tuple[str, str], ...],
    ) -> tuple[tuple[str, str], ...]:
        if value != tuple(sorted(value)) or len({key for key, _ in value}) != len(value):
            raise ValueError("generation_config 必须按键排序且键不得重复")
        return value


class CachedModelResult(FrozenModel, Generic[ResponseModel]):
    response: ResponseModel
    successful_path: SuccessfulPath


class DocumentModelCache:
    """Run 内不可变结构化模型缓存；跨实例/进程使用 POSIX 文件锁。"""

    def __init__(self, run_root: Path, *, max_entry_bytes: int, max_run_bytes: int) -> None:
        if fcntl is None:
            raise VideoDemoError(ErrorCode.INVALID_CONFIGURATION, "当前平台不支持安全模型缓存锁")
        self._run_root = _validated_run_root(run_root)
        self._cache_root = self._run_root / "model-cache"
        self._invocation_lock_root = self._run_root / ".model-invocation-locks"
        self._lock_path = self._run_root / ".model-cache.lock"
        self._max_entry_bytes = max_entry_bytes
        self._max_run_bytes = max_run_bytes

    @contextmanager
    def invocation_lock(
        self,
        identity: ModelInvocationIdentity,
        canonical_input: FrozenModel,
        *,
        wait_timeout_seconds: float,
        is_cancel_requested: Callable[[], bool],
        poll_interval_seconds: float = 0.05,
    ) -> Iterator[None]:
        """按逻辑指纹串行付费调用；调用者需在锁内重新查询缓存。"""

        _require_frozen_instance(canonical_input, "模型缓存规范输入")
        if wait_timeout_seconds <= 0 or poll_interval_seconds <= 0:
            raise VideoDemoError(ErrorCode.INVALID_CONFIGURATION, "模型调用锁等待参数必须大于 0")
        fingerprint = _cache_fingerprint(identity, canonical_input)
        descriptor = self._open_invocation_lock_file(f"{fingerprint}.lock")
        acquired = False
        deadline = time.monotonic() + wait_timeout_seconds
        try:
            while not acquired:
                if is_cancel_requested():
                    raise VideoDemoError(ErrorCode.JOB_CANCELLED, "任务已请求取消")
                try:
                    assert fcntl is not None
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    acquired = True
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise VideoDemoError(
                            ErrorCode.DEPENDENCY_TEMPORARY_FAILURE,
                            "等待同一模型逻辑调用超时",
                        ) from None
                    time.sleep(min(poll_interval_seconds, max(0.0, deadline - time.monotonic())))
            yield
        finally:
            if acquired:
                assert fcntl is not None
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def get(
        self,
        identity: ModelInvocationIdentity,
        canonical_input: FrozenModel,
        response_model: type[ResponseModel],
        validate: Callable[[ResponseModel], None],
    ) -> CachedModelResult[ResponseModel] | None:
        _require_frozen_instance(canonical_input, "模型缓存规范输入")
        with self._run_lock(exclusive=False):
            path = self._path(identity, canonical_input)
            return self._read(
                path,
                identity,
                canonical_input,
                response_model,
                validate,
            )

    def put(
        self,
        identity: ModelInvocationIdentity,
        canonical_input: FrozenModel,
        response: ResponseModel,
        *,
        successful_path: SuccessfulPath,
        validate: Callable[[ResponseModel], None],
    ) -> CachedModelResult[ResponseModel]:
        _require_frozen_instance(canonical_input, "模型缓存规范输入")
        _require_frozen_model(type(response), "模型缓存响应")
        _validate_response(response, validate)
        response_type = type(response)
        result: CachedModelResult[ResponseModel] = CachedModelResult(
            response=response,
            successful_path=successful_path,
        )
        encoded = self._encode(identity, canonical_input, result)
        if len(encoded) > self._max_entry_bytes:
            raise VideoDemoError(ErrorCode.INPUT_BUDGET_EXCEEDED, "模型缓存条目超过大小上限")

        with self._run_lock(exclusive=True):
            path = self._path(identity, canonical_input)
            existing = self._read(
                path,
                identity,
                canonical_input,
                response_type,
                validate,
            )
            if existing is not None:
                return existing
            self._ensure_budget(len(encoded))
            self._ensure_private_directory(path.parent)
            temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
            try:
                _write_private_file(temporary, encoded)
                try:
                    os.link(temporary, path)
                except FileExistsError:
                    existing = self._read(
                        path,
                        identity,
                        canonical_input,
                        response_type,
                        validate,
                    )
                    if existing is not None:
                        return existing
                    raise VideoDemoError(
                        ErrorCode.ARTIFACT_SCHEMA_INVALID,
                        "模型缓存并发条目无效",
                    ) from None
                return result
            finally:
                temporary.unlink(missing_ok=True)

    def _path(
        self,
        identity: ModelInvocationIdentity,
        canonical_input: FrozenModel,
    ) -> Path:
        operation = validate_path_component(identity.logical_operation, "模型缓存操作")
        fingerprint = _cache_fingerprint(identity, canonical_input)
        path = self._cache_root / operation / f"{fingerprint}.json"
        try:
            return reject_symlink_components(
                self._run_root,
                path,
                message="模型缓存路径不能越出当前 Run 或包含符号链接",
            )
        except VideoDemoError as error:
            if error.code == ErrorCode.WORKSPACE_PATH_ESCAPE:
                raise VideoDemoError(
                    ErrorCode.ARTIFACT_SCHEMA_INVALID,
                    "模型缓存路径非法",
                ) from None
            raise

    def _encode(
        self,
        identity: ModelInvocationIdentity,
        canonical_input: FrozenModel,
        result: CachedModelResult[ResponseModel],
    ) -> bytes:
        envelope = {
            "identity_digest": _identity_digest(identity),
            "input_sha256": _input_sha256(canonical_input),
            "response": result.response.model_dump(mode="json"),
            "successful_path": result.successful_path,
        }
        return _canonical_json(envelope)

    def _read(
        self,
        path: Path,
        identity: ModelInvocationIdentity,
        canonical_input: FrozenModel,
        response_model: type[ResponseModel],
        validate: Callable[[ResponseModel], None],
    ) -> CachedModelResult[ResponseModel] | None:
        _require_frozen_model(response_model, "模型缓存响应类型")
        try:
            before = os.lstat(path)
        except FileNotFoundError:
            return None
        except OSError as error:
            raise VideoDemoError(ErrorCode.ARTIFACT_SCHEMA_INVALID, "模型缓存文件非法") from error
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
            raise VideoDemoError(ErrorCode.ARTIFACT_SCHEMA_INVALID, "模型缓存文件非法")
        if before.st_size > self._max_entry_bytes:
            raise VideoDemoError(ErrorCode.INPUT_BUDGET_EXCEEDED, "模型缓存条目超过大小上限")
        raw = _read_private_file(path, before, self._max_entry_bytes)
        try:
            envelope = json.loads(raw)
            if set(envelope) != {
                "identity_digest",
                "input_sha256",
                "response",
                "successful_path",
            }:
                raise ValueError
            if envelope["identity_digest"] != _identity_digest(identity):
                raise ValueError
            if envelope["input_sha256"] != _input_sha256(canonical_input):
                raise ValueError
            response = response_model.model_validate(envelope["response"])
            _validate_response(response, validate)
            return CachedModelResult[ResponseModel](
                response=response,
                successful_path=envelope["successful_path"],
            )
        except (UnicodeDecodeError, ValueError, KeyError, TypeError, ValidationError):
            raise VideoDemoError(ErrorCode.ARTIFACT_SCHEMA_INVALID, "模型缓存内容非法") from None

    def _ensure_budget(self, new_size: int) -> None:
        current = 0
        if self._cache_root.exists():
            if self._cache_root.is_symlink() or not self._cache_root.is_dir():
                raise VideoDemoError(ErrorCode.ARTIFACT_SCHEMA_INVALID, "模型缓存目录非法")
            for path in self._cache_root.glob("*/*.json"):
                try:
                    status = os.lstat(path)
                except OSError as error:
                    raise VideoDemoError(
                        ErrorCode.ARTIFACT_SCHEMA_INVALID,
                        "模型缓存预算扫描失败",
                    ) from error
                if stat.S_ISLNK(status.st_mode) or not stat.S_ISREG(status.st_mode):
                    raise VideoDemoError(ErrorCode.ARTIFACT_SCHEMA_INVALID, "模型缓存文件非法")
                current += status.st_size
        if current + new_size > self._max_run_bytes:
            raise VideoDemoError(ErrorCode.INPUT_BUDGET_EXCEEDED, "模型缓存超过 Run 总预算")

    def _ensure_private_directory(self, directory: Path) -> None:
        _ensure_private_directory(self._cache_root)
        _ensure_private_directory(directory)
        reject_symlink_components(
            self._run_root,
            directory,
            message="模型缓存目录不能包含符号链接",
        )

    def _open_invocation_lock_file(self, lock_name: str) -> int:
        directory_descriptor = _open_private_directory(
            self._invocation_lock_root,
            invalid_message="模型调用锁目录非法",
        )
        try:
            opened_directory = os.fstat(directory_descriptor)
            _require_directory_path_identity(
                self._invocation_lock_root,
                opened_directory,
                "模型调用锁目录在打开锁文件前发生变化",
            )
            descriptor = _open_lock_file(lock_name, dir_fd=directory_descriptor)
            try:
                _require_directory_path_identity(
                    self._invocation_lock_root,
                    opened_directory,
                    "模型调用锁目录在打开锁文件后发生变化",
                )
                return descriptor
            except BaseException:
                os.close(descriptor)
                raise
        finally:
            os.close(directory_descriptor)

    @contextmanager
    def _run_lock(self, *, exclusive: bool):  # type: ignore[no-untyped-def]
        assert fcntl is not None
        descriptor = _open_lock_file(self._lock_path)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)


def _validated_run_root(run_root: Path) -> Path:
    path = run_root.expanduser()
    if not path.is_absolute():
        path = path.resolve(strict=False)
    if path.is_symlink():
        raise VideoDemoError(ErrorCode.WORKSPACE_PATH_ESCAPE, "模型缓存 Run 根不能是符号链接")
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not path.is_dir():
        raise VideoDemoError(ErrorCode.WORKSPACE_PATH_ESCAPE, "模型缓存 Run 根非法")
    return path.resolve(strict=True)


def _open_lock_file(path: Path | str, *, dir_fd: int | None = None) -> int:
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if no_follow is None:
        raise VideoDemoError(ErrorCode.INVALID_CONFIGURATION, "当前平台不支持安全模型缓存锁")
    if dir_fd is not None and not _OPEN_SUPPORTS_DIR_FD:
        raise VideoDemoError(ErrorCode.INVALID_CONFIGURATION, "当前平台不支持安全相对路径锁")
    descriptor: int | None = None
    try:
        if dir_fd is None:
            descriptor = _create_or_open_lock(path, no_follow=no_follow)
        else:
            descriptor = _create_or_open_lock(
                path,
                no_follow=no_follow,
                dir_fd=dir_fd,
            )
        os.fchmod(descriptor, 0o600)
        status = os.fstat(descriptor)
        if not stat.S_ISREG(status.st_mode):
            raise OSError
        return descriptor
    except OSError as error:
        if descriptor is not None:
            os.close(descriptor)
        raise VideoDemoError(ErrorCode.ARTIFACT_SCHEMA_INVALID, "模型缓存锁文件非法") from error


def _create_or_open_lock(
    path: Path | str,
    *,
    no_follow: int,
    dir_fd: int | None = None,
) -> int:
    try:
        return os.open(
            path,
            os.O_RDWR | os.O_CREAT | os.O_EXCL | no_follow,
            0o600,
            dir_fd=dir_fd,
        )
    except FileExistsError:
        return os.open(path, os.O_RDWR | no_follow, dir_fd=dir_fd)


def _ensure_private_directory(path: Path) -> None:
    descriptor = _open_private_directory(path, invalid_message="模型缓存目录非法")
    os.close(descriptor)


def _open_private_directory(path: Path, *, invalid_message: str) -> int:
    no_follow = getattr(os, "O_NOFOLLOW", None)
    directory_flag = getattr(os, "O_DIRECTORY", None)
    if no_follow is None or directory_flag is None:
        raise VideoDemoError(ErrorCode.INVALID_CONFIGURATION, "当前平台不支持安全缓存目录")
    descriptor: int | None = None
    try:
        path.mkdir(mode=0o700, exist_ok=True)
        descriptor = os.open(path, os.O_RDONLY | directory_flag | no_follow)
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise OSError
        os.fchmod(descriptor, 0o700)
        return descriptor
    except OSError as error:
        if descriptor is not None:
            os.close(descriptor)
        raise VideoDemoError(ErrorCode.ARTIFACT_SCHEMA_INVALID, invalid_message) from error


def _require_directory_path_identity(
    path: Path,
    opened: os.stat_result,
    message: str,
) -> None:
    try:
        current = os.lstat(path)
    except OSError as error:
        raise VideoDemoError(ErrorCode.ARTIFACT_SCHEMA_INVALID, message) from error
    if (
        stat.S_ISLNK(current.st_mode)
        or not stat.S_ISDIR(current.st_mode)
        or (current.st_dev, current.st_ino) != (opened.st_dev, opened.st_ino)
    ):
        raise VideoDemoError(ErrorCode.ARTIFACT_SCHEMA_INVALID, message)


def _write_private_file(path: Path, content: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)


def _read_private_file(path: Path, before: os.stat_result, max_bytes: int) -> bytes:
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if no_follow is None:
        raise VideoDemoError(ErrorCode.INVALID_CONFIGURATION, "当前平台不支持安全模型缓存读取")
    descriptor = os.open(path, os.O_RDONLY | no_follow)
    try:
        opened = os.fstat(descriptor)
        _require_same_file(before, opened)
        chunks: list[bytes] = []
        total = 0
        while chunk := os.read(descriptor, min(_READ_CHUNK_BYTES, max_bytes + 1 - total)):
            total += len(chunk)
            if total > max_bytes:
                raise VideoDemoError(ErrorCode.INPUT_BUDGET_EXCEEDED, "模型缓存条目超过大小上限")
            chunks.append(chunk)
        _require_same_file(opened, os.fstat(descriptor))
        return b"".join(chunks)
    except OSError as error:
        raise VideoDemoError(ErrorCode.ARTIFACT_SCHEMA_INVALID, "模型缓存文件非法") from error
    finally:
        os.close(descriptor)


def _require_same_file(before: os.stat_result, after: os.stat_result) -> None:
    if _file_identity(before) != _file_identity(after):
        raise VideoDemoError(ErrorCode.ARTIFACT_SCHEMA_INVALID, "模型缓存读取期间发生变化")


def _file_identity(value: os.stat_result) -> tuple[int, int, int, int]:
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)


def _require_frozen_model(model: type[FrozenModel], label: str) -> None:
    if not issubclass(model, FrozenModel):
        raise VideoDemoError(ErrorCode.INVALID_CONFIGURATION, f"{label}必须继承 FrozenModel")


def _require_frozen_instance(value: object, label: str) -> None:
    if not isinstance(value, FrozenModel):
        raise VideoDemoError(ErrorCode.INVALID_CONFIGURATION, f"{label}必须继承 FrozenModel")


def _validate_response(
    response: ResponseModel,
    validate: Callable[[ResponseModel], None],
) -> None:
    try:
        validate(response)
    except (ValueError, TypeError, ValidationError):
        raise VideoDemoError(ErrorCode.ARTIFACT_SCHEMA_INVALID, "模型缓存响应引用非法") from None


def _cache_fingerprint(
    identity: ModelInvocationIdentity,
    canonical_input: FrozenModel,
) -> str:
    return hashlib.sha256(
        (_identity_digest(identity) + "\n" + _input_sha256(canonical_input)).encode("utf-8"),
    ).hexdigest()


def _identity_digest(identity: ModelInvocationIdentity) -> str:
    return hashlib.sha256(_canonical_json(identity.model_dump(mode="json"))).hexdigest()


def _input_sha256(canonical_input: FrozenModel) -> str:
    return hashlib.sha256(_canonical_json(canonical_input.model_dump(mode="json"))).hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
