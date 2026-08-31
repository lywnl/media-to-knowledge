from __future__ import annotations

import hashlib
import os
import stat
import uuid
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.media.process import ProcessResult
from video_demo.storage.workspace import safe_runtime_path, validate_path_component

CASE_IDS = ("normal_audio", "no_audio", "rotation", "vfr")
MAX_MEDIA_BYTES = 4 * 1024 * 1024 * 1024
_CHUNK_BYTES = 1024 * 1024


class ProcessPort(Protocol):
    def run(
        self,
        args: Sequence[str],
        *,
        timeout_seconds: int,
        pass_fds: tuple[int, ...] = (),
    ) -> ProcessResult: ...


@dataclass(frozen=True, slots=True)
class GeneratedSource:
    path: Path
    process_result: ProcessResult


@dataclass(frozen=True, slots=True)
class LeafSnapshot:
    device: int
    inode: int
    size: int
    sha256: str


@dataclass(frozen=True, slots=True)
class _NodeIdentity:
    device: int
    inode: int


@dataclass(frozen=True, slots=True)
class _DirectoryLease:
    name: str
    descriptor: int
    identity: _NodeIdentity


@dataclass(slots=True)
class _OpenOutput:
    name: str
    descriptor: int
    identity: _NodeIdentity


@dataclass(slots=True)
class StagedCaseOutput:
    relative_path: Path
    parent_descriptor: int
    temporary_name: str
    descriptor: int
    identity: _NodeIdentity


class CaseDirectoryConflict(VideoDemoError):
    """排他创建前已有 case 根；调用方不得把该根视为本次执行所有。"""


class CaseExecutionSession:
    """持有 case 的完整目录身份，直到样本提交或失败清理结束。"""

    def __init__(
        self,
        runtime_root: Path,
        leases: tuple[_DirectoryLease, ...],
    ) -> None:
        self.runtime_root = runtime_root
        self._leases = leases
        self._registered_identities: dict[Path, _NodeIdentity] = {}
        self._source_descriptor: int | None = None
        self._source_snapshot: LeafSnapshot | None = None
        self._closed = False

    @property
    def case_root(self) -> Path:
        return self.runtime_root.joinpath(*(lease.name for lease in self._leases[1:]))

    @property
    def case_descriptor(self) -> int:
        self._require_open()
        return self._leases[-1].descriptor

    def __enter__(self) -> CaseExecutionSession:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def assert_current(self) -> None:
        """复验词法 runtime 根及每个父子目录条目仍指向持有对象。"""

        self._require_open()
        try:
            root_details = os.stat(self.runtime_root, follow_symlinks=False)
        except OSError:
            self._raise_identity_changed()
        if (
            not stat.S_ISDIR(root_details.st_mode)
            or _node_identity(root_details) != self._leases[0].identity
        ):
            self._raise_identity_changed()
        for parent, child in zip(self._leases, self._leases[1:], strict=False):
            try:
                details = os.stat(
                    child.name,
                    dir_fd=parent.descriptor,
                    follow_symlinks=False,
                )
            except OSError:
                self._raise_identity_changed()
            if (
                not stat.S_ISDIR(details.st_mode)
                or _node_identity(details) != child.identity
            ):
                self._raise_identity_changed()

    def snapshot_leaf(self, relative_path: Path, max_bytes: int) -> LeafSnapshot:
        """从持有 case fd 读取稳定普通文件快照，不解析 case 词法路径。"""

        self._validate_relative_leaf(relative_path)
        descriptor = self._open_leaf(relative_path)
        try:
            return _snapshot_descriptor(descriptor, max_bytes)
        finally:
            os.close(descriptor)

    def open_registered_leaf(self, relative_path: Path) -> int:
        """从 held case fd 打开已登记普通叶，并把 fd 所有权交给调用方。"""

        self._validate_relative_leaf(relative_path)
        expected = self._registered_identities.get(relative_path)
        if expected is None:
            raise VideoDemoError(ErrorCode.SYSTEM_FAILURE, "真实媒体输入尚未登记")
        descriptor = self._open_leaf(relative_path)
        if _node_identity(os.fstat(descriptor)) != expected:
            os.close(descriptor)
            raise VideoDemoError(
                ErrorCode.WORKSPACE_PATH_ESCAPE,
                "真实媒体已登记输入身份已改变",
            )
        return descriptor

    def stage_output(self, relative_path: Path) -> StagedCaseOutput:
        """在 held case fd 下预创建空输出，供生产 adapter 直接写 fd。"""

        self._validate_relative_leaf(relative_path)
        self.assert_current()
        parent = self._open_or_create_parent(relative_path)
        temporary_name = f".{relative_path.name}.{uuid.uuid4().hex}.part"
        flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC
        try:
            descriptor = os.open(temporary_name, flags, 0o600, dir_fd=parent)
            details = os.fstat(descriptor)
            if not stat.S_ISREG(details.st_mode) or details.st_size != 0:
                raise VideoDemoError(ErrorCode.VIDEO_OUTPUT_INVALID, "媒体临时产物创建失败")
            return StagedCaseOutput(
                relative_path=relative_path,
                parent_descriptor=parent,
                temporary_name=temporary_name,
                descriptor=descriptor,
                identity=_node_identity(details),
            )
        except BaseException:
            _unlink_at(parent, temporary_name)
            os.close(parent)
            raise

    def publish_output(
        self,
        output: StagedCaseOutput,
        max_bytes: int,
    ) -> LeafSnapshot:
        """只在 stage 持有的同一目录 fd 内验证并发布输出。"""

        self._require_staged_output(output)
        snapshot = _snapshot_descriptor(output.descriptor, max_bytes)
        published = False
        try:
            details = os.stat(
                output.temporary_name,
                dir_fd=output.parent_descriptor,
                follow_symlinks=False,
            )
            if (
                not stat.S_ISREG(details.st_mode)
                or _node_identity(details) != output.identity
                or _NodeIdentity(snapshot.device, snapshot.inode) != output.identity
            ):
                raise VideoDemoError(ErrorCode.VIDEO_OUTPUT_INVALID, "媒体临时产物身份已改变")
            os.link(
                output.temporary_name,
                output.relative_path.name,
                src_dir_fd=output.parent_descriptor,
                dst_dir_fd=output.parent_descriptor,
                follow_symlinks=False,
            )
            published = True
            target = self.snapshot_leaf(output.relative_path, max_bytes)
            if target != snapshot:
                raise VideoDemoError(ErrorCode.VIDEO_OUTPUT_INVALID, "媒体发布产物身份不一致")
            os.unlink(output.temporary_name, dir_fd=output.parent_descriptor)
            os.fsync(output.parent_descriptor)
            return snapshot
        except FileExistsError:
            raise VideoDemoError(
                ErrorCode.IDEMPOTENCY_CONFLICT,
                "真实媒体输出目标被占用",
            ) from None
        except BaseException:
            if published:
                _unlink_at(output.parent_descriptor, output.relative_path.name)
            raise
        finally:
            self.discard_output(output)

    def discard_output(self, output: StagedCaseOutput) -> None:
        """只清理 capability 自己的临时叶与 fd，可重复调用。"""

        if output.descriptor >= 0:
            with suppress(OSError):
                os.close(output.descriptor)
            output.descriptor = -1
        if output.parent_descriptor >= 0:
            _unlink_at(output.parent_descriptor, output.temporary_name)
            with suppress(OSError):
                os.close(output.parent_descriptor)
            output.parent_descriptor = -1

    def write_output_bytes(
        self,
        relative_path: Path,
        payload: bytes,
        max_bytes: int,
    ) -> LeafSnapshot:
        if not payload or len(payload) > max_bytes:
            raise VideoDemoError(ErrorCode.VISUAL_RESULT_INVALID, "媒体输出正文非法")
        output = self.stage_output(relative_path)
        try:
            view = memoryview(payload)
            written = 0
            while written < len(view):
                count = os.write(output.descriptor, view[written:])
                if count <= 0:
                    raise VideoDemoError(ErrorCode.VIDEO_OUTPUT_INVALID, "媒体输出写入失败")
                written += count
            os.fsync(output.descriptor)
            return self.publish_output(output, max_bytes)
        finally:
            self.discard_output(output)

    def remember_registered_leaf(
        self,
        relative_path: Path,
        snapshot: LeafSnapshot,
    ) -> None:
        self._validate_relative_leaf(relative_path)
        self._registered_identities[relative_path] = _NodeIdentity(
            snapshot.device,
            snapshot.inode,
        )

    def assert_published_source(
        self,
        relative_path: Path,
        observed: LeafSnapshot,
        max_bytes: int,
    ) -> None:
        """确认待绑定 source 仍是本会话发布并持续持有的同一文件。"""

        self._validate_relative_leaf(relative_path)
        if (
            relative_path != Path("source.mp4")
            or self._source_descriptor is None
            or self._source_snapshot is None
        ):
            raise VideoDemoError(ErrorCode.SYSTEM_FAILURE, "真实媒体 source 发布状态缺失")
        held = _snapshot_descriptor(self._source_descriptor, max_bytes)
        if observed != self._source_snapshot or held != self._source_snapshot:
            raise VideoDemoError(
                ErrorCode.WORKSPACE_PATH_ESCAPE,
                "真实媒体 source 叶身份已改变",
            )

    def assert_registered_leaves(self, registered: set[Path]) -> None:
        """阶段边界复验祖先和所有已登记叶的身份。"""

        self.assert_current()
        if set(self._registered_identities) != registered:
            raise VideoDemoError(ErrorCode.SYSTEM_FAILURE, "真实媒体登记身份集合不一致")
        for relative_path in registered:
            identity = self._leaf_identity(relative_path)
            if identity != self._registered_identities[relative_path]:
                raise VideoDemoError(
                    ErrorCode.WORKSPACE_PATH_ESCAPE,
                    "真实媒体已登记叶身份已改变",
                )

    def assert_media_closed(self, registered: set[Path]) -> None:
        """基于 held case fd 审计精确叶集合及全部已登记身份。"""

        self.assert_registered_leaves(registered)
        observed = self._walk_leaf_identities()
        if set(observed) != registered:
            raise VideoDemoError(ErrorCode.VIDEO_OUTPUT_INVALID, "真实媒体 case 存在未登记叶文件")
        for relative_path, identity in observed.items():
            if identity != self._registered_identities[relative_path]:
                raise VideoDemoError(
                    ErrorCode.WORKSPACE_PATH_ESCAPE,
                    "真实媒体已登记叶身份已改变",
                )

    def cleanup_unregistered(self, registered: set[Path]) -> None:
        """只经 held case fd 清理本次根，绝不解析同名词法替换目录。"""

        if self._closed:
            return
        self._cleanup_directory(self.case_descriptor, Path(), registered)

    def remove_leaf(self, relative_path: Path) -> None:
        """经 held case fd 删除尚未转移所有权的普通叶。"""

        self._validate_relative_leaf(relative_path)
        parent = os.dup(self.case_descriptor)
        try:
            for component in relative_path.parts[:-1]:
                child = os.open(component, _directory_open_flags(), dir_fd=parent)
                os.close(parent)
                parent = child
            _unlink_at(parent, relative_path.name)
        finally:
            os.close(parent)

    def _open_or_create_parent(self, relative_path: Path) -> int:
        parent = os.dup(self.case_descriptor)
        try:
            for component in relative_path.parts[:-1]:
                with suppress(FileExistsError):
                    os.mkdir(component, dir_fd=parent)
                child = os.open(component, _directory_open_flags(), dir_fd=parent)
                os.close(parent)
                parent = child
            return parent
        except BaseException:
            os.close(parent)
            raise

    @staticmethod
    def _require_staged_output(output: StagedCaseOutput) -> None:
        if output.descriptor < 0 or output.parent_descriptor < 0:
            raise VideoDemoError(ErrorCode.SYSTEM_FAILURE, "媒体输出 capability 已关闭")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._source_descriptor is not None:
            with suppress(OSError):
                os.close(self._source_descriptor)
            self._source_descriptor = None
        self._source_snapshot = None
        for lease in reversed(self._leases):
            with suppress(OSError):
                os.close(lease.descriptor)

    def _create_output(self, suffix: str) -> _OpenOutput:
        self.assert_current()
        name = f".source.{uuid.uuid4().hex}.{suffix}"
        flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC
        descriptor = os.open(name, flags, 0o600, dir_fd=self.case_descriptor)
        details = os.fstat(descriptor)
        if not stat.S_ISREG(details.st_mode) or details.st_size != 0:
            os.close(descriptor)
            _unlink_at(self.case_descriptor, name)
            raise VideoDemoError(ErrorCode.VIDEO_OUTPUT_INVALID, "媒体临时产物创建失败")
        return _OpenOutput(name, descriptor, _node_identity(details))

    def _validate_output(self, output: _OpenOutput, max_bytes: int) -> LeafSnapshot:
        self.assert_current()
        snapshot = _snapshot_descriptor(output.descriptor, max_bytes)
        try:
            details = os.stat(
                output.name,
                dir_fd=self.case_descriptor,
                follow_symlinks=False,
            )
        except OSError:
            raise VideoDemoError(ErrorCode.VIDEO_OUTPUT_INVALID, "媒体临时产物不存在") from None
        if (
            not stat.S_ISREG(details.st_mode)
            or _node_identity(details) != output.identity
            or _NodeIdentity(snapshot.device, snapshot.inode) != output.identity
        ):
            raise VideoDemoError(ErrorCode.VIDEO_OUTPUT_INVALID, "媒体临时产物身份已改变")
        return snapshot

    def _publish_source(self, output: _OpenOutput, snapshot: LeafSnapshot) -> None:
        target_name = "source.mp4"
        published = False
        self.assert_current()
        try:
            os.link(
                output.name,
                target_name,
                src_dir_fd=self.case_descriptor,
                dst_dir_fd=self.case_descriptor,
                follow_symlinks=False,
            )
            published = True
            target = self.snapshot_leaf(Path(target_name), snapshot.size)
            if target != snapshot:
                raise VideoDemoError(ErrorCode.VIDEO_OUTPUT_INVALID, "真实媒体发布产物身份不一致")
            self.assert_current()
            os.unlink(output.name, dir_fd=self.case_descriptor)
            os.fsync(self.case_descriptor)
            self._source_descriptor = output.descriptor
            self._source_snapshot = snapshot
            output.descriptor = -1
        except FileExistsError:
            raise VideoDemoError(
                ErrorCode.IDEMPOTENCY_CONFLICT,
                "真实媒体 source 目标被占用",
            ) from None
        except BaseException:
            if published:
                _unlink_at(self.case_descriptor, target_name)
            raise

    def _open_leaf(self, relative_path: Path) -> int:
        parent = os.dup(self.case_descriptor)
        try:
            for component in relative_path.parts[:-1]:
                child = os.open(component, _directory_open_flags(), dir_fd=parent)
                os.close(parent)
                parent = child
            flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK
            descriptor = os.open(relative_path.name, flags, dir_fd=parent)
        finally:
            os.close(parent)
        try:
            if stat.S_ISREG(os.fstat(descriptor).st_mode):
                return descriptor
            raise VideoDemoError(ErrorCode.VIDEO_OUTPUT_INVALID, "媒体产物不是普通文件")
        except BaseException:
            os.close(descriptor)
            raise

    def _leaf_identity(self, relative_path: Path) -> _NodeIdentity:
        descriptor = self._open_leaf(relative_path)
        try:
            return _node_identity(os.fstat(descriptor))
        finally:
            os.close(descriptor)

    def _walk_leaf_identities(self) -> dict[Path, _NodeIdentity]:
        observed: dict[Path, _NodeIdentity] = {}

        def walk(descriptor: int, relative_root: Path) -> None:
            for name in os.listdir(descriptor):
                # 候选帧会话的锁文件是并发控制元数据，不属于媒体制品。
                if relative_root == Path("visual") and name == ".candidates.lock":
                    continue
                details = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                relative = relative_root / name
                if stat.S_ISDIR(details.st_mode):
                    child = os.open(name, _directory_open_flags(), dir_fd=descriptor)
                    try:
                        walk(child, relative)
                    finally:
                        os.close(child)
                else:
                    observed[relative] = _node_identity(details)

        walk(self.case_descriptor, Path())
        return observed

    def _cleanup_directory(
        self,
        descriptor: int,
        relative_root: Path,
        registered: set[Path],
    ) -> None:
        try:
            names = os.listdir(descriptor)
        except OSError:
            return
        for name in names:
            relative = relative_root / name
            if relative_root == Path("visual") and name == ".candidates.lock":
                continue
            try:
                details = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            except OSError:
                continue
            if stat.S_ISDIR(details.st_mode):
                try:
                    child = os.open(name, _directory_open_flags(), dir_fd=descriptor)
                except OSError:
                    continue
                try:
                    self._cleanup_directory(child, relative, registered)
                finally:
                    os.close(child)
                with suppress(OSError):
                    os.rmdir(name, dir_fd=descriptor)
            elif relative not in registered:
                _unlink_at(descriptor, name)

    @staticmethod
    def _validate_relative_leaf(relative_path: Path) -> None:
        if (
            not relative_path.parts
            or relative_path.is_absolute()
            or any(part in {"", ".", ".."} for part in relative_path.parts)
        ):
            raise VideoDemoError(ErrorCode.WORKSPACE_PATH_ESCAPE, "真实媒体叶路径非法")

    def _require_open(self) -> None:
        if self._closed:
            raise VideoDemoError(ErrorCode.SYSTEM_FAILURE, "真实媒体 case 会话已关闭")

    @staticmethod
    def _raise_identity_changed() -> None:
        raise VideoDemoError(
            ErrorCode.WORKSPACE_PATH_ESCAPE,
            "真实媒体 case 完整祖先身份已改变",
        )


def open_case_execution_session(
    runtime_root: Path,
    evaluation_run_id: str,
    case_id: str,
) -> CaseExecutionSession:
    """排他创建 case，并把完整祖先 fd 的所有权交给执行会话。"""

    if case_id not in CASE_IDS:
        raise VideoDemoError(ErrorCode.INVALID_CONFIGURATION, "真实媒体 case 非法")
    validate_path_component(evaluation_run_id, "evaluation_run_id")
    _require_posix_fd_capabilities()
    relative_root = Path("eval/generated") / evaluation_run_id / case_id
    safe_runtime_path(runtime_root, relative_root)
    root_descriptor = _open_directory_descriptor(runtime_root)
    leases = [
        _DirectoryLease(
            name=runtime_root.name,
            descriptor=root_descriptor,
            identity=_node_identity(os.fstat(root_descriptor)),
        )
    ]
    case_created = False
    case_parent_descriptor: int | None = None
    try:
        for component in ("eval", "generated", evaluation_run_id):
            with suppress(FileExistsError):
                os.mkdir(component, dir_fd=leases[-1].descriptor)
            child = _open_child_directory(leases[-1].descriptor, component)
            leases.append(
                _DirectoryLease(component, child, _node_identity(os.fstat(child)))
            )
        case_parent_descriptor = leases[-1].descriptor
        try:
            os.mkdir(case_id, dir_fd=case_parent_descriptor)
            case_created = True
        except FileExistsError:
            raise CaseDirectoryConflict(
                ErrorCode.IDEMPOTENCY_CONFLICT,
                "真实媒体 case 目录已存在",
            ) from None
        case_descriptor = _open_child_directory(case_parent_descriptor, case_id)
        leases.append(
            _DirectoryLease(
                case_id,
                case_descriptor,
                _node_identity(os.fstat(case_descriptor)),
            )
        )
        session = CaseExecutionSession(runtime_root, tuple(leases))
        session.assert_current()
        return session
    except BaseException as error:
        if case_created and case_parent_descriptor is not None:
            with suppress(OSError):
                os.rmdir(case_id, dir_fd=case_parent_descriptor)
        for lease in reversed(leases):
            with suppress(OSError):
                os.close(lease.descriptor)
        if isinstance(error, OSError):
            raise VideoDemoError(
                ErrorCode.WORKSPACE_PATH_ESCAPE,
                "真实媒体 case 目录安全创建失败",
            ) from None
        raise


def generate_source(
    *,
    session: CaseExecutionSession,
    case_id: str,
    executable: Path,
    runner: ProcessPort,
    max_bytes: int,
    timeout_seconds: int,
) -> GeneratedSource:
    """让 ffmpeg 只写继承的普通文件 fd，再从 held case fd 原子发布 source。"""

    if case_id not in CASE_IDS or max_bytes < 1 or timeout_seconds < 1:
        raise VideoDemoError(ErrorCode.INVALID_CONFIGURATION, "真实媒体生成参数非法")
    limit = min(max_bytes, MAX_MEDIA_BYTES)
    target = session.case_root / "source.mp4"
    output = session._create_output("part.mp4")
    intermediate: _OpenOutput | None = None
    try:
        if case_id == "rotation":
            intermediate = output
            first = runner.run(
                _source_argv(
                    case_id,
                    executable,
                    _fd_path(intermediate.descriptor),
                    limit,
                    rotation_base=True,
                ),
                timeout_seconds=timeout_seconds,
                pass_fds=(intermediate.descriptor,),
            )
            if first.returncode != 0:
                return GeneratedSource(path=target, process_result=first)
            session._validate_output(intermediate, limit)
            os.lseek(intermediate.descriptor, 0, os.SEEK_SET)
            output = session._create_output("part.mp4")
            result = runner.run(
                _rotation_remux_argv(
                    executable,
                    _fd_path(intermediate.descriptor),
                    _fd_path(output.descriptor),
                    limit,
                ),
                timeout_seconds=timeout_seconds,
                pass_fds=(intermediate.descriptor, output.descriptor),
            )
        else:
            result = runner.run(
                _source_argv(
                    case_id,
                    executable,
                    _fd_path(output.descriptor),
                    limit,
                ),
                timeout_seconds=timeout_seconds,
                pass_fds=(output.descriptor,),
            )
        if result.returncode != 0:
            return GeneratedSource(path=target, process_result=result)
        snapshot = session._validate_output(output, limit)
        session._publish_source(output, snapshot)
        session.assert_current()
        return GeneratedSource(path=target, process_result=result)
    except BaseException:
        _unlink_at(session.case_descriptor, "source.mp4")
        raise
    finally:
        handled_outputs: set[int] = set()
        for current in (output, intermediate):
            if current is None or id(current) in handled_outputs:
                continue
            handled_outputs.add(id(current))
            if current.descriptor >= 0:
                with suppress(OSError):
                    os.close(current.descriptor)
                current.descriptor = -1
            _unlink_at(session.case_descriptor, current.name)


def regular_file_size(path: Path, max_bytes: int) -> int:
    if path.is_symlink():
        raise VideoDemoError(ErrorCode.WORKSPACE_PATH_ESCAPE, "媒体产物不能是符号链接")
    try:
        details = path.stat()
    except OSError:
        raise VideoDemoError(ErrorCode.VIDEO_OUTPUT_INVALID, "媒体产物不存在") from None
    if not stat.S_ISREG(details.st_mode) or details.st_size < 1:
        raise VideoDemoError(ErrorCode.VIDEO_OUTPUT_INVALID, "媒体产物不是非空普通文件")
    if details.st_size > max_bytes:
        raise VideoDemoError(ErrorCode.VIDEO_OUTPUT_TOO_LARGE, "媒体产物超过大小限制")
    return details.st_size


def _source_argv(
    case_id: str,
    executable: Path,
    output: str,
    limit: int,
    *,
    rotation_base: bool = False,
) -> list[str]:
    args = [
        str(executable),
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        "testsrc2=size=640x360:rate=30:duration=2",
    ]
    if case_id != "no_audio":
        args.extend(
            ["-f", "lavfi", "-i", "sine=frequency=1000:sample_rate=48000:duration=2"]
        )
    args.extend(["-map", "0:v:0"])
    if case_id != "no_audio":
        args.extend(["-map", "1:a:0", "-c:a", "aac", "-shortest"])
    else:
        args.append("-an")
    if case_id == "vfr":
        args.extend(["-vf", "select=not(eq(mod(n\\,3)\\,0))", "-fps_mode", "vfr"])
    if case_id == "rotation" and not rotation_base:
        args.extend(["-metadata:s:v:0", "rotate=90"])
    args.extend(
        [
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+frag_keyframe+empty_moov+delay_moov",
            "-y",
            "-fs",
            str(limit),
            "-f",
            "mp4",
            output,
        ]
    )
    return args


def _rotation_remux_argv(
    executable: Path,
    source: str,
    output: str,
    limit: int,
) -> list[str]:
    return [
        str(executable),
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "mp4",
        "-i",
        source,
        "-map",
        "0",
        "-c",
        "copy",
        "-metadata:s:v:0",
        "rotate=90",
        "-movflags",
        "+frag_keyframe+empty_moov+delay_moov",
        "-y",
        "-fs",
        str(limit),
        "-f",
        "mp4",
        output,
    ]


def _require_posix_fd_capabilities() -> None:
    dir_fd_functions = {os.open, os.mkdir, os.stat, os.link, os.unlink, os.rmdir}
    if (
        os.name != "posix"
        or not all(hasattr(os, name) for name in ("O_DIRECTORY", "O_NOFOLLOW", "O_CLOEXEC"))
        or not dir_fd_functions.issubset(os.supports_dir_fd)
        or os.listdir not in os.supports_fd
        or os.stat not in os.supports_follow_symlinks
        or os.link not in os.supports_follow_symlinks
    ):
        raise VideoDemoError(
            ErrorCode.INVALID_CONFIGURATION,
            "当前平台缺少真实媒体 fd 安全能力",
        )


def _snapshot_descriptor(descriptor: int, max_bytes: int) -> LeafSnapshot:
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode) or before.st_size < 1:
        raise VideoDemoError(ErrorCode.VIDEO_OUTPUT_INVALID, "媒体产物不是非空普通文件")
    if before.st_size > max_bytes:
        raise VideoDemoError(ErrorCode.VIDEO_OUTPUT_TOO_LARGE, "媒体产物超过大小限制")
    digest = hashlib.sha256()
    offset = 0
    while offset < before.st_size:
        chunk = os.pread(descriptor, min(_CHUNK_BYTES, before.st_size - offset), offset)
        if not chunk:
            break
        digest.update(chunk)
        offset += len(chunk)
    after = os.fstat(descriptor)
    if _leaf_metadata(before) != _leaf_metadata(after) or offset != before.st_size:
        raise VideoDemoError(ErrorCode.VIDEO_OUTPUT_INVALID, "媒体产物读取期间发生变化")
    return LeafSnapshot(
        device=before.st_dev,
        inode=before.st_ino,
        size=before.st_size,
        sha256=digest.hexdigest(),
    )


def _open_directory_descriptor(path: Path) -> int:
    absolute = path.absolute()
    descriptor = os.open(absolute.anchor, _directory_open_flags())
    try:
        for component in absolute.parts[1:]:
            child = os.open(component, _directory_open_flags(), dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _open_child_directory(parent_descriptor: int, name: str) -> int:
    return os.open(name, _directory_open_flags(), dir_fd=parent_descriptor)


def _directory_open_flags() -> int:
    return os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW


def _fd_path(descriptor: int) -> str:
    return f"/dev/fd/{descriptor}"


def _unlink_at(descriptor: int, name: str) -> None:
    with suppress(IsADirectoryError, OSError):
        os.unlink(name, dir_fd=descriptor)


def _node_identity(details: os.stat_result) -> _NodeIdentity:
    return _NodeIdentity(device=details.st_dev, inode=details.st_ino)


def _leaf_metadata(details: os.stat_result) -> tuple[int, int, int, int]:
    return details.st_dev, details.st_ino, details.st_size, details.st_mtime_ns
