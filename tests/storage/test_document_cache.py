from __future__ import annotations

import json
import multiprocessing
import os
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from threading import Event

import pytest
from pydantic import BaseModel, Field

from video_demo.domain.base import FrozenModel
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.storage import document_cache as document_cache_module
from video_demo.storage.document_cache import (
    CachedModelResult,
    DocumentModelCache,
    ModelInvocationIdentity,
)


class _CanonicalInput(FrozenModel):
    segment_ids: tuple[str, ...] = Field(min_length=1)


class _CachedResponse(FrozenModel):
    chapter_drafts: tuple[str, ...] = ()


def _identity(
    *,
    model_id: str = "text-model-v1",
    main_prompt: str = "chapter-planner-v1",
    main_schema: str = "chapter_planning_v1",
) -> ModelInvocationIdentity:
    return ModelInvocationIdentity(
        logical_operation="chapter_planning",
        provider_config_fingerprint="a" * 64,
        model_id=model_id,
        generation_config=(("temperature", "0"),),
        main_response_schema_name=main_schema,
        main_prompt_version=main_prompt,
        repair_response_schema_name="chapter_planning_repair_v1",
        repair_prompt_version="chapter-planner-repair-v1",
    )


def _input(segment: str = "segment_001") -> _CanonicalInput:
    return _CanonicalInput(segment_ids=(segment,))


def _validate_known_chapter(response: _CachedResponse) -> None:
    if any(item != "chapter_001" for item in response.chapter_drafts):
        raise ValueError("未知章节")


def _multiprocess_singleflight_worker(
    run_root: str,
    start_event: object,
    result_queue: object,
) -> None:
    """spawn 进程入口必须位于模块顶层，避免子进程无法反序列化。"""

    start = start_event
    queue = result_queue
    assert hasattr(start, "wait")
    assert hasattr(queue, "put")
    try:
        if not start.wait(timeout=5):
            raise TimeoutError("等待启动信号超时")
        root = Path(run_root)
        cache = DocumentModelCache(root, max_entry_bytes=4096, max_run_bytes=16_384)
        with cache.invocation_lock(
            _identity(),
            _input(),
            wait_timeout_seconds=5,
            is_cancel_requested=lambda: False,
            poll_interval_seconds=0.01,
        ):
            cached = cache.get(
                _identity(),
                _input(),
                _CachedResponse,
                _validate_known_chapter,
            )
            if cached is None:
                marker = root / "provider-calls.marker"
                descriptor = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
                try:
                    os.write(descriptor, f"{os.getpid()}\n".encode("ascii"))
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
                time.sleep(0.05)
                cached = cache.put(
                    _identity(),
                    _input(),
                    _CachedResponse(chapter_drafts=("chapter_001",)),
                    successful_path="MAIN",
                    validate=_validate_known_chapter,
                )
        queue.put(("ok", tuple(cached.response.chapter_drafts)))
    except BaseException as error:
        queue.put(("error", type(error).__name__, str(error)))


def test_cache_roundtrip_returns_successful_path_and_private_envelope(tmp_path: Path) -> None:
    cache = DocumentModelCache(tmp_path, max_entry_bytes=4096, max_run_bytes=16_384)
    response = _CachedResponse(chapter_drafts=("chapter_001",))

    first = cache.put(
        _identity(),
        _input(),
        response,
        successful_path="REPAIR",
        validate=_validate_known_chapter,
    )
    second = cache.get(
        _identity(),
        _input(),
        _CachedResponse,
        _validate_known_chapter,
    )

    expected: CachedModelResult[_CachedResponse] = CachedModelResult(
        response=response,
        successful_path="REPAIR",
    )
    assert first == second == expected
    cache_file = next((tmp_path / "model-cache" / "chapter_planning").glob("*.json"))
    envelope = json.loads(cache_file.read_text(encoding="utf-8"))
    assert cache_file.stat().st_mode & 0o777 == 0o600
    assert cache_file.parent.stat().st_mode & 0o777 == 0o700
    assert set(envelope) == {
        "identity_digest",
        "input_sha256",
        "response",
        "successful_path",
    }
    assert "segment_001" not in cache_file.read_text(encoding="utf-8")


def test_cache_get_resolves_path_only_while_shared_run_lock_is_held(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = DocumentModelCache(tmp_path, max_entry_bytes=4096, max_run_bytes=16_384)
    original_run_lock = cache._run_lock
    original_path = cache._path
    lock_held = False

    @contextmanager
    def tracked_run_lock(*, exclusive: bool) -> Iterator[None]:
        nonlocal lock_held
        assert not exclusive
        with original_run_lock(exclusive=exclusive):
            lock_held = True
            try:
                yield
            finally:
                lock_held = False

    def assert_locked_path(
        identity: ModelInvocationIdentity,
        canonical_input: FrozenModel,
    ) -> Path:
        assert lock_held
        return original_path(identity, canonical_input)

    monkeypatch.setattr(cache, "_run_lock", tracked_run_lock)
    monkeypatch.setattr(cache, "_path", assert_locked_path)

    assert cache.get(
        _identity(),
        _input(),
        _CachedResponse,
        _validate_known_chapter,
    ) is None


def test_cache_rejects_non_frozen_canonical_input_at_runtime(tmp_path: Path) -> None:
    class MutableInput(BaseModel):
        segment_ids: tuple[str, ...]

    cache = DocumentModelCache(tmp_path, max_entry_bytes=4096, max_run_bytes=16_384)
    with pytest.raises(VideoDemoError) as raised:
        cache.get(
            _identity(),
            MutableInput(segment_ids=("segment_001",)),  # type: ignore[arg-type]
            _CachedResponse,
            _validate_known_chapter,
        )
    assert raised.value.code == ErrorCode.INVALID_CONFIGURATION


def test_cache_identity_changes_for_model_prompt_and_schema(tmp_path: Path) -> None:
    cache = DocumentModelCache(tmp_path, max_entry_bytes=4096, max_run_bytes=65_536)
    response = _CachedResponse(chapter_drafts=("chapter_001",))
    identities = (
        _identity(),
        _identity(model_id="text-model-v2"),
        _identity(main_prompt="chapter-planner-v2"),
        _identity(main_schema="chapter_planning_v2"),
    )

    for identity in identities:
        cache.put(
            identity,
            _input(),
            response,
            successful_path="MAIN",
            validate=_validate_known_chapter,
        )

    assert len(list((tmp_path / "model-cache" / "chapter_planning").glob("*.json"))) == 4


def test_cache_identity_requires_canonical_generation_config() -> None:
    with pytest.raises(ValueError, match="generation_config"):
        ModelInvocationIdentity(
            logical_operation="chapter_planning",
            provider_config_fingerprint="a" * 64,
            model_id="text-model",
            generation_config=(("temperature", "0"), ("alpha", "1")),
            main_response_schema_name="chapter_planning_v1",
            main_prompt_version="chapter-planner-v1",
            repair_response_schema_name="chapter_planning_repair_v1",
            repair_prompt_version="chapter-planner-repair-v1",
        )


def test_cache_revalidates_pydantic_and_cross_object_references(tmp_path: Path) -> None:
    cache = DocumentModelCache(tmp_path, max_entry_bytes=4096, max_run_bytes=16_384)
    cache.put(
        _identity(),
        _input(),
        _CachedResponse(chapter_drafts=("chapter_001",)),
        successful_path="MAIN",
        validate=_validate_known_chapter,
    )
    cache_file = next((tmp_path / "model-cache" / "chapter_planning").glob("*.json"))
    envelope = json.loads(cache_file.read_text(encoding="utf-8"))
    envelope["response"] = {"chapter_drafts": ["chapter_unknown"]}
    cache_file.write_text(json.dumps(envelope), encoding="utf-8")

    with pytest.raises(VideoDemoError) as raised:
        cache.get(
            _identity(),
            _input(),
            _CachedResponse,
            _validate_known_chapter,
        )
    assert raised.value.code == ErrorCode.ARTIFACT_SCHEMA_INVALID


def test_cache_rejects_symlink_and_oversized_entry(tmp_path: Path) -> None:
    cache = DocumentModelCache(tmp_path, max_entry_bytes=4096, max_run_bytes=16_384)
    cache.put(
        _identity(),
        _input(),
        _CachedResponse(chapter_drafts=("chapter_001",)),
        successful_path="MAIN",
        validate=_validate_known_chapter,
    )
    cache_file = next((tmp_path / "model-cache" / "chapter_planning").glob("*.json"))
    payload = cache_file.read_bytes()
    cache_file.unlink()
    outside = tmp_path / "outside.json"
    outside.write_bytes(payload)
    cache_file.symlink_to(outside)

    with pytest.raises(VideoDemoError) as raised:
        cache.get(_identity(), _input(), _CachedResponse, _validate_known_chapter)
    assert raised.value.code == ErrorCode.ARTIFACT_SCHEMA_INVALID

    small = DocumentModelCache(tmp_path / "small", max_entry_bytes=20, max_run_bytes=100)
    with pytest.raises(VideoDemoError) as raised:
        small.put(
            _identity(),
            _input(),
            _CachedResponse(chapter_drafts=("chapter_001",)),
            successful_path="MAIN",
            validate=_validate_known_chapter,
        )
    assert raised.value.code == ErrorCode.INPUT_BUDGET_EXCEEDED


def test_cache_rejects_symlinked_operation_directory_without_touching_target(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir(mode=0o755)
    cache_root = tmp_path / "run" / "model-cache"
    cache_root.mkdir(parents=True)
    (cache_root / "chapter_planning").symlink_to(outside)
    cache = DocumentModelCache(
        tmp_path / "run",
        max_entry_bytes=4096,
        max_run_bytes=16_384,
    )

    with pytest.raises(VideoDemoError) as raised:
        cache.put(
            _identity(),
            _input(),
            _CachedResponse(chapter_drafts=("chapter_001",)),
            successful_path="MAIN",
            validate=_validate_known_chapter,
        )
    assert raised.value.code == ErrorCode.ARTIFACT_SCHEMA_INVALID
    assert outside.stat().st_mode & 0o777 == 0o755
    assert tuple(outside.iterdir()) == ()


def test_same_fingerprint_concurrent_put_reuses_single_immutable_entry(tmp_path: Path) -> None:
    caches = [
        DocumentModelCache(tmp_path, max_entry_bytes=4096, max_run_bytes=16_384)
        for _ in range(8)
    ]

    def write(cache: DocumentModelCache) -> CachedModelResult[_CachedResponse]:
        return cache.put(
            _identity(),
            _input(),
            _CachedResponse(chapter_drafts=("chapter_001",)),
            successful_path="MAIN",
            validate=_validate_known_chapter,
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = tuple(executor.map(write, caches))

    assert all(result == results[0] for result in results)
    assert len(list((tmp_path / "model-cache" / "chapter_planning").glob("*.json"))) == 1


def test_different_fingerprints_concurrently_respect_run_budget(tmp_path: Path) -> None:
    probe_root = tmp_path / "probe"
    probe = DocumentModelCache(probe_root, max_entry_bytes=4096, max_run_bytes=16_384)
    probe.put(
        _identity(),
        _input(),
        _CachedResponse(chapter_drafts=("chapter_001",)),
        successful_path="MAIN",
        validate=_validate_known_chapter,
    )
    entry_size = next((probe_root / "model-cache").glob("*/*.json")).stat().st_size
    run_root = tmp_path / "budget"
    caches = [
        DocumentModelCache(run_root, max_entry_bytes=4096, max_run_bytes=entry_size)
        for _ in range(2)
    ]

    def write(index: int) -> ErrorCode | None:
        try:
            caches[index].put(
                _identity(model_id=f"text-model-{index}"),
                _input(f"segment_{index:03d}"),
                _CachedResponse(chapter_drafts=("chapter_001",)),
                successful_path="MAIN",
                validate=_validate_known_chapter,
            )
        except VideoDemoError as error:
            return error.code
        return None

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(executor.map(write, range(2)))

    assert sorted(code.value if code else "OK" for code in results) == [
        ErrorCode.INPUT_BUDGET_EXCEEDED.value,
        "OK",
    ]
    assert len(list((run_root / "model-cache").glob("*/*.json"))) == 1
    assert (run_root / ".model-cache.lock").stat().st_mode & 0o777 == 0o600


def test_invocation_lock_serializes_same_fingerprint_and_requires_recheck(
    tmp_path: Path,
) -> None:
    caches = [
        DocumentModelCache(tmp_path, max_entry_bytes=4096, max_run_bytes=16_384)
        for _ in range(2)
    ]
    provider_calls = 0

    def logical_call(cache: DocumentModelCache) -> _CachedResponse:
        nonlocal provider_calls
        with cache.invocation_lock(
            _identity(),
            _input(),
            wait_timeout_seconds=2,
            is_cancel_requested=lambda: False,
            poll_interval_seconds=0.01,
        ):
            cached = cache.get(
                _identity(),
                _input(),
                _CachedResponse,
                _validate_known_chapter,
            )
            if cached is not None:
                return cached.response
            provider_calls += 1
            result = cache.put(
                _identity(),
                _input(),
                _CachedResponse(chapter_drafts=("chapter_001",)),
                successful_path="MAIN",
                validate=_validate_known_chapter,
            )
            return result.response

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(executor.map(logical_call, caches))

    assert provider_calls == 1
    assert results == (
        _CachedResponse(chapter_drafts=("chapter_001",)),
        _CachedResponse(chapter_drafts=("chapter_001",)),
    )


def test_invocation_lock_singleflight_works_across_spawned_processes(
    tmp_path: Path,
) -> None:
    context = multiprocessing.get_context("spawn")
    start_event = context.Event()
    result_queue = context.Queue()
    processes = tuple(
        context.Process(
            target=_multiprocess_singleflight_worker,
            args=(str(tmp_path), start_event, result_queue),
        )
        for _ in range(2)
    )
    hung_process_ids: list[int | None] = []
    try:
        for process in processes:
            process.start()
        start_event.set()
        results = tuple(result_queue.get(timeout=8) for _ in processes)
    finally:
        for process in processes:
            process.join(timeout=8)
            if process.is_alive():
                hung_process_ids.append(process.pid)
                process.terminate()
                process.join(timeout=2)
        result_queue.close()
        result_queue.join_thread()

    assert not hung_process_ids, f"模型缓存子进程未在期限内退出：{hung_process_ids}"
    assert all(process.exitcode == 0 for process in processes)

    assert all(result == ("ok", ("chapter_001",)) for result in results)
    provider_calls = (tmp_path / "provider-calls.marker").read_text(
        encoding="ascii",
    ).splitlines()
    assert len(provider_calls) == 1


def test_invocation_lock_wait_is_cancelable_and_bounded(tmp_path: Path) -> None:
    holder = DocumentModelCache(tmp_path, max_entry_bytes=4096, max_run_bytes=16_384)
    waiter = DocumentModelCache(tmp_path, max_entry_bytes=4096, max_run_bytes=16_384)
    entered = Event()
    release = Event()

    def hold() -> None:
        with holder.invocation_lock(
            _identity(),
            _input(),
            wait_timeout_seconds=2,
            is_cancel_requested=lambda: False,
            poll_interval_seconds=0.01,
        ):
            entered.set()
            release.wait(timeout=2)

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(hold)
        assert entered.wait(timeout=1)
        with pytest.raises(VideoDemoError) as raised, waiter.invocation_lock(
            _identity(),
            _input(),
            wait_timeout_seconds=0.02,
            is_cancel_requested=lambda: False,
            poll_interval_seconds=0.005,
        ):
            pass
        assert raised.value.code == ErrorCode.DEPENDENCY_TEMPORARY_FAILURE

        with pytest.raises(VideoDemoError) as raised, waiter.invocation_lock(
            _identity(),
            _input(),
            wait_timeout_seconds=2,
            is_cancel_requested=lambda: True,
            poll_interval_seconds=0.005,
        ):
            pass
        assert raised.value.code == ErrorCode.JOB_CANCELLED
        release.set()
        future.result(timeout=1)


def test_invocation_lock_allows_different_fingerprints_to_run_concurrently(
    tmp_path: Path,
) -> None:
    caches = [
        DocumentModelCache(tmp_path, max_entry_bytes=4096, max_run_bytes=16_384)
        for _ in range(2)
    ]
    first_entered = Event()
    second_entered = Event()
    release = Event()

    def hold(index: int) -> None:
        with caches[index].invocation_lock(
            _identity(model_id=f"text-model-{index}"),
            _input(),
            wait_timeout_seconds=2,
            is_cancel_requested=lambda: False,
            poll_interval_seconds=0.01,
        ):
            (first_entered if index == 0 else second_entered).set()
            release.wait(timeout=2)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = (executor.submit(hold, 0), executor.submit(hold, 1))
        assert first_entered.wait(timeout=1)
        assert second_entered.wait(timeout=1)
        release.set()
        for future in futures:
            future.result(timeout=1)


def test_invocation_lock_releases_after_provider_failure(tmp_path: Path) -> None:
    cache = DocumentModelCache(tmp_path, max_entry_bytes=4096, max_run_bytes=16_384)

    with pytest.raises(RuntimeError, match="provider failed"), cache.invocation_lock(
        _identity(),
        _input(),
        wait_timeout_seconds=1,
        is_cancel_requested=lambda: False,
    ):
        raise RuntimeError("provider failed")

    with cache.invocation_lock(
        _identity(),
        _input(),
        wait_timeout_seconds=1,
        is_cancel_requested=lambda: False,
    ):
        pass
    lock_file = next((tmp_path / ".model-invocation-locks").glob("*.lock"))
    assert lock_file.stat().st_mode & 0o777 == 0o600
    assert lock_file.parent.stat().st_mode & 0o777 == 0o700


def test_invocation_lock_rejects_symlinked_lock_file(tmp_path: Path) -> None:
    cache = DocumentModelCache(tmp_path, max_entry_bytes=4096, max_run_bytes=16_384)
    with cache.invocation_lock(
        _identity(),
        _input(),
        wait_timeout_seconds=1,
        is_cancel_requested=lambda: False,
    ):
        pass
    lock_file = next((tmp_path / ".model-invocation-locks").glob("*.lock"))
    lock_file.unlink()
    outside = tmp_path / "outside.lock"
    outside.write_bytes(b"")
    lock_file.symlink_to(outside)

    with pytest.raises(VideoDemoError) as raised, cache.invocation_lock(
        _identity(),
        _input(),
        wait_timeout_seconds=1,
        is_cancel_requested=lambda: False,
    ):
        pass
    assert raised.value.code == ErrorCode.ARTIFACT_SCHEMA_INVALID


def test_invocation_lock_rejects_symlinked_parent_without_touching_target(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir(mode=0o755)
    lock_root = tmp_path / ".model-invocation-locks"
    lock_root.symlink_to(outside, target_is_directory=True)
    cache = DocumentModelCache(tmp_path, max_entry_bytes=4096, max_run_bytes=16_384)

    with pytest.raises(VideoDemoError) as raised, cache.invocation_lock(
        _identity(),
        _input(),
        wait_timeout_seconds=1,
        is_cancel_requested=lambda: False,
    ):
        pass

    assert raised.value.code == ErrorCode.ARTIFACT_SCHEMA_INVALID
    assert outside.stat().st_mode & 0o777 == 0o755
    assert tuple(outside.iterdir()) == ()


def test_invocation_lock_rejects_parent_replaced_while_opening_leaf(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock_root = tmp_path / ".model-invocation-locks"
    lock_root.mkdir(mode=0o700)
    displaced = tmp_path / ".model-invocation-locks.displaced"
    outside = tmp_path / "outside"
    outside.mkdir(mode=0o755)
    original_open = os.open
    replaced = False

    def racing_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal replaced
        if isinstance(path, (bytes, os.PathLike)):
            candidate = Path(os.fsdecode(path))
        else:
            candidate = Path(path)
        opens_path_leaf = dir_fd is None and candidate.parent == lock_root
        opens_dir_fd_leaf = dir_fd is not None and len(candidate.parts) == 1
        if not replaced and candidate.suffix == ".lock" and (
            opens_path_leaf or opens_dir_fd_leaf
        ):
            lock_root.rename(displaced)
            lock_root.symlink_to(outside, target_is_directory=True)
            replaced = True
        if dir_fd is None:
            return original_open(path, flags, mode)
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", racing_open)
    cache = DocumentModelCache(tmp_path, max_entry_bytes=4096, max_run_bytes=16_384)

    with pytest.raises(VideoDemoError) as raised, cache.invocation_lock(
        _identity(),
        _input(),
        wait_timeout_seconds=1,
        is_cancel_requested=lambda: False,
    ):
        pass

    assert replaced
    assert raised.value.code == ErrorCode.ARTIFACT_SCHEMA_INVALID
    assert tuple(outside.iterdir()) == ()


def test_invocation_lock_requires_safe_relative_open_support(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(document_cache_module, "_OPEN_SUPPORTS_DIR_FD", False)
    cache = DocumentModelCache(tmp_path, max_entry_bytes=4096, max_run_bytes=16_384)

    with pytest.raises(VideoDemoError) as raised, cache.invocation_lock(
        _identity(),
        _input(),
        wait_timeout_seconds=1,
        is_cancel_requested=lambda: False,
    ):
        pass

    assert raised.value.code == ErrorCode.INVALID_CONFIGURATION
