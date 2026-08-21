from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import Field

from video_demo.domain.base import FrozenModel
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.storage.artifacts import AtomicArtifactStore
from video_demo.storage.snapshots import SnapshotStore


class Payload(FrozenModel):
    schema_version: str = "1.0.0"
    value: str = Field(min_length=1)


def test_snapshot_publish_uses_envelope_sha_filename_and_loads_verified_payload(
    tmp_path: Path,
) -> None:
    store = SnapshotStore(AtomicArtifactStore(tmp_path))
    run_root = Path("runs/scope/run_001")
    fingerprint = "a" * 64

    receipt = store.publish(run_root, "asr", fingerprint, Payload(value="first"))
    loaded = store.load(run_root, "asr", fingerprint, Payload)

    assert receipt.relative_path.endswith(f"payload-{receipt.sha256}.json")
    assert loaded == (Payload(value="first"), receipt)


@pytest.mark.parametrize(
    "mutation",
    ["pointer_digest", "payload_digest", "kind", "schema"],
)
def test_snapshot_load_treats_corrupted_state_as_cache_miss(
    tmp_path: Path,
    mutation: str,
) -> None:
    atomic = AtomicArtifactStore(tmp_path)
    store = SnapshotStore(atomic)
    run_root = Path("runs/scope/run_001")
    fingerprint = "a" * 64
    receipt = store.publish(run_root, "asr", fingerprint, Payload(value="first"))
    pointer_path = tmp_path / run_root / "speech/snapshots/asr-current.json"
    payload_path = tmp_path / receipt.relative_path
    if mutation == "pointer_digest":
        pointer_path.write_bytes(pointer_path.read_bytes() + b" ")
    elif mutation == "payload_digest":
        payload_path.write_bytes(payload_path.read_bytes() + b" ")
    elif mutation == "kind":
        envelope = json.loads(pointer_path.read_text(encoding="utf-8"))
        envelope["payload"]["kind"] = "speech"
        pointer_path.write_text(
            json.dumps(envelope, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
    else:
        envelope = json.loads(pointer_path.read_text(encoding="utf-8"))
        envelope["payload"]["schema_version"] = "9.0.0"
        _rewrite_pointer(atomic, run_root, fingerprint, envelope["payload"])

    assert store.load(run_root, "asr", fingerprint, Payload) is None


@pytest.mark.parametrize("mutation", ["outside", "absolute", "symlink"])
def test_snapshot_load_fails_closed_for_path_security_violations(
    tmp_path: Path,
    mutation: str,
) -> None:
    atomic = AtomicArtifactStore(tmp_path)
    store = SnapshotStore(atomic)
    run_root = Path("runs/scope/run_001")
    fingerprint = "a" * 64
    receipt = store.publish(run_root, "asr", fingerprint, Payload(value="first"))
    pointer_path = tmp_path / run_root / "speech/snapshots/asr-current.json"
    payload_path = tmp_path / receipt.relative_path
    if mutation in {"outside", "absolute"}:
        outside = tmp_path / "runs/scope/run_002/speech/snapshots/asr/payload.json"
        outside.parent.mkdir(parents=True)
        outside.write_bytes(payload_path.read_bytes())
        envelope = json.loads(pointer_path.read_text(encoding="utf-8"))
        envelope["payload"]["payload_receipt"]["relative_path"] = (
            str(outside)
            if mutation == "absolute"
            else outside.relative_to(tmp_path).as_posix()
        )
        _rewrite_pointer(atomic, run_root, fingerprint, envelope["payload"])
    else:
        target = payload_path.with_name("target.json")
        payload_path.rename(target)
        payload_path.symlink_to(target)

    with pytest.raises(VideoDemoError) as raised:
        store.load(run_root, "asr", fingerprint, Payload)

    assert raised.value.code == ErrorCode.WORKSPACE_PATH_ESCAPE


def test_snapshot_load_rejects_oversized_pointer_without_reading_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import video_demo.storage.snapshots as snapshots

    store = SnapshotStore(AtomicArtifactStore(tmp_path))
    run_root = Path("runs/scope/run_001")
    fingerprint = "a" * 64
    store.publish(run_root, "asr", fingerprint, Payload(value="first"))
    monkeypatch.setattr(snapshots, "_MAX_POINTER_BYTES", 1)

    assert store.load(run_root, "asr", fingerprint, Payload) is None


def test_snapshot_load_rejects_oversized_payload_without_reading_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import video_demo.storage.snapshots as snapshots

    store = SnapshotStore(AtomicArtifactStore(tmp_path))
    run_root = Path("runs/scope/run_001")
    fingerprint = "a" * 64
    store.publish(run_root, "asr", fingerprint, Payload(value="first"))
    monkeypatch.setattr(snapshots, "_MAX_SNAPSHOT_PAYLOAD_BYTES", 1)

    assert store.load(run_root, "asr", fingerprint, Payload) is None


def test_failed_pointer_publish_preserves_previous_pointer(tmp_path: Path) -> None:
    class FailingPointerStore(AtomicArtifactStore):
        fail_pointer = False

        def write_json(self, relative_path: Path, *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
            if self.fail_pointer and relative_path.name == "asr-current.json":
                raise OSError("模拟 pointer 写入失败")
            return super().write_json(relative_path, *args, **kwargs)  # type: ignore[arg-type]

    atomic = FailingPointerStore(tmp_path)
    store = SnapshotStore(atomic)
    run_root = Path("runs/scope/run_001")
    old_fingerprint = "a" * 64
    store.publish(run_root, "asr", old_fingerprint, Payload(value="old"))
    atomic.fail_pointer = True

    with pytest.raises(OSError):
        store.publish(run_root, "asr", "b" * 64, Payload(value="new"))

    loaded = store.load(run_root, "asr", old_fingerprint, Payload)
    assert loaded is not None and loaded[0].value == "old"


def _rewrite_pointer(
    atomic: AtomicArtifactStore,
    run_root: Path,
    fingerprint: str,
    payload: dict[str, object],
) -> None:
    atomic.write_json(
        run_root / "speech/snapshots/asr-current.json",
        payload,
        schema_version="1.0.0",
        upstream_sha256=fingerprint,
    )
