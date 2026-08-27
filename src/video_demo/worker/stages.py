from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.storage.artifacts import ArtifactReceipt, AtomicArtifactStore

StagePayload = dict[str, Any] | list[Any]
StageFunction = Callable[["StageContext"], StagePayload]

PIPELINE_STAGE_ORDER = (
    "REGISTER",
    "PROBE",
    "TRANSCODE",
    "EVIDENCE_PREP",
    "CHAPTER_PLAN",
    "FRAME_SEARCH",
    "VISUAL_EVIDENCE",
    "CHAPTER_WRITE",
    "DOCUMENT_ASSEMBLY",
    "RESULT",
)


@dataclass(frozen=True, slots=True)
class StageContext:
    run_id: str
    run_relative_root: Path
    source_sha256: str
    is_cancel_requested: Callable[[], bool] = lambda: False


@dataclass(frozen=True, slots=True)
class StageDefinition:
    name: str
    schema_version: str
    relative_path: Path
    execute: StageFunction


@dataclass(frozen=True, slots=True)
class StageExecution:
    receipt: ArtifactReceipt
    reused: bool
    duration_ms: int


class StageRunner:
    """只复用可从磁盘重新验证的阶段产物。"""

    def __init__(
        self,
        artifact_store: AtomicArtifactStore,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._artifact_store = artifact_store
        self._clock = clock

    def execute(
        self,
        context: StageContext,
        stages: Sequence[StageDefinition],
    ) -> dict[str, StageExecution]:
        upstream_sha256 = context.source_sha256
        invalidated = False
        executions: dict[str, StageExecution] = {}
        for stage in stages:
            if context.is_cancel_requested():
                raise VideoDemoError(ErrorCode.JOB_CANCELLED, "任务已请求取消")
            relative_path = context.run_relative_root / stage.relative_path
            receipt = None if invalidated else self._verified_receipt(
                relative_path,
                stage.schema_version,
                upstream_sha256,
            )
            if receipt is not None:
                execution = StageExecution(receipt=receipt, reused=True, duration_ms=0)
            else:
                started_at = self._clock()
                payload = stage.execute(context)
                receipt = self._artifact_store.write_json(
                    relative_path,
                    payload,
                    schema_version=stage.schema_version,
                    upstream_sha256=upstream_sha256,
                )
                duration_ms = round((self._clock() - started_at) * 1000)
                execution = StageExecution(
                    receipt=receipt,
                    reused=False,
                    duration_ms=duration_ms,
                )
                invalidated = True
            executions[stage.name] = execution
            upstream_sha256 = receipt.sha256
        return executions

    def _verified_receipt(
        self,
        relative_path: Path,
        schema_version: str,
        upstream_sha256: str,
    ) -> ArtifactReceipt | None:
        artifact_path = self._artifact_store.runtime_root / relative_path
        if not artifact_path.is_file() or artifact_path.is_symlink():
            return None
        encoded = artifact_path.read_bytes()
        receipt = ArtifactReceipt(
            relative_path=relative_path.as_posix(),
            schema_version=schema_version,
            sha256=hashlib.sha256(encoded).hexdigest(),
            upstream_sha256=upstream_sha256,
        )
        try:
            self._artifact_store.read_verified_json(receipt)
        except VideoDemoError:
            return None
        return receipt
