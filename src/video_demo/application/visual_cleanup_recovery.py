"""Worker 启动时按 keyset 编排已发布视觉制品恢复。"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from video_demo.application.document_publication import scope_key
from video_demo.domain.document_artifact import DocumentArtifactPayload
from video_demo.domain.evidence import KeyframeEvidence
from video_demo.persistence.database import Database
from video_demo.persistence.repositories import (
    JobRepository,
    PublishedRunCleanupRecord,
    Scope,
    VideoRunRepository,
)

_LOGGER = logging.getLogger(__name__)


class PublishedArtifactReader(Protocol):
    def get_artifact(
        self,
        scope: Scope,
        run_id: str,
    ) -> tuple[DocumentArtifactPayload, bytes]: ...


class VisualCleanupPort(Protocol):
    def has_residuals(self, run_relative_root: Path) -> bool: ...

    def cleanup(
        self,
        run_relative_root: Path,
        keyframes: tuple[KeyframeEvidence, ...],
    ) -> bool: ...

    def mark_pending(self, run_relative_root: Path) -> None: ...


class PublishedVisualCleanupRecovery:
    """按数据库主键分页恢复已发布 4.1 Run 的视觉临时制品。"""

    _BATCH_SIZE = 100

    def __init__(
        self,
        database: Database,
        artifact_reader: PublishedArtifactReader,
        cleaner: VisualCleanupPort,
        *,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._database = database
        self._artifact_reader = artifact_reader
        self._cleaner = cleaner
        self._clock = clock

    def recover(self) -> int:
        cursor = 0
        recovered = 0
        while True:
            records = self._list_batch(cursor)
            if not records:
                return recovered
            for record in records:
                cursor = record.run_pk
                recovered += int(self._recover_one(record))
            if len(records) < self._BATCH_SIZE:
                return recovered

    def _list_batch(self, cursor: int) -> tuple[PublishedRunCleanupRecord, ...]:
        with self._database.session() as session:
            return VideoRunRepository(session).list_published_4_for_cleanup(
                after_id=cursor,
                limit=self._BATCH_SIZE,
            )

    def _recover_one(self, record: PublishedRunCleanupRecord) -> bool:
        run_root = Path("runs") / scope_key(record.scope) / record.run_id
        stage = "残留探测"
        try:
            if not self._cleaner.has_residuals(run_root):
                return False
            stage = "活跃 owner 查询"
            with self._database.session() as session:
                if JobRepository(session).has_active_owner(
                    record.scope,
                    record.run_id,
                    now=self._clock(),
                ):
                    return False
            stage = "bundle 重读"
            artifact, _ = self._artifact_reader.get_artifact(record.scope, record.run_id)
            keyframes = tuple(
                item for item in artifact.evidence if isinstance(item, KeyframeEvidence)
            )
            stage = "视觉清理"
            return self._cleaner.cleanup(run_root, keyframes)
        except Exception as error:
            _LOGGER.warning(
                "视觉清理恢复单 Run 失败，run_id=%s，阶段=%s，异常类型=%s",
                record.run_id,
                stage,
                type(error).__name__,
            )
            try:
                self._cleaner.mark_pending(run_root)
            except Exception as pending_error:
                _LOGGER.warning(
                    "视觉清理恢复单 Run 失败，run_id=%s，阶段=pending 刷新，异常类型=%s",
                    record.run_id,
                    type(pending_error).__name__,
                )
            return False


__all__ = ["PublishedVisualCleanupRecovery"]
