from __future__ import annotations

from typing import Any

from sqlalchemy import Select, select
from sqlalchemy.orm import Session

from video_demo.persistence.models import (
    AudioObjectModel,
    AudioUnderstandingRunModel,
    ImageObjectModel,
    ImageUnderstandingRunModel,
    RunStatusValue,
    VideoObjectStatus,
)
from video_demo.persistence.scope import Scope


def _scope_where(statement: Select[Any], model: type[Any], scope: Scope) -> Select[Any]:
    return statement.where(
        model.tenant_id == scope.tenant_id,
        model.application_id == scope.application_id,
        model.knowledge_base_id == scope.knowledge_base_id,
    )


class MediaObjectRepository:
    def __init__(self, session: Session, model: type[Any]) -> None:
        self._session = session
        self._model = model

    def add_ready(
        self,
        *,
        scope: Scope,
        object_ref: str,
        original_filename: str,
        declared_mime: str,
        detected_mime: str,
        size_bytes: int,
        sha256: str,
        relative_path: str,
    ) -> Any:
        model = self._model(
            tenant_id=scope.tenant_id,
            application_id=scope.application_id,
            knowledge_base_id=scope.knowledge_base_id,
            object_ref=object_ref,
            original_filename=original_filename,
            declared_mime=declared_mime,
            detected_mime=detected_mime,
            size_bytes=size_bytes,
            sha256=sha256,
            relative_path=relative_path,
            status=VideoObjectStatus.READY,
            scan_details={},
        )
        self._session.add(model)
        self._session.flush()
        return model

    def get_ready(self, scope: Scope, object_ref: str) -> Any | None:
        statement = _scope_where(select(self._model), self._model, scope).where(
            self._model.object_ref == object_ref,
            self._model.status == VideoObjectStatus.READY,
        )
        return self._session.scalar(statement)

    def get(self, scope: Scope, object_ref: str) -> Any | None:
        statement = _scope_where(select(self._model), self._model, scope).where(
            self._model.object_ref == object_ref,
        )
        return self._session.scalar(statement)


class MediaRunRepository:
    def __init__(self, session: Session, model: type[Any]) -> None:
        self._session = session
        self._model = model

    def add(
        self,
        *,
        scope: Scope,
        run_id: str,
        object_ref: str,
        idempotency_key: str,
        config_snapshot: dict[str, Any],
    ) -> Any:
        model = self._model(
            tenant_id=scope.tenant_id,
            application_id=scope.application_id,
            knowledge_base_id=scope.knowledge_base_id,
            run_id=run_id,
            object_ref=object_ref,
            idempotency_key=idempotency_key,
            status=RunStatusValue.PENDING,
            current_stage="REGISTER",
            warning_codes=[],
            config_snapshot=config_snapshot,
        )
        self._session.add(model)
        self._session.flush()
        return model

    def get(self, scope: Scope, run_id: str) -> Any | None:
        statement = _scope_where(select(self._model), self._model, scope).where(
            self._model.run_id == run_id,
        )
        return self._session.scalar(statement)

    def get_by_idempotency(self, scope: Scope, key: str) -> Any | None:
        statement = _scope_where(select(self._model), self._model, scope).where(
            self._model.idempotency_key == key,
        )
        return self._session.scalar(statement)

    def list_with_objects(self, scope: Scope) -> list[Any]:
        statement = _scope_where(
            select(self._model).order_by(self._model.created_at.desc(), self._model.id.desc()),
            self._model,
            scope,
        )
        return list(self._session.scalars(statement))

    def update_owned(
        self,
        scope: Scope,
        run_id: str,
        *,
        status: RunStatusValue | None = None,
        current_stage: str | None = None,
        warnings: tuple[str, ...] | None = None,
        error_code: str | None = None,
    ) -> bool:
        model = self.get(scope, run_id)
        if model is None:
            return False
        if status is not None:
            model.status = status
        if current_stage is not None:
            model.current_stage = current_stage
        if warnings is not None:
            model.warning_codes = list(warnings)
        model.error_code = error_code
        self._session.flush()
        return True


def media_model_for_kind(kind: str) -> tuple[type[Any], type[Any]]:
    if kind == "AUDIO":
        return AudioObjectModel, AudioUnderstandingRunModel
    if kind == "IMAGE":
        return ImageObjectModel, ImageUnderstandingRunModel
    raise ValueError(f"不支持的媒体类型: {kind}")


__all__ = ["MediaObjectRepository", "MediaRunRepository", "media_model_for_kind"]
