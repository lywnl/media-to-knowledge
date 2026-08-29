from __future__ import annotations

from typing import Annotated, Literal

from fastapi import APIRouter, Depends, File, Response, UploadFile, status

from video_demo.api.dependencies import AppContainer, get_container, get_scope
from video_demo.api.schemas import (
    CreateMediaRunRequest,
    MediaObjectResponse,
    MediaRunHistoryItem,
    MediaRunHistoryResponse,
    MediaRunResponse,
)
from video_demo.application.media_runs import MediaRunService
from video_demo.domain.audio_document import AudioUnderstandingResult
from video_demo.domain.image_document import ImageUnderstandingResult
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.persistence.media_repositories import MediaObjectRepository
from video_demo.persistence.models import AudioObjectModel, ImageObjectModel
from video_demo.persistence.repositories import Scope

MediaUploadFile = Annotated[UploadFile, File()]
MediaScope = Annotated[Scope, Depends(get_scope)]
MediaContainer = Annotated[AppContainer, Depends(get_container)]


def build_media_router(kind: Literal["AUDIO", "IMAGE"]) -> APIRouter:
    label = "音频" if kind == "AUDIO" else "图片"
    object_model = AudioObjectModel if kind == "AUDIO" else ImageObjectModel
    result_model = AudioUnderstandingResult if kind == "AUDIO" else ImageUnderstandingResult
    object_prefix = kind.lower()
    router = APIRouter(
        prefix=f"/api/kb/knowledge-bases/{{kb_id}}/{object_prefix}",
        tags=[f"{label}理解"],
    )

    def service(container: AppContainer) -> MediaRunService:
        return container.media_run_services[kind]

    @router.post(
        "-objects",
        response_model=MediaObjectResponse,
        status_code=status.HTTP_201_CREATED,
    )
    def upload_object(
        file: MediaUploadFile,
        scope: MediaScope,
        container: MediaContainer,
    ) -> MediaObjectResponse:
        upload = container.media_upload_services[kind]
        record = upload.upload(file.file, file.filename or "", file.content_type or "", scope)
        return MediaObjectResponse(
            object_ref=record.object_ref,
            original_filename=record.original_filename,
            declared_mime=record.declared_mime,
            detected_mime=record.detected_mime,
            size_bytes=record.size_bytes,
            sha256=record.sha256,
        )

    @router.get("-objects/{object_ref}", response_model=MediaObjectResponse)
    def get_object(
        object_ref: str,
        scope: MediaScope,
        container: MediaContainer,
    ) -> MediaObjectResponse:
        with container.database.session() as session:
            model = MediaObjectRepository(session, object_model).get(scope, object_ref)
            if model is None:
                code = (
                    ErrorCode.AUDIO_OBJECT_NOT_FOUND
                    if kind == "AUDIO"
                    else ErrorCode.IMAGE_OBJECT_NOT_FOUND
                )
                raise VideoDemoError(code, f"{label}对象不存在")
            return MediaObjectResponse(
                object_ref=model.object_ref,
                original_filename=model.original_filename,
                declared_mime=model.declared_mime,
                detected_mime=model.detected_mime,
                size_bytes=model.size_bytes,
                sha256=model.sha256,
            )

    @router.post(
        "-understanding-runs",
        response_model=MediaRunResponse,
        status_code=status.HTTP_202_ACCEPTED,
    )
    def create_run(
        payload: CreateMediaRunRequest,
        scope: MediaScope,
        container: MediaContainer,
    ) -> MediaRunResponse:
        view = service(container).create(
            scope=scope,
            object_ref=payload.object_ref,
            idempotency_key=payload.idempotency_key,
            language_hints=payload.language_hints,
            hotwords=payload.hotwords,
            core_context=payload.core_context,
            document_config=payload.document_config,
        )
        return _run_response(view)

    @router.get("-understanding-runs", response_model=MediaRunHistoryResponse)
    def list_runs(
        scope: MediaScope,
        container: MediaContainer,
    ) -> MediaRunHistoryResponse:
        return MediaRunHistoryResponse(
            items=tuple(
                MediaRunHistoryItem.model_validate(item)
                for item in service(container).list_history(scope)
            ),
        )

    @router.get("-understanding-runs/{run_id}", response_model=MediaRunResponse)
    def get_run(
        run_id: str,
        scope: MediaScope,
        container: MediaContainer,
    ) -> MediaRunResponse:
        return _run_response(service(container).get(scope, run_id))

    @router.get("-understanding-runs/{run_id}/result", response_model=result_model)
    def get_result(
        run_id: str,
        scope: MediaScope,
        container: MediaContainer,
    ) -> object:
        return container.media_query_services[kind].get_result(scope, run_id)

    @router.get("-understanding-runs/{run_id}/document")
    def get_document(
        run_id: str,
        scope: MediaScope,
        container: MediaContainer,
    ) -> Response:
        return Response(
            content=container.media_query_services[kind].get_document(scope, run_id),
            media_type="text/markdown; charset=utf-8",
            headers={"Content-Disposition": 'inline; filename="knowledge-note.md"'},
        )

    return router


def _run_response(view: object) -> MediaRunResponse:
    return MediaRunResponse(
        run_id=view.run_id,  # type: ignore[attr-defined]
        job_id=view.job_id,  # type: ignore[attr-defined]
        status=view.status,  # type: ignore[attr-defined]
        current_stage=view.current_stage,  # type: ignore[attr-defined]
        warning_codes=view.warning_codes,  # type: ignore[attr-defined]
        error_code=view.error_code,  # type: ignore[attr-defined]
    )
