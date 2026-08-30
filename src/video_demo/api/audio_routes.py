"""音频理解 API 路由。"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, File, Response, UploadFile, status

from video_demo.api.dependencies import AppContainer, get_container, get_scope
from video_demo.api.schemas import (
    CreateAudioRunRequest,
    MediaObjectResponse,
    MediaRunHistoryItem,
    MediaRunHistoryResponse,
    MediaRunResponse,
    PublicAudioUnderstandingResult,
)
from video_demo.application.media_runs import MediaRunService
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.persistence.media_repositories import MediaObjectRepository
from video_demo.persistence.models import AudioObjectModel
from video_demo.persistence.scope import Scope

AudioUploadFile = Annotated[UploadFile, File()]
AudioScope = Annotated[Scope, Depends(get_scope)]
AudioContainer = Annotated[AppContainer, Depends(get_container)]

router = APIRouter(
    prefix="/api/kb/knowledge-bases/{kb_id}/audio",
    tags=["音频理解"],
)


@router.post("-objects", response_model=MediaObjectResponse, status_code=status.HTTP_201_CREATED)
def upload_audio_object(
    file: AudioUploadFile,
    scope: AudioScope,
    container: AudioContainer,
) -> MediaObjectResponse:
    record = container.media_upload_services["AUDIO"].upload(
        file.file,
        file.filename or "",
        file.content_type or "",
        scope,
    )
    return _object_response(record)


@router.get("-objects/{object_ref}", response_model=MediaObjectResponse)
def get_audio_object(
    object_ref: str,
    scope: AudioScope,
    container: AudioContainer,
) -> MediaObjectResponse:
    with container.database.session() as session:
        model = MediaObjectRepository(session, AudioObjectModel).get(scope, object_ref)
        if model is None:
            raise VideoDemoError(ErrorCode.AUDIO_OBJECT_NOT_FOUND, "音频对象不存在")
        return _object_response(model)


@router.post(
    "-understanding-runs",
    response_model=MediaRunResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def create_audio_run(
    payload: CreateAudioRunRequest,
    scope: AudioScope,
    container: AudioContainer,
) -> MediaRunResponse:
    view = _audio_runs(container).create(
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
def list_audio_runs(
    scope: AudioScope,
    container: AudioContainer,
) -> MediaRunHistoryResponse:
    return MediaRunHistoryResponse(
        items=tuple(
            MediaRunHistoryItem.model_validate(item)
            for item in _audio_runs(container).list_history(scope)
        ),
    )


@router.get("-understanding-runs/{run_id}", response_model=MediaRunResponse)
def get_audio_run(
    run_id: str,
    scope: AudioScope,
    container: AudioContainer,
) -> MediaRunResponse:
    return _run_response(_audio_runs(container).get(scope, run_id))


@router.get(
    "-understanding-runs/{run_id}/result",
    response_model=PublicAudioUnderstandingResult,
)
def get_audio_result(
    run_id: str,
    scope: AudioScope,
    container: AudioContainer,
) -> PublicAudioUnderstandingResult:
    result = container.media_query_services["AUDIO"].get_result(scope, run_id)
    return PublicAudioUnderstandingResult.model_validate(
        result.model_dump(mode="json", exclude_computed_fields=True),
    )


@router.get("-understanding-runs/{run_id}/document")
def get_audio_document(
    run_id: str,
    scope: AudioScope,
    container: AudioContainer,
) -> Response:
    return Response(
        content=container.media_query_services["AUDIO"].get_document(scope, run_id),
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": 'inline; filename="knowledge-note.md"'},
    )


def _audio_runs(container: AppContainer) -> MediaRunService:
    return container.media_run_services["AUDIO"]


def _object_response(model: object) -> MediaObjectResponse:
    return MediaObjectResponse(
        object_ref=model.object_ref,  # type: ignore[attr-defined]
        original_filename=model.original_filename,  # type: ignore[attr-defined]
        declared_mime=model.declared_mime,  # type: ignore[attr-defined]
        detected_mime=model.detected_mime,  # type: ignore[attr-defined]
        size_bytes=model.size_bytes,  # type: ignore[attr-defined]
        sha256=model.sha256,  # type: ignore[attr-defined]
    )


def _run_response(view: object) -> MediaRunResponse:
    return MediaRunResponse(
        run_id=view.run_id,  # type: ignore[attr-defined]
        job_id=view.job_id,  # type: ignore[attr-defined]
        status=view.status,  # type: ignore[attr-defined]
        current_stage=view.current_stage,  # type: ignore[attr-defined]
        warning_codes=view.warning_codes,  # type: ignore[attr-defined]
        error_code=view.error_code,  # type: ignore[attr-defined]
    )
