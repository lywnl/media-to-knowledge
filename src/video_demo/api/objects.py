from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, File, UploadFile, status

from video_demo.api.dependencies import AppContainer, get_container, get_scope
from video_demo.api.schemas import VideoObjectResponse
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.persistence.repositories import VideoObjectRepository
from video_demo.persistence.scope import Scope

router = APIRouter(prefix="/api/kb/knowledge-bases/{kb_id}/video-objects", tags=["视频对象"])


@router.post("", response_model=VideoObjectResponse, status_code=status.HTTP_201_CREATED)
def upload_video_object(
    file: Annotated[UploadFile, File(description="MP4、MOV、MKV 或 WebM 视频")],
    scope: Annotated[Scope, Depends(get_scope)],
    container: Annotated[AppContainer, Depends(get_container)],
) -> VideoObjectResponse:
    record = container.upload_service.upload(
        file.file,
        file.filename or "",
        file.content_type or "application/octet-stream",
        scope,
    )
    return VideoObjectResponse(
        object_ref=record.object_ref,
        original_filename=record.original_filename,
        declared_mime=record.declared_mime,
        detected_mime=record.detected_mime,
        size_bytes=record.size_bytes,
        sha256=record.sha256,
    )


@router.get("/{object_ref}", response_model=VideoObjectResponse)
def get_video_object(
    object_ref: str,
    scope: Annotated[Scope, Depends(get_scope)],
    container: Annotated[AppContainer, Depends(get_container)],
) -> VideoObjectResponse:
    with container.database.session() as session:
        model = VideoObjectRepository(session).get(scope, object_ref)
        if model is None:
            raise VideoDemoError(ErrorCode.VIDEO_OBJECT_NOT_FOUND, "视频对象不存在")
        return VideoObjectResponse(
            object_ref=model.object_ref,
            original_filename=model.original_filename,
            declared_mime=model.declared_mime,
            detected_mime=model.detected_mime,
            size_bytes=model.size_bytes,
            sha256=model.sha256,
            status="READY",
        )
