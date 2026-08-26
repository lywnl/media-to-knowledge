from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query, Response, status
from pydantic import TypeAdapter

from video_demo.api.dependencies import AppContainer, get_container, get_scope
from video_demo.api.schemas import (
    CreateRunRequest,
    EvidencePageResponse,
    PublicEvidence,
    RunHistoryItem,
    RunHistoryResponse,
    RunResponse,
)
from video_demo.application.runs import RunView
from video_demo.domain.evidence import DocumentEvidenceItem
from video_demo.domain.result import VideoUnderstandingResult
from video_demo.persistence.repositories import Scope

router = APIRouter(
    prefix="/api/kb/knowledge-bases/{kb_id}/video-understanding-runs",
    tags=["视频理解运行"],
)


def _response(view: RunView) -> RunResponse:
    return RunResponse(
        run_id=view.run_id,
        job_id=view.job_id,
        status=str(view.status),
        current_stage=view.current_stage,
        warning_codes=view.warning_codes,
        error_code=view.error_code,
    )


@router.post("", response_model=RunResponse, status_code=status.HTTP_202_ACCEPTED)
def create_video_run(
    payload: CreateRunRequest,
    scope: Annotated[Scope, Depends(get_scope)],
    container: Annotated[AppContainer, Depends(get_container)],
) -> RunResponse:
    return _response(
        container.run_service.create(
            scope=scope,
            object_ref=payload.object_ref,
            idempotency_key=payload.idempotency_key,
            language_hints=payload.language_hints,
            hotwords=payload.hotwords,
            core_context=payload.core_context,
            document_config=payload.document_config,
            result_schema_version=payload.result_schema_version,
        ),
    )


@router.get("", response_model=RunHistoryResponse)
def list_video_runs(
    scope: Annotated[Scope, Depends(get_scope)],
    container: Annotated[AppContainer, Depends(get_container)],
) -> RunHistoryResponse:
    return RunHistoryResponse(
        items=tuple(
            RunHistoryItem(
                run_id=item.run_id,
                object_ref=item.object_ref,
                original_filename=item.original_filename,
                detected_mime=item.detected_mime,
                size_bytes=item.size_bytes,
                status=item.status.value,
                current_stage=item.current_stage,
                warning_codes=item.warning_codes,
                error_code=item.error_code,
                created_at=item.created_at,
                updated_at=item.updated_at,
            )
            for item in container.run_service.list_history(scope)
        ),
    )


@router.get("/{run_id}", response_model=RunResponse)
def get_video_run(
    run_id: str,
    scope: Annotated[Scope, Depends(get_scope)],
    container: Annotated[AppContainer, Depends(get_container)],
) -> RunResponse:
    return _response(container.run_service.get(scope, run_id))


_PUBLIC_EVIDENCE_ADAPTER: TypeAdapter[PublicEvidence] = TypeAdapter(PublicEvidence)


@router.get("/{run_id}/result", response_model=VideoUnderstandingResult)
def get_video_result(
    run_id: str,
    scope: Annotated[Scope, Depends(get_scope)],
    container: Annotated[AppContainer, Depends(get_container)],
) -> VideoUnderstandingResult:
    container.run_service.require_result_ready(scope, run_id)
    return container.result_query_service.get_result(scope, run_id)


@router.get("/{run_id}/evidence", response_model=EvidencePageResponse)
def get_video_evidence(
    run_id: str,
    scope: Annotated[Scope, Depends(get_scope)],
    container: Annotated[AppContainer, Depends(get_container)],
    evidence_type: Annotated[str | None, Query(max_length=64)] = None,
    start_ms: Annotated[int | None, Query(ge=0)] = None,
    cursor: Annotated[str | None, Query(max_length=256)] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
) -> EvidencePageResponse:
    container.run_service.require_result_ready(scope, run_id)
    page = container.result_query_service.get_evidence(
        scope,
        run_id,
        evidence_type=evidence_type,
        start_ms=start_ms,
        cursor=cursor,
        limit=limit,
    )
    return EvidencePageResponse(
        items=tuple(_public_evidence(item) for item in page.items),
        next_cursor=page.next_cursor,
    )


@router.get("/{run_id}/document")
def get_video_document(
    run_id: str,
    scope: Annotated[Scope, Depends(get_scope)],
    container: Annotated[AppContainer, Depends(get_container)],
) -> Response:
    container.run_service.require_result_ready(scope, run_id)
    document = container.result_query_service.get_document(scope, run_id)
    return Response(
        content=document,
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": 'inline; filename="knowledge-note.md"'},
    )


@router.get("/{run_id}/keyframes/{keyframe_id}/content")
def get_keyframe_content(
    run_id: str,
    keyframe_id: str,
    scope: Annotated[Scope, Depends(get_scope)],
    container: Annotated[AppContainer, Depends(get_container)],
) -> Response:
    container.run_service.require_result_ready(scope, run_id)
    content = container.result_query_service.get_keyframe(scope, run_id, keyframe_id)
    return Response(content=content.content, media_type=content.mime_type)


def _public_evidence(item: DocumentEvidenceItem) -> PublicEvidence:
    payload: dict[str, object] = item.model_dump(
        mode="json",
        exclude_computed_fields=True,
    )
    payload.pop("relative_path", None)
    return _PUBLIC_EVIDENCE_ADAPTER.validate_python(payload)
