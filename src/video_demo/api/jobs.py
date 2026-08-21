from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Header, Query

from video_demo.api.dependencies import AppContainer, get_container
from video_demo.api.schemas import JobResponse
from video_demo.application.runs import JobView
from video_demo.persistence.repositories import Scope

router = APIRouter(prefix="/api/kb/jobs", tags=["可靠任务"])


def job_scope(
    knowledge_base_id: Annotated[
        str,
        Query(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$"),
    ],
    tenant_id: Annotated[
        str,
        Header(alias="X-Tenant-Id", min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$"),
    ],
    application_id: Annotated[
        str,
        Header(
            alias="X-Application-Id",
            min_length=1,
            max_length=128,
            pattern=r"^[A-Za-z0-9_-]+$",
        ),
    ],
) -> Scope:
    return Scope(tenant_id, application_id, knowledge_base_id)


def _response(view: JobView) -> JobResponse:
    return JobResponse(
        job_id=view.job_id,
        resource_id=view.resource_id,
        status=str(view.status),
        attempt_count=view.attempt_count,
        max_attempts=view.max_attempts,
        error_code=view.error_code,
    )


@router.get("/{job_id}", response_model=JobResponse)
def get_job(
    job_id: str,
    scope: Annotated[Scope, Depends(job_scope)],
    container: Annotated[AppContainer, Depends(get_container)],
) -> JobResponse:
    return _response(container.run_service.get_job(scope, job_id))


@router.post("/{job_id}/cancel", response_model=JobResponse)
def cancel_job(
    job_id: str,
    scope: Annotated[Scope, Depends(job_scope)],
    container: Annotated[AppContainer, Depends(get_container)],
) -> JobResponse:
    return _response(container.run_service.cancel_job(scope, job_id))


@router.post("/{job_id}/retry", response_model=JobResponse)
def retry_job(
    job_id: str,
    scope: Annotated[Scope, Depends(job_scope)],
    container: Annotated[AppContainer, Depends(get_container)],
) -> JobResponse:
    return _response(container.run_service.retry_job(scope, job_id))
