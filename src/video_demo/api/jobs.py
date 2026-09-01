from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Header, Query

from video_demo.api.dependencies import AppContainer, get_container
from video_demo.api.schemas import JobResponse
from video_demo.application.runs import JobView
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.persistence.repositories import JobRepository
from video_demo.persistence.scope import Scope

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


def _service_for_job(
    container: AppContainer,
    scope: Scope,
    job_id: str,
) -> object:
    """按任务资源类型选择对应媒体服务，禁止音频任务落入视频服务。"""

    with container.database.session() as session:
        job = JobRepository(session).get(scope, job_id)
    if job is None:
        raise VideoDemoError(ErrorCode.JOB_NOT_FOUND, "任务不存在")
    if job.resource_type == "AUDIO_UNDERSTANDING_RUN":
        return container.audio_run_service
    if job.resource_type == "VIDEO_UNDERSTANDING_RUN":
        return container.run_service
    if job.resource_type == "IMAGE_UNDERSTANDING_RUN":
        return container.media_run_services["IMAGE"]
    raise VideoDemoError(ErrorCode.JOB_NOT_FOUND, "任务不存在")


@router.get("/{job_id}", response_model=JobResponse)
def get_job(
    job_id: str,
    scope: Annotated[Scope, Depends(job_scope)],
    container: Annotated[AppContainer, Depends(get_container)],
) -> JobResponse:
    service = _service_for_job(container, scope, job_id)
    return _response(service.get_job(scope, job_id))  # type: ignore[attr-defined]


@router.post("/{job_id}/cancel", response_model=JobResponse)
def cancel_job(
    job_id: str,
    scope: Annotated[Scope, Depends(job_scope)],
    container: Annotated[AppContainer, Depends(get_container)],
) -> JobResponse:
    service = _service_for_job(container, scope, job_id)
    return _response(service.cancel_job(scope, job_id))  # type: ignore[attr-defined]


@router.post("/{job_id}/retry", response_model=JobResponse)
def retry_job(
    job_id: str,
    scope: Annotated[Scope, Depends(job_scope)],
    container: Annotated[AppContainer, Depends(get_container)],
) -> JobResponse:
    service = _service_for_job(container, scope, job_id)
    return _response(service.retry_job(scope, job_id))  # type: ignore[attr-defined]
