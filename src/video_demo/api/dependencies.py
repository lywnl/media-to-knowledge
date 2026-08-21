from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from fastapi import Header, Path, Request

from video_demo.application.queries import ResultQueryService
from video_demo.application.runs import RunService
from video_demo.application.uploads import UploadService
from video_demo.config import Settings
from video_demo.persistence.database import Database
from video_demo.persistence.repositories import Scope
from video_demo.storage.object_store import LocalVideoObjectStore


@dataclass(frozen=True, slots=True)
class AppContainer:
    settings: Settings
    database: Database
    object_store: LocalVideoObjectStore
    upload_service: UploadService
    run_service: RunService
    result_query_service: ResultQueryService


def get_container(request: Request) -> AppContainer:
    container: AppContainer = request.app.state.container
    return container


def get_scope(
    knowledge_base_id: Annotated[
        str,
        Path(alias="kb_id", min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$"),
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
