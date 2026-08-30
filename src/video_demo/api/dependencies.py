from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path as FilePath
from typing import Annotated

from fastapi import Header, Path, Request

from video_demo.application.audio_queries import AudioQueryService
from video_demo.application.media_queries import MediaQueryService
from video_demo.application.media_runs import MediaRunService
from video_demo.application.media_uploads import MediaUploadService
from video_demo.application.queries import ResultQueryService
from video_demo.application.runs import RunService
from video_demo.application.uploads import UploadService
from video_demo.persistence.database import Database
from video_demo.persistence.scope import Scope
from video_demo.storage.object_store import LocalVideoObjectStore


@dataclass(frozen=True, slots=True)
class AppContainer:
    runtime_root: FilePath
    database: Database
    object_store: LocalVideoObjectStore
    upload_service: UploadService
    run_service: RunService
    result_query_service: ResultQueryService
    media_upload_services: dict[str, MediaUploadService]
    media_run_services: dict[str, MediaRunService]
    media_query_services: dict[str, MediaQueryService | AudioQueryService]


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
